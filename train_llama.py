"""
Train Llama on various text datasets

A simple training script for language modeling with Llama,
inspired by https://github.com/lucidrains/x-transformers/blob/main/train_enwik8.py

Uses HuggingFace streaming datasets with shuffle buffers for memory-efficient
training on large datasets. See: https://huggingface.co/docs/datasets/main/stream

Usage:
    uv run python train_llama.py --help
    uv run python train_llama.py --dataset wikipedia --num-tokens 0.1
    uv run python train_llama.py --dataset finepdfs-edu --num-tokens 0.1
    uv run python train_llama.py --dataset fineweb-edu --num-tokens 0.1
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import tqdm
import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import typer
import wandb

from allformers.models.llama.llama import Llama, LlamaConfig
from allformers.data.streaming import StreamingTextDataset
from allformers.data.wikipedia import load_wikipedia_streaming, wikipedia_text_fn
from allformers.data.finepdfs_edu import load_finepdfs_edu_streaming, finepdfs_edu_text_fn
from allformers.data.fineweb_edu import load_fineweb_edu_streaming, fineweb_edu_text_fn
from transformers import AutoTokenizer


app = typer.Typer(
    help="Train Llama on text datasets",
    add_completion=False,
)


# =============================================================================
# Dataset Configuration
# =============================================================================


class DatasetChoice(str, Enum):
    """Available datasets for training."""
    wikipedia = "wikipedia"
    finepdfs_edu = "finepdfs-edu"
    fineweb_edu = "fineweb-edu"


def load_dataset_for_training(
    dataset: DatasetChoice,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    # FinePDFs-Edu options (uses per-page language detection)
    finepdfs_filter_english: bool = True,
    finepdfs_english_threshold: float = 0.8,
    # FineWeb-Edu options (uses language column and language_score)
    fineweb_subset: str = "sample-10BT",
    fineweb_min_edu_score: float | None = None,
    fineweb_language: str | None = "en",
    fineweb_min_lang_score: float | None = None,
) -> tuple:
    """Load train/val streaming datasets for the specified dataset.
    
    Returns:
        Tuple of (train_data, val_data, text_fn)
        - train_data: Training IterableDataset (shuffled)
        - val_data: Validation IterableDataset (shuffled)
        - text_fn: Function to extract text from dataset rows
    """
    if dataset == DatasetChoice.wikipedia:
        train_data, val_data = load_wikipedia_streaming(
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        return train_data, val_data, wikipedia_text_fn
    
    elif dataset == DatasetChoice.finepdfs_edu:
        train_data, val_data = load_finepdfs_edu_streaming(
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            filter_english=finepdfs_filter_english,
            english_threshold=finepdfs_english_threshold,
        )
        return train_data, val_data, finepdfs_edu_text_fn
    
    elif dataset == DatasetChoice.fineweb_edu:
        train_data, val_data = load_fineweb_edu_streaming(
            subset=fineweb_subset,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            min_score=fineweb_min_edu_score,
            filter_language=fineweb_language,
            min_language_score=fineweb_min_lang_score,
        )
        return train_data, val_data, fineweb_edu_text_fn
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# =============================================================================
# Helpers
# =============================================================================


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(tokenizer, tokens):
    """Decode token IDs to string."""
    return tokenizer.decode(tokens.tolist())


# =============================================================================
# Main
# =============================================================================


@app.command()
def train(
    # Dataset selection
    dataset: Annotated[DatasetChoice, typer.Option(help="Dataset to train on")] = DatasetChoice.fineweb_edu,
    shuffle_buffer_size: Annotated[int, typer.Option(help="Shuffle buffer size for streaming datasets")] = 10_000,
    # FinePDFs-Edu specific options (uses per-page language detection from PDF OCR)
    finepdfs_filter_english: Annotated[bool, typer.Option(help="[finepdfs] Filter for majority English documents")] = True,
    finepdfs_english_threshold: Annotated[float, typer.Option(help="[finepdfs] Fraction of pages that must be English (0.0-1.0)")] = 0.8,
    # FineWeb-Edu specific options (uses language column from web crawl metadata)
    fineweb_subset: Annotated[str, typer.Option(help="[fineweb] Subset: default, sample-10BT, sample-100BT, sample-350BT, CC-MAIN-*")] = "sample-10BT",
    fineweb_min_edu_score: Annotated[float | None, typer.Option(help="[fineweb] Min educational score (higher=more educational)")] = None,
    fineweb_language: Annotated[str | None, typer.Option(help="[fineweb] Filter by language column (e.g., 'en'), None to disable")] = "en",
    fineweb_min_lang_score: Annotated[float | None, typer.Option(help="[fineweb] Min language detection confidence (0.0-1.0)")] = None,
    # Training hyperparameters
    num_tokens: Annotated[float, typer.Option(help="Total number of tokens to train on (in billions, e.g., 0.01 = 10M tokens)")] = 0.01,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    gradient_accumulate: Annotated[int, typer.Option(help="Gradient accumulation steps")] = 1,
    learning_rate: Annotated[float, typer.Option(help="Peak learning rate")] = 3e-4,
    warmup_ratio: Annotated[float, typer.Option(help="Fraction of training for LR warmup")] = 0.05,
    cooldown_ratio: Annotated[float, typer.Option(help="Fraction of training for LR cooldown")] = 0.5,
    min_learning_rate_frac: Annotated[float, typer.Option(help="Min LR as fraction of peak LR (e.g., 0.1 = 10%)")] = 0.1,
    seq_len: Annotated[int, typer.Option(help="Sequence length")] = 1024,
    # Performance optimizations
    use_amp: Annotated[bool, typer.Option(help="Use mixed precision training (AMP)")] = True,
    use_compile: Annotated[bool, typer.Option(help="Use torch.compile for model optimization")] = True,
    # Logging intervals
    validate_every: Annotated[int, typer.Option(help="Validate every N batches")] = 100,
    val_batches: Annotated[int, typer.Option(help="Number of batches for validation")] = 64,
    generate_every: Annotated[int, typer.Option(help="Generate samples every N batches")] = 500,
    generate_length: Annotated[int, typer.Option(help="Length of generated samples")] = 256,
    # Model settings (Llama specific)
    embedding_dim: Annotated[int, typer.Option(help="Model embedding dimension")] = 768,
    num_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 12,
    num_kv_heads: Annotated[int | None, typer.Option(help="Number of KV heads for GQA (None = same as num_heads)")] = None,
    num_layers: Annotated[int, typer.Option(help="Number of transformer layers")] = 12,
    intermediate_size: Annotated[int | None, typer.Option(help="MLP intermediate size (None = 4 * embedding_dim)")] = None,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
    rope_theta: Annotated[float, typer.Option(help="RoPE base frequency")] = 10000.0,
    # Tokenizer settings
    tokenizer_name: Annotated[str, typer.Option(help="HuggingFace tokenizer to use")] = "meta-llama/Llama-3.2-1B",
    # Wandb settings
    wandb_project: Annotated[str, typer.Option(help="Wandb project name")] = "allformers-llama",
    wandb_run_name: Annotated[str, typer.Option(help="Wandb run name (auto-generated if not provided)")] = "",
    track_online: Annotated[bool, typer.Option(help="Track experiment online with wandb")] = True,
    # Reproducibility
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
):
    """Train Llama on the specified dataset."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset.value}")

    # Auto-generate run name if not provided
    if not wandb_run_name:
        wandb_run_name = f"llama-{dataset.value}"

    # Initialize tokenizer
    # Note: Llama tokenizer requires HF_TOKEN for gated models
    import os
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    
    # Set pad token if not present (Llama uses eos_token as pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load streaming datasets with shuffle buffers
    train_data, val_data, text_fn = load_dataset_for_training(
        dataset,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        # FinePDFs-Edu options
        finepdfs_filter_english=finepdfs_filter_english,
        finepdfs_english_threshold=finepdfs_english_threshold,
        # FineWeb-Edu options
        fineweb_subset=fineweb_subset,
        fineweb_min_edu_score=fineweb_min_edu_score,
        fineweb_language=fineweb_language,
        fineweb_min_lang_score=fineweb_min_lang_score,
    )

    # Create streaming text datasets that tokenize on-the-fly
    train_dataset = StreamingTextDataset(
        dataset=train_data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_fn=text_fn,
    )
    val_dataset = StreamingTextDataset(
        dataset=val_data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_fn=text_fn,
    )

    # Note: num_workers=0 is required for HuggingFace datasets to avoid pickling issues
    # pin_memory=True speeds up GPU transfers when using CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    
    # Create training iterator
    train_iter = iter(train_loader)
    
    # Cache fixed validation batches for consistent evaluation across training
    # This ensures validation loss is comparable between different steps
    print(f"Caching {val_batches} validation batches...")
    val_iter = iter(val_loader)
    cached_val_batches = [next(val_iter) for _ in range(val_batches)]
    print(f"  Cached {len(cached_val_batches)} batches ({len(cached_val_batches) * batch_size * seq_len:,} tokens)")

    # Create model
    # Note: Use len(tokenizer) not tokenizer.vocab_size because
    # Llama tokenizers have added special tokens beyond the base vocab
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        context_length=seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,  # None means same as num_heads (MHA)
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        dropout=dropout,
        rope_theta=rope_theta,
    )
    model = Llama(config)
    model = model.to(device)

    # Apply torch.compile for faster execution (PyTorch 2.0+)
    if use_compile and device == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
        model = torch.compile(model, mode="default")  # type: ignore[assignment]
        print("Model compiled successfully!")

    # Calculate number of optimizer steps from num_tokens
    # num_tokens is in billions, convert to actual token count
    num_tokens_actual = int(num_tokens * 1e9)
    # StreamingTextDataset yields (seq_len + 1) tokens, giving us seq_len predictions
    # Each optimizer step processes batch_size * seq_len * gradient_accumulate tokens
    tokens_per_batch = batch_size * seq_len
    tokens_per_step = tokens_per_batch * gradient_accumulate
    num_steps = num_tokens_actual // tokens_per_step
    print(f"Training configuration:")
    print(f"  Total tokens: {num_tokens_actual:,} ({num_tokens}B)")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Gradient accumulation: {gradient_accumulate}")
    print(f"  Tokens per microbatch: {tokens_per_batch:,}")
    print(f"  Tokens per optimizer step: {tokens_per_step:,}")
    print(f"  Number of optimizer steps: {num_steps:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler with warmup, constant, and cooldown phases
    warmup_steps = round(warmup_ratio * num_steps)
    cooldown_steps = round(cooldown_ratio * num_steps)
    
    def get_lr_multiplier(step: int) -> float:
        """Get LR multiplier for the current step.
        
        - Warmup: linear increase from min_learning_rate_frac to 1.0
        - Constant: stays at 1.0
        - Cooldown: linear decrease from 1.0 to min_learning_rate_frac
        """
        if step < warmup_steps:
            # Linear warmup from min_learning_rate_frac to 1.0
            progress = (step + 1) / warmup_steps
            return min_learning_rate_frac + progress * (1.0 - min_learning_rate_frac)
        elif step <= num_steps - cooldown_steps:
            # Constant phase at peak LR
            return 1.0
        else:
            # Linear cooldown from 1.0 to min_learning_rate_frac
            progress = (num_steps - step) / cooldown_steps
            return progress * 1.0 + (1 - progress) * min_learning_rate_frac
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    print(f"Learning rate schedule:")
    print(f"  Warmup steps: {warmup_steps:,} ({warmup_ratio*100:.1f}%)")
    print(f"  Cooldown steps: {cooldown_steps:,} ({cooldown_ratio*100:.1f}%)")
    print(f"  Constant steps: {num_steps - warmup_steps - cooldown_steps:,}")
    print(f"  Peak LR: {learning_rate:.2e}")
    print(f"  Min LR: {learning_rate * min_learning_rate_frac:.2e} ({min_learning_rate_frac*100:.1f}% of peak)")

    # Mixed precision training scaler
    scaler = GradScaler(device=device) if use_amp and device == "cuda" else None
    if scaler:
        print("Mixed precision training (AMP) enabled")

    # Initialize wandb
    wandb.init(  # type: ignore[attr-defined]
        project=wandb_project,
        name=wandb_run_name,
        mode="online" if track_online else "offline",
        config={
            "model": "llama",
            "tokenizer": tokenizer_name,
            "dataset": dataset.value,
            "shuffle_buffer_size": shuffle_buffer_size,
            # FinePDFs-Edu config
            "finepdfs_filter_english": finepdfs_filter_english,
            "finepdfs_english_threshold": finepdfs_english_threshold,
            # FineWeb-Edu config
            "fineweb_subset": fineweb_subset,
            "fineweb_min_edu_score": fineweb_min_edu_score,
            "fineweb_language": fineweb_language,
            "fineweb_min_lang_score": fineweb_min_lang_score,
            "seed": seed,
            "num_tokens": num_tokens_actual,
            "num_tokens_billions": num_tokens,
            "num_steps": num_steps,
            "tokens_per_step": tokens_per_step,
            "batch_size": batch_size,
            "gradient_accumulate": gradient_accumulate,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "cooldown_ratio": cooldown_ratio,
            "min_learning_rate_frac": min_learning_rate_frac,
            "warmup_steps": warmup_steps,
            "cooldown_steps": cooldown_steps,
            "use_amp": use_amp and device == "cuda",
            "use_compile": use_compile and device == "cuda",
            "seq_len": seq_len,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "num_layers": num_layers,
            "intermediate_size": intermediate_size,
            "dropout": dropout,
            "rope_theta": rope_theta,
            "model_params": model.get_num_params(),
        },
    )

    # Setup JSON logging
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = logs_dir / f"metrics_{timestamp}_{wandb_run_name}.json"
    metrics_history = []

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    tokens_seen = 0
    for step in tqdm.tqdm(range(num_steps), desc="Training"):
        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for micro_step in range(gradient_accumulate):
            batch = next(train_iter).to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            if scaler:
                with autocast(device_type=device):
                    _, loss = model(input_ids, targets)
                    loss = loss / gradient_accumulate
                scaler.scale(loss).backward()
            else:
                _, loss = model(input_ids, targets)
                loss = loss / gradient_accumulate
                loss.backward()

            accumulated_loss += loss.item()

        # Gradient clipping and optimizer step
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        tokens_seen += tokens_per_step

        wandb.log({"train_loss": accumulated_loss, "lr": current_lr, "step": step, "tokens_seen": tokens_seen})  # type: ignore[attr-defined]
        metrics_history.append({"step": step, "train_loss": accumulated_loss, "lr": current_lr, "tokens_seen": tokens_seen})
        if step % 10 == 0:
            tqdm.tqdm.write(f"step {step:5d} | train loss: {accumulated_loss:.4f} | lr: {current_lr:.2e} | tokens: {tokens_seen:,}")

        # Validation (uses cached batches for consistent comparison across training)
        if step % validate_every == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_batch in cached_val_batches:
                    val_batch = val_batch.to(device, non_blocking=True)
                    val_input = val_batch[:, :-1]
                    val_targets = val_batch[:, 1:]
                    if scaler:
                        with autocast(device_type=device):
                            _, val_loss = model(val_input, val_targets)
                    else:
                        _, val_loss = model(val_input, val_targets)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(cached_val_batches)
            wandb.log({"val_loss": avg_val_loss, "step": step, "tokens_seen": tokens_seen})  # type: ignore[attr-defined]
            metrics_history.append({"step": step, "val_loss": avg_val_loss, "tokens_seen": tokens_seen})
            tqdm.tqdm.write(f"step {step:5d} | val loss: {avg_val_loss:.4f} (avg of {len(cached_val_batches)} batches)")

        # Generation
        if step % generate_every == 0 and step > 0:
            model.eval()

            # Use a fixed prompt for generation
            prompt_text = "An interesting fact about"
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt = tokenizer.decode(prompt_tokens)

            tqdm.tqdm.write("\n" + "=" * 60)
            tqdm.tqdm.write(f"PROMPT:\n{prompt}")
            tqdm.tqdm.write("-" * 60)

            # Generate (stop early if EOS token is produced)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=generate_length,
                temperature=0.8,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
            )

            output = decode_tokens(tokenizer, generated[0])
            tqdm.tqdm.write(f"GENERATED:\n{output}")
            tqdm.tqdm.write("=" * 60 + "\n")

    # Save metrics to JSON
    with open(metrics_file, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    wandb.finish()  # type: ignore[attr-defined]
    print("Training complete!")


if __name__ == "__main__":
    app()

