"""
Train GPT-2 on various text datasets

A simple training script for language modeling with GPT-2,
inspired by https://github.com/lucidrains/x-transformers/blob/main/train_enwik8.py

Uses HuggingFace streaming datasets with shuffle buffers for memory-efficient
training on large datasets. See: https://huggingface.co/docs/datasets/main/stream

Usage:
    uv run python train_gpt2.py --help
    uv run python train_gpt2.py --dataset wikipedia --num-tokens 0.1
    uv run python train_gpt2.py --dataset finepdfs-edu --num-tokens 0.1
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import tqdm
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import typer
import wandb

from allformers.models.gpt2.gpt2 import GPT2, GPT2Config
from allformers.data.streaming import StreamingTextDataset
from allformers.data.wikipedia import load_wikipedia_streaming, wikipedia_text_fn
from allformers.data.finepdfs_edu import load_finepdfs_edu_streaming, finepdfs_edu_text_fn
from transformers import AutoTokenizer


app = typer.Typer(
    help="Train GPT-2 on text datasets",
    add_completion=False,
)


# =============================================================================
# Dataset Configuration
# =============================================================================


class DatasetChoice(str, Enum):
    """Available datasets for training."""
    wikipedia = "wikipedia"
    finepdfs_edu = "finepdfs-edu"


def load_dataset_for_training(
    dataset: DatasetChoice,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    filter_english: bool = True,
    english_threshold: float = 0.5,
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
            filter_english=filter_english,
            english_threshold=english_threshold,
        )
        return train_data, val_data, finepdfs_edu_text_fn
    
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
    dataset: Annotated[DatasetChoice, typer.Option(help="Dataset to train on")] = DatasetChoice.finepdfs_edu,
    shuffle_buffer_size: Annotated[int, typer.Option(help="Shuffle buffer size for streaming datasets")] = 10_000,
    # FinePDFs-Edu specific options
    filter_english: Annotated[bool, typer.Option(help="Filter FinePDFs-Edu for majority English documents")] = True,
    english_threshold: Annotated[float, typer.Option(help="Fraction of pages that must be English (0.0-1.0), docs with > threshold kept")] = 0.8,
    # Training hyperparameters
    num_tokens: Annotated[float, typer.Option(help="Total number of tokens to train on (in billions, e.g., 0.01 = 10M tokens)")] = 0.01,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    gradient_accumulate: Annotated[int, typer.Option(help="Gradient accumulation steps")] = 1,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 3e-4,
    seq_len: Annotated[int, typer.Option(help="Sequence length")] = 512,
    # Performance optimizations
    use_amp: Annotated[bool, typer.Option(help="Use mixed precision training (AMP)")] = True,
    use_compile: Annotated[bool, typer.Option(help="Use torch.compile for model optimization")] = True,
    # Logging intervals
    validate_every: Annotated[int, typer.Option(help="Validate every N batches")] = 100,
    val_batches: Annotated[int, typer.Option(help="Number of batches for validation")] = 64,
    generate_every: Annotated[int, typer.Option(help="Generate samples every N batches")] = 500,
    generate_length: Annotated[int, typer.Option(help="Length of generated samples")] = 256,
    # Model settings (GPT-2 specific)
    embedding_dim: Annotated[int, typer.Option(help="Model embedding dimension")] = 512,
    num_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 8,
    num_layers: Annotated[int, typer.Option(help="Number of transformer layers")] = 6,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
    # Wandb settings
    wandb_project: Annotated[str, typer.Option(help="Wandb project name")] = "allformers-gpt2",
    wandb_run_name: Annotated[str, typer.Option(help="Wandb run name (auto-generated if not provided)")] = "",
    track_online: Annotated[bool, typer.Option(help="Track experiment online with wandb")] = True,
    # Reproducibility
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
):
    """Train GPT-2 on the specified dataset."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset.value}")

    # Auto-generate run name if not provided
    if not wandb_run_name:
        wandb_run_name = f"gpt2-{dataset.value}"

    # Initialize tokenizer (using transformers for flexibility across models)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

    # Load streaming datasets with shuffle buffers
    train_data, val_data, text_fn = load_dataset_for_training(
        dataset,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        filter_english=filter_english,
        english_threshold=english_threshold,
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
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        context_length=seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = GPT2(config)
    model = model.to(device)

    # Apply torch.compile for faster execution (PyTorch 2.0+)
    if use_compile and device == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
        model = torch.compile(model, mode="default")
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

    # Mixed precision training scaler
    scaler = GradScaler(device=device) if use_amp and device == "cuda" else None
    if scaler:
        print("Mixed precision training (AMP) enabled")

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode="online" if track_online else "offline",
        config={
            "dataset": dataset.value,
            "shuffle_buffer_size": shuffle_buffer_size,
            "filter_english": filter_english,
            "english_threshold": english_threshold,
            "seed": seed,
            "num_tokens": num_tokens_actual,
            "num_tokens_billions": num_tokens,
            "num_steps": num_steps,
            "tokens_per_step": tokens_per_step,
            "batch_size": batch_size,
            "gradient_accumulate": gradient_accumulate,
            "learning_rate": learning_rate,
            "use_amp": use_amp and device == "cuda",
            "use_compile": use_compile and device == "cuda",
            "seq_len": seq_len,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
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

        tokens_seen += tokens_per_step

        wandb.log({"train_loss": accumulated_loss, "step": step, "tokens_seen": tokens_seen})
        metrics_history.append({"step": step, "train_loss": accumulated_loss, "tokens_seen": tokens_seen})
        if step % 10 == 0:
            tqdm.tqdm.write(f"step {step:5d} | train loss: {accumulated_loss:.4f} | tokens: {tokens_seen:,}")

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
            wandb.log({"val_loss": avg_val_loss, "step": step, "tokens_seen": tokens_seen})
            metrics_history.append({"step": step, "val_loss": avg_val_loss, "tokens_seen": tokens_seen})
            tqdm.tqdm.write(f"step {step:5d} | val loss: {avg_val_loss:.4f} (avg of {len(cached_val_batches)} batches)")

        # Generation
        if step % generate_every == 0 and step > 0:
            model.eval()

            # Use a fixed prompt for generation
            prompt_text = "The quick brown fox"
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

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    app()
