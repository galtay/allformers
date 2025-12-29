"""
Train GPT-2 on various text datasets

A simple training script for language modeling with GPT-2,
inspired by https://github.com/lucidrains/x-transformers/blob/main/train_enwik8.py

Uses HuggingFace streaming datasets with shuffle buffers for memory-efficient
training on large datasets. See: https://huggingface.co/docs/datasets/main/stream

Supports DDP (Distributed Data Parallel) training for multi-GPU setups.

Usage:
    # Single GPU
    uv run python train_gpt2.py --dataset wikipedia --num-tokens 0.1
    
    # Multi-GPU with DDP (e.g., 4 GPUs)
    uv run torchrun --nproc_per_node=4 train_gpt2.py --dataset wikipedia --num-tokens 0.1
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import tqdm
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import typer
import wandb
import datasets

from allformers.models.gpt2.gpt2 import GPT2, GPT2Config
from allformers.data.streaming import StreamingTextDataset
from allformers.data.wikipedia import load_wikipedia_streaming, wikipedia_text_fn
from allformers.data.finepdfs_edu import load_finepdfs_edu_streaming, finepdfs_edu_text_fn
from allformers.data.fineweb_edu import load_fineweb_edu_streaming, fineweb_edu_text_fn
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
# DDP (Distributed Data Parallel) Helpers
# =============================================================================


def is_ddp() -> bool:
    """Check if we're running in DDP mode."""
    return dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process (0 if not in DDP mode)."""
    return dist.get_rank() if is_ddp() else 0


def get_world_size() -> int:
    """Get the total number of processes (1 if not in DDP mode)."""
    return dist.get_world_size() if is_ddp() else 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_ddp():
    """Initialize DDP if running with torchrun/distributed launch.
    
    Returns:
        Tuple of (rank, world_size, local_rank, device)
    """
    # Check if we're running in distributed mode
    # torchrun sets these environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        
        if rank == 0:
            print(f"DDP initialized: rank {rank}/{world_size}, local_rank {local_rank}")
        
        return rank, world_size, local_rank, device
    else:
        # Single process mode
        device = get_device()
        return 0, 1, 0, device


def cleanup_ddp():
    """Clean up DDP resources."""
    if is_ddp():
        dist.destroy_process_group()


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch CPU
    - PyTorch CUDA (all devices)
    
    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms which may
            impact performance but ensures fully reproducible results.
    """
    import random
    random.seed(seed)
    
    # NumPy (optional, but good to have)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Optionally enable deterministic algorithms (may impact performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN to auto-tune for best performance
        torch.backends.cudnn.benchmark = True


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
    fineweb_subset: Annotated[str, typer.Option(help="[fineweb] Subset: default, sample-10BT, sample-100BT, sample-350BT, CC-MAIN-*")] = "default",
    fineweb_min_edu_score: Annotated[float | None, typer.Option(help="[fineweb] Min educational score (higher=more educational)")] = None,
    fineweb_language: Annotated[str | None, typer.Option(help="[fineweb] Filter by language column (e.g., 'en'), None to disable")] = "en",
    fineweb_min_lang_score: Annotated[float | None, typer.Option(help="[fineweb] Min language detection confidence (0.0-1.0)")] = None,
    # Training hyperparameters
    num_tokens: Annotated[float, typer.Option(help="Total number of tokens to train on (in billions, e.g., 0.01 = 10M tokens)")] = 0.01,
    batch_size: Annotated[int, typer.Option(help="Batch size per GPU")] = 32,
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
    # Model settings (GPT-2 specific)
    embedding_dim: Annotated[int, typer.Option(help="Model embedding dimension")] = 768,
    num_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 12,
    num_layers: Annotated[int, typer.Option(help="Number of transformer layers")] = 12,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
    # Wandb settings
    wandb_project: Annotated[str, typer.Option(help="Wandb project name")] = "allformers-gpt2",
    wandb_run_name: Annotated[str, typer.Option(help="Wandb run name (auto-generated if not provided)")] = "",
    track_online: Annotated[bool, typer.Option(help="Track experiment online with wandb")] = True,
    # Reproducibility
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    deterministic: Annotated[bool, typer.Option(help="Use deterministic algorithms (slower but fully reproducible)")] = False,
):
    """Train GPT-2 on the specified dataset.
    
    Supports both single-GPU and multi-GPU (DDP) training.
    For multi-GPU, launch with torchrun:
        torchrun --nproc_per_node=N train_gpt2.py [options]
    """
    # Setup DDP if running in distributed mode
    rank, world_size, local_rank, device = setup_ddp()
    
    # Disable HuggingFace datasets progress bars on non-main processes
    if not is_main_process():
        datasets.disable_progress_bar()
    
    # Set random seeds for reproducibility (before any model/data initialization)
    set_seed(seed, deterministic=deterministic)
    if is_main_process():
        print(f"Random seed: {seed}" + (" (deterministic mode)" if deterministic else ""))
    
    if is_main_process():
        print(f"Using device: {device}")
        print(f"Dataset: {dataset.value}")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs")

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
    # In DDP mode, each worker gets a unique shard of the data
    ddp_kwargs = {"rank": rank, "world_size": world_size} if world_size > 1 else {}
    train_dataset = StreamingTextDataset(
        dataset=train_data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_fn=text_fn,
        **ddp_kwargs,
    )
    val_dataset = StreamingTextDataset(
        dataset=val_data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_fn=text_fn,
        **ddp_kwargs,
    )

    # Note: num_workers=0 is required for HuggingFace datasets to avoid pickling issues
    # pin_memory=True speeds up GPU transfers when using CUDA
    use_cuda = device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=use_cuda,
    )
    
    # Create training iterator
    train_iter = iter(train_loader)
    
    # Cache fixed validation batches for consistent evaluation across training
    # This ensures validation loss is comparable between different steps
    if is_main_process():
        print(f"Caching {val_batches} validation batches...")
    val_iter = iter(val_loader)
    cached_val_batches = [next(val_iter) for _ in range(val_batches)]
    if is_main_process():
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
    
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"GPT-2 model with {n_params:,} parameters")

    # Apply torch.compile for faster execution (PyTorch 2.0+)
    if use_compile and use_cuda and hasattr(torch, "compile"):
        if is_main_process():
            print("Compiling model with torch.compile...")
        # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
        model = torch.compile(model, mode="default")  # type: ignore[assignment]
        if is_main_process():
            print("Model compiled successfully!")
    
    # Wrap model in DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])  # type: ignore[assignment]
        if is_main_process():
            print(f"Model wrapped in DDP")

    # Calculate number of optimizer steps from num_tokens
    # num_tokens is in billions, convert to actual token count
    num_tokens_actual = int(num_tokens * 1e9)
    # StreamingTextDataset yields (seq_len + 1) tokens, giving us seq_len predictions
    # Each optimizer step processes batch_size * seq_len * gradient_accumulate * world_size tokens
    # (In DDP, each GPU processes batch_size samples, so global batch = batch_size * world_size)
    tokens_per_batch = batch_size * seq_len  # Per GPU
    global_tokens_per_batch = tokens_per_batch * world_size  # Across all GPUs
    tokens_per_step = global_tokens_per_batch * gradient_accumulate
    num_steps = num_tokens_actual // tokens_per_step
    if is_main_process():
        print(f"Training configuration:")
        print(f"  Total tokens: {num_tokens_actual:,} ({num_tokens}B)")
        print(f"  Number of GPUs: {world_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Global batch size: {batch_size * world_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Gradient accumulation: {gradient_accumulate}")
        print(f"  Tokens per microbatch (per GPU): {tokens_per_batch:,}")
        print(f"  Tokens per optimizer step (global): {tokens_per_step:,}")
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
    
    if is_main_process():
        print(f"Learning rate schedule:")
        print(f"  Warmup steps: {warmup_steps:,} ({warmup_ratio*100:.1f}%)")
        print(f"  Cooldown steps: {cooldown_steps:,} ({cooldown_ratio*100:.1f}%)")
        print(f"  Constant steps: {num_steps - warmup_steps - cooldown_steps:,}")
        print(f"  Peak LR: {learning_rate:.2e}")
        print(f"  Min LR: {learning_rate * min_learning_rate_frac:.2e} ({min_learning_rate_frac*100:.1f}% of peak)")

    # Mixed precision training scaler
    scaler = GradScaler(device=device) if use_amp and use_cuda else None
    if scaler and is_main_process():
        print("Mixed precision training (AMP) enabled")

    # Get model params (handle DDP wrapper)
    model_for_params = model.module if isinstance(model, DDP) else model
    num_params = model_for_params.get_num_params()

    # Initialize wandb (only on main process)
    if is_main_process():
        wandb.init(  # type: ignore[attr-defined]
            project=wandb_project,
            name=wandb_run_name,
            mode="online" if track_online else "offline",
            config={
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
                "global_batch_size": batch_size * world_size,
                "gradient_accumulate": gradient_accumulate,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "cooldown_ratio": cooldown_ratio,
                "min_learning_rate_frac": min_learning_rate_frac,
                "warmup_steps": warmup_steps,
                "cooldown_steps": cooldown_steps,
                "use_amp": use_amp and use_cuda,
                "use_compile": use_compile and use_cuda,
                "seq_len": seq_len,
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "model_params": num_params,
                # DDP config
                "world_size": world_size,
                "ddp": world_size > 1,
            },
        )

    # Setup JSON logging (only on main process)
    metrics_history = []
    metrics_file = None
    if is_main_process():
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = logs_dir / f"metrics_{timestamp}_{wandb_run_name}.json"

    # Training loop
    if is_main_process():
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

    tokens_seen = 0
    pbar = tqdm.tqdm(range(num_steps), desc="Training", disable=not is_main_process())
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for micro_step in range(gradient_accumulate):
            batch = next(train_iter).to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # For DDP, we need to specify the device type without the index
            device_type = "cuda" if use_cuda else device
            if scaler:
                with autocast(device_type=device_type):
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

        # Logging (only on main process)
        if is_main_process():
            wandb.log({"train_loss": accumulated_loss, "lr": current_lr, "step": step, "tokens_seen": tokens_seen})  # type: ignore[attr-defined]
            metrics_history.append({"step": step, "train_loss": accumulated_loss, "lr": current_lr, "tokens_seen": tokens_seen})
            if step % 10 == 0:
                tqdm.tqdm.write(f"step {step:5d} | train loss: {accumulated_loss:.4f} | lr: {current_lr:.2e} | tokens: {tokens_seen:,}")

        # Validation (uses cached batches for consistent comparison across training)
        # Only run on main process to avoid redundant computation
        if step % validate_every == 0 and is_main_process():
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_batch in cached_val_batches:
                    val_batch = val_batch.to(device, non_blocking=True)
                    val_input = val_batch[:, :-1]
                    val_targets = val_batch[:, 1:]
                    if scaler:
                        with autocast(device_type=device_type):
                            _, val_loss = model(val_input, val_targets)
                    else:
                        _, val_loss = model(val_input, val_targets)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(cached_val_batches)
            wandb.log({"val_loss": avg_val_loss, "step": step, "tokens_seen": tokens_seen})  # type: ignore[attr-defined]
            metrics_history.append({"step": step, "val_loss": avg_val_loss, "tokens_seen": tokens_seen})
            tqdm.tqdm.write(f"step {step:5d} | val loss: {avg_val_loss:.4f} (avg of {len(cached_val_batches)} batches)")

        # Generation (only on main process)
        if step % generate_every == 0 and step > 0 and is_main_process():
            model.eval()

            # Use a fixed prompt for generation
            prompt_text = "An interesting fact about"
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt = tokenizer.decode(prompt_tokens)

            tqdm.tqdm.write("\n" + "=" * 60)
            tqdm.tqdm.write(f"PROMPT:\n{prompt}")
            tqdm.tqdm.write("-" * 60)

            # Generate (stop early if EOS token is produced)
            # Use the underlying model for generation (not DDP wrapper)
            gen_model = model.module if isinstance(model, DDP) else model
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
            generated = gen_model.generate(
                prompt_tensor,
                max_new_tokens=generate_length,
                temperature=0.8,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
            )

            output = decode_tokens(tokenizer, generated[0])
            tqdm.tqdm.write(f"GENERATED:\n{output}")
            tqdm.tqdm.write("=" * 60 + "\n")

    # Save metrics to JSON (only on main process)
    if is_main_process() and metrics_file is not None:
        with open(metrics_file, "w") as f:
            json.dump(metrics_history, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
        wandb.finish()  # type: ignore[attr-defined]
        print("Training complete!")
    
    # Clean up DDP
    cleanup_ddp()


if __name__ == "__main__":
    app()
