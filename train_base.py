"""
Common utilities for training transformer models.

This module provides shared functionality for training scripts:
- Dataset loading and configuration
- DataLoader creation with validation caching
- Device detection and DDP (Distributed Data Parallel) utilities
- Learning rate scheduling
- Training loop
- Reproducibility utilities

Usage:
    from train_base import (
        DatasetChoice,
        load_dataset_for_training,
        create_dataloaders,
        get_device,
        decode_tokens,
        set_seed,
        setup_ddp,
        cleanup_ddp,
        is_main_process,
        create_lr_schedule,
        run_training_loop,
    )
"""

import json
import os
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import datasets
import tqdm
import torch
import torch.distributed as dist
import wandb
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from allformers.data.streaming import StreamingTextDataset
from allformers.data.wikipedia import load_wikipedia_streaming, wikipedia_text_fn
from allformers.data.finepdfs_edu import load_finepdfs_edu_streaming, finepdfs_edu_text_fn
from allformers.data.fineweb_edu import load_fineweb_edu_streaming, fineweb_edu_text_fn


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
    # Verbose output
    verbose: bool = True,
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
            verbose=verbose,
        )
        return train_data, val_data, wikipedia_text_fn
    
    elif dataset == DatasetChoice.finepdfs_edu:
        train_data, val_data = load_finepdfs_edu_streaming(
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            filter_english=finepdfs_filter_english,
            english_threshold=finepdfs_english_threshold,
            verbose=verbose,
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
            verbose=verbose,
        )
        return train_data, val_data, fineweb_edu_text_fn
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_dataloaders(
    train_data: datasets.IterableDataset,
    val_data: datasets.IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    text_fn: Callable[[dict[str, Any]], str],
    seq_len: int,
    batch_size: int,
    val_batches: int,
    device: str,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Iterator[torch.Tensor], list[torch.Tensor]]:
    """Create training iterator and cached validation batches.
    
    Args:
        train_data: Training IterableDataset from load_dataset_for_training
        val_data: Validation IterableDataset from load_dataset_for_training
        tokenizer: HuggingFace tokenizer
        text_fn: Function to extract text from dataset rows
        seq_len: Sequence length for tokenization
        batch_size: Batch size per GPU
        val_batches: Number of validation batches to cache
        device: Device string (e.g., "cuda", "cuda:0", "cpu")
        rank: Process rank for DDP (0 if not using DDP)
        world_size: Total number of processes (1 if not using DDP)
        
    Returns:
        Tuple of (train_iter, cached_val_batches)
        - train_iter: Iterator over training batches
        - cached_val_batches: List of pre-fetched validation batches
    """
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
    
    return train_iter, cached_val_batches


# =============================================================================
# Device and General Helpers
# =============================================================================


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(tokenizer, tokens) -> str:
    """Decode token IDs to string."""
    return tokenizer.decode(tokens.tolist())


# =============================================================================
# Reproducibility
# =============================================================================


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
# DDP (Distributed Data Parallel) Utilities
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


def setup_ddp() -> tuple[int, int, int, str]:
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


# =============================================================================
# Learning Rate Scheduling
# =============================================================================


def create_lr_schedule(
    num_steps: int,
    warmup_ratio: float = 0.05,
    cooldown_ratio: float = 0.5,
    min_lr_frac: float = 0.1,
) -> tuple[Callable[[int], float], int, int]:
    """Create a learning rate multiplier function for LambdaLR.
    
    The schedule has three phases:
    1. Warmup: Linear increase from min_lr_frac to 1.0
    2. Constant: Stays at 1.0 (peak LR)
    3. Cooldown: Linear decrease from 1.0 to min_lr_frac
    
    Args:
        num_steps: Total number of training steps
        warmup_ratio: Fraction of training for warmup phase
        cooldown_ratio: Fraction of training for cooldown phase
        min_lr_frac: Minimum LR as fraction of peak LR
        
    Returns:
        A function that takes a step number and returns the LR multiplier
    """
    warmup_steps = round(warmup_ratio * num_steps)
    cooldown_steps = round(cooldown_ratio * num_steps)
    
    def get_lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from min_lr_frac to 1.0
            progress = (step + 1) / warmup_steps
            return min_lr_frac + progress * (1.0 - min_lr_frac)
        elif step <= num_steps - cooldown_steps:
            # Constant phase at peak LR
            return 1.0
        else:
            # Linear cooldown from 1.0 to min_lr_frac
            progress = (num_steps - step) / cooldown_steps
            return progress * 1.0 + (1 - progress) * min_lr_frac
    
    return get_lr_multiplier, warmup_steps, cooldown_steps


# =============================================================================
# Training Loop
# =============================================================================


def run_training_loop(
    model: torch.nn.Module,
    train_iter: Iterator[torch.Tensor],
    cached_val_batches: list[torch.Tensor],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler | None,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    num_steps: int,
    gradient_accumulate: int,
    tokens_per_step: int,
    validate_every: int,
    generate_every: int,
    generate_length: int,
    wandb_run_name: str,
    metrics_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the main training loop.
    
    This function handles:
    - Gradient accumulation with optional AMP
    - Gradient clipping and optimizer steps
    - Periodic validation on cached batches
    - Periodic text generation
    - Logging to wandb and JSON metrics file
    
    Args:
        model: The model to train (may be wrapped in DDP)
        train_iter: Iterator over training batches
        cached_val_batches: Pre-fetched validation batches
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for AMP (None if not using AMP)
        tokenizer: HuggingFace tokenizer (for generation)
        device: Device string
        num_steps: Total number of optimizer steps
        gradient_accumulate: Number of gradient accumulation steps
        tokens_per_step: Tokens processed per optimizer step (for logging)
        validate_every: Run validation every N steps
        generate_every: Generate samples every N steps
        generate_length: Max tokens to generate
        wandb_run_name: Run name for metrics file naming
        metrics_file: Path to save JSON metrics (None to skip)
        
    Returns:
        List of metrics dictionaries
    """
    use_cuda = device.startswith("cuda")
    device_type = "cuda" if use_cuda else device
    
    metrics_history: list[dict[str, Any]] = []
    
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
            generated = gen_model.generate(  # type: ignore[union-attr]
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
    
    return metrics_history


def setup_metrics_file(run_name: str) -> Path:
    """Create logs directory and return path for metrics JSON file.
    
    Args:
        run_name: Name of the training run (used in filename)
        
    Returns:
        Path to the metrics JSON file
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"metrics_{timestamp}_{run_name}.json"

