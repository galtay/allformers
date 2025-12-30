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

from typing import Annotated

import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import typer
import wandb
import datasets

from allformers.models.gpt2.gpt2 import GPT2, GPT2Config
from transformers import AutoTokenizer

from train_base import (
    DatasetChoice,
    load_dataset_for_training,
    create_dataloaders,
    set_seed,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    create_lr_schedule,
    run_training_loop,
    setup_metrics_file,
)


app = typer.Typer(
    help="Train GPT-2 on text datasets",
    add_completion=False,
)


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
    use_cuda = device.startswith("cuda")
    
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
        finepdfs_filter_english=finepdfs_filter_english,
        finepdfs_english_threshold=finepdfs_english_threshold,
        fineweb_subset=fineweb_subset,
        fineweb_min_edu_score=fineweb_min_edu_score,
        fineweb_language=fineweb_language,
        fineweb_min_lang_score=fineweb_min_lang_score,
        verbose=is_main_process(),
    )

    # Create dataloaders and cache validation batches
    train_iter, cached_val_batches = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        text_fn=text_fn,
        seq_len=seq_len,
        batch_size=batch_size,
        val_batches=val_batches,
        device=device,
        rank=rank,
        world_size=world_size,
    )

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
        model = torch.compile(model, mode="default")  # type: ignore[assignment]
        if is_main_process():
            print("Model compiled successfully!")
    
    # Wrap model in DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])  # type: ignore[assignment]
        if is_main_process():
            print("Model wrapped in DDP")

    # Calculate number of optimizer steps from num_tokens
    num_tokens_actual = int(num_tokens * 1e9)
    tokens_per_batch = batch_size * seq_len
    global_tokens_per_batch = tokens_per_batch * world_size
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

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    get_lr_multiplier, warmup_steps, cooldown_steps = create_lr_schedule(
        num_steps=num_steps,
        warmup_ratio=warmup_ratio,
        cooldown_ratio=cooldown_ratio,
        min_lr_frac=min_learning_rate_frac,
    )
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
                "model": "gpt2",
                "dataset": dataset.value,
                "shuffle_buffer_size": shuffle_buffer_size,
                "finepdfs_filter_english": finepdfs_filter_english,
                "finepdfs_english_threshold": finepdfs_english_threshold,
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
                "world_size": world_size,
                "ddp": world_size > 1,
            },
        )

    # Setup metrics file (only on main process)
    metrics_file = setup_metrics_file(wandb_run_name) if is_main_process() else None

    # Run training loop
    run_training_loop(
        model=model,
        train_iter=train_iter,
        cached_val_batches=cached_val_batches,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tokenizer=tokenizer,
        device=device,
        num_steps=num_steps,
        gradient_accumulate=gradient_accumulate,
        tokens_per_step=tokens_per_step,
        validate_every=validate_every,
        generate_every=generate_every,
        generate_length=generate_length,
        wandb_run_name=wandb_run_name,
        metrics_file=metrics_file,
    )
    
    # Clean up DDP
    cleanup_ddp()


if __name__ == "__main__":
    app()
