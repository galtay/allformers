"""
Train GPT-2 on English Wikipedia

A simple training script for language modeling on Wikipedia,
inspired by https://github.com/lucidrains/x-transformers/blob/main/train_enwik8.py

Usage:
    uv run python train_wikipedia.py --help
    uv run python train_wikipedia.py --num-batches 1000 --track-online
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated

import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import typer
import wandb

from allformers.models.gpt2.gpt2 import GPT2, GPT2Config
from allformers.data import load_wikipedia, WikipediaConfig, StreamingTextDataset
from transformers import AutoTokenizer


app = typer.Typer(
    help="Train GPT-2 on English Wikipedia",
    add_completion=False,
)


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
# Data Preparation
# =============================================================================


def load_datasets(num_train: int, num_val: int):
    """Load Wikipedia datasets for training and validation.
    
    Args:
        num_train: Number of training articles to load.
        num_val: Number of validation articles to load.
        
    Returns:
        Tuple of (train_dataset, val_dataset) HuggingFace datasets.
    """
    print(f"Loading {num_train} training articles and {num_val} validation articles...")

    # Load training data
    train_config = WikipediaConfig(split=f"train[:{num_train}]")
    train_dataset = load_wikipedia(train_config)

    # Load validation data (from a different slice)
    val_config = WikipediaConfig(split=f"train[{num_train}:{num_train + num_val}]")
    val_dataset = load_wikipedia(val_config)

    print(f"Train articles: {len(train_dataset):,}")
    print(f"Val articles: {len(val_dataset):,}")

    return train_dataset, val_dataset


# =============================================================================
# Main
# =============================================================================


@app.command()
def train(
    # Training hyperparameters
    num_batches: Annotated[int, typer.Option(help="Number of training batches")] = 10000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 4,
    gradient_accumulate: Annotated[int, typer.Option(help="Gradient accumulation steps")] = 1,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 3e-4,
    seq_len: Annotated[int, typer.Option(help="Sequence length")] = 512,
    # Logging intervals
    validate_every: Annotated[int, typer.Option(help="Validate every N batches")] = 100,
    val_batches: Annotated[int, typer.Option(help="Number of batches for validation")] = 10,
    generate_every: Annotated[int, typer.Option(help="Generate samples every N batches")] = 500,
    generate_length: Annotated[int, typer.Option(help="Length of generated samples")] = 256,
    # Data settings
    num_train_articles: Annotated[int, typer.Option(help="Number of training articles")] = 10000,
    num_val_articles: Annotated[int, typer.Option(help="Number of validation articles")] = 1000,
    # Model settings
    embedding_dim: Annotated[int, typer.Option(help="Model embedding dimension")] = 512,
    num_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 8,
    num_layers: Annotated[int, typer.Option(help="Number of transformer layers")] = 6,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
    # Wandb settings
    wandb_project: Annotated[str, typer.Option(help="Wandb project name")] = "allformers-wikipedia",
    wandb_run_name: Annotated[str, typer.Option(help="Wandb run name")] = "gpt2-small",
    track_online: Annotated[bool, typer.Option(help="Track experiment online with wandb")] = True,
):
    """Train GPT-2 on English Wikipedia."""
    device = get_device()
    print(f"Using device: {device}")

    # Initialize tokenizer (using transformers for flexibility across models)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

    # Load datasets (not tokenized yet - tokenization happens on-the-fly)
    train_data, val_data = load_datasets(
        num_train=num_train_articles,
        num_val=num_val_articles,
    )

    # Create streaming datasets that tokenize on-the-fly
    # This avoids loading all tokens into memory upfront
    train_dataset = StreamingTextDataset(
        dataset=train_data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=42,
    )
    val_dataset = StreamingTextDataset(
        dataset=val_data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=123,  # Different seed for validation
    )

    # Note: num_workers=0 is required for HuggingFace datasets to avoid pickling issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    # Create iterators (streaming datasets are already infinite)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

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

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode="online" if track_online else "offline",
        config={
            "num_batches": num_batches,
            "batch_size": batch_size,
            "gradient_accumulate": gradient_accumulate,
            "learning_rate": learning_rate,
            "seq_len": seq_len,
            "num_train_articles": num_train_articles,
            "num_val_articles": num_val_articles,
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

    for i in tqdm.tqdm(range(num_batches), mininterval=10.0, desc="Training"):
        model.train()

        # Gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(gradient_accumulate):
            batch = next(train_iter).to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            _, loss = model(input_ids, targets)
            (loss / gradient_accumulate).backward()
            total_loss += loss.item()

        avg_loss = total_loss / gradient_accumulate

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Logging
        wandb.log({"train_loss": avg_loss, "step": i})
        metrics_history.append({"step": i, "train_loss": avg_loss})
        if i % 10 == 0:
            tqdm.tqdm.write(f"step {i:5d} | train loss: {avg_loss:.4f}")

        # Validation
        if i % validate_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(val_batches):
                    val_batch = next(val_iter).to(device)
                    val_input = val_batch[:, :-1]
                    val_targets = val_batch[:, 1:]
                    _, val_loss = model(val_input, val_targets)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"val_loss": avg_val_loss, "step": i})
            metrics_history.append({"step": i, "val_loss": avg_val_loss})
            tqdm.tqdm.write(f"step {i:5d} | val loss: {avg_val_loss:.4f} (avg of {val_batches} batches)")

        # Generation
        if i % generate_every == 0 and i > 0:
            model.eval()

            # Get a random article from validation set as prompt
            rand_idx = random.randint(0, len(val_data) - 1)
            article = val_data[rand_idx]
            prompt_text = f"{article['title']}\n\n{article['text'][:500]}"
            prompt_tokens = tokenizer.encode(prompt_text)[:seq_len // 4]
            prompt = tokenizer.decode(prompt_tokens)

            tqdm.tqdm.write("\n" + "=" * 60)
            tqdm.tqdm.write(f"PROMPT:\n{prompt}")
            tqdm.tqdm.write("-" * 60)

            # Generate
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=generate_length,
                temperature=0.8,
                top_k=40,
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
