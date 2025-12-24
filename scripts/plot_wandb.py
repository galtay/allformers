#!/usr/bin/env python3
"""
Plot training metrics from local logs.

This script reads from:
1. JSON metrics files (metrics.json) - easiest to parse
2. Wandb offline runs (requires syncing first)

Usage:
    uv run python scripts/plot_wandb.py                    # Plot latest run
    uv run python scripts/plot_wandb.py --list             # List available runs
    uv run python scripts/plot_wandb.py --output loss.png  # Save to file
"""

import json
from pathlib import Path
from typing import Annotated, Optional
from datetime import datetime

import typer
import matplotlib.pyplot as plt

app = typer.Typer(help="Plot training metrics from logs", add_completion=False)

WANDB_DIR = Path("wandb")
LOGS_DIR = Path("logs")


def find_wandb_runs(wandb_dir: Path) -> list[Path]:
    """Find all wandb run directories."""
    if not wandb_dir.exists():
        return []
    runs = []
    for path in wandb_dir.iterdir():
        if path.is_dir() and (path.name.startswith("offline-run-") or path.name.startswith("run-")):
            runs.append(path)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def find_json_logs(logs_dir: Path) -> list[Path]:
    """Find all JSON metric log files."""
    if not logs_dir.exists():
        return []
    return sorted(logs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_run_id(run_dir: Path) -> str:
    """Extract run ID from directory name."""
    name = run_dir.name
    parts = name.split("-")
    return parts[-1] if parts else name


def parse_run_time(run_dir: Path) -> Optional[datetime]:
    """Extract timestamp from run directory name."""
    name = run_dir.name
    try:
        if "offline-run-" in name:
            date_part = name.replace("offline-run-", "").split("-")[0]
        else:
            date_part = name.replace("run-", "").split("-")[0]
        return datetime.strptime(date_part, "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        return None


def read_json_metrics(json_file: Path) -> list[dict]:
    """Read metrics from a JSON log file."""
    try:
        with open(json_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "history" in data:
            return data["history"]
    except Exception as e:
        print(f"Warning: Could not read {json_file}: {e}")
    return []


def read_wandb_history_via_sync(run_dir: Path) -> list[dict]:
    """Try to read wandb history by checking for synced data or summary."""
    history = []
    
    # Check for synced history file
    history_file = run_dir / "files" / "wandb-history.jsonl"
    if history_file.exists():
        try:
            with open(history_file) as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
            return history
        except Exception as e:
            print(f"Warning: Could not read history file: {e}")
    
    # Check for summary (at least gives final values)
    summary_file = run_dir / "files" / "wandb-summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            # Return summary as a single history point
            if "train_loss" in summary or "val_loss" in summary:
                return [summary]
        except Exception:
            pass
    
    return history


@app.command("list")
def list_runs(
    wandb_dir: Annotated[str, typer.Option(help="Wandb directory")] = "wandb",
    logs_dir: Annotated[str, typer.Option(help="Logs directory")] = "logs",
):
    """List available runs."""
    wandb_runs = find_wandb_runs(Path(wandb_dir))
    json_logs = find_json_logs(Path(logs_dir))
    
    if not wandb_runs and not json_logs:
        print("No runs found.")
        print(f"  Looked in: {wandb_dir}/, {logs_dir}/")
        return
    
    if wandb_runs:
        print(f"Wandb runs ({len(wandb_runs)}):\n")
        print(f"  {'Run ID':<12} {'Time':<20} {'Directory'}")
        print("  " + "-" * 65)
        for run_dir in wandb_runs:
            run_id = parse_run_id(run_dir)
            run_time = parse_run_time(run_dir)
            time_str = run_time.strftime("%Y-%m-%d %H:%M:%S") if run_time else "Unknown"
            print(f"  {run_id:<12} {time_str:<20} {run_dir.name}")
        print()
    
    if json_logs:
        print(f"JSON logs ({len(json_logs)}):\n")
        for log_file in json_logs:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"  {log_file.name:<30} {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


@app.command("plot")
def plot(
    run_id: Annotated[Optional[str], typer.Option("--run", "-r", help="Run ID or log file to plot")] = None,
    wandb_dir: Annotated[str, typer.Option(help="Wandb directory")] = "wandb",
    logs_dir: Annotated[str, typer.Option(help="Logs directory")] = "logs",
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output file")] = None,
    title: Annotated[Optional[str], typer.Option(help="Plot title")] = None,
):
    """Plot training metrics."""
    history = []
    run_name = "unknown"
    
    # Try JSON logs first (easier to parse)
    json_logs = find_json_logs(Path(logs_dir))
    if json_logs:
        if run_id:
            # Find matching log
            for log in json_logs:
                if run_id in log.name:
                    history = read_json_metrics(log)
                    run_name = log.stem
                    break
        else:
            # Use latest
            history = read_json_metrics(json_logs[0])
            run_name = json_logs[0].stem
            print(f"Using latest JSON log: {json_logs[0].name}")
    
    # Fall back to wandb if no JSON logs
    if not history:
        wandb_runs = find_wandb_runs(Path(wandb_dir))
        if wandb_runs:
            if run_id:
                run_dir = None
                for r in wandb_runs:
                    if parse_run_id(r) == run_id or r.name == run_id:
                        run_dir = r
                        break
                if run_dir is None:
                    print(f"Run '{run_id}' not found.")
                    raise typer.Exit(1)
            else:
                run_dir = wandb_runs[0]
                print(f"Using latest wandb run: {run_dir.name}")
            
            run_name = parse_run_id(run_dir)
            history = read_wandb_history_via_sync(run_dir)
            
            if not history:
                print(f"\nNo history data found in {run_dir}")
                print("Wandb offline runs need to be synced first to read history.")
                print(f"Run: wandb sync {run_dir}")
                print("\nAlternatively, add JSON logging to your training script.")
                raise typer.Exit(1)
    
    if not history:
        print("No metrics data found.")
        print("Use --list to see available runs.")
        raise typer.Exit(1)
    
    # Extract metrics
    steps = []
    train_losses = []
    val_losses = []
    val_steps = []
    
    for record in history:
        step = record.get("step") or record.get("_step", len(steps))
        if "train_loss" in record:
            steps.append(step)
            train_losses.append(record["train_loss"])
        if "val_loss" in record:
            val_steps.append(step)
            val_losses.append(record["val_loss"])
    
    if not steps and not val_steps:
        print("No loss data found in history")
        raise typer.Exit(1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if steps and train_losses:
        ax.plot(steps, train_losses, label="Train Loss", alpha=0.8, linewidth=1.5)
    
    if val_steps and val_losses:
        ax.plot(val_steps, val_losses, label="Val Loss", marker="o", markersize=4, 
                alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title or f"Training Progress - {run_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add stats annotation
    stats_text = []
    if train_losses:
        stats_text.append(f"Train: {train_losses[0]:.3f} → {train_losses[-1]:.3f}")
    if val_losses:
        stats_text.append(f"Val: {val_losses[0]:.3f} → {val_losses[-1]:.3f}")
    
    if stats_text:
        ax.text(
            0.02, 0.98,
            "\n".join(stats_text),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            alpha=0.8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    
    plt.tight_layout()
    
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output}")
    else:
        plt.show()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="List available runs")] = False,
):
    """Plot training metrics from local logs."""
    if ctx.invoked_subcommand is None:
        if list_all:
            list_runs()
        else:
            plot()


if __name__ == "__main__":
    app()
