#!/usr/bin/env python3
"""
View training runs using wandb's terminal UI (LEET).

LEET = Lightweight Experiment Exploration Tool

This provides a rich terminal-based interface for viewing wandb runs,
including metrics, system info, and more.

Usage:
    uv run python scripts/view_run.py              # View latest run
    uv run python scripts/view_run.py --list       # List available runs
    uv run python scripts/view_run.py --run abc123 # View specific run

Or use wandb directly:
    uv run wandb beta leet                         # View latest run
    uv run wandb beta leet wandb/offline-run-xxx/  # View specific run

Reference:
    https://wandb.ai/wandb_fc/product-announcements-fc/reports/Weights-Biases-gets-a-new-terminal-UI--VmlldzoxNTAxODU5Nw
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional
from datetime import datetime

import typer

app = typer.Typer(help="View wandb runs in terminal UI", add_completion=False)

WANDB_DIR = Path("wandb")


def find_runs(wandb_dir: Path) -> list[Path]:
    """Find all wandb run directories."""
    if not wandb_dir.exists():
        return []
    runs = []
    for path in wandb_dir.iterdir():
        if path.is_dir() and (path.name.startswith("offline-run-") or path.name.startswith("run-")):
            runs.append(path)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


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


@app.command("list")
def list_runs(
    wandb_dir: Annotated[str, typer.Option(help="Wandb directory")] = "wandb",
):
    """List available wandb runs."""
    runs = find_runs(Path(wandb_dir))
    
    if not runs:
        print(f"No runs found in {wandb_dir}/")
        return
    
    print(f"Available runs ({len(runs)}):\n")
    print(f"  {'Run ID':<12} {'Time':<20} {'Directory'}")
    print("  " + "-" * 65)
    
    for run_dir in runs:
        run_id = parse_run_id(run_dir)
        run_time = parse_run_time(run_dir)
        time_str = run_time.strftime("%Y-%m-%d %H:%M:%S") if run_time else "Unknown"
        print(f"  {run_id:<12} {time_str:<20} {run_dir.name}")
    
    print("\nTo view a run:")
    print("  uv run python scripts/view_run.py view")
    print("  uv run python scripts/view_run.py view --run <run_id>")


@app.command("view")
def view_run(
    run_id: Annotated[Optional[str], typer.Option("--run", "-r", help="Run ID to view")] = None,
    wandb_dir: Annotated[str, typer.Option(help="Wandb directory")] = "wandb",
):
    """View a wandb run using the terminal UI (LEET).
    
    This launches wandb's Lightweight Experiment Exploration Tool,
    which provides a rich terminal interface for exploring run data.
    
    Press 'q' to quit the UI.
    """
    runs = find_runs(Path(wandb_dir))
    
    if not runs:
        print(f"No runs found in {wandb_dir}/")
        raise typer.Exit(1)
    
    # Find the requested run
    if run_id is None:
        run_dir = runs[0]
        print(f"Viewing latest run: {run_dir.name}")
    else:
        run_dir = None
        for r in runs:
            if parse_run_id(r) == run_id or r.name == run_id:
                run_dir = r
                break
        if run_dir is None:
            print(f"Run '{run_id}' not found. Use 'list' command to see available runs.")
            raise typer.Exit(1)
    
    print(f"Launching terminal UI for: {run_dir}")
    print("Press 'q' to quit\n")
    
    # Launch wandb beta leet
    try:
        subprocess.run(
            ["wandb", "beta", "leet", str(run_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error launching UI: {e}")
        print("\nYou can also try running directly:")
        print(f"  wandb beta leet {run_dir}")
        raise typer.Exit(1)
    except FileNotFoundError:
        print("Error: 'wandb' command not found.")
        print("Try: uv run wandb beta leet " + str(run_dir))
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="List available runs")] = False,
):
    """View wandb runs using the terminal UI."""
    if ctx.invoked_subcommand is None:
        if list_all:
            list_runs()
        else:
            # Default: view latest run
            view_run()


if __name__ == "__main__":
    app()

