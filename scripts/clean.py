#!/usr/bin/env python3
"""
Clean up local logs and cache files from previous runs.

Usage:
    uv run python scripts/clean.py          # Dry run (show what would be deleted)
    uv run python scripts/clean.py --force  # Actually delete files
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Clean up local logs and cache files", add_completion=False)

# Directories to clean (log directories only - not Python artifacts)
CLEAN_DIRS = [
    "wandb",           # Wandb run logs
]


def get_dirs_to_clean(root: Path) -> list[Path]:
    """Find all directories that should be cleaned."""
    dirs_to_clean = []
    
    for dir_name in CLEAN_DIRS:
        dir_path = root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            dirs_to_clean.append(dir_path)
    
    return sorted(dirs_to_clean)


def get_size_str(path: Path) -> str:
    """Get human-readable size of a directory."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        return "?"
    
    if total < 1024:
        return f"{total} B"
    elif total < 1024 * 1024:
        return f"{total / 1024:.1f} KB"
    elif total < 1024 * 1024 * 1024:
        return f"{total / (1024 * 1024):.1f} MB"
    else:
        return f"{total / (1024 * 1024 * 1024):.1f} GB"


@app.command()
def clean(
    force: Annotated[bool, typer.Option("--force", "-f", help="Actually delete files (default is dry run)")] = False,
    root: Annotated[str, typer.Option(help="Root directory to clean")] = ".",
):
    """Clean up local logs and cache files."""
    root_path = Path(root).resolve()
    
    print(f"Scanning {root_path}...")
    dirs_to_clean = get_dirs_to_clean(root_path)
    
    if not dirs_to_clean:
        print("Nothing to clean!")
        return
    
    print()
    if force:
        print("Deleting:")
    else:
        print("Would delete (use --force to actually delete):")
    
    total_size = 0
    for dir_path in dirs_to_clean:
        rel_path = dir_path.relative_to(root_path)
        size_str = get_size_str(dir_path)
        print(f"  {rel_path}/ ({size_str})")
        
        if force:
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"    Error: {e}")
    
    print()
    if force:
        print(f"✓ Cleaned {len(dirs_to_clean)} directories")
    else:
        print(f"Run with --force to delete {len(dirs_to_clean)} directories")


if __name__ == "__main__":
    app()

