# Agent Guidelines

This document describes conventions and preferences for AI agents working on this codebase.

## Project Management

This project uses **uv** for Python project management.

### Installation

```bash
# Install core dependencies
uv sync

# Install with dev dependencies (pytest, ipython)
uv sync --extra dev
```

### Running Commands

Always use `uv run` to execute Python commands:

```bash
uv run python script.py
uv run pytest
```

## Testing

Tests are managed with **pytest**. Some tests are marked as `slow` because they download data from HuggingFace.

### Running Tests

```bash
# Run all tests (includes slow tests)
uv run pytest

# Run only fast tests (skip slow tests) 
uv run pytest -m "not slow"

# Run only slow tests
uv run pytest -m slow

# Run with verbose output
uv run pytest -v
```

## Code Organization

### One Model, One File

Each model implementation should be contained in a single file for easy reading. For example, the entire GPT-2 implementation lives in `allformers/models/gpt2/gpt2.py`.

### One Dataset, One File

Similarly, each dataset module should be self-contained. Wikipedia utilities are in `allformers/data/wikipedia.py`, FinePDFs-Edu in `allformers/data/finepdfs_edu.py`.

## Import Conventions

### No Import Aliases in `__init__.py`

Do **not** use `__init__.py` files to create import aliases. Keep `__init__.py` files minimal (empty or just containing version info for the root package).

**Bad:**
```python
# allformers/data/__init__.py
from allformers.data.wikipedia import load_wikipedia  # Don't do this
```

**Good:**
```python
# allformers/data/__init__.py
# (empty or minimal)
```

### Use Full Import Paths

Always import from the full module path:

**Bad:**
```python
from allformers.data import load_wikipedia  # Relies on __init__.py alias
```

**Good:**
```python
from allformers.data.wikipedia import load_wikipedia  # Full path
```

### No Docstrings in `__init__.py`

Do not put module docstrings in `__init__.py` files. Instead, create a `README.md` in the directory if documentation is needed.

## HuggingFace Datasets

When working with HuggingFace streaming datasets, be aware of resource cleanup issues that can cause tests to hang.

### Avoiding Hangs in Tests

1. **Use small shuffle buffers in tests**: Large buffers (e.g., 1000) take time to fill. Use small buffers (e.g., 10) for faster tests.

2. **Explicitly delete dataset iterators**: Streaming datasets hold network connections. Delete them when done:
   ```python
   train_data, val_data = load_dataset_streaming(...)
   # ... use the data ...
   del train_data, val_data
   gc.collect()
   ```

3. **Disable filters when not testing them**: Language/score filters slow down iteration. Pass `filter_language=None` when the test doesn't need filtering.

4. **Iterate only what you need**: Don't iterate 1000 items when 10 suffice. Streaming datasets are slow to start.

### Example Test Pattern

```python
def test_streaming_dataset(self):
    import gc
    
    train_data, val_data = load_fineweb_edu_streaming(
        shuffle_buffer_size=10,      # Small buffer for speed
        filter_language=None,         # Disable unless testing filtering
    )
    
    # Get just what you need
    for i, doc in enumerate(train_data):
        # ... test logic ...
        if i >= 4:
            break
    
    # Explicit cleanup to avoid hanging
    del train_data, val_data
    gc.collect()
```

### Cleanup Errors

You may see errors like `'[Errno 9] Bad file descriptor'` during cleanup. These are harmless - they occur when the script exits while background download threads are still active.

## Documentation

- Use docstrings in the actual module files (`.py` files with implementations)
- Use `README.md` files in directories for package-level documentation
- The project README is at the root level

