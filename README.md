# allformers

Readable transformer implementations for learning, built with PyTorch and einops.

## Goals

- **Readability over performance**: Code prioritizes clarity and understanding
- **Educational**: Extensive comments explaining each component
- **Functional**: Can train on small datasets and load HuggingFace weights
- **Modern tooling**: Uses einops for clear tensor operations
- **One model, one file**: Following HuggingFace's philosophy for easy reading

## Installation

```bash
uv sync
```

## Project Structure

```
allformers/
├── __init__.py              # Package version
├── utils.py                 # Device utilities
└── models/
    └── <model_name>/
        └── <model_name>.py  # Complete implementation in one file
```

## Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| [GPT-2](allformers/models/gpt2/) | ✅ | OpenAI's GPT-2 language model |

## Running Tests

```bash
# Run fast tests
uv run pytest

# Run all tests including slow ones (downloads from HuggingFace)
uv run pytest -m slow

# Run with verbose output
uv run pytest -v
```

## License

MIT
