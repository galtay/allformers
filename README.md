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
# Install core dependencies
uv sync

# Install with dev dependencies (pytest, ipython) - required for running tests
uv sync --extra dev
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
| [Llama](allformers/models/llama/) | ✅ | Meta's Llama 3.2 with RoPE, GQA, and SwiGLU |

## HuggingFace Authentication

A HuggingFace token is **required** for accessing gated models like Llama, and recommended for faster downloads:

```bash
# Set for current session
export HF_TOKEN=your_token_here

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
```

You can get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Note**: For Llama models, you must also accept the license at [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).

## Running Tests

First, ensure dev dependencies are installed:

```bash
uv sync --extra dev
```

Then run the tests:

```bash
# Run all tests (includes slow tests that download from HuggingFace)
uv run pytest

# Run only fast tests (skip slow tests)
uv run pytest -m "not slow"

# Run only slow tests
uv run pytest -m slow

# Run with verbose output
uv run pytest -v
```

## License

MIT
