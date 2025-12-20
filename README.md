# allformers

Readable transformer implementations for learning, built with PyTorch and einops.

## Goals

- **Readability over performance**: Code prioritizes clarity and understanding
- **Educational**: Extensive comments explaining each component
- **Functional**: Can train on small datasets and load HuggingFace weights
- **Modern tooling**: Uses einops for clear tensor operations
- **Modular**: Each model in its own directory for easy comparison

## Installation

```bash
uv sync
```

## Quick Start

```python
from allformers.models.gpt2.gpt2 import GPT2, GPT2Config

# Create a tiny model for testing
config = GPT2Config.tiny()
model = GPT2(config)

# Or use standard GPT-2 sizes
config = GPT2Config.gpt2_small()   # 124M params
config = GPT2Config.gpt2_medium()  # 355M params
config = GPT2Config.gpt2_large()   # 774M params
config = GPT2Config.gpt2_xl()      # 1.5B params
```

## Project Structure

Following HuggingFace's "one model, one file" philosophy:

```
allformers/
├── __init__.py              # Package version
├── utils.py                 # Device utilities
└── models/
    └── gpt2/
        └── gpt2.py          # Complete GPT-2 implementation
```

## Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| GPT-2 | ✅ | OpenAI's GPT-2 language model |

## Architecture Overview (GPT-2)

```
Token IDs (batch, seq)
    │
    ▼
┌─────────────────────────────────┐
│  Token Embedding + Position Emb │
│  + Dropout                      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Block (×N)         │
│  ┌─────────────────────────────┐│
│  │ LayerNorm → Attention → +  ││
│  │ LayerNorm → MLP → +        ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Final LayerNorm                │
│  Linear → Logits                │
└─────────────────────────────────┘
    │
    ▼
Logits (batch, seq, vocab_size)
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_hf_comparison.py -v
```

## Key Concepts

### Causal (Masked) Attention
Each token can only attend to previous tokens and itself. This is achieved by masking future positions with -inf before softmax.

### Pre-Norm Architecture
GPT-2 applies LayerNorm *before* each sub-layer (attention/MLP), not after. This improves training stability.

### Weight Tying
The token embedding matrix is reused for the output projection (lm_head), reducing parameters.

### einops Usage
We use einops throughout for clear tensor operations:
```python
# Split into heads
q, k, v = rearrange(qkv, "b s (three h d) -> three b h s d", three=3, h=num_heads)

# Compute attention
scores = einsum(q, k, "b h sq d, b h sk d -> b h sq sk")
```

## License

MIT
