# Llama

A readable implementation of Meta's Llama language model.

## Quick Start

```python
from allformers.models.llama.llama import Llama, LlamaConfig

# Create a tiny model for testing
config = LlamaConfig.tiny()
model = Llama(config)

# Or use Llama 3.2 1B configuration
config = LlamaConfig.llama_3_2_1b()
model = Llama(config)

# Load pretrained weights from HuggingFace
model = Llama.from_pretrained("meta-llama/Llama-3.2-1B")
```

## Key Concepts

### RMSNorm (Root Mean Square Layer Normalization)
Llama uses RMSNorm instead of LayerNorm:
- Simpler: no bias term, normalizes by RMS instead of mean and variance
- More efficient: fewer operations
- Formula: `output = (input / RMS(input)) * weight`
- Where `RMS(input) = sqrt(mean(input^2) + eps)`

### RoPE (Rotary Position Embedding)
Llama uses RoPE instead of learned positional embeddings:
- Encodes position by rotating query and key vectors in the complex plane
- Allows the model to understand relative positions naturally
- More efficient than learned positional embeddings
- Applied directly to Q and K before computing attention scores

### SwiGLU Activation
Llama uses SwiGLU in the MLP instead of GELU:
- SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
- Where Swish(x) = x * sigmoid(x)
- More expressive than standard GELU activation

### Pre-Norm Architecture
Like GPT-2, Llama uses pre-norm:
- RMSNorm is applied *before* each sub-layer (attention/MLP)
- Improves training stability for deeper models

### No Weight Tying
Unlike GPT-2, Llama does NOT tie weights between token embeddings and output projection.

## Architecture Differences from GPT-2

| Feature | GPT-2 | Llama |
|---------|-------|-------|
| Normalization | LayerNorm | RMSNorm |
| Position Encoding | Learned embeddings | RoPE |
| MLP Activation | GELU | SwiGLU |
| Weight Tying | Yes | No |
| Attention Bias | Yes | No |

## Model Configurations

| Config | Layers | Heads | Embedding Dim | Parameters |
|--------|--------|-------|---------------|------------|
| `llama_3_2_1b()` | 16 | 16 | 2048 | ~1B |
| `tiny()` | 2 | 4 | 64 | ~100K |

## References

1. [Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Original Llama paper
2. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE paper
3. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU paper

