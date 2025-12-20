# GPT-2

A readable implementation of OpenAI's GPT-2 language model.

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

# Load pretrained weights from HuggingFace
model = GPT2.from_pretrained("gpt2")
```

## Key Concepts

### Causal (Masked) Attention
Each token can only attend to previous tokens and itself. This is achieved by masking future positions with -inf before softmax.

### Pre-Norm Architecture
GPT-2 applies LayerNorm *before* each sub-layer (attention/MLP), not after. This improves training stability for deeper models.

### Weight Tying
The token embedding matrix is reused for the output projection (lm_head), reducing parameters.

### GELU Activation
GPT-2 uses the GELU activation function with tanh approximation in the MLP layers.

## einops Usage

We use einops throughout for clear tensor operations:

```python
# Split QKV into heads
q, k, v = rearrange(
    qkv,
    "batch seq (three heads head_dim) -> three batch heads seq head_dim",
    three=3,
    heads=num_heads,
)

# Compute attention scores
scores = einsum(
    q, k,
    "batch heads seq_q head_dim, batch heads seq_k head_dim -> batch heads seq_q seq_k"
)
```

## Model Configurations

| Config | Layers | Heads | Embedding Dim | Parameters |
|--------|--------|-------|---------------|------------|
| `gpt2_small()` | 12 | 12 | 768 | 124M |
| `gpt2_medium()` | 24 | 16 | 1024 | 355M |
| `gpt2_large()` | 36 | 20 | 1280 | 774M |
| `gpt2_xl()` | 48 | 25 | 1600 | 1.5B |
| `tiny()` | 2 | 4 | 64 | ~100K |

## References

1. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Original GPT-2 paper
2. [OpenAI GPT-2 TensorFlow implementation](https://github.com/openai/gpt-2/blob/master/src/model.py)
3. [HuggingFace transformers GPT-2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
4. [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)

