"""
Llama Model Implementation

A readable implementation of Meta's Llama, built with PyTorch and einops.
Following Hugging Face's "one model, one file" philosophy, this single file
contains all components: config, RMSNorm, RoPE, attention, MLP, decoder layer, and model.

Architecture:
    token_ids (batch, seq)
    -> Token Embedding (batch, seq, embedding_dim)
    -> N x LlamaDecoderLayer (RMSNorm -> Attention (with RoPE) -> + -> RMSNorm -> MLP (SwiGLU) -> +)
    -> RMSNorm
    -> Linear projection to vocab (batch, seq, vocab_size)
    -> logits

Llama uses:
- Pre-norm architecture (RMSNorm before attention/MLP, not after)
- RMSNorm instead of LayerNorm (no bias, simpler normalization)
- RoPE (Rotary Position Embedding) instead of learned positional embeddings
- SwiGLU activation in MLP (instead of GELU)
- No weight tying (unlike GPT-2)

References:
1) Meta Llama paper: https://arxiv.org/abs/2302.13971
2) HuggingFace transformers implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
3) RoPE paper: https://arxiv.org/abs/2104.09864
"""

import math

from typing import Self, cast

from pydantic import BaseModel, ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from transformers import LlamaForCausalLM


# =============================================================================
# Configuration
# =============================================================================


class LlamaConfig(BaseModel):
    """Configuration for Llama model.

    Attributes:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length the model can process
        embedding_dim: Dimension of token embeddings (hidden_size in HF)
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        intermediate_size: Hidden dimension of the MLP (default: 4 * embedding_dim)
        rms_norm_eps: Epsilon for RMSNorm (default: 1e-6)
        dropout: Dropout probability for regularization
        rope_theta: Base frequency for RoPE (default: 10000.0)
    """

    vocab_size: int = 128256  # Llama 3.2 tokenizer vocab size
    context_length: int = 8192
    embedding_dim: int = 2048
    num_heads: int = 16
    num_key_value_heads: int | None = None  # For GQA, defaults to num_heads if None
    num_layers: int = 16
    intermediate_size: int | None = None  # Will default to 4 * embedding_dim
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    rope_theta: float = 10000.0

    model_config = ConfigDict(frozen=True)  # Make immutable like a frozen dataclass

    @property
    def num_kv_heads(self) -> int:
        """Number of key/value heads (for GQA)."""
        return self.num_key_value_heads if self.num_key_value_heads is not None else self.num_heads

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        assert self.embedding_dim % self.num_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        return self.embedding_dim // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        """Hidden dimension of the MLP (feed-forward) layer."""
        if self.intermediate_size is not None:
            return self.intermediate_size
        return self.embedding_dim * 4

    @classmethod
    def llama_3_2_1b(cls) -> Self:
        """Llama 3.2 1B model configuration."""
        return cls(
            vocab_size=128256,
            context_length=8192,
            embedding_dim=2048,
            num_heads=32,  # Actual HF config has 32 attention heads
            num_key_value_heads=8,  # GQA: 32 query heads, 8 KV heads
            num_layers=16,
            intermediate_size=8192,  # 4 * embedding_dim
            rms_norm_eps=1e-6,
            dropout=0.0,
            rope_theta=10000.0,
        )

    @classmethod
    def tiny(cls) -> Self:
        """A tiny model for testing and debugging."""
        return cls(
            vocab_size=1000,
            context_length=128,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )

    @classmethod
    def for_pretrained(cls, model_name: str) -> Self:
        """Get config for a pretrained HuggingFace model.

        Args:
            model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-3.2-1B')

        Returns:
            LlamaConfig with settings matching the pretrained checkpoint
        """
        from transformers import LlamaConfig as HFLlamaConfig
        import os
        
        token = os.environ.get("HF_TOKEN")
        hf_config = HFLlamaConfig.from_pretrained(model_name, token=token)
        
        # Extract rope_theta from rope_parameters
        # rope_parameters can be None, a dict, or a RopeParameters object
        rope_theta: float = 10000.0  # default
        if hf_config.rope_parameters is not None:
            if isinstance(hf_config.rope_parameters, dict):
                rope_theta_val = hf_config.rope_parameters.get("rope_theta", 10000.0)
                if isinstance(rope_theta_val, (int, float)):
                    rope_theta = float(rope_theta_val)
            elif hasattr(hf_config.rope_parameters, "rope_theta"):
                rope_theta_val = getattr(hf_config.rope_parameters, "rope_theta")
                if isinstance(rope_theta_val, (int, float)):
                    rope_theta = float(rope_theta_val)
            elif hasattr(hf_config.rope_parameters, "get"):
                rope_theta_val = hf_config.rope_parameters.get("rope_theta", 10000.0)
                if isinstance(rope_theta_val, (int, float)):
                    rope_theta = float(rope_theta_val)
        
        return cls(
            vocab_size=hf_config.vocab_size or 128256,
            context_length=hf_config.max_position_embeddings or 8192,
            embedding_dim=hf_config.hidden_size or 2048,
            num_heads=hf_config.num_attention_heads or 16,
            num_key_value_heads=hf_config.num_key_value_heads,  # Can be None, which is handled by property
            num_layers=hf_config.num_hidden_layers or 16,
            intermediate_size=hf_config.intermediate_size,  # Can be None, which is handled by property
            rms_norm_eps=hf_config.rms_norm_eps or 1e-6,
            dropout=0.0,  # No dropout in pretrained models
            rope_theta=rope_theta,
        )


# =============================================================================
# RMSNorm
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simpler alternative to LayerNorm that:
    - Doesn't use a bias term
    - Normalizes by RMS instead of mean and variance
    - Computes: output = (input / RMS(input)) * weight
    - Where RMS(input) = sqrt(mean(input^2) + eps)

    This is more efficient than LayerNorm and works well in practice.

    Args:
        dim: Dimension to normalize over
        eps: Small value to prevent division by zero
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RMSNorm.

        This matches HuggingFace's implementation exactly:
        - Compute in float32 for numerical stability
        - Use rsqrt for efficiency
        - Convert back to original dtype

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input.
    
    This is the HuggingFace implementation: split into two halves,
    swap them, and negate the second half.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys.
    
    This matches HuggingFace's implementation exactly:
    - Uses rotate_half for the rotation
    - cos/sin have shape (seq_len, head_dim) - already full head_dim
    - Unsqueeze at dim=1 for broadcasting to (batch, heads, seq_len, head_dim)
    - Formula: rotated = (x * cos) + (rotate_half(x) * sin)

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim) for GQA
        cos: Cosine values for RoPE of shape (seq_len, head_dim)
        sin: Sine values for RoPE of shape (seq_len, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as input
    """
    # Unsqueeze at dim=1 (heads dimension) for broadcasting
    # cos/sin: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    # This broadcasts to q/k shape: (batch, heads, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply RoPE: rotated = (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def precompute_rope_freqs(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequencies for RoPE.
    
    This matches HuggingFace's LlamaRotaryEmbedding implementation:
    - Compute inverse frequencies: 1 / (theta^(2i/dim))
    - Multiply by positions to get angles
    - Concatenate (freqs, freqs) to get full head_dim
    - Return cos and sin of the angles

    Args:
        head_dim: Dimension of each attention head
        seq_len: Maximum sequence length
        theta: Base frequency for RoPE (rope_theta in config)
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors of shape (seq_len, head_dim)
    """
    # Compute inverse frequencies: 1 / (theta^(2i/dim)) for i in [0, 1, 2, ...]
    # This gives head_dim // 2 frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
    # Create position indices: [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device).float()
    
    # Compute angles: outer product of positions and inverse frequencies
    # (seq_len,) @ (head_dim // 2,) -> (seq_len, head_dim // 2)
    freqs = torch.outer(positions, inv_freq)
    
    # Concatenate freqs with itself to get full head_dim (matches HF exactly)
    # emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Compute cos and sin
    cos = emb.cos()
    sin = emb.sin()
    
    return cos, sin


# =============================================================================
# Causal Self-Attention with RoPE
# =============================================================================


class LlamaAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    This is similar to GPT-2's attention but uses RoPE instead of
    learned positional embeddings. RoPE is applied to queries and keys
    before computing attention scores.

    Args:
        config: LlamaConfig with model hyperparameters
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        # Separate projections for Q, K, V to support GQA
        # Q: num_heads * head_dim, K/V: num_kv_heads * head_dim
        self.num_kv_heads = config.num_kv_heads
        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        
        self.q_projection = nn.Linear(
            config.embedding_dim,
            q_dim,
            bias=False,
        )
        self.k_projection = nn.Linear(
            config.embedding_dim,
            kv_dim,
            bias=False,
        )
        self.v_projection = nn.Linear(
            config.embedding_dim,
            kv_dim,
            bias=False,
        )

        # Output projection
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=False,
        )

        # Dropout for regularization
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE frequencies
        # We'll recompute these if sequence length changes
        # Initialize as None, will be set on first forward pass
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

        # Check if flash attention is available
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Flash Attention not available. Using manual attention.")

    def _update_rope_cache(self, seq_len: int, device: torch.device) -> None:
        """Update RoPE cache if sequence length changed."""
        if (
            self._rope_cos is None
            or self._rope_cos.shape[0] < seq_len
        ):
            cos, sin = precompute_rope_freqs(
                self.head_dim,
                max(seq_len, self.config.context_length),
                self.config.rope_theta,
                device,
            )
            # Register as buffers so they're part of module state
            # Only register if they don't exist yet (check _buffers dict)
            if "rope_cos" not in self._buffers:
                self.register_buffer("rope_cos", cos, persistent=False)
                self.register_buffer("rope_sin", sin, persistent=False)
            else:
                # Update existing buffers
                rope_cos_buffer = cast(torch.Tensor, self.rope_cos)
                rope_sin_buffer = cast(torch.Tensor, self.rope_sin)
                rope_cos_buffer.copy_(cos)
                rope_sin_buffer.copy_(sin)
            # Update internal references (buffers are tensors)
            self._rope_cos = cast(torch.Tensor, self.rope_cos)
            self._rope_sin = cast(torch.Tensor, self.rope_sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of causal self-attention with RoPE.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Update RoPE cache if needed
        self._update_rope_cache(seq_len, x.device)

        # Project to Q, K, V separately
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        # Reshape for multi-head attention
        # Q: (batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim)
        # K, V: (batch, seq, num_kv_heads * head_dim) -> (batch, num_kv_heads, seq, head_dim)
        q = rearrange(
            q,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.num_heads,
        )
        k = rearrange(
            k,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.num_kv_heads,
        )
        v = rearrange(
            v,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.num_kv_heads,
        )

        # Apply RoPE to queries and keys BEFORE repeating (important for GQA!)
        # After _update_rope_cache, these are guaranteed to be tensors
        assert self._rope_cos is not None and self._rope_sin is not None
        cos = self._rope_cos[:seq_len]
        sin = self._rope_sin[:seq_len]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # For GQA, we need to repeat K and V to match Q's number of heads
        # Each KV head is shared by num_heads // num_kv_heads query heads
        # IMPORTANT: Repeat AFTER applying RoPE (this matches HuggingFace's implementation)
        # Use the same method as HF: expand + reshape (not repeat_interleave)
        if self.num_heads != self.num_kv_heads:
            # Repeat K and V heads to match Q
            # k: (batch, num_kv_heads, seq, head_dim) -> (batch, num_heads, seq, head_dim)
            repeat_factor = self.num_heads // self.num_kv_heads
            # Use expand + reshape like HuggingFace (equivalent to repeat_interleave but matches HF exactly)
            k = k[:, :, None, :, :].expand(-1, -1, repeat_factor, -1, -1).reshape(
                k.shape[0], self.num_heads, k.shape[2], k.shape[3]
            )
            v = v[:, :, None, :, :].expand(-1, -1, repeat_factor, -1, -1).reshape(
                v.shape[0], self.num_heads, v.shape[2], v.shape[3]
            )

        if self.flash:
            # Use flash attention (memory-efficient, faster on GPU)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention implementation
            scale = self.head_dim ** 0.5
            attention_scores = einsum(
                q, k,
                "batch heads seq_q head_dim, batch heads seq_k head_dim -> batch heads seq_q seq_k"
            ) / scale

            # Apply causal mask
            # Create lower triangular mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            attention_scores = attention_scores.masked_fill(
                mask == 0,
                float("-inf")
            )

            # Softmax to get attention weights
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)

            # Weighted sum of values
            output = einsum(
                attention_weights, v,
                "batch heads seq_q seq_k, batch heads seq_k head_dim -> batch heads seq_q head_dim"
            )

        # Concatenate heads
        output = rearrange(
            output,
            "batch heads seq head_dim -> batch seq (heads head_dim)"
        )

        # Final projection and dropout
        output = self.output_projection(output)
        output = self.output_dropout(output)

        return output


# =============================================================================
# MLP with SwiGLU
# =============================================================================


class LlamaMLP(nn.Module):
    """Feed-forward network with SwiGLU activation.

    SwiGLU is a variant of GLU (Gated Linear Unit) that uses Swish activation:
    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    where ⊙ is element-wise multiplication.

    Swish is: Swish(x) = x * sigmoid(x)

    The MLP structure:
        input (embedding_dim)
        -> Gate projection (embedding_dim -> intermediate_size)
        -> Up projection (embedding_dim -> intermediate_size)
        -> SwiGLU activation
        -> Down projection (intermediate_size -> embedding_dim)
        -> Dropout
        -> output (embedding_dim)

    Args:
        config: LlamaConfig with model hyperparameters
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # Gate projection (for gating in SwiGLU)
        self.gate_projection = nn.Linear(
            config.embedding_dim,
            config.mlp_hidden_dim,
            bias=False,
        )

        # Up projection (main projection)
        self.up_projection = nn.Linear(
            config.embedding_dim,
            config.mlp_hidden_dim,
            bias=False,
        )

        # Down projection (output projection)
        self.down_projection = nn.Linear(
            config.mlp_hidden_dim,
            config.embedding_dim,
            bias=False,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP with SwiGLU.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        # Gate and up projections
        gate = self.gate_projection(x)
        up = self.up_projection(x)

        # SwiGLU: Swish(gate) * up
        # Swish(x) = x * sigmoid(x)
        swish_gate = gate * torch.sigmoid(gate)
        activated = swish_gate * up

        # Down projection
        output = self.down_projection(activated)

        # Dropout
        output = self.dropout(output)

        return output


# =============================================================================
# Decoder Layer
# =============================================================================


class LlamaDecoderLayer(nn.Module):
    """A single Llama decoder layer with attention and MLP.

    Architecture (Pre-Norm, same as GPT-2):
        x
        -> RMSNorm -> Attention -> + (residual connection with x)
        -> RMSNorm -> MLP       -> + (residual connection)
        -> output

    Args:
        config: LlamaConfig with model hyperparameters
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # RMSNorm before attention
        self.input_layernorm = RMSNorm(
            config.embedding_dim,
            eps=config.rms_norm_eps,
        )

        # Multi-head causal self-attention with RoPE
        self.attention = LlamaAttention(config)

        # RMSNorm before MLP
        self.post_attention_layernorm = RMSNorm(
            config.embedding_dim,
            eps=config.rms_norm_eps,
        )

        # Feed-forward network with SwiGLU
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        # Attention sub-block with residual connection
        # x = x + Attention(RMSNorm(x))
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x)
        x = residual + x

        # MLP sub-block with residual connection
        # x = x + MLP(RMSNorm(x))
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# =============================================================================
# Weight Loading Utilities
# =============================================================================


def copy_weights_from_hf(model: "Llama", hf_model: LlamaForCausalLM) -> None:
    """Copy weights from a HuggingFace Llama model to our implementation.

    This function maps HuggingFace's parameter names to our parameter names
    and copies the weights.

    Args:
        model: Our Llama model to copy weights into
        hf_model: HuggingFace LlamaForCausalLM to copy weights from
    """
    hf_state = hf_model.state_dict()

    with torch.no_grad():
        # Token embeddings
        model.embed_tokens.weight.copy_(hf_state["model.embed_tokens.weight"])

        # Decoder layers
        for i, layer in enumerate(cast(list[LlamaDecoderLayer], model.layers)):
            # Input layer norm (before attention)
            layer.input_layernorm.weight.copy_(
                hf_state[f"model.layers.{i}.input_layernorm.weight"]
            )

            # Attention Q, K, V projections (separate for GQA support)
            layer.attention.q_projection.weight.copy_(
                hf_state[f"model.layers.{i}.self_attn.q_proj.weight"]
            )
            layer.attention.k_projection.weight.copy_(
                hf_state[f"model.layers.{i}.self_attn.k_proj.weight"]
            )
            layer.attention.v_projection.weight.copy_(
                hf_state[f"model.layers.{i}.self_attn.v_proj.weight"]
            )

            # Attention output projection
            layer.attention.output_projection.weight.copy_(
                hf_state[f"model.layers.{i}.self_attn.o_proj.weight"]
            )

            # Post-attention layer norm (before MLP)
            layer.post_attention_layernorm.weight.copy_(
                hf_state[f"model.layers.{i}.post_attention_layernorm.weight"]
            )

            # MLP projections
            layer.mlp.gate_projection.weight.copy_(
                hf_state[f"model.layers.{i}.mlp.gate_proj.weight"]
            )
            layer.mlp.up_projection.weight.copy_(
                hf_state[f"model.layers.{i}.mlp.up_proj.weight"]
            )
            layer.mlp.down_projection.weight.copy_(
                hf_state[f"model.layers.{i}.mlp.down_proj.weight"]
            )

        # Final layer norm
        model.norm.weight.copy_(hf_state["model.norm.weight"])

        # Output projection (lm_head)
        model.lm_head.weight.copy_(hf_state["lm_head.weight"])


# =============================================================================
# Llama Model
# =============================================================================


class Llama(nn.Module):
    """Llama Language Model.

    The complete Llama model that combines all components:
    - Token embeddings: Convert token IDs to vectors
    - Decoder layers: Stack of attention + MLP layers with RoPE
    - Output head: Project back to vocabulary logits

    Unlike GPT-2, Llama does NOT use weight tying.

    Args:
        config: LlamaConfig with model hyperparameters
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # Token embeddings: vocab_size -> embedding_dim
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(
            config.embedding_dim,
            eps=config.rms_norm_eps,
        )

        # Output projection (language model head)
        # Projects from embedding_dim back to vocab_size
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Llama model with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following Llama paper.

        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - RMSNorm: weight=1
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of Llama.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            targets: Optional target token IDs for computing loss (batch, seq_len)

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape

        # Ensure sequence length doesn't exceed context length
        assert seq_len <= self.config.context_length, (
            f"Sequence length {seq_len} exceeds context length "
            f"{self.config.context_length}"
        )

        # Get token embeddings
        # Token embedding: (batch, seq) -> (batch, seq, embedding_dim)
        x = self.embed_tokens(input_ids)
        x = self.embedding_dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy:
            # logits: (batch * seq, vocab_size)
            # targets: (batch * seq,)
            loss = nn.functional.cross_entropy(
                rearrange(logits, "batch seq vocab -> (batch seq) vocab"),
                rearrange(targets, "batch seq -> (batch seq)"),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1 = sharper, >1 = flatter)
            top_k: If set, only sample from top k most likely tokens
            eos_token_id: If set, stop generation when all sequences have generated this token

        Returns:
            Generated token IDs of shape (batch, seq_len + num_generated)
            where num_generated <= max_new_tokens
        """
        batch_size = input_ids.shape[0]
        
        # Track which sequences have finished (generated EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for _ in range(max_new_tokens):
            # Crop to context length if necessary
            input_ids_cond = input_ids
            if input_ids.shape[1] > self.config.context_length:
                input_ids_cond = input_ids[:, -self.config.context_length:]

            # Get predictions
            logits, _ = self.forward(input_ids_cond)

            # Focus on last token's predictions
            # (batch, seq, vocab) -> (batch, vocab)
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                # Get the top k values and indices
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all values below the k-th largest to -inf
                logits[logits < top_k_values[:, [-1]]] = float("-inf")

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude token embeddings from count
                          (useful when comparing to models without them)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls, model_name: str) -> Self:
        """Load a pretrained Llama model from HuggingFace.

        Args:
            model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-3.2-1B')

        Returns:
            Llama model with pretrained weights loaded
        """
        print(f"Loading pretrained weights from HuggingFace: {model_name}")

        # Create our model with the matching config
        config = LlamaConfig.for_pretrained(model_name)
        model = cls(config)

        # Load HuggingFace model and copy weights
        # Use token from environment if available
        import os
        token = os.environ.get("HF_TOKEN")
        hf_model = LlamaForCausalLM.from_pretrained(model_name, token=token)
        copy_weights_from_hf(model, hf_model)

        print(f"Successfully loaded {model_name} weights")
        return model

