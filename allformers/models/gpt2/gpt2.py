"""
GPT-2 Model Implementation

A readable implementation of OpenAI's GPT-2, built with PyTorch and einops.
Following Hugging Face's "one model, one file" philosophy, this single file
contains all components: config, attention, MLP, transformer block, and model.

Architecture:
    token_ids (batch, seq)
    -> Token Embedding + Position Embedding (batch, seq, embedding_dim)
    -> Dropout
    -> N x TransformerBlock (LayerNorm -> Attention -> + -> LayerNorm -> MLP -> +)
    -> LayerNorm
    -> Linear projection to vocab (batch, seq, vocab_size)
    -> logits

GPT-2 uses:
- Pre-norm architecture (LayerNorm before attention/MLP, not after)
- Weight tying between token embedding and output projection
- GELU activation with tanh approximation
- Learned absolute positional embeddings

References:
1) OpenAI GPT-2 TensorFlow implementation:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) Karpathy's nanoGPT:
   https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math

from typing import Self, cast

from pydantic import BaseModel, ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from transformers import GPT2LMHeadModel


# =============================================================================
# Configuration
# =============================================================================


class GPT2Config(BaseModel):
    """Configuration for GPT-2 model.

    Attributes:
        vocab_size: Size of the vocabulary (default: 50257 for GPT-2 BPE tokenizer)
        context_length: Maximum sequence length the model can process
        embedding_dim: Dimension of token and position embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension (default: 4)
        dropout: Dropout probability for regularization
        bias: Whether to use bias in linear layers and layer norms
        scale_residual_init: If True, scale residual projection weights by 1/sqrt(2*num_layers)
            to prevent variance growth through the residual stream. Default False to match
            HuggingFace's initialization.
    """

    vocab_size: int = 50257
    context_length: int = 1024
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = True
    scale_residual_init: bool = False

    model_config = ConfigDict(frozen=True)  # Make immutable like a frozen dataclass

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
        return self.embedding_dim * self.mlp_ratio

    @classmethod
    def gpt2(cls) -> Self:
        """GPT-2 (124M parameters). Matches HuggingFace 'gpt2'."""
        return cls(
            embedding_dim=768,
            num_heads=12,
            num_layers=12,
        )

    @classmethod
    def gpt2_medium(cls) -> Self:
        """GPT-2 Medium (355M parameters). Matches HuggingFace 'gpt2-medium'."""
        return cls(
            embedding_dim=1024,
            num_heads=16,
            num_layers=24,
        )

    @classmethod
    def gpt2_large(cls) -> Self:
        """GPT-2 Large (774M parameters). Matches HuggingFace 'gpt2-large'."""
        return cls(
            embedding_dim=1280,
            num_heads=20,
            num_layers=36,
        )

    @classmethod
    def gpt2_xl(cls) -> Self:
        """GPT-2 XL (1.5B parameters). Matches HuggingFace 'gpt2-xl'."""
        return cls(
            embedding_dim=1600,
            num_heads=25,
            num_layers=48,
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
            model_name: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        Returns:
            GPT2Config with settings matching the pretrained checkpoint
        """
        # Map HuggingFace model names (with hyphens) to class methods (with underscores)
        config_methods = {
            "gpt2": cls.gpt2,
            "gpt2-medium": cls.gpt2_medium,
            "gpt2-large": cls.gpt2_large,
            "gpt2-xl": cls.gpt2_xl,
        }

        if model_name not in config_methods:
            valid = ", ".join(config_methods.keys())
            raise ValueError(f"Unknown model: {model_name}. Choose from: {valid}")

        # Get base config and override with pretrained-specific settings
        base_config = config_methods[model_name]()
        return cls(
            vocab_size=50257,  # GPT-2 BPE tokenizer
            context_length=1024,
            embedding_dim=base_config.embedding_dim,
            num_heads=base_config.num_heads,
            num_layers=base_config.num_layers,
            mlp_ratio=base_config.mlp_ratio,
            dropout=0.0,  # No dropout for inference
            bias=True,
            scale_residual_init=False,  # Loading pretrained weights
        )


# =============================================================================
# Causal Self-Attention
# =============================================================================


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    The attention mechanism is the heart of the transformer. Each token attends
    to all previous tokens (and itself) to gather context.

    Key concepts:
    - Self-attention: Each token attends to all tokens (including itself)
    - Causal/Masked: Each token can only attend to previous tokens (and itself)
    - Multi-head: We run multiple attention operations in parallel, each learning
      different aspects of the relationships between tokens

    The attention computation follows these steps:
    1. Project input to queries (Q), keys (K), and values (V)
    2. Split into multiple heads
    3. Compute attention scores: softmax(QK^T / sqrt(d_k))
    4. Apply causal mask (prevent attending to future tokens)
    5. Multiply scores by values
    6. Concatenate heads and project output

    Flash Attention:
    - When available (PyTorch >= 2.0), uses F.scaled_dot_product_attention
    - This is much faster on GPU due to memory-efficient implementation
    - Falls back to manual attention when not available

    Args:
        config: GPT2Config with model hyperparameters
    """

    # Type annotation for buffer registered conditionally
    causal_mask: torch.Tensor

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        # Combined projection for Q, K, V (more efficient than separate projections)
        # Input: (batch, seq, embedding_dim)
        # Output: (batch, seq, 3 * embedding_dim)
        self.qkv_projection = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
            bias=config.bias,
        )

        # Output projection after attention
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias,
        )

        # Dropout for regularization
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        # Check if flash attention is available (PyTorch >= 2.0)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Flash Attention not available. Using manual attention.")
            # Causal mask for manual attention: a lower triangular matrix of ones
            # We register it as a buffer so it's part of the module state
            # but not a learnable parameter
            causal_mask = torch.tril(
                torch.ones(config.context_length, config.context_length)
            )
            self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project to Q, K, V
        # Shape: (batch, seq, 3 * embedding_dim)
        qkv = self.qkv_projection(x)

        # Step 2: Split into Q, K, V and reshape for multi-head attention
        # Split along last dimension into 3 tensors
        # Then rearrange each to (batch, num_heads, seq, head_dim)
        q, k, v = rearrange(
            qkv,
            "batch seq (three heads head_dim) -> three batch heads seq head_dim",
            three=3,
            heads=self.num_heads,
        )

        if self.flash:
            # Use flash attention (memory-efficient, faster on GPU)
            # scaled_dot_product_attention handles scaling and causal masking
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention implementation (for older PyTorch or debugging)
            # Step 3: Compute attention scores
            # Q @ K^T -> (batch, heads, seq, seq)
            # Scale by sqrt(head_dim) for stable gradients
            scale = self.head_dim ** 0.5
            attention_scores = einsum(
                q, k,
                "batch heads seq_q head_dim, batch heads seq_k head_dim -> batch heads seq_q seq_k"
            ) / scale

            # Step 4: Apply causal mask
            # Set future positions to -inf so softmax gives them 0 probability
            mask = self.causal_mask[:seq_len, :seq_len]
            attention_scores = attention_scores.masked_fill(
                mask == 0,
                float("-inf")
            )

            # Step 5: Softmax to get attention weights (probabilities)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)

            # Step 6: Weighted sum of values
            # (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
            # -> (batch, heads, seq, head_dim)
            output = einsum(
                attention_weights, v,
                "batch heads seq_q seq_k, batch heads seq_k head_dim -> batch heads seq_q head_dim"
            )

        # Step 7: Concatenate heads
        # (batch, heads, seq, head_dim) -> (batch, seq, embedding_dim)
        output = rearrange(
            output,
            "batch heads seq head_dim -> batch seq (heads head_dim)"
        )

        # Step 8: Final projection and dropout
        output = self.output_projection(output)
        output = self.output_dropout(output)

        return output


# =============================================================================
# MLP (Feed-Forward Network)
# =============================================================================


class MLP(nn.Module):
    """Feed-forward network applied after attention.

    The MLP is applied independently to each position after attention.
    It consists of two linear transformations with a GELU activation in between.

    This is where the model does most of its "thinking" - the attention layer
    figures out what information to gather, and the MLP processes that information.

    Architecture:
        input (embedding_dim)
        -> Linear (embedding_dim -> 4 * embedding_dim)
        -> GELU activation
        -> Linear (4 * embedding_dim -> embedding_dim)
        -> Dropout
        -> output (embedding_dim)

    GELU Activation:
        GELU(x) = x * Φ(x), where Φ is the cumulative distribution function
        of the standard normal distribution.

        GPT-2 uses the tanh approximation:
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        We use PyTorch's built-in nn.GELU with approximate='tanh'.

    Args:
        config: GPT2Config with model hyperparameters
    """

    def __init__(self, config: GPT2Config):
        super().__init__()

        # Expand: embedding_dim -> 4 * embedding_dim
        self.fc1 = nn.Linear(
            config.embedding_dim,
            config.mlp_hidden_dim,
            bias=config.bias,
        )

        # GELU activation (GPT-2 uses tanh approximation)
        # PyTorch's nn.GELU supports both exact ('none') and approximate ('tanh')
        self.activation = nn.GELU(approximate="tanh")

        # Contract: 4 * embedding_dim -> embedding_dim
        self.fc2 = nn.Linear(
            config.mlp_hidden_dim,
            config.embedding_dim,
            bias=config.bias,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        # Expand dimension
        x = self.fc1(x)

        # Apply activation
        x = self.activation(x)

        # Contract dimension
        x = self.fc2(x)

        # Apply dropout
        x = self.dropout(x)

        return x


# =============================================================================
# Transformer Block
# =============================================================================


class TransformerBlock(nn.Module):
    """A single transformer block with attention and MLP.

    A transformer block combines attention and MLP with residual connections
    and layer normalization. GPT-2 uses "pre-norm" architecture where LayerNorm
    is applied before each sub-layer rather than after.

    Architecture (Pre-Norm, used in GPT-2):
        x
        -> LayerNorm -> Attention -> + (residual connection with x)
        -> LayerNorm -> MLP       -> + (residual connection)
        -> output

    This differs from the original transformer "post-norm":
        x
        -> Attention -> + (residual) -> LayerNorm
        -> MLP       -> + (residual) -> LayerNorm
        -> output

    Pre-norm tends to train more stably, especially for deeper models.

    Args:
        config: GPT2Config with model hyperparameters
    """

    def __init__(self, config: GPT2Config):
        super().__init__()

        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(
            config.embedding_dim,
            eps=1e-5,
            bias=config.bias,
        )

        # Multi-head causal self-attention
        self.attention = CausalSelfAttention(config)

        # Layer normalization before MLP
        self.ln2 = nn.LayerNorm(
            config.embedding_dim,
            eps=1e-5,
            bias=config.bias,
        )

        # Feed-forward network
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        # Attention sub-block with residual connection
        # x = x + Attention(LayerNorm(x))
        x = x + self.attention(self.ln1(x))

        # MLP sub-block with residual connection
        # x = x + MLP(LayerNorm(x))
        x = x + self.mlp(self.ln2(x))

        return x


# =============================================================================
# Weight Loading Utilities
# =============================================================================


def copy_weights_from_hf(model: "GPT2", hf_model: GPT2LMHeadModel) -> None:
    """Copy weights from a HuggingFace GPT-2 model to our implementation.

    This function maps HuggingFace's parameter names to our parameter names
    and copies the weights. This is essential for verifying our implementation
    produces identical outputs.

    HuggingFace GPT-2 structure:
        transformer.wte.weight              -> token_embedding.weight
        transformer.wpe.weight              -> position_embedding.weight
        transformer.h[i].ln_1.weight/bias   -> blocks[i].ln1.weight/bias
        transformer.h[i].attn.c_attn.weight/bias -> blocks[i].attention.qkv_projection.weight/bias
        transformer.h[i].attn.c_proj.weight/bias -> blocks[i].attention.output_projection.weight/bias
        transformer.h[i].ln_2.weight/bias   -> blocks[i].ln2.weight/bias
        transformer.h[i].mlp.c_fc.weight/bias    -> blocks[i].mlp.fc1.weight/bias
        transformer.h[i].mlp.c_proj.weight/bias  -> blocks[i].mlp.fc2.weight/bias
        transformer.ln_f.weight/bias        -> ln_final.weight/bias
        lm_head.weight                      -> lm_head.weight (tied to token_embedding)

    Note: HuggingFace uses Conv1D for linear layers which stores weights transposed!

    Args:
        model: Our GPT2 model to copy weights into
        hf_model: HuggingFace GPT2LMHeadModel to copy weights from
    """
    hf_state = hf_model.state_dict()

    with torch.no_grad():
        # Token and position embeddings
        model.token_embedding.weight.copy_(hf_state["transformer.wte.weight"])
        model.position_embedding.weight.copy_(hf_state["transformer.wpe.weight"])

        # Transformer blocks
        for i, block in enumerate(cast(list[TransformerBlock], model.blocks)):
            # Layer norm 1 (before attention)
            block.ln1.weight.copy_(hf_state[f"transformer.h.{i}.ln_1.weight"])
            block.ln1.bias.copy_(hf_state[f"transformer.h.{i}.ln_1.bias"])

            # Attention QKV projection (transposed due to Conv1D)
            block.attention.qkv_projection.weight.copy_(
                hf_state[f"transformer.h.{i}.attn.c_attn.weight"].T
            )
            block.attention.qkv_projection.bias.copy_(
                hf_state[f"transformer.h.{i}.attn.c_attn.bias"]
            )

            # Attention output projection (transposed)
            block.attention.output_projection.weight.copy_(
                hf_state[f"transformer.h.{i}.attn.c_proj.weight"].T
            )
            block.attention.output_projection.bias.copy_(
                hf_state[f"transformer.h.{i}.attn.c_proj.bias"]
            )

            # Layer norm 2 (before MLP)
            block.ln2.weight.copy_(hf_state[f"transformer.h.{i}.ln_2.weight"])
            block.ln2.bias.copy_(hf_state[f"transformer.h.{i}.ln_2.bias"])

            # MLP fc1 (transposed)
            block.mlp.fc1.weight.copy_(
                hf_state[f"transformer.h.{i}.mlp.c_fc.weight"].T
            )
            block.mlp.fc1.bias.copy_(hf_state[f"transformer.h.{i}.mlp.c_fc.bias"])

            # MLP fc2 (transposed)
            block.mlp.fc2.weight.copy_(
                hf_state[f"transformer.h.{i}.mlp.c_proj.weight"].T
            )
            block.mlp.fc2.bias.copy_(hf_state[f"transformer.h.{i}.mlp.c_proj.bias"])

        # Final layer norm
        model.ln_final.weight.copy_(hf_state["transformer.ln_f.weight"])
        model.ln_final.bias.copy_(hf_state["transformer.ln_f.bias"])

        # lm_head is tied to token_embedding, already set


# =============================================================================
# GPT-2 Model
# =============================================================================


class GPT2(nn.Module):
    """GPT-2 Language Model.

    The complete GPT-2 model that combines all components:
    - Token embeddings: Convert token IDs to vectors
    - Position embeddings: Add positional information
    - Transformer blocks: Stack of attention + MLP layers
    - Output head: Project back to vocabulary logits

    GPT-2 uses weight tying: the token embedding matrix is reused for the
    output projection (lm_head), reducing parameters and often improving performance.

    Args:
        config: GPT2Config with model hyperparameters
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token embeddings: vocab_size -> embedding_dim
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        # Position embeddings: context_length -> embedding_dim
        # Learned absolute positional embeddings (not sinusoidal)
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )

        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer normalization
        self.ln_final = nn.LayerNorm(
            config.embedding_dim,
            eps=1e-5,
            bias=config.bias,
        )

        # Output projection (language model head)
        # Projects from embedding_dim back to vocab_size
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # Weight tying: share weights between token embedding and output projection
        # This is a key technique in GPT-2 that:
        # 1. Reduces parameters
        # 2. Creates symmetry between input and output representations
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections, per GPT-2 paper
        # This uses the nanoGPT approach: re-initialize with scaled std
        if config.scale_residual_init:
            self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 paper.

        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - LayerNorm: bias=0, weight=1
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _init_residual_projections(self) -> None:
        """Re-initialize residual projection weights with scaled std.

        Per GPT-2 paper, residual projections are initialized with
        std = 0.02 / sqrt(2 * num_layers) to prevent variance from
        growing through the residual stream.

        Each transformer block has 2 residual connections (attention + MLP),
        so we scale by 1/sqrt(2*N) where N is the number of layers.

        The residual projections are:
        - attention.output_projection: projects attention output before residual add
        - mlp.fc2: projects MLP output before residual add

        This follows nanoGPT's approach of using named_parameters to find
        the projection layers by name suffix.
        """
        scaled_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, param in self.named_parameters():
            if name.endswith("output_projection.weight") or name.endswith("fc2.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=scaled_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of GPT-2.

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

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device)

        # Get embeddings
        # Token embedding: (batch, seq) -> (batch, seq, embedding_dim)
        token_emb = self.token_embedding(input_ids)

        # Position embedding: (seq,) -> (seq, embedding_dim) -> broadcast to batch
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_final(x)

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
            non_embedding: If True, exclude position embeddings from count
                          (useful when comparing to models without them)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls, model_name: str) -> Self:
        """Load a pretrained GPT-2 model from HuggingFace.

        Args:
            model_name: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        Returns:
            GPT2 model with pretrained weights loaded
        """
        print(f"Loading pretrained weights from HuggingFace: {model_name}")

        # Create our model with the matching config
        config = GPT2Config.for_pretrained(model_name)
        model = cls(config)

        # Load HuggingFace model and copy weights
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        copy_weights_from_hf(model, hf_model)

        print(f"Successfully loaded {model_name} weights")
        return model

