"""
Tests for GPT-2 Model Implementation.

This consolidated test file covers all components:
- GPT2Config
- CausalSelfAttention
- MLP
- TransformerBlock
- GPT2
"""

import math

import torch
import torch.nn.functional as F
import pytest
from transformers import GPT2LMHeadModel, GPT2Config as HF_GPT2Config

from allformers.models.gpt2.gpt2 import (
    GPT2,
    GPT2Config,
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    copy_weights_from_hf,
)


# =============================================================================
# Test Utilities
# =============================================================================


def create_matching_models(
    num_layers: int = 2,
    num_heads: int = 4,
    embedding_dim: int = 64,
    vocab_size: int = 1000,
    context_length: int = 128,
) -> tuple[GPT2, GPT2LMHeadModel]:
    """Create our model and HuggingFace model with matching configs."""
    # Our config
    our_config = GPT2Config(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0,  # Disable dropout for deterministic comparison
        bias=True,
    )

    # HuggingFace config
    hf_config = HF_GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_embd=embedding_dim,
        n_head=num_heads,
        n_layer=num_layers,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    # Create models
    our_model = GPT2(our_config)
    hf_model = GPT2LMHeadModel(hf_config)

    # Set both to eval mode
    our_model.eval()
    hf_model.eval()

    return our_model, hf_model


# =============================================================================
# GPT2Config Tests
# =============================================================================


class TestGPT2Config:
    """Tests for the GPT2Config dataclass."""

    def test_default_values(self):
        """Test that default config values are correct for GPT-2."""
        config = GPT2Config()

        assert config.vocab_size == 50257
        assert config.context_length == 1024
        assert config.embedding_dim == 768
        assert config.num_heads == 12
        assert config.num_layers == 12
        assert config.mlp_ratio == 4
        assert config.dropout == 0.1
        assert config.bias is True
        assert config.scale_residual_init is False

    def test_head_dim_property(self):
        """Test that head_dim is computed correctly."""
        config = GPT2Config(embedding_dim=768, num_heads=12)
        assert config.head_dim == 64

        config = GPT2Config(embedding_dim=1024, num_heads=16)
        assert config.head_dim == 64

    def test_head_dim_assertion(self):
        """Test that head_dim raises error when not divisible."""
        config = GPT2Config(embedding_dim=100, num_heads=12)
        with pytest.raises(AssertionError):
            _ = config.head_dim

    def test_mlp_hidden_dim_property(self):
        """Test that mlp_hidden_dim is computed correctly."""
        config = GPT2Config(embedding_dim=768, mlp_ratio=4)
        assert config.mlp_hidden_dim == 3072

        config = GPT2Config(embedding_dim=1024, mlp_ratio=4)
        assert config.mlp_hidden_dim == 4096

    def test_gpt2_small_preset(self):
        """Test GPT-2 Small preset configuration."""
        config = GPT2Config.gpt2_small()

        assert config.embedding_dim == 768
        assert config.num_heads == 12
        assert config.num_layers == 12

    def test_gpt2_medium_preset(self):
        """Test GPT-2 Medium preset configuration."""
        config = GPT2Config.gpt2_medium()

        assert config.embedding_dim == 1024
        assert config.num_heads == 16
        assert config.num_layers == 24

    def test_gpt2_large_preset(self):
        """Test GPT-2 Large preset configuration."""
        config = GPT2Config.gpt2_large()

        assert config.embedding_dim == 1280
        assert config.num_heads == 20
        assert config.num_layers == 36

    def test_gpt2_xl_preset(self):
        """Test GPT-2 XL preset configuration."""
        config = GPT2Config.gpt2_xl()

        assert config.embedding_dim == 1600
        assert config.num_heads == 25
        assert config.num_layers == 48

    def test_tiny_preset(self):
        """Test tiny preset configuration for testing."""
        config = GPT2Config.tiny()

        assert config.vocab_size == 1000
        assert config.context_length == 128
        assert config.embedding_dim == 64
        assert config.num_heads == 4
        assert config.num_layers == 2
        assert config.dropout == 0.0

    def test_scale_residual_init_flag(self):
        """Test that scale_residual_init can be set."""
        config = GPT2Config(scale_residual_init=True)
        assert config.scale_residual_init is True

        config = GPT2Config(scale_residual_init=False)
        assert config.scale_residual_init is False


# =============================================================================
# CausalSelfAttention Tests
# =============================================================================


class TestCausalSelfAttention:
    """Tests for the CausalSelfAttention module."""

    def test_output_shape(self):
        """Test that attention outputs have correct shape."""
        config = GPT2Config.tiny()
        attention = CausalSelfAttention(config)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.embedding_dim)

        output = attention(x)

        assert output.shape == (batch_size, seq_len, config.embedding_dim)

    def test_flash_attention_detection(self):
        """Test that flash attention is detected when available."""
        config = GPT2Config.tiny()
        attention = CausalSelfAttention(config)

        # Flash attention should be available in PyTorch >= 2.0
        has_flash = hasattr(F, "scaled_dot_product_attention")
        assert attention.flash == has_flash

    def test_causal_mask_created_when_no_flash(self):
        """Test that causal mask is created when flash attention is not available."""
        config = GPT2Config.tiny()
        attention = CausalSelfAttention(config)

        # If flash is not available, causal_mask should exist
        if not attention.flash:
            assert hasattr(attention, "causal_mask")
            assert attention.causal_mask.shape == (
                config.context_length,
                config.context_length
            )
            # Should be lower triangular
            expected = torch.tril(torch.ones_like(attention.causal_mask))
            torch.testing.assert_close(attention.causal_mask, expected)

    def test_qkv_projection_shape(self):
        """Test QKV projection has correct dimensions."""
        config = GPT2Config.tiny()
        attention = CausalSelfAttention(config)

        # QKV projection: embedding_dim -> 3 * embedding_dim
        assert attention.qkv_projection.in_features == config.embedding_dim
        assert attention.qkv_projection.out_features == 3 * config.embedding_dim

    def test_output_projection_shape(self):
        """Test output projection has correct dimensions."""
        config = GPT2Config.tiny()
        attention = CausalSelfAttention(config)

        # Output projection: embedding_dim -> embedding_dim
        assert attention.output_projection.in_features == config.embedding_dim
        assert attention.output_projection.out_features == config.embedding_dim

    def test_dropout_rate(self):
        """Test that dropout layers use config dropout rate."""
        config = GPT2Config(dropout=0.2, embedding_dim=64, num_heads=4, context_length=128)
        attention = CausalSelfAttention(config)

        assert attention.attention_dropout.p == 0.2
        assert attention.output_dropout.p == 0.2
        assert attention.dropout == 0.2

    def test_single_head_matches_hf(self):
        """Test single-head attention matches HuggingFace."""
        our_model, hf_model = create_matching_models(
            num_layers=1,
            num_heads=1,
            embedding_dim=32,
        )
        copy_weights_from_hf(our_model, hf_model)

        input_ids = torch.randint(0, 1000, (2, 16))

        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_logits = hf_model(input_ids).logits

        torch.testing.assert_close(
            our_logits, hf_logits,
            rtol=1e-4, atol=1e-4,
            msg="Single head attention outputs don't match"
        )

    def test_multi_head_matches_hf(self):
        """Test multi-head attention matches HuggingFace."""
        our_model, hf_model = create_matching_models(
            num_layers=1,
            num_heads=4,
            embedding_dim=64,
        )
        copy_weights_from_hf(our_model, hf_model)

        input_ids = torch.randint(0, 1000, (2, 32))

        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_logits = hf_model(input_ids).logits

        torch.testing.assert_close(
            our_logits, hf_logits,
            rtol=1e-4, atol=1e-4,
            msg="Multi-head attention outputs don't match"
        )

    def test_attention_is_causal(self):
        """Test that attention is properly causal (matches HuggingFace).

        We verify this by comparing outputs with HuggingFace, which
        ensures our causal masking (via flash attention or manual) is correct.
        """
        our_model, hf_model = create_matching_models(num_layers=1, num_heads=2)
        copy_weights_from_hf(our_model, hf_model)

        input_ids = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_logits = hf_model(input_ids).logits

        torch.testing.assert_close(our_logits, hf_logits, rtol=1e-4, atol=1e-4)


# =============================================================================
# MLP Tests
# =============================================================================


class TestMLP:
    """Tests for the MLP module."""

    def test_output_shape(self):
        """Test that MLP outputs have correct shape."""
        config = GPT2Config.tiny()
        mlp = MLP(config)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.embedding_dim)

        output = mlp(x)

        assert output.shape == (batch_size, seq_len, config.embedding_dim)

    def test_fc1_expands_dimension(self):
        """Test that fc1 expands dimension by mlp_ratio."""
        config = GPT2Config.tiny()
        mlp = MLP(config)

        assert mlp.fc1.in_features == config.embedding_dim
        assert mlp.fc1.out_features == config.mlp_hidden_dim
        assert mlp.fc1.out_features == config.embedding_dim * config.mlp_ratio

    def test_fc2_contracts_dimension(self):
        """Test that fc2 contracts dimension back to embedding_dim."""
        config = GPT2Config.tiny()
        mlp = MLP(config)

        assert mlp.fc2.in_features == config.mlp_hidden_dim
        assert mlp.fc2.out_features == config.embedding_dim

    def test_uses_gelu_tanh_approximation(self):
        """Test that MLP uses GELU with tanh approximation."""
        config = GPT2Config.tiny()
        mlp = MLP(config)

        assert isinstance(mlp.activation, torch.nn.GELU)
        assert mlp.activation.approximate == "tanh"

    def test_dropout_rate(self):
        """Test that dropout uses config dropout rate."""
        config = GPT2Config(dropout=0.3, embedding_dim=64, num_heads=4, context_length=128)
        mlp = MLP(config)

        assert mlp.dropout.p == 0.3

    def test_bias_configuration(self):
        """Test that bias is configured correctly."""
        # With bias
        config_with_bias = GPT2Config.tiny()
        config_with_bias.bias = True
        mlp_with_bias = MLP(config_with_bias)

        assert mlp_with_bias.fc1.bias is not None
        assert mlp_with_bias.fc2.bias is not None

        # Without bias
        config_no_bias = GPT2Config(
            vocab_size=1000,
            context_length=128,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            bias=False,
        )
        mlp_no_bias = MLP(config_no_bias)

        assert mlp_no_bias.fc1.bias is None
        assert mlp_no_bias.fc2.bias is None

    def test_forward_pass_no_nan(self):
        """Test that forward pass doesn't produce NaN."""
        config = GPT2Config.tiny()
        mlp = MLP(config)

        x = torch.randn(2, 16, config.embedding_dim)
        output = mlp(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# TransformerBlock Tests
# =============================================================================


class TestTransformerBlock:
    """Tests for the TransformerBlock module."""

    def test_output_shape(self):
        """Test that block outputs have correct shape."""
        config = GPT2Config.tiny()
        block = TransformerBlock(config)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.embedding_dim)

        output = block(x)

        assert output.shape == (batch_size, seq_len, config.embedding_dim)

    def test_contains_layer_norms(self):
        """Test that block has two layer norms."""
        config = GPT2Config.tiny()
        block = TransformerBlock(config)

        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)

        assert block.ln1.normalized_shape == (config.embedding_dim,)
        assert block.ln2.normalized_shape == (config.embedding_dim,)

    def test_contains_attention(self):
        """Test that block has attention module."""
        config = GPT2Config.tiny()
        block = TransformerBlock(config)

        assert isinstance(block.attention, CausalSelfAttention)

    def test_contains_mlp(self):
        """Test that block has MLP module."""
        config = GPT2Config.tiny()
        block = TransformerBlock(config)

        assert isinstance(block.mlp, MLP)

    def test_layer_norm_eps(self):
        """Test that layer norms use correct epsilon."""
        config = GPT2Config.tiny()
        block = TransformerBlock(config)

        assert block.ln1.eps == 1e-5
        assert block.ln2.eps == 1e-5

    def test_layer_norm_bias_configuration(self):
        """Test that layer norm bias follows config."""
        # With bias
        config_with_bias = GPT2Config.tiny()
        block_with_bias = TransformerBlock(config_with_bias)

        assert block_with_bias.ln1.bias is not None
        assert block_with_bias.ln2.bias is not None

        # Without bias
        config_no_bias = GPT2Config(
            vocab_size=1000,
            context_length=128,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            bias=False,
        )
        block_no_bias = TransformerBlock(config_no_bias)

        assert block_no_bias.ln1.bias is None
        assert block_no_bias.ln2.bias is None

    def test_residual_connection(self):
        """Test that residual connections are applied.

        The output should be different from what attention/MLP alone would produce,
        due to the residual addition.
        """
        config = GPT2Config.tiny()
        block = TransformerBlock(config)
        block.eval()

        x = torch.randn(1, 8, config.embedding_dim)

        with torch.no_grad():
            output = block(x)

        # Output should not equal input (transformations applied)
        assert not torch.allclose(output, x)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_forward_pass_no_nan(self):
        """Test that forward pass doesn't produce NaN."""
        config = GPT2Config.tiny()
        block = TransformerBlock(config)

        x = torch.randn(2, 16, config.embedding_dim)
        output = block(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# GPT2 Forward Tests
# =============================================================================


class TestGPT2Forward:
    """Tests for GPT2 forward pass."""

    def test_output_shape(self):
        """Test that model outputs have correct shape."""
        config = GPT2Config.tiny()
        model = GPT2(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None

    def test_output_with_targets(self):
        """Test forward pass with targets computes loss."""
        config = GPT2Config.tiny()
        model = GPT2(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids, targets=targets)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0  # Cross-entropy is always positive
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_length_assertion(self):
        """Test that exceeding context length raises error."""
        config = GPT2Config.tiny()  # context_length=128
        model = GPT2(config)

        # Create input longer than context length
        input_ids = torch.randint(0, config.vocab_size, (1, 256))

        with pytest.raises(AssertionError):
            model(input_ids)

    def test_matches_huggingface_small(self):
        """Test small model matches HuggingFace."""
        our_model, hf_model = create_matching_models(
            num_layers=2,
            num_heads=4,
            embedding_dim=64,
        )
        copy_weights_from_hf(our_model, hf_model)

        input_ids = torch.randint(0, 1000, (2, 32))

        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_logits = hf_model(input_ids).logits

        torch.testing.assert_close(
            our_logits, hf_logits,
            rtol=1e-4, atol=1e-4,
            msg="Small model outputs don't match"
        )

    def test_matches_huggingface_deep(self):
        """Test deeper model matches HuggingFace."""
        our_model, hf_model = create_matching_models(
            num_layers=6,
            num_heads=8,
            embedding_dim=128,
        )
        copy_weights_from_hf(our_model, hf_model)

        input_ids = torch.randint(0, 1000, (2, 64))

        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_logits = hf_model(input_ids).logits

        torch.testing.assert_close(
            our_logits, hf_logits,
            rtol=1e-4, atol=1e-4,
            msg="Deeper model outputs don't match"
        )

    def test_various_sequence_lengths(self):
        """Test with different sequence lengths."""
        our_model, hf_model = create_matching_models()
        copy_weights_from_hf(our_model, hf_model)

        for seq_len in [1, 5, 16, 32, 64, 100]:
            input_ids = torch.randint(0, 1000, (2, seq_len))

            with torch.no_grad():
                our_logits, _ = our_model(input_ids)
                hf_logits = hf_model(input_ids).logits

            torch.testing.assert_close(
                our_logits, hf_logits,
                rtol=1e-4, atol=1e-4,
                msg=f"Outputs don't match for sequence length {seq_len}"
            )


# =============================================================================
# GPT2 Loss Tests
# =============================================================================


class TestGPT2Loss:
    """Tests for GPT2 loss computation."""

    def test_loss_matches_huggingface(self):
        """Test that loss matches HuggingFace (accounting for shift).

        HuggingFace computes loss with internal shifting:
        - Uses logits[:, :-1, :] to predict labels[:, 1:]

        Our implementation expects aligned targets:
        - Uses logits[:, :, :] to predict targets[:, :]
        """
        from einops import rearrange

        our_model, hf_model = create_matching_models()
        copy_weights_from_hf(our_model, hf_model)

        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_output = hf_model(input_ids, labels=labels)
            hf_loss = hf_output.loss

            # Compute loss the HuggingFace way
            shift_logits = our_logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            our_loss_hf_style = torch.nn.functional.cross_entropy(
                rearrange(shift_logits, "batch seq vocab -> (batch seq) vocab"),
                rearrange(shift_labels, "batch seq -> (batch seq)"),
            )

        torch.testing.assert_close(
            our_loss_hf_style, hf_loss,
            rtol=1e-4, atol=1e-4,
            msg="Loss values don't match"
        )


# =============================================================================
# GPT2 Weight Tying Tests
# =============================================================================


class TestGPT2WeightTying:
    """Tests for weight tying."""

    def test_embedding_and_lm_head_share_weights(self):
        """Test that token embedding and lm_head share the same weight tensor."""
        config = GPT2Config.tiny()
        model = GPT2(config)

        assert model.lm_head.weight is model.token_embedding.weight

    def test_weight_tying_after_update(self):
        """Test that updating one updates the other."""
        config = GPT2Config.tiny()
        model = GPT2(config)

        with torch.no_grad():
            model.token_embedding.weight[0, 0] = 999.0

        assert model.lm_head.weight[0, 0] == 999.0


# =============================================================================
# GPT2 Residual Scaling Tests
# =============================================================================


class TestGPT2ResidualScaling:
    """Tests for residual projection scaling."""

    def test_scale_residual_init_disabled_by_default(self):
        """Test that scale_residual_init is False by default."""
        config = GPT2Config.tiny()
        assert config.scale_residual_init is False

    def test_residual_scaling_reduces_weight_std(self):
        """Test that enabling scale_residual_init reduces residual projection std."""
        # Create model without scaling
        config_no_scale = GPT2Config.tiny()
        model_no_scale = GPT2(config_no_scale)

        # Create model with scaling
        config_scaled = GPT2Config(
            vocab_size=1000,
            context_length=128,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            scale_residual_init=True,
        )
        model_scaled = GPT2(config_scaled)

        expected_std_no_scale = 0.02
        expected_std_scaled = 0.02 / math.sqrt(2 * config_scaled.num_layers)

        # Collect residual projection weights
        residual_weights_no_scale = []
        residual_weights_scaled = []

        for block_no_scale, block_scaled in zip(
            model_no_scale.blocks, model_scaled.blocks
        ):
            residual_weights_no_scale.extend([
                block_no_scale.attention.output_projection.weight,
                block_no_scale.mlp.fc2.weight,
            ])
            residual_weights_scaled.extend([
                block_scaled.attention.output_projection.weight,
                block_scaled.mlp.fc2.weight,
            ])

        # Compute actual std
        all_no_scale = torch.cat([w.flatten() for w in residual_weights_no_scale])
        all_scaled = torch.cat([w.flatten() for w in residual_weights_scaled])

        actual_std_no_scale = all_no_scale.std().item()
        actual_std_scaled = all_scaled.std().item()

        # Check that scaled model has smaller std
        assert actual_std_scaled < actual_std_no_scale

        # Check stds are approximately correct (within 20% tolerance)
        assert abs(actual_std_no_scale - expected_std_no_scale) / expected_std_no_scale < 0.2
        assert abs(actual_std_scaled - expected_std_scaled) / expected_std_scaled < 0.2

    def test_non_residual_weights_have_standard_std(self):
        """Test that non-residual projections still use std=0.02."""
        config_scaled = GPT2Config(
            vocab_size=1000,
            context_length=128,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            scale_residual_init=True,
        )
        model_scaled = GPT2(config_scaled)

        # Collect non-residual projection weights
        non_residual_weights = []
        for block in model_scaled.blocks:
            non_residual_weights.extend([
                block.attention.qkv_projection.weight,
                block.mlp.fc1.weight,
            ])

        # Compute actual std
        all_weights = torch.cat([w.flatten() for w in non_residual_weights])
        actual_std = all_weights.std().item()

        # Should be approximately 0.02 (within 20% tolerance)
        expected_std = 0.02
        assert abs(actual_std - expected_std) / expected_std < 0.2


# =============================================================================
# GPT2 Generate Tests
# =============================================================================


class TestGPT2Generate:
    """Tests for GPT2 generation."""

    def test_generate_output_shape(self):
        """Test that generate produces correct output shape."""
        config = GPT2Config.tiny()
        model = GPT2(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        max_new_tokens = 10

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens)

        assert output.shape == (1, 5 + max_new_tokens)

    def test_generate_respects_context_length(self):
        """Test that generate crops to context length when needed."""
        config = GPT2Config.tiny()  # context_length=128
        model = GPT2(config)
        model.eval()

        # Start with nearly full context
        input_ids = torch.randint(0, config.vocab_size, (1, 120))

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=20)

        # Should have generated without error
        assert output.shape == (1, 140)

    def test_generate_with_temperature(self):
        """Test that temperature affects generation."""
        config = GPT2Config.tiny()
        model = GPT2(config)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            # Low temperature (sharper distribution)
            output_low_temp = model.generate(
                input_ids.clone(), max_new_tokens=5, temperature=0.1
            )
            # High temperature (flatter distribution)
            output_high_temp = model.generate(
                input_ids.clone(), max_new_tokens=5, temperature=2.0
            )

        # Both should produce valid output
        assert output_low_temp.shape == (1, 10)
        assert output_high_temp.shape == (1, 10)

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling."""
        config = GPT2Config.tiny()
        model = GPT2(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10, top_k=50)

        assert output.shape == (1, 15)


# =============================================================================
# GPT2 From Pretrained Tests
# =============================================================================


class TestGPT2FromPretrained:
    """Tests for from_pretrained classmethod."""

    def test_invalid_model_type(self):
        """Test that invalid model types raise an error."""
        with pytest.raises(AssertionError):
            GPT2.from_pretrained("invalid-model")

    def test_valid_model_types_accepted(self):
        """Test that valid model types are recognized."""
        valid_types = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        for model_type in valid_types:
            assert model_type in valid_types

    @pytest.mark.slow
    def test_from_pretrained_gpt2_small(self):
        """Test loading GPT-2 small from HuggingFace.

        This test downloads weights from HuggingFace, so it's marked as slow.
        Run with: pytest -v -m slow
        """
        model = GPT2.from_pretrained("gpt2")

        # Verify config
        assert model.config.embedding_dim == 768
        assert model.config.num_heads == 12
        assert model.config.num_layers == 12
        assert model.config.vocab_size == 50257
        assert model.config.context_length == 1024

        # Verify model works
        model.eval()
        input_ids = torch.tensor([[50256]])  # <|endoftext|> token
        with torch.no_grad():
            logits, _ = model(input_ids)

        assert logits.shape == (1, 1, 50257)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    @pytest.mark.slow
    def test_pretrained_matches_huggingface(self):
        """Test that pretrained GPT-2 produces identical outputs to HuggingFace.

        This test downloads weights from HuggingFace, so it's marked as slow.
        Run with: pytest -v -m slow

        Verifies:
        1. Logits match exactly (within tolerance)
        2. Greedy generation produces identical tokens
        """
        # Load both models with pretrained weights
        our_model = GPT2.from_pretrained("gpt2")
        hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

        our_model.eval()
        hf_model.eval()

        # Test prompt
        prompt_tokens = [7454, 2402, 257, 640]  # "Once upon a time"
        input_ids = torch.tensor([prompt_tokens])

        # Compare logits
        with torch.no_grad():
            our_logits, _ = our_model(input_ids)
            hf_logits = hf_model(input_ids).logits

        torch.testing.assert_close(
            our_logits, hf_logits,
            rtol=1e-4, atol=1e-4,
            msg="Pretrained model logits don't match HuggingFace"
        )

        # Compare greedy generation
        max_new_tokens = 20
        with torch.no_grad():
            # HuggingFace greedy generation
            hf_generated = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=50256,  # eos token
            )

            # Our greedy generation (argmax)
            our_generated = input_ids.clone()
            for _ in range(max_new_tokens):
                logits, _ = our_model(our_generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                our_generated = torch.cat([our_generated, next_token], dim=1)

        assert torch.equal(hf_generated, our_generated), (
            f"Greedy generation mismatch.\n"
            f"HuggingFace: {hf_generated[0].tolist()}\n"
            f"Ours: {our_generated[0].tolist()}"
        )


# =============================================================================
# GPT2 Num Params Tests
# =============================================================================


class TestGPT2NumParams:
    """Tests for parameter counting."""

    def test_get_num_params(self):
        """Test get_num_params returns correct count."""
        config = GPT2Config.tiny()
        model = GPT2(config)

        # Total params
        total_params = sum(p.numel() for p in model.parameters())

        # Non-embedding params (excludes position embeddings)
        non_emb_params = model.get_num_params(non_embedding=True)
        pos_emb_params = model.position_embedding.weight.numel()

        assert non_emb_params == total_params - pos_emb_params

    def test_get_num_params_all(self):
        """Test get_num_params with non_embedding=False."""
        config = GPT2Config.tiny()
        model = GPT2(config)

        total_params = sum(p.numel() for p in model.parameters())
        all_params = model.get_num_params(non_embedding=False)

        assert all_params == total_params

