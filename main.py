"""
Simple test script to verify the GPT-2 implementation works.
"""

import torch
from allformers.models.gpt2.gpt2 import GPT2, GPT2Config


def test_tiny_model():
    """Test that a tiny model can do a forward pass."""
    print("=" * 60)
    print("Testing tiny GPT-2 model")
    print("=" * 60)

    # Create a tiny config for testing
    config = GPT2Config.tiny()
    print(f"\nConfig: {config}")

    # Create model
    model = GPT2(config)

    # Create some dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    logits, loss = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss (no targets): {loss}")

    # Forward pass with targets (for training)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, targets=targets)
    print(f"Loss (with targets): {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    start_tokens = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\n✓ Tiny model test passed!")


def test_gpt2_small_shapes():
    """Test that GPT-2 small has correct shapes."""
    print("\n" + "=" * 60)
    print("Testing GPT-2 Small configuration shapes")
    print("=" * 60)

    config = GPT2Config.gpt2_small()
    print(f"\nConfig: {config}")

    model = GPT2(config)

    # Verify shapes with a single forward pass
    batch_size = 1
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, _ = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, "
        f"got {logits.shape}"
    )

    print(f"\n✓ Output shape correct: {logits.shape}")
    print("✓ GPT-2 Small shape test passed!")


def main():
    print("allformers - GPT-2 Implementation Test\n")

    test_tiny_model()
    test_gpt2_small_shapes()

    print("\n" + "=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
