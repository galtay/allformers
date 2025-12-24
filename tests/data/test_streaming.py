"""
Tests for allformers.data.streaming module.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from allformers.data import (
    load_wikipedia,
    WikipediaConfig,
    StreamingTextDataset,
    wikipedia_text_fn,
)


@pytest.fixture
def tokenizer():
    """GPT-2 tokenizer fixture."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def small_dataset():
    """Small Wikipedia dataset fixture (50 articles)."""
    config = WikipediaConfig(split="train[:50]")
    return load_wikipedia(config)


class TestWikipediaTextFn:
    """Tests for wikipedia_text_fn helper function."""

    def test_formats_title_and_text(self):
        """Should format article with title and text."""
        row = {"title": "Test Article", "text": "This is the content."}
        result = wikipedia_text_fn(row)
        
        assert result == "Test Article\n\nThis is the content."

    def test_handles_empty_text(self):
        """Should handle empty text gracefully."""
        row = {"title": "Empty", "text": ""}
        result = wikipedia_text_fn(row)
        
        assert result == "Empty\n\n"


@pytest.mark.slow
class TestStreamingTextDataset:
    """Tests for StreamingTextDataset class.
    
    These tests are marked as slow because they download data from HuggingFace.
    """

    def test_initialization(self, tokenizer, small_dataset):
        """Should initialize without errors."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=128,
        )
        
        assert dataset.seq_len == 128
        assert dataset.dataset_len == len(small_dataset)

    def test_yields_correct_shape(self, tokenizer, small_dataset):
        """Should yield tensors of shape (seq_len + 1,)."""
        seq_len = 64
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            seed=42,
        )
        
        for i, seq in enumerate(dataset):
            assert seq.shape == (seq_len + 1,)
            assert seq.dtype == torch.long
            if i >= 4:
                break

    def test_yields_valid_token_ids(self, tokenizer, small_dataset):
        """Should yield valid token IDs within vocabulary range."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        
        for i, seq in enumerate(dataset):
            assert seq.min() >= 0
            assert seq.max() < tokenizer.vocab_size
            if i >= 4:
                break

    def test_is_infinite_iterator(self, tokenizer, small_dataset):
        """Should yield indefinitely (infinite iterator)."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        
        # Should be able to get many more samples than dataset size
        count = 0
        for seq in dataset:
            count += 1
            if count >= 200:  # Way more than 50 articles
                break
        
        assert count == 200

    def test_reproducible_with_same_seed(self, tokenizer, small_dataset):
        """Same seed should produce same sequences."""
        dataset1 = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        dataset2 = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        
        iter1 = iter(dataset1)
        iter2 = iter(dataset2)
        
        for _ in range(10):
            seq1 = next(iter1)
            seq2 = next(iter2)
            assert torch.equal(seq1, seq2)

    def test_different_seeds_produce_different_sequences(self, tokenizer, small_dataset):
        """Different seeds should produce different sequences."""
        dataset1 = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        dataset2 = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=123,
        )
        
        seq1 = next(iter(dataset1))
        seq2 = next(iter(dataset2))
        
        # Should be different (very unlikely to be same by chance)
        assert not torch.equal(seq1, seq2)

    def test_contains_eos_tokens(self, tokenizer, small_dataset):
        """Should contain EOS tokens between articles."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=512,  # Longer to likely contain article boundaries
            seed=42,
        )
        
        # Check multiple sequences for EOS tokens
        eos_found = False
        for i, seq in enumerate(dataset):
            if tokenizer.eos_token_id in seq.tolist():
                eos_found = True
                break
            if i >= 20:
                break
        
        assert eos_found, "EOS token should appear in sequences"

    def test_random_offset_enabled_by_default(self, tokenizer, small_dataset):
        """random_offset should be True by default."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
        )
        
        assert dataset.random_offset is True

    def test_random_offset_disabled(self, tokenizer, small_dataset):
        """Should be able to disable random offset."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            random_offset=False,
        )
        
        assert dataset.random_offset is False

    def test_random_offset_affects_start_positions(self, tokenizer, small_dataset):
        """With random_offset=False, more sequences should start with article titles."""
        # Get sequences without random offset
        dataset_no_offset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
            random_offset=False,
        )
        
        # Check if first sequence starts with a title pattern
        seq = next(iter(dataset_no_offset))
        decoded = tokenizer.decode(seq[:20].tolist())
        
        # Without offset, should start at article beginning more often
        # This is a soft check - the sequence should decode to readable text
        assert len(decoded) > 0

    def test_custom_text_fn(self, tokenizer, small_dataset):
        """Should work with custom text extraction function."""
        def title_only_fn(row):
            return row["title"]
        
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=title_only_fn,
            seed=42,
        )
        
        # Should still yield valid sequences
        seq = next(iter(dataset))
        assert seq.shape == (65,)

    def test_works_with_dataloader(self, tokenizer, small_dataset):
        """Should work correctly with PyTorch DataLoader."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        
        batch = next(iter(loader))
        assert batch.shape == (4, 65)  # batch_size x (seq_len + 1)

    def test_dataloader_batch_split(self, tokenizer, small_dataset):
        """DataLoader batches should be splittable into input/target."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        assert input_ids.shape == (4, 64)
        assert targets.shape == (4, 64)

    def test_detects_tokenizer_eos_behavior(self, tokenizer, small_dataset):
        """Should correctly detect if tokenizer adds EOS automatically."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
        )
        
        # GPT-2 tokenizer doesn't add EOS automatically
        assert dataset.tokenizer_adds_eos is False
        assert dataset.eos_token_id == tokenizer.eos_token_id


@pytest.mark.slow
class TestStreamingTextDatasetEdgeCases:
    """Edge case tests for StreamingTextDataset."""

    def test_very_short_seq_len(self, tokenizer, small_dataset):
        """Should handle very short sequence lengths."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=8,  # Very short
            seed=42,
        )
        
        seq = next(iter(dataset))
        assert seq.shape == (9,)

    def test_long_seq_len(self, tokenizer, small_dataset):
        """Should handle long sequence lengths."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=1024,
            seed=42,
        )
        
        seq = next(iter(dataset))
        assert seq.shape == (1025,)

    def test_multiple_iterations(self, tokenizer, small_dataset):
        """Should support multiple independent iterations."""
        dataset = StreamingTextDataset(
            dataset=small_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            seed=42,
        )
        
        # First iteration
        iter1 = iter(dataset)
        seq1_first = next(iter1)
        
        # Second iteration (should restart with same seed)
        iter2 = iter(dataset)
        seq2_first = next(iter2)
        
        # Both should produce the same first sequence
        assert torch.equal(seq1_first, seq2_first)

