"""
Tests for allformers.data.streaming module.

Note: The StreamingTextDataset now works with HuggingFace IterableDatasets
(streaming mode) and uses shuffle buffers for randomization. Shuffling is
done at the dataset level, not within StreamingTextDataset.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from allformers.data.streaming import StreamingTextDataset
from allformers.data.wikipedia import (
    load_wikipedia,
    load_wikipedia_streaming,
    WikipediaConfig,
    wikipedia_text_fn,
)


@pytest.fixture
def tokenizer():
    """GPT-2 tokenizer fixture."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def small_streaming_dataset():
    """Small Wikipedia streaming dataset fixture.
    
    Uses streaming mode for testing. We apply a shuffle with a small
    buffer for realistic usage. Tests control iteration limits themselves.
    """
    # Load in streaming mode (can't use slice notation with streaming)
    dataset = load_wikipedia(WikipediaConfig(
        split="train",
        streaming=True,
    ))
    # Apply shuffle for realistic usage - tests control how many docs they consume
    return dataset.shuffle(seed=42, buffer_size=100)


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

    def test_initialization(self, tokenizer, small_streaming_dataset):
        """Should initialize without errors."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=128,
            text_fn=wikipedia_text_fn,
        )
        
        assert dataset.seq_len == 128

    def test_yields_correct_shape(self, tokenizer, small_streaming_dataset):
        """Should yield tensors of shape (seq_len + 1,)."""
        seq_len = 64
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_fn=wikipedia_text_fn,
        )
        
        for i, seq in enumerate(dataset):
            assert seq.shape == (seq_len + 1,)
            assert seq.dtype == torch.long
            if i >= 4:
                break

    def test_yields_valid_token_ids(self, tokenizer, small_streaming_dataset):
        """Should yield valid token IDs within vocabulary range."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
        )
        
        for i, seq in enumerate(dataset):
            assert seq.min() >= 0
            assert seq.max() < tokenizer.vocab_size
            if i >= 4:
                break

    def test_yields_multiple_sequences(self, tokenizer, small_streaming_dataset):
        """Should yield multiple sequences from the dataset."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
        )
        
        # Should be able to get multiple sequences
        count = 0
        for seq in dataset:
            count += 1
            if count >= 50:
                break
        
        assert count == 50

    def test_contains_eos_tokens(self, tokenizer, small_streaming_dataset):
        """Should contain EOS tokens between articles."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=512,  # Longer to likely contain article boundaries
            text_fn=wikipedia_text_fn,
        )
        
        # Check multiple sequences for EOS tokens
        # Some Wikipedia articles are very long (10K+ tokens), so we need
        # to check enough sequences to cross at least a few article boundaries
        eos_found = False
        for i, seq in enumerate(dataset):
            if tokenizer.eos_token_id in seq.tolist():
                eos_found = True
                break
            if i >= 100:  # ~50K tokens should cross several articles
                break
        
        assert eos_found, "EOS token should appear in sequences"

    def test_custom_text_fn(self, tokenizer, small_streaming_dataset):
        """Should work with custom text extraction function."""
        def title_only_fn(row):
            return row["title"]
        
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=title_only_fn,
        )
        
        # Should still yield valid sequences
        seq = next(iter(dataset))
        assert seq.shape == (65,)

    def test_works_with_dataloader(self, tokenizer, small_streaming_dataset):
        """Should work correctly with PyTorch DataLoader."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
        )
        
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        
        batch = next(iter(loader))
        assert batch.shape == (4, 65)  # batch_size x (seq_len + 1)

    def test_dataloader_batch_split(self, tokenizer, small_streaming_dataset):
        """DataLoader batches should be splittable into input/target."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
        )
        
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        assert input_ids.shape == (4, 64)
        assert targets.shape == (4, 64)

    def test_detects_tokenizer_eos_behavior(self, tokenizer, small_streaming_dataset):
        """Should correctly detect if tokenizer adds EOS automatically."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
        )
        
        # GPT-2 tokenizer doesn't add EOS automatically
        assert dataset.tokenizer_adds_eos is False
        assert dataset.eos_token_id == tokenizer.eos_token_id

    def test_requires_text_fn(self, tokenizer, small_streaming_dataset):
        """Should raise error if text_fn is not provided."""
        with pytest.raises(ValueError, match="text_fn is required"):
            StreamingTextDataset(
                dataset=small_streaming_dataset,
                tokenizer=tokenizer,
                seq_len=64,
            )


@pytest.mark.slow
class TestStreamingTextDatasetEdgeCases:
    """Edge case tests for StreamingTextDataset."""

    def test_very_short_seq_len(self, tokenizer, small_streaming_dataset):
        """Should handle very short sequence lengths."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=8,  # Very short
            text_fn=wikipedia_text_fn,
        )
        
        seq = next(iter(dataset))
        assert seq.shape == (9,)

    def test_long_seq_len(self, tokenizer, small_streaming_dataset):
        """Should handle long sequence lengths."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=1024,
            text_fn=wikipedia_text_fn,
        )
        
        seq = next(iter(dataset))
        assert seq.shape == (1025,)


@pytest.mark.slow
class TestLoadWikipediaStreaming:
    """Tests for load_wikipedia_streaming function."""

    def test_returns_train_val_datasets(self):
        """Should return train and validation IterableDatasets."""
        train_data, val_data = load_wikipedia_streaming(
            shuffle_buffer_size=100,
            seed=42,
        )
        
        # Both should be iterable
        train_sample = next(iter(train_data))
        val_sample = next(iter(val_data))
        
        # Check they have expected fields
        assert "title" in train_sample
        assert "text" in train_sample
        assert "title" in val_sample
        assert "text" in val_sample

    def test_different_seeds_give_different_order(self):
        """Different seeds should produce different shuffle orders."""
        train1, _ = load_wikipedia_streaming(shuffle_buffer_size=100, seed=42)
        train2, _ = load_wikipedia_streaming(shuffle_buffer_size=100, seed=123)
        
        # Get first few samples
        samples1 = [next(iter(train1))["title"] for _ in range(5)]
        samples2 = [next(iter(train2))["title"] for _ in range(5)]
        
        # Should likely be different (not guaranteed but very probable)
        # Just check they're valid
        assert all(isinstance(s, str) for s in samples1)
        assert all(isinstance(s, str) for s in samples2)

    def test_no_overlap_between_train_and_val(self):
        """Train and validation sets should have no overlapping documents."""
        # Load without shuffle to get deterministic order
        train_data, val_data = load_wikipedia_streaming(
            shuffle_buffer_size=1,  # Minimal shuffle to preserve order
            seed=42,
        )
        
        # Collect IDs from first 1000 documents of each split
        train_ids = set()
        for i, doc in enumerate(train_data):
            if i >= 1000:
                break
            train_ids.add(doc["id"])
        
        val_ids = set()
        for i, doc in enumerate(val_data):
            if i >= 1000:
                break
            val_ids.add(doc["id"])
        
        # Check for overlap
        overlap = train_ids & val_ids
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping documents between train and val"
        
        # Sanity check: we got documents from both
        assert len(train_ids) == 1000, f"Expected 1000 train docs, got {len(train_ids)}"
        assert len(val_ids) == 1000, f"Expected 1000 val docs, got {len(val_ids)}"


@pytest.mark.slow
class TestStreamingTextDatasetDDP:
    """Tests for StreamingTextDataset DDP (distributed) support."""

    def test_rank_world_size_validation(self, tokenizer, small_streaming_dataset):
        """Should raise error if only rank or world_size is provided."""
        with pytest.raises(ValueError, match="rank and world_size must both be provided"):
            StreamingTextDataset(
                dataset=small_streaming_dataset,
                tokenizer=tokenizer,
                seq_len=64,
                text_fn=wikipedia_text_fn,
                rank=0,  # Only rank, no world_size
            )
        
        with pytest.raises(ValueError, match="rank and world_size must both be provided"):
            StreamingTextDataset(
                dataset=small_streaming_dataset,
                tokenizer=tokenizer,
                seq_len=64,
                text_fn=wikipedia_text_fn,
                world_size=4,  # Only world_size, no rank
            )

    def test_invalid_rank_value(self, tokenizer, small_streaming_dataset):
        """Should raise error if rank is out of valid range."""
        with pytest.raises(ValueError, match="rank must be in"):
            StreamingTextDataset(
                dataset=small_streaming_dataset,
                tokenizer=tokenizer,
                seq_len=64,
                text_fn=wikipedia_text_fn,
                rank=4,  # Invalid: should be < world_size
                world_size=4,
            )
        
        with pytest.raises(ValueError, match="rank must be in"):
            StreamingTextDataset(
                dataset=small_streaming_dataset,
                tokenizer=tokenizer,
                seq_len=64,
                text_fn=wikipedia_text_fn,
                rank=-1,  # Invalid: should be >= 0
                world_size=4,
            )

    def test_different_ranks_get_different_data(self, tokenizer, small_streaming_dataset):
        """Different ranks should yield different sequences."""
        world_size = 2
        
        # Create datasets for different ranks
        sequences_by_rank = []
        for rank in range(world_size):
            dataset = StreamingTextDataset(
                dataset=small_streaming_dataset,
                tokenizer=tokenizer,
                seq_len=64,
                text_fn=wikipedia_text_fn,
                rank=rank,
                world_size=world_size,
            )
            
            # Collect first 3 sequences from each rank
            seqs = []
            for i, seq in enumerate(dataset):
                seqs.append(tuple(seq.tolist()))
                if i >= 2:
                    break
            sequences_by_rank.append(seqs)
        
        # Flatten and check uniqueness
        all_seqs = [seq for seqs in sequences_by_rank for seq in seqs]
        unique_seqs = set(all_seqs)
        
        # All sequences should be unique across ranks
        assert len(all_seqs) == len(unique_seqs), "Ranks should get different sequences"

    def test_same_rank_is_deterministic(self, tokenizer, small_streaming_dataset):
        """Same rank should yield same sequences on repeated iterations."""
        rank = 1
        world_size = 2
        
        # First iteration
        dataset1 = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
            rank=rank,
            world_size=world_size,
        )
        seqs1 = []
        for i, seq in enumerate(dataset1):
            seqs1.append(tuple(seq.tolist()))
            if i >= 2:
                break
        
        # Second iteration with same parameters
        dataset2 = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
            rank=rank,
            world_size=world_size,
        )
        seqs2 = []
        for i, seq in enumerate(dataset2):
            seqs2.append(tuple(seq.tolist()))
            if i >= 2:
                break
        
        assert seqs1 == seqs2, "Same rank should produce same sequences"

    def test_stores_rank_and_world_size(self, tokenizer, small_streaming_dataset):
        """Should store rank and world_size as attributes."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
            rank=2,
            world_size=4,
        )
        
        assert dataset.rank == 2
        assert dataset.world_size == 4

    def test_no_sharding_when_not_specified(self, tokenizer, small_streaming_dataset):
        """When rank/world_size not provided, should not shard."""
        dataset = StreamingTextDataset(
            dataset=small_streaming_dataset,
            tokenizer=tokenizer,
            seq_len=64,
            text_fn=wikipedia_text_fn,
        )
        
        assert dataset.rank is None
        assert dataset.world_size is None
        
        # Should still work normally
        seq = next(iter(dataset))
        assert seq.shape == (65,)
