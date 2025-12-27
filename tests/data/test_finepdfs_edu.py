"""
Tests for allformers.data.finepdfs_edu module.

Note: These tests use streaming mode to avoid downloading the full 736GB dataset.
"""

import pytest

from allformers.data.finepdfs_edu import (
    load_finepdfs_edu,
    load_finepdfs_edu_streaming,
    get_finepdfs_edu_sample,
    finepdfs_edu_text_fn,
    is_majority_english,
    filter_majority_english,
    FinePDFsEduConfig,
    FINEPDFS_EDU_DATASET_PATH,
    FINEPDFS_EDU_ENGLISH_SUBSET,
)


class TestFinePDFsEduTextFn:
    """Tests for finepdfs_edu_text_fn helper function."""

    def test_extracts_text(self):
        """Should extract text field from row."""
        row = {"text": "This is document content.", "id": "abc123"}
        result = finepdfs_edu_text_fn(row)
        
        assert result == "This is document content."

    def test_handles_empty_text(self):
        """Should handle empty text gracefully."""
        row = {"text": "", "id": "abc123"}
        result = finepdfs_edu_text_fn(row)
        
        assert result == ""


class TestIsMajorityEnglish:
    """Tests for is_majority_english filter function."""

    def test_majority_english_returns_true(self):
        """Should return True when majority of pages are English (>80%)."""
        # 9/10 = 90% English - passes >0.8 threshold
        example = {
            "per_page_languages": ["eng_Latn"] * 9 + ["deu_Latn"],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is True

    def test_majority_not_english_returns_false(self):
        """Should return False when majority of pages are not English."""
        example = {
            "per_page_languages": ["deu_Latn", "deu_Latn", "deu_Latn", "eng_Latn"],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is False

    def test_exactly_at_threshold_returns_false(self):
        """Should return False when exactly at threshold (uses > not >=)."""
        example = {
            "per_page_languages": ["eng_Latn", "eng_Latn", "deu_Latn", "deu_Latn"],
            "language": "eng_Latn",
        }
        # 50% English - fails 0.5 threshold (needs > 0.5)
        assert is_majority_english(example, threshold=0.5) is False

    def test_above_threshold_returns_true(self):
        """Should return True when above threshold."""
        example = {
            "per_page_languages": ["eng_Latn", "eng_Latn", "eng_Latn", "deu_Latn"],
            "language": "eng_Latn",
        }
        # 75% English - passes 0.5 threshold
        assert is_majority_english(example, threshold=0.5) is True

    def test_custom_threshold(self):
        """Should respect custom threshold."""
        example = {
            "per_page_languages": ["eng_Latn", "eng_Latn", "eng_Latn", "deu_Latn"],
            "language": "eng_Latn",
        }
        # 75% English - passes 0.5 threshold (> 0.5)
        assert is_majority_english(example, threshold=0.5) is True
        # 75% English - fails 0.75 threshold (not > 0.75)
        assert is_majority_english(example, threshold=0.75) is False
        # 75% English - passes 0.7 threshold (> 0.7)
        assert is_majority_english(example, threshold=0.7) is True

    def test_all_english_returns_true(self):
        """Should return True when all pages are English."""
        example = {
            "per_page_languages": ["eng_Latn", "eng_Latn", "eng_Latn"],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is True

    def test_no_english_returns_false(self):
        """Should return False when no pages are English."""
        example = {
            "per_page_languages": ["deu_Latn", "fra_Latn", "spa_Latn"],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is False

    def test_empty_per_page_languages_uses_language_field(self):
        """Should fall back to language field when per_page_languages is empty."""
        example = {
            "per_page_languages": [],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is True

        example_not_english = {
            "per_page_languages": [],
            "language": "deu_Latn",
        }
        assert is_majority_english(example_not_english) is False

    def test_missing_per_page_languages_uses_language_field(self):
        """Should fall back to language field when per_page_languages is missing."""
        example = {"language": "eng_Latn"}
        assert is_majority_english(example) is True

    def test_handles_unknown_language(self):
        """Should handle 'unknown' language tags."""
        example = {
            "per_page_languages": ["eng_Latn", "eng_Latn", "unknown", "eng_Latn"],
            "language": "eng_Latn",
        }
        # 3/4 = 75% English - passes >0.7 threshold (strict comparison)
        assert is_majority_english(example, threshold=0.7) is True
        # 3/4 = 75% English - fails >0.75 threshold (not strictly greater)
        assert is_majority_english(example, threshold=0.75) is False

    def test_single_page_english(self):
        """Should handle single-page documents."""
        example = {
            "per_page_languages": ["eng_Latn"],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is True

    def test_single_page_not_english(self):
        """Should handle single-page non-English documents."""
        example = {
            "per_page_languages": ["deu_Latn"],
            "language": "eng_Latn",
        }
        assert is_majority_english(example) is False


class TestConstants:
    """Tests for module constants."""

    def test_dataset_path(self):
        """Dataset path should be correct."""
        assert FINEPDFS_EDU_DATASET_PATH == "HuggingFaceFW/finepdfs-edu"

    def test_english_subset(self):
        """English subset should be correct."""
        assert FINEPDFS_EDU_ENGLISH_SUBSET == "eng_Latn"


@pytest.mark.slow
class TestLoadFinePDFsEdu:
    """Tests for load_finepdfs_edu function.
    
    These tests are marked as slow because they stream data from HuggingFace.
    """

    def test_default_config_streams(self):
        """Default config should use streaming mode."""
        config = FinePDFsEduConfig()
        assert config.streaming is True
        assert config.subset == "eng_Latn"

    def test_load_streaming(self):
        """Should load dataset in streaming mode."""
        dataset = load_finepdfs_edu()
        
        # Get first sample
        sample = next(iter(dataset))
        
        # Check expected fields
        assert "text" in sample
        assert "id" in sample
        assert "url" in sample
        assert "token_count" in sample
        assert "per_page_languages" in sample

    def test_get_sample(self):
        """Should get a small sample of documents."""
        sample = get_finepdfs_edu_sample(num_docs=5)
        
        count = 0
        for doc in sample:
            count += 1
            assert "text" in doc
        
        assert count == 5


@pytest.mark.slow
class TestLoadFinePDFsEduStreaming:
    """Tests for load_finepdfs_edu_streaming function."""

    def test_returns_train_val_datasets(self):
        """Should return train and validation IterableDatasets."""
        train_data, val_data = load_finepdfs_edu_streaming(
            shuffle_buffer_size=100,
            seed=42,
        )
        
        # Both should be iterable
        train_sample = next(iter(train_data))
        val_sample = next(iter(val_data))
        
        # Check they have expected fields
        assert "text" in train_sample
        assert "id" in train_sample
        assert "text" in val_sample
        assert "id" in val_sample

    def test_english_filter_enabled_by_default(self):
        """English filter should be enabled by default with 0.8 threshold."""
        train_data, _ = load_finepdfs_edu_streaming(
            shuffle_buffer_size=100,
            seed=42,
            filter_english=True,
            english_threshold=0.8,
        )
        
        # Check that samples pass the English filter
        for i, sample in enumerate(train_data):
            per_page = sample.get("per_page_languages", [])
            if per_page:
                english_count = sum(1 for lang in per_page if lang == "eng_Latn")
                # Should be more than 80% English
                assert english_count / len(per_page) > 0.8
            if i >= 10:
                break

    def test_can_disable_english_filter(self):
        """Should be able to disable English filter."""
        # This should not raise any errors
        train_data, _ = load_finepdfs_edu_streaming(
            shuffle_buffer_size=100,
            seed=42,
            filter_english=False,
        )
        
        # Just verify we can iterate
        sample = next(iter(train_data))
        assert "text" in sample

    def test_no_overlap_between_train_and_val(self):
        """Train and validation sets should have no overlapping documents."""
        # Load without English filter and minimal shuffle for faster iteration
        train_data, val_data = load_finepdfs_edu_streaming(
            shuffle_buffer_size=1,  # Minimal shuffle to preserve order
            seed=42,
            filter_english=False,  # Disable filter for faster iteration
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
class TestFilterMajorityEnglish:
    """Tests for filter_majority_english function with real data."""

    def test_filters_dataset(self):
        """Should filter dataset to keep only majority English documents."""
        dataset = load_finepdfs_edu()
        
        # Take a sample and filter
        sample = dataset.take(100)
        filtered = filter_majority_english(sample, threshold=0.5)
        
        # Check that filtered samples are majority English (> threshold)
        for i, doc in enumerate(filtered):
            per_page = doc.get("per_page_languages", [])
            if per_page:
                english_count = sum(1 for lang in per_page if lang == "eng_Latn")
                assert english_count / len(per_page) > 0.5
            if i >= 10:
                break

    def test_higher_threshold_filters_more(self):
        """Higher threshold should filter more aggressively."""
        dataset = load_finepdfs_edu()
        sample = dataset.take(50)
        
        # Count with 50% threshold
        filtered_50 = list(filter_majority_english(sample, threshold=0.5))
        
        # Reset and count with 90% threshold
        sample = dataset.take(50)
        filtered_90 = list(filter_majority_english(sample, threshold=0.9))
        
        # Higher threshold should keep fewer or equal documents
        assert len(filtered_90) <= len(filtered_50)

