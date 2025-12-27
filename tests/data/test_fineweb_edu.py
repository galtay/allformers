"""Tests for the FineWeb-Edu dataset module."""

import pytest
from datasets import IterableDataset

from allformers.data.fineweb_edu import (
    FINEWEB_EDU_DATASET_PATH,
    FINEWEB_EDU_DEFAULT_SUBSET,
    FineWebEduConfig,
    fineweb_edu_text_fn,
    load_fineweb_edu,
    load_fineweb_edu_streaming,
    iterate_documents,
    sample_documents,
    get_dataset_info,
)


class TestFineWebEduTextFn:
    """Tests for the text extraction function."""

    def test_extracts_text_field(self):
        """Should extract text from the 'text' field."""
        row = {"text": "This is educational content.", "id": "123", "score": 3.5}
        result = fineweb_edu_text_fn(row)
        assert result == "This is educational content."

    def test_handles_empty_text(self):
        """Should handle empty text fields."""
        row = {"text": "", "id": "123"}
        result = fineweb_edu_text_fn(row)
        assert result == ""

    def test_handles_long_text(self):
        """Should handle long text without truncation."""
        long_text = "word " * 10000
        row = {"text": long_text, "id": "123"}
        result = fineweb_edu_text_fn(row)
        assert result == long_text


class TestFineWebEduConfig:
    """Tests for the configuration dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = FineWebEduConfig()
        assert config.subset == FINEWEB_EDU_DEFAULT_SUBSET
        assert config.split == "train"
        assert config.streaming is True
        assert config.num_samples is None
        assert config.cache_dir is None
        assert config.token is None

    def test_custom_values(self):
        """Should accept custom values."""
        config = FineWebEduConfig(
            subset="sample-100BT",
            split="train[:1000]",
            streaming=False,
            num_samples=500,
            cache_dir="/tmp/cache",
            token="hf_test_token",
        )
        assert config.subset == "sample-100BT"
        assert config.split == "train[:1000]"
        assert config.streaming is False
        assert config.num_samples == 500
        assert config.cache_dir == "/tmp/cache"
        assert config.token == "hf_test_token"


@pytest.mark.slow
class TestLoadFineWebEdu:
    """Tests for loading the FineWeb-Edu dataset."""

    def test_loads_streaming_dataset(self):
        """Should load a streaming dataset by default."""
        config = FineWebEduConfig(streaming=True)
        dataset = load_fineweb_edu(config)
        assert isinstance(dataset, IterableDataset)

    def test_can_iterate_documents(self):
        """Should be able to iterate over documents."""
        config = FineWebEduConfig(streaming=True)
        dataset = load_fineweb_edu(config)
        
        doc = next(iter(dataset))
        assert "text" in doc
        assert "id" in doc
        assert isinstance(doc["text"], str)
        assert len(doc["text"]) > 0

    def test_documents_have_expected_fields(self):
        """Documents should have expected fields."""
        config = FineWebEduConfig(streaming=True)
        dataset = load_fineweb_edu(config)
        
        doc = next(iter(dataset))
        # Required fields
        assert "text" in doc
        assert "id" in doc
        # Score field for educational quality
        assert "score" in doc


@pytest.mark.slow
class TestLoadFineWebEduStreaming:
    """Tests for the streaming train/val split function."""

    def test_returns_train_and_val_datasets(self):
        """Should return separate train and validation datasets."""
        train_data, val_data = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10,  # Small buffer for speed
            seed=42,
            filter_language=None,  # Disable for speed
        )
        
        assert isinstance(train_data, IterableDataset)
        assert isinstance(val_data, IterableDataset)

    def test_train_and_val_are_different(self):
        """Train and validation should contain different documents."""
        train_data, val_data = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10,  # Small buffer for speed
            seed=42,
            filter_language=None,  # Disable for speed
        )
        
        # Get some documents from each (small sample for speed)
        train_ids = set()
        for i, doc in enumerate(train_data):
            train_ids.add(doc["id"])
            if i >= 19:
                break
        
        val_ids = set()
        for i, doc in enumerate(val_data):
            val_ids.add(doc["id"])
            if i >= 19:
                break
        
        # There should be no overlap
        overlap = train_ids & val_ids
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping documents"

    def test_deterministic_split(self):
        """Split should be deterministic with same seed.
        
        Note: We verify determinism by checking that the same document IDs
        appear in the same order when using the same seed.
        """
        import gc
        
        # Use small buffer for faster test
        train_data1, val1 = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10,  # Small buffer for speed
            seed=42,
            filter_language=None,  # Disable for speed
        )
        
        # Get first few document IDs from first dataset
        ids1 = []
        for i, doc in enumerate(train_data1):
            ids1.append(doc["id"])
            if i >= 4:
                break
        
        # Cleanup first dataset to avoid resource issues
        del train_data1, val1
        gc.collect()
        
        # Load second dataset with same params
        train_data2, val2 = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10,  # Small buffer for speed
            seed=42,
            filter_language=None,  # Disable for speed
        )
        
        # Get first few document IDs from second dataset
        ids2 = []
        for i, doc in enumerate(train_data2):
            ids2.append(doc["id"])
            if i >= 4:
                break
        
        # Cleanup second dataset
        del train_data2, val2
        gc.collect()
        
        assert ids1 == ids2, "Same seed should produce same order"

    def test_can_filter_by_score(self):
        """Should be able to filter by minimum score."""
        train_data, _ = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10,  # Small buffer for speed
            seed=42,
            min_score=3.0,
            filter_language=None,  # Disable for speed
        )
        
        # Check that documents meet the score threshold
        for i, doc in enumerate(train_data):
            assert doc["score"] >= 3.0, f"Document score {doc['score']} < 3.0"
            if i >= 9:
                break
    
    def test_can_filter_by_language(self):
        """Should be able to filter by language."""
        train_data, _ = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10,  # Small buffer for speed
            seed=42,
            filter_language="en",
        )
        
        # Check that documents have the expected language
        for i, doc in enumerate(train_data):
            assert doc.get("language") == "en", f"Document language {doc.get('language')} != 'en'"
            if i >= 4:
                break


@pytest.mark.slow
class TestIterateDocuments:
    """Tests for the document iteration function."""

    def test_iterates_documents(self):
        """Should iterate over documents."""
        docs = list(iterate_documents(max_docs=5))
        assert len(docs) == 5
        for doc in docs:
            assert "text" in doc
            assert "id" in doc

    def test_respects_max_docs(self):
        """Should stop at max_docs."""
        docs = list(iterate_documents(max_docs=3))
        assert len(docs) == 3


@pytest.mark.slow
class TestSampleDocuments:
    """Tests for the document sampling function."""

    def test_samples_documents(self):
        """Should return requested number of samples."""
        docs = sample_documents(n=3)
        assert len(docs) == 3
        for doc in docs:
            assert "text" in doc


@pytest.mark.slow
class TestGetDatasetInfo:
    """Tests for the dataset info function."""

    def test_returns_info_dict(self):
        """Should return a dictionary with dataset info."""
        info = get_dataset_info()
        
        assert isinstance(info, dict)
        assert info["dataset_path"] == FINEWEB_EDU_DATASET_PATH
        assert "fields" in info
        assert "text" in info["fields"]


@pytest.mark.slow
class TestNoTrainValOverlap:
    """Test that train and validation sets don't overlap."""

    def test_no_overlap_in_100_documents(self):
        """Pull 100 docs from each and ensure no overlap."""
        train_data, val_data = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=50,  # Smaller buffer for speed
            seed=42,
            filter_language=None,  # Disable for speed
        )
        
        # Collect 100 document IDs from train
        train_ids = set()
        for i, doc in enumerate(train_data):
            train_ids.add(doc["id"])
            if i >= 99:
                break
        
        # Collect 100 document IDs from val
        val_ids = set()
        for i, doc in enumerate(val_data):
            val_ids.add(doc["id"])
            if i >= 99:
                break
        
        # Check for overlap
        overlap = train_ids & val_ids
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping document IDs between train and val"
        
        print(f"Verified no overlap: {len(train_ids)} train IDs, {len(val_ids)} val IDs")

