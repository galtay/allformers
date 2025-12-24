"""
Tests for allformers.data.wikipedia module.
"""

import pytest

from allformers.data import (
    load_wikipedia,
    WikipediaConfig,
    WIKIPEDIA_DATASET_PATH,
    WIKIPEDIA_ENGLISH_SUBSET,
)


class TestWikipediaConfig:
    """Tests for WikipediaConfig dataclass."""

    def test_default_values(self):
        """WikipediaConfig should have sensible defaults."""
        config = WikipediaConfig()
        assert config.subset == WIKIPEDIA_ENGLISH_SUBSET
        assert config.split == "train"
        assert config.streaming is False
        assert config.num_samples is None
        assert config.cache_dir is None

    def test_custom_values(self):
        """WikipediaConfig should accept custom values."""
        config = WikipediaConfig(
            subset="20231101.de",
            split="train[:100]",
            streaming=True,
            num_samples=50,
            cache_dir="/tmp/cache",
        )
        assert config.subset == "20231101.de"
        assert config.split == "train[:100]"
        assert config.streaming is True
        assert config.num_samples == 50
        assert config.cache_dir == "/tmp/cache"


class TestWikipediaConstants:
    """Tests for Wikipedia module constants."""

    def test_dataset_path(self):
        """Dataset path should be the wikimedia/wikipedia dataset."""
        assert WIKIPEDIA_DATASET_PATH == "wikimedia/wikipedia"

    def test_english_subset(self):
        """English subset should be 20231101.en."""
        assert WIKIPEDIA_ENGLISH_SUBSET == "20231101.en"


@pytest.mark.slow
class TestLoadWikipedia:
    """Tests for load_wikipedia function.
    
    These tests are marked as slow because they download data from HuggingFace.
    Run with: pytest -m slow
    """

    def test_load_small_sample(self):
        """Should be able to load a small sample of Wikipedia."""
        config = WikipediaConfig(split="train[:5]")
        dataset = load_wikipedia(config)
        
        assert len(dataset) == 5

    def test_dataset_has_required_fields(self):
        """Each article should have id, url, title, and text fields."""
        config = WikipediaConfig(split="train[:1]")
        dataset = load_wikipedia(config)
        
        article = dataset[0]
        assert "id" in article
        assert "url" in article
        assert "title" in article
        assert "text" in article

    def test_article_fields_are_strings(self):
        """Article fields should be strings."""
        config = WikipediaConfig(split="train[:1]")
        dataset = load_wikipedia(config)
        
        article = dataset[0]
        assert isinstance(article["id"], str)
        assert isinstance(article["url"], str)
        assert isinstance(article["title"], str)
        assert isinstance(article["text"], str)

    def test_url_is_wikipedia_url(self):
        """Article URL should be a Wikipedia URL."""
        config = WikipediaConfig(split="train[:1]")
        dataset = load_wikipedia(config)
        
        article = dataset[0]
        assert "wikipedia.org" in article["url"]

    def test_num_samples_limits_dataset(self):
        """num_samples should limit the number of articles."""
        config = WikipediaConfig(split="train[:100]", num_samples=10)
        dataset = load_wikipedia(config)
        
        assert len(dataset) == 10

    def test_default_config_loads_data(self):
        """Loading with default config should work (loads full train split)."""
        # Just load a small slice to test
        config = WikipediaConfig(split="train[:3]")
        dataset = load_wikipedia(config)
        
        assert len(dataset) == 3
        assert dataset[0]["title"]  # Should have a title

