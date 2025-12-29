"""
Tests for allformers.data.wikipedia module.
"""

import pytest
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from allformers.data.wikipedia import (
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


@pytest.mark.slow
class TestWikipediaSharding:
    """Tests for sharding Wikipedia dataset across multiple workers (DDP support)."""

    def test_n_shards_property(self):
        """Wikipedia dataset should have multiple shards for DDP."""
        dataset = load_dataset(
            WIKIPEDIA_DATASET_PATH,
            name=WIKIPEDIA_ENGLISH_SUBSET,
            split="train",
            streaming=True,
        )
        
        n_shards = dataset.n_shards
        assert n_shards == 41, f"Expected 41 shards, got {n_shards}"

    def test_shard_method_splits_data(self):
        """Different ranks should get different documents using .shard()."""
        dataset = load_dataset(
            WIKIPEDIA_DATASET_PATH,
            name=WIKIPEDIA_ENGLISH_SUBSET,
            split="train",
            streaming=True,
        )
        
        world_size = 4
        
        # Collect first 3 document IDs from each rank
        worker_ids = []
        for rank in range(world_size):
            sharded = dataset.shard(num_shards=world_size, index=rank)
            ids = []
            for i, example in enumerate(sharded):
                ids.append(example["id"])
                if i >= 2:
                    break
            worker_ids.append(ids)
        
        # All IDs should be unique across workers
        all_ids = [id for ids in worker_ids for id in ids]
        assert len(all_ids) == len(set(all_ids)), "Workers should get different documents"

    def test_split_dataset_by_node(self):
        """split_dataset_by_node should distribute data across workers."""
        dataset = load_dataset(
            WIKIPEDIA_DATASET_PATH,
            name=WIKIPEDIA_ENGLISH_SUBSET,
            split="train",
            streaming=True,
        )
        
        world_size = 4
        
        # Collect first 3 document IDs from each rank
        worker_ids = []
        for rank in range(world_size):
            sharded = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            ids = []
            for i, example in enumerate(sharded):
                ids.append(example["id"])
                if i >= 2:
                    break
            worker_ids.append(ids)
        
        # All IDs should be unique across workers
        all_ids = [id for ids in worker_ids for id in ids]
        assert len(all_ids) == len(set(all_ids)), "Workers should get different documents"

    def test_sharding_is_deterministic(self):
        """Same rank should get same data on repeated calls."""
        dataset = load_dataset(
            WIKIPEDIA_DATASET_PATH,
            name=WIKIPEDIA_ENGLISH_SUBSET,
            split="train",
            streaming=True,
        )
        
        world_size = 4
        rank = 2
        
        # Get IDs twice for the same rank
        ids1 = []
        sharded1 = dataset.shard(num_shards=world_size, index=rank)
        for i, example in enumerate(sharded1):
            ids1.append(example["id"])
            if i >= 4:
                break
        
        ids2 = []
        sharded2 = dataset.shard(num_shards=world_size, index=rank)
        for i, example in enumerate(sharded2):
            ids2.append(example["id"])
            if i >= 4:
                break
        
        assert ids1 == ids2, "Same rank should get same data"

    def test_sharding_with_shuffle(self):
        """Sharding should maintain separation even after shuffling."""
        dataset = load_dataset(
            WIKIPEDIA_DATASET_PATH,
            name=WIKIPEDIA_ENGLISH_SUBSET,
            split="train",
            streaming=True,
        )
        
        world_size = 2
        seed = 42
        
        worker_ids = []
        for rank in range(world_size):
            # Shard first, then shuffle
            sharded = dataset.shard(num_shards=world_size, index=rank)
            shuffled = sharded.shuffle(seed=seed, buffer_size=100)
            
            ids = set()
            for i, example in enumerate(shuffled):
                ids.add(example["id"])
                if i >= 9:
                    break
            worker_ids.append(ids)
        
        # No overlap between workers
        overlap = worker_ids[0] & worker_ids[1]
        assert len(overlap) == 0, f"Workers should not overlap, found: {overlap}"

