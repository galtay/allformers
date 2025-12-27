"""
Wikipedia dataset loading utilities.

This module provides utilities for downloading and preparing the English Wikipedia
dataset from the wikimedia/wikipedia dataset on HuggingFace.

Dataset: https://huggingface.co/datasets/wikimedia/wikipedia
Subset: 20231101.en (English Wikipedia, November 2023 snapshot)

The full wikimedia/wikipedia dataset contains Wikipedia dumps in 300+ languages.
This module specifically targets the English subset to avoid downloading
unnecessary data.

Example usage:
    from allformers.data.wikipedia import load_wikipedia, WikipediaConfig

    # Load a small sample for testing (streaming mode)
    config = WikipediaConfig(num_samples=1000, streaming=True)
    dataset = load_wikipedia(config)

    # Load with specific split
    config = WikipediaConfig(split="train[:10000]")
    dataset = load_wikipedia(config)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Union, Iterator

from datasets import load_dataset, Dataset, IterableDataset


def _deterministic_hash(value: str) -> int:
    """Compute a deterministic hash of a string.
    
    Uses MD5 to ensure consistent results across Python processes
    (unlike the built-in hash() which is randomized).
    """
    return int(hashlib.md5(value.encode()).hexdigest(), 16)


# The wikimedia/wikipedia dataset on HuggingFace
WIKIPEDIA_DATASET_PATH = "wikimedia/wikipedia"

# We specifically want the English Wikipedia snapshot from November 2023
# This is a subset identifier, not the full dataset
WIKIPEDIA_ENGLISH_SUBSET = "20231101.en"


@dataclass
class WikipediaConfig:
    """Configuration for loading the Wikipedia dataset.

    Attributes:
        subset: The Wikipedia language/date subset to load.
            Default is "20231101.en" for English Wikipedia from Nov 2023.
        split: The dataset split to load. Can be "train" or a slice like "train[:1000]".
            Default is "train".
        streaming: Whether to stream the dataset instead of downloading it all.
            Streaming is recommended for large datasets or quick experimentation.
            Default is False.
        num_samples: If provided, limits the dataset to this many samples.
            Only works when streaming=False. For streaming, use split slicing instead.
        cache_dir: Directory to cache the downloaded dataset.
            If None, uses the default HuggingFace cache directory.
        token: HuggingFace token for authentication. If None, uses the HF_TOKEN
            environment variable or cached token. Set to increase rate limits
            and download speeds.
    """

    subset: str = WIKIPEDIA_ENGLISH_SUBSET
    split: str = "train"
    streaming: bool = False
    num_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None


def load_wikipedia(
    config: Optional[WikipediaConfig] = None,
) -> Union[Dataset, IterableDataset]:
    """Load the Wikipedia dataset from HuggingFace.

    This function loads the English Wikipedia dataset (or another language subset
    if specified in the config). By default, it loads the "20231101.en" subset
    which contains ~6.4 million English Wikipedia articles.

    IMPORTANT: The wikimedia/wikipedia dataset contains 300+ language subsets.
    This function uses the `name` parameter to load ONLY the specified subset,
    avoiding downloading the entire multi-language dataset.

    Args:
        config: Configuration for loading the dataset. If None, uses defaults.

    Returns:
        A HuggingFace Dataset or IterableDataset containing Wikipedia articles.
        Each example has the following fields:
        - id: Unique article identifier
        - url: URL to the Wikipedia article
        - title: Article title
        - text: Full article text

    Examples:
        # Load with default settings (full English Wikipedia)
        >>> dataset = load_wikipedia()

        # Load a small sample for testing
        >>> config = WikipediaConfig(split="train[:1000]")
        >>> dataset = load_wikipedia(config)

        # Stream the dataset (doesn't download everything)
        >>> config = WikipediaConfig(streaming=True)
        >>> dataset = load_wikipedia(config)
        >>> for article in dataset:
        ...     print(article["title"])
        ...     break
    """
    if config is None:
        config = WikipediaConfig()

    # Load the dataset with the specific subset name
    # The `name` parameter is crucial - it tells HuggingFace to only
    # download the specified language subset, not the entire dataset
    dataset = load_dataset(
        WIKIPEDIA_DATASET_PATH,
        name=config.subset,  # This ensures we only get the English subset
        split=config.split,
        streaming=config.streaming,
        cache_dir=config.cache_dir,
        token=config.token,
    )

    # If num_samples is specified and we're not streaming, select a subset
    if config.num_samples is not None and not config.streaming:
        if len(dataset) > config.num_samples:
            dataset = dataset.select(range(config.num_samples))

    return dataset


def iter_wikipedia_texts(
    config: Optional[WikipediaConfig] = None,
    include_title: bool = True,
) -> Iterator[str]:
    """Iterate over Wikipedia article texts.

    This is a convenience function for getting just the text content
    from Wikipedia articles, useful for language model training.

    Args:
        config: Configuration for loading the dataset.
        include_title: Whether to prepend the article title to the text.
            If True, yields "Title\n\nText", otherwise just "Text".

    Yields:
        Article text strings.

    Example:
        >>> config = WikipediaConfig(streaming=True)
        >>> for text in iter_wikipedia_texts(config):
        ...     print(text[:100])
        ...     break
    """
    if config is None:
        config = WikipediaConfig(streaming=True)

    dataset = load_wikipedia(config)

    for article in dataset:
        if include_title:
            yield f"{article['title']}\n\n{article['text']}"
        else:
            yield article["text"]


def wikipedia_text_fn(row: dict) -> str:
    """Text extraction function for Wikipedia articles.
    
    This function is designed to be used with StreamingTextDataset
    to extract formatted text from Wikipedia dataset rows.
    
    Args:
        row: A row from the Wikipedia dataset with 'title' and 'text' fields.
        
    Returns:
        Formatted text with title and content.
        
    Example:
        >>> streaming_ds = StreamingTextDataset(
        ...     dataset=wikipedia_data,
        ...     tokenizer=tokenizer,
        ...     text_fn=wikipedia_text_fn,
        ... )
    """
    return f"{row['title']}\n\n{row['text']}"


def get_wikipedia_sample(
    num_articles: int = 100,
    seed: int = 42,
) -> Dataset:
    """Get a small, reproducible sample of Wikipedia articles.

    This is useful for quick testing and development. It downloads
    a deterministic sample of articles that can be used for unit tests
    or quick experiments.

    Args:
        num_articles: Number of articles to include in the sample.
        seed: Random seed for reproducible sampling.

    Returns:
        A Dataset containing the sampled articles.

    Example:
        >>> sample = get_wikipedia_sample(num_articles=10)
        >>> print(len(sample))
        10
    """
    # Use split slicing for efficiency - this avoids downloading
    # more data than necessary
    config = WikipediaConfig(
        split=f"train[:{num_articles}]",
        streaming=False,
    )
    return load_wikipedia(config)


def load_wikipedia_streaming(
    subset: str = WIKIPEDIA_ENGLISH_SUBSET,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[IterableDataset, IterableDataset]:
    """Load Wikipedia dataset as streaming IterableDatasets with shuffle buffers.

    This is the recommended way to load Wikipedia for training. Uses HuggingFace's
    streaming mode with shuffle buffers for memory-efficient random sampling.
    
    For train/val split, we use sharding: train gets shards 1-19, val gets shard 0.
    This gives approximately 95%/5% split.
    
    See: https://huggingface.co/docs/datasets/main/stream#shuffle

    Args:
        subset: The Wikipedia language/date subset to load.
            Default is "20231101.en" for English Wikipedia.
        shuffle_buffer_size: Size of the shuffle buffer for randomization.
            Larger buffers give better randomization but use more memory.
            Default is 10,000.
        seed: Random seed for shuffling reproducibility.
        cache_dir: Directory to cache the downloaded dataset.
        token: HuggingFace token for authentication. If None, uses the HF_TOKEN
            environment variable or cached token.

    Returns:
        Tuple of (train_dataset, val_dataset) as shuffled IterableDatasets.

    Example:
        >>> train_data, val_data = load_wikipedia_streaming(seed=42)
        >>> for article in train_data:
        ...     print(article["title"])
        ...     break
    """
    # Load dataset in streaming mode
    dataset = load_dataset(
        WIKIPEDIA_DATASET_PATH,
        name=subset,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        token=token,
    )
    
    # Use filter to create train/val split based on deterministic hash of id
    # This gives a deterministic ~95%/5% split that's consistent across runs
    def is_train(example):
        return _deterministic_hash(example["id"]) % 20 != 0  # 19/20 = 95% for training
    
    def is_val(example):
        return _deterministic_hash(example["id"]) % 20 == 0  # 1/20 = 5% for validation
    
    train_dataset = dataset.filter(is_train)
    val_dataset = dataset.filter(is_val)
    
    # Apply shuffle buffers
    # Training uses the provided seed, validation uses a different seed
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    val_dataset = val_dataset.shuffle(seed=seed + 1, buffer_size=shuffle_buffer_size)
    
    print(f"Loaded Wikipedia dataset (streaming mode):")
    print(f"  Training: ~95% of articles (shuffle buffer: {shuffle_buffer_size:,})")
    print(f"  Validation: ~5% of articles")

    return train_dataset, val_dataset

