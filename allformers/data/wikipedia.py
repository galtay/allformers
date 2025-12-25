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

from dataclasses import dataclass, field
from typing import Optional, Union, Iterator

from datasets import load_dataset, Dataset, IterableDataset


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
    """

    subset: str = WIKIPEDIA_ENGLISH_SUBSET
    split: str = "train"
    streaming: bool = False
    num_samples: Optional[int] = None
    cache_dir: Optional[str] = None


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


def load_wikipedia_train_val(
    config: Optional[WikipediaConfig] = None,
    val_fraction: float = 0.05,
) -> tuple[Dataset, Dataset]:
    """Load Wikipedia dataset split into training and validation sets.

    This function automatically handles the train/val split by reserving
    a portion of the dataset for validation. Since the Wikipedia dataset
    doesn't have a built-in validation split, we reserve the last portion
    of the dataset for validation.

    Args:
        config: Configuration for loading the dataset. If None, uses defaults.
            Note: The split in config will be ignored - this function handles
            the split internally.
        val_fraction: Fraction of dataset to reserve for validation (default: 0.05 = 5%).

    Returns:
        Tuple of (train_dataset, val_dataset) HuggingFace datasets.

    Example:
        >>> train_data, val_data = load_wikipedia_train_val()
        >>> print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    """
    if config is None:
        config = WikipediaConfig()

    # Load full dataset to determine size
    full_config = WikipediaConfig(
        subset=config.subset,
        split="train",
        streaming=False,  # Need to know length, so can't use streaming
        cache_dir=config.cache_dir,
    )
    full_dataset = load_wikipedia(full_config)
    full_size = len(full_dataset)

    # Calculate train/val split
    val_size = max(1, int(full_size * val_fraction))
    train_size = full_size - val_size

    # Load training split (first portion)
    # Note: We use streaming=False for splits since we need to slice the dataset
    # The actual streaming happens in StreamingTextDataset during training
    train_config = WikipediaConfig(
        subset=config.subset,
        split=f"train[:{train_size}]",
        streaming=False,
        cache_dir=config.cache_dir,
    )
    train_dataset = load_wikipedia(train_config)

    # Load validation split (last portion)
    val_config = WikipediaConfig(
        subset=config.subset,
        split=f"train[-{val_size}:]",
        streaming=False,
        cache_dir=config.cache_dir,
    )
    val_dataset = load_wikipedia(val_config)
    
    print(f"Loaded Wikipedia dataset:")
    print(f"  Training articles: {len(train_dataset):,} ({train_size/full_size*100:.1f}%)")
    print(f"  Validation articles: {len(val_dataset):,} ({val_size/full_size*100:.1f}%)")

    return train_dataset, val_dataset

