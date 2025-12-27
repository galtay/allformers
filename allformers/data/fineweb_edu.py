"""
FineWeb-Edu dataset loading utilities.

This module provides utilities for loading the FineWeb-Edu dataset from HuggingFace,
which contains highly educational web content filtered from FineWeb using an
educational quality classifier.

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

The dataset was created by:
1. Filtering FineWeb (web content from CommonCrawl) using an educational quality classifier
2. Training a classifier to predict educational value scores
3. Filtering to keep only high-quality educational content

Available subsets:
    - default: Full dataset (~1.53B rows, largest)
    - sample-10BT: 10 billion token sample (~9.67M rows)
    - sample-100BT: 100 billion token sample (~97.3M rows)  
    - sample-350BT: 350 billion token sample (~339M rows)
    - CC-MAIN-*: Individual CommonCrawl crawl subsets (e.g., CC-MAIN-2024-10)

Key fields:
    - text: The document text
    - id: Unique document identifier
    - url: Source URL
    - score: Educational quality score (higher = more educational)

Example usage:
    from allformers.data.fineweb_edu import load_fineweb_edu_streaming, fineweb_edu_text_fn

    # Stream the dataset
    train_data, val_data = load_fineweb_edu_streaming()
    
    for doc in train_data:
        print(doc["text"][:200])
        break
"""

import hashlib
from dataclasses import dataclass
from typing import Optional, Iterator

from datasets import load_dataset, Dataset, IterableDataset


def _deterministic_hash(value: str) -> int:
    """Compute a deterministic hash of a string.
    
    Uses MD5 to ensure consistent results across Python processes
    (unlike the built-in hash() which is randomized).
    """
    return int(hashlib.md5(value.encode()).hexdigest(), 16)


# The FineWeb-Edu dataset on HuggingFace
FINEWEB_EDU_DATASET_PATH = "HuggingFaceFW/fineweb-edu"

# Default subset - sample-10BT is a good starting point (smaller than full dataset)
FINEWEB_EDU_DEFAULT_SUBSET = "sample-10BT"


@dataclass
class FineWebEduConfig:
    """Configuration for loading the FineWeb-Edu dataset.

    Attributes:
        subset: The subset to load.
            Options include:
            - "default": Full dataset (~1.53B rows)
            - "sample-10BT": 10B token sample (~9.67M rows) - good for testing
            - "sample-100BT": 100B token sample (~97.3M rows)
            - "sample-350BT": 350B token sample (~339M rows)
            - "CC-MAIN-*": Individual crawl subsets (e.g., "CC-MAIN-2024-10")
            Default is "sample-10BT".
        split: The dataset split to load. Can be "train" or a slice like "train[:1000]".
            Default is "train".
        streaming: Whether to stream the dataset instead of downloading it all.
            Recommended for larger subsets.
            Default is True.
        num_samples: If provided, limits the dataset to this many samples.
            Only works when streaming=False. For streaming, use .take() on the result.
        cache_dir: Directory to cache the downloaded dataset.
            If None, uses the default HuggingFace cache directory.
        token: HuggingFace token for authentication. If None, uses the HF_TOKEN
            environment variable or cached token.
    """

    subset: str = FINEWEB_EDU_DEFAULT_SUBSET
    split: str = "train"
    streaming: bool = True
    num_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None


def fineweb_edu_text_fn(row: dict) -> str:
    """Text extraction function for FineWeb-Edu documents.
    
    This function is designed to be used with StreamingTextDataset
    to extract text from FineWeb-Edu dataset rows.
    
    Args:
        row: A row from the FineWeb-Edu dataset with a 'text' field.
        
    Returns:
        The document text.
    """
    return row["text"]


def load_fineweb_edu(
    config: Optional[FineWebEduConfig] = None,
) -> Dataset | IterableDataset:
    """Load the FineWeb-Edu dataset.

    This function loads the FineWeb-Edu dataset from HuggingFace.
    By default, it streams the dataset to avoid downloading the entire dataset
    at once.

    Args:
        config: Configuration for loading the dataset.
            If None, uses default configuration (sample-10BT, streaming).

    Returns:
        The loaded dataset, either as a Dataset or IterableDataset depending
        on the streaming setting.

    Example:
        # Load with default settings (streaming)
        dataset = load_fineweb_edu()
        for doc in dataset:
            print(doc["text"][:100])
            break

        # Load a small subset for testing
        config = FineWebEduConfig(streaming=False, num_samples=100)
        dataset = load_fineweb_edu(config)
    """
    if config is None:
        config = FineWebEduConfig()

    dataset = load_dataset(
        FINEWEB_EDU_DATASET_PATH,
        name=config.subset,
        split=config.split,
        streaming=config.streaming,
        cache_dir=config.cache_dir,
        token=config.token,
    )

    # Apply sample limit if specified (only works for non-streaming)
    if config.num_samples is not None and not config.streaming:
        dataset = dataset.select(range(min(config.num_samples, len(dataset))))

    return dataset


def load_fineweb_edu_streaming(
    subset: str = FINEWEB_EDU_DEFAULT_SUBSET,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    min_score: Optional[float] = None,
    filter_language: Optional[str] = "en",
    min_language_score: Optional[float] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[IterableDataset, IterableDataset]:
    """Load FineWeb-Edu as streaming train/val splits with shuffling.
    
    Uses HuggingFace's IterableDataset.shuffle() for memory-efficient
    shuffling via a buffer. See: https://huggingface.co/docs/datasets/main/stream#shuffle
    
    Train/val split is deterministic based on document ID hash (95%/5% split).
    
    Args:
        subset: Dataset subset to load (default: sample-10BT).
        shuffle_buffer_size: Size of the shuffle buffer. Larger = better randomization
            but more memory. Default is 10,000.
        seed: Random seed for shuffling reproducibility.
        min_score: If provided, filter to keep only documents with score >= min_score.
            Higher scores indicate more educational content.
        filter_language: If provided, filter to keep only documents with this language.
            Uses the 'language' column (e.g., "en" for English). Set to None to disable.
            Default is "en".
        min_language_score: If provided, filter to keep only documents with 
            language_score >= this value. The language_score indicates confidence
            in the language detection (0.0-1.0). Default is None (no filtering).
        cache_dir: Directory to cache downloaded data.
        token: HuggingFace token for authentication.
        
    Returns:
        Tuple of (train_dataset, val_dataset) as shuffled IterableDatasets.
        
    Example:
        train_data, val_data = load_fineweb_edu_streaming(
            subset="sample-10BT",
            shuffle_buffer_size=10_000,
            seed=42,
            filter_language="en",  # Only English documents
        )
        
        for doc in train_data:
            print(doc["text"][:100])
            break
    """
    dataset = load_dataset(
        FINEWEB_EDU_DATASET_PATH,
        name=subset,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        token=token,
    )
    
    # Create filter functions that combine all filters with train/val split
    # We use a single filter to avoid the HuggingFace datasets bug with chained filters
    def passes_filters(example) -> bool:
        """Check if example passes all configured filters."""
        # Check language filter if applicable
        if filter_language is not None:
            if example.get("language") != filter_language:
                return False
        
        # Check language score filter if applicable
        if min_language_score is not None:
            lang_score = example.get("language_score", 0)
            if lang_score < min_language_score:
                return False
        
        # Check educational score filter if applicable
        if min_score is not None:
            score = example.get("score", 0)
            if score < min_score:
                return False
        
        return True
    
    def train_filter(example):
        if not passes_filters(example):
            return False
        # 95% train split based on deterministic hash
        return _deterministic_hash(example["id"]) % 20 != 0
    
    def val_filter(example):
        if not passes_filters(example):
            return False
        # 5% val split based on deterministic hash
        return _deterministic_hash(example["id"]) % 20 == 0
    
    # Apply filters and shuffle
    train_dataset = dataset.filter(train_filter).shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed,
    )
    val_dataset = dataset.filter(val_filter).shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed + 1,  # Different seed for validation
    )
    
    print(f"Loaded FineWeb-Edu dataset (streaming mode):")
    print(f"  Subset: {subset}")
    if filter_language is not None:
        print(f"  Language filter: {filter_language}")
    if min_language_score is not None:
        print(f"  Language score filter: >= {min_language_score}")
    if min_score is not None:
        print(f"  Educational score filter: >= {min_score}")
    print(f"  Training: ~95% of documents (shuffle buffer: {shuffle_buffer_size:,})")
    print(f"  Validation: ~5% of documents")
    
    return train_dataset, val_dataset


def iterate_documents(
    config: Optional[FineWebEduConfig] = None,
    max_docs: Optional[int] = None,
) -> Iterator[dict]:
    """Iterate over documents in the FineWeb-Edu dataset.

    A convenience function for iterating over documents one at a time.
    Uses streaming by default to minimize memory usage.

    Args:
        config: Configuration for loading the dataset.
        max_docs: Maximum number of documents to yield.
            If None, yields all documents.

    Yields:
        Document dictionaries with 'text', 'id', 'url', 'score' fields.

    Example:
        for i, doc in enumerate(iterate_documents(max_docs=10)):
            print(f"Doc {i}: {doc['text'][:50]}...")
    """
    if config is None:
        config = FineWebEduConfig(streaming=True)

    dataset = load_fineweb_edu(config)

    for i, doc in enumerate(dataset):
        if max_docs is not None and i >= max_docs:
            break
        yield doc


def sample_documents(
    n: int = 5,
    config: Optional[FineWebEduConfig] = None,
) -> list[dict]:
    """Sample n documents from the FineWeb-Edu dataset.

    A convenience function for quickly getting sample documents.

    Args:
        n: Number of documents to sample.
        config: Configuration for loading the dataset.

    Returns:
        List of document dictionaries.

    Example:
        docs = sample_documents(3)
        for doc in docs:
            print(doc["text"][:100])
    """
    return list(iterate_documents(config=config, max_docs=n))


def get_dataset_info(config: Optional[FineWebEduConfig] = None) -> dict:
    """Get information about the FineWeb-Edu dataset.

    Returns metadata about the dataset including available fields
    and sample statistics.

    Args:
        config: Configuration for loading the dataset.

    Returns:
        Dictionary with dataset information.
    """
    if config is None:
        config = FineWebEduConfig(streaming=True)

    dataset = load_fineweb_edu(config)

    # Get a sample to inspect fields
    sample = next(iter(dataset))

    info = {
        "dataset_path": FINEWEB_EDU_DATASET_PATH,
        "subset": config.subset,
        "streaming": config.streaming,
        "fields": list(sample.keys()),
        "sample_text_length": len(sample.get("text", "")),
    }

    return info

