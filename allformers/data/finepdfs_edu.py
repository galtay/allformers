"""
FinePDFs-Edu dataset loading utilities.

This module provides utilities for loading the FinePDFs-Edu dataset from HuggingFace,
which contains 350B+ tokens of highly educational content extracted from PDFs.

Dataset: https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu
Subset: eng_Latn (English, ~23M rows)

The dataset was created by:
1. Processing PDFs from CommonCrawl using OCR
2. Scoring documents for educational quality using Qwen3-235B-A22B-Instruct-2507
3. Filtering to keep only the top 10% most educational content

Key fields:
    - text: The extracted document text
    - id: Unique document identifier
    - url: Source URL of the PDF
    - token_count: Number of tokens in the document
    - fw_edu_scores: Education quality scores [top_score, bottom_score]

IMPORTANT: This dataset is very large (~736 GB) so streaming mode is strongly
recommended unless you have substantial storage available.

Example usage:
    from allformers.data.finepdfs_edu import load_finepdfs_edu, FinePDFsEduConfig

    # Stream the dataset (recommended for large dataset)
    config = FinePDFsEduConfig(streaming=True)
    dataset = load_finepdfs_edu(config)
    
    for doc in dataset:
        print(doc["text"][:200])
        break
"""

import hashlib
from dataclasses import dataclass
from typing import Optional, Union, Iterator

from datasets import load_dataset, Dataset, IterableDataset


def _deterministic_hash(value: str) -> int:
    """Compute a deterministic hash of a string.
    
    Uses MD5 to ensure consistent results across Python processes
    (unlike the built-in hash() which is randomized).
    """
    return int(hashlib.md5(value.encode()).hexdigest(), 16)


# The FinePDFs-Edu dataset on HuggingFace
FINEPDFS_EDU_DATASET_PATH = "HuggingFaceFW/finepdfs-edu"

# We use the English subset (23M rows, largest language subset)
FINEPDFS_EDU_ENGLISH_SUBSET = "eng_Latn"


@dataclass
class FinePDFsEduConfig:
    """Configuration for loading the FinePDFs-Edu dataset.

    Attributes:
        subset: The language subset to load.
            Default is "eng_Latn" for English.
            Other options include: deu_Latn (German), jpn_Jpan (Japanese), etc.
        split: The dataset split to load. Can be "train" or a slice like "train[:1000]".
            Default is "train".
        streaming: Whether to stream the dataset instead of downloading it all.
            STRONGLY RECOMMENDED due to the dataset size (~736 GB total).
            Default is True.
        num_samples: If provided, limits the dataset to this many samples.
            Only works when streaming=False. For streaming, use .take() on the result.
        cache_dir: Directory to cache the downloaded dataset.
            If None, uses the default HuggingFace cache directory.
        token: HuggingFace token for authentication. If None, uses the HF_TOKEN
            environment variable or cached token. Set to increase rate limits
            and download speeds.
    """

    subset: str = FINEPDFS_EDU_ENGLISH_SUBSET
    split: str = "train"
    streaming: bool = True  # Default to streaming due to large size
    num_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None


def finepdfs_edu_text_fn(row: dict) -> str:
    """Text extraction function for FinePDFs-Edu documents.
    
    This function is designed to be used with StreamingTextDataset
    to extract text from FinePDFs-Edu dataset rows.
    
    Args:
        row: A row from the FinePDFs-Edu dataset with a 'text' field.
        
    Returns:
        The document text.
        
    Example:
        >>> streaming_ds = StreamingTextDataset(
        ...     dataset=finepdfs_edu_data,
        ...     tokenizer=tokenizer,
        ...     text_fn=finepdfs_edu_text_fn,
        ... )
    """
    return row["text"]


def is_majority_english(example: dict, threshold: float = 0.8) -> bool:
    """Check if a document is majority English based on per-page language detection.
    
    The FinePDFs dataset contains documents with code-switching (multiple languages).
    This filter keeps only documents where the majority of pages are detected as English.
    
    See: https://huggingface.co/datasets/HuggingFaceFW/finepdfs#code-switching
    
    Args:
        example: A row from the FinePDFs-Edu dataset with 'per_page_languages' field.
        threshold: Fraction of pages that must be English (default: 0.8 = 80%).
            Documents with English fraction > threshold are kept.
    
    Returns:
        True if the document is majority English, False otherwise.
        
    Example:
        >>> # Filter dataset to keep only majority English documents
        >>> dataset = dataset.filter(is_majority_english)
        
        >>> # Use a lower threshold (50% English)
        >>> dataset = dataset.filter(lambda x: is_majority_english(x, threshold=0.5))
    """
    per_page_languages = example.get("per_page_languages", [])
    
    if not per_page_languages:
        # If no language info, check the overall language field
        return example.get("language", "") == "eng_Latn"
    
    # Count English pages
    english_count = sum(1 for lang in per_page_languages if lang == "eng_Latn")
    total_pages = len(per_page_languages)
    
    if total_pages == 0:
        return False
    
    return (english_count / total_pages) > threshold


def filter_majority_english(dataset: IterableDataset, threshold: float = 0.8) -> IterableDataset:
    """Filter a FinePDFs-Edu dataset to keep only majority English documents.
    
    This is useful for removing documents with significant code-switching where
    large portions are in other languages.
    
    Args:
        dataset: A FinePDFs-Edu IterableDataset to filter.
        threshold: Minimum fraction of pages that must be English (default: 0.5 = 50%).
    
    Returns:
        Filtered IterableDataset containing only majority English documents.
        
    Example:
        >>> dataset = load_finepdfs_edu()
        >>> english_dataset = filter_majority_english(dataset, threshold=0.75)
    """
    return dataset.filter(lambda x: is_majority_english(x, threshold=threshold))


def load_finepdfs_edu(
    config: Optional[FinePDFsEduConfig] = None,
) -> Union[Dataset, IterableDataset]:
    """Load the FinePDFs-Edu dataset from HuggingFace.

    This function loads the English FinePDFs-Edu dataset (or another language subset
    if specified in the config). By default, it uses streaming mode since the
    dataset is very large (~736 GB).

    Args:
        config: Configuration for loading the dataset. If None, uses defaults
            (streaming mode enabled).

    Returns:
        A HuggingFace Dataset or IterableDataset containing PDF documents.
        Each example has the following key fields:
        - text: The extracted document text
        - id: Unique document identifier  
        - url: Source URL of the PDF
        - token_count: Number of tokens in the document
        - fw_edu_scores: Education quality scores

    Examples:
        # Stream the dataset (recommended)
        >>> dataset = load_finepdfs_edu()
        >>> for doc in dataset:
        ...     print(doc["text"][:100])
        ...     break

        # Load a specific number of samples (non-streaming)
        >>> config = FinePDFsEduConfig(streaming=False, num_samples=1000)
        >>> dataset = load_finepdfs_edu(config)
    """
    if config is None:
        config = FinePDFsEduConfig()

    # Load the dataset with the specific subset name
    dataset = load_dataset(
        FINEPDFS_EDU_DATASET_PATH,
        name=config.subset,
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


def load_finepdfs_edu_streaming(
    subset: str = FINEPDFS_EDU_ENGLISH_SUBSET,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    filter_english: bool = True,
    english_threshold: float = 0.8,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[IterableDataset, IterableDataset]:
    """Load FinePDFs-Edu dataset as streaming IterableDatasets with shuffle buffers.

    This is the recommended way to load FinePDFs-Edu for training. Uses HuggingFace's
    streaming mode with shuffle buffers for memory-efficient random sampling.
    
    The dataset is ~736 GB, so streaming is essential.
    For train/val split, we use filtering: train gets 95%, val gets 5%.
    
    See: https://huggingface.co/docs/datasets/main/stream#shuffle

    Args:
        subset: The language subset to load.
            Default is "eng_Latn" for English (~23M documents).
        shuffle_buffer_size: Size of the shuffle buffer for randomization.
            Larger buffers give better randomization but use more memory.
            Default is 10,000.
        seed: Random seed for shuffling reproducibility.
        filter_english: Whether to filter for majority English documents.
            This removes documents with significant code-switching.
            Default is True.
        english_threshold: Fraction of pages that must be English
            when filter_english is True. Default is 0.8 (80%).
            Documents with English fraction > threshold are kept.
        cache_dir: Directory to cache the downloaded dataset.
        token: HuggingFace token for authentication. If None, uses the HF_TOKEN
            environment variable or cached token.

    Returns:
        Tuple of (train_dataset, val_dataset) as shuffled IterableDatasets.

    Example:
        >>> # Default: filter for majority English
        >>> train_data, val_data = load_finepdfs_edu_streaming(seed=42)
        
        >>> # Stricter English filter (75% of pages must be English)
        >>> train_data, val_data = load_finepdfs_edu_streaming(
        ...     seed=42, english_threshold=0.75
        ... )
        
        >>> # No English filtering
        >>> train_data, val_data = load_finepdfs_edu_streaming(
        ...     seed=42, filter_english=False
        ... )
    """
    # Load dataset in streaming mode
    dataset = load_dataset(
        FINEPDFS_EDU_DATASET_PATH,
        name=subset,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        token=token,
    )
    
    # Create combined filter functions that include both:
    # 1. Optional English language filter
    # 2. Train/val split based on deterministic hash of id
    #
    # We combine these into single filters because HuggingFace datasets
    # has a bug where chaining multiple .filter() calls on streaming
    # datasets causes a TypeError (features become None).
    
    def is_train(example):
        # First check English filter if enabled
        if filter_english and not is_majority_english(example, threshold=english_threshold):
            return False
        # Then check train split (95% of documents)
        return _deterministic_hash(example["id"]) % 20 != 0
    
    def is_val(example):
        # First check English filter if enabled
        if filter_english and not is_majority_english(example, threshold=english_threshold):
            return False
        # Then check val split (5% of documents)
        return _deterministic_hash(example["id"]) % 20 == 0
    
    train_dataset = dataset.filter(is_train)
    val_dataset = dataset.filter(is_val)
    
    # Apply shuffle buffers
    # Training uses the provided seed, validation uses a different seed
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    val_dataset = val_dataset.shuffle(seed=seed + 1, buffer_size=shuffle_buffer_size)
    
    print(f"Loaded FinePDFs-Edu dataset (streaming mode):")
    print(f"  Subset: {subset}")
    if filter_english:
        print(f"  English filter: >{english_threshold*100:.0f}% of pages must be English")
    print(f"  Training: ~95% of documents (shuffle buffer: {shuffle_buffer_size:,})")
    print(f"  Validation: ~5% of documents")

    return train_dataset, val_dataset


def iter_finepdfs_edu_texts(
    config: Optional[FinePDFsEduConfig] = None,
) -> Iterator[str]:
    """Iterate over FinePDFs-Edu document texts.

    This is a convenience function for getting just the text content
    from documents, useful for language model training.

    Args:
        config: Configuration for loading the dataset.

    Yields:
        Document text strings.

    Example:
        >>> config = FinePDFsEduConfig(streaming=True)
        >>> for text in iter_finepdfs_edu_texts(config):
        ...     print(text[:100])
        ...     break
    """
    if config is None:
        config = FinePDFsEduConfig(streaming=True)

    dataset = load_finepdfs_edu(config)

    for doc in dataset:
        yield doc["text"]


def get_finepdfs_edu_sample(
    num_docs: int = 100,
) -> IterableDataset:
    """Get a small sample of FinePDFs-Edu documents.

    This is useful for quick testing and development. Since the dataset
    is very large, we use streaming and .take() to get a sample without
    downloading everything.

    Args:
        num_docs: Number of documents to include in the sample.

    Returns:
        An IterableDataset containing the sampled documents.

    Example:
        >>> sample = get_finepdfs_edu_sample(num_docs=10)
        >>> for doc in sample:
        ...     print(doc["text"][:50])
    """
    config = FinePDFsEduConfig(streaming=True)
    dataset = load_finepdfs_edu(config)
    return dataset.take(num_docs)


def get_finepdfs_edu_stats(num_samples: int = 1000) -> dict:
    """Get basic statistics about the FinePDFs-Edu dataset.

    Samples a portion of the dataset to compute statistics like
    average token count, text length distribution, etc.

    Args:
        num_samples: Number of samples to use for computing statistics.

    Returns:
        Dictionary containing dataset statistics.

    Example:
        >>> stats = get_finepdfs_edu_stats(num_samples=100)
        >>> print(f"Avg tokens: {stats['avg_token_count']:.0f}")
    """
    config = FinePDFsEduConfig(streaming=True)
    dataset = load_finepdfs_edu(config)

    token_counts = []
    text_lengths = []
    edu_scores = []

    for i, doc in enumerate(dataset):
        if i >= num_samples:
            break
        token_counts.append(doc["token_count"])
        text_lengths.append(len(doc["text"]))
        # fw_edu_scores contains [top_score, bottom_score]
        if doc["fw_edu_scores"]:
            edu_scores.append(max(doc["fw_edu_scores"]))

    return {
        "num_samples": len(token_counts),
        "avg_token_count": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_token_count": min(token_counts) if token_counts else 0,
        "max_token_count": max(token_counts) if token_counts else 0,
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "avg_edu_score": sum(edu_scores) / len(edu_scores) if edu_scores else 0,
    }

