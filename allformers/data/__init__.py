"""
allformers.data - Dataset loading and preparation utilities

This module provides utilities for downloading and preparing datasets
for training and testing transformer models.

Available datasets:
- Wikipedia: English Wikipedia from wikimedia/wikipedia on HuggingFace

Streaming utilities:
- StreamingTextDataset: On-the-fly tokenization for memory-efficient training
"""

from allformers.data.wikipedia import (
    load_wikipedia,
    load_wikipedia_train_val,
    iter_wikipedia_texts,
    get_wikipedia_sample,
    WikipediaConfig,
    WIKIPEDIA_DATASET_PATH,
    WIKIPEDIA_ENGLISH_SUBSET,
)
from allformers.data.streaming import (
    StreamingTextDataset,
    wikipedia_text_fn,
)

__all__ = [
    # Wikipedia
    "load_wikipedia",
    "load_wikipedia_train_val",
    "iter_wikipedia_texts",
    "get_wikipedia_sample",
    "WikipediaConfig",
    "WIKIPEDIA_DATASET_PATH",
    "WIKIPEDIA_ENGLISH_SUBSET",
    # Streaming
    "StreamingTextDataset",
    "wikipedia_text_fn",
]

