"""
Streaming dataset utilities for on-the-fly tokenization.

This module provides streaming datasets that tokenize text on-the-fly,
avoiding the need to load entire tokenized datasets into memory.

Uses HuggingFace IterableDataset with shuffle buffers for randomization.
See: https://huggingface.co/docs/datasets/main/stream#shuffle
"""

from typing import Callable

import torch
from torch.utils.data import IterableDataset


class StreamingTextDataset(IterableDataset):
    """
    Iterable dataset that tokenizes on-the-fly and yields fixed-length sequences.
    
    Works with HuggingFace IterableDataset (streaming mode). The dataset should
    be pre-shuffled using `.shuffle(buffer_size=N, seed=seed)` before being
    passed to this class.
    
    Benefits:
    - Works with datasets of any size (including 700GB+ datasets)
    - Lower memory usage (no need to store all tokens)
    - Faster startup (no upfront tokenization or download)
    - Uses HuggingFace's built-in shuffle buffer for randomization
    
    Note: Use num_workers=0 with DataLoader when using HuggingFace datasets
    to avoid pickling issues.
    
    Args:
        dataset: HuggingFace IterableDataset (streaming mode). Should be
            pre-shuffled with .shuffle(buffer_size=N, seed=seed).
        tokenizer: Tokenizer to use for encoding text. Should have an
            eos_token_id attribute.
        seq_len: Length of sequences to yield. Each yielded tensor will have
            shape (seq_len + 1,) to provide both input and target.
        text_fn: Function to extract text from a dataset row. Should take a
            dict and return a string. Each dataset module provides its own:
            - Wikipedia: use wikipedia_text_fn from allformers.data.wikipedia
            - FinePDFs-Edu: use finepdfs_edu_text_fn from allformers.data.finepdfs_edu
    
    Yields:
        torch.Tensor of shape (seq_len + 1,) containing token IDs.
        Use tensor[:-1] for input and tensor[1:] for targets.
    
    Example:
        >>> from allformers.data.wikipedia import load_wikipedia_streaming, wikipedia_text_fn
        >>> from transformers import AutoTokenizer
        >>> 
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> train_data, val_data = load_wikipedia_streaming(seed=42)
        >>> 
        >>> train_dataset = StreamingTextDataset(
        ...     dataset=train_data,
        ...     tokenizer=tokenizer,
        ...     seq_len=512,
        ...     text_fn=wikipedia_text_fn,
        ... )
        >>> 
        >>> loader = DataLoader(train_dataset, batch_size=4, num_workers=0)
        >>> batch = next(iter(loader))
        >>> input_ids, targets = batch[:, :-1], batch[:, 1:]
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len: int = 512,
        text_fn: Callable[[dict], str] = None,
    ):
        if text_fn is None:
            raise ValueError(
                "text_fn is required. Use the text function from your dataset module:\n"
                "  - Wikipedia: wikipedia_text_fn from allformers.data.wikipedia\n"
                "  - FinePDFs-Edu: finepdfs_edu_text_fn from allformers.data.finepdfs_edu"
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_fn = text_fn
        
        # Check if tokenizer adds EOS automatically
        test_tokens = tokenizer.encode("test")
        self.tokenizer_adds_eos = (
            tokenizer.eos_token_id is not None
            and len(test_tokens) > 0
            and test_tokens[-1] == tokenizer.eos_token_id
        )
        self.eos_token_id = tokenizer.eos_token_id
    
    def __iter__(self):
        """Iterate over the dataset, yielding fixed-length token sequences."""
        token_buffer = []
        
        # Iterate through the (pre-shuffled) streaming dataset
        for row in self.dataset:
            text = self.text_fn(row)
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)
            
            # Add EOS if tokenizer doesn't do it automatically
            if not self.tokenizer_adds_eos and self.eos_token_id is not None:
                token_buffer.append(self.eos_token_id)
            
            # Yield sequences when we have enough tokens
            while len(token_buffer) >= self.seq_len + 1:
                sequence = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1:]
                yield torch.tensor(sequence, dtype=torch.long)
