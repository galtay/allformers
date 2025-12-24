"""
Streaming dataset utilities for on-the-fly tokenization.

This module provides streaming datasets that tokenize text on-the-fly,
avoiding the need to load entire tokenized datasets into memory.

Inspired by:
https://github.com/huggingface/transformers-research-projects/blob/main/codeparrot/scripts/codeparrot_training.py
"""

import random
from typing import Callable, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info


def wikipedia_text_fn(row: dict) -> str:
    """Default text extraction function for Wikipedia articles.
    
    Args:
        row: A row from the Wikipedia dataset with 'title' and 'text' fields.
        
    Returns:
        Formatted text with title and content.
    """
    return f"{row['title']}\n\n{row['text']}"


class StreamingTextDataset(IterableDataset):
    """
    Iterable dataset that tokenizes on-the-fly and yields fixed-length sequences.
    
    This dataset samples articles randomly from a HuggingFace dataset, tokenizes
    them on the fly, concatenates them with EOS tokens, and yields fixed-length
    sequences suitable for language modeling.
    
    Benefits over pre-tokenization:
    - Lower memory usage (no need to store all tokens)
    - Faster startup (no upfront tokenization)
    - True random sampling across the dataset
    
    Note: Use num_workers=0 with DataLoader when using HuggingFace datasets
    to avoid pickling issues.
    
    Args:
        dataset: HuggingFace dataset with text samples. Must support len() and
            integer indexing (i.e., not a streaming dataset).
        tokenizer: Tokenizer to use for encoding text. Should have an
            eos_token_id attribute.
        seq_len: Length of sequences to yield. Each yielded tensor will have
            shape (seq_len + 1,) to provide both input and target.
        text_fn: Function to extract text from a dataset row. Should take a
            dict and return a string. Defaults to wikipedia_text_fn.
        seed: Random seed for reproducibility. Different DataLoader workers
            will use different seeds based on this.
        random_offset: If True, apply a random offset when adding tokens from
            each article to the buffer. This means sequences can start at any
            position within an article, not just at the beginning. Default True.
    
    Yields:
        torch.Tensor of shape (seq_len + 1,) containing token IDs.
        Use tensor[:-1] for input and tensor[1:] for targets.
    
    Example:
        >>> from allformers.data import load_wikipedia, WikipediaConfig
        >>> from transformers import AutoTokenizer
        >>> 
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = load_wikipedia(WikipediaConfig(split="train[:1000]"))
        >>> 
        >>> streaming = StreamingTextDataset(
        ...     dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     seq_len=512,
        ... )
        >>> 
        >>> loader = DataLoader(streaming, batch_size=4, num_workers=0)
        >>> batch = next(iter(loader))
        >>> input_ids, targets = batch[:, :-1], batch[:, 1:]
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len: int = 512,
        text_fn: Optional[Callable[[dict], str]] = None,
        seed: int = 42,
        random_offset: bool = True,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.text_fn = text_fn if text_fn is not None else wikipedia_text_fn
        self.random_offset = random_offset
        
        # Check if tokenizer adds EOS automatically
        test_tokens = tokenizer.encode("test")
        self.tokenizer_adds_eos = (
            tokenizer.eos_token_id is not None
            and len(test_tokens) > 0
            and test_tokens[-1] == tokenizer.eos_token_id
        )
        self.eos_token_id = tokenizer.eos_token_id
        
        # Cache dataset length for random sampling
        self.dataset_len = len(dataset)
    
    def __iter__(self):
        """Iterate over the dataset, yielding fixed-length token sequences."""
        # Handle multiple workers - each worker gets a different seed
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed
        
        rng = random.Random(worker_seed)
        token_buffer = []
        
        while True:
            # Fill buffer until we have enough tokens for a sequence
            while len(token_buffer) < self.seq_len + 1:
                # Sample a random article
                idx = rng.randint(0, self.dataset_len - 1)
                row = self.dataset[idx]
                text = self.text_fn(row)
                
                # Tokenize
                tokens = self.tokenizer.encode(text)
                
                # Apply random offset - skip a random number of tokens from the start
                # This ensures sequences can start anywhere in an article, not just
                # at the beginning (mimics the original random token offset behavior)
                if self.random_offset and len(tokens) > 1:
                    offset = rng.randint(0, len(tokens) - 1)
                    tokens = tokens[offset:]
                
                token_buffer.extend(tokens)
                
                # Add EOS if tokenizer doesn't do it automatically
                if not self.tokenizer_adds_eos and self.eos_token_id is not None:
                    token_buffer.append(self.eos_token_id)
            
            # Extract a sequence from the buffer
            sequence = token_buffer[:self.seq_len + 1]
            token_buffer = token_buffer[self.seq_len + 1:]
            
            yield torch.tensor(sequence, dtype=torch.long)

