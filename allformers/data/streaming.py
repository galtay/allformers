"""
Streaming dataset utilities for on-the-fly tokenization.

This module provides streaming datasets that tokenize text on-the-fly,
avoiding the need to load entire tokenized datasets into memory.

Uses HuggingFace IterableDataset with shuffle buffers for randomization.
See: https://huggingface.co/docs/datasets/main/stream#shuffle

For DDP (Distributed Data Parallel) training, use the rank and world_size
parameters to shard the dataset across workers.
"""

from typing import Callable, Optional

import torch
from datasets.distributed import split_dataset_by_node
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
    - Supports DDP training via rank/world_size sharding
    
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
        rank: The rank of the current process in DDP training (0 to world_size-1).
            If provided along with world_size, the dataset will be sharded so each
            worker processes a unique subset of the data. Default: None (no sharding).
        world_size: Total number of processes in DDP training. Must be provided
            together with rank. Default: None (no sharding).
    
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
    
    DDP Example:
        >>> # In each DDP process:
        >>> rank = torch.distributed.get_rank()
        >>> world_size = torch.distributed.get_world_size()
        >>> 
        >>> train_dataset = StreamingTextDataset(
        ...     dataset=train_data,
        ...     tokenizer=tokenizer,
        ...     seq_len=512,
        ...     text_fn=wikipedia_text_fn,
        ...     rank=rank,
        ...     world_size=world_size,
        ... )
        >>> # Each worker automatically gets a unique shard of the data
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len: int = 512,
        text_fn: Optional[Callable[[dict], str]] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        if text_fn is None:
            raise ValueError(
                "text_fn is required. Use the text function from your dataset module:\n"
                "  - Wikipedia: wikipedia_text_fn from allformers.data.wikipedia\n"
                "  - FinePDFs-Edu: finepdfs_edu_text_fn from allformers.data.finepdfs_edu"
            )
        
        # Validate rank/world_size consistency
        if (rank is None) != (world_size is None):
            raise ValueError(
                "rank and world_size must both be provided or both be None. "
                f"Got rank={rank}, world_size={world_size}"
            )
        
        # Apply sharding for DDP if rank/world_size are provided
        if rank is not None and world_size is not None:
            if not (0 <= rank < world_size):
                raise ValueError(
                    f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}"
                )
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_fn = text_fn
        self.rank = rank
        self.world_size = world_size
        
        # Disable tokenizer warning about sequence length - we handle chunking ourselves
        # This prevents: "Token indices sequence length is longer than the specified 
        # maximum sequence length for this model (N > 1024)"
        self.tokenizer.model_max_length = 10**12
        
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
