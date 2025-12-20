"""
allformers - Readable transformer implementations for learning

A collection of transformer model implementations built with PyTorch and einops,
prioritizing clarity and understanding over cutting-edge performance.

Supported Models:
- GPT-2: OpenAI's GPT-2 language model

Usage:
    from allformers.models.gpt2.gpt2 import GPT2, GPT2Config
"""

from importlib.metadata import version

__version__ = version("allformers")
