"""
Utility functions for allformers.
"""

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Get the best available device.

    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu'). If None, auto-detect.

    Returns:
        torch.device for the best available or preferred device.
    """
    if preferred is not None:
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cpu":
            return torch.device("cpu")
        else:
            print(f"WARNING: Preferred device '{preferred}' not available, auto-detecting...")

    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary with device availability info.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "best_device": str(get_device()),
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()

    return info

