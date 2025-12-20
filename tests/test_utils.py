"""
Tests for allformers.utils
"""

import torch
import pytest

from allformers.utils import get_device, get_device_info


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_torch_device(self):
        """get_device should return a torch.device object."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cpu_always_available(self):
        """CPU device should always be selectable."""
        device = get_device(preferred="cpu")
        assert device.type == "cpu"

    def test_invalid_preferred_falls_back(self, capsys):
        """Invalid preferred device should trigger warning and fallback."""
        device = get_device(preferred="invalid_device")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert isinstance(device, torch.device)

    def test_cuda_returns_cuda_when_available(self):
        """If CUDA is available, get_device('cuda') should return cuda device."""
        if torch.cuda.is_available():
            device = get_device(preferred="cuda")
            assert device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_mps_returns_mps_when_available(self):
        """If MPS is available, get_device('mps') should return mps device."""
        if torch.backends.mps.is_available():
            device = get_device(preferred="mps")
            assert device.type == "mps"
        else:
            pytest.skip("MPS not available")

    def test_auto_detect_prefers_gpu(self):
        """Auto-detect should prefer GPU (CUDA or MPS) over CPU."""
        device = get_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_returns_dict(self):
        """get_device_info should return a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        """Info dict should contain standard keys."""
        info = get_device_info()
        assert "cuda_available" in info
        assert "mps_available" in info
        assert "best_device" in info

    def test_cuda_info_when_available(self):
        """If CUDA available, should include device name and count."""
        info = get_device_info()
        if info["cuda_available"]:
            assert "cuda_device_name" in info
            assert "cuda_device_count" in info
            assert info["cuda_device_count"] >= 1

    def test_boolean_availability_values(self):
        """Availability values should be booleans."""
        info = get_device_info()
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["mps_available"], bool)

    def test_best_device_is_string(self):
        """Best device should be a string representation."""
        info = get_device_info()
        assert isinstance(info["best_device"], str)
        assert info["best_device"] in ["cpu", "cuda", "mps"]

