from unittest.mock import patch

from helm.common.gpu_utils import get_torch_device_name, is_mps_available


def test_get_torch_device_name_prefers_cuda():
    with patch("helm.common.gpu_utils.is_cuda_available", return_value=True):
        with patch("helm.common.gpu_utils.is_mps_available", return_value=True):
            assert get_torch_device_name() == "cuda"


def test_get_torch_device_name_uses_mps_on_apple_silicon():
    with patch("helm.common.gpu_utils.is_cuda_available", return_value=False):
        with patch("helm.common.gpu_utils.is_mps_available", return_value=True):
            assert get_torch_device_name() == "mps"


def test_get_torch_device_name_falls_back_to_cpu():
    with patch("helm.common.gpu_utils.is_cuda_available", return_value=False):
        with patch("helm.common.gpu_utils.is_mps_available", return_value=False):
            assert get_torch_device_name() == "cpu"


def test_is_mps_available_delegates_to_torch():
    with patch("torch.backends.mps.is_available", return_value=True):
        assert is_mps_available() is True
