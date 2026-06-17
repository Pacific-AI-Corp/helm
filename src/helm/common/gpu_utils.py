import torch


def is_cuda_available() -> bool:
    """Checks if CUDA is available on the machine."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Checks if Apple Metal (MPS) is available on the machine."""
    return torch.backends.mps.is_available()


def get_torch_device_name() -> str:
    """Return the best available PyTorch device: CUDA, then MPS, then CPU."""
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_torch_device() -> torch.device:
    """Return the best available PyTorch device."""
    return torch.device(get_torch_device_name())
