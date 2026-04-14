# ---------------------------------------------------------
# Utility: device selection
# ---------------------------------------------------------
import torch

def get_best_device() -> torch.device:
    """
    Select the most capable available compute device.

    Backends are checked in descending order of typical performance and
    ecosystem maturity:

    1. NVIDIA CUDA (torch.cuda)
    2. Intel XPU (torch.xpu)
    3. AMD ROCm (torch.backends.amd)
    4. Apple Metal / MPS (torch.backends.mps)
    5. CPU fallback

    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if hasattr(torch.backends, "amd") and torch.backends.amd.is_available():
        return torch.device("amd")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
