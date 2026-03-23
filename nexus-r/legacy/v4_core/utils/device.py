"""
SNAP-C1 V4: Centralized Device Resolution
==========================================
All V2/V3/V4 modules import this single function to guarantee they ALL
target the same physical GPU.

Priority order:
  1. NVIDIA CUDA  (RunPod / Cloud GPUs)
  2. AMD DirectML (Local RX 7600)
  3. CPU fallback
"""
import torch

_cached_device = None

def get_device():
    """
    Returns the best available device.
    Priority: CUDA → DirectML (AMD) → CPU
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    # 1. NVIDIA CUDA
    if torch.cuda.is_available():
        _cached_device = torch.device('cuda')
        return _cached_device

    # 2. AMD DirectML — explicitly pick the dGPU (RX 7600 = index 1).
    # Index 0 is the iGPU (AMD Radeon Graphics); index 1 is the discrete RX 7600.
    try:
        import torch_directml
        n = torch_directml.device_count()
        # Prefer any device whose name contains "RX" or "Radeon RX"; fall back to last device.
        target_idx = 0
        for i in range(n):
            name = torch_directml.device_name(i)
            if "RX" in name:
                target_idx = i
                break
        _cached_device = torch_directml.device(target_idx)
        return _cached_device
    except Exception:
        pass

    # 3. CPU fallback
    _cached_device = torch.device('cpu')
    return _cached_device
