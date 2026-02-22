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
    """Returns the best available GPU device, cached after first call."""
    global _cached_device
    if _cached_device is not None:
        return _cached_device
    
    # 1. NVIDIA CUDA (RunPod RTX 6000 Ada, cloud GPUs, etc.)
    if torch.cuda.is_available():
        _cached_device = torch.device('cuda')
        return _cached_device
    
    # 2. AMD DirectML (Local Windows development on RX 7600)
    try:
        import torch_directml
        target_id = 0
        for i in range(torch_directml.device_count()):
            if "RX 7600" in torch_directml.device_name(i):
                target_id = i
                break
        _cached_device = torch_directml.device(target_id)
        return _cached_device
    except ImportError:
        pass
    
    # 3. CPU fallback
    _cached_device = torch.device('cpu')
    return _cached_device
