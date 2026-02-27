import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM: {mem:.1f} GB")
else:
    print("No GPU detected - check HSA_OVERRIDE_GFX_VERSION and AMD drivers")

