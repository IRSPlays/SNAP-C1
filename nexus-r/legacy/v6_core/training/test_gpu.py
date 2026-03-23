"""
V6 WHORMHOLE - Quick GPU Test
==============================
Verifies model works on GPU before full training.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time

def test_gpu():
    """Test V6 on GPU."""
    print("="*60)
    print("V6 GPU Test")
    print("="*60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    # Import V6
    from v6_core.architecture.v6_assembly import V6ResonanceModel
    
    # Create small model for testing
    print("Creating model...")
    model = V6ResonanceModel(
        d_model=512,
        n_blocks=4,
        n_heads=4,
        window_size=64,
        max_seq_len=256,
        vocab_size=1000,
        K_hash=4,
        d_hash=64,
        use_skip=True,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params:.1f}M params")
    
    # Test forward pass
    print("\nTesting forward pass...")
    token_ids = torch.randint(0, 1000, (4, 128), device=device)
    type_ids = torch.zeros(4, 128, device=device, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        result = model.forward_pretrain(token_ids, type_ids, token_ids)
    print(f"Forward pass OK: {result['logits'].shape}")
    
    # Test training
    print("\nTesting training...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    losses = []
    start = time.time()
    
    for step in range(100):
        optimizer.zero_grad()
        
        with autocast():
            result = model.forward_pretrain(token_ids, type_ids, token_ids)
            loss = result['loss']
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (step + 1) / elapsed
            print(f"Step {step+1:3d} | Loss: {loss.item():.4f} | Rate: {rate:.1f} step/s")
    
    avg_loss = sum(losses) / len(losses)
    print(f"\nTraining OK! Avg loss: {avg_loss:.4f}")
    
    # Memory check
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM used: {allocated:.2f} GB (reserved: {reserved:.2f} GB)")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("GPU TEST PASSED!")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_gpu()
    exit(0 if success else 1)
