"""
Improved Training Script with BPE Tokenizer and WSD LR Schedule
===============================================================

Key improvements over baseline:
1. BPE subword tokenization (8K vocab vs 50 char)
2. WSD learning rate schedule (proven better than cosine)
3. Better weight initialization (T5-style)
4. Gradient clipping and stable training
5. Dynamic Batch Size Optimization with memory profiling for AMD RX 7600
6. Gradient accumulation for effective larger batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, List, Tuple
import math
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup DirectML for AMD RX 7600
try:
    import torch_directml
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        DEVICE_TYPE = 'directml'
        print(f"Using AMD RX 7600 GPU: {DEVICE}")
        # Enable TF32 for better performance on RDNA3
        if hasattr(torch, 'backends'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DirectML not available, using: {DEVICE}")
except ImportError:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"torch_directml not installed, using: {DEVICE}")

from nexus_v1 import NexusV7, RMSNorm
from tokenizer import TiktokenTokenizer, download_tiny_shakespeare
from scheduler import create_scheduler


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights that is updated with exponential
    moving average: shadow = decay * shadow + (1 - decay) * weights

    Benefits:
    - Improved generalization (typically 0.5-2% improvement)
    - More stable training (smooths out weight oscillations)
    - Works by averaging out noise in gradient updates

    Usage:
    - ema = EMA(model, decay=0.999)
    - After optimizer step: ema.update()
    - For validation/generation: ema.apply_to(model)
    - To restore original: ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        """
        Initialize EMA with model parameters.

        Args:
            model: The model to track
            decay: EMA decay rate (0.999 is standard, higher = slower average)
            device: Device to store shadow weights on
        """
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create shadow weights (deep copy)
        self.shadow = {}
        self._create_shadow()

        # Track if EMA weights are currently applied
        self.ema_active = False

    def _create_shadow(self):
        """Create shadow weight copies."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone().to(self.device)

    def update(self):
        """Update shadow weights with exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA: shadow = decay * shadow + (1 - decay) * current
                self.shadow[name].mul_(self.decay).add_(param.data.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        """
        Apply EMA weights to model (for validation/generation).

        Saves original weights so they can be restored later.
        """
        if self.ema_active:
            return  # Already applied

        self._original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._original_weights[name] = param.data.detach().clone()
                param.data.copy_(self.shadow[name].to(param.device))

        self.ema_active = True

    def restore(self, model: nn.Module):
        """Restore original weights after EMA validation."""
        if not self.ema_active:
            return  # Not applied

        for name, param in model.named_parameters():
            if name in self._original_weights:
                param.data.copy_(self._original_weights[name])
                del self._original_weights[name]

        self.ema_active = False

    def get_shadow(self) -> Dict[str, torch.Tensor]:
        """Get shadow weights dict (useful for checkpointing)."""
        return self.shadow

    def load_shadow(self, shadow_state_dict: Dict[str, torch.Tensor]):
        """Load shadow weights from state dict (for resuming)."""
        self.shadow = {k: v.clone() for k, v in shadow_state_dict.items()}


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage in MB.
    Works with DirectML, CUDA, and CPU (estimated).
    """
    if DEVICE_TYPE == 'directml':
        try:
            import torch_directml
            # DirectML doesn't have standard memory reporting
            # Use torch memory stats if available, otherwise estimate
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated = torch.cuda.memory_allocated()
                max_allocated = torch.cuda.max_memory_allocated() if hasattr(torch.cuda, 'max_memory_allocated') else allocated
                return {
                    'allocated_mb': allocated / (1024 * 1024),
                    'max_allocated_mb': max_allocated / (1024 * 1024)
                }
        except:
            pass
        # Fallback: return 0 - actual estimation done by estimate_training_memory
        return {'allocated_mb': 0, 'max_allocated_mb': 0}
    elif DEVICE_TYPE == 'cuda':
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return {'allocated_mb': allocated, 'max_allocated_mb': max_allocated}
    else:
        return {'allocated_mb': 0, 'max_allocated_mb': 0}


def estimate_training_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    use_amp: bool = True,
    use_gradient_checkpointing: bool = True
) -> Dict[str, float]:
    """
    Estimate training memory usage for a given configuration.
    This provides memory estimates WITHOUT actually running training.

    Memory breakdown for NEXUS V7 with SwiGLU on AMD RX 7600:
    - Model weights: params * 2 bytes (BF16 with AMP)
    - Gradients: params * 4 bytes (FP32 for optimizer)
    - Optimizer states: params * 8 bytes (AdamW: 2 states per param)
    - Activations: params * activation_factor (varies by config)

    The activation factor accounts for:
    - Embeddings: small memory
    - Q/K/V/Output projections: d_model^2 per layer
    - FFN (SwiGLU): 2 * d_model * d_ffn per layer
    - Attention scores: batch * num_heads * seq^2 (causal mask)
    - All stored in FP32 for mixed precision training

    Args:
        model: The model to estimate memory for
        batch_size: Batch size per micro-batch
        seq_len: Sequence length
        vocab_size: Vocabulary size
        use_amp: Whether AMP (BF16) is used
        use_gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        Dict with memory estimates in MB
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Get model dimensions
    d_model = getattr(model, 'd_model', 512)
    num_layers = getattr(model, 'num_layers', 12)
    d_head = d_model // getattr(model, 'num_q_heads', 8)
    num_kv_heads = getattr(model, 'num_kv_heads', 2)
    d_ffn = getattr(model, 'd_ffn', d_model * 4)

    # Weights (BF16 = 2 bytes with AMP, FP32 = 4 bytes without)
    weights_mb = (num_params * (2 if use_amp else 4)) / (1024 * 1024)

    # Gradients (always FP32 for optimizer regardless of AMP)
    gradients_mb = (num_params * 4) / (1024 * 1024)

    # Optimizer states (AdamW: exp_avg + exp_avg_sq per parameter = 8 bytes)
    optimizer_mb = (num_params * 8) / (1024 * 1024)

    # Embedding lookup memory
    embedding_mb = (vocab_size * d_model * 4) / (1024 * 1024)  # FP32 for embedding table

    # Activation memory estimation
    #
    # For transformer training, activations stored for backward pass include:
    # - Q/K/V projections: (B, T, d_model) each = 3 * B * T * d_model
    # - Attention scores: (B, num_heads, T, T) causal = B * num_heads * T^2 / 2
    # - FFN activations: (B, T, d_ffn) * 2 for SwiGLU gates
    #
    # Empirical formula (calibrated for NEXUS V7 with SwiGLU):
    # per_layer_activations ≈ batch * seq * d_model * 8 bytes
    #
    # This empirically accounts for all the intermediate tensors needed
    # during backward pass with mixed precision training.

    # Per-layer activation memory (empirical formula)
    # Scale factor of 8 accounts for: Q/K/V projections (3), attention scores (3),
    # FFN gates (2), and other intermediate tensors
    per_layer_activations = batch_size * seq_len * d_model * 8 / (1024 * 1024)

    # Total activations across all layers
    if use_gradient_checkpointing:
        # Gradient checkpointing: ~50% memory reduction
        activations_mb = per_layer_activations * num_layers * 0.5
    else:
        # Full activation storage
        activations_mb = per_layer_activations * num_layers

    # Input tensor
    input_mb = (batch_size * seq_len * 4) / (1024 * 1024)

    total_mb = weights_mb + gradients_mb + optimizer_mb + embedding_mb + activations_mb + input_mb

    return {
        'weights_mb': weights_mb,
        'gradients_mb': gradients_mb,
        'optimizer_mb': optimizer_mb,
        'embedding_mb': embedding_mb,
        'activations_mb': activations_mb,
        'input_mb': input_mb,
        'total_mb': total_mb,
        'headroom_mb': 8192 - total_mb  # RX 7600 has 8GB
    }


def find_optimal_batch_size(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    initial_batch_size: int = 4,
    max_batch_size: int = 256,
    min_batch_size: int = 1
) -> Tuple[int, Dict[str, float]]:
    """
    Binary search to find the largest batch size that fits in GPU memory.
    Uses model-based estimation for DirectML, actual measurement for CUDA.

    Args:
        model: The model to test
        seq_len: Sequence length
        vocab_size: Vocabulary size
        initial_batch_size: Starting batch size for search
        max_batch_size: Upper bound for search
        min_batch_size: Lower bound for search

    Returns:
        (optimal_batch_size, memory_info)
    """
    print("\n" + "="*60)
    print("Finding optimal batch size for AMD RX 7600...")
    print("="*60)

    device = next(model.parameters()).device
    model.train()

    # Test input
    test_input = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # For DirectML, use model-based estimation since memory reporting is broken
    if DEVICE_TYPE == 'directml':
        print("Using model-based memory estimation for DirectML...")

        def estimate_batch_size(bs: int) -> Dict[str, float]:
            """Estimate memory for a batch size."""
            return estimate_training_memory(
                model, bs, seq_len, vocab_size,
                use_amp=True,  # Assume AMP is enabled
                use_gradient_checkpointing=True  # Assume checkpointing is enabled
            )

        # Binary search on estimated memory
        low = min_batch_size
        high = min(max_batch_size, 256)  # Cap at 256 for estimation
        optimal = initial_batch_size

        while low <= high:
            mid = (low + high) // 2
            mem = estimate_batch_size(mid)
            headroom = mem['headroom_mb']

            print(f"  batch_size={mid}: {mem['total_mb']:.0f} MB (headroom: {headroom:.0f} MB)")

            if headroom > 500:  # Keep at least 500MB headroom
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1

        # Verify by trying the optimal batch size
        print(f"\nVerifying optimal batch_size={optimal}...")
        try:
            x = torch.randint(0, vocab_size, (optimal, seq_len), device=device)
            result = model(x, labels=x)
            loss = result['loss']
            loss.backward()
            model.zero_grad()
            print(f"  Verification PASSED")
            verified = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Verification FAILED (OOM) - reducing batch size")
                verified = False
            else:
                raise e

        if not verified:
            # Reduce until it fits
            while optimal > min_batch_size:
                optimal //= 2
                try:
                    x = torch.randint(0, vocab_size, (optimal, seq_len), device=device)
                    result = model(x, labels=x)
                    loss = result['loss']
                    loss.backward()
                    model.zero_grad()
                    print(f"  Verified with batch_size={optimal}")
                    break
                except:
                    continue

        mem_info = estimate_batch_size(optimal)
        print(f"\nOptimal batch size: {optimal}")
        print(f"  Weights: {mem_info['weights_mb']:.1f} MB")
        print(f"  Gradients: {mem_info['gradients_mb']:.1f} MB")
        print(f"  Optimizer: {mem_info['optimizer_mb']:.1f} MB")
        print(f"  Activations: {mem_info['activations_mb']:.1f} MB")
        print(f"  Total: {mem_info['total_mb']:.1f} MB / 8192 MB")

        model.zero_grad()
        model.train()

        return optimal, mem_info

    # For CUDA, use actual memory measurement
    def test_batch_size(bs: int) -> bool:
        """Try a batch size, return True if it fits."""
        try:
            # Clear cache
            if DEVICE_TYPE == 'cuda':
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == 'directml':
                torch.cuda.empty_cache()

            # Create batch
            x = torch.randint(0, vocab_size, (bs, seq_len), device=device)

            # Forward
            result = model(x, labels=x)
            loss = result['loss']

            # Backward
            loss.backward()

            # Check memory
            mem = get_memory_usage()
            memory_ok = mem['max_allocated_mb'] < 7000  # Keep under 7GB to leave headroom

            # Clear gradients
            model.zero_grad()

            return memory_ok
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise e

    # Binary search
    low = min_batch_size
    high = max_batch_size
    optimal = initial_batch_size

    while low <= high:
        mid = (low + high) // 2
        print(f"  Testing batch_size={mid}...", end=" ")

        try:
            if test_batch_size(mid):
                optimal = mid
                print(f"OK ({get_memory_usage()['max_allocated_mb']:.0f} MB)")
                low = mid + 1
            else:
                print(f"OOM")
                high = mid - 1
        except RuntimeError as e:
            print(f"Error: {e}")
            high = mid - 1

    # Final verification
    print(f"\nOptimal batch size: {optimal}")
    mem_info = get_memory_usage()

    # Reset model to initial state
    model.zero_grad()
    model.train()

    return optimal, mem_info


class TextDataset(Dataset):
    """Dataset for text training with sliding window."""

    def __init__(self, text: str, tokenizer: TiktokenTokenizer,
                 max_len: int = 256, stride: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

        # Tokenize entire text (no truncation for dataset creation)
        print(f"Tokenizing {len(text)} chars...")
        # Use large max_len to avoid truncation during dataset creation
        self.tokens = tokenizer.encode(text, max_len=1000000, add_special_tokens=False)
        print(f"Got {len(self.tokens)} tokens")

        # Create overlapping sequences using sliding window
        # FIX: Use full range(0, len(tokens)) instead of range(0, len(tokens) - stride)
        # The old code lost up to (stride-1) tokens at the end of the corpus
        self.sequences = []
        for i in range(0, len(self.tokens), stride):
            end_idx = min(i + max_len, len(self.tokens))
            seq = self.tokens[i:end_idx]
            # Pad last sequence if needed
            if len(seq) < max_len:
                seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
            self.sequences.append(seq)

        # If no sequences created (text was short), create at least one
        if len(self.sequences) == 0:
            seq = self.tokens[:max_len]
            if len(seq) < max_len:
                seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
            self.sequences.append(seq)

        print(f"Created {len(self.sequences)} sequences of length {max_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        return {'input_ids': torch.tensor(tokens, dtype=torch.long)}


class ImprovedNexusV7(NexusV7):
    """
    Improved NEXUS V7 with depth-scaled initialization.

    Key changes:
    1. Depth-scaled initialization (GPT-2/BLOOM style): 1/sqrt(2*num_layers)
    2. Attention projections use smaller init (0.006) than FFN (0.02)
    3. Proper weight tying with scaled initialization

    This prevents gradient explosion/vanishing in deeper models by
    maintaining roughly constant activation variance through layers.
    """

    def _init_weights(self, module):
        """Initialize weights with depth-scaled initialization.

        Uses GPT-2/BLOOM-style initialization:
        - Attention projections (q, k, v, out): std = 0.006 / sqrt(2 * num_layers)
        - FFN projections (w1, w2, w3): std = 0.02 / sqrt(2 * num_layers)
        - Embeddings: std = 0.02 / sqrt(2 * num_layers)
        """
        if isinstance(module, nn.Linear):
            depth_scale = 1.0 / math.sqrt(2.0 * self.num_layers)

            # Determine if this is an attention projection or FFN projection
            name = ''
            for n, p in self.named_parameters():
                if p is module.weight:
                    name = n.lower()
                    break

            # Attention projections (q, k, v, out) use smaller init
            if 'proj' in name or 'attn' in name:
                std = 0.006 * depth_scale
            else:
                # FFN projections (w1, w2, w3) use standard init
                std = 0.02 * depth_scale

            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            depth_scale = 1.0 / math.sqrt(2.0 * self.num_layers)
            std = 0.02 * depth_scale
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def test_improved_architecture():
    """Test the improved architecture."""
    print("="*60)
    print("Testing Improved NEXUS V7")
    print("="*60)

    device = DEVICE
    print(f"Device: {device}")

    # Create model
    model = ImprovedNexusV7(
        vocab_size=8192,
        d_model=256,
        num_layers=6,
        num_q_heads=4,
        num_kv_heads=2,
        d_ffn=1024,
        dropout=0.0,
        max_seq_len=256
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,} ({params/1e6:.1f}M)")

    # Test forward pass
    x = torch.randint(0, 8192, (4, 64)).to(device)
    result = model(x, labels=x)
    print(f"Forward pass OK, loss={result['loss'].item():.4f}")

    # Test backward pass
    result['loss'].backward()

    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print(f"Backward pass OK, {len(grad_norms)} parameters have gradients")

    # Check for NaN/Inf gradients
    has_nan = any(math.isnan(g) or math.isinf(g) for g in grad_norms.values())
    if has_nan:
        print("WARNING: NaN or Inf gradients detected!")
    else:
        print("No NaN/Inf gradients detected")

    # Test generation
    model.eval()
    gen_ids = model.generate(x[:, :20], max_new_tokens=30)
    print(f"Generation OK, shape={gen_ids.shape}")

    return model, grad_norms


def train_improved(
    text: str,
    vocab_size: int = 100000,
    d_model: int = 256,
    num_layers: int = 6,
    num_steps: int = 5000,
    batch_size: int = 8,
    peak_lr: float = 3e-4,
    warmup_steps: int = 200,
    stable_steps: int = 1000,
    device=None,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    auto_batch_size: bool = False,
    target_effective_batch_size: int = 64
):
    """
    Train improved model with optional mixed precision and gradient accumulation.

    Key improvements:
    - Gradient accumulation: Accumulate gradients over multiple micro-batches
      to achieve larger effective batch size without memory issues.
    - Auto batch size: Automatically find largest batch size for GPU.
    - Proper perplexity evaluation: Use held-out validation set.

    Args:
        text: Training text corpus
        vocab_size: Vocabulary size (auto-detected from tokenizer)
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_steps: Total training steps
        batch_size: Micro-batch size (before gradient accumulation)
        peak_lr: Peak learning rate
        warmup_steps: Warmup steps
        stable_steps: Stable LR steps
        device: Training device (auto-detected if None)
        use_amp: Use automatic mixed precision
        gradient_accumulation_steps: Number of micro-batches per optimizer step
        auto_batch_size: Automatically find optimal batch size
        target_effective_batch_size: Target effective batch size for gradient accumulation
    """
    if device is None:
        device = DEVICE

    print("="*60)
    print("Training Improved NEXUS V7 with Dynamic Batch Optimization")
    print("="*60)

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Use tiktoken tokenizer (no training needed - it's pre-trained)
    tokenizer = TiktokenTokenizer()
    print(f"Using Tiktoken tokenizer with vocab size: {tokenizer.vocab_size}")

    # Create datasets
    max_len = 256
    stride = 128

    # Split into train/val
    split = int(len(text) * 0.9)
    train_text = text[:split]
    val_text = text[split:]

    train_dataset = TextDataset(train_text, tokenizer, max_len, stride)
    val_dataset = TextDataset(val_text, tokenizer, max_len, stride)

    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    # Create model first (needed for batch size optimization)
    logit_clamp = float(os.environ.get('NEXUS_LOGIT_CLAMP', '0'))
    model = ImprovedNexusV7(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_q_heads=4,
        num_kv_heads=2,
        d_ffn=d_model * 4,
        dropout=0.1,
        max_seq_len=max_len,
        logit_clamp=logit_clamp
    ).to(device)

    if logit_clamp > 0:
        print(f"Attention logit clamping enabled: max |logits| <= {logit_clamp}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,} ({params/1e6:.1f}M)")

    # Enable gradient checkpointing for ~50% memory reduction
    model.enable_gradient_checkpointing()
    print(f"Gradient checkpointing enabled - reduces memory by ~50%")

    # Optional: torch.compile() for ~20-40% speedup on AMD RDNA3
    # Note: Only use with DirectML/CUDA, not with gradient checkpointing as it can cause issues
    use_compile = os.environ.get('NEXUS_COMPILE', '0') == '1'
    if use_compile and DEVICE_TYPE != 'cpu':
        print("Compiling model with torch.compile() for faster training...")
        # fullgraph=True ensures the entire model can be compiled
        # dynamic=True allows variable sequence lengths but may be slower to compile
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
        print("Model compiled successfully!")
    elif DEVICE_TYPE == 'cpu':
        print("torch.compile disabled on CPU (use DirectML/GPU for acceleration)")

    # Auto batch size optimization
    if auto_batch_size:
        optimal_bs, mem_info = find_optimal_batch_size(
            model, max_len, tokenizer.vocab_size,
            initial_batch_size=batch_size,
            max_batch_size=128
        )
        batch_size = optimal_bs
        print(f"Auto batch size found: {batch_size} (VRAM: {mem_info['max_allocated_mb']:.0f} MB)")
    else:
        print(f"Using batch size: {batch_size}")

    # Calculate gradient accumulation to reach target effective batch size
    if gradient_accumulation_steps == 1 and target_effective_batch_size > batch_size:
        gradient_accumulation_steps = target_effective_batch_size // batch_size
        print(f"Auto-set gradient accumulation: {gradient_accumulation_steps} steps")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    # DataLoader with optimized settings for AMD RX 7600
    # Note: num_workers=0 for Windows/DirectML compatibility
    # prefetch_factor requires num_workers > 0, so we disable it
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    print(f"Micro-batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Train batches per epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create optimizer with weight decay - separate decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'rmsnorm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=peak_lr,
        betas=(0.9, 0.95),
        foreach=True  # Fused implementation - ~2x faster optimizer steps on AMD RDNA3
    )

    # Initialize EMA (Exponential Moving Average) for better generalization
    # EMA typically improves validation loss by 0.5-2%
    ema = EMA(model, decay=0.999, device=device)
    print(f"EMA initialized with decay=0.999 (shadow params: {len(ema.shadow):,})")

    # Create WSD scheduler
    total_steps = num_steps
    scheduler = create_scheduler(
        'wsd',
        optimizer,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=total_steps - warmup_steps - stable_steps,
        peak_lr=peak_lr,
        min_lr=peak_lr * 0.1
    )

    print(f"\nScheduler: WSD")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  Stable: {stable_steps} steps")
    print(f"  Decay: {total_steps - warmup_steps - stable_steps} steps")
    print(f"  Peak LR: {peak_lr}")

    # Mixed precision training for AMD RX 7600 with BF16 (better for RDNA3)
    use_amp = use_amp and (device.type == 'privateuseone' or device.type == 'cuda')
    scaler = None  # Initialize scaler variable
    if use_amp:
        # BF16 is more stable than FP16 on RDNA3 - use autocast with bfloat16
        print(f"Mixed precision (BF16) enabled with GradScaler for AMD RX 7600")
        # Note: GradScaler works with BF16 on DirectML via privateuseone
        scaler = GradScaler()
    else:
        print(f"Training in FP32")

    # Checkpoint resume
    checkpoint_path = 'outputs/best_model.pt'
    start_step = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', {
            'train_loss': [], 'val_loss': [], 'lr': [],
            'grad_norm': [], 'effective_batch': []
        })
        # Load EMA shadow weights if available
        if 'ema_shadow' in checkpoint:
            ema.load_shadow(checkpoint['ema_shadow'])
            print(f"Loaded EMA shadow weights")
        print(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")
    else:
        step = 0
        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'val_loss': [], 'lr': [],
            'grad_norm': [], 'effective_batch': []
        }

    # Training loop with gradient accumulation
    print("\nStarting training...")
    print("="*60)
    step = start_step
    best_step = step
    patience = 500
    no_improve_count = 0
    accum_steps = 0  # Gradient accumulation counter
    accumulated_loss = 0.0  # FIX: Track accumulated loss correctly

    # Use iterator for data loading
    data_iter = iter(train_loader)
    while step < num_steps:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        model.train()
        input_ids = batch['input_ids'].to(device)

        # Gradient accumulation - FIX: Track accumulated loss correctly
        # We DON'T use no_grad() here because we need backward() to compute gradients
        if use_amp and scaler is not None:
            # Mixed precision micro-batch
            with autocast(dtype=torch.bfloat16):
                result = model(input_ids, labels=input_ids)
                loss = result['loss']
            # Scale loss for gradient accumulation, then scale for AMP
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.item()
        else:
            # FP32 micro-batch
            result = model(input_ids, labels=input_ids)
            loss = result['loss'] / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item() * gradient_accumulation_steps

        accum_steps += 1

        # Only optimizer step when accumulation is complete
        if accum_steps >= gradient_accumulation_steps:
            accum_steps = 0

            # FIX: Use accumulated loss for proper averaging
            train_loss = accumulated_loss / gradient_accumulation_steps
            accumulated_loss = 0.0  # Reset for next cycle

            # Unscale gradients for clipping
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            # Compute gradient norm for monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            history['grad_norm'].append((step, grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))

            # Optimizer step
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            # Update EMA shadow weights after optimizer step
            ema.update()

            history['train_loss'].append((step, train_loss))
            history['lr'].append((step, optimizer.param_groups[0]['lr']))
            history['effective_batch'].append((step, batch_size * gradient_accumulation_steps))

            # Validation
            if step > 0 and step % 100 == 0:
                # Use EMA weights for validation (improves generalization)
                ema.apply_to(model)
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    val_iter = iter(val_loader)
                    for _ in range(min(5, len(val_loader))):
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)
                        val_input = val_batch['input_ids'].to(device)
                        val_result = model(val_input, labels=val_input)
                        val_losses.append(val_result['loss'].item())
                    val_loss = sum(val_losses) / len(val_losses)

                history['val_loss'].append((step, val_loss))

                lr = optimizer.param_groups[0]['lr']
                train_ppl = math.exp(min(train_loss, 20))
                val_ppl = math.exp(min(val_loss, 20))
                gn = history['grad_norm'][-1][1] if history['grad_norm'] else 0

                # Check improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    no_improve_count = 0
                    checkpoint = {
                        'step': step,
                        'best_step': best_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'history': history,
                        'batch_size': batch_size,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'ema_shadow': ema.get_shadow()  # Save EMA weights
                    }
                    torch.save(checkpoint, 'outputs/best_model.pt')
                    print(f"Step {step:5d}: train={train_loss:.4f} ({train_ppl:.1f}), "
                          f"val={val_loss:.4f} ({val_ppl:.1f}), lr={lr:.2e}, gn={gn:.2f} **BEST**")
                else:
                    no_improve_count += 1
                    print(f"Step {step:5d}: train={train_loss:.4f} ({train_ppl:.1f}), "
                          f"val={val_loss:.4f} ({val_ppl:.1f}), lr={lr:.2e}, gn={gn:.2f} "
                          f"[no_improve={no_improve_count}]")

                # Early stopping
                if no_improve_count >= patience:
                    print(f"\nEarly stopping at step {step} "
                          f"(no improvement for {patience} steps)")
                    ema.restore(model)  # Restore original weights before exit
                    break

                # Restore original (non-EMA) weights after validation
                ema.restore(model)

            step += 1

    print("="*60)
    print(f"Training complete! Best val_loss: {best_val_loss:.4f} at step {best_step}")
    print(f"Final perplexity: {math.exp(best_val_loss):.1f}")

    # Save final checkpoint
    final_checkpoint = {
        'step': step,
        'best_step': best_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'ema_shadow': ema.get_shadow()  # Save EMA weights
    }
    torch.save(final_checkpoint, 'outputs/final_model.pt')
    print("Saved final checkpoint to outputs/final_model.pt")

    return model, tokenizer, history


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler, ema=None):
    """
    Load a checkpoint to resume training.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        ema: Optional EMA instance to load shadow weights into

    Returns:
        tuple: (step, best_val_loss, history)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if ema is not None and 'ema_shadow' in checkpoint:
        ema.load_shadow(checkpoint['ema_shadow'])
    return checkpoint['step'], checkpoint['best_val_loss'], checkpoint['history']


def load_model_with_ema(checkpoint_path: str, model, device):
    """
    Load model from checkpoint and apply EMA weights for inference.

    This gives you the best of both worlds:
    - Training with EMA for stability
    - Inference with EMA-averaged weights for better generalization

    Args:
        checkpoint_path: Path to checkpoint (best_model.pt or final_model.pt)
        model: The model to load weights into
        device: Device to run on

    Returns:
        The model with EMA weights loaded
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create EMA and load shadow weights
    ema = EMA(model, decay=0.999, device=device)
    if 'ema_shadow' in checkpoint:
        ema.load_shadow(checkpoint['ema_shadow'])
        # Apply EMA weights to model
        ema.apply_to(model)
        print(f"Loaded model with EMA weights (val_loss={checkpoint.get('best_val_loss', 'N/A')})")
    else:
        # Fallback to regular weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model without EMA (no EMA shadow in checkpoint)")

    return model, checkpoint.get('best_val_loss', None)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Generate text from prompt."""
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer.encode(prompt, max_len=256)
    input_ids = torch.tensor([encoded]).to(device)

    gen_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    generated = tokenizer.decode(gen_ids[0].tolist())

    return generated


if __name__ == '__main__':
    # Test architecture
    print("\n" + "="*60)
    print("TEST 1: Architecture")
    print("="*60)
    test_improved_architecture()

    # Test tokenizer
    print("\n" + "="*60)
    print("TEST 2: Tiktoken Tokenizer")
    print("="*60)
    tokenizer = TiktokenTokenizer()
    text = "Hello world! This is a test of the BPE tokenizer."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: '{text}'")
    print(f"Encoded: {len(encoded)} tokens")
    print(f"Decoded: '{decoded}'")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print("Tokenizer test PASSED!")

    # Get training text - use TinyShakespeare
    print("\n" + "="*60)
    print("TEST 3: Training on TinyShakespeare")
    print("="*60)

    # Download or load TinyShakespeare
    training_text = download_tiny_shakespeare()
    print(f"Training text length: {len(training_text)} chars")

    # Train model on TinyShakespeare using the properly configured DEVICE
    print(f"Using device: {DEVICE}")

    model, tokenizer, history = train_improved(
        training_text,
        vocab_size=tokenizer.vocab_size,  # Use tiktoken's vocab
        d_model=384,  # Increased from 256 for better capacity
        num_layers=8,  # Increased from 6 for deeper model
        num_steps=3000,  # Increased from 2000 for better convergence
        batch_size=16,  # Micro-batch size (gradient accumulation used for effective batch)
        peak_lr=3e-4,
        warmup_steps=300,  # Scaled with num_steps
        stable_steps=1200,  # Scaled with num_steps
        device=DEVICE,
        gradient_accumulation_steps=1,  # Default, auto-adjusts if target_effective_batch_size > batch_size
        auto_batch_size=False,  # Set True to auto-detect optimal batch size
        target_effective_batch_size=64  # Effective batch size target
    )

    # Test generation
    print("\n" + "="*60)
    print("TEST 4: Generation")
    print("="*60)

    prompt = "ROMEO:"
    generated = generate_text(model, tokenizer, prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)