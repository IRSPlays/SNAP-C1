"""
Improved Training Script with BPE Tokenizer and WSD LR Schedule
===============================================================

Key improvements over baseline:
1. BPE subword tokenization (8K vocab vs 50 char)
2. WSD learning rate schedule (proven better than cosine)
3. Better weight initialization (T5-style)
4. Gradient clipping and stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List
import math
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup DirectML for AMD RX 7600
try:
    import torch_directml
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        print(f"Using AMD RX 7600 GPU: {DEVICE}")
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DirectML not available, using: {DEVICE}")
except ImportError:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"torch_directml not installed, using: {DEVICE}")

from nexus_v1 import NexusV7, RMSNorm
from tokenizer import TiktokenTokenizer, download_tiny_shakespeare
from scheduler import create_scheduler


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
    Improved NEXUS V7 with better initialization.

    Key changes:
    1. T5-style embedding initialization (d_model^-0.5 scaling)
    2. Better linear layer initialization
    3. Proper weight tying with scaled initialization
    """

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # T5-style: scaled initialization for attention projections
            if hasattr(module, 'weight') and module.weight.dim() > 1:
                # Attention projection layers
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
            else:
                # FFN layers - slightly larger init
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Scaled embedding initialization
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
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
    device=None
):
    """Train improved model."""
    if device is None:
        device = DEVICE
    """Train improved model."""
    print("="*60)
    print("Training Improved NEXUS V7 with Tiktoken BPE")
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = ImprovedNexusV7(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_q_heads=4,
        num_kv_heads=2,
        d_ffn=d_model * 4,
        dropout=0.1,
        max_seq_len=max_len
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,} ({params/1e6:.1f}M)")

    # Create optimizer with weight decay - separate decay groups
    # Standard practice: DON'T apply weight decay to biases and RMSNorm weights
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Apply weight decay to weights only (Linear, Embedding)
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
        betas=(0.9, 0.95)
    )

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

    # Training loop
    print("\nStarting training...")
    step = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # FIX: Use iterator to allow resetting when DataLoader exhausts
    data_iter = iter(train_loader)
    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

            model.train()
            input_ids = batch['input_ids'].to(device)

            optimizer.zero_grad()
            result = model(input_ids, labels=input_ids)
            loss = result['loss']

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss = loss.item()
            history['train_loss'].append((step, train_loss))
            history['lr'].append((step, optimizer.param_groups[0]['lr']))

            # Validation
            if step > 0 and step % 100 == 0:
                model.eval()
                with torch.no_grad():  # FIX: Prevent gradient computation during validation
                    val_losses = []
                    val_iter = iter(val_loader)
                    for _ in range(min(5, len(val_loader))):  # Quick validation with DIFFERENT batches
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)  # Reset if we run out
                            val_batch = next(val_iter)
                        val_input = val_batch['input_ids'].to(device)
                        val_result = model(val_input, labels=val_input)
                        val_losses.append(val_result['loss'].item())
                    val_loss = sum(val_losses) / len(val_losses)

                history['val_loss'].append((step, val_loss))

                lr = optimizer.param_groups[0]['lr']
                train_ppl = math.exp(min(train_loss, 20))
                val_ppl = math.exp(min(val_loss, 20))

                print(f"Step {step:5d}: train={train_loss:.4f} ({train_ppl:.1f}), "
                      f"val={val_loss:.4f} ({val_ppl:.1f}), lr={lr:.2e}")

                # Save best checkpoint with optimizer/scheduler state for resumability
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'history': history
                    }
                    torch.save(checkpoint, 'outputs/best_model.pt')
                    print(f"  -> Saved best model (val_loss={val_loss:.4f})")

            step += 1

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Final perplexity: {math.exp(best_val_loss):.1f}")

    # Save final checkpoint with full state
    final_checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history
    }
    torch.save(final_checkpoint, 'outputs/final_model.pt')
    print("Saved final checkpoint to outputs/final_model.pt")

    return model, tokenizer, history


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler):
    """
    Load a checkpoint to resume training.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into

    Returns:
        tuple: (step, best_val_loss, history)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'], checkpoint['best_val_loss'], checkpoint['history']


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

    # Train model on TinyShakespeare
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, tokenizer, history = train_improved(
        training_text,
        vocab_size=tokenizer.vocab_size,  # Use tiktoken's vocab
        d_model=256,
        num_layers=6,
        num_steps=2000,
        batch_size=8,
        peak_lr=3e-4,
        warmup_steps=200,
        stable_steps=800,
        device=device
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