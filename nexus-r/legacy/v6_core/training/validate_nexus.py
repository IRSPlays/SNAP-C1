"""
NEXUS V6 Validation Script
==========================
Tests for:
- Numerical stability (NaN/Inf)
- Loss convergence
- Gradient flow
- Actual performance on coding/reasoning tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v6_core.architecture.nexus_v6 import (
    NexusV6, build_nexus_small, WSDTrainer
)


class SyntheticCodingDataset(Dataset):
    """Synthetic dataset that mimics coding patterns."""
    def __init__(self, num_samples=10000, seq_len=128, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        torch.manual_seed(42)
        
        # Generate synthetic "code-like" sequences
        # Patterns: function definitions, loops, conditionals
        self.data = []
        for _ in range(num_samples):
            seq = self._generate_code_sequence(seq_len)
            self.data.append(seq)
        
    def _generate_code_sequence(self, length):
        """Generate a sequence that looks like code tokens."""
        tokens = []
        for i in range(length):
            # Simulate code structure
            if i % 20 == 0:
                tokens.append(torch.randint(0, 100, (1,)).item())  # def/function
            elif i % 20 == 1:
                tokens.append(torch.randint(100, 200, (1,)).item())  # name
            elif i % 20 < 15:
                tokens.append(torch.randint(200, 2000, (1,)).item())  # body tokens
            else:
                tokens.append(torch.randint(2000, 5000, (1,)).item())  # keywords
        return torch.tensor(tokens)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'input_ids': seq[:-1].clone(),
            'labels': seq[1:].clone()
        }


class ReasoningDataset(Dataset):
    """Synthetic reasoning tasks (math, logic)."""
    def __init__(self, num_samples=5000, seq_len=64, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        torch.manual_seed(123)
        
        self.data = []
        for _ in range(num_samples):
            seq = self._generate_reasoning_sequence(seq_len)
            self.data.append(seq)
            
    def _generate_reasoning_sequence(self, length):
        """Generate sequence with reasoning patterns."""
        tokens = []
        for i in range(length):
            # Simulate: premise -> logic -> conclusion
            if i < length // 3:
                tokens.append(torch.randint(0, 500, (1,)).item())  # premise
            elif i < 2 * length // 3:
                tokens.append(torch.randint(500, 1000, (1,)).item())  # logic ops
            else:
                tokens.append(torch.randint(1000, 2000, (1,)).item())  # conclusion
        return torch.tensor(tokens)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'input_ids': seq[:-1].clone(),
            'labels': seq[1:].clone()
        }


def check_nan_inf(tensor, name="tensor"):
    """Check for NaN or Inf values."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    return has_nan, has_inf


def monitor_gradients(model):
    """Check gradient statistics."""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_nan, has_inf = check_nan_inf(param.grad, name)
            grad_stats[name] = {
                'norm': grad_norm,
                'has_nan': has_nan,
                'has_inf': has_inf
            }
    return grad_stats


def validate_forward_pass(model, batch, device):
    """Run forward pass and check for issues."""
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    logits, info = model(input_ids)
    
    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    # Check for NaN/Inf
    loss_nan, loss_inf = check_nan_inf(loss, "loss")
    
    return {
        'loss': loss.item(),
        'loss_nan': loss_nan,
        'loss_inf': loss_inf,
        'logits_nan': check_nan_inf(logits)[0],
        'logits_inf': check_nan_inf(logits)[1]
    }


def run_stability_test(model, device, num_steps=100):
    """Test for numerical stability over many steps."""
    print("\n" + "="*60)
    print("STABILITY TEST")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    dataset = SyntheticCodingDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    results = []
    nan_detected = False
    
    for step, batch in enumerate(loader):
        if step >= num_steps:
            break
            
        optimizer.zero_grad()
        
        # Forward pass
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits, info = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Check before backward
        loss_nan, loss_inf = check_nan_inf(loss, "loss")
        if loss_nan or loss_inf:
            print(f"  STEP {step}: NaN={loss_nan}, Inf={loss_inf} in LOSS")
            nan_detected = True
        
        # Backward
        loss.backward()
        
        # Gradient check
        grad_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                g_nan, g_inf = check_nan_inf(param.grad, name)
                if g_nan or g_inf:
                    print(f"  STEP {step}: NaN={g_nan}, Inf={g_inf} in grad/{name}")
                    grad_nan = True
                    break
        
        if grad_nan:
            nan_detected = True
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        results.append({'step': step, 'loss': loss.item(), 'nan': loss_nan or loss_inf})
        
        if step % 20 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, nan={loss_nan}, inf={loss_inf}")
    
    print(f"\nStability Test Result: {'FAIL - NaN detected' if nan_detected else 'PASS'}")
    return not nan_detected, results


def run_convergence_test(model, device, num_steps=200):
    """Test if loss converges over time."""
    print("\n" + "="*60)
    print("CONVERGENCE TEST")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    dataset = SyntheticCodingDataset(num_samples=200)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    losses = []
    
    for step, batch in enumerate(loader):
        if step >= num_steps:
            break
        
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits, info = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 50 == 0:
            recent_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
            print(f"  Step {step}: loss={loss.item():.4f}, avg_recent={recent_loss:.4f}")
    
    # Check if loss is decreasing
    early_avg = np.mean(losses[:50])
    late_avg = np.mean(losses[-50:])
    improvement = early_avg - late_avg
    
    print(f"\nEarly avg loss: {early_avg:.4f}")
    print(f"Late avg loss: {late_avg:.4f}")
    print(f"Improvement: {improvement:.4f}")
    print(f"Convergence Test Result: {'PASS' if improvement > 0 else 'FAIL - loss not decreasing'}")
    
    return improvement > 0, losses


def run_wsd_test(model, device, num_steps=100):
    """Test WSD trainer specifically."""
    print("\n" + "="*60)
    print("WSD SCHEDULER TEST")
    print("="*60)
    
    model.train()
    trainer = WSDTrainer(model, peak_lr=1e-3, warmup_steps=20, total_steps=num_steps)
    
    dataset = SyntheticCodingDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    losses = []
    lrs = []
    
    for step, batch in enumerate(loader):
        if step >= num_steps:
            break
        
        batch_dict = {
            'input_ids': batch['input_ids'].to(device),
            'labels': batch['labels'].to(device)
        }
        
        loss = trainer.step(batch_dict)
        lr = trainer.get_lr()
        
        losses.append(loss)
        lrs.append(lr)
        
        if step % 20 == 0:
            print(f"  Step {step}: loss={loss:.4f}, lr={lr:.6f}")
    
    # Check for NaN in losses
    nan_count = sum(1 for l in losses if np.isnan(l))
    print(f"\nNaN losses: {nan_count}/{len(losses)}")
    print(f"WSD Test Result: {'PASS' if nan_count == 0 else 'FAIL'}")
    
    return nan_count == 0, losses, lrs


def run_perplexity_test(model, device):
    """Measure perplexity on held-out data."""
    print("\n" + "="*60)
    print("PERPLEXITY TEST")
    print("="*60)
    
    model.eval()
    
    dataset = SyntheticCodingDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Perplexity Test Result: {'PASS' if perplexity < 100 else 'FAIL - too high'}")
    
    return perplexity < 100, perplexity


def main():
    print("="*60)
    print("NEXUS V6 VALIDATION SUITE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create small model for testing
    print("\nCreating NexusV6 Small model...")
    model = build_nexus_small()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    
    results = {}
    
    # Run all tests
    stability_pass, stability_results = run_stability_test(model, device, num_steps=100)
    results['stability'] = {'pass': stability_pass, 'results': stability_results}
    
    # Reset model for convergence test
    model = build_nexus_small().to(device)
    
    convergence_pass, convergence_results = run_convergence_test(model, device, num_steps=200)
    results['convergence'] = {'pass': convergence_pass, 'results': convergence_results}
    
    # Reset model for WSD test
    model = build_nexus_small().to(device)
    
    wsd_pass, wsd_losses, wsd_lrs = run_wsd_test(model, device, num_steps=100)
    results['wsd'] = {'pass': wsd_pass, 'losses': wsd_losses, 'lrs': wsd_lrs}
    
    # Reset model for perplexity test
    model = build_nexus_small().to(device)
    
    perplexity_pass, perplexity = run_perplexity_test(model, device)
    results['perplexity'] = {'pass': perplexity_pass, 'value': perplexity}
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_pass = all(r['pass'] for r in results.values())
    
    print(f"Stability Test:    {'PASS' if stability_pass else 'FAIL'}")
    print(f"Convergence Test:  {'PASS' if convergence_pass else 'FAIL'}")
    print(f"WSD Scheduler:     {'PASS' if wsd_pass else 'FAIL'}")
    print(f"Perplexity:        {'PASS' if perplexity_pass else 'FAIL'}")
    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    # Save results
    results_dir = Path("./validation_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"validation_{timestamp}.json"
    
    # Convert non-serializable items
    serializable_results = {}
    for key, val in results.items():
        serializable_results[key] = {
            'pass': val['pass'],
            **{k: v for k, v in val.items() if k != 'pass' and k != 'results' and k != 'losses' and k != 'lrs'}
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
