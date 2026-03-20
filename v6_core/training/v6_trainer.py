"""
SNAP-C1 V6: Training Loop
===========================
The "Alive Protocol" training for V6.

Key difference from V5:
1. Plastic weights are updated during inference (Hebbian)
2. Between training steps, weights are consolidated
3. Dynamic layer skip is trained via auxiliary loss
4. Self-verification is trained via reinforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for V6 training."""
    d_model: int = 1024
    n_blocks: int = 8
    n_heads: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 4
    max_seq_len: int = 2048
    plasticity_rate: float = 0.001
    skip_loss_weight: float = 0.1  # Weight for skip rate regularization
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    checkpoint_every: int = 1000
    eval_every: int = 500


class V6Trainer:
    """
    V6 Alive Protocol Trainer.
    
    Training procedure:
    1. Forward pass through model (plastic weights modify during inference)
    2. Compute losses (LM loss + skip regularization + verification loss)
    3. Backward pass for gradients
    4. Optimizer step
    5. Reset plasticity traces between batches
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
    def train_step(self, batch: Dict) -> Dict:
        """
        Single training step.
        
        Args:
            batch: dict with 'token_ids', 'type_ids', 'labels'
        
        Returns:
            dict with losses and metrics
        """
        self.model.train()
        
        # Reset plasticity traces at start of each step
        self._reset_plasticity()
        
        # Forward pass
        token_ids = batch['token_ids']
        type_ids = batch.get('type_ids')
        labels = batch.get('labels')
        
        result = self.model.forward_pretrain(token_ids, type_ids, labels)
        lm_loss = result['loss']
        
        # Skip rate regularization
        # We want to encourage ~40% skip rate (optimal efficiency)
        skip_rate = self.model.get_skip_rate() if hasattr(self.model, 'get_skip_rate') else 0.0
        target_skip = 0.4
        skip_reg_loss = self.config.skip_loss_weight * (skip_rate - target_skip) ** 2
        
        # Total loss
        total_loss = lm_loss + skip_reg_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update step
        self.step += 1
        
        # Record metrics
        metrics = {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'skip_reg_loss': skip_reg_loss.item(),
            'skip_rate': skip_rate,
            'lr': self.scheduler.get_last_lr()[0],
        }
        
        self.loss_history.append(metrics)
        
        return metrics
    
    def _reset_plasticity(self):
        """Reset plasticity traces between training steps."""
        if hasattr(self.model, 'resonance'):
            for block in self.model.resonance.blocks:
                if hasattr(block, 'skip_router'):
                    continue  # Skip routers don't have plasticity
                # For plastic components
                if hasattr(block, 'attn'):
                    for proj in [block.attn.q_proj, block.attn.k_proj, 
                                  block.attn.v_proj, block.attn.o_proj]:
                        if hasattr(proj, 'reset_plasticity'):
                            proj.reset_plasticity()
    
    def train_loop(self, train_loader: DataLoader, 
                   num_steps: int,
                   eval_fn: Optional[Callable] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader for training data
            num_steps: Number of training steps
            eval_fn: Optional evaluation function called every eval_every steps
        
        Returns:
            dict with final metrics
        """
        print(f"Starting V6 training for {num_steps} steps...")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"LR: {self.config.learning_rate}, Batch: {self.config.batch_size}")
        print()
        
        self.model.train()
        iter_loader = iter(train_loader)
        
        for step in range(num_steps):
            # Get batch
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
            
            # Move to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = self.train_step(batch)
            
            # Logging
            if step % 10 == 0:
                print(f"Step {step}: loss={metrics['loss']:.4f}, "
                      f"skip_rate={metrics['skip_rate']:.2%}, "
                      f"lr={metrics['lr']:.2e}")
            
            # Checkpointing
            if step % self.config.checkpoint_every == 0 and step > 0:
                self._save_checkpoint(f"step_{step}")
            
            # Evaluation
            if eval_fn and step % self.config.eval_every == 0 and step > 0:
                eval_metrics = eval_fn(self.model)
                print(f"Eval: {eval_metrics}")
                self.model.train()
        
        print(f"\nTraining complete! {num_steps} steps.")
        return {'final_step': step, 'loss_history': self.loss_history}
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = f"v6_checkpoint_{name}.pt"
        torch.save({
            'step': self.step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, path)
        print(f"Checkpoint saved: {path}")


class AliveProtocolLoss(nn.Module):
    """
    Combined loss for V6 "Alive Protocol".
    
    Components:
    1. Language modeling loss (next token prediction)
    2. Skip regularization (encourage ~40% skip rate)
    3. Verification loss (encourage correct self-verification)
    """
    
    def __init__(self, skip_weight: float = 0.1, verify_weight: float = 0.1):
        super().__init__()
        self.skip_weight = skip_weight
        self.verify_weight = verify_weight
    
    def forward(self, lm_logits, labels, skip_rate, verify_probs=None, verify_targets=None):
        """
        Args:
            lm_logits: [B, T, V] language model logits
            labels: [B, T] token labels
            skip_rate: float current skip rate
            verify_probs: [B, 3] verification probs (optional)
            verify_targets: [B] verification targets (optional)
        """
        # LM loss
        lm_loss = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )
        
        # Skip regularization
        target_skip = 0.4
        skip_loss = self.skip_weight * (skip_rate - target_skip) ** 2
        
        # Verification loss
        verify_loss = 0.0
        if verify_probs is not None and verify_targets is not None:
            verify_loss = self.verify_weight * F.cross_entropy(
                verify_probs, verify_targets
            )
        
        total = lm_loss + skip_loss + verify_loss
        
        return total, {
            'lm_loss': lm_loss.item(),
            'skip_loss': skip_loss.item(),
            'verify_loss': verify_loss.item(),
        }
