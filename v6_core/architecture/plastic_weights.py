"""
SNAP-C1 V6: Plastic Weight Layers
=================================
KEY INNOVATION: Synapses change during inference via Hebbian learning.

Standard models: weights are CONSTANT during inference (dead synapses)
Biological neurons: synapses STRENGTHEN when neurons fire together (Hebbian: "neurons that fire together, wire together")

This implementation modifies weights DURING INFERENCE - weights are alive.

Full PyTorch CPU implementation (no DirectML constraints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EligibilityTrace:
    """
    Tracks correlation between pre-synaptic and post-synaptic activity.
    This is what makes Hebbian learning work in biological brains.
    
    The trace decays over time (forgetting), but is reinforced by repeated co-activation.
    """
    def __init__(self, shape: tuple, decay: float = 0.95, learning_rate: float = 0.01):
        self.decay = decay
        self.lr = learning_rate
        self.shape = shape  # Store shape for later use
        # Trace: correlation between input and output activation
        self.trace = torch.zeros(shape)
    
    def update(self, input_activation: torch.Tensor, output_activation: torch.Tensor):
        """
        Update eligibility trace based on correlation between input and output.
        
        Args:
            input_activation: [B, ..., in_features] or [B, in_features] - pre-synaptic
            output_activation: [B, ..., out_features] or [B, out_features] - post-synaptic
        """
        # Flatten spatial dimensions if needed
        if len(input_activation.shape) > 2:
            input_flat = input_activation.view(-1, self.shape[1])  # [N, in_features]
        else:
            input_flat = input_activation
        
        if len(output_activation.shape) > 2:
            output_flat = output_activation.view(-1, self.shape[0])  # [N, out_features]
        else:
            output_flat = output_activation
        
        # Compute correlation
        batch_corr = torch.zeros(self.shape, device=input_activation.device)
        
        for b in range(input_flat.shape[0]):
            # Outer product: [out, in]
            op = output_flat[b].unsqueeze(-1) * input_flat[b].unsqueeze(0)
            batch_corr = batch_corr + op
        
        batch_corr = batch_corr / max(1, input_flat.shape[0])
        
        # Decay old trace, add new correlation
        self.trace = self.decay * self.trace + (1 - self.decay) * batch_corr.detach()
    
    def get_trace(self) -> torch.Tensor:
        return self.trace
    
    def reset(self):
        self.trace.zero_()


class PlasticLinear(nn.Module):
    """
    Linear layer with LIVE PLASTIC WEIGHTS.
    
    During inference, this layer MODIFIES ITS OWN WEIGHTS based on Hebbian learning.
    - When input and output neurons co-activate, strengthen their connection
    - Weak connections decay over time
    
    This is what makes biological neurons 1000x more efficient than artificial ones.
    
    Key insight: We use no_grad() to modify weights during inference WITHOUT breaking autograd.
    The forward pass uses the MODIFIED weights, but gradients still flow through the computation graph.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 plasticity_rate: float = 0.001, trace_decay: float = 0.95):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.plasticity_rate = plasticity_rate
        self.trace_decay = trace_decay
        
        # Standard learnable weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Eligibility trace for Hebbian updates
        self.eligibility = EligibilityTrace(
            (out_features, in_features),
            decay=trace_decay,
            learning_rate=plasticity_rate
        )
        
        # Initial weight norm for conservation
        self.register_buffer('initial_weight_norm', self.weight.norm().detach())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with IN-PLACE weight modification.
        
        Args:
            x: [B, in_features] - input tensor
        
        Returns:
            output: [B, out_features] - weighted sum
        """
        # Save input for Hebbian update
        input_save = x.detach()
        
        # Standard linear computation (uses CURRENT weights, which may be modified)
        output = F.linear(x, self.weight, self.bias)
        
        # Save output for Hebbian update
        output_save = output.detach()
        
        # Apply Hebbian plasticity AFTER forward
        # This happens in no_grad() so it doesn't break backprop
        # In training: always apply
        # In inference: apply (this is what makes weights "alive")
        with torch.no_grad():
            self._apply_hebbian_update(input_save, output_save)
        
        return output
    
    def _apply_hebbian_update(self, input_act: torch.Tensor, output_act: torch.Tensor):
        """
        Apply Hebbian weight update: Δw ∝ input × output
        
        "When neuron A fires and neuron B is active, the synapse between them strengthens."
        """
        # Update eligibility trace first
        self.eligibility.update(input_act, output_act)
        
        # Get trace - this tells us which connections were recently active together
        trace = self.eligibility.get_trace()
        
        # Hebbian update: Δw = lr * trace
        # This strengthens connections that have been correlated
        delta_w = self.plasticity_rate * trace
        
        # Apply update IN-PLACE to modify the actual weights
        self.weight.data = self.weight.data + delta_w
        
        # Conservation: maintain weight magnitude to prevent runaway growth
        current_norm = self.weight.norm().detach()
        if current_norm > 0:
            self.weight.data = self.weight.data * (self.initial_weight_norm / current_norm)
    
    def reset_plasticity(self):
        """Reset eligibility trace between distinct inferences."""
        self.eligibility.reset()
    
    def get_plasticity_info(self) -> dict:
        """Get info about current plasticity state."""
        return {
            'trace_norm': self.eligibility.trace.norm().item(),
            'weight_norm': self.weight.norm().item(),
            'plasticity_rate': self.plasticity_rate,
        }


class PlasticAttention(nn.Module):
    """
    Multi-head attention with PLASTIC Q, K, V, O projections.
    
    During inference, the attention weights themselves adapt based on what
    patterns are being attended to. This is like the brain's synaptic plasticity
    in visual cortex - patterns that appear frequently get stronger attention pathways.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, plasticity_rate: float = 0.001):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.plasticity_rate = plasticity_rate
        
        # Q, K, V, O projections - ALL plastic
        self.q_proj = PlasticLinear(d_model, d_model, plasticity_rate)
        self.k_proj = PlasticLinear(d_model, d_model, plasticity_rate)
        self.v_proj = PlasticLinear(d_model, d_model, plasticity_rate)
        self.o_proj = PlasticLinear(d_model, d_model, plasticity_rate)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: optional attention mask
        
        Returns:
            output: [B, T, d_model]
        """
        B, T, D = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d_h]
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, H, T, d_h]
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.o_proj(context)
        
        return output
    
    def reset_plasticity(self):
        """Reset all plastic projections."""
        self.q_proj.reset_plasticity()
        self.k_proj.reset_plasticity()
        self.v_proj.reset_plasticity()
        self.o_proj.reset_plasticity()


class PlasticBlock(nn.Module):
    """
    Transformer block with FULLY PLASTIC weights.
    
    All components (attention + FFN) adapt during inference.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = None,
                 plasticity_rate: float = 0.001, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        # Plastic attention
        self.attn = PlasticAttention(d_model, n_heads, plasticity_rate)
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Plastic FFN (SwiGLU style)
        self.w_gate = PlasticLinear(d_model, d_ff, plasticity_rate)
        self.w_up = PlasticLinear(d_model, d_ff, plasticity_rate)
        self.w_down = PlasticLinear(d_ff, d_model, plasticity_rate)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        
        # FFN with residual (SwiGLU-like)
        gate = F.silu(self.w_gate(self.norm2(x)))
        up = self.w_up(self.norm2(x))
        x = x + self.dropout(self.w_down(gate * up))
        
        return x
    
    def reset_plasticity(self):
        self.attn.reset_plasticity()
        self.w_gate.reset_plasticity()
        self.w_up.reset_plasticity()
        self.w_down.reset_plasticity()


class PlasticStack(nn.Module):
    """
    Stack of PLASTIC transformer blocks.
    
    The entire stack is "alive" - every layer modifies its weights during inference.
    """
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int = 8,
                 d_ff: int = None, plasticity_rate: float = 0.001, dropout: float = 0.0):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            PlasticBlock(d_model, n_heads, d_ff, plasticity_rate, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)
    
    def reset_plasticity(self):
        """Reset all blocks - call between distinct inference episodes."""
        for block in self.blocks:
            block.reset_plasticity()
    
    def get_plasticity_stats(self) -> dict:
        """Get statistics about plasticity across all layers."""
        total_trace_norm = 0
        total_weight_norm = 0
        for block in self.blocks:
            total_trace_norm += block.attn.q_proj.get_plasticity_info()['trace_norm']
            total_weight_norm += block.attn.q_proj.get_plasticity_info()['weight_norm']
        return {
            'avg_trace_norm': total_trace_norm / len(self.blocks),
            'avg_weight_norm': total_weight_norm / len(self.blocks),
        }


def convert_to_plastic(model: nn.Module, plasticity_rate: float = 0.001) -> nn.Module:
    """
    Convert a standard nn.Module to use plastic weights.
    
    Replaces all nn.Linear layers with PlasticLinear.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # Replace with plastic version
            setattr(model, name, PlasticLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                plasticity_rate=plasticity_rate
            ))
            print(f"  Converted {name}: {child.in_features} → {child.out_features}")
        elif len(list(child.children())) > 0:
            # Recurse into children
            convert_to_plastic(child, plasticity_rate)
    
    return model
