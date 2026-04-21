"""
NEXUS V7: Simplified Working Architecture
=========================================

Stripped down to essentials that actually work.
No innovations until we verify it learns.

Architecture:
- Embedding + Positional Encoding
- Flash Attention layers (standard, proven)
- Simple FFN layers
- Proper training with validation
- Weight decay and regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization
    
    Instead of computing mean and variance, only RMS is computed.
    Faster than LayerNorm, similar or better performance.
    
    Reference: "Root Mean Square Layer Normalization" (Zhang & Sablayrolles, 2019)
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: normalize by sqrt(mean(x²)) instead of mean(x) and var(x)
        # x = w * (x / rms(x))
        # where rms(x) = sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class FlashAttention(nn.Module):
    """
    Flash Attention with GQA (Grouped Query Attention) and RoPE.
    
    Features:
    - Real Flash Attention via F.scaled_dot_product_attention
    - Grouped Query Attention: fewer KV heads than Q heads
    - RoPE (Rotary Positional Embedding) for length generalization
    
    Reference: Flash Attention (Dao et al., 2022)
    Reference: GQA (Ainslie et al., 2023)
    Reference: RoPE (Su et al., 2022)
    """
    
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int = None,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads or num_q_heads  # Default to MHA
        self.d_head = d_model // num_q_heads
        self.max_seq_len = max_seq_len
        
        assert d_model % num_q_heads == 0, "d_model must be divisible by num_q_heads"
        assert num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        
        self.num_groups = num_q_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)  # Q: full size
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.d_head)  # K: fewer heads
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.d_head)  # V: fewer heads
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE
        self._init_rope()
    
    def _init_rope(self):
        """Initialize RoPE rotation angles."""
        # RoPE works on pairs of dimensions
        # For each position, we rotate q and k by angles based on position
        
        # Build frequency bands: 1 / (10000^(2i/d_head)) for i in [0, d_head/2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        
        # Create position indices
        positions = torch.arange(self.max_seq_len)
        
        # Compute angles: position * inv_freq
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Stack to create (cos, sin) pairs
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Register as buffer (not trainable, but saved with model)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply RoPE to tensor x.
        
        x: (B, num_heads, seq_len, d_head)
        Returns: x with RoPE applied
        """
        # Handle dynamic sequence length - extend buffers if needed
        if seq_len > self.rope_cos.shape[0]:
            # Recompute for longer sequence on the same device as x
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2, device=x.device).float() / self.d_head))
            positions = torch.arange(seq_len, device=x.device)
            angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        else:
            cos = self.rope_cos[:seq_len].to(x.device)
            sin = self.rope_sin[:seq_len].to(x.device)
        
        # Reshape for broadcasting
        # x[:, :, t] has shape (B, num_heads, d_head)
        # We rotate pairs of dimensions: (d_head//2) pairs
        
        # Split into even and odd indices
        x1 = x[..., ::2]  # dimensions 0, 2, 4, ...
        x2 = x[..., 1::2]  # dimensions 1, 3, 5, ...
        
        # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        # For each position t: rotate by angle[t]
        
        # Reshape cos, sin for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head/2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head/2)
        
        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Interleave back
        # Create empty tensor and fill in alternating positions
        x_rot = torch.zeros_like(x)
        x_rot[..., ::2] = x1_rot
        x_rot[..., 1::2] = x2_rot
        
        return x_rot
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, num_kv_heads * d_head)
        v = self.v_proj(x)  # (B, T, num_kv_heads * d_head)
        
        # Reshape Q: (B, T, num_q_heads, d_head) -> (B, num_q_heads, T, d_head)
        q = q.view(B, T, self.num_q_heads, self.d_head).transpose(1, 2)
        
        # Reshape K, V: (B, T, num_kv_heads, d_head) -> (B, num_kv_heads, T, d_head)
        k = k.view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)
        
        # Handle GQA: repeat K,V for each Q group if num_kv_heads < num_q_heads
        if self.num_q_heads != self.num_kv_heads:
            # Repeat K,V along Q dimension: (B, num_kv_heads, T, d) -> (B, num_q_heads, T, d)
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        
        # REAL Flash Attention - single line, hardware accelerated
        # Note: When is_causal=True, PyTorch creates causal mask automatically
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0 if not self.training else 0.1
        )
        
        # Reshape back: (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.out_proj(attn_output)


class NexusBlock(nn.Module):
    """Transformer block with GQA, RMSNorm, and SwiGLU."""
    
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        # RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, num_q_heads, num_kv_heads, max_seq_len)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(d_model)
        # SwiGLU FFN
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.dropout2 = nn.Dropout(dropout)
    
    def ffn_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation function.
        
        FFN_SwiGLU(x) = (Silu(W1(x)) * W3(x)) @ W2
        
        Where:
        - W1: d_model → d_ffn
        - W3: d_model → d_ffn
        - W2: d_ffn → d_model
        """
        # Gate path and up path
        gate = self.w1(x)  # [B, T, d_ffn]
        up = self.w3(x)  # [B, T, d_ffn]
        
        # Element-wise multiply with silu activation on gate
        intermediate = F.silu(gate) * up  # [B, T, d_ffn]
        
        # Down projection
        return intermediate @ self.w2.weight.T  # [B, T, d_model]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (more stable)
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        x = x + self.dropout2(self.ffn_swiglu(self.norm2(x)))
        return x


class NexusV7(nn.Module):
    """
    NEXUS V7: Simplified working architecture.
    
    Just a standard transformer with Flash Attention.
    No MoE, no SSM, no Hebbian, no Tree Evolution.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 384,
        num_layers: int = 8,
        num_q_heads: int = 6,
        num_kv_heads: int = 2,
        d_ffn: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Embeddings (NO positional encoding - RoPE handles it inside FlashAttention!)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Transformer blocks with GQA, RMSNorm, SwiGLU
        self.layers = nn.ModuleList([
            NexusBlock(d_model, num_q_heads, num_kv_heads, d_ffn, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Output with RMSNorm
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive decoding."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) mask for padding
            labels: (B, T) labels for loss computation
            return_loss: Whether to compute loss
        
        Returns:
            dict with 'logits', 'loss', and other metrics
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        mask = self.create_causal_mask(T, device)
        
        # Handle padding mask
        if attention_mask is not None:
            # Convert padding mask to attention mask format
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask = mask * padding_mask
        
        # Embeddings (NO positional encoding - RoPE handles it inside FlashAttention!)
        x = self.dropout(self.embedding(input_ids))
        
        # Transformer layers (RoPE is applied inside FlashAttention)
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        result = {'logits': logits}
        
        # Compute loss
        if labels is not None and return_loss:
            # Shift for teacher forcing: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if needed
            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, -2048:]
            
            # Forward
            result = self.forward(input_ids, return_loss=False)
            logits = result['logits']
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def build_nexus_v7_tiny():
    """Tiny model with GQA (6 Q heads, 2 KV heads)."""
    return NexusV7(
        vocab_size=32000,
        d_model=256,
        num_layers=6,
        num_q_heads=4,
        num_kv_heads=2,
        d_ffn=1024,
        dropout=0.1
    )


def build_nexus_v7_small():
    """Small model with GQA (6 Q heads, 2 KV heads)."""
    return NexusV7(
        vocab_size=32000,
        d_model=384,
        num_layers=8,
        num_q_heads=6,
        num_kv_heads=2,
        d_ffn=1536,
        dropout=0.1
    )


def build_nexus_v7_medium():
    """Medium model with GQA (8 Q heads, 2 KV heads)."""
    return NexusV7(
        vocab_size=32000,
        d_model=512,
        num_layers=12,
        num_q_heads=8,
        num_kv_heads=2,
        d_ffn=2048,
        dropout=0.1
    )


class SimpleTrainer:
    """
    Simple trainer with validation to detect memorization.
    
    Includes:
    - Gradient clipping
    - Warmup + cosine LR schedule
    - Validation monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        learning_rate: float = 1e-3,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step = 0
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Create LR scheduler with warmup + cosine decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )
        
        # Warmup is handled separately in train_step
        self.base_lr = learning_rate
    
    def get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.step + 1) / self.warmup_steps
        else:
            # Cosine decay
            return self.scheduler.get_last_lr()[0]
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        # Apply warmup LR
        if self.step < self.warmup_steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        self.optimizer.zero_grad()
        result = self.model(input_ids, labels=labels)
        loss = result['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Step scheduler after warmup
        if self.step >= self.warmup_steps:
            self.scheduler.step()
        
        self.step += 1
        return loss.item()
    
    @torch.no_grad()
    def validate(self, val_batch: Dict[str, torch.Tensor]) -> float:
        """Validation step - detects memorization."""
        self.model.eval()
        
        input_ids = val_batch['input_ids'].to(self.device)
        labels = val_batch.get('labels', input_ids).to(self.device)
        
        result = self.model(input_ids, labels=labels)
        return result['loss'].item()
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        val_every: int = 100,
        print_every: int = 10
    ) -> Dict:
        """Full training loop with validation."""
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': []
        }
        
        step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            for batch in train_loader:
                train_loss = self.train_step(batch)
                history['train_loss'].append((step, train_loss))
                history['train_perplexity'].append((step, math.exp(train_loss)))
                
                # Validation
                if step % val_every == 0:
                    val_batch = next(iter(val_loader))
                    val_loss = self.validate(val_batch)
                    history['val_loss'].append((step, val_loss))
                    history['val_perplexity'].append((step, math.exp(val_loss)))
                    
                    # Check for memorization
                    if val_loss > train_loss * 1.5:
                        print(f"[WARN] Possible memorization! train={train_loss:.4f}, val={val_loss:.4f}")
                    
                    # Save best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), 'nexus_v7_best.pt')
                    
                    print(f"Epoch {epoch} Step {step}: train={train_loss:.4f} ({math.exp(train_loss):.1f}), val={val_loss:.4f} ({math.exp(val_loss):.1f})")
                
                if step % print_every == 0:
                    print(f"Step {step}: loss={train_loss:.4f}")
                
                step += 1
        
        return history


if __name__ == '__main__':
    print("Testing NEXUS V7...")
    
    model = build_nexus_v7_tiny()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,} ({params/1e6:.1f}M)")
    
    # Test forward
    x = torch.randint(0, 32000, (4, 64))
    result = model(x, labels=x)
    print(f"Forward pass OK, loss={result['loss'].item():.4f}")
    
    # Test backward
    result['loss'].backward()
    print("Backward pass OK")
    
    # Test generation
    model.eval()
    gen_ids = model.generate(x[:, :10], max_new_tokens=20)
    print(f"Generation OK, shape={gen_ids.shape}")
    
    print("\nNEXUS V7 test PASSED!")
