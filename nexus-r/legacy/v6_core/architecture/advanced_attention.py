"""
SNAP-C1 V6: Advanced Attention Mechanisms
==========================================
Implements modern attention innovations from the LLM Architecture Gallery:

1. QK-Norm - Normalizes Q and K before attention for training stability
2. GQA - Grouped Query Attention reduces KV cache by ~8x
3. DeltaNet - Linear attention hybrid for long contexts
4. MTP - Multi-Token Prediction for 2-4x faster generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QKNorm(nn.Module):
    """QK-Normalization: Normalize Query and Key before attention."""
    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.q_norm = nn.LayerNorm(d_model // n_heads, eps=eps)
        self.k_norm = nn.LayerNorm(d_model // n_heads, eps=eps)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> tuple:
        B, H, T, d = Q.shape
        Q = Q.transpose(1, 2).reshape(B * T * H, d)
        K = K.transpose(1, 2).reshape(B * T * H, d)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        Q = Q.reshape(B, T, H, d).transpose(1, 2)
        K = K.reshape(B, T, H, d).transpose(1, 2)
        return Q, K


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) - Used by most modern models."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None,
                 qk_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 8)
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.qk_norm = QKNorm(d_model, n_heads) if qk_norm else None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        
        Q = self.q_proj(x)
        KV = self.kv_proj(x)
        
        Q = Q.view(B, T, self.n_heads, self.head_dim)
        KV = KV.view(B, T, 2, self.n_kv_heads, self.head_dim)
        K = KV[:, :, 0]
        V = KV[:, :, 1]
        
        Q = Q.transpose(1, 2)  # [B, T, H, d] -> [B, H, T, d]
        K = K.transpose(1, 2)  # [B, T, n_kv, d] -> [B, n_kv, T, d]
        V = V.transpose(1, 2)  # [B, T, n_kv, d] -> [B, n_kv, T, d]
        
        if self.n_kv_heads < self.n_heads:
            num_groups = self.n_heads // self.n_kv_heads
            K = K.unsqueeze(1).expand(B, num_groups, self.n_kv_heads, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
            V = V.unsqueeze(1).expand(B, num_groups, self.n_kv_heads, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        
        if self.qk_norm is not None:
            Q, K = self.qk_norm(Q, K)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.o_proj(out)


class DeltaNetAttention(nn.Module):
    """DeltaNet - Linear attention variant with recurrences."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.delta = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        h = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device)
        all_outputs = []
        
        for t in range(T):
            Q_t = Q[:, t].reshape(B * self.n_heads, self.head_dim)
            K_t = K[:, t].reshape(B * self.n_heads, self.head_dim)
            V_t = V[:, t].reshape(B, self.n_heads, self.head_dim)
            
            gate = torch.sigmoid(torch.sum(Q_t * K_t, dim=-1, keepdim=True) / (self.head_dim ** 0.5))
            gate = gate.view(B, self.n_heads, 1, 1)
            
            qk_outer = torch.einsum('bd,bh->bhd', Q_t, K_t).view(B, self.n_heads, self.head_dim, self.head_dim)
            h = gate * h + (1 - gate) * qk_outer
            
            out_t = torch.einsum('bhdd,bhd->bhd', h, V_t)
            all_outputs.append(out_t)
        
        out = torch.stack(all_outputs, dim=2)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        
        return self.norm(self.o_proj(out))


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction (MTP) - Predicts N tokens simultaneously."""
    def __init__(self, d_model: int, n_tokens: int = 4, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.head_dim = d_model // n_heads
        
        self.mtp_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_tokens)
        ])
        self.pred_heads = nn.ModuleList([
            nn.Linear(d_model, d_model // n_tokens) for _ in range(n_tokens)
        ])
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden: torch.Tensor, context: torch.Tensor = None) -> tuple:
        B = hidden.shape[0]
        preds = []
        for i, (proj, head) in enumerate(zip(self.mtp_proj, self.pred_heads)):
            shifted = torch.roll(hidden, shifts=-i-1, dims=1)
            proj_out = proj(shifted)
            pred = head(proj_out)
            preds.append(pred)
        
        predictions = torch.stack(preds, dim=1)
        all_hidden = torch.cat([hidden] + preds, dim=-1)
        confidence = self.confidence_head(all_hidden.mean(dim=1))
        
        return predictions, confidence
    
    def decode_speculative(self, hidden: torch.Tensor, vocab_size: int) -> tuple:
        preds, confidence = self.forward(hidden, None)
        tokens = []
        for pred in preds:
            token = pred.argmax(dim=-1)
            tokens.append(token[:, -1])
        tokens = torch.stack(tokens, dim=1)
        return tokens, confidence


class HybridAttention(nn.Module):
    """Hybrid attention combining sliding window + DeltaNet."""
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 128, use_delta: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.window_attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads=max(1, n_heads // 4), qk_norm=True
        )
        self.delta_attn = DeltaNetAttention(d_model, n_heads) if use_delta else None
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        local = self.window_attn(x, mask)
        if self.delta_attn is not None:
            global_out = self.delta_attn(x)
        else:
            global_out = local
        combined = torch.cat([local, global_out], dim=-1)
        gate = self.gate(combined)
        out = gate * local + (1 - gate) * global_out
        return out


class SlidingWindowGQA(nn.Module):
    """Sliding Window Attention with GQA for local dependencies."""
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 128, n_kv_heads: int = None):
        super().__init__()
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, qk_norm=True
        )
        self.window_size = window_size
        self.mask_cache = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if T not in self.mask_cache:
            rows = torch.arange(T, device=x.device)
            cols = torch.arange(T, device=x.device)
            diff = rows.unsqueeze(1) - cols.unsqueeze(0)
            mask = (diff >= 0) & (diff < self.window_size)
            self.mask_cache[T] = mask
        mask = self.mask_cache[T]
        return self.attn(x, mask)


def apply_gqa_to_model(model: nn.Module, n_kv_heads: int = 2) -> nn.Module:
    """Convert model's attention to GQA."""
    for name, child in model.named_children():
        if isinstance(child, nn.MultiheadAttention):
            d_model = child.embed_dim
            n_heads = child.num_heads
            setattr(model, name, GroupedQueryAttention(
                d_model, n_heads, n_kv_heads=n_kv_heads, qk_norm=True
            ))
        else:
            apply_gqa_to_model(child, n_kv_heads)
    return model
