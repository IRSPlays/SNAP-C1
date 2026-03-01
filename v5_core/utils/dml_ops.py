"""
SNAP-C1 V5: DirectML-Safe Operations
=====================================
Battle-tested ops that work on AMD RX 7600 via torch-directml.
Every op here has been verified: forward + backward, no scatter_ crashes.

Banned on DirectML:
  - scatter_, scatter_add_
  - nn.Embedding.backward
  - F.one_hot
  - torch.gather.backward
  - torch.max(dim=).backward (use .detach())
  - aten::_thnn_fused_gru_cell
  - aten::sigmoid (some dims — use StableSigmoid)
  - aten::lerp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# StableSigmoid — avoids AMD aten::sigmoid crash on certain tensor dims
# ---------------------------------------------------------------------------
class StableSigmoid(nn.Module):
    """tanh-based sigmoid: 0.5 * (1 + tanh(0.5 * x)). Proven safe since V3."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.tanh(0.5 * x))


def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Functional version of StableSigmoid."""
    return 0.5 * (1.0 + torch.tanh(0.5 * x))


# ---------------------------------------------------------------------------
# RMSNorm — simpler and faster than LayerNorm, no mean subtraction
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    x * rsqrt(mean(x²) + eps), with learnable scale.
    No scatter, no centering — pure elementwise + reduction.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# SwiGLU FFN — gated feed-forward, used in Resonance Blocks
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.
    gate = silu(W_gate @ x)
    out  = W_down @ (gate * (W_up @ x))
    Pure matmul + elementwise. DirectML safe.
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# ---------------------------------------------------------------------------
# chunked_softmax — for 100K+ vocab dims on DirectML
# ---------------------------------------------------------------------------
def chunked_softmax(x: torch.Tensor, dim: int = -1, chunk_size: int = 20000) -> torch.Tensor:
    """
    DirectML-safe Softmax for 100k+ dimensions.
    Chunks the exp+sum to avoid driver hangs on AMD.
    Only activates on DirectML — CUDA/CPU use F.softmax directly.
    """
    if x.device.type != 'privateuseone':
        return F.softmax(x, dim=dim)

    # Detach max — torch.max(dim=).backward uses scatter_, which DirectML rejects.
    # Subtracting a constant doesn't change softmax gradients.
    max_val = torch.max(x, dim=dim, keepdim=True)[0].detach()
    x_stable = x - max_val

    exp_sum = torch.zeros_like(max_val)
    num_elements = x.shape[dim]

    for start in range(0, num_elements, chunk_size):
        size = min(chunk_size, num_elements - start)
        chunk = x_stable.narrow(dim, start, size)
        exp_sum = exp_sum + torch.exp(chunk).sum(dim=dim, keepdim=True)

    return torch.exp(x_stable) / (exp_sum + 1e-10)


# ---------------------------------------------------------------------------
# DML_GRUCell — manual GRU that avoids aten::_thnn_fused_gru_cell
# ---------------------------------------------------------------------------
class DML_GRUCell(nn.Module):
    """Manual GRU Cell bypassing fused_gru_cell (unsupported on DirectML).
    Optional context gate for pointer-generator attention injection.
    """
    def __init__(self, input_size: int, hidden_size: int, context_size: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.has_context = context_size > 0
        if self.has_context:
            self.weight_ctx = nn.Linear(context_size, hidden_size)

    def forward(self, input_tensor: torch.Tensor, hx: torch.Tensor,
                context_t: torch.Tensor = None) -> torch.Tensor:
        ih = self.weight_ih(input_tensor)
        hh = self.weight_hh(hx)

        i_r, i_z, i_n = ih.chunk(3, dim=-1)
        h_r, h_z, h_n = hh.chunk(3, dim=-1)

        resetgate = stable_sigmoid(i_r + h_r)
        updategate = stable_sigmoid(i_z + h_z)

        ctx_bias = self.weight_ctx(context_t) if (self.has_context and context_t is not None) else 0
        newgate = torch.tanh(i_n + resetgate * h_n + ctx_bias)

        return newgate + updategate * (hx - newgate)


# ---------------------------------------------------------------------------
# Device detection — picks RX 7600 (privateuseone:1)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Auto-detect best available device. Prefers dGPU (privateuseone:1) on AMD."""
    try:
        import torch_directml
        # privateuseone:0 = iGPU, privateuseone:1 = dGPU (RX 7600)
        count = torch_directml.device_count()
        if count >= 2:
            return torch.device("privateuseone:1")
        elif count >= 1:
            return torch.device("privateuseone:0")
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
