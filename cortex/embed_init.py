"""GPT-2 Embedding Initialization for Eidos V1.

Loads GPT-2-small's pretrained token embeddings, maps them to our restricted
vocabulary, and projects to our d_model via PCA. Caches to disk.
"""
import torch
import os
import pickle
from typing import Dict, Tuple, Optional


def load_gpt2_embeddings(bpe_to_local: Dict[int, int], vocab_size: int,
                         d_model: int = 512,
                         cache_dir: Optional[str] = None) -> torch.Tensor:
    """Load GPT-2 embeddings mapped to our restricted vocabulary.
    
    Returns:
        embed_weight: [vocab_size, d_model] tensor for nn.Embedding
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'embed_cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f'gpt2_embed_{vocab_size}v_{d_model}d.pt')

    if os.path.exists(cache_path):
        print(f"  Embed: loaded cached {cache_path}")
        return torch.load(cache_path, map_location='cpu')

    print(f"  Embed: loading GPT-2-small from HuggingFace...")
    from transformers import GPT2Model
    import torch.nn.functional as F

    gpt2 = GPT2Model.from_pretrained('gpt2')
    gpt2_embed = gpt2.wte.weight.data  # [50257, 768]
    del gpt2

    # Build mapping: local_id → GPT-2 BPE token → GPT-2 embedding
    # bpe_to_local maps BPE_ID → local_id
    # We need local_id → BPE_ID (reverse mapping)
    local_to_bpe = {}
    for bpe_id, local_id in bpe_to_local.items():
        local_to_bpe[local_id] = bpe_id

    # Collect embeddings for each local token
    gpt2_vectors = []
    for local_id in range(vocab_size):
        bpe_id = local_to_bpe.get(local_id, None)
        if bpe_id is not None and bpe_id < gpt2_embed.size(0):
            gpt2_vectors.append(gpt2_embed[bpe_id])
        else:
            # Unknown token or PAD — use random init
            gpt2_vectors.append(torch.randn(768) * 0.02)

    gpt2_matrix = torch.stack(gpt2_vectors)  # [V, 768]

    # PCA from 768 → d_model
    print(f"  Embed: PCA {gpt2_matrix.size(1)} -> {d_model}...")
    gpt2_matrix = gpt2_matrix.to(torch.float32)

    # If vocab is tiny (testing), can't SVD to full d_model — use random init
    if vocab_size < d_model:
        embed_weight = torch.randn(vocab_size, d_model) * (1.0 / (d_model ** 0.5))
        n_copy = min(gpt2_matrix.size(1), d_model)
        embed_weight[:, :n_copy] = gpt2_matrix[:, :n_copy]
    else:
        # Center
        mean = gpt2_matrix.mean(dim=0)
        centered = gpt2_matrix - mean

        # SVD
        U, S, V = torch.linalg.svd(centered, full_matrices=False)
        # U: [V, K], S: [K], Vh: [K, 768] where K = min(V, 768)
        # Project to d_model: take first d_model components
        k = min(U.size(1), d_model)
        projected = U[:, :k] @ torch.diag(S[:k])  # [V, k]
        if k < d_model:
            # Pad with random columns
            pad = torch.randn(projected.size(0), d_model - k) * (1.0 / (d_model ** 0.5))
            projected = torch.cat([projected, pad], dim=1)

        # Scale to match expected embedding std
        target_std = 1.0 / (d_model ** 0.5)
        projected = projected * (target_std / (projected.std() + 1e-8))
        embed_weight = projected.to(torch.float32).cpu()

    torch.save(embed_weight, cache_path)
    print(f"  Embed: saved to {cache_path}")
    return embed_weight
