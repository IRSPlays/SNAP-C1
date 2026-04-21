"""
SNAP-C1 V6: State Space Hopper
===============================
KEY INNOVATION: JUMP to relevant memory states, don't crawl.

Human brain: "I remember the answer before I finish reading the question."
Standard LLM: Sequential processing of all tokens.
V6 SSH: Content-addressable memory jumping.

How it works:
1. Learn canonical "states" that represent common patterns
2. Given a query, compute similarity to all states
3. JUMP directly to most similar state(s)
4. Aggregate hop results

This is O(1) retrieval vs O(n) attention over context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_core.architecture.dml_ops import RMSNorm


class StateMemory(nn.Module):
    """
    Learnable state memory - stores canonical patterns.
    
    Unlike attention which computes O(n × d) over all tokens,
    we store N learned states and retrieve in O(1).
    
    Each state is a d_model dimensional vector.
    """
    
    def __init__(self, d_model: int = 1024, n_states: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        
        # Learned state vectors
        self.states = nn.Parameter(
            torch.randn(n_states, d_model) * 0.02
        )
        
        # State importance weights (learned)
        self.state_weights = nn.Parameter(torch.ones(n_states))
        
        # Normalize states for cosine similarity
        self.register_buffer('states_norm', 
                           torch.ones(n_states))
    
    def forward(self):
        """Return normalized states for retrieval."""
        # L2 normalize for cosine similarity
        norm = self.states.norm(dim=-1, keepdim=True)
        normalized = self.states / (norm + 1e-8)
        return normalized
    
    def get_states(self):
        """Return raw states + weights."""
        return self.states, torch.softmax(self.state_weights, dim=0)


class StateHopper(nn.Module):
    """
    Content-addressable memory hopper.
    
    Given a query, find most similar states and "hop" to them.
    Instead of sequential processing, we JUMP to relevant patterns.
    """
    
    def __init__(self, d_model: int = 1024, n_states: int = 512, 
                 n_hops: int = 3, n_retrieve: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        self.n_hops = n_hops
        self.n_retrieve = n_retrieve
        
        # State memory
        self.state_memory = StateMemory(d_model, n_states)
        
        # Query encoder: projects hidden state to retrieval space
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Hop aggregator: combines multiple hops
        self.hop_aggregator = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Gating: how much to incorporate retrieved state
        self.hop_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output norm
        self.norm = RMSNorm(d_model)
    
    def retrieve(self, query: torch.Tensor) -> tuple:
        """
        Retrieve most similar states to query.
        
        Args:
            query: [B, d_model] or [B, T, d_model]
        
        Returns:
            states: [B, k, d_model] - top k similar states
            weights: [B, k] - similarity weights
            indices: [B, k] - state indices
        """
        if len(query.shape) == 3:
            query = query.mean(dim=1)  # [B, d_model]
        
        # Encode query
        q = self.query_encoder(query)  # [B, d_model]
        
        # Get normalized states
        states = self.state_memory()  # [N, d_model]
        
        # Cosine similarity: [B, d] · [d, N] → [B, N]
        # Reshape: q [B, 1, d], states [1, N, d]
        q_expanded = q.unsqueeze(1)  # [B, 1, d_model]
        states_expanded = states.unsqueeze(0)  # [1, N, d_model]
        
        # Cosine similarity
        similarity = (q_expanded * states_expanded).sum(dim=-1)  # [B, N]
        
        # Get top-k
        weights, indices = torch.topk(similarity, k=self.n_retrieve, dim=-1)  # [B, k]
        
        # Softmax over weights
        weights = F.softmax(weights, dim=-1)
        
        # Gather states
        retrieved_states = states[indices]  # [B, k, d_model]
        
        return retrieved_states, weights, indices
    
    def forward(self, hidden: torch.Tensor, 
                context: torch.Tensor = None) -> torch.Tensor:
        """
        Hop to relevant states and aggregate.
        
        Args:
            hidden: [B, T, d_model] - current hidden states
            context: [B, T_ctx, d_model] - optional context
        
        Returns:
            hopped: [B, T, d_model] - hidden with state information
        """
        B, T, D = hidden.shape
        
        # Query from last position or mean
        if context is not None:
            query = context.mean(dim=1)  # [B, d_model]
        else:
            query = hidden.mean(dim=1)  # [B, d_model]
        
        # Retrieve states
        states, weights, indices = self.retrieve(query)  # [B, k, d], [B, k], [B, k]
        
        # Weighted combination of retrieved states
        retrieved = (states * weights.unsqueeze(-1)).sum(dim=1)  # [B, d_model]
        
        # Hop aggregation via GRU
        # Input: [B, n_hops, d_model], hidden: [2, B, d_model]
        hop_inputs = retrieved.unsqueeze(1).expand(-1, self.n_hops, -1)  # [B, n_hops, d_model]
        
        # Process hops
        gru_out, _ = self.hop_aggregator(hop_inputs)  # [B, n_hops, d_model]
        
        # Use last hop output
        final_hop = gru_out[:, -1, :]  # [B, d_model]
        
        # Gated incorporation: how much to trust retrieved vs original
        combined = torch.cat([hidden[:, -1, :], final_hop], dim=-1)  # [B, 2*d_model]
        gate = self.hop_gate(combined)  # [B, d_model]
        
        # Apply gate
        output = gate * final_hop + (1 - gate) * hidden[:, -1, :]
        
        # Expand to full sequence
        output = output.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model]
        
        return self.norm(output)


class StateSpaceHopper(nn.Module):
    """
    Full State Space Hopper module.
    
    Integrates with the model to provide O(1) memory retrieval.
    
    Usage:
        # After resonance blocks
        hopped = state_hopper(resonance_output, context)
        # Use hopped for action decoder
    """
    
    def __init__(self, d_model: int = 1024, n_states: int = 512,
                 n_hops: int = 3, n_retrieve: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        
        self.hopper = StateHopper(
            d_model=d_model, 
            n_states=n_states,
            n_hops=n_hops,
            n_retrieve=n_retrieve
        )
        
        # Learnable skip connection around hopper
        self.skip_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden: [B, T, d_model]
            context: [B, T_ctx, d_model] or None

        Returns:
            output: [B, T, d_model] - possibly hopped hidden state
        """
        B, T, D = hidden.shape
        
        # Decide whether to hop or skip
        skip_prob = self.skip_gate(hidden.mean(dim=1))  # [B, 1]
        
        # Deterministic during eval, stochastic during train
        if self.training:
            skip = torch.bernoulli(1 - skip_prob).bool()
        else:
            skip = skip_prob < 0.5
        
        # Always compute hopper output (for training signal)
        hopped = self.hopper(hidden, context)
        
        # Apply skip or hopped
        skip_3d = skip.view(B, 1, 1).expand(-1, T, D)
        output = torch.where(skip_3d, hidden, hopped)
        
        return output


class AssociativeMemory(nn.Module):
    """
    Key-value associative memory.
    
    Unlike content-addressable retrieval (StateSpaceHopper),
    this learns KEY → VALUE mappings.
    
    Useful for:
    - Remembering tool argument patterns
    - Storing common solution patterns
    - Caching frequent operations
    """
    
    def __init__(self, d_model: int = 1024, n_slots: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        
        # Key memory
        self.keys = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        # Value memory
        self.values = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        
        # Update gate
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve value given key query."""
        # Cosine similarity
        norm_query = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        norm_keys = self.keys / (self.keys.norm(dim=-1, keepdim=True) + 1e-8)
        
        sim = torch.matmul(norm_query, norm_keys.T)  # [B, n_slots]
        
        # Softmax over slots
        weights = F.softmax(sim, dim=-1)  # [B, n_slots]
        
        # Weighted sum of values
        retrieved = torch.matmul(weights, self.values)  # [B, d_model]
        
        return retrieved
    
    def update(self, key: torch.Tensor, value: torch.Tensor):
        """Update a memory slot with new key-value pair."""
        # Find least-used slot (simplified replacement)
        with torch.no_grad():
            # In practice, would use usage counters
            slot_idx = torch.randint(0, self.n_slots, (1,)).item()
            self.keys[slot_idx] = key.detach()
            self.values[slot_idx] = value.detach()
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.retrieve(query)


def integrate_hopper(model: nn.Module, d_model: int = 1024,
                     n_states: int = 512) -> nn.Module:
    """
    Integrate StateSpaceHopper into an existing model.
    
    Adds hopper after the resonance stack for O(1) memory retrieval.
    """
    model.state_hopper = StateSpaceHopper(
        d_model=d_model,
        n_states=n_states,
        n_hops=3,
        n_retrieve=8
    )
    
    # Hook to modify forward
    original_forward = model.forward_agent
    
    def hooked_forward(token_ids, type_ids=None):
        # Run encoder
        context = model.encoder(token_ids, type_ids)
        slot_token_ids = model._build_slot_token_ids(token_ids)
        
        # Resonance
        hidden = model.resonance(context, causal=False)
        
        # NEW: State hopping
        hidden = model.state_hopper(hidden, context)
        
        # Action decision
        action = model.action_decoder(hidden, context, slot_token_ids)
        
        # Outcome prediction
        outcome = model.outcome_predictor(action['hidden'], action['tool_id'])
        
        return {
            'tool_id': action['tool_id'],
            'tool_logits': action['tool_logits'],
            'confidence': action['confidence'],
            'p_success': outcome['p_success'],
            'outcome_logit': outcome['outcome_logits'],
            'hidden': action['hidden'],
            'context': context,
            'slot_token_ids': slot_token_ids,
            'think_steps': 0,
        }
    
    model.forward_agent = hooked_forward
    return model
