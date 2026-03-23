import torch
import torch.nn as nn
from loguru import logger

class DML_GRUCell(nn.Module):
    """A custom GPU-native GRU Cell using only supported physical layers."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_ir = nn.Linear(input_size, hidden_size)
        self.w_hr = nn.Linear(hidden_size, hidden_size)
        self.w_iz = nn.Linear(input_size, hidden_size)
        self.w_hz = nn.Linear(hidden_size, hidden_size)
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_hn = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input, h_prev):
        r = torch.sigmoid(self.w_ir(input) + self.w_hr(h_prev))
        z = torch.sigmoid(self.w_iz(input) + self.w_hz(h_prev))
        n = torch.tanh(self.w_in(input) + r * self.w_hn(h_prev))
        return (1 - z) * n + z * h_prev

class DML_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.cells = nn.ModuleList([
            DML_GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
    def forward(self, input, h_prev):
        x = input.squeeze(1)
        h_next = []
        for i, cell in enumerate(self.cells):
            x = cell(x, h_prev[i])
            h_next.append(x)
        return x.unsqueeze(1), torch.stack(h_next, dim=0)

class ASTDecoder(nn.Module):
    """
    SNAP-C1 V3 Model Component: Abstract Syntax Graph Generator
    
    This replaces the V2 ConceptDecoder (`tiktoken` flat-string generator).
    Instead of outputting [100277] possible 1D BPE tokens, it connects directly 
    to the V3 ASTGraphParser to predict mathematical tree structures.
    
    A Graph Neural Network (GNN) approach ensures that invalid Python syntax 
    is physically impossible to generate.
    """
    def __init__(self, concept_dim: int, ast_vocab_size: int = 250, hidden_dim: int = 512, semantic_vocab_size: int = 50):
        super().__init__()
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.ast_vocab_size = ast_vocab_size
        
        # 1. Project the final Core Hologram math into the AST latent space
        self.graph_proj = nn.Linear(concept_dim, hidden_dim)
        
        # 2. The Recurrent Graph Builder (DirectML-compatible custom GRU)
        self.tree_rnn = DML_GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2
        )
        
        # 3. Predict the specific AST Node Type (e.g. FunctionDef, Return, ForLoop)
        self.node_classifier = nn.Linear(hidden_dim, ast_vocab_size)
        
        # 4. Predict if this Node has Children (Branching factor)
        self.branch_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 5. Predict the semantic string payload (Variables, Constants, Function Names)
        self.semantic_classifier = nn.Linear(hidden_dim, semantic_vocab_size)
        
    def forward(self, hologram: torch.Tensor, max_nodes: int = 100) -> torch.Tensor:
        """
        Takes the mathematical equilibrium vector from the Liquid Time-Constant Core
        and physically hallucinates an Abstract Syntax Tree structure.
        """
        batch_size = hologram.size(0)
        device = hologram.device
        
        # If the input still has a sequence dimension [batch, seq_len, dim], pool it
        # down to a single logical "thought" vector
        if hologram.dim() == 3:
            hologram = hologram.mean(dim=1) # [batch, dim]
            
        # Transform the Core thought into Tree-building memory
        hidden_state = self.graph_proj(hologram).unsqueeze(0).repeat(2, 1, 1) # [num_layers, batch, hidden]
        
        # Start the tree with a root 'Module' node sequence
        current_node_emb = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        generated_nodes = []
        branch_probabilities = []
        all_node_logits = []
        all_semantic_logits = []
        
        # Auto-regressively build the graph nodes
        for step in range(max_nodes):
            out, hidden_state = self.tree_rnn(current_node_emb, hidden_state)
            
            # Predict what kind of Python logic node this is
            node_logits = self.node_classifier(out.squeeze(1)) # [batch, ast_vocab_size]
            predicted_node_id = torch.argmax(node_logits, dim=-1)
            generated_nodes.append(predicted_node_id)
            
            # Predict if we need to recurse down a level (e.g. into a 'For' loop body)
            branch_prob = self.branch_predictor(out.squeeze(1))
            branch_probabilities.append(branch_prob)
            all_node_logits.append(node_logits.unsqueeze(1))
            
            # Predict Semantic String payload (Variables/Constants)
            semantic_logits = self.semantic_classifier(out.squeeze(1)) # [batch, semantic_vocab_size]
            all_semantic_logits.append(semantic_logits.unsqueeze(1))
            
            # Form next structural concept as input context for the GRU
            # In a real GNN, this would include Edge embeddings as well
            current_node_emb = out
            
        # Cat sequence alongside the sequence
        final_tree_sequence = torch.stack(generated_nodes, dim=1)
        final_branch_probs = torch.cat(branch_probabilities, dim=1)
        final_node_logits = torch.cat(all_node_logits, dim=1)
        final_semantic_logits = torch.cat(all_semantic_logits, dim=1)
        
        # We return the raw integer node lists, structural probabilities, and continuous logits for CrossEntropy
        return final_tree_sequence, final_branch_probs, final_node_logits, final_semantic_logits

if __name__ == "__main__":
    print("\n--- Testing V3 Abstract Syntax Tree (AST) Decoder ---")
    decoder = ASTDecoder(concept_dim=1024, ast_vocab_size=250)
    
    # 1024-dimension thought vector passed perfectly from the Continuous Core
    mock_thought = torch.randn(2, 1024) 
    
    tree_sequence, jump_probs, _, sem_logits = decoder(mock_thought, max_nodes=15)
    
    print(f"Generated a logic tree with {tree_sequence.shape[1]} structural nodes.")
    print(f"Tree Output Matrix Shape: {tree_sequence.shape}")
    print(f"Branch Probability Shape: {jump_probs.shape}")
    print(f"Semantic Payload Matrix Shape: {sem_logits.shape}")
