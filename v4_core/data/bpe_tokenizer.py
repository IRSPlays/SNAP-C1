import sys
import os
import tiktoken
import torch
import torch.nn as nn
from loguru import logger
from typing import List, Dict, Any, Tuple

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Steal the perfectly working geometrical tree parser Native from V3
from v3_core.data.ast_parser import ASTGraphParser

class HybridTokenDecoder:
    """
    SNAP-C1 V4: Hybrid Semantic Tokenizer (BPE + AST)
    
    This replaces V3's hardcoded 1000-variable integer dictionary.
    Instead of predicting an entire complex framework object 
    (like `DjangoSQLDatabaseRouter`) as a single ID, this class acts as a 
    bridge.
    
    1. It delegates geometric syntax (FunctionDef, Return, Assign) to V3's AST Parser.
    2. It intercepts Semantic payloads (Name, Constant) and shreds them into 
       Byte-Pair Encoded (BPE) sub-tokens using OpenAI's tiktoken.
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.bpe = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.bpe.n_vocab
        self.ast_geometry_parser = ASTGraphParser()
        
        # We need special tokens to signal when the BPE sub-network 
        # should start and stop generating a variable string.
        self.BOS_TOKEN = self.vocab_size # Begin Variable String
        self.EOS_TOKEN = self.vocab_size + 1 # End Variable String
        self.extended_vocab_size = self.vocab_size + 2
        
        logger.info(f"V4 Hybrid Tokenizer Online. BPE Vocabulary size: {self.extended_vocab_size}")
        
    def encode_ast_with_bpe(self, code_string: str) -> Dict[str, Any]:
        """
        Takes raw Python code and generates the V4 Mathematical Graph.
        Returns:
            - "nodes": The geometric shape (from V3)
            - "edges": The routing paths
            - "bpe_payloads": A dictionary mapping Node IDs to their sequence of BPE tokens.
        """
        # 1. Parse the flawless geometry using the legacy V3 engine
        base_graph = self.ast_geometry_parser.parse_to_graph(code_string)
        
        bpe_payloads = {}
        
        # 2. Iterate through the graph and intercept textual payloads
        for node in base_graph["nodes"]:
            # Let's say the node is an ast.Name with the value 'HttpResponseRedirect'
            raw_string_value = node.get("value")
            
            if raw_string_value is not None:
                # Shred the massive variable name into smaller BPE tokens
                # e.g., 'HttpResponseRedirect' -> [1245, 889, 1024]
                sub_tokens = self.bpe.encode(str(raw_string_value))
                
                # Wrap it in generation bounds
                sequence = [self.BOS_TOKEN] + sub_tokens + [self.EOS_TOKEN]
                bpe_payloads[node["id"]] = sequence
                
        # Inject the new BPE mapping directly into the graph package
        base_graph["bpe_payloads"] = bpe_payloads
        
        return base_graph
        
    def decode_bpe_to_string(self, token_sequence: List[int]) -> str:
        """
        Takes a sequence of predicted BPE integers and reconstructs the Python framework variable.
        Strips the EOS and BOS tokens natively.
        """
        clean_sequence = [t for t in token_sequence if t not in (self.BOS_TOKEN, self.EOS_TOKEN)]
        return self.bpe.decode(clean_sequence)

if __name__ == "__main__":
    print("\n=== Testing V4 Hybrid BPE Pipeline ===")
    v4_tokenizer = HybridTokenDecoder()
    
    # A prompt with a massive external dependency name that would instantly crash V3
    complex_prompt = "def execute():\n    db_router = DjangoSQLDatabaseRouter()\n    return db_router"
    
    print("\nOrigin Code:")
    print(complex_prompt)
    
    graph_package = v4_tokenizer.encode_ast_with_bpe(complex_prompt)
    
    print(f"\nSuccessfully Extracted {len(graph_package['nodes'])} Geometric AST Nodes.")
    print("Intercepted Variable Payloads (Shredded via BPE):")
    
    for node_id, bpe_sequence in graph_package['bpe_payloads'].items():
        # Find the original node name for debugging
        node = next(n for n in graph_package['nodes'] if n['id'] == node_id)
        
        # Reconstruct it to prove no data loss
        reconstructed = v4_tokenizer.decode_bpe_to_string(bpe_sequence)
        print(f"Node {node_id} ({node['type_str']}): {bpe_sequence} -> '{reconstructed}'")
