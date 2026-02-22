import ast
import json
import torch
from typing import Dict, Any, List

class ASTGraphParser:
    """
    SNAP-C1 V3 Model Component: Mathematical Graph Serialization
    
    This parser replaces the standard `tiktoken` BPE tokenizer. 
    Instead of ripping a python string into a flat sequence of integer IDs 
    (which generates millions of invalid syntax states), this class parses 
    code directly into its underlying Abstract Syntax Tree Graph (Nodes + Edges).
    """
    
    def __init__(self, language: str = "python"):
        self.language = language
        # A globally aligned dictionary mapping AST Node types to unique mathematical logic IDs
        self.node_embeddings = {
            "Module": 0, "FunctionDef": 1, "arguments": 2, "arg": 3, 
            "Assign": 4, "Name": 5, "Store": 6, "Constant": 7, "BinOp": 8,
            "Load": 9, "Add": 10, "Mult": 11, "Return": 12, "For": 13, "If": 14
        }
        self.next_id = 15
        
    def _get_node_id(self, node_type: str) -> int:
        """Assigns a persistent mathematical ID to a specific node type."""
        if node_type not in self.node_embeddings:
            self.node_embeddings[node_type] = self.next_id
            self.next_id += 1
        return self.node_embeddings[node_type]
        
    def parse_to_graph(self, code_string: str) -> Dict[str, Any]:
        """
        Takes raw string code and returns a mathematical Graph definition:
        Nodes: [Node_Type_ID, Node_Value_Embedding]
        Edges: [Parent_Idx, Child_Idx, Edge_Type]
        """
        if self.language == "python":
            try:
                tree = ast.parse(code_string)
            except SyntaxError as e:
                raise ValueError(f"V3 Graph Generator requires valid seed syntax. Parse failed: {e}")
                
            nodes = []
            edges = []
            
            # Recursive function to walk the tree and build the Math Graph
            def _walk_tree(node, parent_idx=None, edge_label=None):
                current_idx = len(nodes)
                node_type = type(node).__name__
                
                # Extract any raw value (like an integer '5' or string 'hello')
                value = None
                if isinstance(node, ast.Constant):
                    value = str(node.value)
                elif isinstance(node, ast.Name):
                    value = node.id
                    
                nodes.append({
                    "id": current_idx,
                    "type": self._get_node_id(node_type),
                    "type_str": node_type,
                    "value": value
                })
                
                # Link to parent
                if parent_idx is not None:
                    edges.append({
                        "source": parent_idx,
                        "target": current_idx,
                        "label": edge_label
                    })
                    
                # Walk children
                for field_name, child in ast.iter_fields(node):
                    if isinstance(child, ast.AST):
                        _walk_tree(child, current_idx, field_name)
                    elif isinstance(child, list):
                        for item in child:
                            if isinstance(item, ast.AST):
                                _walk_tree(item, current_idx, field_name)
                                
            _walk_tree(tree)
            return {"nodes": nodes, "edges": edges}
            
        else:
            raise NotImplementedError("Multi-language Graph Compilation (JS/Rust) mapped for Phase 2.")
            
    def graph_to_tensor(self, graph_dict: Dict) -> torch.Tensor:
        """
        Converts the serialized graph dictionary into a sequence of Node+Edge vectors
        ready for the Liquid Time-Constant (LTC) neural core.
        """
        # Node Type Tensors
        node_types = torch.tensor([n["type"] for n in graph_dict["nodes"]], dtype=torch.long)
        
        # In a full model, Edge connectivity matrices (Adjacency Matrix) would be built here
        # for Graph Neural Network message passing.
        
        return node_types

if __name__ == "__main__":
    print("Testing V3 AST Graph Parsing...")
    parser = ASTGraphParser()
    
    code = "def add(a, b): return a + b"
    graph = parser.parse_to_graph(code)
    
    print(f"Original Code: {code}")
    print(f"Extracted {len(graph['nodes'])} Mathematical Nodes.")
    print(f"Top Level AST Shapes:")
    for i in range(5):
        print(f"  Node {i}: {graph['nodes'][i]['type_str']}")
    
    tensor = parser.graph_to_tensor(graph)
    print(f"\nFinal Input Tensor Shape for V3 Core: {tensor.shape}")
