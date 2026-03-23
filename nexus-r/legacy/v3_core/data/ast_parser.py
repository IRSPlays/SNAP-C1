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
        
        # A dictionary mapping text values (Variables, Constants, Function Names) to Semantic IDs
        self.semantic_embeddings = {
            "<pad>": 0, "solve_math": 1, "a": 2, "b": 3, "c": 4, "result": 5,
            "15": 6, "25": 7,
            "sum_array": 8, "arr": 9, "total": 10, "num": 11, "ans": 12,
            "1": 13, "2": 14, "3": 15, "0": 16,
            "reverse_string": 17, "s": 18, "rev": 19, "char": 20, "out": 21,
            "RX7600": 22, "": 23,
            "check_even": 24, "val": 25, "is_even": 26, "res": 27,
            "10": 28, "True": 29, "False": 30,
            "fib": 31, "i": 32, "temp": 33, "fib_ans": 34
        }
        self.max_semantic_vocab = 1000
        
    def _get_semantic_id(self, value: str) -> int:
        """Dynamically retrieves or auto-registers a specific token ID for the Variable/Constant."""
        if value not in self.semantic_embeddings:
            if len(self.semantic_embeddings) < self.max_semantic_vocab:
                self.semantic_embeddings[value] = len(self.semantic_embeddings)
            else:
                return 0 # Fallback to <pad> if the graph vocabulary limit is reached
        return self.semantic_embeddings[value]
        
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
                semantic_id = 0
                if isinstance(node, ast.Constant):
                    value = str(node.value)
                    semantic_id = self._get_semantic_id(value)
                elif isinstance(node, ast.Name):
                    value = node.id
                    semantic_id = self._get_semantic_id(value)
                elif isinstance(node, ast.arg):
                    value = node.arg
                    semantic_id = self._get_semantic_id(value)
                elif isinstance(node, ast.FunctionDef):
                    value = node.name
                    semantic_id = self._get_semantic_id(value)
                    
                nodes.append({
                    "id": current_idx,
                    "type": self._get_node_id(node_type),
                    "type_str": node_type,
                    "value": value,
                    "semantic_id": semantic_id
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
