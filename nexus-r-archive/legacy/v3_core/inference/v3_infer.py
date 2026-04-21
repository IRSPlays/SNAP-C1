import sys
import os
import torch
from pathlib import Path
from loguru import logger
import ast

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from v3_core.architecture.v3_assembly import V3GenerativeReasoningArchitecture
from v2_core.architecture.holographic_compressor import HolographicCompressor
from v3_core.data.ast_parser import ASTGraphParser

class V3InferenceEngine:
    """
    SNAP-C1 V3 Model Component: Mathematical Graph Inference
    
    This loads the new PyTorch-Native Generative Weights 
    (snapshot_v3_generative_LTC_core.pt) and executes a prompt 
    through the Liquid Time-Constant (LTC) core offline!
    """
    def __init__(self, device: str = None):
        if device is None:
            try:
                import torch_directml
                dml_device = torch_directml.device() 
                for i in range(torch_directml.device_count()):
                    name = torch_directml.device_name(i).lower()
                    if "rx" in name or "tx" in name:
                        dml_device = torch_directml.device(i)
                        break
                self.device = dml_device
            except ImportError:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Booting V3 Inference Engine on: {self.device}")
        
        # Load the architecture
        self.model = V3GenerativeReasoningArchitecture().to(self.device)
        self.compressor = HolographicCompressor(d_model=1024).to(self.device)
        self.model.eval()
        self.compressor.eval()
        
        # Pull the new 744MB Continuous Training Checkpoint
        weights_path = Path(__file__).parent.parent / "snapshot_v3_generative_LTC_core.pt"
        if weights_path.exists():
            logger.info(f"Loading Generative Weights: {weights_path}")
            
            # Handle torch.compile '_orig_mod.' prefix stripping
            # Load into CPU RAM first to prevent DirectML MemoryError alloc crashes
            # AMD Tensors require custom deserialization, so weights_only=False is mandatory here.
            raw_state_dict = torch.load(weights_path, map_location='cpu', weights_only=False, mmap=True)
            clean_state_dict = {}
            for k, v in raw_state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                clean_state_dict[new_key] = v
                
            self.model.load_state_dict(clean_state_dict, strict=False)
            
            # Flush the CPU-staged weights BACK to the dedicated AMD RX 7600 VRAM!
            self.model = self.model.to(self.device)
            logger.success("V3 Offline Synapses physically mapped onto DirectML VRAM successfully.")
        else:
            logger.error(f"Cannot perform inference. Missing V3 Checkpoint at {weights_path}")
            sys.exit(1)
            
        # Reverse mapping for the UI interpretation of Graph Nodes
        # Instead of 250 arbitrary numbers, we map them back to Python AST object names
        self.reverse_ast_vocab = {
            # Dummy map for visualization sake based on the trained seed suite length
            0: "Module", 1: "FunctionDef", 2: "arguments", 3: "arg", 
            4: "Assign", 5: "Name", 6: "Store", 7: "Constant", 8: "BinOp",
            9: "Load", 10: "Add", 11: "Mult", 12: "Return", 13: "For", 14: "If"
        }
        
        # Load the newly defined 50-space string variable/constant vocabulary
        parser = ASTGraphParser()
        self.semantic_vocab_reverse = {v: k for k, v in parser.semantic_embeddings.items()}

    @torch.no_grad()
    def generate_ast_graph(self, prompt: str, max_nodes: int = 15):
        """
        Passes a human prompt into the Continuous logic layers and physical extracts
        the Graph Neural Network branching predictions.
        """
        logger.info(f"\n--- User Prompt Input ---")
        logger.info(f"'{prompt}'\n")
        
        # 1. Compress prompt
        hologram = self.compressor.process_string(prompt, device=self.device)
        hologram = hologram.mean(dim=1) # Pool for the generic V3 build context vector
        
        # 2. Fluid continuous-time thinking
        # In the real V3 layout, we'd pass raw Hologram sequence. For this skeleton, we simulated 2D pooling.
        # But `ContinuousRecurrentCore` expects [batch, seq, dim], so we re-unsqueeze.
        math_vector = hologram.unsqueeze(1) 
        equilibrium_vector, time_steps = self.model.core(math_vector)
        
        # 3. Predict the continuous AST Math structure natively on GPU (No timeouts!)
        predicted_node_ids, branch_probabilities, _, semantic_logits = self.model.ast_decoder(equilibrium_vector, max_nodes=max_nodes)
        
        # Determine the maximum likelihood payload variable names dynamically
        predicted_semantics = torch.argmax(semantic_logits, dim=-1)[0].cpu().tolist()
        
        return predicted_node_ids[0].cpu().tolist(), branch_probabilities[0].cpu().tolist(), time_steps, predicted_semantics

    def construct_python_code(self, node_list: list, semantics_list: list = None) -> str:
        """
        Takes the mathematical branch tree (sequence of classification IDs)
        and attempts to structurally build a valid Python ast.AST object,
        then unparses it into readable text code!
        """
        module = ast.Module(body=[], type_ignores=[])
        current_func_node = None
        current_args = []
        
        # Keep track of the last Assign node to dynamically populate its target vs its logic
        active_assign = None
        
        for i, node_id in enumerate(node_list):
            safe_id = node_id % 15 
            node_name = self.reverse_ast_vocab.get(safe_id, "Unknown")
            
            # Map the exact Semantic string onto the Graph geometry
            semantic_id = semantics_list[i] if semantics_list and i < len(semantics_list) else 0
            semantic_val = self.semantic_vocab_reverse.get(semantic_id, "var_x")
            if semantic_val == "<pad>":
                semantic_val = "var_x" # default algorithmic safe fallback
                
            if node_name == "FunctionDef":
                func_name = semantic_val if semantic_val != "var_x" else "generated_logic"
                current_func_node = ast.FunctionDef(
                    name=func_name,
                    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=[], decorator_list=[]
                )
                module.body.append(current_func_node)
                
            elif node_name == "arg" and current_func_node is not None:
                arg_name = semantic_val if semantic_val != "var_x" else f"var_{len(current_args)}"
                current_args.append(ast.arg(arg=arg_name))
                current_func_node.args.args = current_args
                
            elif node_name == "Assign" and current_func_node is not None:
                active_assign = ast.Assign(targets=[], value=ast.Constant(value=None))
                current_func_node.body.append(active_assign)
                
            elif node_name == "Name" and active_assign is not None:
                if len(active_assign.targets) == 0:
                    active_assign.targets.append(ast.Name(id=semantic_val, ctx=ast.Store()))
                else:
                    active_assign.value = ast.Name(id=semantic_val, ctx=ast.Load())
                    
            elif node_name == "Constant" and active_assign is not None:
                val = semantic_val
                if val.isdigit(): val = int(val)
                elif val == "True": val = True
                elif val == "False": val = False
                active_assign.value = ast.Constant(value=val)
                
            elif node_name == "Return" and current_func_node is not None:
                return_node = ast.Return(value=ast.Name(id=semantic_val, ctx=ast.Load()))
                current_func_node.body.append(return_node)
                
        if current_func_node and not current_func_node.body:
            current_func_node.body.append(ast.Pass())
            
        try:
            ast.fix_missing_locations(module)
            return ast.unparse(module)
        except Exception as e:
            return f"# [V3 Grammar Error] Failed to Unparse Mathematical Tree: {e}"

if __name__ == "__main__":
    print("\n=== Initializing V3 Live AST Evaluation ===")
    engine = V3InferenceEngine()
    
    test_prompt = "chat: Write a function that multiples two inputs."
    
    nodes, branching_weights, step_time, semantics = engine.generate_ast_graph(test_prompt, max_nodes=15)
    
    print(f"\nThe V3 Generative Engine arrived at a Mathematical Conclusion in {step_time} continuous steps.")
    print(f"Generated Mathematical Branch Tree:\n")
    
    for i, (node_id, prob, sem_id) in enumerate(zip(nodes, branching_weights, semantics)):
        # If the network predicted a completely wild ID, cap it for the demo dictionary map
        safe_id = node_id % 15 
        node_name = engine.reverse_ast_vocab.get(safe_id, "Unknown")
        payload = engine.semantic_vocab_reverse.get(sem_id, "<pad>")
        
        # Print a simple visual graph structure
        indent = "  " * (i % 3)
        print(f"{indent}├── [{node_id}] {node_name} ~ Payload: '{payload}' (Confidence: {prob:.2f})")
        
    print("\n-----------------------------------------------------------")
    print("Translating Abstract Logic & Dynamic Semantics into executable Python Text:")
    print("-----------------------------------------------------------")
    
    generated_python_code = engine.construct_python_code(nodes, semantics_list=semantics)
    print(generated_python_code)
    
    print("\n-----------------------------------------------------------")
    print("Notice we never booted up a Python subprocess. The logic generated natively inside the VRAM!")
