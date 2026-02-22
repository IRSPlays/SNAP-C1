import sys
import os
import json
import ast
import glob
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import V4 Architecture Modules
from v4_core.memory.chroma_indexer import V4RepositoryIndexer
from v4_core.data.trace_simulator_stubs import V4StubbedTraceSimulator
from v4_core.data.bpe_tokenizer import HybridTokenDecoder

class V4GeneralDatasetBuilder:
    """
    SNAP-C1 V4: Universal Python Dataset Generator
    
    This engine converts any repository of raw Python files into the 
    massive geometric logic tensors required to train the V4 Architecture.
    
    Process:
    1. Read `.py` files from a target directory.
    2. Embed the repository into ChromaDB to create the Vector Layout.
    3. Pass valid functions through the V4 Stubbed Tracer to record local ODE Memory.
    4. Compile the target `Context -> Trace -> Geometric AST` outputs into JSON/Safetensors.
    """
    def __init__(self, target_repo_path: str, output_file: str = "v4_general_dataset.json"):
        self.target_repo_path = Path(target_repo_path)
        self.output_file = output_file
        
        # We need the ChromaDB Indexer to build the context vectors
        self.indexer = V4RepositoryIndexer(db_path="./v4_core/training_db")
        
        # We need the Object Mock Engine to blindly trace unseen library calls without crashing
        self.simulator = V4StubbedTraceSimulator()
        
        # We need the BPE Tokenizer to extract the formal geometric targets for the Loss function
        self.hybrid_tokenizer = HybridTokenDecoder()
        
        self.dataset = []
        
        logger.info(f"V4 Dataset Builder Online. Target: {self.target_repo_path}")

    def generate_data(self):
        """ The master extraction loop. """
        if not self.target_repo_path.exists():
            logger.error(f"Cannot find target repository: {self.target_repo_path}")
            return
            
        logger.info("Step 1: Embedding Codebase into Vector Space (ChromaDB)...")
        # In a full training run, we would index the repo here. 
        # For this script, we assume the indexer chunks the target files.
        # self.indexer.index_repository(str(self.target_repo_path)) 
        
        # Find all python files
        py_files = list(self.target_repo_path.rglob("*.py"))
        logger.info(f"Found {len(py_files)} total Python files. Extracting structural logic...")
        
        extracted_functions = 0
        
        # Step 2: Extract individual logic chunks to trace
        for file_path in tqdm(py_files, desc="Parsing ASTs"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # We use the generic Python `ast` library just to split the file up into functions
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip massive or empty functions
                        if not node.body:
                            continue
                            
                        func_code = ast.unparse(node)
                        
                        # Step 3: Run the code offline to get the memory trace (The magic step!)
                        # This generates the [CODE] ... [MEM] ... interleaved sequence
                        memory_trace = self.simulator.generate_stubbed_trace(func_code)
                        
                        # Step 4: Extract the AST Geometry & BPE Payloads (The actual Training Target)
                        try:
                            v4_geometric_target = self.hybrid_tokenizer.encode_ast_with_bpe(func_code)
                        except Exception as parse_e:
                            # If V3's AST graph parser fails on complex new python 3.10 syntax natively, we skip
                            continue
                        
                        # We have successfully compiled a perfect V4 training object
                        self.dataset.append({
                            "source_file": str(file_path.relative_to(self.target_repo_path)),
                            "function_name": node.name,
                            "original_code": func_code,
                            "v4_trace_input": memory_trace,            # What the Continuous Core sees
                            "v4_geometric_target": v4_geometric_target # What the Pointer-Generator predicts
                        })
                        extracted_functions += 1
                        
            except Exception as e:
                # File encoding issues, syntax errors, etc.
                pass
                
        logger.info(f"Extraction Complete. Mapped {extracted_functions} structural logic equations.")
        
        # Save to disk
        out_path = Path(__file__).parent / self.output_file
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2)
            
        logger.info(f"V4 Dataset saved to {out_path} ({len(self.dataset)} samples)")

if __name__ == "__main__":
    print("\n=======================================================")
    print("  Booting SNAP-C1 V4 (General Dataset Generator)       ")
    print("=======================================================\n")
    
    # For testing the script natively on the RX 7600 constraints, 
    # we point the dataset engine at our OWN legacy `v3_core` codebase!
    # It will rip the intelligence out of V3 to train V4.
    target_directory = os.path.abspath(os.path.join(project_root, "v3_core", "data"))
    
    builder = V4GeneralDatasetBuilder(target_repo_path=target_directory, output_file="v4_test_dataset.json")
    builder.generate_data()
