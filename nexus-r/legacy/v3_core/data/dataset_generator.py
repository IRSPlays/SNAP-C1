import json
import random
import sys
import os
from pathlib import Path

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from loguru import logger
from v3_core.data.trace_simulator import ExecutionTraceSimulator

class AlgorithmicDatasetGenerator:
    """
    SNAP-C1 V3: Synthetic Logic Generator (Phase 12)
    
    Generates hundreds of diverse Python algorithms procedurally to teach the 
    Continuous AST Decoder how to reason about nested loops, conditionals, 
    and recursive logic structures before hitting real-world repositories.
    """
    def __init__(self):
        self.simulator = ExecutionTraceSimulator()
        
    def generate_math_operations(self):
        ops = [
            ("add", "+"), ("sub", "-"), ("mult", "*"), ("div", "//"),
            ("mod", "%"), ("power", "**")
        ]
        
        snippets = []
        for name, symbol in ops:
            for _ in range(3):
                a = random.randint(1, 100)
                b = random.randint(1, 100)
                code = f"def {name}_op():\n    var_a = {a}\n    var_b = {b}\n    res = var_a {symbol} var_b\n    return res\nout = {name}_op()"
                snippets.append({"id": f"math_{name}_{a}_{b}", "code": code})
        return snippets

    def generate_loops(self):
        snippets = []
        for limit in [5, 10, 15]:
            code = f"def sum_up_to():\n    total = 0\n    for i in range({limit}):\n        total += i\n    return total\nout = sum_up_to()"
            snippets.append({"id": f"loop_sum_{limit}", "code": code})
            
            code_while = f"def while_count():\n    count = 0\n    while count < {limit}:\n        count += 1\n    return count\nout = while_count()"
            snippets.append({"id": f"while_count_{limit}", "code": code_while})
        return snippets
        
    def generate_conditionals(self):
        snippets = []
        for threshold in [10, 50, 100]:
            val = random.randint(1, 120)
            code = f"def check_limit():\n    val = {val}\n    limit = {threshold}\n    if val > limit:\n        return True\n    else:\n        return False\nout = check_limit()"
            snippets.append({"id": f"cond_limit_{threshold}_{val}", "code": code})
        return snippets

    def generate_list_ops(self):
        snippets = []
        for _ in range(5):
            arr = [random.randint(1, 20) for _ in range(random.randint(3, 8))]
            arr_str = str(arr)
            code = f"def find_max():\n    arr = {arr_str}\n    max_val = arr[0]\n    for num in arr:\n        if num > max_val:\n            max_val = num\n    return max_val\nout = find_max()"
            snippets.append({"id": f"list_max_{random.randint(1000,9999)}", "code": code})
            
            code_min = f"def find_min():\n    arr = {arr_str}\n    min_val = arr[0]\n    for num in arr:\n        if num < min_val:\n            min_val = num\n    return min_val\nout = find_min()"
            snippets.append({"id": f"list_min_{random.randint(1000,9999)}", "code": code_min})
        return snippets

    def synthesize_dataset(self, output_file: str):
        logger.info("Synthesizing algorithmic base logic...")
        
        all_snippets = []
        all_snippets.extend(self.generate_math_operations())
        all_snippets.extend(self.generate_loops())
        all_snippets.extend(self.generate_conditionals())
        all_snippets.extend(self.generate_list_ops())
        
        logger.info(f"Generated {len(all_snippets)} raw python functions. Tracing execution memory...")
        
        final_dataset = []
        for snippet in all_snippets:
            try:
                # The trace simulator physically executes the code and logs the RAM state line-by-line
                stringified_trace = self.simulator.generate_trace_dataset(snippet["code"])
                
                final_dataset.append({
                    "id": snippet["id"],
                    "original_code": snippet["code"],
                    "v3_training_sequence": stringified_trace
                })
            except Exception as e:
                logger.warning(f"Failed to trace {snippet['id']}: {e}")
                
        out_path = Path(__file__).parent / output_file
        with open(out_path, "w") as f:
            json.dump(final_dataset, f, indent=4)
            
        logger.success(f"Successfully generated offline algorithmic dataset: {out_path} ({len(final_dataset)} items)")

if __name__ == "__main__":
    generator = AlgorithmicDatasetGenerator()
    generator.synthesize_dataset("v3_algorithmic_dataset.json")
