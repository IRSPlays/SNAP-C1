import sys
import os
import json
from pathlib import Path
from loguru import logger

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from v3_core.data.trace_simulator import ExecutionTraceSimulator

def generate_v3_dataset():
    """
    Reads a raw Python dataset and mathematically rips the RAM state
    at each line of execution to generate the V3 Generative Offline dataset.
    """
    simulator = ExecutionTraceSimulator()
    
    # In a real build, we'd load HumanEval or MBPP from HuggingFace
    # For this initialization, we use a 5-problem hardcoded seed suite.
    raw_dataset = [
        # 1. Basic Arithmetic
        """
def solve_math():
    a = 15
    b = 25
    c = a + b
    return c
result = solve_math()
""",
        # 2. Array Iteration
        """
def sum_array():
    arr = [1, 2, 3]
    total = 0
    for num in arr:
        total += num
    return total
ans = sum_array()
""",
        # 3. String Manipulation
        """
def reverse_string():
    s = "RX7600"
    rev = ""
    for char in s:
        rev = char + rev
    return rev
out = reverse_string()
""",
        # 4. Conditional Logic
        """
def check_even():
    val = 10
    is_even = False
    if val % 2 == 0:
        is_even = True
    return is_even
res = check_even()
""",
        # 5. Fib Sequence
        """
def fib():
    a = 0
    b = 1
    for i in range(3):
        temp = a + b
        a = b
        b = temp
    return b
fib_ans = fib()
"""
    ]

    logger.info("Initializing V3 Dataset Generation (Execution-Trace Ripping)...")
    
    output_dataset = []
    
    for i, code_block in enumerate(raw_dataset):
        code_block = code_block.strip()
        
        # 1. Fire the code through the native native Python C-debugger
        v3_sequence = simulator.generate_trace_dataset(code_block)
        
        if v3_sequence:
            output_dataset.append({
                "id": f"task_{i}",
                "original_code": code_block,
                "v3_training_sequence": v3_sequence
            })
            logger.success(f"Generated [CODE]/[MEM] Mathematical Sequence for task_{i}.")
        else:
            logger.error(f"Failed to trace dataset entry {i}.")
            
    # Save the ripped memory sequences to disk
    output_path = Path(__file__).parent / "v3_seed_dataset.json"
    
    with open(output_path, "w") as f:
        json.dump(output_dataset, f, indent=4)
        
    logger.info(f"\nV3 Dataset Generation Complete. Saved 5 offline sequences to {output_path}")

if __name__ == "__main__":
    generate_v3_dataset()
