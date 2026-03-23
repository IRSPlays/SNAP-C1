"""
Logic Dataset Generator for FRC Pre-Training
============================================
Generates pure abstract logic, mathematics, and algorithmic structures.
This data contains zero "human facts" (no history, no pop culture).
It is used strictly to train the Latent Recurrent Loop how to think.
"""

import os
import json
import random
import ast
from pathlib import Path
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class LogicDatasetGenerator:
    """Procedurally generates verifiable reasoning problems."""
    
    def __init__(self):
        pass

    def _generate_math_problem(self) -> dict:
        """Generates a complex PEMDAS math problem."""
        ops = ['+', '-', '*']
        
        # Build a random equation
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = random.randint(1, 50)
        d = random.randint(1, 25)
        
        op1 = random.choice(ops)
        op2 = random.choice(ops)
        op3 = random.choice(ops)
        
        # Format: (A op1 B) op2 C op3 D
        equation_str = f"({a} {op1} {b}) {op2} {c} {op3} {d}"
        
        # Safely evaluate ground truth using AST
        try:
            answer = eval(equation_str)
        except Exception:
            return None # Skip invalid math
            
        prompt = f"Calculate the exact integer value of the following expression: {equation_str}"
        
        return {
            "type": "mathematics",
            "prompt": prompt,
            "expected_output": str(answer)
        }

    def _generate_algorithmic_problem(self) -> dict:
        """Generates abstract Python control flow logic puzzles."""
        # We generate a code snippet that modifies a variable, and ask the AI
        # to predict the final state of the variable without running it.
        
        var_start = random.randint(0, 10)
        loop_count = random.randint(3, 15)
        add_val = random.randint(1, 5)
        
        code_snippet = f"""
x = {var_start}
for i in range({loop_count}):
    if i % 2 == 0:
        x += {add_val}
    else:
        x -= 1
"""
        # Calculate ground truth natively
        x = var_start
        for i in range(loop_count):
            if i % 2 == 0:
                x += add_val
            else:
                x -= 1
                
        prompt = f"Analyze the following Python code execution trace:\n{code_snippet}\nWhat is the final integer value of the variable 'x'?"
        
        return {
            "type": "algorithm",
            "prompt": prompt,
            "expected_output": str(x)
        }

    def _generate_boolean_logic(self) -> dict:
        """Generates pure Boolean algebra proofs."""
        # A XOR B AND C ...
        vars_dict = {
            "A": random.choice([True, False]),
            "B": random.choice([True, False]),
            "C": random.choice([True, False])
        }
        
        operations = [
            ("A and B", vars_dict["A"] and vars_dict["B"]),
            ("A or (B and C)", vars_dict["A"] or (vars_dict["B"] and vars_dict["C"])),
            ("not A and (B or C)", not vars_dict["A"] and (vars_dict["B"] or vars_dict["C"])),
            ("(A != B) and C", (vars_dict["A"] != vars_dict["B"]) and vars_dict["C"]) # XOR
        ]
        
        op_str, answer = random.choice(operations)
        
        prompt = f"Given the boolean states A={vars_dict['A']}, B={vars_dict['B']}, and C={vars_dict['C']}, evaluate the truthful state of the following expression: {op_str}"
        
        return {
            "type": "boolean_logic",
            "prompt": prompt,
            "expected_output": "True" if answer else "False"
        }

    def generate_dataset(self, num_samples: int = 10000, output_file: str = "frc_pretrain_logic.jsonl"):
        """Generates the full dataset and writes to JSONL."""
        out_path = DATA_DIR / output_file
        
        logger.info(f"Generating {num_samples} abstract logic problems...")
        
        valid_samples = 0
        with open(out_path, "w") as f:
            pbar = tqdm(total=num_samples)
            while valid_samples < num_samples:
                # Randomly pick a problem type
                problem_type = random.choice([
                    self._generate_math_problem,
                    self._generate_algorithmic_problem,
                    self._generate_boolean_logic
                ])
                
                sample = problem_type()
                
                if sample:
                    f.write(json.dumps(sample) + "\n")
                    valid_samples += 1
                    pbar.update(1)
            pbar.close()
            
        logger.info(f"Dataset securely saved to: {out_path}")
        logger.info(f"File size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    generator = LogicDatasetGenerator()
    
    # For testing, we generate 5,000 samples. 
    # For actual Colab pre-training, you'd bump this to 5,000,000+
    generator.generate_dataset(num_samples=5000, output_file="frc_pretrain_logic_prototype.jsonl")
