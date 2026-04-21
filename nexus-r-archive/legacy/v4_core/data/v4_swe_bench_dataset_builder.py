import json
import argparse
from pathlib import Path
from loguru import logger

def build_swe_dataset(input_json: str, output_json: str):
    """
    Reads SWE-Bench Verified instances and creates an instruction dataset
    mapping problem statements directly to their git patches.
    """
    logger.info(f"Loading SWE-Bench data from {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = []
    for item in data:
        # We need the model to learn to output valid unified diff patches
        # based on the problem statement.
        prompt = item['problem_statement']
        target_code = item['patch']

        if not target_code or not target_code.strip():
            continue

        dataset.append({
            "prompt": prompt,
            "target_code": target_code
        })

    logger.info(f"Created {len(dataset)} training pairs from real GitHub patches.")
    
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    logger.info(f"Saved instruction dataset to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="v4_core/data/swe_bench_verified.json")
    parser.add_argument("--output", type=str, default="v4_core/data/v4_swe_instruction_dataset.json")
    args = parser.parse_args()
    build_swe_dataset(args.input, args.output)