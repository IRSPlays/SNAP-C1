import json

with open('v4_core/data/swe_bench_verified.json', 'r') as f:
    original_data = json.load(f)

# Wait, previously we overwrote 'swe_bench_verified.json' with ONLY the 4 shortest instances.
# Let's download a fresh batch from huggingface datasets to get completely new ones.
from datasets import load_dataset
print("Downloading fresh SWE-Bench instances for zero-shot testing...")
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

test_instances = []
for i, item in enumerate(ds):
    if i >= 10:  # Skip the first 10 just to be absolutely sure they weren't in the cache
        test_instances.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "base_commit": item["base_commit"],
        })
    if len(test_instances) == 4:
        break

with open('v4_core/data/swe_bench_verified_test.json', 'w') as f:
    json.dump(test_instances, f, indent=2)

print('Created zero-shot test set with 4 unseen instances:')
for item in test_instances:
    print(f"- {item['instance_id']}")
