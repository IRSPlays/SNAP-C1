"""
NEXUS V6 Benchmark Suite
========================
Real benchmarks to validate architecture claims:
1. Perplexity on standard datasets
2. Reasoning tasks
3. Efficiency comparison vs larger models
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import time
from typing import Dict, List, Tuple

# Import NEXUS
import sys
sys.path.insert(0, '/workspaces/SNAP-C1')
from v6_core.architecture.nexus_v6 import build_nexus_tiny, build_nexus_small


def load_nexus_model(checkpoint_path: str = None):
    """Load NEXUS model with optional checkpoint."""
    model = build_nexus_tiny()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'])
        elif 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, max_samples: int = 500) -> float:
    """Compute perplexity on text dataset. Lower is better."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            
            text = example['text'] if 'text' in example else example.get('content', '')
            if not text:
                continue
            
            # Tokenize
            ids = tokenizer.encode(text, truncation=True, max_length=128, 
                                  return_tensors='pt')
            
            # Forward - handle both NEXUS (returns tuple) and HuggingFace (returns object)
            output = model(ids)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output.logits
            
            # Compute loss (shift by 1 for causal LM)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            total_loss += loss.item() * (ids.size(1) - 1)
            total_tokens += ids.size(1) - 1
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity


def benchmark_vs_baseline(nexus_model, nexus_tokenizer, 
                          baseline_model_name: str = "gpt2") -> Dict:
    """
    Compare NEXUS vs baseline (GPT-2) on perplexity.
    Uses same tokenizer for fair comparison.
    """
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print(f"{'='*60}")
    
    # Load baseline
    print(f"Loading baseline: {baseline_model_name}...")
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_name)
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
    baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
    
    # Dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # NEXUS perplexity
    print("\nEvaluating NEXUS...")
    nexus_params = sum(p.numel() for p in nexus_model.parameters())
    start = time.time()
    nexus_ppl = compute_perplexity(nexus_model, nexus_tokenizer, dataset, max_samples=200)
    nexus_time = time.time() - start
    
    # Baseline perplexity
    print("Evaluating baseline...")
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    start = time.time()
    baseline_ppl = compute_perplexity(baseline_model, baseline_tokenizer, dataset, max_samples=200)
    baseline_time = time.time() - start
    
    # Results
    results = {
        'nexus': {
            'params_M': nexus_params / 1e6,
            'perplexity': nexus_ppl,
            'time_sec': nexus_time,
        },
        'baseline': {
            'name': baseline_model_name,
            'params_M': baseline_params / 1e6,
            'perplexity': baseline_ppl,
            'time_sec': baseline_time,
        },
        'comparison': {
            'size_ratio': baseline_params / nexus_params,
            'ppl_ratio': baseline_ppl / nexus_ppl if nexus_ppl > 0 else float('inf'),
        }
    }
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"NEXUS:  {nexus_params/1e6:.1f}M params | PPL: {nexus_ppl:.2f} | Time: {nexus_time:.1f}s")
    print(f"BASELINE: {baseline_params/1e6:.1f}M params | PPL: {baseline_ppl:.2f} | Time: {baseline_time:.1f}s")
    print(f"\nSize ratio: {results['comparison']['size_ratio']:.1f}x smaller")
    print(f"PPL ratio: {results['comparison']['ppl_ratio']:.2f}x {'BETTER' if results['comparison']['ppl_ratio'] > 1 else 'WORSE'}")
    
    return results


def test_reasoning(model, tokenizer) -> Dict:
    """Test basic reasoning tasks."""
    print(f"\n{'='*60}")
    print("REASONING TASKS")
    print(f"{'='*60}")
    
    model.eval()
    results = {}
    
    tasks = [
        {
            'name': 'Pattern Completion',
            'prompt': 'Complete the pattern: 1, 2, 4, 8, 16, ',
            'expected': '32',
        },
        {
            'name': 'Simple Addition',
            'prompt': 'What is 25 + 17? The answer is ',
            'expected': '42',
        },
        {
            'name': 'Word Analogy',
            'prompt': 'King is to Queen as Man is to ',
            'expected': 'Woman',
        },
        {
            'name': 'Sentence Completion',
            'prompt': 'The sky is blue because it ',
            'expected': 'reflects',
        },
    ]
    
    for task in tasks:
        with torch.no_grad():
            ids = tokenizer.encode(task['prompt'], return_tensors='pt')
            logits, _ = model(ids)
            
            # Get top prediction
            probs = torch.softmax(logits[0, -1], dim=-1)
            top5 = torch.topk(probs, 5)
            
            decoded = [tokenizer.decode([t]) for t in top5.indices]
            
            # Check if expected is in top 5
            hit = any(task['expected'].lower() in d.lower() for d in decoded)
            
            results[task['name']] = {
                'prompt': task['prompt'],
                'expected': task['expected'],
                'top5': decoded,
                'hit': hit,
            }
            
            status = "✓" if hit else "✗"
            print(f"{status} {task['name']}: expected '{task['expected']}', got {decoded[:3]}")
    
    return results


def test_code_generation(model, tokenizer) -> Dict:
    """Test code generation capability."""
    print(f"\n{'='*60}")
    print("CODE GENERATION")
    print(f"{'='*60}")
    
    model.eval()
    results = {}
    
    prompts = [
        "def fibonacci(n):",
        "class HelloWorld:",
        "import torch\nx = torch",
    ]
    
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            ids = tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate a few tokens
            for _ in range(20):
                logits, _ = model(ids)
                next_token = torch.argmax(logits[0, -1])
                ids = torch.cat([ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                if next_token == tokenizer.eos_token_id:
                    break
            
            generated = tokenizer.decode(ids[0])
            results[f'test_{i}'] = {
                'prompt': prompt,
                'generated': generated[len(prompt):],
            }
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated[len(prompt):][:100]}...")
    
    return results


def run_full_benchmark(checkpoint_path: str = None) -> Dict:
    """Run complete benchmark suite."""
    print("\n" + "="*60)
    print("NEXUS V6 BENCHMARK SUITE")
    print("="*60)
    
    # Load model
    print("\nLoading NEXUS model...")
    model, tokenizer = load_nexus_model(checkpoint_path)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    all_results = {}
    
    # 1. Perplexity benchmark
    print("\n[1/4] Running perplexity benchmark...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    all_results['perplexity'] = compute_perplexity(model, tokenizer, dataset, max_samples=300)
    print(f"WikiText-2 perplexity: {all_results['perplexity']:.2f}")
    
    # 2. Baseline comparison
    print("\n[2/4] Running baseline comparison...")
    all_results['baseline_comparison'] = benchmark_vs_baseline(model, tokenizer)
    
    # 3. Reasoning tasks
    print("\n[3/4] Running reasoning tasks...")
    all_results['reasoning'] = test_reasoning(model, tokenizer)
    
    # 4. Code generation
    print("\n[4/4] Testing code generation...")
    all_results['code_generation'] = test_code_generation(model, tokenizer)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Perplexity: {all_results['perplexity']:.2f}")
    
    if 'baseline_comparison' in all_results:
        bc = all_results['baseline_comparison']
        print(f"vs {bc['baseline']['name']}: {bc['comparison']['size_ratio']:.1f}x smaller, "
              f"{bc['comparison']['ppl_ratio']:.2f}x {'better' if bc['comparison']['ppl_ratio'] > 1 else 'worse'}")
    
    reasoning_score = sum(1 for r in all_results['reasoning'].values() if r['hit'])
    print(f"Reasoning: {reasoning_score}/4 tasks passed")
    
    return all_results


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_full_benchmark(checkpoint)
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        # Convert tensors to floats for JSON
        json.dump(results, f, indent=2, default=lambda x: float(x) if torch.is_tensor(x) else str(x))
    print("\n✓ Results saved to benchmark_results.json")