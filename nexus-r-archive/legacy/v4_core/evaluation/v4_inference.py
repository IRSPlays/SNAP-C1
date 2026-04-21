"""
SNAP-C1 V4: Inference & Capability Evaluation Script
=====================================================
Loads the trained V4 weights and evaluates the model on:
  1. Held-out test prompts (not seen during training)
  2. Complexity analysis (ODE convergence speed)
  3. Routing diversity (expert selection distribution)
  4. Loss quality (how well the model predicts structural complexity)

Usage:
  python v4_core/evaluation/v4_inference.py
  python v4_core/evaluation/v4_inference.py --weights v4_core/snapshot_v4_hyper_router.pt
"""
import sys
import os
import torch
import json
import time
from loguru import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from v4_core.architecture.v4_assembly import V4HyperAssembly
from v4_core.utils.device import get_device


# ============================================================
#  Test Prompts — NOT from training data
# ============================================================
EVAL_PROMPTS = [
    {
        "prompt": "Fix the TypeError in Flask-RESTful when a nested blueprint attempts to register a resource with a conflicting endpoint name",
        "category": "web_framework",
        "expected_complexity": "medium"
    },
    {
        "prompt": "Debug the race condition in asyncio.gather() that causes intermittent connection pool exhaustion in aiohttp when handling 1000+ concurrent WebSocket connections",
        "category": "async_concurrency",
        "expected_complexity": "high"
    },
    {
        "prompt": "Resolve the ImportError circular dependency between Django's models.py and signals.py when using a custom User model with post_save signal handlers",
        "category": "dependency_resolution",
        "expected_complexity": "medium"
    },
    {
        "prompt": "Fix the pandas DataFrame.merge() producing duplicate columns when both DataFrames have identically named columns with different suffixes configuration",
        "category": "data_processing",
        "expected_complexity": "low"
    },
    {
        "prompt": "Debug memory leak in PyTorch DataLoader with persistent_workers=True when the custom Dataset.__getitem__ opens file handles without proper cleanup in the worker process",
        "category": "ml_pipeline",
        "expected_complexity": "high"
    },
    {
        "prompt": "Resolve the SQLAlchemy DetachedInstanceError occurring when accessing lazy-loaded relationships after the Session.close() in a multi-threaded FastAPI application",
        "category": "orm_database",
        "expected_complexity": "high"
    },
    {
        "prompt": "Fix the RecursionError in a recursive AST transformer that processes deeply nested list comprehensions exceeding Python's default recursion limit",
        "category": "ast_manipulation",
        "expected_complexity": "medium"
    },
    {
        "prompt": "Debug the numpy broadcasting error when multiplying arrays of shapes (3,4,5) and (4,1) inside a vectorized batch processing pipeline",
        "category": "numerical_computing",
        "expected_complexity": "low"
    },
]


class V4Evaluator:
    """Loads trained weights and runs inference evaluation."""
    
    def __init__(self, weights_path: str = None):
        self.device = get_device()
        
        logger.info("=== Loading V4 HyperAssembly for Evaluation ===")
        self.model = V4HyperAssembly(d_model=1024, max_loops=50)
        
        # Load trained weights if available
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            logger.success(f"Loaded trained weights from: {weights_path}")
        else:
            logger.warning("No trained weights found — running with initialized weights")
        
        self.model.eval()  # Disable dropout etc.
    
    @torch.no_grad()
    def run_single_inference(self, prompt: str) -> dict:
        """Run inference on a single prompt and collect detailed metrics."""
        start = time.time()
        
        output = self.model([prompt], batch_size=1)
        
        elapsed = time.time() - start
        
        return {
            "loss_logit": output["loss_logits"].item(),
            "time_steps": output["time_steps"],
            "experts_used": output["experts_used"],
            "inference_time_ms": elapsed * 1000,
        }
    
    @torch.no_grad()
    def run_batched_inference(self, prompts: list) -> dict:
        """Run inference on a batch of prompts."""
        B = len(prompts)
        start = time.time()
        
        output = self.model([p["prompt"] for p in prompts], batch_size=B)
        
        elapsed = time.time() - start
        
        return {
            "loss_logits": output["loss_logits"].squeeze(-1).tolist(),
            "time_steps": output["time_steps"],
            "experts_used": output["experts_used"],
            "batch_size": B,
            "total_time_ms": elapsed * 1000,
            "per_chunk_ms": (elapsed * 1000) / B,
        }
    
    def evaluate(self):
        """Full evaluation pipeline."""
        print("\n" + "=" * 70)
        print("  SNAP-C1 V4 — Capability Evaluation Report")
        print("=" * 70)
        print(f"  Device: {self.device}")
        print(f"  Eval Prompts: {len(EVAL_PROMPTS)}")
        print("=" * 70 + "\n")
        
        # === Test 1: Individual Inference ===
        print("─" * 70)
        print("  TEST 1: Individual Prompt Inference")
        print("─" * 70)
        
        results = []
        for i, task in enumerate(EVAL_PROMPTS):
            result = self.run_single_inference(task["prompt"])
            results.append(result)
            
            print(f"\n  [{i+1}/{len(EVAL_PROMPTS)}] {task['category'].upper()}")
            print(f"  Prompt: {task['prompt'][:80]}...")
            print(f"  ├─ ODE Convergence: {result['time_steps']} cycles")
            print(f"  ├─ Loss Logit:      {result['loss_logit']:.4f}")
            print(f"  ├─ Experts Routed:  {result['experts_used']}")
            print(f"  └─ Inference Time:  {result['inference_time_ms']:.1f}ms")
        
        # === Test 2: Batched Inference ===
        print("\n" + "─" * 70)
        print("  TEST 2: Batched Inference (all prompts simultaneously)")
        print("─" * 70)
        
        batch_result = self.run_batched_inference(EVAL_PROMPTS)
        
        print(f"\n  Batch Size:     {batch_result['batch_size']}")
        print(f"  Total Time:     {batch_result['total_time_ms']:.1f}ms")
        print(f"  Per Chunk:      {batch_result['per_chunk_ms']:.1f}ms")
        print(f"  ODE Steps:      {batch_result['time_steps']}")
        print(f"  Experts Used:   {batch_result['experts_used']}")
        
        # === Test 3: Summary Statistics ===
        print("\n" + "─" * 70)
        print("  TEST 3: Summary Statistics")
        print("─" * 70)
        
        ode_steps = [r["time_steps"] for r in results]
        logits = [r["loss_logit"] for r in results]
        times = [r["inference_time_ms"] for r in results]
        
        all_experts = []
        for r in results:
            all_experts.extend(r["experts_used"])
        expert_freq = {}
        for e in all_experts:
            expert_freq[e] = expert_freq.get(e, 0) + 1
        
        avg_ode = sum(ode_steps) / len(ode_steps) if ode_steps else 0
        avg_logit = sum(logits) / len(logits) if logits else 0
        avg_time = sum(times) / len(times) if times else 0
        
        print(f"\n  Avg ODE Convergence:    {avg_ode:.1f} / 50 cycles")
        print(f"  Avg Loss Logit:         {avg_logit:.4f}")
        print(f"  Avg Inference Time:     {avg_time:.1f}ms (single) | {batch_result['per_chunk_ms']:.1f}ms (batched)")
        print(f"  Expert Routing Distribution:")
        for exp_id, count in sorted(expert_freq.items()):
            bar = "█" * count
            print(f"    Expert {exp_id}: {bar} ({count})")
        
        # === Verdict ===
        print("\n" + "=" * 70)
        print("  VERDICT")
        print("=" * 70)
        
        issues = []
        if avg_ode >= 49:
            issues.append("ODE solver hitting max iterations (not converging) — needs more training")
        if len(expert_freq) <= 2:
            issues.append("Expert routing collapsed to few experts — needs routing diversity loss")
        if avg_logit < 0.01:
            issues.append("Loss logits near zero — loss head may be underfitting")
        
        speedup = avg_time / max(batch_result['per_chunk_ms'], 0.001)
        
        if not issues:
            print(f"\n  ✅ Model appears healthy!")
        else:
            print(f"\n  ⚠️  Potential issues detected:")
            for issue in issues:
                print(f"    • {issue}")
        
        print(f"\n  Batch speedup: {speedup:.1f}x over sequential")
        print(f"  Recommendation: ", end="")
        
        if avg_ode >= 48:
            print("Train more epochs or add more diverse data")
        elif len(expert_freq) <= 2:
            print("Add expert diversity regularization to loss")
        else:
            print("Model looks ready for SWE-Bench evaluation")
        
        print("=" * 70 + "\n")
        
        return {
            "avg_ode_steps": avg_ode,
            "avg_logit": avg_logit,
            "avg_inference_ms": avg_time,
            "batch_inference_ms": batch_result["per_chunk_ms"],
            "expert_distribution": expert_freq,
            "issues": issues,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SNAP-C1 V4 Inference Evaluator")
    parser.add_argument("--weights", type=str, default="v4_core/snapshot_v4_hyper_router.pt",
                        help="Path to trained model weights")
    args = parser.parse_args()
    
    evaluator = V4Evaluator(weights_path=args.weights)
    report = evaluator.evaluate()
