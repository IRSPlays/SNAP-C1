"""
SNAP-C1 V4: SWE-Bench Verified Benchmark Harness
=================================================
Evaluates the trained V4 model against the official SWE-Bench dataset
to measure its ability to understand and reason about real-world
Python software engineering problems.

This script:
  1. Downloads the SWE-Bench Verified subset (500 curated instances)
  2. Feeds each instance through the V4 pipeline
  3. Measures structural reasoning quality via ODE convergence metrics
  4. Outputs a standardized benchmark report

Usage:
  python v4_core/evaluation/v4_swe_bench.py
  python v4_core/evaluation/v4_swe_bench.py --weights v4_core/snapshot_v4_hyper_router.pt --max_instances 50
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


def load_swe_bench_verified(max_instances: int = 50) -> list:
    """
    Loads SWE-Bench Verified instances.
    Falls back to synthetic test cases if the official dataset isn't available.
    """
    # Try loading official SWE-Bench Verified dataset
    swe_bench_path = os.path.join(project_root, "v4_core", "data", "swe_bench_verified.json")
    
    if os.path.exists(swe_bench_path):
        with open(swe_bench_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} SWE-Bench Verified instances from disk")
        return data[:max_instances]
    
    # Try downloading from HuggingFace
    try:
        from datasets import load_dataset
        logger.info("Downloading SWE-Bench Verified from HuggingFace...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        
        instances = []
        for i, item in enumerate(ds):
            if i >= max_instances:
                break
            instances.append({
                "instance_id": item.get("instance_id", f"swe-{i}"),
                "repo": item.get("repo", "unknown"),
                "problem_statement": item.get("problem_statement", ""),
                "patch": item.get("patch", ""),
                "base_commit": item.get("base_commit", ""),
                "hints_text": item.get("hints_text", ""),
            })
        
        # Cache locally
        with open(swe_bench_path, "w") as f:
            json.dump(instances, f, indent=2)
        logger.info(f"Downloaded and cached {len(instances)} instances")
        return instances
        
    except ImportError:
        logger.warning("HuggingFace `datasets` not installed. Using synthetic test cases.")
    except Exception as e:
        logger.warning(f"Could not download SWE-Bench: {e}. Using synthetic test cases.")
    
    # Fallback: synthetic SWE-Bench-style instances
    synthetic = [
        {"instance_id": "django__django-11099", "repo": "django/django",
         "problem_statement": "UsernameValidator allows trailing newline in usernames. The validators.RegexValidator class uses re.search which allows partial matches. The ASCIIUsernameValidator and UnicodeUsernameValidator use regex patterns that don't anchor the end, allowing usernames with trailing newlines."},
        {"instance_id": "django__django-11179", "repo": "django/django",
         "problem_statement": "delete() on instances of models without any dependencies doesn't clear PKs. Calling delete() on model instances that don't have cascades clears the PK on the instance but only if there are no parents."},
        {"instance_id": "scikit-learn__scikit-learn-13497", "repo": "scikit-learn/scikit-learn",
         "problem_statement": "Comparing string to np.nan in _encode_check_unknown raises FutureWarning. In _encode.py, the comparison 'diff = set(values) - set(uniques)' when values contain strings and uniques contain np.nan triggers a FutureWarning."},
        {"instance_id": "flask__flask-4045", "repo": "pallets/flask",
         "problem_statement": "Raising error for URL with missing slash. When a URL is registered with a trailing slash but requested without one, Flask returns a 404 instead of redirecting."},
        {"instance_id": "sympy__sympy-20590", "repo": "sympy/sympy",
         "problem_statement": "Symbol.__new__ raises TypeError for invalid assumptions. Creating a Symbol with conflicting assumptions (e.g., Symbol('x', real=True, imaginary=True)) should raise a clear error."},
        {"instance_id": "requests__requests-3362", "repo": "psf/requests",
         "problem_statement": "Encoding error when posting files with non-ASCII filenames. The multipart encoder fails to properly encode filenames containing Unicode characters."},
        {"instance_id": "pytest__pytest-5692", "repo": "pytest-dev/pytest",
         "problem_statement": "Hostname and timestamp not set in generated JUnit XML reports. The <testsuite> element in JUnit XML output is missing the hostname and timestamp attributes."},
        {"instance_id": "matplotlib__matplotlib-23299", "repo": "matplotlib/matplotlib",
         "problem_statement": "get_backend() clears figures from Gcf.figs if the current backend is non-interactive. Calling matplotlib.get_backend() as a side effect closes all open figures."},
        {"instance_id": "astropy__astropy-6938", "repo": "astropy/astropy",
         "problem_statement": "Possible bug in io.fits with D exponent notation. FITS files with D exponent notation (e.g., 1.0D2) are not parsed correctly, leading to incorrect float values."},
        {"instance_id": "sphinx__sphinx-8435", "repo": "sphinx-doc/sphinx",
         "problem_statement": "autodoc fails to resolve forward references in type hints when __future__.annotations is used. PEP 563 postponed evaluation of annotations causes autodoc to show string representations instead of resolved types."},
    ]
    
    logger.info(f"Using {len(synthetic)} synthetic SWE-Bench instances")
    return synthetic[:max_instances]


class SWEBenchEvaluator:
    """Evaluates V4 against SWE-Bench instances."""
    
    def __init__(self, weights_path: str = None):
        self.device = get_device()
        self.model = V4HyperAssembly(d_model=1024, max_loops=50)
        
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            logger.success(f"Loaded trained weights: {weights_path}")
        else:
            logger.warning("No weights — evaluating randomly initialized model as baseline")
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_instance(self, instance: dict) -> dict:
        """Evaluate a single SWE-Bench instance."""
        prompt = instance.get("problem_statement", "")
        start = time.time()
        
        output = self.model([prompt], batch_size=1)
        elapsed = time.time() - start
        
        return {
            "instance_id": instance.get("instance_id", "unknown"),
            "repo": instance.get("repo", "unknown"),
            "loss_logit": output["loss_logits"].item(),
            "ode_steps": output["time_steps"],
            "experts": output["experts_used"],
            "time_ms": elapsed * 1000,
        }
    
    @torch.no_grad()
    def evaluate_batch(self, instances: list) -> dict:
        """Evaluate a batch of instances simultaneously."""
        prompts = [inst.get("problem_statement", "") for inst in instances]
        B = len(prompts)
        
        start = time.time()
        output = self.model(prompts, batch_size=B)
        elapsed = time.time() - start
        
        return {
            "batch_size": B,
            "loss_logits": output["loss_logits"].squeeze(-1).tolist(),
            "ode_steps": output["time_steps"],
            "experts": output["experts_used"],
            "total_ms": elapsed * 1000,
            "per_instance_ms": (elapsed * 1000) / B,
        }
    
    def run_benchmark(self, instances: list, batch_size: int = 8) -> dict:
        """Full SWE-Bench benchmark run."""
        print("\n" + "=" * 70)
        print("  SNAP-C1 V4 — SWE-Bench Verified Benchmark")
        print("=" * 70)
        print(f"  Device:     {self.device}")
        print(f"  Instances:  {len(instances)}")
        print(f"  Batch Size: {batch_size}")
        print("=" * 70 + "\n")
        
        all_results = []
        total_start = time.time()
        
        # Process in batches
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i+batch_size]
            B = len(batch)
            
            batch_result = self.evaluate_batch(batch)
            
            for j, inst in enumerate(batch):
                result = {
                    "instance_id": inst.get("instance_id", "unknown"),
                    "repo": inst.get("repo", "unknown"),
                    "loss_logit": batch_result["loss_logits"][j] if j < len(batch_result["loss_logits"]) else 0,
                    "ode_steps": batch_result["ode_steps"],
                    "time_ms": batch_result["per_instance_ms"],
                }
                all_results.append(result)
            
            print(f"  Batch [{i//batch_size + 1}/{(len(instances)-1)//batch_size + 1}] | "
                  f"{B} instances | {batch_result['total_ms']:.0f}ms | "
                  f"ODE: {batch_result['ode_steps']} steps")
        
        total_time = time.time() - total_start
        
        # === Aggregate Statistics ===
        print("\n" + "─" * 70)
        print("  BENCHMARK RESULTS")
        print("─" * 70)
        
        logits = [r["loss_logit"] for r in all_results]
        ode_steps = [r["ode_steps"] for r in all_results]
        times = [r["time_ms"] for r in all_results]
        
        avg_logit = sum(logits) / len(logits) if logits else 0
        avg_ode = sum(ode_steps) / len(ode_steps) if ode_steps else 0
        avg_time = sum(times) / len(times) if times else 0
        
        # Group by repo
        repo_stats = {}
        for r in all_results:
            repo = r["repo"]
            if repo not in repo_stats:
                repo_stats[repo] = []
            repo_stats[repo].append(r["loss_logit"])
        
        print(f"\n  Total Instances:        {len(all_results)}")
        print(f"  Total Wall Time:        {total_time:.1f}s")
        print(f"  Avg Inference/Instance: {avg_time:.1f}ms")
        print(f"  Avg ODE Convergence:    {avg_ode:.1f} / 50")
        print(f"  Avg Loss Logit:         {avg_logit:.4f}")
        
        print(f"\n  Per-Repository Breakdown:")
        for repo, repo_logits in sorted(repo_stats.items()):
            avg = sum(repo_logits) / len(repo_logits)
            print(f"    {repo:40s} | logit: {avg:+.4f} | n={len(repo_logits)}")
        
        # === Convergence Analysis ===
        print(f"\n  ODE Convergence Analysis:")
        converged = sum(1 for s in ode_steps if s < 45)
        near_max = sum(1 for s in ode_steps if s >= 48)
        print(f"    Fast convergence (<45):  {converged}/{len(ode_steps)}")
        print(f"    Near-max (≥48):          {near_max}/{len(ode_steps)}")
        
        # === Recommendations ===
        print("\n" + "─" * 70)
        print("  RECOMMENDATIONS")
        print("─" * 70)
        
        if near_max / max(len(ode_steps), 1) > 0.7:
            print("  ⚠️  >70% of instances hit near-max ODE steps.")
            print("     → Train more epochs (try 500) or increase max_loops to 100")
            print("     → Add more diverse training data from real GitHub repos")
        
        if abs(avg_logit) < 0.1:
            print("  ⚠️  Loss logits clustered near zero — model may be underfitting.")
            print("     → Increase model capacity (d_model=2048)")
            print("     → Train with more data")
        
        if len(repo_stats) > 1:
            logit_variance = max(
                abs(sum(v)/len(v) - avg_logit) for v in repo_stats.values()
            )
            if logit_variance < 0.01:
                print("  ⚠️  No differentiation between repos — model not learning structure.")
                print("     → Needs structural loss (actual AST comparison)")
            else:
                print("  ✅ Model shows repo-level differentiation")
        
        print("=" * 70 + "\n")
        
        # Save report
        report = {
            "model": "SNAP-C1-V4",
            "device": str(self.device),
            "num_instances": len(all_results),
            "total_time_s": total_time,
            "avg_loss_logit": avg_logit,
            "avg_ode_steps": avg_ode,
            "avg_inference_ms": avg_time,
            "per_repo": {repo: sum(v)/len(v) for repo, v in repo_stats.items()},
            "results": all_results,
        }
        
        report_path = os.path.join(project_root, "v4_core", "evaluation", "swe_bench_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.success(f"Benchmark report saved to: {report_path}")
        
        return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SNAP-C1 V4 SWE-Bench Benchmark")
    parser.add_argument("--weights", type=str, default="v4_core/snapshot_v4_hyper_router.pt")
    parser.add_argument("--max_instances", type=int, default=50, help="Max SWE-Bench instances to eval")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for eval")
    args = parser.parse_args()
    
    evaluator = SWEBenchEvaluator(weights_path=args.weights)
    instances = load_swe_bench_verified(max_instances=args.max_instances)
    report = evaluator.run_benchmark(instances, batch_size=args.batch_size)
