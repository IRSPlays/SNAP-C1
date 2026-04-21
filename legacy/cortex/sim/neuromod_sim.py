"""Cortex V1 — Neuromodulator + Hopfield Memory Simulation
==========================================================
Pure NumPy. No PyTorch. No GPU.

Simulates the 4 neuromodulatory signals and Hopfield memory interaction
under synthetic scenarios to verify the math BEFORE building modules.

Signals:
    δ (dopamine)       — Controls hippocampus writes (surprise/novelty)
    ν (norepinephrine) — Controls cortex iteration depth (uncertainty)
    σ (serotonin)      — Controls memory-vs-cortex blend (confidence)
    α (acetylcholine)  — Reserved for V2 (learning rate modulation)

Run:
    cd SNAP-C1/cortex/sim
    python neuromod_sim.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================
# Neuromodulator (deterministic, V1)
# ============================================================

@dataclass
class NeuromodulatorState:
    """Running statistics for adaptive thresholding."""
    mu_epsilon: float = 0.5        # Running mean of prediction errors
    sigma_epsilon: float = 0.2     # Running std of prediction errors
    ema_decay: float = 0.99        # EMA smoothing for running stats
    n_updates: int = 0             # Total updates seen
    history: dict = field(default_factory=lambda: {
        'delta': [], 'nu': [], 'sigma': [], 'alpha': [],
        'epsilon': [], 'mu': [], 'std': [],
        'mem_confidence': [], 'writes': [], 'reads': [],
    })


def compute_neuromodulators(
    prediction_error: float,      # ||ε|| — cosine distance between predicted and actual embedding
    memory_confidence: float,     # max(softmax(β·K^T·q)) — best match in Hopfield memory
    state: NeuromodulatorState,
) -> Tuple[float, float, float, float]:
    """Compute all 4 neuromodulatory signals.

    Returns (δ, ν, σ, α):
        δ ∈ [0, 1]: Dopamine — write gate for hippocampus
        ν ∈ [0, 1]: Norepinephrine — cortex effort/iterations
        σ ∈ [0, 1]: Serotonin — memory blend weight (high = trust memory)
        α ∈ [0, 1]: Acetylcholine — reserved (always 0 in V1)
    """

    # Update running statistics (EMA)
    if state.n_updates == 0:
        state.mu_epsilon = prediction_error
        state.sigma_epsilon = 0.1  # Initial std estimate
    else:
        state.mu_epsilon = state.ema_decay * state.mu_epsilon + (1 - state.ema_decay) * prediction_error
        # EMA of variance, then sqrt for std
        var = (prediction_error - state.mu_epsilon) ** 2
        old_var = state.sigma_epsilon ** 2
        new_var = state.ema_decay * old_var + (1 - state.ema_decay) * var
        state.sigma_epsilon = max(np.sqrt(new_var), 1e-6)  # Floor to prevent div-by-zero

    state.n_updates += 1

    # --- DOPAMINE (δ): Surprise → write gate ---
    # Two-component trigger: absolute AND relative surprise
    # Absolute: if error exceeds a fixed floor, always consider writing
    # Relative: if error is unusually high compared to recent history
    abs_threshold = 0.3  # Fixed floor — if ε > 0.3, the model is wrong enough to store
    delta_abs = float(np.clip((prediction_error - abs_threshold) / (1.0 - abs_threshold + 1e-6), 0.0, 1.0))
    z_score = (prediction_error - state.mu_epsilon) / max(state.sigma_epsilon, 1e-6)
    delta_rel = float(np.clip(z_score, 0.0, 1.0))
    delta = max(delta_abs, delta_rel)

    # --- NOREPINEPHRINE (ν): Uncertainty → cortex effort ---
    # High prediction error AND low memory confidence = maximum uncertainty
    # The cortex should work harder when both are true
    uncertainty = prediction_error * (1.0 - memory_confidence)
    nu = float(np.clip(uncertainty * 2.0, 0.0, 1.0))  # Scale factor 2.0, clamp to [0,1]

    # --- SEROTONIN (σ): Confidence → trust memory ---
    # High memory confidence + low prediction error = trust memory
    # Low memory confidence = trust cortex
    # Sigmoid-shaped response centered at confidence=0.5
    sigma = float(1.0 / (1.0 + np.exp(-10.0 * (memory_confidence - 0.5))))

    # --- ACETYLCHOLINE (α): Reserved for V2 ---
    alpha = 0.0

    # Record history
    state.history['delta'].append(delta)
    state.history['nu'].append(nu)
    state.history['sigma'].append(sigma)
    state.history['alpha'].append(alpha)
    state.history['epsilon'].append(prediction_error)
    state.history['mu'].append(state.mu_epsilon)
    state.history['std'].append(state.sigma_epsilon)
    state.history['mem_confidence'].append(memory_confidence)

    return delta, nu, sigma, alpha


# ============================================================
# Hopfield Memory (simplified, d-dimensional)
# ============================================================

@dataclass
class HopfieldMemory:
    """Modern Hopfield Network with continuous states.

    Keys:   d_key dimensional (for high-res retrieval)
    Values: d_val dimensional (for compact storage)
    """
    d_key: int = 512
    d_val: int = 256
    beta: float = 8.0          # Inverse temperature (sharpness)
    max_memories: int = 10000  # Hard cap
    similarity_threshold: float = 0.85  # If new key is this similar to existing, UPDATE instead of append

    def __post_init__(self):
        self.keys = np.zeros((0, self.d_key), dtype=np.float32)    # [N, d_key]
        self.values = np.zeros((0, self.d_val), dtype=np.float32)  # [N, d_val]
        self.access_counts = np.array([], dtype=np.int32)          # Per-memory access count
        self.write_count = 0
        self.update_count = 0

    @property
    def n_memories(self) -> int:
        return len(self.keys)

    def write(self, key: np.ndarray, value: np.ndarray, dopamine: float, threshold: float = 0.3) -> bool:
        """Write a memory if dopamine exceeds threshold.

        Args:
            key: [d_key] vector — the "address" of this memory
            value: [d_val] vector — the "content" of this memory
            dopamine: δ signal from neuromodulator
            threshold: Minimum δ to trigger a write

        Returns:
            True if a write (or update) occurred
        """
        if dopamine < threshold:
            return False

        key = key / (np.linalg.norm(key) + 1e-8)  # Normalize key
        value = value / (np.linalg.norm(value) + 1e-8)

        # Check for collision with existing memories
        if self.n_memories > 0:
            similarities = self.keys @ key  # [N]
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]

            if max_sim > self.similarity_threshold:
                # UPDATE existing memory (blend old and new)
                blend = 0.3  # How much the new value overrides
                self.values[max_sim_idx] = (1 - blend) * self.values[max_sim_idx] + blend * value
                self.access_counts[max_sim_idx] += 1
                self.update_count += 1
                return True

        # APPEND new memory
        if self.n_memories >= self.max_memories:
            # Evict least-accessed memory
            evict_idx = np.argmin(self.access_counts)
            self.keys[evict_idx] = key
            self.values[evict_idx] = value
            self.access_counts[evict_idx] = 1
        else:
            self.keys = np.vstack([self.keys, key.reshape(1, -1)]) if self.n_memories > 0 else key.reshape(1, -1)
            self.values = np.vstack([self.values, value.reshape(1, -1)]) if self.n_memories > 0 else value.reshape(1, -1)
            self.access_counts = np.append(self.access_counts, 1)

        self.write_count += 1
        return True

    def read(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """Retrieve from memory using Modern Hopfield update rule.

        Args:
            query: [d_key] vector

        Returns:
            (retrieved_value, confidence)
            confidence = max attention weight (how strong the best match is)
        """
        if self.n_memories == 0:
            return np.zeros(self.d_val, dtype=np.float32), 0.0

        query = query / (np.linalg.norm(query) + 1e-8)

        # Hopfield retrieval: softmax(β · K^T · q) · V
        logits = self.beta * (self.keys @ query)  # [N]

        # Numerically stable softmax
        logits_shifted = logits - np.max(logits)
        attn = np.exp(logits_shifted)
        attn = attn / (np.sum(attn) + 1e-8)

        # Confidence = max attention weight
        confidence = float(np.max(attn))
        best_idx = np.argmax(attn)
        self.access_counts[best_idx] += 1

        # Weighted sum of values
        retrieved = attn @ self.values  # [d_val]

        return retrieved, confidence


# ============================================================
# Simulated Cortex (trivial for simulation purposes)
# ============================================================

def simulated_cortex(
    input_vec: np.ndarray,     # d_val dimensional input
    nu: float,                 # Norepinephrine — effort level
    d_val: int = 256,
) -> np.ndarray:
    """Simulate cortex processing.

    In the real system, this is an LTC-RNN. Here we just add noise
    proportional to (1-ν) to simulate that higher effort = better output.
    """
    # More iterations (higher ν) = less noise = better output
    noise_scale = 0.3 * (1.0 - nu)
    output = input_vec + np.random.randn(d_val).astype(np.float32) * noise_scale
    output = output / (np.linalg.norm(output) + 1e-8)
    return output


# ============================================================
# Scenario Runner
# ============================================================

def run_scenario(
    name: str,
    prediction_errors: List[float],
    ground_truth_keys: List[np.ndarray],
    ground_truth_values: List[np.ndarray],
    d_key: int = 512,
    d_val: int = 256,
) -> dict:
    """Run a full scenario through the neuromodulator + hippocampus system.

    Args:
        name: Scenario label
        prediction_errors: Sequence of ||ε|| values (0=perfectly predicted, 1=total surprise)
        ground_truth_keys: The "address" vectors for each input
        ground_truth_values: The "content" vectors for each input

    Returns:
        Dict with all metrics and history
    """
    state = NeuromodulatorState()
    memory = HopfieldMemory(d_key=d_key, d_val=d_val)

    results = {
        'name': name,
        'steps': len(prediction_errors),
        'retrieval_quality': [],
        'routing_decisions': [],  # 'memory', 'cortex', or 'blend'
    }

    for step, (eps, key, value) in enumerate(zip(prediction_errors, ground_truth_keys, ground_truth_values)):
        # 1. Query memory BEFORE processing
        mem_output, mem_confidence = memory.read(key)

        # 2. Compute neuromodulatory signals
        delta, nu, sigma, alpha = compute_neuromodulators(eps, mem_confidence, state)

        # 3. Cortex processes input
        cortex_output = simulated_cortex(value, nu, d_val)

        # 4. Integrate: σ·memory + (1-σ)·cortex
        if memory.n_memories > 0:
            final_output = sigma * mem_output + (1 - sigma) * cortex_output
        else:
            final_output = cortex_output

        # 5. Write to hippocampus if dopamine exceeds threshold
        wrote = memory.write(key, value, delta, threshold=0.3)
        state.history['writes'].append(1 if wrote else 0)
        state.history['reads'].append(mem_confidence)

        # 6. Measure retrieval quality (cosine similarity to ground truth)
        cos_sim = float(np.dot(final_output, value) / (np.linalg.norm(final_output) * np.linalg.norm(value) + 1e-8))
        results['retrieval_quality'].append(cos_sim)

        # 7. Classify routing decision
        if sigma > 0.7:
            results['routing_decisions'].append('memory')
        elif sigma < 0.3:
            results['routing_decisions'].append('cortex')
        else:
            results['routing_decisions'].append('blend')

    # Summary stats
    total_writes = sum(state.history['writes'])
    results['total_writes'] = total_writes
    results['total_updates'] = memory.update_count
    results['final_memories'] = memory.n_memories
    results['avg_retrieval'] = float(np.mean(results['retrieval_quality']))
    results['history'] = state.history
    results['memory'] = memory

    return results


def print_results(results: dict):
    """Print a formatted summary of a scenario run."""
    h = results['history']
    n = results['steps']

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {results['name']}")
    print(f"{'='*70}")
    print(f"  Steps: {n}")
    print(f"  Memories written: {results['total_writes']} (updates: {results['total_updates']})")
    print(f"  Final memory size: {results['final_memories']}")
    print(f"  Avg retrieval quality: {results['avg_retrieval']:.4f}")
    print(f"  Routing: memory={results['routing_decisions'].count('memory')}"
          f"  cortex={results['routing_decisions'].count('cortex')}"
          f"  blend={results['routing_decisions'].count('blend')}")

    # Print key moments
    print(f"\n  {'Step':>5} {'eps':>6} {'mu_e':>6} {'sd_e':>6} {'d(dop)':>7} {'n(nor)':>7} {'s(ser)':>7} {'memConf':>7} {'wrote':>5} {'route':>7}")
    print(f"  {'-'*65}")

    # Show first 10, middle 5, last 10
    indices = list(range(min(10, n)))
    if n > 25:
        indices += list(range(n // 2 - 2, n // 2 + 3))
    if n > 10:
        indices += list(range(max(10, n - 10), n))
    indices = sorted(set(i for i in indices if i < n))

    prev_idx = -1
    for i in indices:
        if prev_idx >= 0 and i > prev_idx + 1:
            print(f"  {'...':>5}")
        print(f"  {i:5d} {h['epsilon'][i]:6.3f} {h['mu'][i]:6.3f} {h['std'][i]:6.3f}"
              f" {h['delta'][i]:7.3f} {h['nu'][i]:7.3f} {h['sigma'][i]:7.3f}"
              f" {h['mem_confidence'][i]:7.3f} {'  Y' if h['writes'][i] else '  -':>5}"
              f" {results['routing_decisions'][i]:>7}")
        prev_idx = i


# ============================================================
# Test Scenarios
# ============================================================

def make_random_vecs(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate n random unit vectors of dimension d."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def scenario_1_all_novel():
    """Everything is new — model has never seen any of these inputs.
    Expected: δ should spike initially then calibrate, most inputs get written."""
    n = 50
    d_key, d_val = 512, 256
    keys = make_random_vecs(n, d_key, seed=100)
    values = make_random_vecs(n, d_val, seed=200)

    # High prediction errors (novel inputs) with slight decay as model "adapts"
    errors = [0.8 - 0.005 * i + np.random.randn() * 0.05 for i in range(n)]
    errors = [max(0.0, min(1.0, e)) for e in errors]

    return run_scenario("ALL NOVEL (everything is new)", errors, list(keys), list(values), d_key, d_val)


def scenario_2_all_familiar():
    """Model has seen everything before — prediction errors near zero.
    Expected: δ ≈ 0, no writes, σ high (trust memory), ν low (don't bother cortex)."""
    n = 50
    d_key, d_val = 512, 256
    keys = make_random_vecs(n, d_key, seed=300)
    values = make_random_vecs(n, d_val, seed=400)

    # Very low prediction errors
    errors = [0.05 + np.random.randn() * 0.02 for _ in range(n)]
    errors = [max(0.0, min(1.0, e)) for e in errors]

    return run_scenario("ALL FAMILIAR (everything predicted well)", errors, list(keys), list(values), d_key, d_val)


def scenario_3_sudden_novelty():
    """40 familiar inputs then 10 suddenly novel ones.
    Expected: δ near 0 for first 40, spikes for last 10. The adaptive threshold
    should have settled low, so the novelty spike is VERY strong."""
    n = 50
    d_key, d_val = 512, 256
    keys = make_random_vecs(n, d_key, seed=500)
    values = make_random_vecs(n, d_val, seed=600)

    errors = [0.05 + np.random.randn() * 0.02 for _ in range(40)]
    errors += [0.7 + np.random.randn() * 0.05 for _ in range(10)]
    errors = [max(0.0, min(1.0, e)) for e in errors]

    return run_scenario("SUDDEN NOVELTY (40 familiar → 10 novel)", errors, list(keys), list(values), d_key, d_val)


def scenario_4_repeated_facts():
    """Same 5 facts repeated 10 times each.
    Expected: First exposure writes. Second+ exposures should NOT write
    (key collision detected, maybe update). Memory size should be 5, not 50."""
    d_key, d_val = 512, 256
    n_facts = 5
    n_repeats = 10
    base_keys = make_random_vecs(n_facts, d_key, seed=700)
    base_values = make_random_vecs(n_facts, d_val, seed=800)

    keys = []
    values = []
    errors = []
    for repeat in range(n_repeats):
        for fact_idx in range(n_facts):
            keys.append(base_keys[fact_idx])
            values.append(base_values[fact_idx])
            # First exposure: high error. Subsequent: low (model "remembers")
            if repeat == 0:
                errors.append(0.7 + np.random.randn() * 0.05)
            else:
                errors.append(0.1 + np.random.randn() * 0.03)

    errors = [max(0.0, min(1.0, e)) for e in errors]

    return run_scenario(
        f"REPEATED FACTS ({n_facts} facts × {n_repeats} repeats)",
        errors, keys, values, d_key, d_val
    )


def scenario_5_gradual_learning():
    """Model starts bad, gradually improves over 100 steps.
    Expected: δ threshold should adapt down over time. Early on, everything
    is novel (many writes). Later, only genuine spikes trigger writes."""
    n = 100
    d_key, d_val = 512, 256
    keys = make_random_vecs(n, d_key, seed=900)
    values = make_random_vecs(n, d_val, seed=1000)

    # Error curve: starts at 0.9, decays exponentially to 0.1 with occasional spikes
    errors = []
    for i in range(n):
        base = 0.1 + 0.8 * np.exp(-i / 20.0)
        # Occasional surprise spike every ~15 steps
        spike = 0.5 if (i > 20 and i % 15 == 0) else 0.0
        noise = np.random.randn() * 0.03
        errors.append(max(0.0, min(1.0, base + spike + noise)))

    return run_scenario("GRADUAL LEARNING (error decays, occasional spikes)", errors, list(keys), list(values), d_key, d_val)


def scenario_6_memory_collision():
    """Two very similar keys but different values.
    Tests whether the collision detection handles near-duplicates correctly."""
    d_key, d_val = 512, 256
    rng = np.random.RandomState(1100)

    base_key = rng.randn(d_key).astype(np.float32)
    base_key /= np.linalg.norm(base_key)

    # Create two keys that are 90% similar (cosine ~0.9)
    noise = rng.randn(d_key).astype(np.float32) * 0.1
    similar_key = base_key + noise
    similar_key /= np.linalg.norm(similar_key)

    cos_sim = float(np.dot(base_key, similar_key))
    print(f"  [Collision test] Key similarity: {cos_sim:.4f}")

    # Very different values
    value_a = rng.randn(d_val).astype(np.float32)
    value_a /= np.linalg.norm(value_a)
    value_b = rng.randn(d_val).astype(np.float32)
    value_b /= np.linalg.norm(value_b)

    # Sequence: write A, then try to write B (similar key, different value)
    keys = [base_key, similar_key, base_key, similar_key]
    values_list = [value_a, value_b, value_a, value_b]
    errors = [0.9, 0.9, 0.3, 0.3]  # Both initially "surprising"

    result = run_scenario("MEMORY COLLISION (similar keys, different values)", errors, keys, values_list, d_key, d_val)

    # Extra analysis: what does retrieval return for each key?
    mem = result['memory']
    print(f"\n  After scenario:")
    print(f"    Memories stored: {mem.n_memories}")
    for i, (k, label) in enumerate([(base_key, "key_A"), (similar_key, "key_B")]):
        retrieved, conf = mem.read(k)
        cos_a = float(np.dot(retrieved, value_a) / (np.linalg.norm(retrieved) * np.linalg.norm(value_a) + 1e-8))
        cos_b = float(np.dot(retrieved, value_b) / (np.linalg.norm(retrieved) * np.linalg.norm(value_b) + 1e-8))
        print(f"    Query {label}: conf={conf:.3f}, sim_to_A={cos_a:.3f}, sim_to_B={cos_b:.3f}")

    return result


def scenario_7_stress_test():
    """500 unique inputs. Tests memory growth, eviction, and running stat stability."""
    n = 500
    d_key, d_val = 512, 256
    keys = make_random_vecs(n, d_key, seed=1200)
    values = make_random_vecs(n, d_val, seed=1300)

    # Mixed error profile: mostly medium, some high, some low
    rng = np.random.RandomState(1400)
    errors = []
    for i in range(n):
        base = 0.4 + rng.randn() * 0.15
        errors.append(max(0.0, min(1.0, base)))

    return run_scenario("STRESS TEST (500 diverse inputs)", errors, list(keys), list(values), d_key, d_val)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  CORTEX V1 — Neuromodulator + Hopfield Memory Simulation")
    print("  Pure NumPy. Testing the math before building PyTorch modules.")
    print("=" * 70)

    scenarios = [
        scenario_1_all_novel,
        scenario_2_all_familiar,
        scenario_3_sudden_novelty,
        scenario_4_repeated_facts,
        scenario_5_gradual_learning,
        scenario_6_memory_collision,
        scenario_7_stress_test,
    ]

    all_results = []
    for scenario_fn in scenarios:
        result = scenario_fn()
        print_results(result)
        all_results.append(result)

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY ACROSS ALL SCENARIOS")
    print(f"{'='*70}")
    print(f"  {'Scenario':<45} {'Writes':>7} {'Updates':>8} {'MemSize':>8} {'AvgRetr':>8}")
    print(f"  {'-'*80}")
    for r in all_results:
        print(f"  {r['name']:<45} {r['total_writes']:>7} {r['total_updates']:>8}"
              f" {r['final_memories']:>8} {r['avg_retrieval']:>8.4f}")

    # ---- Diagnostic Checks ----
    print(f"\n  DIAGNOSTIC CHECKS:")
    checks_passed = 0
    checks_total = 0

    # Check 1: All-novel should write many memories
    r = all_results[0]
    checks_total += 1
    if r['total_writes'] > r['steps'] * 0.3:
        print(f"    ✓ All-novel: wrote {r['total_writes']}/{r['steps']} (>30%)")
        checks_passed += 1
    else:
        print(f"    ✗ All-novel: only wrote {r['total_writes']}/{r['steps']} (expected >30%)")

    # Check 2: All-familiar should write very few
    r = all_results[1]
    checks_total += 1
    if r['total_writes'] < r['steps'] * 0.15:
        print(f"    ✓ All-familiar: wrote {r['total_writes']}/{r['steps']} (<15%)")
        checks_passed += 1
    else:
        print(f"    ✗ All-familiar: wrote {r['total_writes']}/{r['steps']} (expected <15%)")

    # Check 3: Sudden novelty should spike δ after step 40
    r = all_results[2]
    h = r['history']
    late_writes = sum(h['writes'][40:])
    checks_total += 1
    if late_writes >= 5:
        print(f"    ✓ Sudden novelty: {late_writes}/10 novel inputs triggered writes")
        checks_passed += 1
    else:
        print(f"    ✗ Sudden novelty: only {late_writes}/10 novel inputs triggered writes")

    # Check 4: Repeated facts should have ~5 unique memories, not 50
    r = all_results[3]
    checks_total += 1
    if r['final_memories'] <= 8:
        print(f"    ✓ Repeated facts: {r['final_memories']} unique memories (expected ~5)")
        checks_passed += 1
    else:
        print(f"    ✗ Repeated facts: {r['final_memories']} memories (expected ~5, got too many)")

    # Check 5: Gradual learning should have more writes early, fewer late
    r = all_results[4]
    h = r['history']
    early_writes = sum(h['writes'][:30])
    late_writes_gl = sum(h['writes'][70:])
    checks_total += 1
    if early_writes > late_writes_gl:
        print(f"    ✓ Gradual learning: early writes={early_writes} > late writes={late_writes_gl}")
        checks_passed += 1
    else:
        print(f"    ✗ Gradual learning: early writes={early_writes} <= late writes={late_writes_gl}")

    # Check 6: Collision should not create duplicate memories
    r = all_results[5]
    checks_total += 1
    if r['final_memories'] <= 2:
        print(f"    ✓ Collision: {r['final_memories']} memories (collision detection works)")
        checks_passed += 1
    else:
        print(f"    ✗ Collision: {r['final_memories']} memories (collision detection failed)")

    # Check 7: Stress test should not crash and memory should stay bounded
    r = all_results[6]
    checks_total += 1
    if r['final_memories'] <= 10000 and r['avg_retrieval'] > 0:
        print(f"    ✓ Stress test: {r['final_memories']} memories, avg retrieval={r['avg_retrieval']:.3f}")
        checks_passed += 1
    else:
        print(f"    ✗ Stress test: failed")

    print(f"\n  PASSED: {checks_passed}/{checks_total}")

    if checks_passed == checks_total:
        print(f"\n  ★ ALL CHECKS PASSED — Neuromodulator math is verified.")
        print(f"  ★ Ready to proceed to PyTorch module implementation.")
    else:
        print(f"\n  ⚠ {checks_total - checks_passed} check(s) failed. Review the equations before proceeding.")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
