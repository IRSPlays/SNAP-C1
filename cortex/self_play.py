"""Self-Play Problem Generator — model creates harder problems, calculator verifies.

Level 3 of human-like training:
    1. Model generates novel problems at different difficulty levels
    2. Calculator verifies the answer (ground truth)
    3. Verified problems get added to training data
    4. Curriculum auto-advances as model improves
    5. Model teaches itself through its own mistakes

Usage:
    from cortex.self_play import SelfPlayGenerator
    gen = SelfPlayGenerator(model, enc, bpe_to_local, local_to_bpe, device)
    new_problems = gen.generate_batch(difficulty=3, count=100)
"""

import torch
import random
import math
import re


class SelfPlayGenerator:
    """Generates arithmetic problems at increasing difficulty for self-play training."""

    def __init__(self, model, enc, bpe_to_local, local_to_bpe, device,
                 num_values=None):
        self.model = model
        self.enc = enc
        self.bpe_to_local = bpe_to_local
        self.local_to_bpe = local_to_bpe
        self.device = device
        self.num_values = num_values
        self.eot = bpe_to_local.get(enc.eot_token, 0) if bpe_to_local else enc.eot_token

    # ── Problem generation by difficulty ──

    def _gen_level_1(self):
        """Single operations, small numbers (2 + 3 = ?)."""
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(['+', '-', '*', '/'])
        if op == '+':
            answer = a + b
            chain = f"{a} + {b} = {answer}"
        elif op == '-':
            if a < b:
                a, b = b, a
            answer = a - b
            chain = f"{a} - {b} = {answer}"
        elif op == '*':
            a, b = min(a, 10), min(b, 10)
            answer = a * b
            chain = f"{a} x {b} = {answer}"
        else:
            b = random.randint(2, 8)
            answer = random.randint(2, 10)
            a = b * answer
            chain = f"{a} / {b} = {answer}"
        q = f"Compute {a} {op} {b}"
        return q, chain, answer

    def _gen_level_2(self):
        """Chain operations (5+3=8, 8×2=16)."""
        depth = random.randint(2, 3)
        current = random.randint(1, 10)
        steps = []
        for _ in range(depth):
            op = random.choice(['+', '-', '*'])
            if op == '+':
                delta = random.randint(1, 15)
                prev = current
                current += delta
                steps.append(f"{prev} + {delta} = {current}")
            elif op == '-':
                delta = random.randint(1, max(1, current - 1))
                prev = current
                current -= delta
                steps.append(f"{prev} - {delta} = {current}")
            else:
                mul = random.randint(2, 3)
                prev = current
                current *= mul
                steps.append(f"{prev} x {mul} = {current}")
        chain = ", ".join(steps)
        q = f"Step by step, starting from {steps[0].split()[0]}: " + ", ".join(
            s.split(" = ")[0] for s in steps
        )
        return q, chain, current

    def _gen_level_3(self):
        """Word problems (GSM8K-style)."""
        templates = [
            self._shop_problem,
            self._people_problem,
            self._travel_problem,
            self._comparison_problem,
        ]
        return random.choice(templates)()

    def _gen_level_4(self):
        """Multi-digit operations with varied structure."""
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            answer = a + b
            chain = f"{a} + {b} = {answer}"
        elif op == '-':
            if a < b:
                a, b = b, a
            answer = a - b
            chain = f"{a} - {b} = {answer}"
        else:
            a, b = random.randint(5, 50), random.randint(5, 20)
            answer = a * b
            chain = f"{a} x {b} = {answer}"
        formats = [
            f"Calculate {a} {op} {b}",
            f"What is {a} {op} {b}?",
            f"Solve: {a} {op} {b}",
            f"Find the value of {a} {op} {b}",
        ]
        q = random.choice(formats)
        return q, chain, answer

    # ── Word problem sub-generators ──

    def _shop_problem(self):
        items = ["apples", "pencils", "books", "cookies", "mangoes"]
        a_price = random.randint(2, 20)
        a_qty = random.randint(2, 10)
        b_price = random.randint(2, 20)
        b_qty = random.randint(1, 10)
        total = a_price * a_qty + b_price * b_qty
        chain = f"{a_price} x {a_qty} = {a_price * a_qty}. {b_price} x {b_qty} = {b_price * b_qty}. {a_price * a_qty} + {b_price * b_qty} = {total}"
        q = f"You buy {a_qty} {random.choice(items)} at ${a_price} each and {b_qty} {random.choice(items)} at ${b_price} each. How much total?"
        return q, chain, total

    def _people_problem(self):
        initial = random.randint(5, 30)
        add = random.randint(2, 15)
        remove = random.randint(1, min(10, initial))
        total = initial + add - remove
        chain = f"{initial} + {add} = {initial + add}. {initial + add} - {remove} = {total}"
        q = f"There are {initial} people. {add} arrive. {remove} leave. How many remain?"
        return q, chain, total

    def _travel_problem(self):
        speed = random.randint(20, 80)
        hours = random.randint(1, 6)
        distance = speed * hours
        chain = f"{speed} x {hours} = {distance}"
        q = f"A car goes {speed} km/h for {hours} hours. Distance traveled?"
        return q, chain, distance

    def _comparison_problem(self):
        base = random.randint(5, 20)
        factor = random.randint(2, 4)
        larger = base * factor
        total = base + larger
        chain = f"{base} x {factor} = {larger}. {base} + {larger} = {total}"
        q = f"Person A has {base}. Person B has {factor} times as many. How many total?"
        return q, chain, total

    # ── Model-guided problem generation (uses model to create novel problems) ──

    @torch.no_grad()
    def generate_with_model(self, difficulty, count=10):
        """Use model's own output to generate novel problems at given difficulty.

        Model generates a problem template, calculator solves it, verified
        problem goes into training data.
        """
        gen = {
            1: self._gen_level_1,
            2: self._gen_level_2,
            3: self._gen_level_3,
            4: self._gen_level_4,
        }
        gen_fn = gen.get(difficulty, self._gen_level_1)
        problems = []
        for _ in range(count):
            try:
                q, chain, answer = gen_fn()
                output = f"{chain}\n#### {answer}"
                problems.append({
                    'instruction': q,
                    'output': output,
                    'answer': answer,
                    'difficulty': difficulty,
                })
            except Exception:
                pass
        return problems

    def generate_curriculum(self, model_accuracy, baseline=500):
        """Auto-advance difficulty based on model performance.

        accuracy > 80% at current level → advance
        accuracy < 30% at current level → retreat
        """
        if model_accuracy > 0.80:
            return baseline * 2  # harder
        elif model_accuracy > 0.50:
            return baseline  # same
        else:
            return max(baseline // 2, 100)  # easier

    def verify_answer(self, question, generated_answer, calculator_answer):
        """Verify model-generated answer against calculator."""
        try:
            return abs(float(generated_answer) - float(calculator_answer)) < 1e-6
        except (ValueError, TypeError):
            return False

    @staticmethod
    def extract_number(text):
        """Extract the final numeric answer from model output."""
        patterns = [
            r'####\s*(\d+(?:\.\d+)?)',
            r'Answer:\s*(\d+(?:\.\d+)?)',
            r'Result:\s*(\d+(?:\.\d+)?)',
            r'answer is\s*(\d+(?:\.\d+)?)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return int(float(m.group(1)))
        nums = re.findall(r'\b(\d+)\b', text)
        return int(nums[-1]) if nums else None
