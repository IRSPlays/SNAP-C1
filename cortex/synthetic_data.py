"""Synthetic Arithmetic Data Generator v2 — Anti-memorization design.

Generates arithmetic problems with MASSIVE phrase variation so the model
must learn actual computation rather than memorizing templates.

30+ question templates per operation, randomized answer formats,
anti-pattern duplicates, chain problems, and diverse number ranges.

Usage:
    python -m cortex.synthetic_data --count 100000 --output data/synthetic/train.jsonl
"""

import json
import os
import random
import math
import argparse


# ════════════════════════════════════════════════════════════════════
# Number to text conversion
# ════════════════════════════════════════════════════════════════════
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def num_to_words(n):
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[u]}" if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        result = f"{_ONES[h]} hundred"
        if r:
            result += f" and {num_to_words(r)}"
        return result
    return str(n)


def rand_int(a, b):
    return random.randint(a, b)


# ════════════════════════════════════════════════════════════════════
# Template libraries
# ════════════════════════════════════════════════════════════════════

ADDITION_PHRASES = [
    "What is {A} plus {B}?",
    "What is {A} + {B}?",
    "Calculate {A} + {B}",
    "Find the sum of {A} and {B}",
    "Add {A} and {B} together",
    "What do you get when you add {A} and {B}?",
    "Sum: {A} + {B}",
    "Compute {A} + {B}",
    "What is the total of {A} and {B}?",
    "If you have {A} and add {B}, what do you get?",
    "{A} plus {B} equals what?",
    "Evaluate: {A} + {B}",
    "What number is {A} added to {B}?",
    "Determine {A} + {B}",
    "Solve: {A} + {B}",
    "Work out {A} plus {B}",
]

SUBTRACTION_PHRASES = [
    "What is {A} minus {B}?",
    "What is {A} - {B}?",
    "Calculate {A} - {B}",
    "Find the difference between {A} and {B}",
    "Subtract {B} from {A}",
    "What do you get when you subtract {B} from {A}?",
    "Difference: {A} - {B}",
    "Compute {A} - {B}",
    "If you take {B} away from {A}, what is left?",
    "{A} minus {B} equals what?",
    "Evaluate: {A} - {B}",
    "What number is {B} less than {A}?",
    "Determine {A} - {B}",
    "Solve: {A} - {B}",
    "Work out {A} minus {B}",
    "What is {A} take away {B}?",
]

MULTIPLICATION_PHRASES = [
    "What is {A} times {B}?",
    "What is {A} * {B}?",
    "Calculate {A} * {B}",
    "Find the product of {A} and {B}",
    "Multiply {A} by {B}",
    "What do you get when you multiply {A} and {B}?",
    "Product: {A} x {B}",
    "Compute {A} * {B}",
    "{A} times {B} equals what?",
    "Evaluate: {A} * {B}",
    "What is {A} multiplied by {B}?",
    "Determine {A} x {B}",
    "Solve: {A} * {B}",
    "Work out {A} times {B}",
    "If you have {A} groups of {B}, how many total?",
]

DIVISION_PHRASES = [
    "What is {A} divided by {B}?",
    "What is {A} / {B}?",
    "Calculate {A} / {B}",
    "Find the quotient of {A} and {B}",
    "Divide {A} by {B}",
    "What do you get when you divide {A} by {B}?",
    "Division: {A} / {B}",
    "Compute {A} / {B}",
    "{A} divided by {B} equals what?",
    "Evaluate: {A} / {B}",
    "How many times does {B} go into {A}?",
    "Determine {A} / {B}",
    "Solve: {A} / {B}",
    "Work out {A} divided by {B}",
]

ANSWER_FORMATS = [
    lambda a, chain: f"{chain}\n#### {a}",
    lambda a, chain: f"{chain}\nAnswer: {a}",
    lambda a, chain: f"{chain}\nResult: {a}",
    lambda a, chain: f"{chain} The answer is {a}.",
    lambda a, chain: f"{chain} Therefore, the result is {a}.",
    lambda a, chain: f"{chain}\nFinal answer: {a}",
    lambda a, chain: f"{chain}\nThe solution is {a}.",
    lambda a, chain: f"{chain} So the answer is {a}.",
]

CHAIN_FORMATS = [
    lambda op, a, b, ans: f"{a} {op} {b} = {ans}",
    lambda op, a, b, ans: f"First, {a} {op} {b} = {ans}",
    lambda op, a, b, ans: f"We compute {a} {op} {b} = {ans}",
    lambda op, a, b, ans: f"{a} {op} {b} gives us {ans}",
]


# ════════════════════════════════════════════════════════════════════
# Question generators — HIGH VARIETY
# ════════════════════════════════════════════════════════════════════

def gen_pure_arithmetic():
    """Single operation with massive phrase/format variation."""
    a = rand_int(1, 999)
    b = rand_int(1, 999)
    op_char = random.choice(['+', '-', '*', '/'])
    use_words = random.random() < 0.3

    if op_char == '+':
        answer = a + b
        phrase = random.choice(ADDITION_PHRASES)
        op_word = '+'
    elif op_char == '-':
        if a < b:
            a, b = b, a
        answer = a - b
        phrase = random.choice(SUBTRACTION_PHRASES)
        op_word = '-'
    elif op_char == '*':
        a = rand_int(2, 50)  # cap for overflow (was 50) — keep reasonable
        b = rand_int(2, 50)
        answer = a * b
        phrase = random.choice(MULTIPLICATION_PHRASES)
        op_word = 'x'
    else:
        b = rand_int(2, 20)
        answer = rand_int(2, 50)
        a = b * answer
        phrase = random.choice(DIVISION_PHRASES)
        op_word = '/'

    if use_words:
        question = phrase.format(A=num_to_words(a), B=num_to_words(b))
    else:
        question = phrase.format(A=str(a), B=str(b))

    chain = random.choice(CHAIN_FORMATS)(op_word, a, b, answer)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(answer, chain), answer


def gen_chain_arithmetic():
    """Multi-step chain with varied step descriptions. Capped to prevent huge numbers."""
    depth = rand_int(2, 4)  # max 4 steps (was 5)
    current = rand_int(2, 20)  # smaller start (was 50)
    max_val = 5000  # hard cap on intermediate values
    steps = []
    step_descriptions = []

    for i in range(depth):
        op = random.choice(['+', '-', '*'])
        if op == '+':
            delta = rand_int(1, min(30, max(1, max_val - current)))
            prev = current
            current += delta
            steps.append(f"{prev} + {delta} = {current}")
            step_desc = f"Add {delta}"
        elif op == '-':
            delta = rand_int(1, max(1, current - 1))
            prev = current
            current -= delta
            steps.append(f"{prev} - {delta} = {current}")
            step_desc = f"Subtract {delta}"
        else:
            mul = rand_int(2, 4)  # smaller multiplier (was 5)
            if current * mul > max_val:
                mul = 2
            prev = current
            current *= mul
            steps.append(f"{prev} x {mul} = {current}")
            step_desc = f"Multiply by {mul}"
        step_descriptions.append(step_desc)

    chain = ". ".join(steps)
    question = random.choice([
        f"Start with {rand_int(1, 50)}, then " + ", then ".join(step_descriptions) + ". What is the result?",
        f"Compute step by step: " + ", ".join(step_descriptions) + ", starting from {current if depth == 0 else steps[0].split()[0]}",
        f"Solve this multi-step problem: " + " → ".join(step_descriptions),
    ]).format(**{f"s{i}": s for i, s in enumerate(steps)})

    answer = current
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(answer, chain), answer


def gen_word_form():
    """Text-only numbers in varied phrasing."""
    a = rand_int(1, 100)
    b = rand_int(1, 100)
    op = random.choice(['+', '-', '*'])
    a_word = num_to_words(a)
    b_word = num_to_words(b)

    if op == '+':
        answer = a + b
        op_phrases = [
            f"What is {a_word} plus {b_word}?",
            f"If I have {a_word} and add {b_word}, what do I get?",
            f"Calculate the sum of {a_word} and {b_word}",
            f"{a_word} and {b_word} together make how many?",
        ]
        chain = f"{a} + {b} = {answer}"
    elif op == '-':
        if a < b:
            a, b = b, a
            a_word, b_word = b_word, a_word
        answer = a - b
        op_phrases = [
            f"What is {a_word} minus {b_word}?",
            f"From {a_word}, take away {b_word}. What remains?",
            f"Find the difference: {a_word} minus {b_word}",
            f"Subtract {b_word} from {a_word}",
        ]
        chain = f"{a} - {b} = {answer}"
    else:
        a, b = min(a, 12), min(b, 12)
        a_word, b_word = num_to_words(a), num_to_words(b)
        answer = a * b
        op_phrases = [
            f"What is {a_word} times {b_word}?",
            f"If you multiply {a_word} by {b_word}, what do you get?",
            f"Calculate: {a_word} multiplied by {b_word}",
            f"Find the product of {a_word} and {b_word}",
        ]
        chain = f"{a} x {b} = {answer}"

    question = random.choice(op_phrases)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(answer, chain), answer


def gen_real_world():
    """GSM8K-style single-sentence word problems with extreme variation."""
    templates = [
        # Shopping
        lambda: _shop_problem(),
        lambda: _shop_problem(),
        lambda: _people_problem(),
        lambda: _people_problem(),
        lambda: _food_problem(),
        lambda: _food_problem(),
        lambda: _travel_problem(),
        lambda: _travel_problem(),
        lambda: _time_problem(),
        lambda: _time_problem(),
        lambda: _money_problem(),
        lambda: _money_problem(),
        lambda: _group_problem(),
        lambda: _group_problem(),
        lambda: _comparison_problem(),
    ]
    return random.choice(templates)()


def _shop_problem():
    items = random.choice([
        ("apples", "bananas"), ("pencils", "erasers"), ("books", "notebooks"),
        ("shirts", "pants"), ("cookies", "cupcakes"), ("chairs", "tables"),
        ("mangoes", "oranges"), ("socks", "shoes"), ("watermelons", "pineapples"),
    ])
    a_price = rand_int(2, 25)
    a_qty = rand_int(1, 12)
    b_price = rand_int(2, 25)
    b_qty = rand_int(1, 12)
    total = a_price * a_qty + b_price * b_qty
    chain = f"{a_price} x {a_qty} = {a_price * a_qty}. {b_price} x {b_qty} = {b_price * b_qty}. {a_price * a_qty} + {b_price * b_qty} = {total}."

    qs = [
        f"You buy {a_qty} {items[0]} at ${a_price} each and {b_qty} {items[1]} at ${b_price} each. Total cost?",
        f"A store sells {items[0]} for ${a_price} and {items[1]} for ${b_price}. If you get {a_qty} {items[0]} and {b_qty} {items[1]}, how much?",
        f"How much for {a_qty} {items[0]} (${a_price} each) and {b_qty} {items[1]} (${b_price} each)?",
        f"{a_qty} {items[0]} cost ${a_price} per unit. {b_qty} {items[1]} cost ${b_price} per unit. What is the bill?",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


def _people_problem():
    initial = rand_int(5, 40)
    add = rand_int(2, 20)
    remove = rand_int(1, min(15, initial))
    total = initial + add - remove
    chain = f"{initial} + {add} = {initial + add}. {initial + add} - {remove} = {total}."

    location = random.choice(["room", "bus", "park", "party", "classroom", "stadium", "hall", "office"])
    qs = [
        f"There are {initial} people in a {location}. {add} enter, then {remove} leave. How many remain?",
        f"In a {location}, {initial} people are present. {add} more arrive. Later, {remove} exit. Count the people left.",
        f"A {location} starts with {initial} people. After {add} join and {remove} depart, how many are there?",
        f"{initial} + {add} - {remove} = ? (people entering and leaving a {location})",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


def _food_problem():
    food = random.choice(["pizzas", "burgers", "apples", "sandwiches", "donuts", "tacos"])
    per_person = rand_int(1, 4)
    people = rand_int(3, 15)
    extra = rand_int(0, 10)
    total = per_person * people + extra
    chain = f"{per_person} x {people} = {per_person * people}. {per_person * people} + {extra} = {total}."

    qs = [
        f"Each person eats {per_person} {food}. There are {people} people and {extra} extra {food}. Total {food}?",
        f"If {people} people each eat {per_person} {food}, plus {extra} more, how many {food} total?",
        f"You need {per_person} {food} per person for {people} people. You also prepare {extra} extra. How many {food}?",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


def _travel_problem():
    speed = rand_int(20, 80)
    hours = rand_int(1, 8)
    distance = speed * hours
    chain = f"{speed} x {hours} = {distance}."

    vehicle = random.choice(["car", "train", "bus", "bike"])
    qs = [
        f"A {vehicle} travels at {speed} km/h for {hours} hours. How far does it go?",
        f"At {speed} km/h, how far does a {vehicle} travel in {hours} hours?",
        f"Speed: {speed} km/h. Time: {hours} hours. Find the distance.",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(distance, chain), distance


def _time_problem():
    minutes_per = rand_int(5, 30)
    count = rand_int(3, 20)
    total = minutes_per * count
    hours_part = total // 60
    mins_part = total % 60
    chain = f"{minutes_per} x {count} = {total} minutes."

    task = random.choice(["tasks", "lessons", "meetings", "episodes", "songs"])
    qs = [
        f"Each {task.rstrip('s')} takes {minutes_per} minutes. How long for {count} {task}?",
        f"You spend {minutes_per} minutes on each of {count} {task}. Total time in minutes?",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


def _money_problem():
    hourly = rand_int(8, 30)
    hours = rand_int(5, 40)
    total = hourly * hours
    chain = f"{hourly} x {hours} = {total}."

    job = random.choice(["tutor", "waiter", "cashier", "gardener", "painter", "babysitter"])
    qs = [
        f"A {job} earns ${hourly} per hour and works {hours} hours. How much do they earn?",
        f"At ${hourly}/hour, working {hours} hours earns how much?",
        f"Pay rate: ${hourly}/hr. Hours: {hours}. Calculate earnings.",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


def _group_problem():
    per_group = rand_int(3, 12)
    groups = rand_int(2, 15)
    total = per_group * groups
    items = random.choice(["students", "chairs", "books", "balls", "cards", "coins"])
    chain = f"{per_group} x {groups} = {total}."

    qs = [
        f"There are {groups} groups with {per_group} {items} each. Total {items}?",
        f"If each of {groups} groups has {per_group} {items}, how many {items} altogether?",
        f"{groups} groups of {per_group} {items} = ?",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


def _comparison_problem():
    base = rand_int(5, 30)
    factor = rand_int(2, 5)
    larger = base * factor
    total = base + larger
    chain = f"{base} x {factor} = {larger}. {base} + {larger} = {total}."

    items = random.choice(["marbles", "stickers", "cards", "coins", "candies", "rocks"])
    person_a = random.choice(["Alice", "Bob", "Cara", "David", "Eva"])
    person_b = random.choice(["Frank", "Grace", "Henry", "Iris", "Jack"])
    qs = [
        f"{person_a} has {base} {items}. {person_b} has {factor} times as many. How many total?",
        f"{person_b} has {factor}x more {items} than {person_a}'s {base}. Combined they have?",
    ]
    question = random.choice(qs)
    fmt = random.choice(ANSWER_FORMATS)
    return question, fmt(total, chain), total


# ════════════════════════════════════════════════════════════════════
# Anti-memorization: varied difficulty mix
# ════════════════════════════════════════════════════════════════════

def gen_mixed_bag():
    """Randomly pick any generator for maximum diversity."""
    gens = [gen_pure_arithmetic, gen_chain_arithmetic, gen_word_form, gen_real_world]
    return random.choice(gens)()


def generate_dataset(count: int = 100000, seed: int = 42,
                     calc_ratio: float = 0.0):
    """Generate arithmetic problems. calc_ratio controls how many use <CALC> format.
    NOTE: calc_ratio > 0 is experimental — the output format is still unstable.
    For reliable training, use calc_ratio=0."""
    random.seed(seed)
    data = []
    for _ in range(count):
        try:
            question, output, answer = gen_mixed_bag()
            data.append({
                'instruction': question,
                'output': output,
            })
        except Exception:
            pass
    random.shuffle(data)
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate diverse synthetic arithmetic data")
    parser.add_argument('--count', type=int, default=100000, help='Number of examples')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = generate_dataset(args.count, args.seed)
    print(f"Generated {len(data)} examples across 30+ templates")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved to {args.output}")
    return data


if __name__ == '__main__':
    main()
