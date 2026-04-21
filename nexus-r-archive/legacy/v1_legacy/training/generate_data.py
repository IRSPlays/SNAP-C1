"""
SNAP-C1 Synthetic Training Data Generator
==========================================
Generates training data for all SNAP-C1 capabilities:
- Team Thinking (multi-perspective reasoning)
- Self-Correction (review → fix → validate)
- Tool Use (structured tool calls)

Uses templates + manual examples as seeds, then can optionally
use a running LLM to expand the dataset synthetically.

Usage:
    python generate_data.py --capability team_thinking --count 100
    python generate_data.py --capability self_correction --count 100
    python generate_data.py --capability tool_use --count 100
    python generate_data.py --capability all --count 100
"""

import argparse
import json
import random
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# TEAM THINKING DATA TEMPLATES
# =============================================================================

TEAM_THINKING_SEEDS = [
    {
        "instruction": "Design a caching system for a web application that handles 10,000 requests per second.",
        "output": """<think>
[Architect] We need a multi-layer cache: L1 in-process (fastest, limited by RAM), L2 distributed cache (Redis/Memcached), and L3 CDN for static assets. The key design decisions are eviction policy, cache invalidation strategy, and consistency guarantees.

[Critic] A multi-layer cache adds complexity. What happens during cache stampede (many requests for expired key simultaneously)? What about cache poisoning? Also, 10K rps is not that high — are we over-engineering? We need to profile first to identify actual bottlenecks.

[Researcher] Industry benchmarks show Redis handles 100K+ ops/sec on a single node. Cache stampede is typically solved with request coalescing (singleflight pattern) or probabilistic early expiration. Facebook's Memcached paper (2013) describes scaling to billions of requests with lease-based invalidation.

[Implementer] Concrete stack: Application-level LRU cache (e.g., Python's `cachetools` or Go's `groupcache`), Redis cluster for distributed state, Varnish/CloudFlare for edge caching. TTL-based expiration with write-through invalidation. Implement singleflight to prevent stampede.

[Synthesizer] The optimal approach is a 3-tier cache with singleflight protection. Start with Redis + application LRU (covers 90% of cases), add CDN only if static asset latency is a measured problem. Use write-through invalidation for consistency. Profile before adding complexity.
</think>

Here's the recommended caching architecture for 10K rps:

**Architecture: 3-Tier Cache with Stampede Protection**

1. **L1 - In-Process Cache**: Application-level LRU cache (e.g., `cachetools`). Sub-millisecond access, limited by process memory. TTL: 30-60 seconds.

2. **L2 - Distributed Cache**: Redis cluster. Handles cache misses from L1. TTL: 5-15 minutes. Use write-through invalidation on data mutations.

3. **L3 - Edge Cache** (optional): CDN for static assets only. Add when profiling shows static asset latency is a bottleneck.

**Key protections:**
- **Singleflight pattern**: Coalesce concurrent requests for the same expired key into one database query
- **Probabilistic early expiration**: Randomly refresh keys before TTL to prevent synchronized expiration
- **Circuit breaker**: If the database is slow, serve stale cache data rather than cascading failures

Start with L1 + L2 and measure. Don't add L3 until you have data showing it's needed."""
    },
    {
        "instruction": "Should I use microservices or a monolith for a new startup MVP?",
        "output": """<think>
[Architect] For an MVP, the primary constraints are speed of development and ability to iterate. A monolith provides simpler deployment, easier debugging, and faster development. Microservices add operational overhead (service discovery, distributed tracing, API contracts) that slows down early development.

[Critic] The "start with a monolith" advice is common but not universal. If the team already has microservices expertise and CI/CD infrastructure, the overhead is lower. Also, if the product has genuinely independent domains (e.g., a marketplace with separate buyer/seller/payment systems), microservices from day 1 might prevent painful extraction later.

[Researcher] Basecamp, Shopify, and Stack Overflow all scaled as monoliths well past their MVP stage. Amazon and Netflix moved to microservices after reaching scaling limits. Martin Fowler's "MonolithFirst" pattern recommends starting monolithic because it's hard to get service boundaries right without domain understanding.

[Implementer] Practical approach: Build a modular monolith. Single deployable unit, but internally organized into clean modules with clear boundaries (separate directories, defined interfaces between modules). This gives you monolith simplicity now with a clear migration path to microservices later when you actually need to scale independently.

[Synthesizer] Start with a modular monolith. You get the speed of monolith development with the organizational discipline that makes future extraction to microservices straightforward. The key is clean module boundaries from day 1, not microservices infrastructure.
</think>

**Go with a modular monolith for your MVP.** Here's why:

1. **Speed**: Single codebase, single deployment, single database. You ship faster.
2. **Debugging**: One process, one log stream. Stack traces make sense.
3. **Iteration**: Refactoring across module boundaries is trivial in a monolith, painful across services.

**But do it right — modular monolith, not a big ball of mud:**
- Organize code into clean modules (e.g., `users/`, `billing/`, `products/`)
- Define explicit interfaces between modules (no reaching into another module's database tables)
- Each module owns its own data models

This gives you a clear extraction path when (if) you need microservices later. Most startups never reach the scale where microservices are necessary. Don't solve problems you don't have yet."""
    },
    {
        "instruction": "Explain how a neural network learns, for someone who knows basic programming but no ML.",
        "output": """<think>
[Architect] Structure the explanation in layers: (1) What a neural network IS (function that maps inputs to outputs), (2) What "learning" means (adjusting parameters to minimize error), (3) How it adjusts (gradient descent + backpropagation), (4) Concrete analogy to make it stick.

[Critic] Most explanations of neural networks lean too heavily on biological neuron analogies, which are misleading. The "brain" metaphor creates false expectations. Also, skipping the math entirely leaves people with a shallow understanding — we should include just enough math to be precise without overwhelming.

[Researcher] Effective pedagogy for neural networks: Karpathy's "Micrograd" video teaches backprop by building it from scratch. The key insight to convey is that learning = optimization = finding the minimum of a loss function. For programmers specifically, framing it as "automated parameter tuning" resonates well.

[Implementer] Use a concrete code analogy: a function with adjustable coefficients. Show how you'd manually tune parameters, then explain gradient descent as the automated version of that tuning process. Maybe show a 3-line pseudocode example.

[Synthesizer] Lead with programmer-friendly framing (function with tunable params), use gradient descent as "automated parameter search", skip the brain metaphor, include minimal pseudocode to make it concrete.
</think>

Think of a neural network as a **function with millions of adjustable knobs**.

**Step 1: The function**
```python
def predict(input, knobs):
    # Imagine thousands of multiply-and-add operations
    # chained together, with 'knobs' controlling each one
    result = input
    for layer_knobs in knobs:
        result = activate(result * layer_knobs.weights + layer_knobs.bias)
    return result
```

Each "knob" (called a **weight**) is just a number. The network has millions of them. Initially, they're random — so the outputs are garbage.

**Step 2: Measuring how wrong it is**
```python
error = how_wrong(predict(input, knobs), correct_answer)
# e.g., error = (predicted - actual) ** 2
```

This is called the **loss function**. It gives you a single number: how far off the prediction is.

**Step 3: Adjusting the knobs (learning)**
Here's the key insight: calculus lets us compute, for each knob, "if I turn this knob slightly up, does the error go up or down?"

That's called the **gradient** — it's a direction. Then we nudge each knob slightly in the direction that reduces the error:

```python
for knob in all_knobs:
    direction = compute_gradient(error, knob)  # calculus magic
    knob -= learning_rate * direction           # nudge it
```

**Step 4: Repeat a million times**
Feed in thousands of examples, compute the error each time, nudge the knobs each time. Gradually, the knobs settle into values that produce good predictions.

That's it. "Learning" = running this loop until the error is small enough. No magic, no consciousness — just automated knob-turning guided by calculus."""
    },
]

TEAM_THINKING_QUERY_TEMPLATES = [
    "How would you implement {topic} in a production system?",
    "What are the tradeoffs between {option_a} and {option_b}?",
    "Debug this issue: {problem_description}",
    "Design a system that handles {requirement}",
    "Explain {concept} to someone who understands {prerequisite}",
    "Review this approach: {approach_description}",
    "What's the best way to {task} given {constraints}?",
    "How should I structure {project_type} for {goal}?",
]

TEAM_THINKING_TOPICS = [
    ("a rate limiter", "API gateway", "token bucket vs sliding window"),
    ("a real-time notification system", "WebSocket", "polling vs SSE vs WebSocket"),
    ("database sharding", "horizontal scaling", "hash-based vs range-based sharding"),
    ("a CI/CD pipeline", "deployment automation", "GitHub Actions vs Jenkins"),
    ("error handling in distributed systems", "retry logic", "exponential backoff vs circuit breaker"),
    ("a search engine", "inverted index", "Elasticsearch vs building from scratch"),
    ("authentication", "JWT", "session-based vs token-based auth"),
    ("a job queue", "async processing", "Redis Queue vs RabbitMQ vs Kafka"),
    ("data validation", "input sanitization", "schema validation vs runtime checks"),
    ("a plugin system", "extensibility", "dependency injection vs event hooks"),
    ("memory management in Python", "garbage collection", "reference counting vs generational GC"),
    ("a recommendation engine", "collaborative filtering", "content-based vs collaborative filtering"),
    ("logging and monitoring", "observability", "structured logging vs unstructured"),
    ("API versioning", "backward compatibility", "URL versioning vs header versioning"),
    ("a file upload system", "large file handling", "chunked upload vs multipart"),
]


# =============================================================================
# SELF-CORRECTION DATA TEMPLATES
# =============================================================================

SELF_CORRECTION_SEEDS = [
    {
        "instruction": "Write a Python function to find the longest palindrome substring in a string.",
        "initial_response": """```python
def longest_palindrome(s):
    if len(s) < 2:
        return s
    
    longest = ""
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            substring = s[i:j]
            if substring == substring[::-1] and len(substring) > len(longest):
                longest = substring
    return longest
```""",
        "output": """<review>
Error identified: Off-by-one error in the inner loop. `range(i+1, len(s))` should be `range(i+1, len(s)+1)` because string slicing `s[i:j]` excludes index `j`. As written, the function never checks substrings that include the last character.
Type: code
Severity: high

Additional issue: O(n^3) time complexity — O(n^2) substrings each taking O(n) to reverse-check. This can be solved in O(n^2) using the expand-around-center technique.
</review>
<fix>
```python
def longest_palindrome(s: str) -> str:
    if len(s) < 2:
        return s
    
    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = s[0]
    for i in range(len(s)):
        # Odd-length palindromes (centered on s[i])
        odd = expand_around_center(i, i)
        # Even-length palindromes (centered between s[i] and s[i+1])
        even = expand_around_center(i, i + 1)
        
        candidate = odd if len(odd) >= len(even) else even
        if len(candidate) > len(longest):
            longest = candidate
    
    return longest
```

Changes made:
1. Fixed off-by-one: replaced O(n^3) brute force with O(n^2) expand-around-center
2. Handles both odd and even length palindromes
3. Added type hints
4. Correctly initializes `longest` to first character (handles single-char edge case)
</fix>
<validate>
Verification: pass
Test cases:
- "babad" → "bab" or "aba" ✓
- "cbbd" → "bb" ✓
- "a" → "a" ✓
- "" → "" ✓
- "racecar" → "racecar" ✓
Confidence: 0.95
Remaining issues: none
</validate>"""
    },
    {
        "instruction": "What causes a deadlock in multithreaded programming?",
        "initial_response": "A deadlock happens when two threads are waiting for each other to release a lock. Thread A holds Lock 1 and wants Lock 2, while Thread B holds Lock 2 and wants Lock 1. Neither can proceed.",
        "output": """<review>
Error identified: The explanation only covers one specific deadlock scenario (circular wait with 2 threads and 2 locks). It omits the four necessary conditions for deadlock (Coffman conditions) and doesn't mention that deadlocks can involve more than 2 threads. Also missing: prevention strategies.
Type: reasoning
Severity: medium

The given example is correct but incomplete — a partial explanation can be worse than no explanation because it creates false confidence.
</review>
<fix>
A deadlock occurs when threads are permanently blocked, each waiting for a resource held by another. Four conditions must ALL be present simultaneously (Coffman conditions):

1. **Mutual exclusion**: Resources can't be shared (only one thread holds a lock at a time)
2. **Hold and wait**: A thread holds one resource while waiting for another
3. **No preemption**: Resources can't be forcibly taken from a thread
4. **Circular wait**: A cycle exists: Thread A waits for B's resource, B waits for C's, C waits for A's

**Example with 2 threads:**
```
Thread A: acquires Lock1 → tries to acquire Lock2 (blocked, B has it)
Thread B: acquires Lock2 → tries to acquire Lock1 (blocked, A has it)
```

**Prevention strategies** (break any one condition):
- **Lock ordering**: Always acquire locks in the same global order (breaks circular wait)
- **Timeout**: Use `tryLock(timeout)` instead of blocking forever (breaks hold-and-wait)
- **Lock-free data structures**: Use atomics/CAS instead of locks (breaks mutual exclusion)

Changes made: Added all 4 Coffman conditions, clarified that deadlocks can involve N threads, and added prevention strategies.
</fix>
<validate>
Verification: pass
Confidence: 0.95
Remaining issues: none
</validate>"""
    },
]


# =============================================================================
# TOOL USE DATA TEMPLATES
# =============================================================================

TOOL_USE_SEEDS = [
    {
        "instruction": "Read the file 'config.json' and update the 'version' field to '2.0.0'.",
        "output": """<think>
I need to read the file first, then modify the version field and write it back.
</think>

Let me read the current config file first.

<tool_call>
{"name": "read_file", "args": {"path": "config.json"}}
</tool_call>
<tool_result>
{"status": "success", "content": "{\\n  \\"name\\": \\"my-app\\",\\n  \\"version\\": \\"1.5.3\\",\\n  \\"description\\": \\"A sample application\\"\\n}"}
</tool_result>

The current version is 1.5.3. I'll update it to 2.0.0.

<tool_call>
{"name": "edit_file", "args": {"path": "config.json", "old_text": "\"version\": \"1.5.3\"", "new_text": "\"version\": \"2.0.0\""}}
</tool_call>
<tool_result>
{"status": "success", "changes_made": 1}
</tool_result>

Done. Updated `config.json` version from `1.5.3` to `2.0.0`."""
    },
    {
        "instruction": "Find all Python files that import 'requests' and list them.",
        "output": """<think>
I need to search for Python files containing 'import requests' or 'from requests import'.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "(?:import requests|from requests import)", "path": ".", "include": "*.py"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/api_client.py", "line": 3, "match": "import requests"}, {"file": "src/downloader.py", "line": 1, "match": "import requests"}, {"file": "tests/test_api.py", "line": 5, "match": "from requests import Session"}]}
</tool_result>

Found 3 Python files that import `requests`:

1. `src/api_client.py` (line 3): `import requests`
2. `src/downloader.py` (line 1): `import requests`
3. `tests/test_api.py` (line 5): `from requests import Session`"""
    },
    {
        "instruction": "Write a Python script that fetches the current Bitcoin price from CoinGecko API and saves it to a file.",
        "output": """<think>
I need to write a Python script that:
1. Makes an HTTP request to CoinGecko's API
2. Parses the JSON response
3. Extracts the Bitcoin price
4. Saves it to a file with a timestamp
</think>

<tool_call>
{"name": "write_file", "args": {"path": "fetch_btc_price.py", "content": "import json\\nimport urllib.request\\nfrom datetime import datetime\\n\\ndef fetch_btc_price():\\n    url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'\\n    \\n    with urllib.request.urlopen(url) as response:\\n        data = json.loads(response.read().decode())\\n    \\n    price = data['bitcoin']['usd']\\n    timestamp = datetime.now().isoformat()\\n    \\n    result = {'price_usd': price, 'timestamp': timestamp}\\n    \\n    with open('btc_price.json', 'w') as f:\\n        json.dump(result, f, indent=2)\\n    \\n    return result\\n\\nif __name__ == '__main__':\\n    result = fetch_btc_price()\\n    print(f'Bitcoin: ${result[\\\"price_usd\\\"]:,.2f} at {result[\\\"timestamp\\\"]}')"}}
</tool_call>
<tool_result>
{"status": "success", "bytes_written": 542}
</tool_result>

Now let me run it to verify it works:

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "exec(open('fetch_btc_price.py').read())"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Bitcoin: $67,432.00 at 2025-01-15T14:23:45.123456"}
</tool_result>

Script created and tested. The file `btc_price.json` now contains the current Bitcoin price with a timestamp."""
    },
]


# =============================================================================
# GENERATION LOGIC
# =============================================================================

def generate_team_thinking_data(count: int) -> list[dict]:
    """Generate team thinking training examples."""
    examples = list(TEAM_THINKING_SEEDS)  # Start with seeds
    
    # Generate additional examples from templates
    for i in range(count - len(examples)):
        topic, context, tradeoff = random.choice(TEAM_THINKING_TOPICS)
        template = random.choice(TEAM_THINKING_QUERY_TEMPLATES)
        
        # Create instruction from template
        instruction = template.format(
            topic=topic,
            option_a=tradeoff.split(" vs ")[0] if " vs " in tradeoff else topic,
            option_b=tradeoff.split(" vs ")[1] if " vs " in tradeoff else context,
            problem_description=f"issues with {topic}",
            requirement=topic,
            concept=topic,
            prerequisite=context,
            approach_description=f"using {tradeoff} for {topic}",
            task=f"implement {topic}",
            constraints=f"using {context}",
            project_type=context,
            goal=topic,
        )
        
        # Placeholder output - in real pipeline, this would be generated by a strong LLM
        examples.append({
            "instruction": instruction,
            "output": _generate_team_thinking_placeholder(topic, context, tradeoff),
        })
    
    return examples[:count]


def _generate_team_thinking_placeholder(topic: str, context: str, tradeoff: str) -> str:
    """Generate a placeholder team thinking response.
    
    NOTE: In production, replace this with actual LLM-generated responses.
    These placeholders establish the FORMAT the model should learn.
    """
    return f"""<think>
[Architect] When designing {topic}, the key architectural considerations are scalability, maintainability, and performance. The system should be built with {context} in mind, using established patterns that support future growth.

[Critic] Before committing to this approach, we need to consider: What are the failure modes? How does this perform under edge cases? The tradeoff between {tradeoff} deserves careful analysis — the "obvious" choice isn't always correct.

[Researcher] Industry best practices for {topic} suggest using proven solutions where possible. Key papers and implementations show that {context} is well-understood, with known tradeoffs between {tradeoff}.

[Implementer] Concrete implementation: Start with the simpler option ({tradeoff.split(' vs ')[0] if ' vs ' in tradeoff else topic}), measure performance, and only add complexity if measurements justify it. Use existing libraries where possible.

[Synthesizer] The team agrees: start simple with {tradeoff.split(' vs ')[0] if ' vs ' in tradeoff else topic}, measure before optimizing, and keep the architecture modular enough to switch approaches if needed.
</think>

Based on multi-perspective analysis of {topic}:

**Recommendation**: Start with {tradeoff.split(' vs ')[0] if ' vs ' in tradeoff else 'the simpler approach'} and measure before adding complexity.

**Key considerations**:
1. Prioritize simplicity and measurability in the initial implementation
2. Design for modularity so components can be swapped later
3. Monitor the tradeoff between {tradeoff} with real production data
4. Plan for the failure modes identified during review"""


def generate_self_correction_data(count: int) -> list[dict]:
    """Generate self-correction training examples."""
    examples = list(SELF_CORRECTION_SEEDS)
    
    # Bug patterns for synthetic generation
    bug_patterns = [
        ("off-by-one error", "Loop boundary is wrong, missing last element", "code", "high"),
        ("null/None check missing", "Function crashes on None input", "code", "high"),
        ("incorrect type handling", "String/int confusion in comparison", "code", "medium"),
        ("missing edge case", "Empty input not handled", "code", "medium"),
        ("logical fallacy", "Correlation treated as causation", "reasoning", "high"),
        ("incomplete explanation", "Key concept omitted from explanation", "reasoning", "medium"),
        ("outdated information", "Using deprecated API/approach", "factual", "medium"),
        ("wrong complexity analysis", "Claiming O(n) when it's O(n^2)", "reasoning", "high"),
    ]
    
    for i in range(count - len(examples)):
        bug_name, bug_desc, bug_type, severity = random.choice(bug_patterns)
        
        examples.append({
            "instruction": f"Review and fix the following code/response that contains a {bug_name}.",
            "initial_response": f"[Placeholder initial response with {bug_name}: {bug_desc}]",
            "output": f"""<review>
Error identified: {bug_desc}
Type: {bug_type}
Severity: {severity}
</review>
<fix>
Corrected response: [Fixed version addressing the {bug_name}]
Changes made: Fixed the {bug_name} by [specific correction approach]
</fix>
<validate>
Verification: pass
Confidence: 0.9
Remaining issues: none
</validate>"""
        })
    
    return examples[:count]


def generate_tool_use_data(count: int) -> list[dict]:
    """Generate tool use training examples."""
    examples = list(TOOL_USE_SEEDS)
    
    tool_tasks = [
        "List all files larger than 1MB in the current directory",
        "Create a new directory called 'backup' and copy important files to it",
        "Find and replace all occurrences of 'old_api' with 'new_api' in the src/ directory",
        "Run the test suite and report any failures",
        "Check the disk usage of the current project",
        "Install a Python package and verify it imported correctly",
        "Read a CSV file and compute the average of a numeric column",
        "Create a git branch, make changes, and commit them",
        "Download a file from a URL and verify its checksum",
        "Parse a log file and extract error messages from the last hour",
    ]
    
    for i in range(count - len(examples)):
        task = tool_tasks[i % len(tool_tasks)]
        examples.append({
            "instruction": task,
            "output": f"""<think>
I need to break this task into steps and use the appropriate tools.
</think>

<tool_call>
{{"name": "shell_command", "args": {{"command": "echo 'Executing: {task}'"}}}}
</tool_call>
<tool_result>
{{"status": "success", "output": "Executing: {task}"}}
</tool_result>

Task completed: {task}"""
        })
    
    return examples[:count]


def save_dataset(examples: list[dict], output_dir: Path, train_ratio: float = 0.9):
    """Save examples as train/eval JSONL splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    
    for path, data in [(train_path, train_examples), (eval_path, eval_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(data)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 Training Data Generator")
    parser.add_argument(
        "--capability",
        type=str,
        required=True,
        choices=["team_thinking", "self_correction", "tool_use", "all"],
        help="Which capability to generate data for",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of examples to generate per capability",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    generators = {
        "team_thinking": (generate_team_thinking_data, DATA_DIR / "team_thinking"),
        "self_correction": (generate_self_correction_data, DATA_DIR / "self_correction"),
        "tool_use": (generate_tool_use_data, DATA_DIR / "tool_use"),
    }
    
    capabilities = list(generators.keys()) if args.capability == "all" else [args.capability]
    
    for cap in capabilities:
        logger.info(f"Generating {args.count} examples for: {cap}")
        gen_fn, output_dir = generators[cap]
        examples = gen_fn(args.count)
        save_dataset(examples, output_dir)
        logger.info(f"Done: {cap}")
    
    logger.info("Data generation complete!")
    logger.info("")
    logger.info("IMPORTANT: The generated data contains placeholder/template outputs.")
    logger.info("For high-quality training, you should:")
    logger.info("  1. Review and edit the seed examples for accuracy")
    logger.info("  2. Use a strong LLM to generate proper outputs for template examples")
    logger.info("  3. Manually curate at least 50-100 high-quality examples per capability")


if __name__ == "__main__":
    main()
