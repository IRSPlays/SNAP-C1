import json
import re
import time
import argparse
import random
from pathlib import Path
from loguru import logger

try:
    import requests as _requests
except ImportError:  # allow import without network deps
    _requests = None

# ---------------------------------------------------------------------------
# Bug templates – each entry carries reasoning for CoT wrapping
# ---------------------------------------------------------------------------
TEMPLATES = [
    # ── original 7 ──────────────────────────────────────────────────────────
    {
        "desc": "Fix the off-by-one error",
        "before": "def get_last_element(items):\n    return items[len(items)]",
        "after":  "def get_last_element(items):\n    return items[len(items)-1]",
        "reasoning": (
            "The bug is on the return line: `items[len(items)]` tries to access an\n"
            "index equal to the length of the list.  Python uses 0-based indexing, so\n"
            "valid indices run from 0 to len(items)-1.  Accessing index len(items)\n"
            "always raises IndexError.\n"
            "The fix: subtract 1 → `items[len(items)-1]`, or use `items[-1]`."
        ),
    },
    {
        "desc": "Fix the variable name typo",
        "before": "def calculate_total(price, tax):\n    return price + txa",
        "after":  "def calculate_total(price, tax):\n    return price + tax",
        "reasoning": (
            "`txa` is not defined anywhere in scope; the developer mistyped the\n"
            "parameter name `tax`.  Python raises NameError at runtime.\n"
            "The fix: correct the typo → `tax`."
        ),
    },
    {
        "desc": "Fix the undefined variable error",
        "before": "def greet(name):\n    print('Hello ' + nmae)",
        "after":  "def greet(name):\n    print('Hello ' + name)",
        "reasoning": (
            "`nmae` is an undefined name – the letters of `name` were transposed.\n"
            "This causes a NameError at runtime.\n"
            "The fix: correct the typo → `name`."
        ),
    },
    {
        "desc": "Fix the incorrect comparison operator",
        "before": "def is_positive(num):\n    return num < 0",
        "after":  "def is_positive(num):\n    return num > 0",
        "reasoning": (
            "A positive number satisfies `num > 0`, not `num < 0`.  The operator is\n"
            "inverted, so the function returns True for negatives and False for\n"
            "positives – the exact opposite of its name.\n"
            "The fix: replace `<` with `>`."
        ),
    },
    {
        "desc": "Fix the missing return statement",
        "before": "def add(a, b):\n    result = a + b",
        "after":  "def add(a, b):\n    return a + b",
        "reasoning": (
            "The function computes `a + b` but never returns the value.  Without an\n"
            "explicit `return`, Python implicitly returns None, so every caller\n"
            "receives None instead of the sum.\n"
            "The fix: add `return` before the expression."
        ),
    },
    {
        "desc": "Fix the incorrect list initialization",
        "before": "def create_matrix(n):\n    return [[0] * n] * n",
        "after":  "def create_matrix(n):\n    return [[0] * n for _ in range(n)]",
        "reasoning": (
            "`[[0]*n]*n` creates n references to the *same* inner list object.\n"
            "Modifying one row modifies all rows because they all point to the same\n"
            "list in memory.\n"
            "The fix: use a list comprehension so each row is an independent object."
        ),
    },
    {
        "desc": "Fix the mutable default argument",
        "before": (
            "def append_to(element, to=[]):\n"
            "    to.append(element)\n"
            "    return to"
        ),
        "after": (
            "def append_to(element, to=None):\n"
            "    if to is None:\n"
            "        to = []\n"
            "    to.append(element)\n"
            "    return to"
        ),
        "reasoning": (
            "Default argument values are evaluated once at function definition time,\n"
            "not on every call.  Using a mutable object (list) as a default means all\n"
            "calls that omit `to` share the same list, causing state to accumulate\n"
            "across calls.\n"
            "The fix: use None as the default and create a fresh list inside the body."
        ),
    },
    # ── 15+ additional templates ─────────────────────────────────────────────
    {
        "desc": "Fix wrong indentation causing logic to always execute",
        "before": (
            "def process(items):\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item > 0:\n"
            "            result.append(item)\n"
            "        return result"
        ),
        "after": (
            "def process(items):\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item > 0:\n"
            "            result.append(item)\n"
            "    return result"
        ),
        "reasoning": (
            "`return result` is indented inside the `for` loop, so the function\n"
            "returns after processing only the first item regardless of list length.\n"
            "The fix: dedent `return result` to the function body level so it runs\n"
            "after the loop completes."
        ),
    },
    {
        "desc": "Fix bare except that swallows all exceptions",
        "before": (
            "def read_config(path):\n"
            "    try:\n"
            "        with open(path) as f:\n"
            "            return json.load(f)\n"
            "    except:\n"
            "        return {}"
        ),
        "after": (
            "def read_config(path):\n"
            "    try:\n"
            "        with open(path) as f:\n"
            "            return json.load(f)\n"
            "    except (FileNotFoundError, json.JSONDecodeError) as exc:\n"
            "        logger.warning(f'Could not read config: {exc}')\n"
            "        return {}"
        ),
        "reasoning": (
            "A bare `except:` clause catches *everything*, including SystemExit,\n"
            "KeyboardInterrupt, and programming errors like AttributeError.\n"
            "This makes debugging nearly impossible and can hide real bugs.\n"
            "The fix: catch only the expected exception types explicitly."
        ),
    },
    {
        "desc": "Fix missing await in async function",
        "before": (
            "import asyncio\n\n"
            "async def fetch_data(url):\n"
            "    async with aiohttp.ClientSession() as session:\n"
            "        response = session.get(url)\n"
            "        return response.text()"
        ),
        "after": (
            "import asyncio\n\n"
            "async def fetch_data(url):\n"
            "    async with aiohttp.ClientSession() as session:\n"
            "        response = await session.get(url)\n"
            "        return await response.text()"
        ),
        "reasoning": (
            "`session.get(url)` returns a coroutine, not the response object.\n"
            "Without `await`, `response` holds an un-awaited coroutine, and calling\n"
            "`.text()` on it raises AttributeError.\n"
            "The fix: `await` both the `.get()` call and the `.text()` call."
        ),
    },
    {
        "desc": "Fix wrong regex pattern anchoring",
        "before": (
            "import re\n\n"
            "def is_valid_email(email):\n"
            "    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'\n"
            "    return bool(re.match(pattern, email))"
        ),
        "after": (
            "import re\n\n"
            "def is_valid_email(email):\n"
            "    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n"
            "    return bool(re.fullmatch(pattern, email))"
        ),
        "reasoning": (
            "`re.match` only anchors at the start, not the end.  A string like\n"
            "'user@example.com!!!invalid' would still pass the check because the\n"
            "regex matches the valid prefix.\n"
            "The fix: use `re.fullmatch` (or add `^` and `$` anchors) to require the\n"
            "entire string to conform to the pattern."
        ),
    },
    {
        "desc": "Fix integer division instead of float division",
        "before": (
            "def average(values):\n"
            "    total = sum(values)\n"
            "    return total / len(values) * 100 // 1"
        ),
        "after": (
            "def average(values):\n"
            "    if not values:\n"
            "        return 0.0\n"
            "    return sum(values) / len(values)"
        ),
        "reasoning": (
            "The original expression multiplies by 100 and floor-divides, which\n"
            "truncates decimal precision and returns a float that looks like an\n"
            "integer percentage.  The intent is a plain arithmetic mean.\n"
            "Additionally, there is no guard against an empty list (ZeroDivisionError).\n"
            "The fix: return `sum(values) / len(values)` with an empty-list guard."
        ),
    },
    {
        "desc": "Fix missing None check before attribute access",
        "before": (
            "def get_user_name(user):\n"
            "    return user.profile.name.strip()"
        ),
        "after": (
            "def get_user_name(user):\n"
            "    if user is None or user.profile is None:\n"
            "        return ''\n"
            "    name = user.profile.name\n"
            "    return name.strip() if name else ''"
        ),
        "reasoning": (
            "Any of `user`, `user.profile`, or `user.profile.name` could be None,\n"
            "which would raise AttributeError on the chained access.\n"
            "The fix: add explicit None guards before each attribute dereference and\n"
            "return a safe default when data is absent."
        ),
    },
    {
        "desc": "Fix incorrect f-string (missing f prefix)",
        "before": (
            "def greet_user(username, score):\n"
            "    msg = 'Hello {username}, your score is {score}'\n"
            "    return msg"
        ),
        "after": (
            "def greet_user(username, score):\n"
            "    msg = f'Hello {username}, your score is {score}'\n"
            "    return msg"
        ),
        "reasoning": (
            "Without the `f` prefix the string is a plain literal; curly-brace\n"
            "expressions are NOT substituted and the variables `username` and `score`\n"
            "are never interpolated.\n"
            "The fix: add the `f` prefix to make it an f-string."
        ),
    },
    {
        "desc": "Fix dictionary mutation during iteration",
        "before": (
            "def remove_zeros(d):\n"
            "    for key, value in d.items():\n"
            "        if value == 0:\n"
            "            del d[key]\n"
            "    return d"
        ),
        "after": (
            "def remove_zeros(d):\n"
            "    return {key: value for key, value in d.items() if value != 0}"
        ),
        "reasoning": (
            "Deleting keys from a dictionary while iterating over it raises\n"
            "RuntimeError: dictionary changed size during iteration.\n"
            "The fix: build a new dict with a comprehension that skips zero-value\n"
            "entries, leaving the original untouched during iteration."
        ),
    },
    {
        "desc": "Fix using `is` instead of `==` for value comparison",
        "before": (
            "def check_status(code):\n"
            "    if code is 200:\n"
            "        return 'OK'\n"
            "    return 'Error'"
        ),
        "after": (
            "def check_status(code):\n"
            "    if code == 200:\n"
            "        return 'OK'\n"
            "    return 'Error'"
        ),
        "reasoning": (
            "`is` tests object identity, not value equality.  CPython only interns\n"
            "small integers (-5 to 256), so `code is 200` may work in tests but\n"
            "silently fails with large integers or in other Python implementations.\n"
            "The fix: use `==` for value comparison."
        ),
    },
    {
        "desc": "Fix missing super().__init__() call",
        "before": (
            "class Animal:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n\n"
            "class Dog(Animal):\n"
            "    def __init__(self, name, breed):\n"
            "        self.breed = breed"
        ),
        "after": (
            "class Animal:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n\n"
            "class Dog(Animal):\n"
            "    def __init__(self, name, breed):\n"
            "        super().__init__(name)\n"
            "        self.breed = breed"
        ),
        "reasoning": (
            "`Dog.__init__` shadows `Animal.__init__` without calling it, so\n"
            "`self.name` is never set.  Any access to `dog.name` raises AttributeError.\n"
            "The fix: call `super().__init__(name)` at the start of `Dog.__init__`."
        ),
    },
    {
        "desc": "Fix accidental string concatenation in loop",
        "before": (
            "def join_words(words):\n"
            "    result = ''\n"
            "    for word in words:\n"
            "        result = result + word + ' '\n"
            "    return result.strip()"
        ),
        "after": (
            "def join_words(words):\n"
            "    return ' '.join(words)"
        ),
        "reasoning": (
            "String concatenation in a loop is O(n²) because each `+` creates a new\n"
            "string object.  For large lists this is severely inefficient.\n"
            "The fix: use `str.join()` which is O(n) and idiomatic Python."
        ),
    },
    {
        "desc": "Fix shadowing built-in name",
        "before": (
            "def count_items(list):\n"
            "    return len(list)"
        ),
        "after": (
            "def count_items(items):\n"
            "    return len(items)"
        ),
        "reasoning": (
            "Using `list` as a parameter name shadows the built-in `list` type.\n"
            "Any code inside (or downstream of) this function that tries to call\n"
            "`list(...)` will find the local variable instead of the built-in and\n"
            "raise TypeError.\n"
            "The fix: rename the parameter to a non-reserved name like `items`."
        ),
    },
    {
        "desc": "Fix incorrect use of `extend` vs `append` for nested lists",
        "before": (
            "def collect_sublists(data):\n"
            "    result = []\n"
            "    for sublist in data:\n"
            "        result.append(sublist)\n"
            "    return result"
        ),
        "after": (
            "def collect_sublists(data):\n"
            "    result = []\n"
            "    for sublist in data:\n"
            "        result.extend(sublist)\n"
            "    return result"
        ),
        "reasoning": (
            "The function is supposed to flatten the sublists into a single list.\n"
            "`append` adds each sublist as a single nested element, producing a list\n"
            "of lists – the same as the input.\n"
            "The fix: use `extend` to unpack each sublist into `result`."
        ),
    },
    {
        "desc": "Fix async generator not being awaited",
        "before": (
            "async def fetch_all(urls):\n"
            "    results = []\n"
            "    for url in urls:\n"
            "        results.append(fetch_data(url))\n"
            "    return results"
        ),
        "after": (
            "async def fetch_all(urls):\n"
            "    results = []\n"
            "    for url in urls:\n"
            "        results.append(await fetch_data(url))\n"
            "    return results"
        ),
        "reasoning": (
            "`fetch_data(url)` is a coroutine; calling it without `await` appends the\n"
            "coroutine object itself to `results`, not the fetched data.\n"
            "The fix: `await fetch_data(url)` to execute the coroutine and get the\n"
            "actual response value."
        ),
    },
    {
        "desc": "Fix wrong exception type caught",
        "before": (
            "def parse_int(s):\n"
            "    try:\n"
            "        return int(s)\n"
            "    except TypeError:\n"
            "        return None"
        ),
        "after": (
            "def parse_int(s):\n"
            "    try:\n"
            "        return int(s)\n"
            "    except (ValueError, TypeError):\n"
            "        return None"
        ),
        "reasoning": (
            "`int('abc')` raises ValueError, not TypeError.  TypeError is raised when\n"
            "an argument of the wrong *type* is passed (e.g. a list).  Catching only\n"
            "TypeError means invalid string inputs propagate as unhandled exceptions.\n"
            "The fix: catch both ValueError and TypeError."
        ),
    },
    {
        "desc": "Fix use of `print` instead of returning value",
        "before": (
            "def square(n):\n"
            "    print(n * n)"
        ),
        "after": (
            "def square(n):\n"
            "    return n * n"
        ),
        "reasoning": (
            "The function computes `n * n` but only prints it; no value is returned.\n"
            "Callers receive None, making the function useless in any expression.\n"
            "The fix: replace `print` with `return`."
        ),
    },
    {
        "desc": "Fix global variable modified without global declaration",
        "before": (
            "counter = 0\n\n"
            "def increment():\n"
            "    counter += 1\n"
            "    return counter"
        ),
        "after": (
            "counter = 0\n\n"
            "def increment():\n"
            "    global counter\n"
            "    counter += 1\n"
            "    return counter"
        ),
        "reasoning": (
            "Inside `increment`, Python treats `counter` as a local variable because\n"
            "it appears on the left side of assignment (`counter += 1` expands to\n"
            "`counter = counter + 1`).  Reading an unassigned local raises\n"
            "UnboundLocalError.\n"
            "The fix: declare `global counter` so Python knows to use the module-level\n"
            "variable."
        ),
    },
    {
        "desc": "Fix missing closing resource (context manager)",
        "before": (
            "def write_file(path, content):\n"
            "    f = open(path, 'w')\n"
            "    f.write(content)\n"
            "    return True"
        ),
        "after": (
            "def write_file(path, content):\n"
            "    with open(path, 'w') as f:\n"
            "        f.write(content)\n"
            "    return True"
        ),
        "reasoning": (
            "Opening a file without a `with` statement means `f.close()` is never\n"
            "called if `f.write()` raises an exception, leaking the file descriptor.\n"
            "The fix: use a `with` statement so the file is always closed on exit,\n"
            "whether normal or due to an exception."
        ),
    },
    {
        "desc": "Fix deep copy needed instead of shallow copy",
        "before": (
            "import copy\n\n"
            "def clone_config(config):\n"
            "    return config.copy()"
        ),
        "after": (
            "import copy\n\n"
            "def clone_config(config):\n"
            "    return copy.deepcopy(config)"
        ),
        "reasoning": (
            "`dict.copy()` performs a shallow copy: top-level keys are duplicated but\n"
            "nested dicts/lists still share the same references as the original.\n"
            "Mutating a nested value in the clone mutates the original too.\n"
            "The fix: use `copy.deepcopy()` to recursively copy all nested objects."
        ),
    },
]


# ---------------------------------------------------------------------------
# CoT wrapper
# ---------------------------------------------------------------------------

def _wrap_cot(reasoning: str, code: str) -> str:
    """Wrap code with a <think> reasoning block."""
    return f"<think>\n{reasoning.strip()}\n</think>\n{code}"


# ---------------------------------------------------------------------------
# 1. Synthetic pair generation
# ---------------------------------------------------------------------------

def generate_synthetic_pairs(num_samples: int = 1000) -> list:
    """Generate synthetic (prompt, target_code) pairs with CoT targets."""
    dataset = []
    for _ in range(num_samples):
        tpl = random.choice(TEMPLATES)
        # light variadic substitution for the variable-name family.
        # Use a word-boundary-aware regex so that `.items()` dict-method calls
        # (e.g. `d.items()`) and compound names (e.g. `count_items`) are never
        # corrupted – only standalone `items` identifiers (parameters/variables)
        # are swapped for the chosen var_name.
        var_name = random.choice(["items", "arr", "lst", "data_list"])
        _items_re = re.compile(r'(?<![.\w])items(?!\w)')
        before    = _items_re.sub(var_name, tpl["before"])
        after     = _items_re.sub(var_name, tpl["after"])
        reasoning = _items_re.sub(var_name, tpl["reasoning"])

        prompt = (
            f"{tpl['desc']} in this Python function.\n\n"
            f"```python\n{before}\n```"
        )
        target = _wrap_cot(reasoning, after)
        dataset.append({"prompt": prompt, "target_code": target})
    return dataset


# ---------------------------------------------------------------------------
# 2. GitHub PR miner
# ---------------------------------------------------------------------------

def mine_github_patches(
    repos: list,
    token: str,
    per_repo: int = 200,
    output_path: str = None,
) -> list:
    """
    Fetch merged Python-only PRs from GitHub and build CoT training samples.

    Parameters
    ----------
    repos       : list of "owner/repo" strings
    token       : GitHub personal access token (needs public_repo scope)
    per_repo    : max number of qualifying PRs to collect per repo
    output_path : optional path; if given, appends results to existing JSON

    Returns
    -------
    list of dicts with keys: prompt, target_code
    """
    if _requests is None:
        raise ImportError("pip install requests to use mine_github_patches()")

    API = "https://api.github.com"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    def _check_rate(resp):
        remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
        if remaining < 10:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_s  = max(reset_ts - int(time.time()) + 2, 5)
            logger.warning(
                f"GitHub rate limit low ({remaining} left). "
                f"Sleeping {sleep_s}s until reset."
            )
            time.sleep(sleep_s)

    collected = []

    for repo in repos:
        logger.info(f"Mining repo: {repo}")
        page = 1
        repo_count = 0

        while repo_count < per_repo:
            url = f"{API}/repos/{repo}/pulls"
            params = {
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            }
            resp = _requests.get(url, headers=headers, params=params, timeout=30)
            _check_rate(resp)

            if resp.status_code == 404:
                logger.warning(f"Repo not found or no access: {repo}")
                break
            if resp.status_code != 200:
                logger.error(f"Error fetching PRs for {repo}: {resp.status_code}")
                break

            prs = resp.json()
            if not prs:
                break  # no more pages

            for pr in prs:
                if repo_count >= per_repo:
                    break

                # only merged PRs
                if not pr.get("merged_at"):
                    continue

                pr_number = pr["number"]
                pr_title  = pr.get("title", "")
                pr_body   = (pr.get("body") or "")[:300]

                # fetch files changed
                files_url = f"{API}/repos/{repo}/pulls/{pr_number}/files"
                fr = _requests.get(files_url, headers=headers, timeout=30)
                _check_rate(fr)
                if fr.status_code != 200:
                    continue

                files = fr.json()

                # filter: all files must be .py
                if not files or not all(
                    f.get("filename", "").endswith(".py") for f in files
                ):
                    continue

                # filter: total diff lines 2..60
                total_changes = sum(
                    f.get("additions", 0) + f.get("deletions", 0) for f in files
                )
                if not (2 <= total_changes <= 60):
                    continue

                # build unified diff text from patch fields
                diff_parts = []
                for f in files:
                    patch = f.get("patch", "")
                    if patch:
                        diff_parts.append(
                            f"--- a/{f['filename']}\n"
                            f"+++ b/{f['filename']}\n"
                            f"{patch}"
                        )
                diff_text = "\n".join(diff_parts) if diff_parts else "(binary or empty)"

                reasoning = (
                    f"PR #{pr_number} in {repo}: {pr_title}\n"
                    + (f"\n{pr_body}" if pr_body.strip() else "")
                )
                target = _wrap_cot(reasoning, diff_text)

                prompt = (
                    f"Apply the patch described by this GitHub pull request.\n\n"
                    f"Repository: {repo}\n"
                    f"PR #{pr_number}: {pr_title}\n\n"
                    f"{pr_body}"
                )
                collected.append({"prompt": prompt, "target_code": target})
                repo_count += 1

            if len(prs) < 100:
                break  # fewer than a full page → no more
            page += 1

        logger.info(f"  collected {repo_count} qualifying PRs from {repo}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if out.exists():
            with open(out, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        merged = existing + collected
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        logger.info(
            f"GitHub samples appended to {out} "
            f"(+{len(collected)}, total {len(merged)})"
        )

    return collected


# ---------------------------------------------------------------------------
# 3. Negative example augmentation
# ---------------------------------------------------------------------------

# Each mutation is (pattern, replacement, description)
_MUTATIONS = [
    # flip comparisons
    (r"(?<![=!<>])>(?![>=])",  "<",    "flip > → <"),
    (r"(?<![=!<>])<(?![<=])",  ">",    "flip < → >"),
    (r"(?<![=!])==(?!=)",       "!=",   "flip == → !="),
    (r"!=",                      "==",   "flip != → =="),
    # negate return booleans
    (r"\breturn True\b",         "return False", "negate return True→False"),
    (r"\breturn False\b",        "return True",  "negate return False→True"),
    # swap arithmetic operators (conservative – only isolated + / -)
    (r"(?<=\s)\+(?=\s)",         "-",    "swap + → -"),
    (r"(?<=\s)-(?=\s)",          "+",    "swap - → +"),
    # off-by-one on integer literals (only small integers 1-9 to stay sensible)
    (r"(?<!\w)([1-9])(?!\w)",    lambda m: str(int(m.group(1)) + 1), "off-by-one +1"),
]


def generate_negative_examples(dataset: list) -> list:
    """
    For each positive sample, attempt one random mutation to create a negative
    (broken) version.  Returns a list of negative dicts with is_negative=True.

    Only samples where a mutation actually changes the code are included.
    """
    negatives = []
    for sample in dataset:
        target = sample.get("target_code", "")
        if not target:
            continue

        # strip <think> block to mutate only the code portion
        think_match = re.match(r"(<think>.*?</think>\n?)(.*)", target, re.DOTALL)
        if think_match:
            think_block = think_match.group(1)
            code_part    = think_match.group(2)
        else:
            think_block = ""
            code_part    = target

        # shuffle to try different mutations
        mutations = list(_MUTATIONS)
        random.shuffle(mutations)

        mutated = None
        for pattern, replacement, _desc in mutations:
            # re.sub handles both string and callable replacements natively;
            # no branching needed.
            new_code = re.sub(pattern, replacement, code_part, count=1)
            if new_code != code_part:
                mutated = new_code
                break

        if mutated is None:
            continue  # no mutation succeeded

        neg_target = think_block + mutated
        negatives.append({
            "prompt":      sample["prompt"],
            "target_code": neg_target,
            "is_negative": True,
        })

    return negatives


# ---------------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V4 Supervised Dataset Builder – synthetic + GitHub + negatives"
    )
    parser.add_argument(
        "--output", type=str,
        default="v4_core/data/v4_instruction_dataset.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "github", "both"],
        help="Data source: synthetic only, github only, or both.",
    )
    parser.add_argument(
        "--github_token", type=str, default="",
        help="GitHub personal access token (required when mode=github or both).",
    )
    parser.add_argument(
        "--github_repos", type=str,
        default="django/django,pallets/flask,psf/requests,pytest-dev/pytest,astropy/astropy",
        help="Comma-separated list of owner/repo pairs to mine.",
    )
    parser.add_argument(
        "--github_per_repo", type=int, default=200,
        help="Max qualifying PRs to collect per repo.",
    )
    parser.add_argument(
        "--negatives", action="store_true",
        help="Also generate negative (mutated-wrong) examples for DPO training.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset: list = []

    # ── synthetic ─────────────────────────────────────────────────────────────
    if args.mode in ("synthetic", "both"):
        logger.info(f"Generating {args.samples} synthetic CoT pairs …")
        synth = generate_synthetic_pairs(args.samples)
        dataset.extend(synth)
        logger.info(f"Synthetic: {len(synth)} samples")

    # ── GitHub ────────────────────────────────────────────────────────────────
    if args.mode in ("github", "both"):
        if not args.github_token:
            logger.error("--github_token is required for mode=github or mode=both")
            raise SystemExit(1)
        repos = [r.strip() for r in args.github_repos.split(",") if r.strip()]
        logger.info(f"Mining GitHub PRs from {len(repos)} repos …")
        gh_samples = mine_github_patches(
            repos=repos,
            token=args.github_token,
            per_repo=args.github_per_repo,
        )
        dataset.extend(gh_samples)
        logger.info(f"GitHub: {len(gh_samples)} samples")

    # ── negatives ─────────────────────────────────────────────────────────────
    if args.negatives:
        logger.info("Generating negative (mutated) examples …")
        negs = generate_negative_examples(dataset)
        dataset.extend(negs)
        logger.info(f"Negatives: {len(negs)} samples")

    # ── write ─────────────────────────────────────────────────────────────────
    logger.info(f"Total samples: {len(dataset)}.  Writing to {out_path} …")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info("Done.")

    if dataset:
        logger.info(f"\nExample prompt:\n{dataset[0]['prompt']}")
        logger.info(f"\nExample target (first 400 chars):\n{dataset[0]['target_code'][:400]}")


if __name__ == "__main__":
    main()
