"""Generate high-quality team_thinking training data."""
import json

examples = [
{
"instruction": "Should I use a SQL or NoSQL database for a social media app?",
"output": """<think>
[Architect] The core data model matters here. Social media has users, posts, comments, likes, follows — these are inherently relational (user HAS MANY posts, post HAS MANY comments). A relational model handles this naturally with JOINs. However, the read patterns are feed-centric: "show me posts from people I follow, sorted by time." That's a fan-out-on-read problem that SQL handles poorly at scale.

[Critic] Hold on — "social media app" is too vague. A Twitter clone with 280-char text posts has completely different storage needs than an Instagram clone with media. Also, the assumption that SQL "handles poorly at scale" ignores that Twitter ran on MySQL for years. The real bottleneck is usually the feed generation algorithm, not the database engine. We're premature-optimizing.

[Researcher] Instagram uses PostgreSQL as their primary database and scaled to billions of rows. Twitter migrated from MySQL to Manhattan (their custom NoSQL) only after hitting serious sharding pain at massive scale. Discord uses Cassandra for messages but PostgreSQL for everything else. The pattern is clear: start relational, migrate specific hot paths to NoSQL when you have concrete evidence of bottlenecks.

[Implementer] Start with PostgreSQL. Schema: users table, posts table with user_id FK and created_at index, follows table (follower_id, following_id) with composite index, likes table. For the feed query:
```sql
SELECT p.* FROM posts p
JOIN follows f ON p.user_id = f.following_id
WHERE f.follower_id = $1
ORDER BY p.created_at DESC LIMIT 20;
```
Add Redis for caching hot feeds. This handles 100K+ users easily.

[Synthesizer] Start with PostgreSQL — it handles the relational nature of social data well, and real-world evidence shows it scales further than most apps will ever need. Add Redis for feed caching. Only consider NoSQL migration for specific hot paths (like message storage) when you have measured evidence of bottlenecks, not before.
</think>

**Use PostgreSQL** for your social media app. Here's why:

1. Social data is inherently relational — users, posts, follows, likes all have clear relationships
2. PostgreSQL scales to billions of rows (Instagram proves this)
3. Add Redis for feed caching to handle read-heavy patterns
4. Only migrate specific hot paths to NoSQL when you have concrete performance data justifying it

Don't premature-optimize. Build, measure, then optimize."""
},
{
"instruction": "How do I handle race conditions in a web application's inventory system?",
"output": """<think>
[Architect] The core problem is concurrent writes to a shared resource (inventory count). When two users buy the last item simultaneously, both read count=1, both decrement, and you end up with count=-1 and two orders for one item. We need atomicity at the database level.

[Critic] The naive solution is "just use transactions" but that's insufficient. A standard READ-then-UPDATE inside a transaction still has a TOCTOU race unless you use the right isolation level or explicit locking. Serializable isolation would work but kills throughput. We need a more surgical approach.

[Researcher] There are three proven patterns: (1) Optimistic locking with version columns — read the version, UPDATE WHERE version=X, retry on conflict. Used by Shopify. (2) Pessimistic locking with SELECT FOR UPDATE — locks the row during the transaction. (3) Atomic decrements — UPDATE inventory SET count = count - 1 WHERE id = X AND count > 0. The atomic decrement is simplest and fastest since it's a single statement, no round-trips.

[Implementer] Atomic decrement is the cleanest:
```sql
UPDATE products SET stock = stock - 1
WHERE id = $1 AND stock > 0
RETURNING stock;
```
If zero rows affected, the item is sold out. In Python:
```python
result = db.execute(
    "UPDATE products SET stock = stock - 1 WHERE id = %s AND stock > 0 RETURNING stock",
    [product_id]
)
if result.rowcount == 0:
    raise OutOfStockError(product_id)
```
No explicit locks needed. The database handles atomicity of the single UPDATE.

[Synthesizer] Use atomic decrements — a single UPDATE statement with a WHERE guard. It's the simplest, fastest, and most correct solution. Optimistic locking adds complexity for no benefit here. Pessimistic locking hurts throughput. The single atomic UPDATE lets the database handle concurrency natively.
</think>

**Use atomic decrements** — the simplest and most reliable pattern:

```sql
UPDATE products SET stock = stock - 1 WHERE id = $1 AND stock > 0;
```

If zero rows affected → sold out. No explicit locks, no version columns, no retries. The database guarantees atomicity of a single UPDATE statement. This is what production e-commerce systems use."""
},
{
"instruction": "What's the best way to learn a new programming language quickly?",
"output": """<think>
[Architect] Learning a language has three phases: syntax (days), idioms (weeks), and ecosystem mastery (months). Most people stall at phase 1 by reading tutorials endlessly. The fastest path is to build something real immediately and look up syntax as you go.

[Critic] "Just build something" is common advice but it's incomplete. If you don't understand the language's mental model first, you'll write Java-in-Python or Python-in-Rust. Spending 2-3 hours understanding the paradigm (ownership in Rust, channels in Go, prototypes in JS) before building saves time vs. fighting the language later.

[Researcher] Studies on skill acquisition show that interleaved practice (mixing reading and doing) outperforms blocked practice (read everything then code). The "20 hour rule" from Josh Kaufman suggests focused, deliberate practice on the core 20% of features covers 80% of use cases. For most languages, that's: variables, functions, control flow, data structures, error handling, and the package manager.

[Implementer] Concrete 5-day plan for any language: Day 1: Install, hello world, read a "Tour of X" (Go has a great one). Day 2: Solve 5 easy LeetCode problems in the language. Day 3: Build a CLI tool (todo app or file renamer). Day 4: Build a REST API with the most popular web framework. Day 5: Read someone else's well-written project on GitHub. After that, you're productive enough to learn on the job.

[Synthesizer] Spend 2-3 hours understanding the language's core paradigm first, then immediately start building. Follow the 5-day plan: tour → exercises → CLI tool → web API → read real code. Don't try to "learn everything" before writing code — that's a trap.
</think>

**5-day fast-track for any new language:**

1. **Day 1**: Official tour/intro + hello world
2. **Day 2**: Solve 5 easy coding problems (forces you to learn syntax by doing)
3. **Day 3**: Build a small CLI tool
4. **Day 4**: Build a REST API with the popular framework
5. **Day 5**: Read a well-written open source project

Key insight: spend 2-3 hours understanding the paradigm (ownership, channels, prototypes, etc.) before building. Then learn by doing, not by reading."""
},
{
"instruction": "Is it worth writing unit tests for a prototype?",
"output": """<think>
[Architect] It depends on the prototype's purpose. A throwaway prototype to validate a UI concept? No tests. A prototype that will evolve into production code? Test the core logic, skip the glue code.

[Critic] The question assumes a binary choice but there's a middle ground most people miss. The real risk isn't "no tests" — it's "prototype that accidentally becomes production." Every dev has seen a prototype ship to customers because it "worked." If there's any chance of that, test the parts that would be dangerous to get wrong.

[Researcher] Kent Beck's original formulation: "I get paid for code that works, not for tests." He tests things that might break, skips things that are obviously correct. Google's testing philosophy uses a 70/20/10 split (unit/integration/e2e) for production code, but their internal guidance says prototypes should have zero tests until the prototype is approved for productionization.

[Implementer] Pragmatic approach: write tests only for pure functions with complex logic. Skip tests for API routes, UI, database queries, config. Example — if your prototype has a pricing calculator, test that. If it has a CRUD API, don't.
```python
# YES - test this, complex logic
def calculate_shipping(weight, distance, expedited):
    ...

# NO - don't test this, it's just glue
@app.route("/api/orders", methods=["POST"])
def create_order():
    ...
```

[Synthesizer] Don't test prototypes by default. Exception: test complex pure logic that would be hard to debug manually (algorithms, calculations, parsers). Skip everything else. And be honest with yourself — if this prototype might ship, treat the core as production code from day 1.
</think>

**Default: no tests for prototypes.** Two exceptions:

1. **Complex pure logic** (algorithms, calculations) — test these, they're hard to debug manually
2. **Prototype that might ship** — if there's >30% chance this becomes production, test the core business logic from the start

Skip tests for routes, UI, CRUD, config. Those change too fast in prototypes to justify the overhead."""
},
{
"instruction": "How should I implement authentication in a REST API?",
"output": """<think>
[Architect] Authentication has two parts: identity verification (who are you?) and session management (remembering who you are between requests). For a REST API, stateless JWT tokens are the standard approach — the server doesn't store sessions, the client sends a signed token with each request.

[Critic] JWTs are overused. They have a critical flaw: you can't revoke them before expiry. If a user's account is compromised, you can't invalidate their token without maintaining a blacklist — which defeats the "stateless" benefit. For most apps with <10K users, server-side sessions with Redis are simpler, more secure, and easier to reason about. JWTs only make sense in microservice architectures where services need to verify identity without calling the auth service.

[Researcher] OWASP recommends: use bcrypt or argon2 for password hashing (never SHA-256), short-lived access tokens (15 min) with longer-lived refresh tokens (7 days), HttpOnly secure cookies over Authorization headers (prevents XSS token theft). Auth0's data shows 86% of breaches involve stolen credentials — the token format matters less than proper password handling and HTTPS enforcement.

[Implementer] Using Python FastAPI with JWT:
```python
from fastapi import Depends, HTTPException
from jose import jwt
from passlib.hash import bcrypt

SECRET = os.environ["JWT_SECRET"]

def create_token(user_id: int) -> str:
    return jwt.encode(
        {"sub": user_id, "exp": datetime.utcnow() + timedelta(minutes=15)},
        SECRET, algorithm="HS256"
    )

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
```
Store refresh tokens in the database. Hash passwords with bcrypt on signup.

[Synthesizer] Use short-lived JWTs (15 min) with database-stored refresh tokens (7 days). This gives you stateless verification for most requests while maintaining revocation ability through refresh tokens. Always use bcrypt for passwords, HTTPS everywhere, and HttpOnly cookies for token storage. For apps under 10K users, server-side sessions with Redis are a valid simpler alternative.
</think>

**Recommended auth setup for REST APIs:**

1. **Passwords**: Hash with bcrypt or argon2 (never SHA-256)
2. **Access tokens**: JWT, 15-minute expiry, signed with HS256
3. **Refresh tokens**: Stored in database, 7-day expiry, rotated on use
4. **Transport**: HttpOnly secure cookies (not localStorage — XSS risk)
5. **Always HTTPS**

For small apps (<10K users), server-side sessions with Redis are simpler and equally secure."""
},
{
"instruction": "Explain the CAP theorem in simple terms.",
"output": """<think>
[Architect] CAP theorem states a distributed system can only guarantee two of three properties: Consistency (all nodes see the same data), Availability (every request gets a response), Partition tolerance (system works despite network splits). Since network partitions are inevitable in distributed systems, the real choice is between C and A during a partition.

[Critic] The way CAP is usually taught is misleading. It implies you pick two and lose one permanently, like a menu. In reality, you only sacrifice C or A during a network partition, which is rare. 99.9% of the time your system can be both consistent and available. The theorem only constrains behavior during failure. Also, "consistency" in CAP means linearizability specifically, not just "data is correct."

[Researcher] Eric Brewer (who proposed CAP) wrote a 2012 clarification titled "CAP Twelve Years Later" where he says the 2-of-3 framing is misleading. He recommends thinking in terms of a spectrum. Modern systems like CockroachDB and Google Spanner effectively achieve all three by using synchronized clocks (TrueTime) to minimize the consistency-availability tradeoff window.

[Implementer] Practical examples: PostgreSQL with single-node = CA (no partitions possible). Cassandra = AP (returns stale data during partition, reconciles later). MongoDB with majority write concern = CP (rejects writes during partition if majority unreachable). Choose based on your actual failure tolerance — do stale reads matter more than write downtime?

[Synthesizer] CAP theorem means: during a network partition, you choose between consistent but potentially unavailable (CP) or available but potentially stale (AP). The choice only matters during failures. Pick based on your domain — banking needs CP, social media feeds can tolerate AP.
</think>

**CAP theorem simplified:**

When network issues split your distributed database, you choose:
- **CP**: Reject requests rather than return wrong data (banks, inventory)
- **AP**: Return potentially stale data rather than go down (social feeds, caches)

Key nuance: this tradeoff only applies during network partitions. Normal operation can be both consistent and available."""
},
{
"instruction": "How do I reduce Docker image size?",
"output": """<think>
[Architect] Docker image size affects build time, push/pull speed, storage costs, and attack surface. The three main strategies are: multi-stage builds (separate build and runtime), smaller base images (alpine/distroless), and layer optimization (fewer layers, better caching).

[Critic] People obsess over shaving megabytes with alpine when their real problem is copying node_modules or build artifacts into the final image. A Python app with a 1GB venv doesn't benefit from switching ubuntu to alpine — fix the venv first. Also, alpine uses musl libc which causes subtle bugs with some Python/Node packages. It's not always the right choice.

[Researcher] Benchmarks show multi-stage builds typically reduce image size by 60-80%. Google's distroless images are even smaller than alpine (distroless Python is ~50MB vs alpine's ~80MB) and have zero shell access, reducing attack surface. The official Node alpine image is 180MB vs 1GB for the full image.

[Implementer] Multi-stage build for a Python app:
```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /install /usr/local
COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
```
Key tricks: `--no-cache-dir` on pip, `.dockerignore` for .git/node_modules/venv, combine RUN commands to reduce layers.

[Synthesizer] Three steps in priority order: (1) Multi-stage builds — separate build deps from runtime. (2) Use slim/distroless base images. (3) Add a proper .dockerignore. These three alone typically cut image size by 70%+. Don't use alpine unless you've verified your dependencies work with musl.
</think>

**Reduce Docker image size in 3 steps:**

1. **Multi-stage build** — build in one stage, copy only artifacts to a clean runtime stage (biggest win, ~70% reduction)
2. **Use `slim` or `distroless` base** — avoid full OS images. Avoid alpine if you use Python/Node (musl compatibility issues)
3. **`.dockerignore`** — exclude .git, node_modules, venv, __pycache__, tests"""
},
{
"instruction": "What sorting algorithm should I use?",
"output": """<think>
[Architect] The answer is almost always "use your language's built-in sort." Python's Timsort, Java's dual-pivot quicksort, and C++'s introsort are all highly optimized, cache-friendly, and handle edge cases. Custom sorting algorithms are only needed in very specific scenarios.

[Critic] Agreed that built-in sorts are the default, but there are legitimate cases where a custom algorithm wins: (1) Data is nearly sorted → insertion sort is O(n). (2) Fixed small range of integers → counting sort is O(n). (3) Sorting millions of strings by prefix → radix sort. (4) External sort for data that doesn't fit in memory. The blanket "just use built-in" advice misses these cases.

[Researcher] Timsort (used in Python and Java) is specifically designed for real-world data — it exploits existing ordered subsequences and achieves O(n) on nearly-sorted data. Benchmarks show it outperforms pure mergesort by 2-5x on typical data. For integers in a small range, counting sort is 10-50x faster than comparison sorts. Radix sort beats quicksort on large datasets of fixed-width keys.

[Implementer] Decision tree:
```
Is data already in memory? 
  → YES: Use built-in sort (99% of cases)
  → NO: External merge sort

Is the range of values small and known?
  → YES: Counting sort (O(n+k))

Are you sorting strings/fixed-width keys?
  → YES, millions+: Radix sort

Is data nearly sorted?
  → Built-in Timsort already handles this in O(n)
```

[Synthesizer] Use your language's built-in sort. It's Timsort or introsort, both battle-tested and cache-optimized. Only reach for alternatives in three cases: counting sort for small integer ranges, radix sort for millions of fixed-width keys, or external merge sort for data too large for memory.
</think>

**Use your language's built-in sort.** It's already optimized (Timsort, introsort).

Only three exceptions worth knowing:
1. **Counting sort** — integers in a small known range (O(n+k))
2. **Radix sort** — millions of fixed-width keys (strings, IPs)
3. **External merge sort** — data doesn't fit in RAM"""
},
{
"instruction": "How do I prevent SQL injection?",
"output": """<think>
[Architect] SQL injection happens when user input is concatenated directly into SQL strings. The fundamental fix is parameterized queries — the database driver separates SQL structure from data, making injection impossible by design. This is a solved problem.

[Critic] Parameterized queries are necessary but not sufficient. ORMs can still produce injection in raw query methods (Django's `.extra()`, SQLAlchemy's `text()`). Dynamic ORDER BY and table names can't be parameterized — those need whitelisting. And stored procedures are NOT a defense by themselves if they concatenate internally.

[Researcher] OWASP ranks injection as #3 in their 2021 Top 10 (down from #1 in 2017, because parameterized queries are now the default in most frameworks). Despite this, Verizon's 2023 DBIR shows SQLi is still present in 25% of web app breaches — mostly in legacy code and dynamic query builders. Every major framework now defaults to parameterized queries.

[Implementer] Python examples:
```python
# VULNERABLE — string concatenation
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# SAFE — parameterized query
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# SAFE — ORM (SQLAlchemy)
session.query(User).filter(User.id == user_id).first()

# STILL VULNERABLE — raw SQL in ORM
session.execute(text(f"SELECT * FROM {table_name}"))  # Can't parameterize table names!

# SAFE — whitelist for dynamic identifiers
ALLOWED_TABLES = {"users", "orders", "products"}
if table_name not in ALLOWED_TABLES:
    raise ValueError("Invalid table")
```

[Synthesizer] Three layers of defense: (1) Parameterized queries for all user input — this is non-negotiable. (2) Whitelist validation for dynamic identifiers (table names, column names, ORDER BY). (3) Least-privilege database user — the app's DB account should never have DROP or GRANT permissions.
</think>

**Three layers of SQL injection prevention:**

1. **Parameterized queries** — use `%s` placeholders, never f-strings or concatenation
2. **Whitelist dynamic identifiers** — table/column names can't be parameterized, so validate against an allowed set
3. **Least-privilege DB user** — app account gets SELECT/INSERT/UPDATE/DELETE only, never DROP/GRANT"""
},
{
"instruction": "Should I use GraphQL or REST for my API?",
"output": """<think>
[Architect] REST is resource-oriented (GET /users/123), GraphQL is query-oriented (ask for exactly the fields you need). REST is simpler, better tooling, easier caching. GraphQL solves the overfetching/underfetching problem — mobile clients get exactly the data they need in one request instead of chaining REST calls.

[Critic] GraphQL's flexibility is also its biggest risk. Without proper query complexity limits, a malicious client can send deeply nested queries that DoS your server. Caching is harder because every query is a POST to a single endpoint — you lose HTTP caching entirely. And the "overfetching" problem in REST is solved 90% of the time by sparse fieldsets (?fields=id,name) or BFF pattern, without GraphQL's complexity.

[Researcher] GitHub, Shopify, and Facebook use GraphQL for their public APIs. But Netflix, Stripe, and Twilio use REST. GitHub's own engineering blog notes that GraphQL increased their API maintenance burden. Shopify reported that GraphQL reduced mobile API calls by 40%. The pattern: GraphQL wins when you have many different clients consuming the same data in different shapes.

[Implementer] Decision criteria:
- Single client (web app only)? → REST. Simpler, better caching.
- Multiple clients with different data needs (web + mobile + third-party)? → GraphQL might be worth it.
- Public API for external developers? → REST. Better understood, self-documenting with OpenAPI.
- Internal API with rapidly changing frontend? → GraphQL. Frontend can evolve without backend changes.

[Synthesizer] Default to REST — it's simpler, cacheable, and has better tooling. Consider GraphQL only if you have multiple clients with genuinely different data shape needs (web + mobile + third-party). Never use GraphQL because it's trendy. The decision should be driven by client diversity, not technology hype.
</think>

**Default to REST.** Switch to GraphQL only if:
- Multiple clients need the same data in different shapes (web vs mobile vs third-party)
- Frontend is iterating faster than the API can keep up

REST wins on: caching, simplicity, tooling, documentation, security defaults.
GraphQL wins on: client flexibility, reducing API calls for complex data needs."""
},
{
"instruction": "How do I design a good database schema?",
"output": """<think>
[Architect] Good schema design follows three principles: normalize to 3NF to eliminate redundancy, denormalize strategically for read performance, and index based on actual query patterns. Start normalized, measure, then denormalize specific hot paths.

[Critic] "Normalize to 3NF" is textbook advice that oversimplifies. Over-normalization creates excessive JOINs that kill performance. An orders table that JOINs to customers, products, addresses, and shipping_methods for every query is worse than storing the shipping address directly on the order. The right level of normalization depends on read/write ratio and query patterns.

[Researcher] Markus Winand's "SQL Performance Explained" recommends designing schemas query-first: identify the 5-10 most common queries, then design tables and indexes to serve those queries efficiently. This is the opposite of the traditional approach of designing tables first and hoping queries work out.

[Implementer] Practical rules:
```
1. Every table gets an 'id' primary key (bigint or UUID)
2. Every FK gets an index
3. created_at and updated_at on every table
4. Use ENUMs or lookup tables for status fields
5. Store money as integers (cents), not decimals
6. Soft delete (deleted_at) instead of hard delete
```
Example — store prices as cents:
```sql
-- BAD: floating point money
price DECIMAL(10,2)  -- rounding errors in aggregations

-- GOOD: integer cents
price_cents INTEGER  -- $19.99 stored as 1999
```

[Synthesizer] Design query-first: list your top queries, then design tables to serve them. Start normalized (3NF), then denormalize specific hot paths after measuring. Always index foreign keys, use integer cents for money, and add created_at/updated_at to everything.
</think>

**Database schema design process:**

1. **List your top 10 queries first** — design tables to serve them, not the other way around
2. **Start at 3NF**, denormalize only after measuring slow queries
3. **Always**: index FKs, add created_at/updated_at, use integer cents for money, soft delete
4. **Never**: store money as floats, skip indexes on FKs, hard-delete user data"""
},
{
"instruction": "What's the difference between concurrency and parallelism?",
"output": """<think>
[Architect] Concurrency is about structure — dealing with multiple things at once by interleaving. Parallelism is about execution — doing multiple things simultaneously on multiple cores. A single-core CPU can be concurrent (switching between tasks) but never parallel. A multi-core CPU can be both.

[Critic] The textbook definitions are clear but people still confuse them because most languages blur the line. Python's asyncio is concurrent but not parallel (single thread, event loop). Python's multiprocessing is parallel but not necessarily concurrent in the traditional sense. Go's goroutines are concurrent by default and parallel when GOMAXPROCS > 1.

[Researcher] Rob Pike's talk "Concurrency is not Parallelism" gives the best analogy: imagine a gopher loading a book-burning pile. One gopher with two arms (concurrency) vs. two gophers each with one arm (parallelism). Go was designed around this distinction — goroutines are cheap concurrent tasks, the runtime maps them to OS threads for parallelism automatically.

[Implementer] Python examples:
```python
# CONCURRENCY (asyncio) — one thread, interleaved I/O
async def fetch_all():
    results = await asyncio.gather(
        fetch("url1"), fetch("url2"), fetch("url3")
    )  # 3 requests, 1 thread, overlapped I/O

# PARALLELISM (multiprocessing) — multiple CPU cores
with ProcessPoolExecutor(max_workers=4) as pool:
    results = pool.map(cpu_heavy_task, data_chunks)
    # 4 processes, 4 cores, true simultaneous execution
```
Use concurrency for I/O-bound work (web requests, DB queries). Use parallelism for CPU-bound work (image processing, number crunching).

[Synthesizer] Concurrency = structure (interleaving tasks). Parallelism = execution (simultaneous tasks on multiple cores). Use asyncio/threads for I/O-bound work (concurrency). Use multiprocessing for CPU-bound work (parallelism). Most real applications need both.
</think>

**Concurrency vs Parallelism:**

- **Concurrency**: dealing with many things at once (structure). One barista handling multiple orders by switching between them.
- **Parallelism**: doing many things at once (execution). Four baristas each making a drink simultaneously.

**When to use what:**
- I/O-bound (network, disk) → concurrency (asyncio, threads)
- CPU-bound (math, images) → parallelism (multiprocessing, workers)"""
},
{
"instruction": "How do I handle errors properly in a Python application?",
"output": """<think>
[Architect] Error handling should follow a layered approach: catch specific exceptions close to where they occur, let unexpected errors bubble up, and have a top-level handler that logs and returns clean error responses. Never silently swallow exceptions.

[Critic] The biggest error handling anti-pattern is `except Exception: pass` — it hides bugs. The second biggest is catching too broadly and logging without context. A log message "Error occurred" with no traceback or variable state is useless for debugging. Also, using exceptions for flow control (like checking if a key exists) is slower and less readable than using `.get()` or `in`.

[Researcher] Python's exception hierarchy matters: BaseException includes KeyboardInterrupt and SystemExit, which you almost never want to catch. Exception is the right base for application errors. PEP 20 says "Errors should never pass silently. Unless explicitly silenced." The `logging` module's `exception()` method automatically includes the traceback.

[Implementer] Pattern for a web app:
```python
# Custom exception hierarchy
class AppError(Exception):
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code

class NotFoundError(AppError):
    def __init__(self, resource, id):
        super().__init__(f"{resource} {id} not found", 404)

class ValidationError(AppError):
    def __init__(self, field, reason):
        super().__init__(f"Invalid {field}: {reason}", 422)

# Usage
def get_user(user_id):
    user = db.query(User).get(user_id)
    if not user:
        raise NotFoundError("User", user_id)
    return user

# Top-level handler
@app.errorhandler(AppError)
def handle_app_error(e):
    logger.warning(f"{e.status_code}: {e.message}")
    return {"error": e.message}, e.status_code

@app.errorhandler(Exception)
def handle_unexpected(e):
    logger.exception("Unexpected error")  # Logs full traceback
    return {"error": "Internal server error"}, 500
```

[Synthesizer] Create a custom exception hierarchy for your app's domain errors. Catch specific exceptions, never bare `except:`. Use `logger.exception()` for unexpected errors (includes traceback automatically). Have a top-level handler that converts exceptions to clean API responses. Never silently swallow exceptions.
</think>

**Python error handling rules:**

1. **Custom exception hierarchy** — `AppError` base with `NotFoundError`, `ValidationError` subclasses
2. **Catch specific, not broad** — `except ValueError:` not `except Exception:`
3. **Never `except: pass`** — this hides bugs
4. **Use `logger.exception()`** — automatically includes traceback
5. **Top-level handler** — catches everything, returns clean responses, logs full details"""
},
{
"instruction": "When should I use microservices?",
"output": """<think>
[Architect] Microservices solve organizational scaling — they let independent teams deploy independently. The technical benefits (independent scaling, polyglot tech stack) are secondary to the organizational ones. If you have one team, microservices add distributed systems complexity with no organizational benefit.

[Critic] The microservices hype has caused enormous damage to the industry. Teams of 3-5 developers running 15 microservices with Kubernetes, service meshes, and distributed tracing — all to serve 1000 users. The operational overhead of microservices (network reliability, data consistency, deployment orchestration, monitoring) dwarfs the complexity of a well-structured monolith for 90% of applications.

[Researcher] Amazon's "two-pizza team" rule drove their microservices adoption — each service owned by a team you can feed with two pizzas. Segment.com famously migrated from microservices BACK to a monolith, citing operational overhead. Basecamp runs a monolithic Rails app serving millions of users. The common pattern: successful microservices adoptions are driven by organizational needs (many teams), not technical ones.

[Implementer] Decision framework:
```
Team size < 10 people? → Monolith
Single deployment pipeline? → Monolith
Need independent scaling of specific components? → Extract that one service
Multiple teams needing independent releases? → Microservices make sense
```
If you do go microservices, minimum infrastructure: container orchestration (K8s), service discovery, centralized logging, distributed tracing, API gateway. If you're not ready for ALL of these, you're not ready for microservices.

[Synthesizer] Microservices are an organizational pattern, not a technical one. Use them when you have multiple teams that need to deploy independently. For teams under 10 people, a modular monolith gives you the code organization benefits without the distributed systems tax.
</think>

**Use microservices when:**
- Multiple teams (3+) need independent deploy cycles
- Specific components have genuinely different scaling needs

**Don't use microservices when:**
- Team is under 10 people
- You don't have the infrastructure (K8s, tracing, service discovery)
- You're doing it because "everyone else is"

Default to a modular monolith. Extract services only when organizational or scaling pain demands it."""
},
{
"instruction": "How do I make my website faster?",
"output": """<think>
[Architect] Web performance has three pillars: reduce what you send (smaller payloads), reduce how often you send it (caching), and reduce how far it travels (CDN/edge). Most gains come from the first pillar — shipping less JavaScript.

[Critic] Teams spend weeks optimizing image compression while shipping 2MB of unminified JavaScript. The browser can render an image progressively, but JavaScript blocks the main thread. A 500KB JS bundle on a mid-range phone takes 2-3 seconds just to parse and compile. Always optimize JavaScript first, images second.

[Researcher] Google's Core Web Vitals data shows that the biggest factor in LCP (Largest Contentful Paint) is render-blocking resources. HTTP Archive data: the median website loads 500KB of JavaScript and 1MB of images. Reducing JS by 100KB improves Time to Interactive by 0.5-1 second on mobile. Cloudflare reports their CDN reduces TTFB by 60% on average.

[Implementer] Quick wins checklist:
```
1. Enable gzip/brotli compression (nginx: gzip on;)
2. Add Cache-Control headers (static assets: max-age=31536000)
3. Lazy-load images below the fold (<img loading="lazy">)
4. Code-split JavaScript (dynamic import())
5. Use a CDN for static assets (Cloudflare free tier)
6. Optimize images (WebP format, srcset for responsive)
7. Preconnect to third-party origins (<link rel="preconnect">)
```
These 7 changes typically improve load time by 40-60%.

[Synthesizer] Focus on JavaScript first — it's the biggest performance bottleneck on mobile. Then implement the quick wins: compression, caching headers, lazy loading, CDN. Measure with Lighthouse before and after to verify improvements. These changes are low-effort, high-impact.
</think>

**Web performance quick wins (in priority order):**

1. **Code-split JavaScript** — dynamic `import()`, ship less JS
2. **Enable brotli/gzip** compression on your server
3. **Cache-Control headers** — `max-age=31536000` for static assets
4. **CDN** for static files (Cloudflare free tier works)
5. **Lazy-load images** — `<img loading="lazy">`
6. **WebP images** with responsive `srcset`

Measure before/after with Lighthouse. JavaScript is almost always the #1 bottleneck."""
},
{
"instruction": "How do I structure a Python project properly?",
"output": """<think>
[Architect] Python project structure should separate concerns: source code in a package directory, tests in a parallel directory, configuration at the root. The two dominant patterns are the flat layout (src/ at root) and the src layout (src/package_name/).

[Critic] People overthink this. For most projects, a simple flat layout works fine. The src layout exists to prevent accidental imports of the development version instead of the installed version — a problem most people never encounter. Don't add complexity you don't need. And stop creating __init__.py files that import everything — it causes circular imports.

[Researcher] The Python Packaging Authority (PyPA) recommends the src layout for distributable packages. However, major projects use both: Flask uses flat layout, pytest uses src layout. Kenneth Reitz's "Hitchhiker's Guide to Python" recommends flat layout for applications (not libraries). PEP 517/518 with pyproject.toml has replaced setup.py as the standard.

[Implementer] For an application (not a library):
```
myproject/
├── pyproject.toml        # Build config, dependencies
├── myproject/
│   ├── __init__.py       # Keep empty or minimal
│   ├── main.py           # Entry point
│   ├── config.py         # Settings
│   ├── models/           # Data models
│   ├── services/         # Business logic
│   ├── api/              # Routes/endpoints
│   └── utils/            # Helpers
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_models.py
│   └── test_services.py
├── .env                  # Local env vars (gitignored)
└── README.md
```
Use `pyproject.toml` instead of `setup.py` + `requirements.txt`.

[Synthesizer] Use flat layout for applications, src layout for distributable libraries. Keep __init__.py files minimal. Use pyproject.toml for dependencies. Separate code into models/services/api layers. Don't overthink it — the best structure is one your team agrees on and maintains consistently.
</think>

**Python project structure for applications:**

```
myproject/
├── pyproject.toml          # Dependencies + config (replaces setup.py)
├── myproject/              # Source package
│   ├── __init__.py         # Keep empty
│   ├── main.py             # Entry point
│   ├── models/services/api/
├── tests/                  # Parallel test directory
└── .env                    # Gitignored
```

Rules: flat layout for apps, src layout for libraries. Use pyproject.toml. Keep __init__.py empty."""
},
{
"instruction": "How does garbage collection work in Python?",
"output": """<think>
[Architect] Python uses a dual strategy: reference counting as the primary mechanism (immediate, deterministic cleanup) and a cyclic garbage collector as a backup for reference cycles. Most objects are cleaned up by reference counting; the cyclic GC only handles the edge case of circular references.

[Critic] What most people don't know: the reference counting mechanism means Python objects are cleaned up immediately when their last reference dies — no GC pauses, no unpredictable latency. This is actually better than Java/Go's tracing GC for most workloads. The cyclic GC only runs periodically and only for container objects (lists, dicts, classes) that could form cycles. Primitive types like ints and strings are never tracked by the cyclic GC.

[Researcher] CPython's cyclic GC uses a generational approach with three generations (0, 1, 2). Gen 0 is collected most frequently. Objects that survive collections get promoted. The thresholds default to (700, 10, 10) — gen 0 collects after 700 allocations, gen 1 after 10 gen 0 collections, gen 2 after 10 gen 1 collections. Instagram famously disabled the cyclic GC entirely and saw a 10% memory improvement because they had no reference cycles.

[Implementer] Practical implications:
```python
# Reference counting handles this automatically
def process():
    data = load_big_file()  # refcount = 1
    result = transform(data)
    return result
    # data goes out of scope, refcount hits 0, immediately freed

# Cyclic GC needed here
class Node:
    def __init__(self):
        self.parent = None
        self.children = []

a = Node()
b = Node()
a.children.append(b)  # a -> b
b.parent = a           # b -> a (cycle!)
del a, b  # refcount never hits 0, cyclic GC must collect
```
Use `weakref` to break cycles intentionally. Use `gc.disable()` only if you're certain you have no cycles.

[Synthesizer] Python's memory management is a two-layer system: reference counting (handles 95% of cases instantly) and cyclic GC (handles circular references periodically). For most code, you never need to think about it. Use `weakref` for parent-child relationships that would create cycles. Avoid `gc.disable()` unless you've profiled and proven no cycles exist.
</think>

**Python GC in a nutshell:**

1. **Reference counting** (primary) — object freed instantly when refcount hits 0. Handles 95% of cases.
2. **Cyclic GC** (backup) — periodic sweep for circular references (a→b→a). Only tracks container types.

**Practical advice:**
- Most code needs zero GC tuning
- Use `weakref` for parent-child patterns to avoid cycles
- Never call `gc.collect()` in production unless you're debugging memory leaks"""
},
{
"instruction": "Should I use TypeScript or JavaScript for a new project?",
"output": """<think>
[Architect] TypeScript adds static typing on top of JavaScript. For any project that will live longer than a weekend hackathon, TypeScript pays for itself — IDE autocomplete, catch errors before runtime, self-documenting code. The migration cost is near-zero since any valid JS is valid TS.

[Critic] TypeScript isn't free. The type system adds cognitive overhead, especially generics and conditional types. Over-engineering types (making everything generic, complex mapped types) can make code harder to read than plain JavaScript. Also, TS doesn't help at runtime — a validated API response type only guarantees the shape at compile time, not from the actual server. You still need runtime validation (Zod, Joi).

[Researcher] The 2024 State of JS survey shows 89% of TypeScript users would use it again. GitHub's Octoverse data shows TypeScript is the 4th most popular language, growing faster than JavaScript. Companies that adopted TypeScript report 15-25% fewer production bugs related to type errors (Airbnb engineering blog). However, Svelte recently moved internal code FROM TypeScript back to JSDoc annotations, citing build complexity.

[Implementer] Setup is trivial:
```bash
npm init -y
npm install -D typescript @types/node
npx tsc --init
```
Start with `strict: false` in tsconfig.json, enable strict checks gradually. Key settings:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true
  }
}
```
Use Zod for runtime type validation at API boundaries.

[Synthesizer] Use TypeScript for any project expected to last more than a few weeks or involve more than one developer. The IDE experience alone justifies it. Start with strict mode off, enable gradually. Use Zod for runtime validation at API boundaries since TS types are compile-time only.
</think>

**Use TypeScript** for any project that:
- Will live more than a month
- Has more than 1 developer
- Needs to be maintained by future you

**Stick with JavaScript** for:
- Quick scripts and prototypes
- Learning projects where you don't want tooling overhead

Start with `strict: false`, enable gradually. Always pair with Zod/Joi for runtime validation."""
},
{
"instruction": "How should I handle secrets and environment variables?",
"output": """<think>
[Architect] Secrets management has a clear hierarchy: never hardcode → use env vars → use a secrets manager. Environment variables are the minimum standard. For production, use a dedicated secrets manager (Vault, AWS Secrets Manager) that provides encryption at rest, access auditing, and automatic rotation.

[Critic] .env files are the standard for local development but they create a false sense of security. Developers commit them to git, share them in Slack, copy them between machines. The .env file is just a plaintext file with your database password. Also, environment variables are visible in process listings (`ps aux`), in /proc on Linux, and in crash dumps.

[Researcher] The 12-Factor App methodology mandates configuration in environment variables. However, it was written in 2011 before secrets managers existed. Modern guidance from HashiCorp and AWS recommends secrets managers for anything beyond development. GitGuardian's 2023 report found 10 million hardcoded secrets in public GitHub repos, and 27% of leaked secrets are still valid when discovered.

[Implementer] Layered approach:
```python
# config.py — load from env with sensible defaults
import os
from dotenv import load_dotenv

load_dotenv()  # Local .env file, gitignored

DATABASE_URL = os.environ["DATABASE_URL"]  # Required — crash if missing
DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # Optional with default
API_KEY = os.environ["API_KEY"]  # Required

# .gitignore — ALWAYS include
# .env
# .env.local
# *.pem
# credentials.json
```
For production: use cloud provider's secrets manager. Never store secrets in docker-compose.yml, CI config, or Terraform state.

[Synthesizer] Three rules: (1) Never hardcode secrets — crash on startup if a required secret is missing rather than using a default. (2) .env + dotenv for development, secrets manager for production. (3) .gitignore .env files and audit git history if one was ever committed. Use `os.environ["KEY"]` (crashes if missing) not `os.getenv("KEY", "default")` for secrets.
</think>

**Secrets management rules:**

1. **Never hardcode** — use `os.environ["KEY"]` (crashes if missing, which is what you want)
2. **Development**: `.env` file + `python-dotenv`, .gitignored
3. **Production**: secrets manager (AWS Secrets Manager, Vault)
4. **Always**: .gitignore `.env`, `*.pem`, `credentials.json`
5. **Audit**: if a secret was ever committed, rotate it immediately — git history is forever"""
},
{
"instruction": "Explain how HTTPS works.",
"output": """<think>
[Architect] HTTPS = HTTP over TLS. The key process is the TLS handshake, which establishes an encrypted channel. It has three goals: verify the server's identity (authentication via certificates), agree on encryption keys (key exchange), and encrypt all subsequent data (symmetric encryption).

[Critic] Most explanations skip the critical detail: TLS uses ASYMMETRIC encryption only for the handshake (to exchange keys securely). All actual data transfer uses SYMMETRIC encryption (AES), which is 100-1000x faster. The certificate doesn't encrypt your data — it just proves the server is who it claims to be and facilitates secure key exchange.

[Researcher] TLS 1.3 (2018) reduced the handshake from 2 round-trips to 1 (and 0 with 0-RTT resumption). It removed insecure cipher suites (RC4, 3DES, SHA-1) and eliminated the RSA key exchange in favor of ephemeral Diffie-Hellman only, providing forward secrecy by default. Let's Encrypt has issued over 3 billion free certificates, making HTTPS universal.

[Implementer] TLS 1.3 handshake simplified:
```
Client → Server: "Hello, I support these cipher suites, here's my DH public key"
Server → Client: "Using TLS_AES_256_GCM_SHA384, here's my DH public key + certificate"
[Both derive shared secret from DH key exchange]
Client → Server: "Finished" (encrypted with shared key)
[Encrypted communication begins]
```
One round-trip. Both sides now have the same symmetric key without ever transmitting it.

[Synthesizer] HTTPS uses asymmetric crypto (certificates + Diffie-Hellman) for the handshake to securely establish a shared symmetric key, then switches to fast symmetric encryption (AES) for all data. TLS 1.3 does this in a single round-trip. The certificate proves identity, DH provides key exchange with forward secrecy, AES encrypts the data.
</think>

**HTTPS in 4 steps:**

1. **Client Hello** — "here are the cipher suites I support + my DH public key"
2. **Server Hello** — "using this cipher, here's my certificate + my DH public key"
3. **Key derivation** — both sides compute the same symmetric key from DH exchange (never transmitted!)
4. **Encrypted data** — all HTTP traffic encrypted with AES using the shared key

Key insight: certificates prove identity, they don't encrypt data. AES does the actual encryption."""
},
{
"instruction": "What are the best practices for code review?",
"output": """<think>
[Architect] Code review serves three purposes: catching bugs, sharing knowledge, and maintaining consistency. The most valuable reviews focus on design and logic, not style (that's what linters are for). Reviews should be small (under 400 lines), fast (under 24 hours), and focused.

[Critic] The biggest code review failure mode is rubber-stamping — "LGTM" after 30 seconds on a 500-line PR. The second is bikeshedding — arguing about variable names while missing a race condition. Reviews need structure: check correctness first, then design, then readability. Also, large PRs get worse reviews because reviewers suffer from fatigue.

[Researcher] Google's engineering practices documentation says PRs should be reviewed within one business day. Their data shows that PRs over 400 lines take disproportionately longer and get lower quality feedback. Microsoft Research found that code review catches about 15% of bugs — less than testing — but its knowledge-sharing value is irreplaceable.

[Implementer] Code review checklist:
```
1. Does it do what the PR description says? (Correctness)
2. Are edge cases handled? (null, empty, overflow)
3. Are there security issues? (injection, auth, data exposure)
4. Is the test coverage adequate?
5. Is it readable? Would a new team member understand it?
6. Does it match existing patterns in the codebase?
```
Automate what you can: formatters (black, prettier), linters (eslint, ruff), type checkers. Don't argue about what a tool can enforce.

[Synthesizer] Keep PRs under 400 lines. Review within 24 hours. Use the checklist: correctness → edge cases → security → tests → readability. Automate style with formatters and linters. Focus human review on logic and design, not formatting.
</think>

**Code review best practices:**

1. **Keep PRs under 400 lines** — large PRs get rubber-stamped
2. **Review within 24 hours** — stale PRs slow everyone down
3. **Check in order**: correctness → edge cases → security → tests → readability
4. **Automate style** — use formatters (black/prettier) and linters (ruff/eslint)
5. **Be specific** — "this could NPE if user is null" beats "needs improvement"
6. **Approve with comments** when changes are minor — don't block on nitpicks"""
},
{
"instruction": "How do I debug a memory leak in a Node.js application?",
"output": """<think>
[Architect] Memory leaks in Node.js usually come from four sources: growing arrays/maps that are never cleaned, event listeners that are registered but never removed, closures retaining references to large objects, and global variables accumulating data. The debugging approach: identify the leak with heap snapshots, then find the retaining path.

[Critic] Before debugging, verify it's actually a leak. Node.js's V8 garbage collector is generational and lazy — memory usage rising doesn't mean leak. Watch RSS over hours, not minutes. A real leak shows monotonically increasing heap used after GC cycles. Also, `process.memoryUsage()` can be misleading because `rss` includes memory mapped files.

[Researcher] The most common Node.js leak pattern is event emitter accumulation. If you add a listener in a request handler without removing it, each request adds a listener permanently. Node.js warns at 11 listeners by default (MaxListenersExceededWarning). The Chrome DevTools heap profiler connected to `--inspect` is the most effective tool — it shows retainer trees that trace exactly what's keeping an object alive.

[Implementer] Step-by-step debugging:
```bash
# 1. Start with inspector
node --inspect app.js

# 2. Open chrome://inspect in Chrome

# 3. Take heap snapshot at t=0, t=5min, t=10min

# 4. Compare snapshots — look for objects that grow
# Sort by "Objects allocated between snapshots"
# The Retainers panel shows WHY the object is alive
```
Common fixes:
```javascript
// LEAK: listener added per request, never removed
app.get('/', (req, res) => {
  emitter.on('data', handler);  // Accumulates!
});

// FIX: use once() or remove listener
app.get('/', (req, res) => {
  emitter.once('data', handler);
});

// LEAK: growing cache with no eviction
const cache = {};
app.get('/user/:id', (req, res) => {
  cache[req.params.id] = fetchUser(req.params.id);  // Never evicted!
});

// FIX: use LRU cache with max size
const LRU = require('lru-cache');
const cache = new LRU({ max: 1000 });
```

[Synthesizer] Debug process: (1) Confirm it's a real leak (heap used rising after GC cycles). (2) Take heap snapshots via Chrome DevTools --inspect. (3) Compare snapshots to find growing objects. (4) Check retainer trees to find what's keeping them alive. The fix is usually: remove event listeners, add cache eviction, or break closure references.
</think>

**Node.js memory leak debugging process:**

1. **Confirm the leak** — watch heap used after GC, not just RSS
2. **Start with `--inspect`** — connect Chrome DevTools
3. **Take heap snapshots** at t=0, t=5min, t=10min — compare growth
4. **Check retainer trees** — shows exactly what's keeping objects alive

**Top 3 causes:** unremoved event listeners, unbounded caches, closures holding references"""
},
{
"instruction": "What's the best way to handle file uploads in a web app?",
"output": """<think>
[Architect] File uploads have three concerns: transport (how bytes get from client to server), storage (where files live), and processing (validation, resize, thumbnail). Never store uploads on the application server's filesystem — it breaks when you scale to multiple servers.

[Critic] The biggest mistake is processing uploads synchronously in the request handler. A user uploading a 50MB video shouldn't block a web worker for 30 seconds. Also, trusting the file extension or MIME type for validation is insecure — always check the file's magic bytes. And never serve uploaded files from the same domain as your app — it enables XSS via SVG/HTML uploads.

[Researcher] AWS S3 presigned URLs are the industry standard for production file uploads — the client uploads directly to S3, bypassing your server entirely. This reduces server load, bandwidth costs, and eliminates the upload size limit on your app server. Cloudflare R2 is an S3-compatible alternative with zero egress fees.

[Implementer] Presigned URL flow:
```python
# Server — generate upload URL
import boto3

s3 = boto3.client('s3')

@app.post("/upload-url")
def get_upload_url(filename: str):
    key = f"uploads/{uuid4()}/{filename}"
    url = s3.generate_presigned_url(
        'put_object',
        Params={'Bucket': 'my-bucket', 'Key': key},
        ExpiresIn=300  # 5 minutes
    )
    return {"upload_url": url, "key": key}
```
```javascript
// Client — upload directly to S3
const { upload_url } = await fetch('/upload-url', ...);
await fetch(upload_url, { method: 'PUT', body: file });
```
Process files async (Lambda/worker) after upload completes.

[Synthesizer] Use presigned URLs to upload directly to S3/R2 — your server never touches the file bytes. Process async in a worker (resize, validate magic bytes, generate thumbnails). Serve from a separate domain or CDN to prevent XSS. Never trust client-provided MIME types.
</think>

**File upload architecture:**

1. **Upload**: Client → S3 directly via presigned URL (server generates URL, never touches file bytes)
2. **Process**: Async worker validates magic bytes, resizes images, generates thumbnails
3. **Serve**: CDN or separate domain (never same origin as app — XSS risk)

Never: store on app server filesystem, process synchronously, trust MIME type from client"""
},
{
"instruction": "How do I choose between Redis and Memcached?",
"output": """<think>
[Architect] Both are in-memory key-value stores, but Redis is a superset of Memcached's functionality. Redis supports data structures (lists, sets, sorted sets, hashes), persistence, pub/sub, Lua scripting, and clustering. Memcached is simpler — just key-value with multi-threaded performance.

[Critic] "Just use Redis" is the default answer but Memcached has legitimate advantages. Memcached is multi-threaded — it uses all cores naturally. Redis is single-threaded per instance (you run multiple instances for parallelism). For pure cache workloads (GET/SET of serialized objects), Memcached can outperform Redis due to its simpler memory allocator and multi-threading.

[Researcher] Redis Labs benchmarks show Redis 7.0 handling 1M+ ops/sec on a single instance. Memcached benchmarks show similar throughput but with lower latency at high connection counts due to multi-threading. Facebook uses Memcached as their primary cache (handling billions of operations per day) because their workload is pure cache with no need for data structures. Instagram uses Redis for counters and queues.

[Implementer] Decision:
```
Need data structures (sorted sets, lists, hashes)? → Redis
Need pub/sub or message queues? → Redis
Need persistence/durability? → Redis
Need Lua scripting? → Redis
Pure key-value cache, nothing else? → Either works
Multi-threaded performance at scale? → Memcached
Already have Redis in your stack? → Just use Redis for caching too
```
Most projects should just use Redis — it does everything Memcached does plus more.

[Synthesizer] Use Redis unless you have a specific reason not to. It does everything Memcached does plus data structures, persistence, and pub/sub. Memcached only wins on multi-threaded performance for pure cache workloads at massive scale (Facebook-level). For 99% of projects, Redis is the right choice.
</think>

**Use Redis.** It's a superset of Memcached with data structures, persistence, and pub/sub.

Memcached only wins at Facebook-scale pure caching where multi-threaded performance matters. For everyone else, Redis does more with comparable performance."""
},
{
"instruction": "How do I write good commit messages?",
"output": """<think>
[Architect] Commit messages serve two audiences: present-you (reviewing recent changes) and future-you (understanding why code changed during debugging). The message should explain WHY the change was made, not WHAT was changed — the diff already shows the what.

[Critic] Conventional Commits (feat/fix/chore prefixes) are useful for automated changelogs but they incentivize lazy messages like "fix: fix bug" or "feat: add feature." The prefix isn't a substitute for a descriptive message. Also, squashing all commits into one on merge destroys valuable history of the thought process.

[Researcher] The most-cited convention is Chris Beams' "How to Write a Git Commit Message" — 7 rules including imperative mood, 50-char subject, wrap body at 72 chars. The Linux kernel's commit messages are considered gold standard — each one explains the problem, the root cause, and why this specific fix is correct. Angular's commit convention inspired Conventional Commits.

[Implementer] Template:
```
<type>: <what changed in imperative mood, 50 chars max>

<Why this change was needed. What problem it solves.
What alternative approaches were considered and rejected.
2-3 sentences max.>

<Optional: link to issue, breaking changes, etc.>
```
Examples:
```
fix: prevent race condition in order processing

Two concurrent requests could both read stock=1 and decrement,
resulting in negative inventory. Changed to atomic UPDATE with
WHERE stock > 0 guard.

Closes #234
```
Bad: `fix: fixed stuff`, `update code`, `WIP`, `asdf`

[Synthesizer] Write the WHY, not the WHAT. Use imperative mood ("add" not "added"). Keep subject under 50 chars. Body explains the problem and why this approach was chosen. Use conventional commit prefixes (feat/fix/refactor) for automated tooling but don't let the prefix replace a real message.
</think>

**Commit message formula:**

```
<type>: <imperative description, 50 chars>

<Why was this needed? What problem does it solve?
What alternatives were considered?>
```

**Good**: `fix: prevent negative inventory via atomic UPDATE`
**Bad**: `fix: fix bug`, `update code`, `WIP`

The diff shows WHAT changed. The message should say WHY."""
},
{
"instruction": "What is dependency injection and when should I use it?",
"output": """<think>
[Architect] Dependency injection means passing dependencies into a component rather than having the component create them internally. Instead of a function calling `db = Database()` inside itself, the caller passes `db` as a parameter. This makes the component testable (pass a mock DB) and flexible (swap implementations).

[Critic] DI is overapplied in some ecosystems. Java's Spring framework turns simple code into a maze of annotations, interfaces, and XML config. Python developers rarely need a DI framework — just pass arguments to constructors. The pattern is useful, the frameworks are often overkill. Also, DI doesn't mean you need interfaces for everything — a concrete class as a parameter is still DI.

[Researcher] Martin Fowler coined the term in 2004 to clarify "Inversion of Control." The three types are constructor injection (most common, preferred), setter injection (for optional deps), and interface injection (rarely used). Google's Guice and Java's Spring popularized DI frameworks, but the Go community explicitly avoids DI frameworks in favor of plain constructor injection.

[Implementer] Python example — no framework needed:
```python
# WITHOUT DI — hard to test
class OrderService:
    def __init__(self):
        self.db = PostgresDatabase()  # Hardcoded!
        self.emailer = SMTPEmailer()  # Hardcoded!

# WITH DI — testable, flexible
class OrderService:
    def __init__(self, db, emailer):
        self.db = db
        self.emailer = emailer

# Production
service = OrderService(PostgresDatabase(), SMTPEmailer())

# Testing
service = OrderService(MockDatabase(), MockEmailer())
```
That's it. No framework, no interfaces, no annotations. Just pass your dependencies as constructor arguments.

[Synthesizer] DI is just passing dependencies as arguments instead of hardcoding them. You don't need a framework — constructor parameters are sufficient in Python, Go, and most dynamic languages. Use DI when you need testability (mock external services) or flexibility (swap implementations). Skip DI frameworks unless your language/ecosystem demands them (Spring in Java).
</think>

**Dependency injection = passing dependencies as arguments.** That's it.

```python
# Bad: hardcoded dependency
class UserService:
    def __init__(self):
        self.db = PostgresDB()  # Can't test without real DB

# Good: injected dependency
class UserService:
    def __init__(self, db):
        self.db = db  # Pass real DB or mock
```

No framework needed. Just use constructor arguments. Use DI when you need testability or swappable implementations."""
},
{
"instruction": "How do I handle pagination in an API?",
"output": """<think>
[Architect] Three pagination strategies: offset-based (OFFSET/LIMIT), cursor-based (WHERE id > cursor), and keyset-based (similar to cursor but using indexed columns). Offset is simplest, cursor is most performant for large datasets.

[Critic] Offset pagination has a fatal flaw at scale: `OFFSET 10000 LIMIT 20` forces the database to scan and discard 10,000 rows. It gets slower linearly with page depth. Also, offset pagination is inconsistent — if a new item is inserted while a user is paginating, they'll see duplicates or miss items. Cursor-based pagination solves both problems.

[Researcher] Slack, Twitter, and GitHub all use cursor-based pagination in their APIs. Slack's API docs specifically explain why: offset pagination causes performance degradation at high page numbers. PostgreSQL's EXPLAIN ANALYZE shows OFFSET 100000 LIMIT 20 takes ~200ms vs cursor-based WHERE id > X LIMIT 20 taking ~0.2ms, regardless of position.

[Implementer] Cursor-based pagination:
```python
@app.get("/api/posts")
def list_posts(cursor: str = None, limit: int = 20):
    query = db.query(Post).order_by(Post.created_at.desc())
    
    if cursor:
        # Decode cursor (base64-encoded timestamp)
        ts = decode_cursor(cursor)
        query = query.filter(Post.created_at < ts)
    
    posts = query.limit(limit + 1).all()  # Fetch one extra
    
    has_more = len(posts) > limit
    if has_more:
        posts = posts[:limit]
    
    return {
        "data": posts,
        "next_cursor": encode_cursor(posts[-1].created_at) if has_more else None,
        "has_more": has_more
    }
```
The `limit + 1` trick avoids a separate COUNT query to determine if there's a next page.

[Synthesizer] Use cursor-based pagination for any dataset that could grow large. Offset pagination is fine for admin panels with <1000 total items. Always return `has_more` and `next_cursor` in the response. Use the `limit + 1` trick instead of COUNT queries.
</think>

**Use cursor-based pagination:**

```json
GET /api/posts?cursor=abc123&limit=20

Response:
{
  "data": [...],
  "next_cursor": "def456",
  "has_more": true
}
```

Offset (`OFFSET 10000`) gets slower with page depth. Cursor (`WHERE id > X`) is constant time. Use the `limit + 1` trick to determine `has_more` without a COUNT query."""
},
{
"instruction": "How do I set up CI/CD for a small team?",
"output": """<think>
[Architect] CI/CD has two parts: Continuous Integration (automated testing on every push) and Continuous Deployment (automated deploy on merge to main). For a small team, keep it simple: GitHub Actions + one deploy target. No Kubernetes, no ArgoCD, no service mesh.

[Critic] Small teams often over-engineer CI/CD. A 3-person team doesn't need separate staging, QA, and production environments with manual approval gates. Ship to production on merge to main. If something breaks, roll back. The cost of a bug reaching production for 5 minutes is far less than the cost of a complex release process slowing down every deployment.

[Researcher] GitHub Actions is free for public repos and has 2000 minutes/month for private repos. It has the lowest setup overhead of any CI system. Vercel and Railway offer zero-config deployments for web apps. Fly.io handles Dockerized apps with a single command. None of these require DevOps expertise.

[Implementer] Minimal GitHub Actions CI/CD:
```yaml
# .github/workflows/ci.yml
name: CI/CD
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```
That's it. Tests on every PR, deploy on merge to main. 20 lines.

[Synthesizer] GitHub Actions + one-command deploy (Fly.io, Railway, or Vercel). Test on every PR, deploy on merge to main. No staging environment needed for small teams — use feature flags instead. The entire CI/CD should fit in under 30 lines of YAML.
</think>

**Minimal CI/CD for small teams:**

1. **GitHub Actions** — test on every PR
2. **Auto-deploy on merge to main** — Fly.io, Railway, or Vercel
3. **No staging environment** — use feature flags instead
4. **Total config: ~20 lines of YAML**

Don't over-engineer. If a bug ships, roll back. The speed of deploying 10x per day beats the safety of deploying once a week."""
},
{
"instruction": "Explain how indexes work in databases.",
"output": """<think>
[Architect] A database index is a separate data structure (usually a B-tree) that maps column values to row locations. Without an index, the database must scan every row (full table scan). With an index, it navigates the B-tree in O(log n) to find matching rows directly.

[Critic] Indexes aren't free. Every INSERT, UPDATE, and DELETE must also update every index on that table. A table with 10 indexes means 10 B-tree updates per write. Over-indexing write-heavy tables (like event logs or metrics) can make writes 5-10x slower. Index only what you query, and review unused indexes regularly.

[Researcher] PostgreSQL supports multiple index types: B-tree (default, works for =, <, >, BETWEEN), Hash (only equality, rarely useful), GIN (full-text search, arrays, JSONB), GiST (geometric data, range types). The query planner chooses whether to use an index based on selectivity — if a query matches >10-15% of rows, a full table scan is actually faster than an index lookup.

[Implementer] Index rules of thumb:
```sql
-- Always index: foreign keys, WHERE clauses, JOIN conditions
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Composite index: column order matters (leftmost prefix rule)
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
-- This index supports:
--   WHERE user_id = 1 (yes)
--   WHERE user_id = 1 AND created_at > '2024-01-01' (yes)
--   WHERE created_at > '2024-01-01' (NO — leftmost column not used)

-- Partial index: index only relevant rows
CREATE INDEX idx_active_users ON users(email) WHERE active = true;

-- Check what indexes exist and if they're used
SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;  -- Unused indexes
```

[Synthesizer] Indexes trade write performance for read performance. Always index foreign keys and frequently filtered columns. Use composite indexes with the most selective column first. Check for unused indexes regularly — they slow writes for no benefit. Use EXPLAIN ANALYZE to verify your queries actually use the indexes you created.
</think>

**How indexes work:**
- B-tree data structure mapping column values → row locations
- Turns O(n) full table scans into O(log n) lookups

**Rules:**
1. Always index foreign keys and WHERE clause columns
2. Composite indexes: put most selective column first
3. Every index slows writes — don't over-index
4. Use `EXPLAIN ANALYZE` to verify index usage
5. Drop unused indexes (`pg_stat_user_indexes WHERE idx_scan = 0`)"""
},
{
"instruction": "How do I prevent burnout as a developer?",
"output": """<think>
[Architect] Burnout has three components: exhaustion (too much work), cynicism (not caring about the work), and reduced efficacy (feeling unproductive despite effort). Addressing only one component doesn't fix burnout — you need to identify which one is dominant.

[Critic] Most burnout advice focuses on rest: "take breaks, exercise, sleep more." That treats the symptom, not the cause. If the root cause is a toxic codebase, unclear requirements, or a manager who creates urgency out of nothing, no amount of meditation fixes that. Sometimes the right answer is to change your environment, not your habits.

[Researcher] Research from the Maslach Burnout Inventory shows the top predictor of developer burnout is sustained context switching — being pulled between meetings, Slack messages, code reviews, and coding tasks. Cal Newport's "Deep Work" research shows that knowledge workers need 3-4 hour uninterrupted blocks to do meaningful work. Developers who get fewer than 2 uninterrupted hours per day report 3x higher burnout rates.

[Implementer] Tactical changes that help:
```
1. Block 3-4 hours daily as "no meetings" time
2. Turn off Slack notifications during deep work
3. Say "no" or "not now" to interrupt-driven requests
4. Finish one task before starting another (WIP limit of 1)
5. Ship something small every day (progress = motivation)
6. Stop working at a fixed time (e.g., 6pm hard stop)
7. Work on a personal project that excites you (prevents cynicism)
```

[Synthesizer] Protect uninterrupted deep work time (3-4 hours daily). Limit work-in-progress to one task. Ship small things daily for a sense of progress. Stop at a fixed time. And honestly evaluate whether the problem is your habits or your environment — if it's the environment, changing habits won't fix it.
</think>

**Preventing developer burnout:**

1. **Protect deep work** — 3-4 hour blocks with no meetings/Slack
2. **WIP limit of 1** — finish before starting something new
3. **Ship daily** — small progress compounds and keeps motivation alive
4. **Hard stop time** — leave at 6pm, guilt-free
5. **Honest assessment** — is the problem your habits or your environment? If the environment is toxic, rest won't fix it"""
}
]

# Write train.jsonl
with open("data/team_thinking/train.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} training examples")

# Eval examples
eval_examples = [
{
"instruction": "Should I use WebSockets or Server-Sent Events for real-time updates?",
"output": "<think>\n[Architect] SSE is unidirectional (server→client), simpler, and works over HTTP/1.1 with automatic reconnection. WebSockets are bidirectional (server↔client), lower latency, but more complex (custom protocol, no auto-reconnect). Choose based on whether the client needs to send real-time data back.\n\n[Critic] WebSockets are overused. Most 'real-time' features are actually server-push only — notifications, live feeds, dashboards. SSE handles all of these with less complexity. WebSockets add connection management burden, custom heartbeats, and don't work through some corporate proxies. Only use WebSockets for truly bidirectional apps like chat or collaborative editing.\n\n[Researcher] HTTP/2 makes SSE even more efficient since multiple SSE streams can share a single TCP connection. EventSource API has native browser support with automatic reconnection and last-event-ID resumption. WebSocket connections each consume a separate TCP connection. Mercure (an SSE-based protocol) is used by Symfony and API Platform for real-time without WebSocket complexity.\n\n[Implementer] SSE in Python:\n```python\n@app.get('/events')\nasync def stream():\n    async def generator():\n        while True:\n            data = await get_updates()\n            yield f'data: {json.dumps(data)}\\n\\n'\n    return StreamingResponse(generator(), media_type='text/event-stream')\n```\nClient: `new EventSource('/events')` — 2 lines, auto-reconnect built in.\n\n[Synthesizer] Use SSE for server-push (notifications, feeds, dashboards) — simpler, auto-reconnect, works through proxies. Use WebSockets only for bidirectional real-time (chat, collaborative editing, gaming). SSE covers 80%+ of real-time use cases with less complexity.\n</think>\n\n**SSE for server-push (80% of cases):** notifications, live feeds, dashboards. Auto-reconnect, simpler.\n**WebSockets for bidirectional:** chat, collaborative editing, gaming. More complex but necessary for two-way communication."
},
{
"instruction": "How do I choose between a monorepo and multiple repositories?",
"output": "<think>\n[Architect] Monorepo puts all projects in one repository with shared tooling. Multi-repo gives each project its own repository with independent versioning. The tradeoff is between cross-project coordination (monorepo wins) and independent deployment autonomy (multi-repo wins).\n\n[Critic] Monorepos are romanticized because Google and Facebook use them, but those companies built custom tooling worth millions to make monorepos work (Bazel, Buck). Without those tools, a large monorepo in git becomes painfully slow — git status takes seconds, CI runs everything on every change. If you don't have the tooling budget, multi-repo with good automation is more practical.\n\n[Researcher] Google's monorepo contains 2 billion lines of code but uses Piper (custom VCS), not git. Git performance degrades significantly above 1-2 million files. Tools like Nx, Turborepo, and Bazel mitigate this with smart caching and affected-project detection. Vercel's Turborepo reduced CI times by 65-85% in monorepo setups.\n\n[Implementer] Decision framework:\n- Tight coupling between projects? → Monorepo (shared types, atomic cross-project changes)\n- Independent teams, independent release cycles? → Multi-repo\n- Shared libraries used across projects? → Monorepo with Turborepo/Nx\n- < 5 projects? → Monorepo is simpler\n- > 20 projects? → Multi-repo unless you invest in tooling\n\n[Synthesizer] Under 5 projects with shared code: monorepo with Turborepo. Independent teams with separate deploy cycles: multi-repo. Don't copy Google's monorepo without Google's tooling budget.\n</think>\n\n**Monorepo** if: shared code between projects, tight coupling, small team.\n**Multi-repo** if: independent teams, separate deploy cycles, >20 projects without tooling budget."
},
{
"instruction": "What are the tradeoffs of using an ORM vs raw SQL?",
"output": "<think>\n[Architect] ORMs (SQLAlchemy, Prisma, ActiveRecord) abstract SQL behind objects and methods. They provide safety (parameterized queries by default), portability (switch databases), and faster development for CRUD operations. Raw SQL gives full control, better performance for complex queries, and direct access to database-specific features.\n\n[Critic] ORMs have two failure modes: (1) The N+1 query problem — loading a list of users then lazily loading each user's posts generates N+1 separate queries instead of one JOIN. (2) Complex queries that fight the ORM — trying to express a window function or CTE through ORM methods produces unreadable code that's worse than raw SQL. Most teams end up writing raw SQL for 20-30% of queries anyway.\n\n[Researcher] Django's ORM documentation explicitly recommends raw SQL for complex reporting queries. Rails' ActiveRecord has `.find_by_sql()` for the same reason. Prisma's approach is interesting — it generates typed client code from a schema, giving you ORM convenience with type safety, but still allows raw SQL via `$queryRaw`. SQLAlchemy's Core (not ORM) layer is a middle ground — SQL expression builder without object mapping.\n\n[Implementer] Use both:\n```python\n# ORM for CRUD (90% of queries)\nuser = User.query.get(id)\nuser.name = 'new_name'\ndb.session.commit()\n\n# Raw SQL for complex analytics (10% of queries)\nresult = db.session.execute(text(\"\"\"\n    SELECT department, AVG(salary), COUNT(*)\n    FROM employees\n    WHERE hired_at > :date\n    GROUP BY department\n    HAVING COUNT(*) > 5\n    ORDER BY AVG(salary) DESC\n\"\"\"), {'date': cutoff_date})\n```\n\n[Synthesizer] Use ORM for CRUD and simple queries (80-90% of your code). Drop to raw SQL for complex reporting, analytics, and performance-critical queries. Watch for N+1 queries — use eager loading (joinedload/selectinload). This hybrid approach gives you productivity and control.\n</think>\n\n**Use both:** ORM for CRUD (fast development, safe defaults), raw SQL for complex queries (performance, full control).\n\nWatch for: N+1 queries (use eager loading), fighting the ORM on complex joins (just write SQL)."
},
{
"instruction": "How do I design an API that won't break when I change it?",
"output": "<think>\n[Architect] API stability comes from versioning, backward compatibility, and additive-only changes. The key principle: you can ADD fields and endpoints, but never REMOVE or RENAME them without a version bump. Clients should ignore unknown fields (Postel's Law).\n\n[Critic] API versioning is necessary but often over-applied. Most breaking changes can be avoided by designing carefully upfront. Adding a field is backward compatible. Removing a field breaks clients. Changing a field's type breaks clients. If you find yourself versioning frequently, the real problem is insufficient upfront design.\n\n[Researcher] Stripe is considered the gold standard for API design. Their approach: URL-based versioning (v1, v2), all changes are backward compatible within a version, and old versions are supported for years. They maintain a compatibility layer that translates between internal models and each API version. Stripe has never removed a v1 endpoint since 2011.\n\n[Implementer] Practical rules:\n```\n1. Use URL versioning: /api/v1/users\n2. Never remove fields — deprecate with docs\n3. Never change field types — add a new field instead\n4. Use optional fields with defaults for new additions\n5. Return envelope: {\"data\": ..., \"meta\": ...}\n6. Document deprecations with sunset dates\n```\nExample of additive change:\n```json\n// v1 response (original)\n{\"name\": \"John Doe\"}\n\n// v1 response (after change — backward compatible)\n{\"name\": \"John Doe\", \"first_name\": \"John\", \"last_name\": \"Doe\"}\n// 'name' kept for backward compat, new fields added\n```\n\n[Synthesizer] Design for additive-only changes. Add fields freely, never remove or rename. Use URL versioning for genuine breaking changes. Study Stripe's API design — it's the best example of long-term API stability in production.\n</think>\n\n**API stability rules:**\n1. Add fields freely, never remove or rename\n2. URL versioning for breaking changes (/api/v1/)\n3. Keep deprecated fields alive with sunset dates\n4. New fields are always optional with defaults\n5. Study Stripe's API — gold standard since 2011"
},
{
"instruction": "What's the difference between TCP and UDP?",
"output": "<think>\n[Architect] TCP provides reliable, ordered delivery with connection state. UDP provides unreliable, unordered delivery with no connection overhead. TCP guarantees every byte arrives correctly; UDP is fire-and-forget. The tradeoff is reliability vs latency.\n\n[Critic] The 'TCP is slow, UDP is fast' framing is oversimplified. TCP's overhead is in connection setup (3-way handshake) and congestion control, not in the actual data transfer. For bulk data transfer, TCP is often faster because its congestion control optimizes throughput. UDP wins on latency for small, frequent messages where occasional loss is acceptable.\n\n[Researcher] HTTP/3 (QUIC) uses UDP as its transport layer but implements its own reliability on top — proving that UDP+custom reliability can outperform TCP for specific use cases. Gaming uses UDP because a dropped position update from 50ms ago is useless — the next update at 16ms will override it. VoIP uses UDP because a dropped audio frame causes a tiny glitch, while TCP's retransmission would cause noticeable delay.\n\n[Implementer] When to use which:\n```\nTCP: Web (HTTP), email (SMTP), file transfer (FTP/SCP)\n     → Any time data must arrive completely and correctly\n\nUDP: Gaming (real-time state), VoIP, video streaming,\n     DNS queries, IoT sensor data\n     → Any time speed matters more than completeness\n\nQUIC (UDP-based): HTTP/3, modern web\n     → Reliability of TCP + speed of UDP\n```\n\n[Synthesizer] TCP for correctness-critical data (web, files, email). UDP for latency-critical data where loss is tolerable (gaming, VoIP, streaming). QUIC (HTTP/3) is the modern hybrid — UDP transport with application-level reliability, giving TCP's guarantees with UDP's speed.\n</think>\n\n**TCP**: Reliable, ordered, connection-based. Use for: web, files, email.\n**UDP**: Fast, unreliable, connectionless. Use for: gaming, VoIP, streaming.\n**QUIC/HTTP3**: UDP transport + custom reliability = best of both worlds."
},
{
"instruction": "How do I manage technical debt effectively?",
"output": "<think>\n[Architect] Technical debt is a spectrum from intentional (\"we'll ship fast now and refactor later\") to accidental (\"we didn't know better\"). Managing it requires tracking, prioritizing by impact, and dedicating time to pay it down — not just acknowledging it exists.\n\n[Critic] Most 'technical debt' conversations are vague hand-waving. Saying 'we have tech debt' is like saying 'we have expenses.' The question is: which specific debt is costing you the most? A messy but stable module that nobody touches isn't urgent. A fragile module that breaks every sprint and slows every feature is critical.\n\n[Researcher] Martin Fowler's Technical Debt Quadrant categorizes debt as deliberate vs inadvertent and reckless vs prudent. The most dangerous type is 'inadvertent prudent' — debt you don't know you have because you didn't know better at the time. Thoughtworks recommends spending 15-20% of each sprint on tech debt reduction, tracked as regular backlog items.\n\n[Implementer] Practical system:\n```\n1. Label tech debt tickets with impact:\n   - P1: Blocks features or causes outages\n   - P2: Slows development by >20%\n   - P3: Annoyance, cosmetic, no impact\n\n2. 20% rule: 1 day per sprint for tech debt\n\n3. Boy Scout Rule: leave code better than you found it\n   (Small refactors during feature work, not separate tickets)\n\n4. Never rewrite from scratch — refactor incrementally\n```\n\n[Synthesizer] Track tech debt as prioritized tickets, not vague complaints. Spend 20% of sprint time on the highest-impact items. Apply the Boy Scout Rule for small improvements during feature work. Never rewrite from scratch — incremental refactoring is always safer.\n</think>\n\n**Managing tech debt:**\n1. **Track it** — concrete tickets, not vague complaints\n2. **Prioritize by impact** — what slows you down most?\n3. **20% rule** — 1 day per sprint dedicated to debt reduction\n4. **Boy Scout Rule** — improve code you touch during feature work\n5. **Never full rewrite** — incremental refactoring is safer"
},
{
"instruction": "What's the best way to store passwords?",
"output": "<think>\n[Architect] Passwords must be hashed with a slow, salted algorithm. The system should never store or be able to recover the original password. On login, hash the input and compare hashes. The hash function must be deliberately slow to make brute-force attacks expensive.\n\n[Critic] Using SHA-256 or MD5 for passwords is a critical security mistake, even with a salt. These algorithms are designed to be fast — a GPU can compute billions of SHA-256 hashes per second. Password hashing algorithms (bcrypt, argon2) are designed to be slow, specifically to resist brute force. This is one area where 'faster' is worse.\n\n[Researcher] OWASP recommends Argon2id as the first choice (winner of the 2015 Password Hashing Competition), bcrypt as second choice. Argon2id is memory-hard (resists GPU/ASIC attacks). Bcrypt has a 20-year track record. A bcrypt hash with cost factor 12 takes ~250ms per hash — fast enough for login, too slow for brute force (4 hashes/second vs billions for SHA-256).\n\n[Implementer] Python with bcrypt:\n```python\nimport bcrypt\n\n# Registration\ndef hash_password(password: str) -> str:\n    salt = bcrypt.gensalt(rounds=12)\n    return bcrypt.hashpw(password.encode(), salt).decode()\n\n# Login\ndef verify_password(password: str, hashed: str) -> bool:\n    return bcrypt.checkpw(password.encode(), hashed.encode())\n\n# Store the hash in your database, never the password\nhashed = hash_password('user_password_123')\nassert verify_password('user_password_123', hashed)  # True\nassert not verify_password('wrong_password', hashed)  # False\n```\nNever implement your own hashing. Never use MD5/SHA for passwords.\n\n[Synthesizer] Use bcrypt (cost 12) or argon2id. Never SHA-256/MD5. The algorithm must be deliberately slow. Store only the hash. Use a well-tested library — never implement your own password hashing.\n</think>\n\n**Password storage: use bcrypt or argon2id. Never SHA-256/MD5.**\n\n```python\nhashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))\n```\n\nRules: slow algorithm (bcrypt/argon2), never store plaintext, never roll your own crypto, cost factor 12+ for bcrypt."
},
{
"instruction": "How do I decide between a queue and a pub/sub system?",
"output": "<think>\n[Architect] Queues (RabbitMQ, SQS) deliver each message to exactly one consumer — work distribution. Pub/sub (Kafka, Redis pub/sub, SNS) delivers each message to all subscribers — event broadcasting. The choice depends on whether you need competing consumers (queue) or fan-out (pub/sub).\n\n[Critic] Kafka blurs this line because it supports both patterns. Consumer groups in Kafka act as competing consumers (queue semantics), while multiple consumer groups each get all messages (pub/sub semantics). Don't choose a technology based on queue-vs-pubsub — choose based on durability, throughput, and ordering requirements.\n\n[Researcher] RabbitMQ handles ~50K messages/sec with strong ordering guarantees per queue. Kafka handles millions of messages/sec with ordering per partition. SQS is fully managed with no infrastructure overhead but has higher latency (~20ms vs <1ms). For most applications under 10K messages/sec, any of these work — the operational complexity differs more than the performance.\n\n[Implementer] Decision tree:\n```\nNeed exactly-once processing of tasks? → SQS or RabbitMQ\nNeed event replay/reprocessing? → Kafka (stores events persistently)\nNeed fan-out to multiple services? → SNS+SQS or Kafka consumer groups\nWant zero infrastructure? → SQS (AWS managed)\nNeed <1ms latency? → RabbitMQ or Redis Streams\n```\n\n[Synthesizer] Use queues (SQS/RabbitMQ) for work distribution where each task runs once. Use pub/sub (Kafka/SNS) for event broadcasting where multiple services react to the same event. Kafka works for both but adds operational complexity. SQS is the lowest-effort choice for most applications.\n</think>\n\n**Queue** (SQS, RabbitMQ): each message processed by ONE consumer. Use for: background jobs, task distribution.\n**Pub/Sub** (Kafka, SNS): each message received by ALL subscribers. Use for: event broadcasting, service decoupling.\n\nStart with SQS (zero ops) unless you need event replay (Kafka) or sub-millisecond latency (RabbitMQ)."
},
{
"instruction": "How should I handle database migrations in production?",
"output": "<think>\n[Architect] Database migrations must be backward compatible with the running application code. The deploy sequence is: apply migration → deploy new code. If the migration breaks the old code, you get downtime during the deploy window. Every migration should be safe to run while the previous version of the app is still serving traffic.\n\n[Critic] The most dangerous migration pattern is renaming a column. Old code references the old name, new code references the new name. During deployment, one version will fail. The safe approach is a 3-step process: (1) add new column, (2) deploy code using both, (3) drop old column. This is slow but safe. Also, never run migrations manually — automate them in the deploy pipeline.\n\n[Researcher] GitHub's approach (documented in their engineering blog): every migration must pass the 'dual version' test — both the current and next version of the application must work with the schema. They use the `gh-ost` tool for zero-downtime schema changes on large tables. For PostgreSQL, `pg_repack` and `CREATE INDEX CONCURRENTLY` avoid table locks.\n\n[Implementer] Safe migration patterns:\n```sql\n-- SAFE: add nullable column (no lock, no rewrite)\nALTER TABLE users ADD COLUMN phone VARCHAR(20);\n\n-- SAFE: add index without locking\nCREATE INDEX CONCURRENTLY idx_users_email ON users(email);\n\n-- DANGEROUS: rename column (breaks old code)\n-- Instead, do it in 3 deploys:\n-- Deploy 1: ADD new_col, backfill, code writes to both\n-- Deploy 2: Code reads from new_col only\n-- Deploy 3: DROP old_col\n\n-- DANGEROUS: add NOT NULL without default (locks table)\n-- Instead:\nALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active';\n```\n\n[Synthesizer] Every migration must be backward compatible with the currently running code. Automate migrations in deploy pipeline. Use `CONCURRENTLY` for index creation. Never rename columns in one step — use the 3-deploy expand/migrate/contract pattern. Test migrations against a production-sized dataset before deploying.\n</think>\n\n**Production migration rules:**\n1. Every migration must work with BOTH old and new code\n2. Automate in deploy pipeline — never run manually\n3. Add columns as nullable or with defaults\n4. Use `CREATE INDEX CONCURRENTLY`\n5. Never rename/drop columns in one step — use 3-deploy expand/migrate/contract\n6. Test against production-sized data first"
},
{
"instruction": "What's the difference between horizontal and vertical scaling?",
"output": "<think>\n[Architect] Vertical scaling means bigger machines (more CPU, RAM). Horizontal scaling means more machines behind a load balancer. Vertical is simpler (no distributed systems complexity) but has a ceiling. Horizontal scales further but requires stateless application design.\n\n[Critic] The industry has a bias toward horizontal scaling because it sounds more impressive, but vertical scaling is underrated. A single modern server with 128 cores and 512GB RAM can handle enormous workloads. Stack Overflow serves 1.3 billion page views per month from a handful of servers through vertical scaling. Horizontal scaling adds distributed systems headaches — session management, data consistency, deploy complexity.\n\n[Researcher] AWS's largest instance (u-24tb1.metal) has 448 vCPUs and 24TB of RAM. That handles most applications. The point where vertical scaling becomes impractical varies — for compute-bound workloads it's when you need more than ~400 cores; for storage-bound workloads it's when you need more than one machine's disk throughput. Cloudflare's entire DNS infrastructure runs on horizontal scaling because no single machine can handle their request volume.\n\n[Implementer] When to use which:\n```\nVertical scaling (scale up):\n  + Simpler — no distributed systems\n  + Works with stateful apps\n  - Has a ceiling\n  - Single point of failure\n  Use for: databases, monoliths, early-stage apps\n\nHorizontal scaling (scale out):\n  + No ceiling\n  + Redundancy built in\n  - Requires stateless design\n  - Distributed systems complexity\n  Use for: web servers, API tiers, microservices\n```\nMost apps: vertically scale the database, horizontally scale the app tier.\n\n[Synthesizer] Scale vertically first — it's simpler and goes further than people think. Scale horizontally when you hit the ceiling or need redundancy. Most architectures: vertical database + horizontal app servers. Don't add distributed systems complexity until you have concrete evidence it's needed.\n</think>\n\n**Vertical** (bigger machine): simpler, no distributed complexity, has a ceiling. Scale your database this way.\n**Horizontal** (more machines): no ceiling, requires stateless design. Scale your app servers this way.\n\nDefault strategy: vertical database + horizontal app tier. Don't go horizontal until vertical hits its limit."
}
]

with open("data/team_thinking/eval.jsonl", "w", encoding="utf-8") as f:
    for ex in eval_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(eval_examples)} eval examples")
print("Done!")
