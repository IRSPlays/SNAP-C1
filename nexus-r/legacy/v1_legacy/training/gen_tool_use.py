"""Generate high-quality tool_use training data for SNAP-C1.

All 7 tools:
  execute_code, read_file, write_file, edit_file,
  search_files, shell_command, web_search

50 training + 10 eval examples with:
  - Real tool calls using <tool_call> / <tool_result> tags
  - Multi-tool workflows
  - Error handling
  - Diverse tasks
"""
import json

examples = [
# ============================================================
# 1. read_file — simple file read
# ============================================================
{
"instruction": "Show me the contents of config.yaml",
"output": """<think>
The user wants to see the contents of a configuration file. I'll use read_file to retrieve it.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "config.yaml"}}
</tool_call>
<tool_result>
{"status": "success", "content": "database:\\n  host: localhost\\n  port: 5432\\n  name: myapp_db\\n  user: admin\\n\\nserver:\\n  port: 8080\\n  debug: false\\n  workers: 4\\n\\nlogging:\\n  level: INFO\\n  file: logs/app.log"}
</tool_result>

Here's the contents of `config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: myapp_db
  user: admin

server:
  port: 8080
  debug: false
  workers: 4

logging:
  level: INFO
  file: logs/app.log
```

The config defines database connection settings (PostgreSQL on localhost:5432), a server running on port 8080 with 4 workers, and INFO-level logging to `logs/app.log`."""
},
# ============================================================
# 2. execute_code — run Python code
# ============================================================
{
"instruction": "Calculate the first 20 Fibonacci numbers",
"output": """<think>
I'll write and execute a Python script to generate the first 20 Fibonacci numbers.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "def fibonacci(n):\\n    fibs = [0, 1]\\n    for i in range(2, n):\\n        fibs.append(fibs[-1] + fibs[-2])\\n    return fibs[:n]\\n\\nresult = fibonacci(20)\\nfor i, f in enumerate(result):\\n    print(f'F({i}) = {f}')\\nprint(f'\\\\nSum: {sum(result)}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "F(0) = 0\\nF(1) = 1\\nF(2) = 1\\nF(3) = 2\\nF(4) = 3\\nF(5) = 5\\nF(6) = 8\\nF(7) = 13\\nF(8) = 21\\nF(9) = 34\\nF(10) = 55\\nF(11) = 89\\nF(12) = 144\\nF(13) = 233\\nF(14) = 377\\nF(15) = 610\\nF(16) = 987\\nF(17) = 1597\\nF(18) = 2584\\nF(19) = 4181\\n\\nSum: 10945"}
</tool_result>

The first 20 Fibonacci numbers are:

```
F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13,
F(8)=21, F(9)=34, F(10)=55, F(11)=89, F(12)=144, F(13)=233,
F(14)=377, F(15)=610, F(16)=987, F(17)=1597, F(18)=2584, F(19)=4181
```

Their sum is **10,945**."""
},
# ============================================================
# 3. search_files + read_file — find then read
# ============================================================
{
"instruction": "Find where the database connection is configured in this project",
"output": """<think>
I need to search for database connection configuration. I'll search for common patterns like "database", "DB_HOST", or "connection string".
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "database|DB_HOST|connection.*string|create_engine", "path": "src/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/config/database.py", "line": 8, "text": "DB_HOST = os.getenv('DB_HOST', 'localhost')"}, {"file": "src/config/database.py", "line": 9, "text": "DB_PORT = int(os.getenv('DB_PORT', '5432'))"}, {"file": "src/config/database.py", "line": 14, "text": "engine = create_engine(DATABASE_URL)"}, {"file": "src/models/base.py", "line": 3, "text": "from src.config.database import engine"}]}
</tool_result>

Found the database configuration in `src/config/database.py`. Let me read the full file:

<tool_call>
{"name": "read_file", "args": {"path": "src/config/database.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "import os\\nfrom sqlalchemy import create_engine\\nfrom sqlalchemy.orm import sessionmaker\\n\\n# Database configuration from environment\\nDB_HOST = os.getenv('DB_HOST', 'localhost')\\nDB_PORT = int(os.getenv('DB_PORT', '5432'))\\nDB_NAME = os.getenv('DB_NAME', 'myapp')\\nDB_USER = os.getenv('DB_USER', 'postgres')\\nDB_PASS = os.getenv('DB_PASS', '')\\n\\nDATABASE_URL = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'\\n\\nengine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)\\nSessionLocal = sessionmaker(bind=engine)\\n\\ndef get_db():\\n    db = SessionLocal()\\n    try:\\n        yield db\\n    finally:\\n        db.close()"}
</tool_result>

The database connection is configured in `src/config/database.py`:

- Uses environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`) with localhost defaults
- Creates a SQLAlchemy engine with PostgreSQL, pool_size=10 and max_overflow=20
- Provides a `get_db()` dependency that yields sessions and auto-closes them

It's also imported in `src/models/base.py` to connect the ORM models to the engine."""
},
# ============================================================
# 4. write_file — create a new file
# ============================================================
{
"instruction": "Create a .gitignore file for a Python project",
"output": """<think>
I'll create a comprehensive .gitignore for a Python project covering common patterns: virtual environments, bytecode, IDE files, OS files, and secrets.
</think>

<tool_call>
{"name": "write_file", "args": {"path": ".gitignore", "content": "# Python bytecode\\n__pycache__/\\n*.py[cod]\\n*.pyo\\n\\n# Virtual environments\\nvenv/\\nenv/\\n.venv/\\n\\n# Distribution / packaging\\ndist/\\nbuild/\\n*.egg-info/\\n*.egg\\n\\n# IDE\\n.vscode/\\n.idea/\\n*.swp\\n*.swo\\n\\n# Environment variables / secrets\\n.env\\n.env.local\\n.env.production\\ncredentials.json\\n*.pem\\n*.key\\n\\n# Testing\\n.pytest_cache/\\nhtmlcov/\\n.coverage\\n\\n# OS files\\n.DS_Store\\nThumbs.db\\n\\n# Logs\\nlogs/\\n*.log\\n\\n# Database\\n*.sqlite3\\n*.db"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: .gitignore (42 lines)"}
</tool_result>

Created `.gitignore` with coverage for:
- **Python**: `__pycache__/`, `.pyc` files, dist/build artifacts
- **Environments**: `venv/`, `.env` files
- **IDE**: `.vscode/`, `.idea/`
- **Security**: credentials, PEM keys, env files
- **Testing**: pytest cache, coverage reports
- **OS**: `.DS_Store`, `Thumbs.db`"""
},
# ============================================================
# 5. shell_command — git operations
# ============================================================
{
"instruction": "Show me the recent git history and current branch",
"output": """<think>
I'll run git commands to show the current branch and recent commit history.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "git branch --show-current"}}
</tool_call>
<tool_result>
{"status": "success", "output": "feature/user-auth"}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "git log --oneline -10"}}
</tool_call>
<tool_result>
{"status": "success", "output": "a3f7c21 feat: add JWT token refresh endpoint\\n8b2e1d4 fix: handle expired token gracefully\\n5c9a0f3 feat: implement login endpoint with bcrypt\\n2d1b8e7 feat: add User model with password hashing\\ne4f6a92 chore: install auth dependencies (pyjwt, bcrypt)\\n1a3c5d8 refactor: split routes into separate modules\\n9f2b4e1 docs: add API documentation\\n7c8d3a6 feat: add health check endpoint\\nb5e1f29 init: project setup with FastAPI\\nc0a2d47 init: initial commit"}
</tool_result>

You're on branch `feature/user-auth`. Recent history shows:

1. `a3f7c21` — JWT token refresh endpoint (latest)
2. `8b2e1d4` — Fix for expired token handling
3. `5c9a0f3` — Login endpoint with bcrypt
4. `2d1b8e7` — User model with password hashing
5. `e4f6a92` — Auth dependencies installed

Looks like you're actively building out the authentication system."""
},
# ============================================================
# 6. edit_file — fix a bug in existing code
# ============================================================
{
"instruction": "Fix the off-by-one error in the pagination function in src/utils/pagination.py",
"output": """<think>
I need to first read the file to see the current code, then identify and fix the off-by-one error.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/utils/pagination.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "def paginate(items, page, per_page=20):\\n    total = len(items)\\n    total_pages = total // per_page\\n    start = (page - 1) * per_page\\n    end = start + per_page\\n    return {\\n        'items': items[start:end],\\n        'page': page,\\n        'per_page': per_page,\\n        'total': total,\\n        'total_pages': total_pages,\\n    }"}
</tool_result>

Found it. The `total_pages` calculation uses integer division (`//`) which drops the remainder. If there are 25 items with per_page=20, it returns `total_pages=1` instead of `2`, hiding the last 5 items.

<tool_call>
{"name": "edit_file", "args": {"path": "src/utils/pagination.py", "old_text": "    total_pages = total // per_page", "new_text": "    total_pages = (total + per_page - 1) // per_page"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/utils/pagination.py"}
</tool_result>

Fixed the off-by-one error. The ceiling division formula `(total + per_page - 1) // per_page` correctly rounds up:
- 25 items, 20 per page → `(25 + 19) // 20 = 2` pages (correct)
- 40 items, 20 per page → `(40 + 19) // 20 = 2` pages (correct)
- 0 items, 20 per page → `(0 + 19) // 20 = 0` pages (correct)"""
},
# ============================================================
# 7. web_search — look up information
# ============================================================
{
"instruction": "What's the latest version of FastAPI?",
"output": """<think>
I'll search the web for the latest FastAPI release information.
</think>

<tool_call>
{"name": "web_search", "args": {"query": "FastAPI latest version release 2025"}}
</tool_call>
<tool_result>
{"status": "success", "results": [{"title": "Release v0.115.0 - tiangolo/fastapi - GitHub", "url": "https://github.com/tiangolo/fastapi/releases", "snippet": "FastAPI 0.115.0 released with support for Pydantic v2.9, improved OpenAPI 3.1 generation, and performance improvements."}, {"title": "FastAPI - PyPI", "url": "https://pypi.org/project/fastapi/", "snippet": "Latest version: 0.115.0. Python web framework for building APIs with automatic validation and documentation."}]}
</tool_result>

The latest version of FastAPI is **0.115.0**. Key updates include:
- Support for Pydantic v2.9
- Improved OpenAPI 3.1 generation
- Performance improvements

You can install/upgrade with:
```bash
pip install fastapi==0.115.0
```"""
},
# ============================================================
# 8. Multi-tool: search + read + edit — refactor a function
# ============================================================
{
"instruction": "Find the function that sends email notifications and add error handling to it",
"output": """<think>
I need to find the email notification function, read it, then add proper error handling with try/except.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "def send.*email|def send.*notification|smtp|sendmail", "path": "src/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/services/notifications.py", "line": 12, "text": "def send_email(to, subject, body):"}, {"file": "src/services/notifications.py", "line": 15, "text": "    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)"}]}
</tool_result>

Found it in `src/services/notifications.py`. Let me read the full function:

<tool_call>
{"name": "read_file", "args": {"path": "src/services/notifications.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "import smtplib\\nfrom email.mime.text import MIMEText\\nimport os\\n\\nSMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')\\nSMTP_PORT = int(os.getenv('SMTP_PORT', '587'))\\nSMTP_USER = os.getenv('SMTP_USER')\\nSMTP_PASS = os.getenv('SMTP_PASS')\\nFROM_EMAIL = os.getenv('FROM_EMAIL', 'noreply@myapp.com')\\n\\ndef send_email(to, subject, body):\\n    msg = MIMEText(body)\\n    msg['Subject'] = subject\\n    msg['From'] = FROM_EMAIL\\n    msg['To'] = to\\n    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)\\n    server.starttls()\\n    server.login(SMTP_USER, SMTP_PASS)\\n    server.send_message(msg)\\n    server.quit()"}
</tool_result>

The function has no error handling — if SMTP fails, the whole request crashes. I'll add retry logic with proper exception handling:

<tool_call>
{"name": "edit_file", "args": {"path": "src/services/notifications.py", "old_text": "def send_email(to, subject, body):\n    msg = MIMEText(body)\n    msg['Subject'] = subject\n    msg['From'] = FROM_EMAIL\n    msg['To'] = to\n    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)\n    server.starttls()\n    server.login(SMTP_USER, SMTP_PASS)\n    server.send_message(msg)\n    server.quit()", "new_text": "def send_email(to, subject, body, retries=3):\n    msg = MIMEText(body)\n    msg['Subject'] = subject\n    msg['From'] = FROM_EMAIL\n    msg['To'] = to\n    \n    for attempt in range(1, retries + 1):\n        try:\n            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)\n            server.starttls()\n            server.login(SMTP_USER, SMTP_PASS)\n            server.send_message(msg)\n            server.quit()\n            return True\n        except smtplib.SMTPAuthenticationError:\n            raise  # Don't retry auth failures\n        except (smtplib.SMTPException, ConnectionError, TimeoutError) as e:\n            if attempt == retries:\n                raise RuntimeError(f'Failed to send email after {retries} attempts: {e}')\n            time.sleep(2 ** attempt)  # Exponential backoff\n    return False"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/services/notifications.py"}
</tool_result>

Added error handling with:
- **3 retries** with exponential backoff (2s, 4s, 8s)
- **10-second timeout** on SMTP connection
- **No retry on auth failures** (SMTPAuthenticationError) — those won't fix themselves
- **Retry on transient errors** (network issues, timeouts, SMTP errors)
- Returns `True` on success, raises `RuntimeError` after exhausting retries"""
},
# ============================================================
# 9. shell_command — system diagnostics
# ============================================================
{
"instruction": "Check how much disk space is available",
"output": """<think>
I'll use a shell command to check disk space usage.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "df -h --total 2>/dev/null || wmic logicaldisk get size,freespace,caption"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Filesystem      Size  Used Avail Use% Mounted on\\n/dev/sda1       500G  312G  188G  63% /\\n/dev/sda2       1.0T  456G  544G  46% /data\\ntmpfs           16G   12M   16G   1% /tmp\\ntotal           1.5T  768G  732G  52% -"}
</tool_result>

Disk space summary:

| Mount   | Size  | Used  | Free  | Usage |
|---------|-------|-------|-------|-------|
| `/`     | 500GB | 312GB | 188GB | 63%   |
| `/data` | 1.0TB | 456GB | 544GB | 46%   |
| **Total** | **1.5TB** | **768GB** | **732GB** | **52%** |

You have **732GB free** across all drives. The root partition is at 63% — healthy but worth monitoring. The `/data` partition has plenty of room."""
},
# ============================================================
# 10. execute_code + write_file — generate and save results
# ============================================================
{
"instruction": "Generate a CSV report of prime numbers under 100 and save it",
"output": """<think>
I'll write Python to find primes under 100, format as CSV, then save the file.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            return False\\n    return True\\n\\nprimes = [n for n in range(2, 100) if is_prime(n)]\\nprint(f'Found {len(primes)} primes under 100')\\nprint(f'First 5: {primes[:5]}')\\nprint(f'Last 5: {primes[-5:]}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Found 25 primes under 100\\nFirst 5: [2, 3, 5, 7, 11]\\nLast 5: [73, 79, 83, 89, 97]"}
</tool_result>

Found 25 primes. Now saving to CSV:

<tool_call>
{"name": "write_file", "args": {"path": "reports/primes_under_100.csv", "content": "index,prime,is_twin_prime\n1,2,false\n2,3,true\n3,5,true\n4,7,true\n5,11,true\n6,13,true\n7,17,true\n8,19,true\n9,23,false\n10,29,true\n11,31,true\n12,37,false\n13,41,true\n14,43,true\n15,47,false\n16,53,false\n17,59,true\n18,61,true\n19,67,false\n20,71,true\n21,73,true\n22,79,false\n23,83,false\n24,89,false\n25,97,false"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: reports/primes_under_100.csv (26 lines)"}
</tool_result>

Saved `reports/primes_under_100.csv` with 25 primes. The CSV includes:
- **index**: 1-25
- **prime**: the prime number
- **is_twin_prime**: whether it's part of a twin prime pair (differ by 2)

There are 8 twin prime pairs under 100: (3,5), (5,7), (11,13), (17,19), (29,31), (41,43), (59,61), (71,73)."""
},
# ============================================================
# 11. search_files — find TODO comments
# ============================================================
{
"instruction": "Find all TODO and FIXME comments in the codebase",
"output": """<think>
I'll search for TODO and FIXME patterns across all source files.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "TODO|FIXME|HACK|XXX", "path": "."}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/api/routes.py", "line": 45, "text": "# TODO: add rate limiting to this endpoint"}, {"file": "src/api/routes.py", "line": 89, "text": "# FIXME: this query is N+1, need to add eager loading"}, {"file": "src/services/auth.py", "line": 23, "text": "# TODO: implement token refresh rotation"}, {"file": "src/services/auth.py", "line": 67, "text": "# HACK: hardcoded expiry, should come from config"}, {"file": "src/models/user.py", "line": 31, "text": "# TODO: add email validation"}, {"file": "tests/test_api.py", "line": 12, "text": "# TODO: add test for unauthorized access"}]}
</tool_result>

Found **6 items** across 4 files:

**FIXME (critical):**
- `src/api/routes.py:89` — N+1 query needs eager loading

**HACK:**
- `src/services/auth.py:67` — Hardcoded token expiry should use config

**TODO (features):**
- `src/api/routes.py:45` — Add rate limiting
- `src/services/auth.py:23` — Token refresh rotation
- `src/models/user.py:31` — Email validation
- `tests/test_api.py:12` — Unauthorized access test

I'd prioritize the **FIXME** (N+1 query is a performance bug) and the **HACK** (hardcoded expiry is a maintenance risk) first."""
},
# ============================================================
# 12. execute_code — data analysis
# ============================================================
{
"instruction": "Parse this JSON and tell me the average response time: [{\"endpoint\": \"/api/users\", \"time_ms\": 45}, {\"endpoint\": \"/api/posts\", \"time_ms\": 120}, {\"endpoint\": \"/api/auth\", \"time_ms\": 230}, {\"endpoint\": \"/api/search\", \"time_ms\": 890}, {\"endpoint\": \"/api/health\", \"time_ms\": 5}]",
"output": """<think>
I'll parse the JSON data and compute statistics on the response times.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import json\\nimport statistics\\n\\ndata = [\\n    {'endpoint': '/api/users', 'time_ms': 45},\\n    {'endpoint': '/api/posts', 'time_ms': 120},\\n    {'endpoint': '/api/auth', 'time_ms': 230},\\n    {'endpoint': '/api/search', 'time_ms': 890},\\n    {'endpoint': '/api/health', 'time_ms': 5}\\n]\\n\\ntimes = [d['time_ms'] for d in data]\\nprint(f'Average: {statistics.mean(times):.1f}ms')\\nprint(f'Median:  {statistics.median(times):.1f}ms')\\nprint(f'Std Dev: {statistics.stdev(times):.1f}ms')\\nprint(f'Min:     {min(times)}ms ({[d[\"endpoint\"] for d in data if d[\"time_ms\"]==min(times)][0]})')\\nprint(f'Max:     {max(times)}ms ({[d[\"endpoint\"] for d in data if d[\"time_ms\"]==max(times)][0]})')\\nprint(f'P95:     {sorted(times)[int(len(times)*0.95)]}ms')\\nprint()\\nfor d in sorted(data, key=lambda x: x['time_ms'], reverse=True):\\n    bar = '#' * (d['time_ms'] // 20)\\n    print(f\"{d['endpoint']:15s} {d['time_ms']:4d}ms {bar}\")"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Average: 258.0ms\\nMedian:  120.0ms\\nStd Dev: 348.1ms\\nMin:     5ms (/api/health)\\nMax:     890ms (/api/search)\\nP95:     890ms\\n\\n/api/search     890ms #############################################\\n/api/auth       230ms ###########\\n/api/posts      120ms ######\\n/api/users       45ms ##\\n/api/health       5ms "}
</tool_result>

**Response time analysis:**

| Metric | Value |
|--------|-------|
| Average | 258.0ms |
| Median | 120.0ms |
| Std Dev | 348.1ms |
| Min | 5ms (`/api/health`) |
| Max | 890ms (`/api/search`) |

The **average is misleading** here — the median (120ms) is a better representation. `/api/search` at 890ms is a clear outlier pulling the average up. That endpoint likely needs optimization (adding an index, caching, or pagination)."""
},
# ============================================================
# 13. shell_command + read_file — check running processes
# ============================================================
{
"instruction": "Check if the app server is running and show me its logs",
"output": """<think>
I'll check for running processes matching the app, then read the log file.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "ps aux | grep -E 'uvicorn|gunicorn|python.*main' | grep -v grep"}}
</tool_call>
<tool_result>
{"status": "success", "output": "user  12345  0.5  2.1  256MB  python -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4"}
</tool_result>

The server is running: **uvicorn** with 4 workers on port 8080, using 256MB RAM. Let me check recent logs:

<tool_call>
{"name": "shell_command", "args": {"command": "tail -20 logs/app.log"}}
</tool_call>
<tool_result>
{"status": "success", "output": "2025-01-15 14:23:01 INFO     Server started on 0.0.0.0:8080\\n2025-01-15 14:23:15 INFO     GET /api/health 200 5ms\\n2025-01-15 14:24:02 INFO     POST /api/auth/login 200 230ms\\n2025-01-15 14:24:45 WARNING  Rate limit approaching for IP 192.168.1.50\\n2025-01-15 14:25:12 ERROR    GET /api/search?q=test 500 1200ms - TimeoutError: database query exceeded 1000ms\\n2025-01-15 14:25:13 INFO     GET /api/health 200 3ms\\n2025-01-15 14:26:01 INFO     GET /api/users 200 45ms"}
</tool_result>

**Server status:** Running (PID 12345, 4 workers, port 8080)

**Recent log highlights:**
- `14:24:45` **WARNING** — Rate limit approaching for `192.168.1.50`
- `14:25:12` **ERROR** — `/api/search` returned 500 after 1200ms due to a database query timeout
- All other endpoints responding normally

The `/api/search` timeout error needs attention — the database query is exceeding the 1000ms limit. This likely needs a query index or pagination."""
},
# ============================================================
# 14. write_file — create a Python script
# ============================================================
{
"instruction": "Create a Python script that monitors a directory for new files and logs them",
"output": """<think>
I'll create a file watcher script using the watchdog library, with fallback to polling if watchdog isn't installed.
</think>

<tool_call>
{"name": "write_file", "args": {"path": "scripts/watch_directory.py", "content": "#!/usr/bin/env python3\\n\\\"\\\"\\\"Watch a directory for new files and log them.\\\"\\\"\\\"\\nimport os\\nimport sys\\nimport time\\nimport logging\\nfrom pathlib import Path\\nfrom datetime import datetime\\n\\nlogging.basicConfig(\\n    level=logging.INFO,\\n    format='%(asctime)s [%(levelname)s] %(message)s',\\n    handlers=[\\n        logging.StreamHandler(),\\n        logging.FileHandler('file_watcher.log')\\n    ]\\n)\\nlogger = logging.getLogger(__name__)\\n\\ndef watch_directory(path, interval=2.0):\\n    \\\"\\\"\\\"Poll directory for new files.\\\"\\\"\\\"\\n    watch_path = Path(path)\\n    if not watch_path.is_dir():\\n        logger.error(f'Directory not found: {path}')\\n        sys.exit(1)\\n\\n    known_files = set(watch_path.rglob('*'))\\n    logger.info(f'Watching {path} ({len(known_files)} existing files)')\\n\\n    try:\\n        while True:\\n            current_files = set(watch_path.rglob('*'))\\n            new_files = current_files - known_files\\n            removed_files = known_files - current_files\\n\\n            for f in new_files:\\n                size = f.stat().st_size if f.is_file() else 0\\n                logger.info(f'NEW: {f} ({size:,} bytes)')\\n\\n            for f in removed_files:\\n                logger.info(f'REMOVED: {f}')\\n\\n            known_files = current_files\\n            time.sleep(interval)\\n    except KeyboardInterrupt:\\n        logger.info('Watcher stopped')\\n\\nif __name__ == '__main__':\\n    target = sys.argv[1] if len(sys.argv) > 1 else '.'\\n    watch_directory(target)"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: scripts/watch_directory.py (44 lines)"}
</tool_result>

Created `scripts/watch_directory.py`. Usage:

```bash
python scripts/watch_directory.py /path/to/watch
```

Features:
- Polls directory every 2 seconds for new or removed files
- Logs to both console and `file_watcher.log`
- Shows file sizes for new files
- Recursive watching (all subdirectories)
- Clean shutdown with Ctrl+C"""
},
# ============================================================
# 15. execute_code + error handling — code fails and is fixed
# ============================================================
{
"instruction": "Run this code and fix any errors: import json; data = json.loads('{name: \"test\"}')",
"output": """<think>
The user's code has invalid JSON — keys must be quoted in JSON. Let me run it to show the error, then fix it.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import json\\ndata = json.loads('{name: \\\"test\\\"}')\\nprint(data)"}}
</tool_call>
<tool_result>
{"status": "error", "error": "json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}
</tool_result>

The error is clear: JSON requires double-quoted keys. `{name: "test"}` is JavaScript syntax, not valid JSON. The correct JSON is `{"name": "test"}`.

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import json\\ndata = json.loads('{\\\"name\\\": \\\"test\\\"}')\\nprint(data)\\nprint(f'Name: {data[\"name\"]}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "{'name': 'test'}\\nName: test"}
</tool_result>

Fixed. The issue was unquoted keys in the JSON string. JSON requires `{"name": "test"}`, not `{name: "test"}`. The corrected code parses successfully."""
},
# ============================================================
# 16. Multi-tool: read + edit + execute — fix and test
# ============================================================
{
"instruction": "The calculate_discount function in src/pricing.py is returning wrong values. Fix it and test it.",
"output": """<think>
Let me read the function first to understand what's wrong, fix it, then test it.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/pricing.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "def calculate_discount(price, discount_percent):\\n    discount = price * discount_percent\\n    return price - discount\\n\\ndef apply_coupon(price, coupon_code):\\n    coupons = {'SAVE10': 10, 'SAVE20': 20, 'HALF': 50}\\n    if coupon_code in coupons:\\n        return calculate_discount(price, coupons[coupon_code])\\n    return price"}
</tool_result>

Found it. `calculate_discount` treats `discount_percent` as a decimal multiplier, but `apply_coupon` passes whole numbers (10, 20, 50). So `calculate_discount(100, 10)` computes `100 * 10 = 1000` discount, returning `-900` instead of `90`.

<tool_call>
{"name": "edit_file", "args": {"path": "src/pricing.py", "old_text": "def calculate_discount(price, discount_percent):\n    discount = price * discount_percent\n    return price - discount", "new_text": "def calculate_discount(price, discount_percent):\n    if not 0 <= discount_percent <= 100:\n        raise ValueError(f'Discount must be 0-100, got {discount_percent}')\n    discount = price * (discount_percent / 100)\n    return round(price - discount, 2)"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/pricing.py"}
</tool_result>

Now let me verify the fix:

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "def calculate_discount(price, discount_percent):\\n    if not 0 <= discount_percent <= 100:\\n        raise ValueError(f'Discount must be 0-100, got {discount_percent}')\\n    discount = price * (discount_percent / 100)\\n    return round(price - discount, 2)\\n\\n# Test cases\\nassert calculate_discount(100, 10) == 90.0, f'Got {calculate_discount(100, 10)}'\\nassert calculate_discount(100, 20) == 80.0\\nassert calculate_discount(100, 50) == 50.0\\nassert calculate_discount(49.99, 15) == 42.49\\nassert calculate_discount(100, 0) == 100.0\\nassert calculate_discount(100, 100) == 0.0\\n\\ntry:\\n    calculate_discount(100, 150)\\n    assert False, 'Should have raised ValueError'\\nexcept ValueError:\\n    pass\\n\\nprint('All tests passed!')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "All tests passed!"}
</tool_result>

Fixed and verified. Changes:
1. **Divided by 100** — `discount_percent / 100` converts percentage to decimal
2. **Added validation** — rejects values outside 0-100
3. **Rounded to 2 decimals** — prevents floating point artifacts like `42.491500000000004`"""
},
# ============================================================
# 17. shell_command — install dependencies
# ============================================================
{
"instruction": "Install the required packages from requirements.txt",
"output": """<think>
I'll install the packages from requirements.txt using pip.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "pip install -r requirements.txt"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Collecting fastapi==0.115.0\\n  Using cached fastapi-0.115.0-py3-none-any.whl\\nCollecting uvicorn==0.30.0\\n  Using cached uvicorn-0.30.0-py3-none-any.whl\\nCollecting sqlalchemy==2.0.35\\n  Downloading sqlalchemy-2.0.35-cp312-cp312-linux_x86_64.whl (3.2 MB)\\nCollecting pydantic==2.9.0\\n  Using cached pydantic-2.9.0-py3-none-any.whl\\nCollecting pytest==8.3.0\\n  Using cached pytest-8.3.0-py3-none-any.whl\\nSuccessfully installed fastapi-0.115.0 uvicorn-0.30.0 sqlalchemy-2.0.35 pydantic-2.9.0 pytest-8.3.0"}
</tool_result>

All packages installed successfully:
- **fastapi** 0.115.0
- **uvicorn** 0.30.0
- **sqlalchemy** 2.0.35
- **pydantic** 2.9.0
- **pytest** 8.3.0"""
},
# ============================================================
# 18. search + edit — rename a variable across files
# ============================================================
{
"instruction": "Rename the variable `usr` to `current_user` everywhere in the auth module",
"output": """<think>
I need to find all occurrences of the variable `usr` in the auth module, then update each one. I should be careful to only rename the variable, not substrings in other words.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "\\busr\\b", "path": "src/auth/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/auth/middleware.py", "line": 15, "text": "    usr = get_user_from_token(token)"}, {"file": "src/auth/middleware.py", "line": 16, "text": "    if not usr:"}, {"file": "src/auth/middleware.py", "line": 18, "text": "    request.state.user = usr"}, {"file": "src/auth/routes.py", "line": 32, "text": "    usr = authenticate(email, password)"}, {"file": "src/auth/routes.py", "line": 33, "text": "    if not usr:"}, {"file": "src/auth/routes.py", "line": 35, "text": "    token = create_token(usr.id)"}]}
</tool_result>

Found 6 occurrences across 2 files. Renaming:

<tool_call>
{"name": "edit_file", "args": {"path": "src/auth/middleware.py", "old_text": "    usr = get_user_from_token(token)\n    if not usr:\n", "new_text": "    current_user = get_user_from_token(token)\n    if not current_user:\n"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/auth/middleware.py"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/auth/middleware.py", "old_text": "    request.state.user = usr", "new_text": "    request.state.user = current_user"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/auth/middleware.py"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/auth/routes.py", "old_text": "    usr = authenticate(email, password)\n    if not usr:\n", "new_text": "    current_user = authenticate(email, password)\n    if not current_user:\n"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/auth/routes.py"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/auth/routes.py", "old_text": "    token = create_token(usr.id)", "new_text": "    token = create_token(current_user.id)"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/auth/routes.py"}
</tool_result>

Renamed all 6 occurrences of `usr` → `current_user` in:
- `src/auth/middleware.py` (3 occurrences)
- `src/auth/routes.py` (3 occurrences)

The new name is clearer and follows Python naming conventions."""
},
# ============================================================
# 19. execute_code — complex data processing
# ============================================================
{
"instruction": "Write a script to parse Apache access logs and show the top 10 most visited URLs",
"output": """<think>
I'll write a Python script to parse the standard Apache combined log format and aggregate URL visit counts.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import re\\nfrom collections import Counter\\n\\n# Simulated Apache log entries (combined format)\\nlogs = \\\"\\\"\\\"\\n192.168.1.1 - - [15/Jan/2025:10:00:01 +0000] \\\"GET / HTTP/1.1\\\" 200 5423\\n192.168.1.2 - - [15/Jan/2025:10:00:02 +0000] \\\"GET /api/users HTTP/1.1\\\" 200 1234\\n192.168.1.1 - - [15/Jan/2025:10:00:03 +0000] \\\"GET / HTTP/1.1\\\" 200 5423\\n192.168.1.3 - - [15/Jan/2025:10:00:04 +0000] \\\"POST /api/login HTTP/1.1\\\" 200 89\\n192.168.1.4 - - [15/Jan/2025:10:00:05 +0000] \\\"GET /api/users HTTP/1.1\\\" 200 1234\\n192.168.1.2 - - [15/Jan/2025:10:00:06 +0000] \\\"GET /static/style.css HTTP/1.1\\\" 200 8901\\n192.168.1.1 - - [15/Jan/2025:10:00:07 +0000] \\\"GET / HTTP/1.1\\\" 200 5423\\n192.168.1.5 - - [15/Jan/2025:10:00:08 +0000] \\\"GET /api/posts HTTP/1.1\\\" 200 4567\\n192.168.1.3 - - [15/Jan/2025:10:00:09 +0000] \\\"GET /api/users HTTP/1.1\\\" 200 1234\\n192.168.1.4 - - [15/Jan/2025:10:00:10 +0000] \\\"GET /about HTTP/1.1\\\" 200 3456\\n\\\"\\\"\\\".strip()\\n\\npattern = r'\\\"(GET|POST|PUT|DELETE) (.+?) HTTP'\\nurls = Counter()\\nfor match in re.finditer(pattern, logs):\\n    method, url = match.groups()\\n    urls[url] += 1\\n\\nprint(f'Total requests: {sum(urls.values())}')\\nprint(f'Unique URLs: {len(urls)}')\\nprint()\\nprint('Top URLs:')\\nfor url, count in urls.most_common(10):\\n    bar = '█' * count\\n    print(f'  {count:3d}  {bar:10s}  {url}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Total requests: 10\\nUnique URLs: 5\\n\\nTop URLs:\\n    3  ███        /\\n    3  ███        /api/users\\n    1  █          /api/login\\n    1  █          /static/style.css\\n    1  █          /api/posts\\n    1  █          /about"}
</tool_result>

Top URLs from the log:

| Rank | Hits | URL |
|------|------|-----|
| 1 | 3 | `/` |
| 2 | 3 | `/api/users` |
| 3 | 1 | `/api/login` |
| 4 | 1 | `/static/style.css` |
| 5 | 1 | `/api/posts` |

The homepage and `/api/users` are tied as the most visited. To run this on real logs, replace the simulated data with `open('/var/log/apache2/access.log').read()`."""
},
# ============================================================
# 20. read_file + write_file — create tests for existing code
# ============================================================
{
"instruction": "Read the User model and create unit tests for it",
"output": """<think>
I'll read the User model to understand its structure, then generate appropriate unit tests.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/models/user.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "from dataclasses import dataclass, field\\nfrom datetime import datetime\\nimport re\\nimport bcrypt\\n\\n@dataclass\\nclass User:\\n    id: int\\n    email: str\\n    name: str\\n    password_hash: str = ''\\n    created_at: datetime = field(default_factory=datetime.utcnow)\\n    is_active: bool = True\\n\\n    def set_password(self, password):\\n        if len(password) < 8:\\n            raise ValueError('Password must be at least 8 characters')\\n        self.password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()\\n\\n    def check_password(self, password):\\n        return bcrypt.checkpw(password.encode(), self.password_hash.encode())\\n\\n    def validate_email(self):\\n        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$'\\n        return bool(re.match(pattern, self.email))\\n\\n    def deactivate(self):\\n        self.is_active = False\\n\\n    def to_dict(self):\\n        return {\\n            'id': self.id,\\n            'email': self.email,\\n            'name': self.name,\\n            'is_active': self.is_active,\\n            'created_at': self.created_at.isoformat()\\n        }"}
</tool_result>

Good model with password hashing, email validation, and serialization. Let me create comprehensive tests:

<tool_call>
{"name": "write_file", "args": {"path": "tests/test_user.py", "content": "import pytest\\nfrom src.models.user import User\\n\\n\\nclass TestUserCreation:\\n    def test_create_user(self):\\n        user = User(id=1, email='test@example.com', name='Test User')\\n        assert user.id == 1\\n        assert user.email == 'test@example.com'\\n        assert user.name == 'Test User'\\n        assert user.is_active is True\\n\\n    def test_default_is_active(self):\\n        user = User(id=1, email='a@b.com', name='A')\\n        assert user.is_active is True\\n\\n\\nclass TestPassword:\\n    def test_set_and_check_password(self):\\n        user = User(id=1, email='a@b.com', name='A')\\n        user.set_password('securepass123')\\n        assert user.check_password('securepass123') is True\\n        assert user.check_password('wrongpass') is False\\n\\n    def test_short_password_raises(self):\\n        user = User(id=1, email='a@b.com', name='A')\\n        with pytest.raises(ValueError, match='at least 8'):\\n            user.set_password('short')\\n\\n    def test_password_hash_is_not_plaintext(self):\\n        user = User(id=1, email='a@b.com', name='A')\\n        user.set_password('mypassword123')\\n        assert user.password_hash != 'mypassword123'\\n\\n\\nclass TestEmailValidation:\\n    @pytest.mark.parametrize('email,expected', [\\n        ('user@example.com', True),\\n        ('user.name@domain.co', True),\\n        ('invalid-email', False),\\n        ('@no-user.com', False),\\n        ('user@', False),\\n    ])\\n    def test_validate_email(self, email, expected):\\n        user = User(id=1, email=email, name='A')\\n        assert user.validate_email() == expected\\n\\n\\nclass TestDeactivate:\\n    def test_deactivate_user(self):\\n        user = User(id=1, email='a@b.com', name='A')\\n        user.deactivate()\\n        assert user.is_active is False\\n\\n\\nclass TestSerialization:\\n    def test_to_dict_excludes_password(self):\\n        user = User(id=1, email='a@b.com', name='A')\\n        user.set_password('securepass123')\\n        d = user.to_dict()\\n        assert 'password_hash' not in d\\n        assert d['id'] == 1\\n        assert d['email'] == 'a@b.com'\\n        assert 'created_at' in d"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: tests/test_user.py (57 lines)"}
</tool_result>

Created `tests/test_user.py` with 9 tests covering:
- **Creation** — default values, required fields
- **Password** — set/check, short password rejection, hash not plaintext
- **Email validation** — valid and invalid formats using parametrize
- **Deactivation** — is_active flag
- **Serialization** — to_dict excludes password_hash

Run with: `pytest tests/test_user.py -v`"""
},
# ============================================================
# 21. shell_command — Docker operations
# ============================================================
{
"instruction": "Build and start the Docker containers for this project",
"output": """<think>
I'll build the Docker image and start the containers using docker-compose.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "docker-compose build --no-cache"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Building web\\nStep 1/8 : FROM python:3.12-slim\\n ---> a1b2c3d4e5f6\\nStep 2/8 : WORKDIR /app\\nStep 3/8 : COPY requirements.txt .\\nStep 4/8 : RUN pip install --no-cache-dir -r requirements.txt\\nStep 5/8 : COPY . .\\nStep 6/8 : EXPOSE 8080\\nStep 7/8 : CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]\\nSuccessfully built f6e5d4c3b2a1\\nSuccessfully tagged myapp-web:latest"}
</tool_result>

Build complete. Starting containers:

<tool_call>
{"name": "shell_command", "args": {"command": "docker-compose up -d"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Creating network \"myapp_default\" with the default driver\\nCreating myapp_db_1  ... done\\nCreating myapp_redis_1 ... done\\nCreating myapp_web_1   ... done"}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "docker-compose ps"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Name              Command                 State           Ports\\n----------------------------------------------------------------------\\nmyapp_db_1      docker-entrypoint.sh postgres   Up      5432/tcp\\nmyapp_redis_1   docker-entrypoint.sh redis       Up      6379/tcp\\nmyapp_web_1     uvicorn main:app --host 0...     Up      0.0.0.0:8080->8080/tcp"}
</tool_result>

All 3 containers are running:

| Container | Status | Port |
|-----------|--------|------|
| **web** | Up | 0.0.0.0:8080 |
| **db** (PostgreSQL) | Up | 5432 (internal) |
| **redis** | Up | 6379 (internal) |

The app is accessible at `http://localhost:8080`."""
},
# ============================================================
# 22. execute_code — debugging / profiling
# ============================================================
{
"instruction": "Profile this function and tell me where the bottleneck is: def process(data): sorted_data = sorted(data); unique = list(set(sorted_data)); result = [x**2 for x in unique]; return sum(result)",
"output": """<think>
I'll profile each step of the function to identify the bottleneck.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import time\\nimport random\\n\\ndef process(data):\\n    sorted_data = sorted(data)\\n    unique = list(set(sorted_data))\\n    result = [x**2 for x in unique]\\n    return sum(result)\\n\\n# Generate test data\\ndata = [random.randint(1, 10000) for _ in range(1_000_000)]\\n\\n# Profile each step\\nt0 = time.perf_counter()\\nsorted_data = sorted(data)\\nt1 = time.perf_counter()\\nunique = list(set(sorted_data))\\nt2 = time.perf_counter()\\nresult = [x**2 for x in unique]\\nt3 = time.perf_counter()\\ntotal = sum(result)\\nt4 = time.perf_counter()\\n\\nprint(f'Input size:  {len(data):>10,} items')\\nprint(f'Unique:      {len(unique):>10,} items')\\nprint()\\nprint(f'sorted()     {(t1-t0)*1000:8.1f}ms  {(t1-t0)/(t4-t0)*100:5.1f}%')\\nprint(f'set+list     {(t2-t1)*1000:8.1f}ms  {(t2-t1)/(t4-t0)*100:5.1f}%')\\nprint(f'x**2 list    {(t3-t2)*1000:8.1f}ms  {(t3-t2)/(t4-t0)*100:5.1f}%')\\nprint(f'sum()        {(t4-t3)*1000:8.1f}ms  {(t4-t3)/(t4-t0)*100:5.1f}%')\\nprint(f'TOTAL        {(t4-t0)*1000:8.1f}ms')\\nprint()\\nprint('Optimization: sort is unnecessary since set() removes duplication anyway')\\nt5 = time.perf_counter()\\nresult2 = sum(x**2 for x in set(data))\\nt6 = time.perf_counter()\\nprint(f'Optimized:   {(t6-t5)*1000:8.1f}ms (vs {(t4-t0)*1000:.1f}ms original)')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Input size:   1,000,000 items\\nUnique:           9,999 items\\n\\nsorted()       312.4ms   68.2%\\nset+list        89.7ms   19.6%\\nx**2 list        1.2ms    0.3%\\nsum()            0.1ms    0.0%\\nTOTAL          458.4ms\\n\\nOptimization: sort is unnecessary since set() removes duplication anyway\\nOptimized:      95.3ms (vs 458.4ms original)"}
</tool_result>

**Bottleneck: `sorted()` takes 68% of the time and is completely unnecessary.**

The `sorted()` call sorts 1M items in 312ms, but then `set()` immediately discards the ordering. The sort does nothing useful.

**Optimized version** (4.8x faster):
```python
def process(data):
    return sum(x**2 for x in set(data))
```

This goes from 458ms → 95ms by removing the useless sort and combining the list comprehension + sum into a generator expression."""
},
# ============================================================
# 23. web_search + execute_code — research and implement
# ============================================================
{
"instruction": "How do I implement exponential backoff for API retries? Show me working code.",
"output": """<think>
Let me search for best practices on exponential backoff, then implement it.
</think>

<tool_call>
{"name": "web_search", "args": {"query": "exponential backoff retry pattern best practices jitter"}}
</tool_call>
<tool_result>
{"status": "success", "results": [{"title": "Exponential Backoff And Jitter - AWS Architecture Blog", "url": "https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/", "snippet": "Full jitter provides the best results: sleep = random_between(0, min(cap, base * 2 ** attempt)). This decorrelates concurrent retry storms."}, {"title": "Retry pattern - Azure Architecture Center", "url": "https://learn.microsoft.com/en-us/azure/architecture/patterns/retry", "snippet": "Use exponential backoff for transient faults. Add jitter to prevent synchronized retries from multiple clients."}]}
</tool_result>

AWS recommends full jitter for best results. Let me implement it:

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import random\\nimport time\\n\\ndef retry_with_backoff(func, max_retries=5, base_delay=1.0, max_delay=60.0):\\n    \\\"\\\"\\\"Retry with exponential backoff and full jitter.\\\"\\\"\\\"\\n    for attempt in range(max_retries):\\n        try:\\n            return func()\\n        except Exception as e:\\n            if attempt == max_retries - 1:\\n                raise\\n            # Full jitter: random between 0 and exponential delay\\n            exp_delay = min(max_delay, base_delay * (2 ** attempt))\\n            sleep_time = random.uniform(0, exp_delay)\\n            print(f'  Attempt {attempt+1} failed: {e}. Retrying in {sleep_time:.2f}s')\\n            time.sleep(sleep_time)\\n\\n# Demo with a function that fails 3 times then succeeds\\ncall_count = 0\\ndef flaky_api_call():\\n    global call_count\\n    call_count += 1\\n    if call_count < 4:\\n        raise ConnectionError(f'Connection refused (attempt {call_count})')\\n    return {'status': 'ok', 'data': [1, 2, 3]}\\n\\nresult = retry_with_backoff(flaky_api_call)\\nprint(f'\\\\nSuccess after {call_count} attempts: {result}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "  Attempt 1 failed: Connection refused (attempt 1). Retrying in 0.73s\\n  Attempt 2 failed: Connection refused (attempt 2). Retrying in 1.42s\\n  Attempt 3 failed: Connection refused (attempt 3). Retrying in 2.89s\\n\\nSuccess after 4 attempts: {'status': 'ok', 'data': [1, 2, 3]}"}
</tool_result>

**Working exponential backoff with full jitter:**

```python
sleep = random.uniform(0, min(max_delay, base_delay * 2**attempt))
```

The key insight from AWS: **full jitter** (randomizing the entire delay window) prevents retry storms when many clients fail simultaneously. Without jitter, all clients retry at exactly 1s, 2s, 4s, etc., creating synchronized spikes."""
},
# ============================================================
# 24. read_file + edit_file — add type hints
# ============================================================
{
"instruction": "Add type hints to the functions in src/utils/helpers.py",
"output": """<think>
Let me read the file and add proper type annotations.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/utils/helpers.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "import hashlib\\nimport os\\nfrom datetime import datetime\\n\\ndef generate_id():\\n    return hashlib.sha256(os.urandom(32)).hexdigest()[:16]\\n\\ndef format_timestamp(dt):\\n    return dt.strftime('%Y-%m-%d %H:%M:%S')\\n\\ndef truncate(text, max_length, suffix='...'):\\n    if len(text) <= max_length:\\n        return text\\n    return text[:max_length - len(suffix)] + suffix\\n\\ndef chunk_list(lst, size):\\n    return [lst[i:i+size] for i in range(0, len(lst), size)]\\n\\ndef safe_get(dictionary, key, default=None):\\n    try:\\n        return dictionary[key]\\n    except (KeyError, TypeError):\\n        return default"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/utils/helpers.py", "old_text": "import hashlib\nimport os\nfrom datetime import datetime\n\ndef generate_id():\n    return hashlib.sha256(os.urandom(32)).hexdigest()[:16]\n\ndef format_timestamp(dt):\n    return dt.strftime('%Y-%m-%d %H:%M:%S')\n\ndef truncate(text, max_length, suffix='...'):\n    if len(text) <= max_length:\n        return text\n    return text[:max_length - len(suffix)] + suffix\n\ndef chunk_list(lst, size):\n    return [lst[i:i+size] for i in range(0, len(lst), size)]\n\ndef safe_get(dictionary, key, default=None):\n    try:\n        return dictionary[key]\n    except (KeyError, TypeError):\n        return default", "new_text": "import hashlib\nimport os\nfrom datetime import datetime\nfrom typing import Any, TypeVar\n\nT = TypeVar('T')\n\ndef generate_id() -> str:\n    return hashlib.sha256(os.urandom(32)).hexdigest()[:16]\n\ndef format_timestamp(dt: datetime) -> str:\n    return dt.strftime('%Y-%m-%d %H:%M:%S')\n\ndef truncate(text: str, max_length: int, suffix: str = '...') -> str:\n    if len(text) <= max_length:\n        return text\n    return text[:max_length - len(suffix)] + suffix\n\ndef chunk_list(lst: list[T], size: int) -> list[list[T]]:\n    return [lst[i:i+size] for i in range(0, len(lst), size)]\n\ndef safe_get(dictionary: dict[str, Any], key: str, default: Any = None) -> Any:\n    try:\n        return dictionary[key]\n    except (KeyError, TypeError):\n        return default"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/utils/helpers.py"}
</tool_result>

Added type hints to all 5 functions:
- `generate_id() -> str`
- `format_timestamp(dt: datetime) -> str`
- `truncate(text: str, max_length: int, suffix: str) -> str`
- `chunk_list(lst: list[T], size: int) -> list[list[T]]` — uses TypeVar for generic typing
- `safe_get(dictionary: dict[str, Any], key: str) -> Any`"""
},
# ============================================================
# 25. shell_command — run tests
# ============================================================
{
"instruction": "Run the test suite and show me the results",
"output": """<think>
I'll run pytest with verbose output to see all test results.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "python -m pytest tests/ -v --tb=short"}}
</tool_call>
<tool_result>
{"status": "success", "output": "============================= test session starts =============================\\ncollected 24 items\\n\\ntests/test_user.py::TestUserCreation::test_create_user PASSED          [  4%]\\ntests/test_user.py::TestUserCreation::test_default_is_active PASSED    [  8%]\\ntests/test_user.py::TestPassword::test_set_and_check PASSED            [ 12%]\\ntests/test_user.py::TestPassword::test_short_password PASSED            [ 16%]\\ntests/test_user.py::TestPassword::test_not_plaintext PASSED             [ 20%]\\ntests/test_user.py::TestEmailValidation::test_valid PASSED              [ 25%]\\ntests/test_user.py::TestEmailValidation::test_invalid PASSED            [ 29%]\\ntests/test_pricing.py::test_discount_10_percent PASSED                  [ 33%]\\ntests/test_pricing.py::test_discount_50_percent PASSED                  [ 37%]\\ntests/test_pricing.py::test_discount_zero PASSED                        [ 41%]\\ntests/test_api.py::test_health_check PASSED                             [ 45%]\\ntests/test_api.py::test_create_user PASSED                              [ 50%]\\ntests/test_api.py::test_login PASSED                                    [ 54%]\\ntests/test_api.py::test_unauthorized PASSED                             [ 58%]\\ntests/test_helpers.py::test_generate_id_length PASSED                   [ 62%]\\ntests/test_helpers.py::test_generate_id_unique PASSED                   [ 66%]\\ntests/test_helpers.py::test_truncate PASSED                             [ 70%]\\ntests/test_helpers.py::test_truncate_short PASSED                       [ 75%]\\ntests/test_helpers.py::test_chunk_list PASSED                           [ 79%]\\ntests/test_helpers.py::test_safe_get PASSED                             [ 83%]\\ntests/test_helpers.py::test_safe_get_missing PASSED                     [ 87%]\\ntests/test_helpers.py::test_format_timestamp PASSED                     [ 91%]\\ntests/test_pagination.py::test_basic_pagination PASSED                  [ 95%]\\ntests/test_pagination.py::test_last_page PASSED                         [100%]\\n\\n============================= 24 passed in 1.82s =============================="}
</tool_result>

**All 24 tests passed** in 1.82 seconds.

| Test File | Tests | Status |
|-----------|-------|--------|
| test_user.py | 7 | All passed |
| test_pricing.py | 3 | All passed |
| test_api.py | 4 | All passed |
| test_helpers.py | 7 | All passed |
| test_pagination.py | 3 | All passed |

Everything is green. No failures or warnings."""
},
# ============================================================
# 26. web_search — technology comparison
# ============================================================
{
"instruction": "What are the main differences between PostgreSQL and MySQL for a new project?",
"output": """<think>
I'll search for a current comparison to give accurate, up-to-date information.
</think>

<tool_call>
{"name": "web_search", "args": {"query": "PostgreSQL vs MySQL comparison 2025 features performance"}}
</tool_call>
<tool_result>
{"status": "success", "results": [{"title": "PostgreSQL vs MySQL: A Comprehensive Comparison (2025)", "url": "https://www.postgresqltutorial.com/postgresql-vs-mysql/", "snippet": "PostgreSQL excels in complex queries, JSON support, and extensibility. MySQL is faster for simple read-heavy workloads and has better replication out of the box."}, {"title": "Choosing Between PostgreSQL and MySQL - DB-Engines", "url": "https://db-engines.com/en/system/MySQL%3BPostgreSQL", "snippet": "PostgreSQL: ranked #4 overall, supports advanced types (arrays, JSONB, ranges, custom types). MySQL: ranked #2 overall, wider adoption, simpler setup."}]}
</tool_result>

**PostgreSQL vs MySQL for a new project:**

| Feature | PostgreSQL | MySQL |
|---------|-----------|-------|
| Complex queries | Excellent (CTEs, window functions, JSONB) | Basic (improving in 8.0+) |
| JSON support | Native JSONB with indexing | JSON type (no indexing until 8.0) |
| Extensibility | Custom types, extensions, PL/pgSQL | Limited |
| Simple reads | Good | Slightly faster |
| Replication | Logical + streaming | Built-in, simpler setup |
| Adoption | Growing fast (#4) | Wider (#2) |

**Recommendation:** PostgreSQL for most new projects — it handles more use cases without needing workarounds. MySQL only if you have a specific reason (existing expertise, specific hosting requirements, or extremely simple read-heavy workloads)."""
},
# ============================================================
# 27. Multi-tool: search + read + write — create migration
# ============================================================
{
"instruction": "Look at the User model and create a database migration to add a 'role' field",
"output": """<think>
I'll check the current User model to understand the schema, then create a migration.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/models/user.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "from sqlalchemy import Column, Integer, String, Boolean, DateTime\\nfrom sqlalchemy.sql import func\\nfrom src.config.database import Base\\n\\nclass User(Base):\\n    __tablename__ = 'users'\\n    \\n    id = Column(Integer, primary_key=True)\\n    email = Column(String(255), unique=True, nullable=False)\\n    name = Column(String(100), nullable=False)\\n    password_hash = Column(String(255), nullable=False)\\n    is_active = Column(Boolean, default=True)\\n    created_at = Column(DateTime, server_default=func.now())"}
</tool_result>

Good, SQLAlchemy model with 5 columns. I'll add the role field and create the migration:

<tool_call>
{"name": "edit_file", "args": {"path": "src/models/user.py", "old_text": "    is_active = Column(Boolean, default=True)\n    created_at = Column(DateTime, server_default=func.now())", "new_text": "    role = Column(String(20), nullable=False, default='user')\n    is_active = Column(Boolean, default=True)\n    created_at = Column(DateTime, server_default=func.now())"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/models/user.py"}
</tool_result>

<tool_call>
{"name": "write_file", "args": {"path": "migrations/versions/002_add_user_role.py", "content": "\"\"\"Add role column to users table.\n\nRevision: 002\nDate: 2025-01-15\n\"\"\"\nfrom alembic import op\nimport sqlalchemy as sa\n\nrevision = '002'\ndown_revision = '001'\n\ndef upgrade():\n    op.add_column('users', sa.Column('role', sa.String(20), nullable=False, server_default='user'))\n    # Set existing users to 'user' role\n    op.execute(\"UPDATE users SET role = 'user' WHERE role IS NULL\")\n\ndef downgrade():\n    op.drop_column('users', 'role')"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: migrations/versions/002_add_user_role.py (17 lines)"}
</tool_result>

Done. Created:
1. **Model update** — added `role = Column(String(20), nullable=False, default='user')` to User
2. **Migration** — `migrations/versions/002_add_user_role.py` with `server_default='user'` so existing rows get the default value

Run with: `alembic upgrade head`"""
},
# ============================================================
# 28. execute_code + error handling — runtime error
# ============================================================
{
"instruction": "Why is this code crashing? import requests; r = requests.get('https://api.example.com/data'); data = r.json()['results'][0]['value']",
"output": """<think>
The code has no error handling for network failures, missing keys, or empty results. Let me reproduce the failure and show the fix.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "# Simulating the failure modes\\nimport json\\n\\n# Scenario 1: API returns empty results\\nresponse_body = '{\"results\": []}'\\ndata = json.loads(response_body)\\ntry:\\n    value = data['results'][0]['value']\\nexcept IndexError as e:\\n    print(f'Scenario 1 - Empty results: {e}')\\n\\n# Scenario 2: API returns results without 'value' key\\nresponse_body = '{\"results\": [{\"name\": \"test\"}]}'\\ndata = json.loads(response_body)\\ntry:\\n    value = data['results'][0]['value']\\nexcept KeyError as e:\\n    print(f'Scenario 2 - Missing key: {e}')\\n\\n# Scenario 3: API returns unexpected structure\\nresponse_body = '{\"error\": \"unauthorized\"}'\\ndata = json.loads(response_body)\\ntry:\\n    value = data['results'][0]['value']\\nexcept KeyError as e:\\n    print(f'Scenario 3 - No results key: {e}')\\n\\nprint()\\nprint('--- Fixed version ---')\\n\\ndef get_value_safe(response_data):\\n    results = response_data.get('results', [])\\n    if not results:\\n        return None\\n    return results[0].get('value')\\n\\nprint(f'Empty results:  {get_value_safe({\"results\": []})}')\\nprint(f'Missing value:  {get_value_safe({\"results\": [{\"name\": \"x\"}]})}')\\nprint(f'No results key: {get_value_safe({\"error\": \"unauthorized\"})}')\\nprint(f'Valid data:     {get_value_safe({\"results\": [{\"value\": 42}]})}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Scenario 1 - Empty results: list index out of range\\nScenario 2 - Missing key: 'value'\\nScenario 3 - No results key: 'results'\\n\\n--- Fixed version ---\\nEmpty results:  None\\nMissing value:  None\\nNo results key: None\\nValid data:     42"}
</tool_result>

Your code crashes because `r.json()['results'][0]['value']` chains 3 assumptions:

1. **`['results']` exists** → crashes with `KeyError` if API returns error response
2. **`[0]` exists** → crashes with `IndexError` if results is empty
3. **`['value']` exists** → crashes with `KeyError` if the field is missing

**Fixed version:**
```python
import requests

r = requests.get('https://api.example.com/data')
r.raise_for_status()  # Raises on 4xx/5xx

data = r.json()
results = data.get('results', [])
value = results[0].get('value') if results else None
```"""
},
# ============================================================
# 29. shell_command — network diagnostics
# ============================================================
{
"instruction": "Check if port 5432 is in use and what process is using it",
"output": """<think>
I'll check for processes listening on port 5432 (PostgreSQL default port).
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "lsof -i :5432 2>/dev/null || netstat -tlnp 2>/dev/null | grep 5432 || ss -tlnp | grep 5432"}}
</tool_call>
<tool_result>
{"status": "success", "output": "COMMAND     PID     USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME\\npostgres  1823  postgres    5u  IPv4  25891      0t0  TCP localhost:5432 (LISTEN)\\npostgres  1823  postgres    6u  IPv6  25892      0t0  TCP [::1]:5432 (LISTEN)"}
</tool_result>

Port 5432 is in use by **PostgreSQL**:

| Field | Value |
|-------|-------|
| Process | `postgres` |
| PID | 1823 |
| User | postgres |
| Listening | localhost:5432 (IPv4 + IPv6) |

PostgreSQL is running and listening. If you need to use a different port for your app, either change your app's config or stop this instance with `sudo systemctl stop postgresql`."""
},
# ============================================================
# 30. Multi-tool: write + execute — create and run a test
# ============================================================
{
"instruction": "Write a unit test for the paginate function and run it",
"output": """<think>
I'll write a test file for the pagination function, then execute it.
</think>

<tool_call>
{"name": "write_file", "args": {"path": "tests/test_paginate.py", "content": "from src.utils.pagination import paginate\\n\\ndef test_basic_pagination():\\n    items = list(range(50))\\n    result = paginate(items, page=1, per_page=10)\\n    assert result['items'] == [0,1,2,3,4,5,6,7,8,9]\\n    assert result['page'] == 1\\n    assert result['total'] == 50\\n    assert result['total_pages'] == 5\\n\\ndef test_middle_page():\\n    items = list(range(50))\\n    result = paginate(items, page=3, per_page=10)\\n    assert result['items'] == [20,21,22,23,24,25,26,27,28,29]\\n\\ndef test_last_page_partial():\\n    items = list(range(25))\\n    result = paginate(items, page=2, per_page=20)\\n    assert result['items'] == [20,21,22,23,24]\\n    assert result['total_pages'] == 2\\n\\ndef test_empty_list():\\n    result = paginate([], page=1, per_page=10)\\n    assert result['items'] == []\\n    assert result['total'] == 0\\n    assert result['total_pages'] == 0\\n\\ndef test_exact_fit():\\n    items = list(range(20))\\n    result = paginate(items, page=1, per_page=20)\\n    assert len(result['items']) == 20\\n    assert result['total_pages'] == 1"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: tests/test_paginate.py (30 lines)"}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "python -m pytest tests/test_paginate.py -v"}}
</tool_call>
<tool_result>
{"status": "success", "output": "============================= test session starts =============================\\ncollected 5 items\\n\\ntests/test_paginate.py::test_basic_pagination PASSED                    [ 20%]\\ntests/test_paginate.py::test_middle_page PASSED                        [ 40%]\\ntests/test_paginate.py::test_last_page_partial PASSED                  [ 60%]\\ntests/test_paginate.py::test_empty_list PASSED                         [ 80%]\\ntests/test_paginate.py::test_exact_fit PASSED                          [100%]\\n\\n============================= 5 passed in 0.12s ==============================="}
</tool_result>

All 5 tests passed:
- **Basic pagination** — first page, correct items and metadata
- **Middle page** — page 3 of 5 returns correct slice
- **Last page partial** — 25 items with 20 per page → page 2 has 5 items
- **Empty list** — handles edge case gracefully
- **Exact fit** — 20 items at 20 per page = 1 page"""
},
# ============================================================
# 31. search_files — find security issues
# ============================================================
{
"instruction": "Search for potential security vulnerabilities in the codebase",
"output": """<think>
I'll search for common security anti-patterns: hardcoded secrets, SQL injection, eval(), unsafe deserialization, etc.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "password\\s*=\\s*['\"]|api_key\\s*=\\s*['\"]|secret\\s*=\\s*['\"]", "path": "src/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/config/settings.py", "line": 5, "text": "    JWT_SECRET = 'super-secret-key-change-me'"}, {"file": "src/tests/fixtures.py", "line": 8, "text": "    password = 'testpass123'"}]}
</tool_result>

<tool_call>
{"name": "search_files", "args": {"pattern": "eval\\(|exec\\(|pickle\\.loads|subprocess\\.call.*shell=True|f['\"].*SELECT.*{", "path": "src/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/services/report.py", "line": 34, "text": "    query = f\"SELECT * FROM reports WHERE user_id = {user_id}\""}, {"file": "src/utils/dynamic.py", "line": 12, "text": "    result = eval(expression)"}]}
</tool_result>

Found **4 security issues**:

**CRITICAL:**
1. **SQL Injection** — `src/services/report.py:34`
   ```python
   query = f"SELECT * FROM reports WHERE user_id = {user_id}"  # INJECTABLE!
   ```
   Fix: Use parameterized query: `"SELECT * FROM reports WHERE user_id = %s", (user_id,)`

2. **Code Injection** — `src/utils/dynamic.py:12`
   ```python
   result = eval(expression)  # Arbitrary code execution!
   ```
   Fix: Use `ast.literal_eval()` for safe evaluation, or remove entirely.

**HIGH:**
3. **Hardcoded secret** — `src/config/settings.py:5`
   ```python
   JWT_SECRET = 'super-secret-key-change-me'  # Hardcoded in source!
   ```
   Fix: Use `os.environ['JWT_SECRET']` with no fallback.

**LOW:**
4. **Test fixture password** — `src/tests/fixtures.py:8` — acceptable for tests only."""
},
# ============================================================
# 32. execute_code — algorithm implementation
# ============================================================
{
"instruction": "Implement binary search and prove it works",
"output": """<think>
I'll implement binary search with tests to prove correctness for various edge cases.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "def binary_search(arr, target):\\n    left, right = 0, len(arr) - 1\\n    while left <= right:\\n        mid = left + (right - left) // 2  # Avoid overflow\\n        if arr[mid] == target:\\n            return mid\\n        elif arr[mid] < target:\\n            left = mid + 1\\n        else:\\n            right = mid - 1\\n    return -1\\n\\n# Test cases\\ntests = [\\n    ([1,3,5,7,9,11,13,15], 7, 3),    # Middle element\\n    ([1,3,5,7,9,11,13,15], 1, 0),    # First element\\n    ([1,3,5,7,9,11,13,15], 15, 7),   # Last element\\n    ([1,3,5,7,9,11,13,15], 4, -1),   # Not found\\n    ([], 5, -1),                       # Empty array\\n    ([42], 42, 0),                     # Single element found\\n    ([42], 99, -1),                    # Single element not found\\n    (list(range(0, 1000000, 2)), 500000, 250000),  # Large array\\n]\\n\\nall_passed = True\\nfor arr, target, expected in tests:\\n    result = binary_search(arr, target)\\n    status = 'PASS' if result == expected else 'FAIL'\\n    if result != expected:\\n        all_passed = False\\n    desc = f'arr[{len(arr)}], target={target}'\\n    print(f'  {status}  {desc:40s} expected={expected}, got={result}')\\n\\nprint(f'\\\\n{\"All tests passed!\" if all_passed else \"SOME TESTS FAILED\"}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "  PASS  arr[8], target=7                          expected=3, got=3\\n  PASS  arr[8], target=1                          expected=0, got=0\\n  PASS  arr[8], target=15                         expected=7, got=7\\n  PASS  arr[8], target=4                          expected=-1, got=-1\\n  PASS  arr[0], target=5                          expected=-1, got=-1\\n  PASS  arr[1], target=42                         expected=0, got=0\\n  PASS  arr[1], target=99                         expected=-1, got=-1\\n  PASS  arr[500000], target=500000                expected=250000, got=250000\\n\\nAll tests passed!"}
</tool_result>

Binary search implementation — **all 8 test cases passed**, including edge cases (empty array, single element, large array with 500K elements).

Key implementation detail: `mid = left + (right - left) // 2` instead of `(left + right) // 2` to prevent integer overflow in languages with fixed-width integers."""
},
# ============================================================
# 33. read_file + edit_file — fix environment variable loading
# ============================================================
{
"instruction": "The app crashes on startup saying DB_PORT is not set. Fix the config loading.",
"output": """<think>
Let me check the config file to see how DB_PORT is being loaded.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/config.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "import os\\n\\nDB_HOST = os.environ['DB_HOST']\\nDB_PORT = int(os.environ['DB_PORT'])\\nDB_NAME = os.environ['DB_NAME']\\nDB_USER = os.environ['DB_USER']\\nDB_PASS = os.environ['DB_PASS']\\nDEBUG = os.environ['DEBUG']"}
</tool_result>

The problem is clear: all variables use `os.environ['KEY']` which raises `KeyError` if the variable isn't set. DB_PORT and other common settings should have defaults.

<tool_call>
{"name": "edit_file", "args": {"path": "src/config.py", "old_text": "import os\n\nDB_HOST = os.environ['DB_HOST']\nDB_PORT = int(os.environ['DB_PORT'])\nDB_NAME = os.environ['DB_NAME']\nDB_USER = os.environ['DB_USER']\nDB_PASS = os.environ['DB_PASS']\nDEBUG = os.environ['DEBUG']", "new_text": "import os\n\n# Required — crash on startup if missing (intentional)\nDB_USER = os.environ['DB_USER']\nDB_PASS = os.environ['DB_PASS']\n\n# Optional — sensible defaults\nDB_HOST = os.getenv('DB_HOST', 'localhost')\nDB_PORT = int(os.getenv('DB_PORT', '5432'))\nDB_NAME = os.getenv('DB_NAME', 'myapp')\nDEBUG = os.getenv('DEBUG', 'false').lower() == 'true'"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/config.py"}
</tool_result>

Fixed. Changes:
- **DB_HOST, DB_PORT, DB_NAME, DEBUG** — now use `os.getenv()` with sensible defaults (localhost, 5432, myapp, false)
- **DB_USER, DB_PASS** — kept as `os.environ[]` (should crash if missing — these are secrets with no valid default)
- **DEBUG** — properly parses string to boolean"""
},
# ============================================================
# 34. write_file — create Dockerfile
# ============================================================
{
"instruction": "Create a Dockerfile for this Python FastAPI application",
"output": """<think>
I'll create a multi-stage Dockerfile for a FastAPI app, optimized for size and security.
</think>

<tool_call>
{"name": "write_file", "args": {"path": "Dockerfile", "content": "FROM python:3.12-slim AS builder\\n\\nWORKDIR /app\\nCOPY requirements.txt .\\nRUN pip install --no-cache-dir --prefix=/install -r requirements.txt\\n\\nFROM python:3.12-slim\\n\\n# Create non-root user\\nRUN useradd --create-home appuser\\n\\nWORKDIR /app\\nCOPY --from=builder /install /usr/local\\nCOPY . .\\n\\n# Don't run as root\\nUSER appuser\\n\\nEXPOSE 8080\\n\\nHEALTHCHECK --interval=30s --timeout=3s \\\\\\n    CMD curl -f http://localhost:8080/health || exit 1\\n\\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: Dockerfile (20 lines)"}
</tool_result>

Created a production-ready Dockerfile with:
- **Multi-stage build** — installs deps in builder, copies only runtime to final image
- **`python:3.12-slim`** — smaller than full image (~150MB vs ~1GB)
- **Non-root user** — runs as `appuser` for security
- **Health check** — Docker monitors `/health` endpoint every 30s
- **No cache** — `--no-cache-dir` keeps image small

Build and run:
```bash
docker build -t myapp .
docker run -p 8080:8080 --env-file .env myapp
```"""
},
# ============================================================
# 35. Multi-tool: read + search + edit — fix an import error
# ============================================================
{
"instruction": "I'm getting 'ImportError: cannot import name UserService from services'. Fix it.",
"output": """<think>
I need to find where UserService is defined and where it's being imported from to identify the mismatch.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "class UserService|from.*import.*UserService", "path": "src/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/services/user_service.py", "line": 5, "text": "class UserService:"}, {"file": "src/api/routes.py", "line": 3, "text": "from src.services import UserService"}]}
</tool_result>

Found the mismatch. `UserService` is defined in `src/services/user_service.py`, but `routes.py` imports from `src.services` (the package), not `src.services.user_service` (the module). Let me check if `__init__.py` exports it:

<tool_call>
{"name": "read_file", "args": {"path": "src/services/__init__.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "# Services package"}
</tool_result>

The `__init__.py` doesn't re-export `UserService`. Two options: fix the import in routes.py, or add the export to `__init__.py`. I'll fix the import since it's more explicit:

<tool_call>
{"name": "edit_file", "args": {"path": "src/api/routes.py", "old_text": "from src.services import UserService", "new_text": "from src.services.user_service import UserService"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/api/routes.py"}
</tool_result>

Fixed. The import was `from src.services import UserService` but `UserService` lives in `src/services/user_service.py`, not in `src/services/__init__.py`.

Changed to: `from src.services.user_service import UserService`"""
},
# ============================================================
# 36. execute_code — regex helper
# ============================================================
{
"instruction": "Help me write a regex to validate email addresses and test it",
"output": """<think>
I'll write a practical email regex and test it against various valid and invalid email addresses.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import re\\n\\n# Practical email regex (not RFC 5322 complete, but handles 99% of real emails)\\nEMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$'\\n\\ntest_cases = [\\n    # (email, should_match)\\n    ('user@example.com', True),\\n    ('first.last@company.co.uk', True),\\n    ('user+tag@gmail.com', True),\\n    ('name@sub.domain.org', True),\\n    ('user123@test.io', True),\\n    ('a@b.cd', True),\\n    ('', False),\\n    ('@no-user.com', False),\\n    ('no-domain@', False),\\n    ('no-at-sign.com', False),\\n    ('spaces in@email.com', False),\\n    ('user@.com', False),\\n    ('user@com', False),\\n]\\n\\nall_passed = True\\nfor email, expected in test_cases:\\n    matches = bool(re.match(EMAIL_REGEX, email))\\n    status = 'PASS' if matches == expected else 'FAIL'\\n    if matches != expected:\\n        all_passed = False\\n    print(f'  {status}  {email:35s}  expected={expected}, got={matches}')\\n\\nprint(f'\\\\n{\"All tests passed!\" if all_passed else \"SOME TESTS FAILED\"}')\\nprint(f'\\\\nRegex: {EMAIL_REGEX}')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "  PASS  user@example.com                     expected=True, got=True\\n  PASS  first.last@company.co.uk              expected=True, got=True\\n  PASS  user+tag@gmail.com                    expected=True, got=True\\n  PASS  name@sub.domain.org                   expected=True, got=True\\n  PASS  user123@test.io                       expected=True, got=True\\n  PASS  a@b.cd                                expected=True, got=True\\n  PASS                                        expected=False, got=False\\n  PASS  @no-user.com                          expected=False, got=False\\n  PASS  no-domain@                            expected=False, got=False\\n  PASS  no-at-sign.com                        expected=False, got=False\\n  PASS  spaces in@email.com                   expected=False, got=False\\n  PASS  user@.com                             expected=False, got=False\\n  PASS  user@com                              expected=False, got=False\\n\\nAll tests passed!"}
</tool_result>

All 13 test cases passed. The regex:

```python
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
```

Breakdown:
- `^[a-zA-Z0-9._%+-]+` — local part (letters, digits, dots, underscores, plus, hyphens)
- `@` — literal at sign
- `[a-zA-Z0-9.-]+` — domain (letters, digits, dots, hyphens)
- `\.[a-zA-Z]{2,}$` — TLD must be at least 2 letters

Note: For production, consider using a library like `email-validator` instead of regex."""
},
# ============================================================
# 37. shell_command + write_file — set up environment
# ============================================================
{
"instruction": "Set up a Python virtual environment and install the project dependencies",
"output": """<think>
I'll create a virtual environment and install dependencies from requirements.txt.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "python -m venv venv"}}
</tool_call>
<tool_result>
{"status": "success", "output": ""}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "venv/bin/pip install -r requirements.txt"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Collecting fastapi==0.115.0\\n  Using cached fastapi-0.115.0-py3-none-any.whl\\nCollecting uvicorn==0.30.0\\nCollecting sqlalchemy==2.0.35\\nCollecting pytest==8.3.0\\nCollecting python-dotenv==1.0.0\\nSuccessfully installed fastapi-0.115.0 uvicorn-0.30.0 sqlalchemy-2.0.35 pytest-8.3.0 python-dotenv-1.0.0"}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "venv/bin/python --version && venv/bin/pip list --format=columns | head -20"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Python 3.12.1\\nPackage         Version\\n--------------- --------\\nfastapi         0.115.0\\npydantic        2.9.0\\npython-dotenv   1.0.0\\npytest          8.3.0\\nsqlalchemy      2.0.35\\nuvicorn         0.30.0"}
</tool_result>

Virtual environment set up and ready:

- **Python**: 3.12.1
- **Location**: `./venv/`
- **Packages installed**: fastapi, uvicorn, sqlalchemy, pytest, python-dotenv

Activate with:
```bash
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate      # Windows
```"""
},
# ============================================================
# 38. Multi-tool: complex debugging workflow
# ============================================================
{
"instruction": "The /api/orders endpoint is returning 500 errors. Debug it.",
"output": """<think>
I need to find the orders endpoint, check the logs for the error, read the code, and fix the issue.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "tail -5 logs/error.log"}}
</tool_call>
<tool_result>
{"status": "success", "output": "2025-01-15 15:30:12 ERROR GET /api/orders 500\\nTraceback (most recent call last):\\n  File \"src/api/orders.py\", line 23, in list_orders\\n    orders = db.query(Order).filter(Order.user_id == user.id).all()\\nAttributeError: 'NoneType' object has no attribute 'id'"}
</tool_result>

The error is clear: `user` is `None` when accessing `user.id`. Let me check the code:

<tool_call>
{"name": "read_file", "args": {"path": "src/api/orders.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "from fastapi import APIRouter, Depends\\nfrom src.config.database import get_db\\nfrom src.models.order import Order\\nfrom src.auth.middleware import get_current_user\\n\\nrouter = APIRouter()\\n\\n@router.get('/api/orders')\\ndef list_orders(db=Depends(get_db), user=Depends(get_current_user)):\\n    orders = db.query(Order).filter(Order.user_id == user.id).all()\\n    return {'orders': [o.to_dict() for o in orders]}"}
</tool_result>

<tool_call>
{"name": "read_file", "args": {"path": "src/auth/middleware.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "from fastapi import Request, Depends\\nfrom fastapi.security import HTTPBearer\\nfrom src.models.user import User\\nfrom src.config.database import get_db\\nimport jwt\\nimport os\\n\\nsecurity = HTTPBearer(auto_error=False)\\n\\ndef get_current_user(credentials=Depends(security), db=Depends(get_db)):\\n    if not credentials:\\n        return None\\n    try:\\n        payload = jwt.decode(credentials.credentials, os.environ['JWT_SECRET'], algorithms=['HS256'])\\n        return db.query(User).get(payload['sub'])\\n    except jwt.InvalidTokenError:\\n        return None"}
</tool_result>

Found the root cause. `get_current_user` returns `None` on failure (invalid token, missing header) instead of raising a 401. Then `list_orders` crashes accessing `None.id`.

<tool_call>
{"name": "edit_file", "args": {"path": "src/auth/middleware.py", "old_text": "def get_current_user(credentials=Depends(security), db=Depends(get_db)):\n    if not credentials:\n        return None\n    try:\n        payload = jwt.decode(credentials.credentials, os.environ['JWT_SECRET'], algorithms=['HS256'])\n        return db.query(User).get(payload['sub'])\n    except jwt.InvalidTokenError:\n        return None", "new_text": "def get_current_user(credentials=Depends(security), db=Depends(get_db)):\n    if not credentials:\n        raise HTTPException(status_code=401, detail='Authentication required')\n    try:\n        payload = jwt.decode(credentials.credentials, os.environ['JWT_SECRET'], algorithms=['HS256'])\n        user = db.query(User).get(payload['sub'])\n        if not user:\n            raise HTTPException(status_code=401, detail='User not found')\n        return user\n    except jwt.InvalidTokenError:\n        raise HTTPException(status_code=401, detail='Invalid token')"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/auth/middleware.py"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/auth/middleware.py", "old_text": "from fastapi import Request, Depends", "new_text": "from fastapi import Request, Depends, HTTPException"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/auth/middleware.py"}
</tool_result>

Fixed. Root cause: `get_current_user` returned `None` instead of raising HTTP 401, causing a downstream `AttributeError`.

Changes:
- Missing credentials → **401 "Authentication required"**
- Invalid/expired token → **401 "Invalid token"**
- Token valid but user deleted → **401 "User not found"**

The 500 error is now a proper 401 response."""
},
# ============================================================
# 39. execute_code — data transformation
# ============================================================
{
"instruction": "Convert this flat list of records into a nested tree structure by parent_id",
"output": """<think>
I'll implement a standard algorithm to build a tree from a flat list using parent_id references.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import json\\n\\n# Flat list with parent_id references\\nrecords = [\\n    {'id': 1, 'name': 'Electronics', 'parent_id': None},\\n    {'id': 2, 'name': 'Computers', 'parent_id': 1},\\n    {'id': 3, 'name': 'Laptops', 'parent_id': 2},\\n    {'id': 4, 'name': 'Desktops', 'parent_id': 2},\\n    {'id': 5, 'name': 'Phones', 'parent_id': 1},\\n    {'id': 6, 'name': 'iPhones', 'parent_id': 5},\\n    {'id': 7, 'name': 'Android', 'parent_id': 5},\\n    {'id': 8, 'name': 'Clothing', 'parent_id': None},\\n    {'id': 9, 'name': 'Shirts', 'parent_id': 8},\\n    {'id': 10, 'name': 'Pants', 'parent_id': 8},\\n]\\n\\ndef build_tree(records):\\n    # Index all nodes by id\\n    nodes = {r['id']: {**r, 'children': []} for r in records}\\n    roots = []\\n    \\n    for node in nodes.values():\\n        parent_id = node['parent_id']\\n        if parent_id is None:\\n            roots.append(node)\\n        else:\\n            nodes[parent_id]['children'].append(node)\\n    \\n    return roots\\n\\ntree = build_tree(records)\\nprint(json.dumps(tree, indent=2))"}}
</tool_call>
<tool_result>
{"status": "success", "output": "[\\n  {\\n    \"id\": 1,\\n    \"name\": \"Electronics\",\\n    \"parent_id\": null,\\n    \"children\": [\\n      {\\n        \"id\": 2,\\n        \"name\": \"Computers\",\\n        \"parent_id\": 1,\\n        \"children\": [\\n          {\"id\": 3, \"name\": \"Laptops\", \"parent_id\": 2, \"children\": []},\\n          {\"id\": 4, \"name\": \"Desktops\", \"parent_id\": 2, \"children\": []}\\n        ]\\n      },\\n      {\\n        \"id\": 5,\\n        \"name\": \"Phones\",\\n        \"parent_id\": 1,\\n        \"children\": [\\n          {\"id\": 6, \"name\": \"iPhones\", \"parent_id\": 5, \"children\": []},\\n          {\"id\": 7, \"name\": \"Android\", \"parent_id\": 5, \"children\": []}\\n        ]\\n      }\\n    ]\\n  },\\n  {\\n    \"id\": 8,\\n    \"name\": \"Clothing\",\\n    \"parent_id\": null,\\n    \"children\": [\\n      {\"id\": 9, \"name\": \"Shirts\", \"parent_id\": 8, \"children\": []},\\n      {\"id\": 10, \"name\": \"Pants\", \"parent_id\": 8, \"children\": []}\\n    ]\\n  }\\n]"}
</tool_result>

Converted flat list to tree in O(n) time:

```
Electronics
├── Computers
│   ├── Laptops
│   └── Desktops
└── Phones
    ├── iPhones
    └── Android
Clothing
├── Shirts
└── Pants
```

The algorithm: (1) index all nodes by ID, (2) single pass to attach each node to its parent's children list. O(n) time and space."""
},
# ============================================================
# 40. shell_command + edit_file — fix broken deployment
# ============================================================
{
"instruction": "The deploy failed because of a syntax error in the nginx config. Find and fix it.",
"output": """<think>
I'll check the nginx config syntax using nginx -t, then read and fix the file.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "nginx -t 2>&1"}}
</tool_call>
<tool_result>
{"status": "error", "error": "nginx: [emerg] unexpected \"}\" in /etc/nginx/sites-enabled/myapp:18\\nnginx: configuration file /etc/nginx/nginx.conf test failed"}
</tool_result>

Syntax error on line 18. Let me read the config:

<tool_call>
{"name": "read_file", "args": {"path": "nginx/myapp.conf"}}
</tool_call>
<tool_result>
{"status": "success", "content": "server {\\n    listen 80;\\n    server_name myapp.com;\\n\\n    location / {\\n        proxy_pass http://localhost:8080;\\n        proxy_set_header Host $host;\\n        proxy_set_header X-Real-IP $remote_addr;\\n    }\\n\\n    location /static {\\n        alias /var/www/myapp/static;\\n        expires 30d;\\n    }\\n\\n    location /api {\\n        proxy_pass http://localhost:8080\\n    }\\n}"}
</tool_result>

Found it — line 17 is missing a semicolon after the `proxy_pass` directive in the `/api` location block.

<tool_call>
{"name": "edit_file", "args": {"path": "nginx/myapp.conf", "old_text": "        proxy_pass http://localhost:8080\n    }", "new_text": "        proxy_pass http://localhost:8080;\n    }"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to nginx/myapp.conf"}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "nginx -t 2>&1"}}
</tool_call>
<tool_result>
{"status": "success", "output": "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok\\nnginx: configuration file /etc/nginx/nginx.conf test is successful"}
</tool_result>

Fixed. Missing semicolon on line 17 — `proxy_pass http://localhost:8080` → `proxy_pass http://localhost:8080;`. Config now passes `nginx -t` validation."""
},
# ============================================================
# 41. write_file + execute_code — create and test a utility
# ============================================================
{
"instruction": "Create a password generator utility that generates secure passwords",
"output": """<think>
I'll create a password generator with configurable options and test it.
</think>

<tool_call>
{"name": "write_file", "args": {"path": "src/utils/password_gen.py", "content": "import secrets\\nimport string\\n\\ndef generate_password(\\n    length: int = 16,\\n    uppercase: bool = True,\\n    lowercase: bool = True,\\n    digits: bool = True,\\n    symbols: bool = True,\\n) -> str:\\n    chars = ''\\n    required = []\\n    \\n    if uppercase:\\n        chars += string.ascii_uppercase\\n        required.append(secrets.choice(string.ascii_uppercase))\\n    if lowercase:\\n        chars += string.ascii_lowercase\\n        required.append(secrets.choice(string.ascii_lowercase))\\n    if digits:\\n        chars += string.digits\\n        required.append(secrets.choice(string.digits))\\n    if symbols:\\n        chars += '!@#$%^&*()-_=+'\\n        required.append(secrets.choice('!@#$%^&*()-_=+'))\\n    \\n    if not chars:\\n        raise ValueError('At least one character type must be enabled')\\n    if length < len(required):\\n        raise ValueError(f'Length must be at least {len(required)}')\\n    \\n    remaining = [secrets.choice(chars) for _ in range(length - len(required))]\\n    password = required + remaining\\n    \\n    # Shuffle to avoid predictable positions\\n    result = list(password)\\n    for i in range(len(result) - 1, 0, -1):\\n        j = secrets.randbelow(i + 1)\\n        result[i], result[j] = result[j], result[i]\\n    \\n    return ''.join(result)"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: src/utils/password_gen.py (36 lines)"}
</tool_result>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import secrets\\nimport string\\nimport re\\n\\ndef generate_password(length=16, uppercase=True, lowercase=True, digits=True, symbols=True):\\n    chars = ''\\n    required = []\\n    if uppercase:\\n        chars += string.ascii_uppercase\\n        required.append(secrets.choice(string.ascii_uppercase))\\n    if lowercase:\\n        chars += string.ascii_lowercase\\n        required.append(secrets.choice(string.ascii_lowercase))\\n    if digits:\\n        chars += string.digits\\n        required.append(secrets.choice(string.digits))\\n    if symbols:\\n        chars += '!@#$%^&*()-_=+'\\n        required.append(secrets.choice('!@#$%^&*()-_=+'))\\n    remaining = [secrets.choice(chars) for _ in range(length - len(required))]\\n    password = required + remaining\\n    result = list(password)\\n    for i in range(len(result) - 1, 0, -1):\\n        j = secrets.randbelow(i + 1)\\n        result[i], result[j] = result[j], result[i]\\n    return ''.join(result)\\n\\n# Test\\nfor i in range(5):\\n    p = generate_password(20)\\n    has_upper = bool(re.search(r'[A-Z]', p))\\n    has_lower = bool(re.search(r'[a-z]', p))\\n    has_digit = bool(re.search(r'[0-9]', p))\\n    has_symbol = bool(re.search(r'[!@#$%^&*()\\-_=+]', p))\\n    print(f'{p}  len={len(p)}  U={has_upper} L={has_lower} D={has_digit} S={has_symbol}')\\n\\nprint()\\nprint('Digits only:', generate_password(8, uppercase=False, lowercase=False, digits=True, symbols=False))\\nprint('Letters only:', generate_password(12, digits=False, symbols=False))"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Kx#7mR&p2NqW+bF5yT$c  len=20  U=True L=True D=True S=True\\n9Fa!nX2h%GjR-kB7wM&q  len=20  U=True L=True D=True S=True\\nW+3dPm@8rYn$fK5bQ-xH  len=20  U=True L=True D=True S=True\\np7R#mN2cX&kF9bT!wJq+  len=20  U=True L=True D=True S=True\\nH5@xWn3kR%pY7mF-cBq&  len=20  U=True L=True D=True S=True\\n\\nDigits only: 49270358\\nLetters only: kRpNxWmFbJqH"}
</tool_result>

Created `src/utils/password_gen.py`. Features:
- Uses `secrets` module (cryptographically secure, not `random`)
- **Guarantees** at least one character from each enabled type
- Fisher-Yates shuffle to avoid predictable positions
- Configurable length and character types
- All 5 test passwords have the required character diversity"""
},
# ============================================================
# 42. read_file + execute_code — explain code
# ============================================================
{
"instruction": "Explain what this code does: src/utils/cache.py",
"output": """<think>
I'll read the file and analyze the code to explain its purpose and behavior.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/utils/cache.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "import time\\nfrom functools import wraps\\n\\ndef cached(ttl=300):\\n    def decorator(func):\\n        cache = {}\\n        \\n        @wraps(func)\\n        def wrapper(*args, **kwargs):\\n            key = (args, tuple(sorted(kwargs.items())))\\n            now = time.time()\\n            \\n            if key in cache:\\n                value, timestamp = cache[key]\\n                if now - timestamp < ttl:\\n                    return value\\n            \\n            result = func(*args, **kwargs)\\n            cache[key] = (result, now)\\n            return result\\n        \\n        wrapper.cache_clear = lambda: cache.clear()\\n        return wrapper\\n    return decorator"}
</tool_result>

This is a **TTL-based memoization decorator**. Here's how it works:

1. **`@cached(ttl=300)`** — decorates a function to cache its return values for 300 seconds (5 minutes)
2. **Cache key** — built from the function arguments (`args` + sorted `kwargs`), so the same inputs always hit the same cache entry
3. **TTL check** — on each call, checks if the cached value is still fresh. If expired, re-calls the function
4. **`cache_clear()`** — exposed method to manually clear the cache

**Usage example:**
```python
@cached(ttl=60)  # Cache for 60 seconds
def get_user(user_id):
    return db.query(User).get(user_id)  # Only hits DB once per minute
```

**Limitations:** This is an in-memory cache — doesn't survive restarts, no size limit (could grow unbounded), not thread-safe. For production, use Redis or `functools.lru_cache`."""
},
# ============================================================
# 43. shell_command — database operations
# ============================================================
{
"instruction": "Show me the tables in the PostgreSQL database and their row counts",
"output": """<think>
I'll query PostgreSQL for table information and row counts.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "psql -U postgres -d myapp -c \"SELECT schemaname, tablename, n_live_tup as row_count FROM pg_stat_user_tables ORDER BY n_live_tup DESC;\""}}
</tool_call>
<tool_result>
{"status": "success", "output": " schemaname |   tablename    | row_count \\n------------+----------------+-----------\\n public     | events         |    245891\\n public     | users          |     15234\\n public     | orders         |      8721\\n public     | order_items    |     34205\\n public     | products       |      1523\\n public     | categories     |        45\\n public     | sessions       |      3201\\n(7 rows)"}
</tool_result>

Database `myapp` has **7 tables**:

| Table | Rows | Notes |
|-------|------|-------|
| events | 245,891 | Largest — consider archiving old data |
| order_items | 34,205 | ~4 items per order average |
| users | 15,234 | |
| orders | 8,721 | |
| sessions | 3,201 | Active user sessions |
| products | 1,523 | |
| categories | 45 | |

The `events` table is by far the largest. If it's growing fast, consider partitioning by date or archiving old events."""
},
# ============================================================
# 44. Multi-tool: search + read + write — create API docs
# ============================================================
{
"instruction": "Generate API documentation from the route definitions",
"output": """<think>
I'll find all route definitions, read them, and generate documentation.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "@router\\.(get|post|put|delete|patch)\\(", "path": "src/api/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/api/users.py", "line": 10, "text": "@router.get('/api/users')"}, {"file": "src/api/users.py", "line": 18, "text": "@router.post('/api/users')"}, {"file": "src/api/users.py", "line": 30, "text": "@router.get('/api/users/{user_id}')"}, {"file": "src/api/users.py", "line": 38, "text": "@router.delete('/api/users/{user_id}')"}, {"file": "src/api/orders.py", "line": 8, "text": "@router.get('/api/orders')"}, {"file": "src/api/orders.py", "line": 15, "text": "@router.post('/api/orders')"}, {"file": "src/api/auth.py", "line": 10, "text": "@router.post('/api/auth/login')"}, {"file": "src/api/auth.py", "line": 22, "text": "@router.post('/api/auth/register')"}, {"file": "src/api/health.py", "line": 5, "text": "@router.get('/api/health')"}]}
</tool_result>

Found 9 endpoints. Let me generate the documentation:

<tool_call>
{"name": "write_file", "args": {"path": "docs/api.md", "content": "# API Documentation\n\n## Authentication\n\n### POST /api/auth/register\nCreate a new user account.\n- **Auth**: None\n- **Body**: `{\"email\": \"string\", \"name\": \"string\", \"password\": \"string\"}`\n- **Response**: `201 {\"id\": int, \"email\": \"string\", \"name\": \"string\"}`\n\n### POST /api/auth/login\nAuthenticate and receive a JWT token.\n- **Auth**: None\n- **Body**: `{\"email\": \"string\", \"password\": \"string\"}`\n- **Response**: `200 {\"access_token\": \"string\", \"token_type\": \"bearer\"}`\n\n## Users\n\n### GET /api/users\nList all users (paginated).\n- **Auth**: Required (Bearer token)\n- **Query**: `?page=1&per_page=20`\n- **Response**: `200 {\"data\": [...], \"total\": int, \"page\": int}`\n\n### GET /api/users/{user_id}\nGet a specific user by ID.\n- **Auth**: Required\n- **Response**: `200 {\"id\": int, \"email\": \"string\", \"name\": \"string\"}`\n\n### POST /api/users\nCreate a new user (admin only).\n- **Auth**: Required (admin role)\n- **Body**: `{\"email\": \"string\", \"name\": \"string\", \"password\": \"string\"}`\n- **Response**: `201 {\"id\": int, ...}`\n\n### DELETE /api/users/{user_id}\nDelete a user (admin only).\n- **Auth**: Required (admin role)\n- **Response**: `204 No Content`\n\n## Orders\n\n### GET /api/orders\nList orders for the authenticated user.\n- **Auth**: Required\n- **Response**: `200 {\"orders\": [...]}`\n\n### POST /api/orders\nCreate a new order.\n- **Auth**: Required\n- **Body**: `{\"items\": [{\"product_id\": int, \"quantity\": int}]}`\n- **Response**: `201 {\"order_id\": int, \"total\": float}`\n\n## Health\n\n### GET /api/health\nHealth check endpoint.\n- **Auth**: None\n- **Response**: `200 {\"status\": \"ok\"}`"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: docs/api.md (52 lines)"}
</tool_result>

Generated `docs/api.md` with documentation for all 9 endpoints:
- **Auth**: 2 endpoints (register, login)
- **Users**: 4 endpoints (CRUD)
- **Orders**: 2 endpoints (list, create)
- **Health**: 1 endpoint

Each entry includes method, auth requirements, request body, and response format."""
},
# ============================================================
# 45. execute_code — benchmark comparison
# ============================================================
{
"instruction": "Which is faster for checking membership: list, set, or dict?",
"output": """<think>
I'll benchmark membership testing across different collection types and sizes.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import time\\nimport random\\n\\ndef benchmark(name, collection, lookups, iterations=100):\\n    start = time.perf_counter()\\n    for _ in range(iterations):\\n        for item in lookups:\\n            _ = item in collection\\n    elapsed = (time.perf_counter() - start) / iterations * 1000\\n    return elapsed\\n\\nfor size in [100, 1000, 10000, 100000]:\\n    data = list(range(size))\\n    random.shuffle(data)\\n    lookups = [random.randint(0, size * 2) for _ in range(1000)]  # 50% hit rate\\n    \\n    lst = data\\n    st = set(data)\\n    dct = {x: True for x in data}\\n    \\n    t_list = benchmark('list', lst, lookups)\\n    t_set = benchmark('set', st, lookups)\\n    t_dict = benchmark('dict', dct, lookups)\\n    \\n    print(f'Size {size:>7,}:  list={t_list:8.3f}ms  set={t_set:8.3f}ms  dict={t_dict:8.3f}ms  list/set={t_list/t_set:6.0f}x slower')"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Size     100:  list=   0.852ms  set=   0.052ms  dict=   0.054ms  list/set=    16x slower\\nSize   1,000:  list=   7.234ms  set=   0.053ms  dict=   0.055ms  list/set=   136x slower\\nSize  10,000:  list=  72.451ms  set=   0.054ms  dict=   0.056ms  list/set=  1342x slower\\nSize 100,000:  list= 745.612ms  set=   0.055ms  dict=   0.057ms  list/set= 13557x slower"}
</tool_result>

**Results are dramatic:**

| Size | list | set | dict | List slowdown |
|------|------|-----|------|---------------|
| 100 | 0.85ms | 0.05ms | 0.05ms | 16x |
| 1,000 | 7.23ms | 0.05ms | 0.05ms | 136x |
| 10,000 | 72.5ms | 0.05ms | 0.05ms | 1,342x |
| 100,000 | 745ms | 0.05ms | 0.05ms | 13,557x |

**Key insight:** `set` and `dict` lookups are O(1) — constant time regardless of size. List lookup is O(n) — linearly slower with size. At 100K items, list is **13,557x slower**.

**Rule:** Always use `set` for membership testing if you're checking more than a few times."""
},
# ============================================================
# 46. Multi-tool: search + edit — update deprecated API calls
# ============================================================
{
"instruction": "Find all uses of the deprecated datetime.utcnow() and replace with datetime.now(UTC)",
"output": """<think>
I need to find all occurrences of the deprecated datetime.utcnow() and update them.
</think>

<tool_call>
{"name": "search_files", "args": {"pattern": "datetime\\.utcnow\\(\\)", "path": "src/"}}
</tool_call>
<tool_result>
{"status": "success", "matches": [{"file": "src/models/user.py", "line": 12, "text": "    created_at: datetime = field(default_factory=datetime.utcnow)"}, {"file": "src/services/auth.py", "line": 18, "text": "    exp = datetime.utcnow() + timedelta(hours=1)"}, {"file": "src/services/orders.py", "line": 25, "text": "    order.created_at = datetime.utcnow()"}]}
</tool_result>

Found 3 occurrences in 3 files. Updating each:

<tool_call>
{"name": "edit_file", "args": {"path": "src/models/user.py", "old_text": "    created_at: datetime = field(default_factory=datetime.utcnow)", "new_text": "    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/models/user.py"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/services/auth.py", "old_text": "    exp = datetime.utcnow() + timedelta(hours=1)", "new_text": "    exp = datetime.now(timezone.utc) + timedelta(hours=1)"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/services/auth.py"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/services/orders.py", "old_text": "    order.created_at = datetime.utcnow()", "new_text": "    order.created_at = datetime.now(timezone.utc)"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/services/orders.py"}
</tool_result>

Updated all 3 files:
- `src/models/user.py` — default factory now uses `datetime.now(timezone.utc)`
- `src/services/auth.py` — JWT expiry calculation
- `src/services/orders.py` — order creation timestamp

`datetime.utcnow()` was deprecated in Python 3.12 because it returns a naive datetime that doesn't carry timezone info. `datetime.now(timezone.utc)` returns a timezone-aware datetime, which is safer for serialization and comparison."""
},
# ============================================================
# 47. shell_command + execute_code — git analysis
# ============================================================
{
"instruction": "Show me the code churn — which files have been changed the most in the last month",
"output": """<think>
I'll use git log to analyze file change frequency over the last month.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "git log --since='1 month ago' --name-only --pretty=format: | sort | uniq -c | sort -rn | head -15"}}
</tool_call>
<tool_result>
{"status": "success", "output": "     23 src/api/routes.py\\n     18 src/services/auth.py\\n     15 src/models/user.py\\n     12 tests/test_api.py\\n     11 src/config/database.py\\n      9 src/services/orders.py\\n      8 requirements.txt\\n      7 src/utils/helpers.py\\n      5 docker-compose.yml\\n      4 src/api/middleware.py\\n      3 README.md\\n      3 .github/workflows/ci.yml\\n      2 Dockerfile\\n      2 nginx/myapp.conf\\n      1 .gitignore"}
</tool_result>

**Top 15 most-changed files (last 30 days):**

| Changes | File | Risk Level |
|---------|------|------------|
| 23 | `src/api/routes.py` | HIGH churn |
| 18 | `src/services/auth.py` | HIGH churn |
| 15 | `src/models/user.py` | HIGH churn |
| 12 | `tests/test_api.py` | Normal (follows code changes) |
| 11 | `src/config/database.py` | Concerning |
| 9 | `src/services/orders.py` | Moderate |
| 8 | `requirements.txt` | Normal |
| 7 | `src/utils/helpers.py` | Moderate |

**Insights:**
- `routes.py` (23 changes) is a hotspot — consider splitting into separate route files per resource
- `auth.py` (18 changes) is stabilizing but still volatile
- `database.py` (11 changes) in config is concerning — config should be stable"""
},
# ============================================================
# 48. read_file + edit_file — add logging
# ============================================================
{
"instruction": "Add proper logging to the order processing function in src/services/orders.py",
"output": """<think>
I'll read the current function and add structured logging at appropriate points.
</think>

<tool_call>
{"name": "read_file", "args": {"path": "src/services/orders.py"}}
</tool_call>
<tool_result>
{"status": "success", "content": "from src.models.order import Order\\nfrom src.models.product import Product\\nfrom src.config.database import get_db\\n\\ndef process_order(user_id, items):\\n    db = next(get_db())\\n    order = Order(user_id=user_id)\\n    total = 0\\n    \\n    for item in items:\\n        product = db.query(Product).get(item['product_id'])\\n        if product.stock < item['quantity']:\\n            raise ValueError(f'Insufficient stock for {product.name}')\\n        product.stock -= item['quantity']\\n        total += product.price * item['quantity']\\n    \\n    order.total = total\\n    db.add(order)\\n    db.commit()\\n    return order"}
</tool_result>

<tool_call>
{"name": "edit_file", "args": {"path": "src/services/orders.py", "old_text": "from src.models.order import Order\nfrom src.models.product import Product\nfrom src.config.database import get_db\n\ndef process_order(user_id, items):\n    db = next(get_db())\n    order = Order(user_id=user_id)\n    total = 0\n    \n    for item in items:\n        product = db.query(Product).get(item['product_id'])\n        if product.stock < item['quantity']:\n            raise ValueError(f'Insufficient stock for {product.name}')\n        product.stock -= item['quantity']\n        total += product.price * item['quantity']\n    \n    order.total = total\n    db.add(order)\n    db.commit()\n    return order", "new_text": "import logging\nimport time\nfrom src.models.order import Order\nfrom src.models.product import Product\nfrom src.config.database import get_db\n\nlogger = logging.getLogger(__name__)\n\ndef process_order(user_id, items):\n    start = time.perf_counter()\n    logger.info(f'Processing order for user={user_id}, items={len(items)}')\n    \n    db = next(get_db())\n    order = Order(user_id=user_id)\n    total = 0\n    \n    for item in items:\n        product = db.query(Product).get(item['product_id'])\n        if not product:\n            logger.error(f'Product not found: id={item[\"product_id\"]}')\n            raise ValueError(f'Product {item[\"product_id\"]} not found')\n        if product.stock < item['quantity']:\n            logger.warning(f'Insufficient stock: product={product.name}, available={product.stock}, requested={item[\"quantity\"]}')\n            raise ValueError(f'Insufficient stock for {product.name}')\n        product.stock -= item['quantity']\n        total += product.price * item['quantity']\n        logger.debug(f'Added item: product={product.name}, qty={item[\"quantity\"]}, subtotal={product.price * item[\"quantity\"]}')\n    \n    order.total = total\n    db.add(order)\n    db.commit()\n    \n    elapsed = (time.perf_counter() - start) * 1000\n    logger.info(f'Order completed: order_id={order.id}, total=${total:.2f}, items={len(items)}, duration={elapsed:.1f}ms')\n    return order"}}
</tool_call>
<tool_result>
{"status": "success", "message": "Edit applied to src/services/orders.py"}
</tool_result>

Added structured logging at 4 levels:
- **INFO** — order start and completion (with timing)
- **WARNING** — insufficient stock (not an error, but noteworthy)
- **ERROR** — product not found (data integrity issue)
- **DEBUG** — individual item details (verbose, for troubleshooting)

Also added a null check for `product` — the original code would crash with `AttributeError` if a product ID didn't exist."""
},
# ============================================================
# 49. execute_code — data validation
# ============================================================
{
"instruction": "Write a function to validate and sanitize user input for a registration form",
"output": """<think>
I'll create a validation function that checks email, password strength, name format, and returns clean data or errors.
</think>

<tool_call>
{"name": "execute_code", "args": {"language": "python", "code": "import re\\nfrom typing import Any\\n\\ndef validate_registration(data: dict[str, Any]) -> tuple[dict | None, list[str]]:\\n    errors = []\\n    clean = {}\\n    \\n    # Name validation\\n    name = data.get('name', '').strip()\\n    if not name:\\n        errors.append('Name is required')\\n    elif len(name) < 2:\\n        errors.append('Name must be at least 2 characters')\\n    elif len(name) > 100:\\n        errors.append('Name must be under 100 characters')\\n    else:\\n        clean['name'] = name\\n    \\n    # Email validation\\n    email = data.get('email', '').strip().lower()\\n    if not email:\\n        errors.append('Email is required')\\n    elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$', email):\\n        errors.append('Invalid email format')\\n    else:\\n        clean['email'] = email\\n    \\n    # Password validation\\n    password = data.get('password', '')\\n    if not password:\\n        errors.append('Password is required')\\n    else:\\n        if len(password) < 8:\\n            errors.append('Password must be at least 8 characters')\\n        if not re.search(r'[A-Z]', password):\\n            errors.append('Password must contain an uppercase letter')\\n        if not re.search(r'[a-z]', password):\\n            errors.append('Password must contain a lowercase letter')\\n        if not re.search(r'[0-9]', password):\\n            errors.append('Password must contain a digit')\\n        if not errors or 'Password' not in errors[-1]:\\n            clean['password'] = password\\n    \\n    return (clean if not errors else None), errors\\n\\n# Test cases\\ntests = [\\n    {'name': 'John Doe', 'email': 'john@example.com', 'password': 'Secure123'},\\n    {'name': '', 'email': 'bad-email', 'password': 'weak'},\\n    {'name': 'A', 'email': 'a@b.com', 'password': 'nouppercase1'},\\n    {'name': '  Jane  ', 'email': '  JANE@GMAIL.COM  ', 'password': 'Good1Pass'},\\n]\\n\\nfor data in tests:\\n    clean, errors = validate_registration(data)\\n    print(f'Input: {data}')\\n    if clean:\\n        print(f'  Valid: {clean}')\\n    else:\\n        print(f'  Errors: {errors}')\\n    print()"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Input: {'name': 'John Doe', 'email': 'john@example.com', 'password': 'Secure123'}\\n  Valid: {'name': 'John Doe', 'email': 'john@example.com', 'password': 'Secure123'}\\n\\nInput: {'name': '', 'email': 'bad-email', 'password': 'weak'}\\n  Errors: ['Name is required', 'Invalid email format', 'Password must be at least 8 characters', 'Password must contain an uppercase letter']\\n\\nInput: {'name': 'A', 'email': 'a@b.com', 'password': 'nouppercase1'}\\n  Errors: ['Name must be at least 2 characters', 'Password must contain an uppercase letter']\\n\\nInput: {'name': '  Jane  ', 'email': '  JANE@GMAIL.COM  ', 'password': 'Good1Pass'}\\n  Valid: {'name': 'Jane', 'email': 'jane@gmail.com', 'password': 'Good1Pass'}"}
</tool_result>

Validation function handles:
- **Name**: required, 2-100 chars, whitespace trimmed
- **Email**: required, regex validated, lowercased, trimmed
- **Password**: min 8 chars, requires uppercase + lowercase + digit

Returns `(clean_data, errors)` tuple — clean data is `None` if any validation fails. Notice how `"  Jane  "` gets trimmed and `"JANE@GMAIL.COM"` gets lowercased."""
},
# ============================================================
# 50. Multi-tool: comprehensive project setup
# ============================================================
{
"instruction": "Initialize a new Python project with proper structure, virtual environment, and git",
"output": """<think>
I'll create the project structure, set up git, create a virtual environment, and install initial dependencies.
</think>

<tool_call>
{"name": "shell_command", "args": {"command": "mkdir -p myproject/src myproject/tests myproject/docs"}}
</tool_call>
<tool_result>
{"status": "success", "output": ""}
</tool_result>

<tool_call>
{"name": "write_file", "args": {"path": "myproject/pyproject.toml", "content": "[build-system]\nrequires = [\"setuptools>=68.0\", \"wheel\"]\nbuild-backend = \"setuptools.backends._legacy:_Backend\"\n\n[project]\nname = \"myproject\"\nversion = \"0.1.0\"\nrequires-python = \">=3.11\"\ndependencies = [\n    \"fastapi>=0.115.0\",\n    \"uvicorn>=0.30.0\",\n    \"python-dotenv>=1.0.0\",\n]\n\n[project.optional-dependencies]\ndev = [\n    \"pytest>=8.0\",\n    \"ruff>=0.5.0\",\n]\n\n[tool.ruff]\nline-length = 100\n\n[tool.pytest.ini_options]\ntestpaths = [\"tests\"]"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: myproject/pyproject.toml (25 lines)"}
</tool_result>

<tool_call>
{"name": "write_file", "args": {"path": "myproject/.gitignore", "content": "__pycache__/\n*.py[cod]\nvenv/\n.env\n.vscode/\n.idea/\ndist/\nbuild/\n*.egg-info/\n.pytest_cache/\n.coverage\n.ruff_cache/"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: myproject/.gitignore (12 lines)"}
</tool_result>

<tool_call>
{"name": "write_file", "args": {"path": "myproject/src/__init__.py", "content": ""}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: myproject/src/__init__.py (0 lines)"}
</tool_result>

<tool_call>
{"name": "write_file", "args": {"path": "myproject/src/main.py", "content": "from fastapi import FastAPI\n\napp = FastAPI(title=\"MyProject\", version=\"0.1.0\")\n\n@app.get(\"/health\")\ndef health():\n    return {\"status\": \"ok\"}"}}
</tool_call>
<tool_result>
{"status": "success", "message": "File written: myproject/src/main.py (7 lines)"}
</tool_result>

<tool_call>
{"name": "shell_command", "args": {"command": "cd myproject && python -m venv venv && venv/bin/pip install -e '.[dev]' && git init && git add . && git commit -m 'init: project setup with FastAPI'"}}
</tool_call>
<tool_result>
{"status": "success", "output": "Collecting fastapi>=0.115.0\\nCollecting uvicorn>=0.30.0\\nCollecting python-dotenv>=1.0.0\\nCollecting pytest>=8.0\\nCollecting ruff>=0.5.0\\nSuccessfully installed fastapi uvicorn python-dotenv pytest ruff\\nInitialized empty Git repository in /home/user/myproject/.git/\\n[main (root-commit) a1b2c3d] init: project setup with FastAPI\\n 6 files changed, 44 insertions(+)"}
</tool_result>

Project initialized with:

```
myproject/
├── pyproject.toml    # Dependencies, build config, tool settings
├── .gitignore        # Python, IDE, env exclusions
├── src/
│   ├── __init__.py
│   └── main.py       # FastAPI app with /health endpoint
├── tests/
├── docs/
└── venv/             # Virtual environment (gitignored)
```

- **Dependencies**: FastAPI, uvicorn, python-dotenv
- **Dev tools**: pytest, ruff (linter)
- **Git**: initialized with first commit
- **Run**: `venv/bin/uvicorn src.main:app --reload`"""
},
]

# Write training data
with open("data/tool_use/train.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} training examples")

# ============================================================
# EVAL EXAMPLES (10)
# ============================================================
eval_examples = [
{
"instruction": "Read the README.md file and tell me what this project is about",
"output": "<think>\nI'll read the README to understand the project.\n</think>\n\n<tool_call>\n{\"name\": \"read_file\", \"args\": {\"path\": \"README.md\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"content\": \"# TaskFlow\\n\\nA lightweight task queue for Python applications.\\n\\n## Features\\n- Redis-backed job queue\\n- Retry with exponential backoff\\n- Priority queues\\n- Dead letter queue for failed jobs\\n- Dashboard for monitoring\\n\\n## Quick Start\\n```bash\\npip install taskflow\\ntaskflow worker start\\n```\"}\n</tool_result>\n\n**TaskFlow** is a lightweight Python task queue backed by Redis. Key features:\n- Job queuing with priority support\n- Automatic retry with exponential backoff\n- Dead letter queue for failed jobs\n- Monitoring dashboard\n\nIt's similar to Celery but aims to be simpler and more lightweight."
},
{
"instruction": "Create a simple HTTP server that serves static files from the public/ directory",
"output": "<think>\nI'll create a Python HTTP server using the standard library's http.server module.\n</think>\n\n<tool_call>\n{\"name\": \"write_file\", \"args\": {\"path\": \"server.py\", \"content\": \"#!/usr/bin/env python3\\nimport http.server\\nimport os\\nimport sys\\n\\nPORT = int(os.getenv('PORT', '3000'))\\nDIRECTORY = 'public'\\n\\nclass Handler(http.server.SimpleHTTPRequestHandler):\\n    def __init__(self, *args, **kwargs):\\n        super().__init__(*args, directory=DIRECTORY, **kwargs)\\n\\nif __name__ == '__main__':\\n    if not os.path.isdir(DIRECTORY):\\n        os.makedirs(DIRECTORY)\\n        print(f'Created {DIRECTORY}/ directory')\\n    with http.server.HTTPServer(('', PORT), Handler) as server:\\n        print(f'Serving {DIRECTORY}/ at http://localhost:{PORT}')\\n        try:\\n            server.serve_forever()\\n        except KeyboardInterrupt:\\n            print('\\\\nServer stopped')\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"message\": \"File written: server.py (18 lines)\"}\n</tool_result>\n\nCreated `server.py` — a static file server for the `public/` directory.\n\nRun with: `python server.py`\n\nIt serves on port 3000 by default (configurable via `PORT` env var). Creates the `public/` directory if it doesn't exist."
},
{
"instruction": "Find all Python files larger than 500 lines",
"output": "<think>\nI'll use a shell command to find Python files and count their lines.\n</think>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"find . -name '*.py' -exec wc -l {} + | sort -rn | awk '$1 > 500 {print}'\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"   823 ./src/services/analytics.py\\n   612 ./src/api/routes.py\\n   567 ./src/models/order.py\\n   534 ./tests/test_integration.py\"}\n</tool_result>\n\nFound **4 Python files** over 500 lines:\n\n| Lines | File | Suggestion |\n|-------|------|------------|\n| 823 | `src/services/analytics.py` | Consider splitting by report type |\n| 612 | `src/api/routes.py` | Split into per-resource route files |\n| 567 | `src/models/order.py` | Separate model from business logic |\n| 534 | `tests/test_integration.py` | Normal for integration tests |\n\nFiles over 500 lines are candidates for refactoring. The routes file especially should be split."
},
{
"instruction": "What Python version is this project using?",
"output": "<think>\nI'll check multiple sources to determine the Python version requirement.\n</think>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"python --version\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"Python 3.12.1\"}\n</tool_result>\n\n<tool_call>\n{\"name\": \"search_files\", \"args\": {\"pattern\": \"python_requires|python-version|FROM python\", \"path\": \".\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"matches\": [{\"file\": \"pyproject.toml\", \"line\": 8, \"text\": \"requires-python = \\\">=3.11\\\"\"}, {\"file\": \"Dockerfile\", \"line\": 1, \"text\": \"FROM python:3.12-slim\"}, {\"file\": \".github/workflows/ci.yml\", \"line\": 15, \"text\": \"        python-version: ['3.11', '3.12']\"}]}\n</tool_result>\n\nPython version info:\n- **System**: Python 3.12.1\n- **Minimum required**: 3.11+ (from `pyproject.toml`)\n- **Docker**: 3.12 (from `Dockerfile`)\n- **CI testing**: 3.11 and 3.12 (from GitHub Actions)"
},
{
"instruction": "Run the linter and fix any issues",
"output": "<think>\nI'll run ruff to check for linting issues, then auto-fix what it can.\n</think>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"ruff check src/ --output-format=concise\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"src/api/routes.py:5:1: F401 `os` imported but unused\\nsrc/api/routes.py:45:80: E501 Line too long (92 > 88 characters)\\nsrc/services/auth.py:12:5: E722 Do not use bare `except`\\nsrc/utils/helpers.py:8:1: F401 `sys` imported but unused\\nFound 4 errors.\"}\n</tool_result>\n\nFound 4 issues. Let me auto-fix what ruff can handle:\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"ruff check src/ --fix\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"Found 4 errors (3 fixed, 1 remaining).\\nsrc/services/auth.py:12:5: E722 Do not use bare `except`\"}\n</tool_result>\n\nRuff auto-fixed 3 of 4 issues:\n- Removed unused `os` import from routes.py\n- Shortened long line in routes.py\n- Removed unused `sys` import from helpers.py\n\nThe remaining issue needs a manual fix — bare `except:` in auth.py should specify the exception type. Let me fix it:\n\n<tool_call>\n{\"name\": \"read_file\", \"args\": {\"path\": \"src/services/auth.py\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"content\": \"import jwt\\nfrom datetime import datetime, timedelta\\n\\ndef verify_token(token):\\n    try:\\n        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\\n        return payload\\n    except:\\n        return None\"}\n</tool_result>\n\n<tool_call>\n{\"name\": \"edit_file\", \"args\": {\"path\": \"src/services/auth.py\", \"old_text\": \"    except:\\n        return None\", \"new_text\": \"    except (jwt.InvalidTokenError, jwt.ExpiredSignatureError):\\n        return None\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"message\": \"Edit applied to src/services/auth.py\"}\n</tool_result>\n\nAll 4 lint issues fixed. The bare `except` was catching everything including `KeyboardInterrupt` and `SystemExit` — now it only catches JWT-specific errors."
},
{
"instruction": "Compute the factorial of 100 without overflow",
"output": "<think>\nPython handles arbitrary precision integers natively, so factorial(100) works without overflow. Let me compute it.\n</think>\n\n<tool_call>\n{\"name\": \"execute_code\", \"args\": {\"language\": \"python\", \"code\": \"import math\\n\\nresult = math.factorial(100)\\nprint(f'100! = {result}')\\nprint(f'\\\\nDigits: {len(str(result))}')\\nprint(f'Trailing zeros: {len(str(result)) - len(str(result).rstrip(\\\"0\\\"))}')\\n\\n# Verify: trailing zeros = sum of floor(100/5^k)\\ntrailing = sum(100 // (5**k) for k in range(1, 10) if 5**k <= 100)\\nprint(f'Expected trailing zeros (Legendre formula): {trailing}')\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"100! = 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000\\n\\nDigits: 158\\nTrailing zeros: 24\\nExpected trailing zeros (Legendre formula): 24\"}\n</tool_result>\n\n**100! = 93326215443944...000000000000000000** (158 digits)\n\nPython handles this natively — no overflow because Python integers have arbitrary precision. The result has **24 trailing zeros**, which matches the Legendre formula (counting factors of 5)."
},
{
"instruction": "Check if there are any outdated dependencies",
"output": "<think>\nI'll use pip to check for outdated packages.\n</think>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"pip list --outdated --format=columns\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"Package     Version Latest Type\\n----------- ------- ------ -----\\nfastapi     0.110.0 0.115.0 wheel\\npydantic    2.6.0   2.9.0   wheel\\nsqlalchemy  2.0.25  2.0.35  wheel\\npytest      8.0.0   8.3.0   wheel\"}\n</tool_result>\n\n**4 outdated packages:**\n\n| Package | Current | Latest | Priority |\n|---------|---------|--------|----------|\n| fastapi | 0.110.0 | 0.115.0 | High — security + features |\n| pydantic | 2.6.0 | 2.9.0 | High — validation improvements |\n| sqlalchemy | 2.0.25 | 2.0.35 | Medium — bug fixes |\n| pytest | 8.0.0 | 8.3.0 | Low — dev dependency |\n\nUpdate all with: `pip install -U fastapi pydantic sqlalchemy pytest`"
},
{
"instruction": "Create a cron job script that backs up the database daily",
"output": "<think>\nI'll create a backup script and show the crontab entry.\n</think>\n\n<tool_call>\n{\"name\": \"write_file\", \"args\": {\"path\": \"scripts/backup_db.sh\", \"content\": \"#!/bin/bash\\nset -euo pipefail\\n\\nBACKUP_DIR=\\\"/var/backups/myapp\\\"\\nDB_NAME=\\\"myapp\\\"\\nDB_USER=\\\"postgres\\\"\\nDATE=$(date +%Y%m%d_%H%M%S)\\nFILENAME=\\\"${BACKUP_DIR}/${DB_NAME}_${DATE}.sql.gz\\\"\\n\\n# Create backup directory\\nmkdir -p \\\"$BACKUP_DIR\\\"\\n\\n# Dump and compress\\npg_dump -U \\\"$DB_USER\\\" \\\"$DB_NAME\\\" | gzip > \\\"$FILENAME\\\"\\n\\n# Keep only last 7 days of backups\\nfind \\\"$BACKUP_DIR\\\" -name '*.sql.gz' -mtime +7 -delete\\n\\necho \\\"Backup complete: $FILENAME ($(du -h \\\"$FILENAME\\\" | cut -f1))\\\"\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"message\": \"File written: scripts/backup_db.sh (17 lines)\"}\n</tool_result>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"chmod +x scripts/backup_db.sh\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"\"}\n</tool_result>\n\nCreated `scripts/backup_db.sh`. Features:\n- Compressed pg_dump with gzip\n- Timestamped filenames\n- Auto-deletes backups older than 7 days\n\nAdd to crontab for daily 2am backups:\n```bash\ncrontab -e\n# Add this line:\n0 2 * * * /path/to/scripts/backup_db.sh >> /var/log/backup.log 2>&1\n```"
},
{
"instruction": "The tests are failing because of a missing environment variable. Fix the test configuration.",
"output": "<think>\nLet me check the test output and the test configuration.\n</think>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"python -m pytest tests/ -x --tb=short 2>&1 | head -30\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"============================= test session starts ============================\\nERROR collecting tests/test_api.py\\n  File \\\"tests/test_api.py\\\", line 5, in <module>\\n    from src.config import DB_HOST\\n  File \\\"src/config.py\\\", line 5, in <module>\\n    DB_PASS = os.environ['DB_PASS']\\nKeyError: 'DB_PASS'\"}\n</tool_result>\n\nThe config file requires `DB_PASS` to be set. Tests need test-specific environment variables. Let me create a pytest configuration with test defaults:\n\n<tool_call>\n{\"name\": \"search_files\", \"args\": {\"pattern\": \"conftest|pytest.ini|pyproject.*pytest\", \"path\": \".\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"matches\": [{\"file\": \"pyproject.toml\", \"line\": 22, \"text\": \"testpaths = [\\\"tests\\\"]\"}]}\n</tool_result>\n\n<tool_call>\n{\"name\": \"write_file\", \"args\": {\"path\": \"tests/conftest.py\", \"content\": \"import os\\n\\n# Set test environment variables before any imports\\nos.environ.setdefault('DB_HOST', 'localhost')\\nos.environ.setdefault('DB_PORT', '5432')\\nos.environ.setdefault('DB_NAME', 'myapp_test')\\nos.environ.setdefault('DB_USER', 'postgres')\\nos.environ.setdefault('DB_PASS', 'test_password')\\nos.environ.setdefault('JWT_SECRET', 'test-secret-key')\\nos.environ.setdefault('DEBUG', 'true')\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"message\": \"File written: tests/conftest.py (10 lines)\"}\n</tool_result>\n\n<tool_call>\n{\"name\": \"shell_command\", \"args\": {\"command\": \"python -m pytest tests/ -x --tb=short 2>&1 | tail -5\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"output\": \"tests/test_api.py ....                                                [40%]\\ntests/test_helpers.py .......                                           [70%]\\ntests/test_user.py .......                                              [100%]\\n\\n========================= 18 passed in 1.23s ================================\"}\n</tool_result>\n\nFixed. Created `tests/conftest.py` that sets test environment variables using `os.environ.setdefault()` — this only sets them if they aren't already set, so CI environments can still override them. All 18 tests pass now."
},
{
"instruction": "Add a rate limiter middleware to the FastAPI application",
"output": "<think>\nI'll create a simple in-memory rate limiter middleware and add it to the FastAPI app.\n</think>\n\n<tool_call>\n{\"name\": \"write_file\", \"args\": {\"path\": \"src/middleware/rate_limiter.py\", \"content\": \"import time\\nfrom collections import defaultdict\\nfrom fastapi import Request, HTTPException\\nfrom starlette.middleware.base import BaseHTTPMiddleware\\n\\nclass RateLimiter(BaseHTTPMiddleware):\\n    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):\\n        super().__init__(app)\\n        self.max_requests = max_requests\\n        self.window = window_seconds\\n        self.requests = defaultdict(list)\\n\\n    async def dispatch(self, request: Request, call_next):\\n        client_ip = request.client.host\\n        now = time.time()\\n        \\n        # Clean old entries\\n        self.requests[client_ip] = [\\n            t for t in self.requests[client_ip] if now - t < self.window\\n        ]\\n        \\n        if len(self.requests[client_ip]) >= self.max_requests:\\n            raise HTTPException(\\n                status_code=429,\\n                detail='Too many requests. Try again later.',\\n                headers={'Retry-After': str(self.window)}\\n            )\\n        \\n        self.requests[client_ip].append(now)\\n        return await call_next(request)\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"message\": \"File written: src/middleware/rate_limiter.py (29 lines)\"}\n</tool_result>\n\n<tool_call>\n{\"name\": \"read_file\", \"args\": {\"path\": \"src/main.py\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"content\": \"from fastapi import FastAPI\\n\\napp = FastAPI(title='MyApp')\\n\\n@app.get('/health')\\ndef health():\\n    return {'status': 'ok'}\"}\n</tool_result>\n\n<tool_call>\n{\"name\": \"edit_file\", \"args\": {\"path\": \"src/main.py\", \"old_text\": \"from fastapi import FastAPI\\n\\napp = FastAPI(title='MyApp')\", \"new_text\": \"from fastapi import FastAPI\\nfrom src.middleware.rate_limiter import RateLimiter\\n\\napp = FastAPI(title='MyApp')\\napp.add_middleware(RateLimiter, max_requests=60, window_seconds=60)\"}}\n</tool_call>\n<tool_result>\n{\"status\": \"success\", \"message\": \"Edit applied to src/main.py\"}\n</tool_result>\n\nAdded rate limiting middleware:\n- **60 requests per minute** per IP address (configurable)\n- Returns **429 Too Many Requests** with `Retry-After` header when exceeded\n- In-memory sliding window — for production, use Redis instead\n\nNote: This is fine for single-server deployments. For multiple servers behind a load balancer, switch to Redis-backed rate limiting."
},
]

with open("data/tool_use/eval.jsonl", "w", encoding="utf-8") as f:
    for ex in eval_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(eval_examples)} eval examples")
print("Done!")
