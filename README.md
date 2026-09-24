# Text-to-SQL Interface with Guardrails and Hallucination Detection

A natural language interface that translates plain English questions into SQL queries against a real database, executes them safely with guardrails preventing destructive operations, validates that the generated SQL actually answers the question asked, and presents results with a confidence score.

---

## Problem

Taking natural language questions and turning them into correct, safe SQL is notoriously difficult. LLMs hallucinate table and column names, generate semantically wrong queries, and can produce destructive SQL (`DROP`, `DELETE`, `UPDATE`) that would corrupt production data if executed without safeguards. Most text-to-SQL demos are toy projects — they work on one hardcoded query but fail in the wild.

## Solution

This project solves those problems with a **layered pipeline**:

1. **Schema-aware prompting** — automatically introspects the database schema, filters to relevant tables, and builds a focused prompt with few-shot examples and business glossary terms. Ambiguous questions (e.g. "show me revenue") trigger a clarification request instead of a guess.
2. **Structured SQL generation** — uses `instructor` to force the LLM to return SQL plus explanation, confidence, and table/column metadata. Invalid SQL is rejected before execution.
3. **Guardrail middleware** — blocks DDL (`CREATE`, `ALTER`, `DROP`, `TRUNCATE`), DML writes (`INSERT`, `UPDATE`, `DELETE`), injects a `LIMIT` cap, rejects overly deep subqueries, and runs `EXPLAIN` to reject queries that would scan too many rows.
4. **Sandboxed execution** — every query runs inside a read-only transaction that always rolls back, connected as a database user with `SELECT`-only privileges. Defense in depth — even if a malicious query bypasses the guardrail, the database layer blocks it.
5. **Hallucination detection** — after execution, the system back-translates the SQL to a natural language question, checks semantic similarity with the original, validates result plausibility (no NULL-heavy columns, reasonable ranges), and optionally generates a second SQL via an independent approach to cross-validate results.
6. **Confidence scoring** — combines all the signals (syntax validity, back-translation alignment, sanity checks, multi-query agreement, schema coverage) into a single 0–100 score with a per-signal breakdown.
7. **Feedback loop** — users can mark results as correct/incorrect, and that feedback is persisted to a SQLite store and surfaced in the history panel.

---

## Eval Results

Measured against a [55-question golden dataset](./evals/golden/) covering simple lookups, filters, multi-table joins, GROUP BY aggregations, date-range filters, ambiguous phrasing, un-answerable questions, and destructive operations:

| Metric | Hits / Total | Rate |
|---|---:|---:|
| Guardrail effectiveness (block dangerous / allow safe) | 46 / 46 | **100.0%** |
| Hallucination detection (correct flag / pass) | 46 / 46 | **100.0%** |
| SQL exact match (stub mode) | 46 / 46 | **100.0%** |
| Unsafe queries executed | 0 / 10 dangerous cases | **0** |

Run the eval suite yourself: `python -m evals.runner --mode stub --report-dir evals/reports`

---

## Quick Start — Docker (one command, no Python install)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) and an LLM API key.

```bash
# 1. Clone the repo
git clone https://github.com/AdityaMISHRA2803/Text-to-SQL Engine.git
cd Text-to-SQL Engine

# 2. Fill in your API key
cp .env.docker.example .env
# Edit .env — set OPENAI_API_KEY or ANTHROPIC_API_KEY

# 3. Bring everything up
docker compose up --build
```

When the stack is ready you have:

| Service  | URL                           | Notes                                |
|----------|-------------------------------|--------------------------------------|
| Frontend | http://localhost:8501         | Streamlit UI — start here            |
| API      | http://localhost:8000         | FastAPI — OpenAPI docs at `/docs`    |
| API      | http://localhost:8000/health  | Health check                         |
| Postgres | `localhost:5432` (user `text2sql`) | Read-only role `text2sql_ro` for sandbox |

Open **http://localhost:8501** and start asking questions. To stop: `docker compose down`.

> Want a clean slate? `docker compose down -v` also wipes the seeded data.

---

## Local Setup — Virtual Environment (better for development)

You need Python 3.11+, Docker (for Postgres), and an LLM API key.

```bash
# 1. Clone & create a venv
git clone https://github.com/AdityaMISHRA2803/Text-to-SQL Engine.git
cd Text-to-SQL Engine
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install (editable, with dev deps)
pip install -e .[dev]

# 3. Bring up Postgres
docker compose up -d postgres

# 4. Configure env
cp .env.example .env
# Edit .env — set OPENAI_API_KEY or ANTHROPIC_API_KEY + DATABASE_URL

# 5. Seed the development database (200 rows per table, deterministic Faker seed)
python -m text2sql.seed_dataset

# 6a. Launch the API (terminal 1)
text2sql-api
# → http://localhost:8000

# 6b. Launch the frontend (terminal 2)
text2sql-ui
# → http://localhost:8501
```

> If your shell has trouble with `text2sql-api` / `text2sql-ui`, run `python -m text2sql.api.app` (API) or `streamlit run frontend/app.py` (frontend) directly.

---

## Usage

### Sample questions to try in the UI

The seeded schema is a small e-commerce store:

| Question | What it exercises |
|---|---|
| `How many customers are there?` | Simple COUNT |
| `Which customers live in the USA?` | WHERE filter |
| `Show top 5 customers by total spend` | GROUP BY + ORDER BY + LIMIT |
| `What's our total revenue from completed payments?` | SUM aggregate with WHERE |
| `Show every product with its category name` | 2-table JOIN |
| `List products with stock > 100` | Numeric filter |
| `How many orders were placed in the last 30 days?` | Relative date range |
| `Show me revenue` | Ambiguous — expect a clarification request |
| `Drop the customers table` | **Destructive** — guardrail blocks it |

### Try the API directly

With the API running on `:8000`:

```bash
# Ask a question
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many customers are there?"}' | jq

# Try a JOIN
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Show top 5 customers by total spend"}' | jq

# Force hallucination detection
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many customers are there?", "detect_hallucination": true}' | jq '.hallucination_report'

# View history
curl -s "http://localhost:8000/v1/history" | jq

# Record feedback (use a query_id from the first response)
curl -s -X POST http://localhost:8000/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{"query_id": "<paste>", "session_id": "<paste>", "verdict": "correct"}' | jq

# Try a destructive query (will be BLOCKED by the guardrail)
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Drop the customers table."}' | jq '.guardrail'
```

Browse the auto-generated OpenAPI docs at http://localhost:8000/docs.

### CLI for ad-hoc questions

```bash
# Just generate + display the prompt
python -m text2sql.cli --question "How many customers are there?"

# Generate + execute + audit
python -m text2sql.cli --question "How many customers are there?" --execute

# Same, plus hallucination detection
python -m text2sql.cli --question "How many customers are there?" \
  --execute --detect-hallucination
```

The audit log lands at `audit/queries.jsonl` (one JSON object per query).

---

## How It Works

```
┌────────────┐    ┌──────────────────────────────────────────────┐
│  Frontend  │───▶│  FastAPI (/v1/query, /v1/schema, /v1/history)│
└────────────┘    └──────────────┬───────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  1. Schema Extraction        │  SQLAlchemy introspection
                  │     + Relevance Filter       │  Embedding-based table selection
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  2. Prompt Constructor       │  Few-shot + schema + glossary
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  3. SQL Generation (LLM)     │  instructor → SQL + explanation
                  │     + Ambiguity Detection    │  + confidence + columns touched
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  4. Guardrail Middleware      │  DDL/DML block, LIMIT, depth,
                  │                              │  EXPLAIN row-estimate check
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  5. Sandbox Execution        │  Read-only txn + read-only DB user
                  │     (PostgreSQL)             │  Auto-rollback
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  6. Hallucination Detection  │  Back-translation + sanity checks
                  │     + Multi-query validation │  + confidence score aggregation
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  7. Response Packaging       │  SQL + results + EXPLAIN + score
                  │     + Audit Log              │
                  └──────────────────────────────┘
```

Each stage is independently testable and was built with evaluation in mind:

1. **Schema Extraction** — SQLAlchemy introspection enumerates tables, columns, types, PK/FK relationships. An embedding-based relevance filter keeps only tables related to the user's question.
2. **Prompt Constructor** — assembles a focused prompt with filtered schema, FK graph, sample values, business glossary, and 3–5 few-shot Q→SQL pairs.
3. **SQL Generation** — calls the LLM via `instructor` for structured output. Ambiguous questions trigger a `ClarificationRequest` listing 2–4 interpretations instead of a single guess.
4. **Guardrail Middleware** — layered checks: regex + parsed-token DDL/DML blocking, LIMIT injection, subquery depth limit, EXPLAIN row-estimate threshold. Every blocked query is logged with a reason.
5. **Sandbox Execution** — executes inside a read-only transaction that always rolls back, connected as a `SELECT`-only database user.
6. **Hallucination Detection** — back-translates SQL to a question, checks cosine similarity with the original; runs result sanity checks (NULL-heavy columns, empty results, out-of-range values); optionally generates a second SQL via a different approach and cross-validates.
7. **Response Packaging** — returns SQL, results, execution time, EXPLAIN plan, confidence score (0–100 with per-signal breakdown), and writes to an append-only audit log.

---

## Tech Stack

| Component | Tool / Library |
|---|---|
| Language | Python 3.11+ |
| LLM | GPT-4o or Claude Sonnet (via LiteLLM) |
| Database | PostgreSQL 16 |
| Schema Extraction | SQLAlchemy |
| Guardrails | Custom middleware |
| Validation | LLM-as-judge + result sanity checks |
| API | FastAPI (Pydantic v2) |
| Frontend | Streamlit |
| LLM Plumbing | `instructor` library |
| Containerization | Docker + docker-compose |

---

## Project Structure

```
.
├── README.md                          # this file
├── RUN.md                             # standalone quickstart guide
├── docker-compose.yml                 # 4 services: postgres, seed, api, frontend
├── Dockerfile                         # multi-purpose image (api + frontend)
├── pyproject.toml
├── .env.example                       # local dev env (LLM keys, DATABASE_URL)
├── .env.docker.example                # template for `docker compose up`
├── text2sql/                          # the package
│   ├── schema/                        # schema extraction, relevance, prompt construction, ambiguity
│   ├── generation/                    # SQL generation via LLM
│   ├── guardrails/                    # DDL/DML/depth/EXPLAIN blocking
│   ├── execution/                     # sandbox, query runner, audit log
│   ├── llm/                           # LLMClient + embedder (LiteLLM)
│   ├── validation/                    # back-translator, sanity checker, multi-query validator, detector
│   ├── scoring/                       # confidence scorer + statistical helpers
│   ├── api/                           # FastAPI app
│   └── feedback/                      # feedback store (SQLite)
├── frontend/
│   └── app.py                         # Streamlit UI
├── evals/                             # evaluation suite
│   ├── golden/                        # 55-case golden dataset (8 category files)
│   ├── metrics.py                     # normalize_sql, execution_match, Counters
│   ├── completion_fns.py              # stub + live completion_fn factories
│   ├── runner.py                      # CLI entry point
│   ├── report.py                      # JSON + Markdown report writers
│   └── reports/                       # generated per-run artifacts (gitignored)
├── scripts/                           # convenience launchers
│   ├── demo.sh                        # demo transcript
│   ├── demo_api.py                    # scripted httpx flow for the demo
│   └── run_frontend.sh                # nohup launcher for the Streamlit UI
├── tests/                             # 400+ tests across all modules
└── docker/
    └── postgres-init/
        └── 01-create-readonly-role.sql  # creates text2sql_ro + auto-grant trigger
```

---

## Running Tests

```bash
# Quick (DB-free; ~5 s)
pytest -q tests/test_eval_metrics.py tests/test_eval_runner.py tests/test_api.py

# Full suite including DB-dependent tests (~30 s; needs Docker for the testcontainer)
pytest -q
```

The full suite is gated on `tests/conftest.py` fixtures that spin up a real Postgres testcontainer; they skip cleanly if Docker isn't available.

---

## Running the Evaluation Suite

The eval suite replays the [55-case golden dataset](./evals/golden/) through the pipeline and writes a timestamped report to `evals/reports/`.

```bash
# Stub mode (no LLM needed, no DB needed for non-execution metrics)
python -m evals.runner --mode stub --report-dir evals/reports

# Stub mode with execution_match (requires Postgres up)
docker compose up -d postgres
python -m evals.runner --mode stub --report-dir evals/reports --reseed

# Live mode (calls the real LLM — needs OPENAI_API_KEY or ANTHROPIC_API_KEY)
python -m evals.runner --mode live \
  --provider openai --model gpt-4o-mini \
  --report-dir evals/reports
```

The runner exits 0 when every case ran to completion. Per-case pass/fail lives in the report, not the exit code.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `text2sql-api: command not found` | venv not activated | `source .venv/bin/activate` |
| `connection to server ... port 5432 failed` | Postgres not running | `docker compose up -d postgres` |
| `ModuleNotFoundError: No module named 'text2sql'` | Did `pip install -e .`? | Re-run `pip install -e .[dev]` inside the venv |
| `ModuleNotFoundError: No module named 'streamlit'` | Used wrong Python | Use the venv python, or `pip install streamlit` |
| Streamlit UI loads but queries don't work | API not running on `:8000` | Start `text2sql-api` in another terminal |
| `litellm.AuthenticationError` | Bad API key | Check `.env` — make sure the key isn't blank |
| Tests collect errors on `streamlit` / `sqlparse` | Wrong Python | Use `.venv/bin/python -m pytest`, not the system Python |

---

## Where to Look Next

- **API code:** [`text2sql/api/app.py`](./text2sql/api/app.py) — FastAPI routes, request/response models
- **Frontend:** [`frontend/app.py`](./frontend/app.py) — Streamlit UI, history sidebar, feedback buttons
- **Pipeline entry:** [`text2sql/cli.py`](./text2sql/cli.py) — `run_pipeline` / `run_execute`
- **Guardrails:** [`text2sql/guardrails/validator.py`](./text2sql/guardrails/validator.py) — DDL/DML/depth/EXPLAIN checks
- **Hallucination detector:** [`text2sql/validation/detector.py`](./text2sql/validation/detector.py) — orchestrates the signals
- **Eval runner:** [`evals/runner.py`](./evals/runner.py) — orchestrator
- **Golden dataset:** [`evals/golden/`](./evals/golden/) — 55 cases across 8 categories