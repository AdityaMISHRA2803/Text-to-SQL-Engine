# RUN — How to play with the text2sql stack

This guide gets the project running end-to-end on your machine in under five minutes.
Pick **Docker** (zero-setup, recommended) or **Local venv** (better for hacking on the code).

---

## Option A — Docker (one command, no Python install)

You need: [Docker Desktop](https://www.docker.com/products/docker-desktop/) and an LLM API key.

```bash
# 1. Clone & enter the repo
git clone https://github.com/AdityaMISHRA2803/Text-to-SQL Engine.git
cd Text-to-SQL Engine

# 2. Fill in your API key
cp .env.docker.example .env
$EDITOR .env        # set OPENAI_API_KEY or ANTHROPIC_API_KEY

# 3. Bring everything up
docker compose up --build
```

When it settles you have:

| Service  | URL                              | Notes                                    |
|----------|----------------------------------|------------------------------------------|
| Frontend | http://localhost:8501            | Streamlit UI — start here                |
| API      | http://localhost:8000            | FastAPI — OpenAPI docs at `/docs`        |
| API      | http://localhost:8000/health     | Health check                             |
| Postgres | `localhost:5432` (user `text2sql`) | Read-only role `text2sql_ro` for sandbox |

Open **http://localhost:8501** and start asking questions. To stop: `docker compose down`.

> Want a clean slate? `docker compose down -v` also wipes the seeded data.

---

## Option B — Local venv (better for development)

You need: Python 3.11+, Docker (for Postgres), and an LLM API key.

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
$EDITOR .env        # set OPENAI_API_KEY or ANTHROPIC_API_KEY + DATABASE_URL

# 5. Seed the dev DB (200 rows per table, deterministic Faker seed)
python -m text2sql.seed_dataset

# 6a. Launch the API (terminal 1)
text2sql-api
# → http://localhost:8000

# 6b. Launch the frontend (terminal 2)
text2sql-ui
# → http://localhost:8501
```

> **Tip:** if your shell has trouble with `text2sql-api` / `text2sql-ui`, you can also run
> `python -m text2sql.api.app` (uvicorn-style) or
> `streamlit run frontend/app.py` directly.

---

## Try the API directly

With the API running, hit it from any terminal:

```bash
# 1. Ask a question
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many customers are there?"}' | jq

# 2. Try a JOIN
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Show top 5 customers by total spend"}' | jq

# 3. Force hallucination detection
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How many customers are there?",
    "detect_hallucination": true
  }' | jq '.hallucination_report'

# 4. View history
curl -s "http://localhost:8000/v1/history" | jq

# 5. Record feedback (use a query_id from step 1's response)
curl -s -X POST http://localhost:8000/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{"query_id": "<paste>", "session_id": "<paste>", "verdict": "correct"}' | jq

# 6. Try a destructive query (will be BLOCKED by the guardrail)
curl -s -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Drop the customers table."}' | jq '.guardrail'
```

**Or browse the auto-generated OpenAPI docs:** http://localhost:8000/docs

---

## Sample questions to try in the UI

The seeded schema is a small e-commerce store. Try these:

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

---

## Run the test suite

```bash
# Quick (DB-free; ~5 s)
pytest -q tests/test_eval_metrics.py tests/test_eval_runner.py tests/test_api.py

# Full suite including DB-dependent tests (~30 s; needs Docker for the testcontainer)
pytest -q
```

The full suite is gated on `tests/conftest.py` fixtures that spin up a real Postgres
testcontainer; they skip cleanly if Docker isn't available.

---

## Run the eval suite

The eval suite replays the [55-case golden dataset](./evals/golden/) through the
pipeline and writes a timestamped report to `evals/reports/`.

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

The runner exits 0 when every case ran to completion. Per-case pass/fail lives in
the report, not the exit code, so CI can ingest the JSON without arguing with the
runner about what "passing" means.

---

## CLI for ad-hoc questions

Want to ask one question from your terminal? The CLI is also there.

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

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `text2sql-api: command not found` | venv not activated | `source .venv/bin/activate` |
| `connection to server ... port 5432 failed` | Postgres not running | `docker compose up -d postgres` |
| `ModuleNotFoundError: No module named 'text2sql'` | Did `pip install -e .`? | Re-run `pip install -e .[dev]` inside the venv |
| `ModuleNotFoundError: No module named 'streamlit'` | Used `python` instead of `.venv/bin/python` | Use the venv python, or `pip install streamlit` |
| Streamlit UI loads but asks don't work | API not running on `:8000` | Start `text2sql-api` in another terminal |
| `litellm.AuthenticationError` | Bad API key | Check `.env` / `.env.docker` — make sure the key isn't blank |
| Tests collect errors on `streamlit` / `sqlparse` | Wrong Python | Use `.venv/bin/python -m pytest`, not the system `python` |

---

## Where to look next

- **API code:** [`text2sql/api/app.py`](./text2sql/api/app.py) — FastAPI routes, request/response models.
- **Frontend:** [`frontend/app.py`](./frontend/app.py) — Streamlit UI, history sidebar, feedback buttons.
- **Pipeline (CLI entry):** [`text2sql/cli.py`](./text2sql/cli.py) — `run_pipeline` / `run_execute`.
- **Guardrails:** [`text2sql/guardrails/validator.py`](./text2sql/guardrails/validator.py) — DDL/DML/depth/EXPLAIN checks.
- **Hallucination detector:** [`text2sql/validation/detector.py`](./text2sql/validation/detector.py) — orchestrates the 5 signals.
- **Eval runner:** [`evals/runner.py`](./evals/runner.py) — orchestrator.
- **Golden dataset:** [`evals/golden/`](./evals/golden/) — 55 cases across 8 categories.

---

## The shortest possible demo

If you have 60 seconds and Docker already running:

```bash
docker compose up --build -d
open http://localhost:8501
```

Ask: **"Show top 5 customers by total spend"** — you'll see SQL, results, and
a confidence score. Click 👍 in the feedback box, then ask another question
and you'll see your verdict reflected in the sidebar history.
