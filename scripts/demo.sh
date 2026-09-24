#!/usr/bin/env bash
# End-to-end Phase 4 demo: launch the (stubbed) API + Streamlit UI,
# run a scripted question/feedback/history cycle, capture the transcript.
#
# Usage: scripts/demo.sh
#   - PID files: logs/demo_api.pid, logs/streamlit.pid
#   - Log files: logs/demo_api.log, logs/streamlit.log
#   - Transcript: .kimchi/docs/phase4_demo.txt
#   - Exit trap kills both background jobs even on Ctrl-C / script error.
#
# Exit code: 0 on full success (question + feedback + history all verified),
# 1 if any step fails.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs .kimchi/docs

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

API_HOST="127.0.0.1"
API_PORT="8000"
UI_HOST="127.0.0.1"
UI_PORT="8501"
DOCS_OUT="$REPO_ROOT/.kimchi/docs/phase4_demo.txt"

API_PID=""
UI_PID=""

cleanup() {
    set +e
    echo "[demo] cleanup: stopping background processes"
    if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    if [[ -n "$UI_PID" ]] && kill -0 "$UI_PID" 2>/dev/null; then
        kill "$UI_PID" 2>/dev/null || true
        wait "$UI_PID" 2>/dev/null || true
    fi
    rm -f logs/demo_api.pid logs/streamlit.pid
}
trap cleanup EXIT INT TERM

wait_for_url() {
    local url="$1"
    local label="$2"
    local max="${3:-30}"
    for i in $(seq 1 "$max"); do
        if curl -sf -o /dev/null "$url"; then
            echo "[demo] $label ready: $url (after ${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "[demo] ERROR: $label never became ready at $url after ${max}s" >&2
    return 1
}

echo "[demo] starting demo API on :$API_PORT (background) ..."
nohup "$PYTHON_BIN" "$REPO_ROOT/scripts/demo_api.py" > logs/demo_api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/demo_api.pid

echo "[demo] starting Streamlit UI on :$UI_PORT (background) ..."
nohup "$PYTHON_BIN" -m streamlit run "$REPO_ROOT/frontend/app.py" \
    --server.port="$UI_PORT" \
    --server.address="$UI_HOST" \
    --server.headless=true \
    > logs/streamlit.log 2>&1 &
UI_PID=$!
echo "$UI_PID" > logs/streamlit.pid

echo "[demo] waiting for API health endpoint ..."
wait_for_url "http://$API_HOST:$API_PORT/health" "API" 30

echo "[demo] waiting for Streamlit UI ..."
wait_for_url "http://$UI_HOST:$UI_PORT/_stcore/health" "Streamlit" 30

echo "[demo] both services up; running scripted Phase 4 demo ..."

SESSION_ID="phase4-demo-$(date +%s)"

DEMO_OUTPUT="$(
    SESSION_ID="$SESSION_ID" \
    API_BASE="http://$API_HOST:$API_PORT" \
    "$PYTHON_BIN" - <<'PY'
import os
import sys
import httpx

BASE = os.environ["API_BASE"]
SID = os.environ["SESSION_ID"]

print(f"Session ID: {SID}")
print()

print("=" * 60)
print("Step 1: POST /v1/query")
print("=" * 60)
question = "How many users signed up last week?"
r = httpx.post(
    f"{BASE}/v1/query",
    json={
        "question": question,
        "session_id": SID,
        "detect_hallucination": True,
    },
    timeout=15.0,
)
r.raise_for_status()
qr = r.json()
print(f"  question:    {qr['question']}")
print(f"  query_id:    {qr['query_id']}")
print(f"  sql:         {qr['sql']}")
print(f"  confidence:  {qr['confidence']:.2f}")
print(f"  guardrail:   allowed={qr['guardrail']['allowed']}")
if qr.get("execution"):
    print(f"  row_count:   {qr['execution']['row_count']}")
if qr.get("hallucination_report"):
    hr = qr["hallucination_report"]
    print(f"  hallucination: verdict={hr['verdict']} overall_score={hr['overall_score']:.1f}")
query_id = qr["query_id"]
print()

print("=" * 60)
print("Step 2: POST /v1/feedback")
print("=" * 60)
r = httpx.post(
    f"{BASE}/v1/feedback",
    json={
        "query_id": query_id,
        "session_id": SID,
        "verdict": "correct",
        "note": "matches expectation (demo stub)",
    },
    timeout=10.0,
)
r.raise_for_status()
fr = r.json()
print(f"  feedback_id: {fr['id']}")
print(f"  verdict:     {fr['verdict']}")
print(f"  note:        {fr['note']}")
print()

print("=" * 60)
print("Step 3: GET /v1/history (verify feedback reflected)")
print("=" * 60)
r = httpx.get(f"{BASE}/v1/history", params={"session_id": SID}, timeout=10.0)
r.raise_for_status()
hist = r.json()
print(f"  history_items: {len(hist)}")
for i, item in enumerate(hist, 1):
    fb = item.get("feedback", [])
    print(f"  item {i}: query_id={item['query_id'][:8]}... sql={item['sql'][:40]!r} feedback_count={len(fb)}")
    for f in fb:
        print(f"    - verdict={f['verdict']} note={f.get('note')!r}")

assert len(hist) >= 1, "history must contain at least one query"
assert any(len(item.get("feedback", [])) >= 1 for item in hist), "feedback not reflected in /v1/history"

print()
print("=" * 60)
print("Demo complete: question -> feedback -> history (all green)")
print("=" * 60)
PY
)"

# Capture to file and echo to stdout.
echo "$DEMO_OUTPUT" | tee "$DOCS_OUT"

echo "[demo] transcript captured to $DOCS_OUT"
echo "[demo] done"
