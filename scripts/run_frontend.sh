#!/usr/bin/env bash
# Launch the Streamlit frontend on :8501 in the background.
#
# Usage: scripts/run_frontend.sh
#   - PID written to logs/streamlit.pid
#   - stdout/stderr written to logs/streamlit.log
#   - Stop with: kill $(cat logs/streamlit.pid)
#
# Pre-requisites: text2sql pip extras installed (streamlit) and the FastAPI
# service running on :8000 (e.g. via scripts/demo.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

nohup "$PYTHON_BIN" -m streamlit run frontend/app.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    > logs/streamlit.log 2>&1 &

echo $! > logs/streamlit.pid
echo "Streamlit launched (pid=$(cat logs/streamlit.pid)). Logs: logs/streamlit.log"
