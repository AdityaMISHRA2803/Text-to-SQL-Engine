"""Console-script entrypoints for the Text-to-SQL service (Phase 4)."""
from __future__ import annotations


def run_api() -> None:
    """Launch the FastAPI service via uvicorn."""
    import uvicorn
    from text2sql.api.app import app
    uvicorn.run(app, host="127.0.0.1", port=8000)


def run_ui() -> None:
    """Launch the Streamlit frontend via streamlit run."""
    import os
    import sys
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port=8501", "--server.address=127.0.0.1"],
    )
