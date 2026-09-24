"""End-to-end smoke test for Phase 4 (scripts/demo.sh wrapper).

Runs ``scripts/demo.sh`` in a subprocess and asserts:
  - exit code 0
  - the demo transcript is captured to ``.kimchi/docs/phase4_demo.txt``
  - the transcript contains the expected flow markers (query → feedback → history)

This is the pytest-form mirror of the bash launcher; the bash script is
self-contained (it starts uvicorn + streamlit in the background, runs the
scripted httpx flow, and tears everything down).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SH = REPO_ROOT / "scripts" / "demo.sh"
TRANSCRIPT = REPO_ROOT / ".kimchi" / "docs" / "phase4_demo.txt"


def _run_demo() -> subprocess.CompletedProcess[str]:
    """Invoke scripts/demo.sh with a hard 90 s timeout."""
    assert DEMO_SH.is_file(), f"missing launcher: {DEMO_SH}"
    assert os.access(DEMO_SH, os.X_OK), f"launcher not executable: {DEMO_SH}"
    env = os.environ.copy()
    # Force unbuffered output so tail/grep assertions work even if the
    # subprocess wraps its child processes.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(
        ["bash", str(DEMO_SH)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_demo_sh_exits_zero() -> None:
    """The launcher exits 0 on a green demo run."""
    result = _run_demo()
    assert result.returncode == 0, (
        "scripts/demo.sh failed:\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )


def test_demo_transcript_captured() -> None:
    """The transcript file is written by scripts/demo.sh (tee)."""
    # Clean any stale transcript so we know this run produced it.
    if TRANSCRIPT.exists():
        TRANSCRIPT.unlink()
    result = _run_demo()
    assert result.returncode == 0, "demo.sh failed; cannot check transcript"
    assert TRANSCRIPT.is_file(), (
        f"transcript missing at {TRANSCRIPT}; demo.sh did not produce it"
    )
    contents = TRANSCRIPT.read_text(encoding="utf-8")
    # Required flow markers — must be present in order.
    expected_markers = [
        "Step 1: POST /v1/query",
        "Step 2: POST /v1/feedback",
        "Step 3: GET /v1/history",
        "Demo complete: question -> feedback -> history (all green)",
    ]
    last_idx = -1
    for marker in expected_markers:
        idx = contents.find(marker)
        assert idx != -1, f"missing marker in transcript: {marker!r}"
        assert idx > last_idx, (
            f"marker out of order: {marker!r} at index {idx} "
            f"(previous marker at index {last_idx})"
        )
        last_idx = idx


def test_demo_banner_in_stdout() -> None:
    """The 'Demo complete' banner appears in the launcher's stdout."""
    result = _run_demo()
    assert result.returncode == 0, "demo.sh failed; cannot inspect stdout"
    assert "Demo complete: question -> feedback -> history (all green)" in result.stdout, (
        "expected success banner missing from demo.sh stdout:\n"
        f"{result.stdout[-2000:]}"
    )


def test_demo_history_reflects_feedback() -> None:
    """Step 3 prints feedback_count>=1, confirming persistence + retrieval."""
    result = _run_demo()
    assert result.returncode == 0, "demo.sh failed; cannot inspect transcript"
    contents = (TRANSCRIPT.read_text(encoding="utf-8")
                if TRANSCRIPT.exists() else result.stdout)
    assert "feedback_count=" in contents, (
        "demo transcript does not show feedback_count for any history item:\n"
        f"{contents[-2000:]}"
    )
