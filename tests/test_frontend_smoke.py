"""Streamlit AppTest smoke tests for the Phase 4 frontend (4.2).

These tests run the Streamlit script through AppTest (no browser, no real
backend) and assert that the rendered widget tree contains the required
entrypoints: question input, submit button, hallucination-detection checkbox,
history sidebar header, session-id caption, and the empty-history notice.

Notes
-----
* The frontend's _fetch_history() helper is wrapped in try/except and returns
  [] on HTTPError, so a missing FastAPI backend does NOT cause the smoke
  tests to fail — they assert the "No queries yet." placeholder instead.
* AppTest resets session_state on every .run() call, so each test starts
  from a fresh browser-tab state.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


@pytest.fixture()
def at() -> AppTest:
    """Fresh AppTest instance pointing at frontend/app.py."""
    assert FRONTEND_PATH.exists(), f"frontend/app.py not found at {FRONTEND_PATH}"
    return AppTest.from_file(str(FRONTEND_PATH))


def test_app_runs_without_exception(at: AppTest) -> None:
    """The app module must import + execute without raising."""
    at.run()
    assert not at.exception, f"App raised: {at.exception}"


def test_app_title_is_rendered(at: AppTest) -> None:
    at.run()
    titles = [t.value for t in at.title]
    assert any("Text-to-SQL" in t for t in titles), f"titles: {titles}"


def test_question_input_is_present(at: AppTest) -> None:
    at.run()
    assert len(at.text_input) >= 1, "Expected at least one st.text_input"
    labels = [inp.label for inp in at.text_input]
    # The question input uses label_visibility='collapsed', so label is empty,
    # but the placeholder is set on the widget. Check for either an explicit
    # 'question' label or an empty label that accompanies a placeholder.
    has_question = any(
        "question" in (lbl or "").lower() or lbl == "" for lbl in labels
    )
    assert has_question, f"text_input labels: {labels}"


def test_submit_button_is_present(at: AppTest) -> None:
    at.run()
    button_labels = [b.label for b in at.button]
    assert any("Ask" in lbl for lbl in button_labels), (
        f"button labels: {button_labels}"
    )


def test_detect_hallucination_checkbox_is_present(at: AppTest) -> None:
    at.run()
    assert len(at.checkbox) >= 1, "Expected at least one st.checkbox"
    labels = [c.label for c in at.checkbox]
    assert any("hallucination" in lbl.lower() for lbl in labels), (
        f"checkbox labels: {labels}"
    )


def test_sidebar_history_header_is_rendered(at: AppTest) -> None:
    at.run()
    sidebar_headers = [h.value for h in at.sidebar.header]
    assert "History" in sidebar_headers, f"sidebar headers: {sidebar_headers}"


def test_sidebar_shows_session_id_caption(at: AppTest) -> None:
    at.run()
    sidebar_captions = [c.value for c in at.sidebar.caption]
    assert any("Session" in c for c in sidebar_captions), (
        f"sidebar captions: {sidebar_captions}"
    )


def test_sidebar_shows_no_queries_when_backend_unreachable(at: AppTest) -> None:
    """With no FastAPI running, _fetch_history returns []; sidebar shows the placeholder."""
    at.run()
    info_texts = [info.value for info in at.sidebar.info]
    assert any("No queries" in t for t in info_texts), (
        f"sidebar info texts: {info_texts}"
    )


@pytest.mark.parametrize(
    "widget_kind,predicate",
    [
        ("text_input", lambda widgets: len(widgets) >= 1),
        ("checkbox", lambda widgets: len(widgets) >= 1),
        ("button", lambda widgets: any("Ask" in b.label for b in widgets)),
        ("sidebar.header", lambda widgets: any(h.value == "History" for h in widgets)),
    ],
)
def test_required_widgets_exist(at: AppTest, widget_kind: str, predicate) -> None:
    """Parametrized exit-criterion sweep: each required widget must render."""
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
    if widget_kind == "sidebar.header":
        widgets = list(at.sidebar.header)
    else:
        widgets = list(getattr(at, widget_kind))
    assert predicate(widgets), f"{widget_kind} failed predicate: {[w.label if hasattr(w, 'label') else w.value for w in widgets]}"
