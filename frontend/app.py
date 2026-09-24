"""Streamlit frontend for the Text-to-SQL API (Phase 4 / 4.2).

This app is the user-facing demo: it consumes the FastAPI service via HTTP
(httpx) and never imports text2sql internals directly. The backend is a
separate process (run via `text2sql-api`); this script only talks to it.

Architecture:
    user → streamlit (:8501) → httpx → fastapi (:8000) → text2sql pipeline

Run:
    text2sql-api   # starts FastAPI on :8000
    text2sql-ui    # starts Streamlit on :8501
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

# Resolve API base URL. Three sources, in order:
#   1. TEXT2SQL_API_URL environment variable (preferred for tests / CI).
#   2. Streamlit secrets.toml entry [api_base_url] (preferred for production).
#   3. Default http://127.0.0.1:8000 (local dev).
# NOTE: st.secrets does NOT support .get(); it raises on missing keys, so we
# only touch it when explicitly opted in via TEXT2SQL_USE_STREAMLIT_SECRETS=1.
_API_URL = os.environ.get("TEXT2SQL_API_URL")
if _API_URL:
    API_BASE_URL = _API_URL
elif os.environ.get("TEXT2SQL_USE_STREAMLIT_SECRETS") == "1":
    try:
        API_BASE_URL = st.secrets["api_base_url"]  # type: ignore[index]
    except Exception:
        API_BASE_URL = "http://127.0.0.1:8000"
else:
    API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Text-to-SQL", page_icon="🔍", layout="wide")
st.title("🔍 Text-to-SQL")
st.caption("Ask a question, inspect the generated SQL, run it, and rate the result.")

# Stable session id for the lifetime of this browser tab.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_response" not in st.session_state:
    st.session_state.last_response = None


def _fetch_history() -> list[dict]:
    """Fetch session history from /v1/history; return [] on any error."""
    try:
        r = httpx.get(
            f"{API_BASE_URL}/v1/history",
            params={"session_id": st.session_state.session_id},
            timeout=10.0,
        )
        r.raise_for_status()
        items = r.json()
        return items if isinstance(items, list) else []
    except httpx.HTTPError:
        return []


# Cache /v1/history in session_state so the sidebar doesn't refetch on every rerun.
if "history_cache" not in st.session_state:
    st.session_state.history_cache: list[dict] = _fetch_history()

with st.sidebar:
    st.header("History")
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")
    history_items = st.session_state.history_cache or []
    if history_items:
        # Most-recent first; each entry has a clickable 'Open in editor' button.
        for idx, item in enumerate(reversed(history_items)):
            qid = item.get("query_id", "")
            label = (item.get("question") or "")[:60]
            with st.expander(f"{label}…"):
                st.write(f"**SQL:** `{item.get('sql', '')[:120]}`")
                st.write(f"**Confidence:** {float(item.get('confidence', 0)):.2f}")
                feedback_list = item.get("feedback") or []
                if feedback_list:
                    st.write(f"**Feedback:** {len(feedback_list)} record(s)")
                if st.button("Open in editor", key=f"open_{qid}_{idx}"):
                    target = next((h for h in history_items if h.get("query_id") == qid), None)
                    if target:
                        st.session_state.last_response = target
                        st.rerun()
        if st.button("Refresh history", key="refresh_history"):
            st.session_state.history_cache = _fetch_history()
            st.rerun()
    else:
        st.info("No queries yet.")

st.subheader("Ask a question")
with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Natural-language question",
        placeholder="e.g. How many users signed up last week?",
        label_visibility="collapsed",
    )
    detect_hallucination = st.checkbox("Run hallucination detection", value=False)
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Generating SQL..."):
        try:
            r = httpx.post(
                f"{API_BASE_URL}/v1/query",
                json={
                    "question": question.strip(),
                    "session_id": st.session_state.session_id,
                    "detect_hallucination": detect_hallucination,
                },
                timeout=60.0,
            )
            r.raise_for_status()
            st.session_state.last_response = r.json()
            # Refresh the sidebar history cache after a successful query.
            st.session_state.history_cache = _fetch_history()
        except httpx.HTTPError as e:
            st.error(f"API request failed: {e}")

if st.session_state.last_response:
    resp = st.session_state.last_response
    st.subheader("Generated SQL (editable)")
    # st.text_area is editable; user can tweak the generated SQL before re-executing.
    edited_sql = st.text_area(
        label="Generated SQL (editable)",
        value=resp.get("sql", ""),
        height=180,
        key=f"sql_editor_{resp.get('query_id', uuid.uuid4())}",
    )
    # Syntax-highlighted preview as well (read-only).
    with st.expander("Syntax-highlighted preview", expanded=True):
        st.code(edited_sql, language="sql")

    # Re-execute button: posts the edited SQL back via sql_override.
    if st.button("Re-execute", type="primary", key=f"re_exec_{resp.get('query_id', uuid.uuid4())}"):
        with st.spinner("Re-executing..."):
            try:
                # re.execute: send edited_sql as sql_override; backend skips generation.
                r = httpx.post(
                    f"{API_BASE_URL}/v1/query",
                    json={
                        "question": resp.get("question", ""),
                        "session_id": resp.get("session_id", st.session_state.session_id),
                        "sql_override": edited_sql,
                        "detect_hallucination": detect_hallucination,
                    },
                    timeout=60.0,
                )
                r.raise_for_status()
                st.session_state.last_response = r.json()
                st.rerun()
            except httpx.HTTPError as e:
                st.error(f"Re-execute failed: {e}")

    # Score block — confidence always shown; hallucination report shown when present.
    st.metric(
        label="Generator confidence",
        value=f"{float(resp.get('confidence', 0)):.2f}",
    )
    hr = resp.get("hallucination_report")
    if hr:
        overall = float(hr.get("overall_score", 0))
        verdict = str(hr.get("verdict", "low"))
        st.metric(label="Hallucination score", value=f"{overall:.1f}")
        verdict_label = {"high": "✅ High confidence", "moderate": "⚠️ Moderate", "low": "❌ Low"}.get(
            verdict, verdict
        )
        st.markdown(f"**Verdict:** {verdict_label}")
        with st.expander("Detection details", expanded=False):
            st.json(hr)

    if resp.get("execution"):
        st.subheader("Results")
        st.dataframe(resp["execution"].get("results", []), use_container_width=True)

    # st.feedback section: thumbs up/down buttons + optional note -> POST /v1/feedback.
    query_id = resp.get("query_id")
    if query_id:
        st.subheader("Feedback")
        fb_cols = st.columns([1, 1, 4])
        with fb_cols[0]:
            thumbs_up = st.button("👍 Correct", key=f"thumb_up_{query_id}")
        with fb_cols[1]:
            thumbs_down = st.button("👎 Incorrect", key=f"thumb_down_{query_id}")
        note = st.text_input(
            "Optional note",
            key=f"fb_note_{query_id}",
            placeholder="Anything to add?",
            label_visibility="collapsed",
        )
        verdict = None
        if thumbs_up:
            verdict = "correct"
        elif thumbs_down:
            verdict = "incorrect"
        if verdict is not None:
            with st.spinner("Submitting feedback..."):
                try:
                    r = httpx.post(
                        f"{API_BASE_URL}/v1/feedback",
                        json={
                            "query_id": query_id,
                            "session_id": resp.get("session_id", st.session_state.session_id),
                            "verdict": verdict,
                            "note": note or None,
                        },
                        timeout=10.0,
                    )
                    r.raise_for_status()
                    st.success(f"Recorded '{verdict}' feedback.")
                    # Refresh history so the sidebar shows the new feedback entry.
                    st.session_state.history_cache = _fetch_history()
                except httpx.HTTPError as e:
                    st.error(f"Feedback failed: {e}")
