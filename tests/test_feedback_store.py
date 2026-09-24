"""Unit tests for text2sql.feedback.store.FeedbackStore and get_store factory."""
from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from text2sql.feedback import get_store
from text2sql.feedback.store import FeedbackRecord, FeedbackStore


def test_record_then_list_roundtrip():
    s = FeedbackStore(":memory:")
    rec = s.record_feedback("q1", "sess1", "correct", "looks good")
    assert rec.query_id == "q1"
    assert rec.session_id == "sess1"
    assert rec.verdict == "correct"
    assert rec.note == "looks good"
    assert rec.id  # uuid, truthy
    items = s.list_feedback("sess1")
    assert len(items) == 1
    assert items[0].id == rec.id
    assert items[0].query_id == "q1"
    assert items[0].verdict == "correct"
    assert items[0].note == "looks good"


def test_list_feedback_filters_by_session():
    s = FeedbackStore(":memory:")
    s.record_feedback("q1", "sess-A", "correct", "a-good")
    s.record_feedback("q2", "sess-B", "incorrect", "b-bad")
    s.record_feedback("q3", "sess-A", "partially_correct", "a-partial")

    a_items = s.list_feedback("sess-A")
    b_items = s.list_feedback("sess-B")
    empty_items = s.list_feedback("sess-MISSING")

    assert len(a_items) == 2
    assert {i.verdict for i in a_items} == {"correct", "partially_correct"}
    assert len(b_items) == 1
    assert b_items[0].query_id == "q2"
    assert empty_items == []


def test_get_store_factory():
    store = get_store(":memory:")
    assert isinstance(store, FeedbackStore)
    # Factory should return a usable store: round-trip a record.
    rec = store.record_feedback("qid-factory", "sess-factory", "correct")
    listed = store.list_feedback("sess-factory")
    assert len(listed) == 1
    assert listed[0].id == rec.id


def test_in_memory_store_persists_within_session():
    s1 = FeedbackStore(":memory:")
    s1.record_feedback("q1", "sess1", "correct", "s1-only")

    # A NEW in-memory store should NOT see s1's data: ':memory:' is per-store.
    s2 = FeedbackStore(":memory:")
    assert s2.list_feedback("sess1") == []
    # And s1 still has its own record.
    assert len(s1.list_feedback("sess1")) == 1


def test_thread_safety_concurrent_writes():
    s = FeedbackStore(":memory:")
    n = 20

    def _worker(i: int) -> None:
        s.record_feedback(f"q-{i}", "sess-thread", "correct", f"note-{i}")

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    items = s.list_feedback("sess-thread")
    assert len(items) == n
    # Every distinct query_id should appear exactly once.
    query_ids = [it.query_id for it in items]
    assert sorted(query_ids) == sorted(f"q-{i}" for i in range(n))
    # All verdicts should match what we wrote.
    assert all(it.verdict == "correct" for it in items)
    # Notes should be the worker index, preserved.
    notes = sorted(it.note for it in items)
    assert notes == sorted(f"note-{i}" for i in range(n))


def test_invalid_verdict_rejected_by_pydantic():
    # Pydantic Literal validation must reject an unknown verdict at model construction.
    with pytest.raises(ValidationError):
        FeedbackRecord(
            query_id="q1",
            session_id="sess1",
            verdict="wrong",  # type: ignore[arg-type]
        )
