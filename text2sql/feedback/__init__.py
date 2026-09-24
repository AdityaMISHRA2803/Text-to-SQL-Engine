"""Human-in-the-loop feedback persistence (SQLite-backed)."""
from text2sql.feedback.store import (
    FeedbackRecord,
    FeedbackStore,
    QueryRecord,
    Verdict,
    get_store,
)

__all__ = [
    "FeedbackRecord",
    "FeedbackStore",
    "QueryRecord",
    "Verdict",
    "get_store",
]
