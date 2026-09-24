"""Confidence scoring package (README 3.4).

Combines per-signal hallucination detection scores into a single
weighted 0-100 confidence score with verdict classification.

Public API:
- HallucinationScorer: weights + thresholds, `score(signals) -> ConfidenceScore`
- ConfidenceScore: aggregate result with score / verdict / flagged / breakdown
- rank: manual 1-based average ranks (used by spearman)
- pearson: manual Pearson correlation (no scipy/numpy)
- spearman: Spearman rank correlation via pearson on ranks
"""

from .scorer import (
    ConfidenceScore,
    HallucinationScorer,
    pearson,
    rank,
    spearman,
)

__all__ = [
    "ConfidenceScore",
    "HallucinationScorer",
    "pearson",
    "rank",
    "spearman",
]
