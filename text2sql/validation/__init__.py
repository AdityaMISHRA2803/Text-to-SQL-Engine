"""SQL validation utilities (Phase 2 — README 2.3 / 3.1 / 3.2 / 3.3).

This package exposes the back-translation component used to detect
when generated SQL does not match the user's original question, the
result-sanity checker that flags implausible values (out-of-range
percentages, negative counts, NULL-heavy columns, empty results, and
suspicious aggregations), and the multi-query agreement validator
that compares two independent SQL attempts on the same question.
"""
from .back_translator import BackTranslationResult, BackTranslator
from .multi_query import MultiQueryResult, MultiQueryValidator
from .sanity_checker import SanityAnomaly, SanityCheckResult, SanityChecker

__all__ = [
    "BackTranslationResult",
    "BackTranslator",
    "MultiQueryResult",
    "MultiQueryValidator",
    "SanityAnomaly",
    "SanityCheckResult",
    "SanityChecker",
]
