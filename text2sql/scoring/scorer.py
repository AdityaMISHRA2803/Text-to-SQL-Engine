"""Weighted confidence scoring (README 3.4).

Combines per-signal scores into a single 0-100 confidence score with
configurable weights from Settings. Includes a manual Spearman
correlation helper for offline evaluation against ground-truth
correctness (no scipy/numpy dependency).
"""
from __future__ import annotations

import logging
import math
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from text2sql.config import get_settings

logger = logging.getLogger("text2sql.scoring.scorer")


#: Canonical signal names. Any signal not in this set is still scored
#: using whatever weight the caller provides, but unknown names are not
#: auto-filled with a default 1.0 (only missing KNOWN signals get that
#: defensive default).
_KNOWN_SIGNALS = ("syntax", "back_translation", "sanity", "multi_query", "schema_coverage")


class ConfidenceScore(BaseModel):
    """Aggregate weighted confidence score for one SQL generation."""

    score: float = Field(ge=0.0, le=100.0, description="Final 0-100 confidence score.")
    verdict: str  # "pass" | "warn" | "fail"
    flagged: bool  # True iff verdict == "fail"
    signal_scores: dict[str, float]  # normalised per-signal scores in [0,1]
    weights: dict[str, float]  # weights after re-normalisation
    notes: list[str] = Field(default_factory=list)


def rank(values: Sequence[float]) -> list[float]:
    """Return 1-based average ranks. Ties get the mean of tied ranks.

    >>> rank([10, 30, 20])
    [1.0, 3.0, 2.0]
    >>> rank([10, 10, 20])
    [1.5, 1.5, 3.0]
    """
    n = len(values)
    if n == 0:
        raise ValueError("rank() requires at least one value")

    # Stable sort by value; record the original index so we can place
    # the computed rank back into the right slot.
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks: list[Optional[float]] = [None] * n
    i = 0
    while i < n:
        j = i
        # Walk forward while values are equal (ties get the average rank).
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # Average rank for positions i..j inclusive is (i+1 + j+1) / 2.
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = float(avg_rank)
        i = j + 1
    return [float(r) for r in ranks]  # type: ignore[arg-type]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Standard Pearson correlation. Returns 0.0 for zero-variance inputs."""
    if len(x) != len(y):
        raise ValueError("pearson() requires x and y of equal length")
    n = len(x)
    if n == 0:
        raise ValueError("pearson() requires at least one pair")

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    return cov / denom


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy). Pearson on ranks."""
    return pearson(rank(x), rank(y))


def _default_weights() -> dict[str, float]:
    """Load weights from Settings, falling back to README 3.4 defaults."""
    s = get_settings()
    return {
        "syntax": float(s.weight_syntax),
        "back_translation": float(s.weight_back_translation),
        "sanity": float(s.weight_sanity),
        "multi_query": float(s.weight_multi_query),
        "schema_coverage": float(s.weight_schema_coverage),
    }


def _renormalise(weights: Mapping[str, float]) -> dict[str, float]:
    """Defensive re-normalisation: divide each weight by the total sum.

    Settings defaults sum to 1.0 exactly; this is a safety net for
    env-var overrides that drift. If the sum is non-positive (shouldn't
    happen, but possible via misconfiguration) we fall back to a uniform
    distribution over the supplied keys so the score remains bounded.
    """
    total = sum(weights.values())
    if total <= 0.0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: float(v) / total for k, v in weights.items()}


class HallucinationScorer:
    """Combine per-signal scores into a single weighted confidence score.

    `score(signals)` expects a mapping of signal name → value in [0.0, 1.0]
    (signal semantics documented at module top). Missing known signals
    default to 1.0 (conservative: don't penalise absence of evidence)
    with a note appended.
    """

    def __init__(
        self,
        *,
        weights: Optional[Mapping[str, float]] = None,
        pass_threshold: float = 80.0,
        warn_threshold: float = 50.0,
    ) -> None:
        if pass_threshold < warn_threshold:
            raise ValueError(
                "pass_threshold must be >= warn_threshold "
                f"(got pass={pass_threshold}, warn={warn_threshold})"
            )

        raw = dict(weights) if weights is not None else _default_weights()
        self.weights: dict[str, float] = _renormalise(raw)
        self.pass_threshold = float(pass_threshold)
        self.warn_threshold = float(warn_threshold)

    def score(self, signals: Mapping[str, float]) -> ConfidenceScore:
        """Combine `signals` into a `ConfidenceScore` using `self.weights`.

        - Any canonical signal absent from `signals` defaults to 1.0
          and a note is added (conservative: don't penalise for missing
          evidence, but record that it was missing).
        - Unknown signal names are passed through but ignored when the
          configured weights don't mention them.
        - The final 0-100 score is `100 * sum(weight_i * signal_i)` over
          the configured weight set; weights have already been
          re-normalised to sum to 1.0 in `__init__`.
        - Verdict: "pass" if score >= pass_threshold, "warn" if
          score >= warn_threshold, else "fail". `flagged = (verdict == "fail")`.
        """
        normalised: dict[str, float] = {}
        notes: list[str] = []

        # Fill in defaults for any canonical signal that wasn't supplied.
        for name in _KNOWN_SIGNALS:
            if name in signals:
                v = float(signals[name])
                # Clamp to [0, 1] defensively; warn if we had to clip.
                if v < 0.0:
                    notes.append(f"signal '{name}' was {v}; clamped to 0.0")
                    v = 0.0
                elif v > 1.0:
                    notes.append(f"signal '{name}' was {v}; clamped to 1.0")
                    v = 1.0
                normalised[name] = v
            else:
                normalised[name] = 1.0
                notes.append(
                    f"signal '{name}' missing from input; defaulted to 1.0"
                )

        # Pick up any extra caller-supplied signals (e.g. custom
        # signals) so they show up in the breakdown even if not weighted.
        for name, value in signals.items():
            if name not in normalised:
                v = float(value)
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                normalised[name] = v

        # Weighted sum restricted to signals we have weights for.
        weighted_total = 0.0
        covered_weight = 0.0
        for name, w in self.weights.items():
            if name in normalised:
                weighted_total += w * normalised[name]
                covered_weight += w
        # Defensive: covered_weight can drift from 1.0 if the caller
        # supplied a weights dict that only partially overlaps with
        # supplied signals; rescale by the actually-covered mass.
        if covered_weight > 0.0 and covered_weight < 1.0:
            weighted_total = weighted_total / covered_weight
            notes.append(
                "weights did not cover all supplied signals; "
                f"rescaled by covered fraction {covered_weight:.3f}"
            )

        score_0_100 = 100.0 * weighted_total
        # Clamp into the documented range.
        if score_0_100 < 0.0:
            score_0_100 = 0.0
        elif score_0_100 > 100.0:
            score_0_100 = 100.0

        if score_0_100 >= self.pass_threshold:
            verdict = "pass"
        elif score_0_100 >= self.warn_threshold:
            verdict = "warn"
        else:
            verdict = "fail"
        flagged = verdict == "fail"

        logger.info(
            "hallucination_score.computed",
            extra={
                "score": score_0_100,
                "verdict": verdict,
                "flagged": flagged,
                "signals": normalised,
                "weights": self.weights,
            },
        )

        return ConfidenceScore(
            score=score_0_100,
            verdict=verdict,
            flagged=flagged,
            signal_scores=normalised,
            weights=dict(self.weights),
            notes=notes,
        )
