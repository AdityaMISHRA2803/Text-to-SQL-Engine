"""Tests for text2sql.scoring — HallucinationScorer + manual Spearman helpers (README 3.4)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from text2sql.config import get_settings
from text2sql.scoring.scorer import (
    ConfidenceScore,
    HallucinationScorer,
    pearson,
    rank,
    spearman,
)


# ===========================================================================
# Group A — helpers (rank / pearson / spearman)
# ===========================================================================


def test_rank_simple_sorted_values() -> None:
    """rank([10, 30, 20]) == [1.0, 3.0, 2.0] (already in value order except middle)."""
    assert rank([10, 30, 20]) == [1.0, 3.0, 2.0]


def test_rank_handles_pair_tie() -> None:
    """rank([10, 10, 20]) == [1.5, 1.5, 3.0] (average rank for the tied pair)."""
    assert rank([10, 10, 20]) == [1.5, 1.5, 3.0]


def test_rank_all_equal_returns_middle() -> None:
    """rank([5, 5, 5]) == [2.0, 2.0, 2.0] (average rank of (1+2+3)/3 = 2)."""
    assert rank([5, 5, 5]) == [2.0, 2.0, 2.0]


def test_pearson_perfect_positive_correlation() -> None:
    """pearson on identical sequences returns 1.0."""
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation() -> None:
    """pearson on perfectly reversed sequences returns -1.0."""
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_constant_y_returns_zero() -> None:
    """Constant y has zero variance → pearson returns 0.0 (denom guard)."""
    assert pearson([1, 2, 3], [2, 2, 2]) == 0.0


def test_spearman_perfect_negative() -> None:
    """spearman on perfectly reversed sequences equals -1.0."""
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_tied_ranks_perfect_positive() -> None:
    """spearman with tied ranks on both sides that agree → 1.0."""
    # Both rank identically to [1.5, 1.5, 3.5, 3.5]; perfect positive.
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


# ===========================================================================
# Group B — ConfidenceScore pydantic model
# ===========================================================================


def test_confidence_score_full_construction() -> None:
    """ConfidenceScore accepts every documented field."""
    cs = ConfidenceScore(
        score=72.5,
        verdict="warn",
        flagged=False,
        signal_scores={"syntax": 1.0, "back_translation": 0.5},
        weights={"syntax": 0.5, "back_translation": 0.5},
        notes=["note 1"],
    )
    assert cs.score == 72.5
    assert cs.verdict == "warn"
    assert cs.flagged is False
    assert cs.signal_scores == {"syntax": 1.0, "back_translation": 0.5}
    assert cs.weights == {"syntax": 0.5, "back_translation": 0.5}
    assert cs.notes == ["note 1"]


def test_confidence_score_score_field_constrained_to_0_100() -> None:
    """Pydantic enforces score in [0.0, 100.0] (Field ge/le)."""
    with pytest.raises(ValidationError):
        ConfidenceScore(
            score=-0.1,
            verdict="fail",
            flagged=True,
            signal_scores={},
            weights={},
        )
    with pytest.raises(ValidationError):
        ConfidenceScore(
            score=100.1,
            verdict="pass",
            flagged=False,
            signal_scores={},
            weights={},
        )


# ===========================================================================
# Group C — HallucinationScorer.score semantics
# ===========================================================================


def test_all_signals_one_hundred_gives_pass() -> None:
    """All signals = 1.0 → score = 100.0 and verdict = 'pass'."""
    result = HallucinationScorer().score(
        {
            "syntax": 1.0,
            "back_translation": 1.0,
            "sanity": 1.0,
            "multi_query": 1.0,
            "schema_coverage": 1.0,
        }
    )
    assert result.score == pytest.approx(100.0)
    assert result.verdict == "pass"
    assert result.flagged is False


def test_all_signals_zero_gives_fail() -> None:
    """All signals = 0.0 → score = 0.0 and verdict = 'fail', flagged=True."""
    result = HallucinationScorer().score(
        {
            "syntax": 0.0,
            "back_translation": 0.0,
            "sanity": 0.0,
            "multi_query": 0.0,
            "schema_coverage": 0.0,
        }
    )
    assert result.score == pytest.approx(0.0)
    assert result.verdict == "fail"
    assert result.flagged is True


def test_mixed_medium_signals_pass_or_warn() -> None:
    """Moderate signals produce either 'pass' or 'warn' (never 'fail')."""
    result = HallucinationScorer().score(
        {
            "syntax": 1.0,
            "back_translation": 0.8,
            "sanity": 0.9,
            "multi_query": 0.7,
            "schema_coverage": 0.85,
        }
    )
    assert result.verdict in ("pass", "warn")
    assert result.flagged is False
    assert result.score >= 50.0  # nothing below warn_threshold


def test_weights_renormalised_to_sum_one() -> None:
    """weights summing to 4 are renormalised to sum to 1.0 in the result."""
    scorer = HallucinationScorer(
        weights={"syntax": 2.0, "back_translation": 2.0},
    )
    result = scorer.score({"syntax": 1.0, "back_translation": 1.0})
    assert sum(result.weights.values()) == pytest.approx(1.0)
    # Re-normalised values: 0.5 / 0.5
    assert result.weights["syntax"] == pytest.approx(0.5)
    assert result.weights["back_translation"] == pytest.approx(0.5)


def test_missing_signal_defaults_to_one_with_note() -> None:
    """A canonical signal missing from input → 1.0 plus a 'missing' note."""
    result = HallucinationScorer().score({"syntax": 1.0})  # others omitted
    assert result.signal_scores["back_translation"] == 1.0
    assert any("back_translation" in n and "missing" in n for n in result.notes)
    # Without the missing-default-to-1.0 behaviour, this score would be
    # ~20 (only syntax=1 weighted 0.2). With it, we get 100.
    assert result.score == pytest.approx(100.0)


def test_custom_pass_warn_thresholds() -> None:
    """Lowering pass_threshold to 60 lets a uniform 0.7 signal set count as 'pass'."""
    scorer = HallucinationScorer(pass_threshold=60.0, warn_threshold=30.0)
    result = scorer.score(
        {
            "syntax": 0.7,
            "back_translation": 0.7,
            "sanity": 0.7,
            "multi_query": 0.7,
            "schema_coverage": 0.7,
        }
    )
    # 100 * 0.7 = 70 → above custom pass_threshold of 60.
    assert result.score == pytest.approx(70.0)
    assert result.verdict == "pass"


# ===========================================================================
# Group D — README exit criterion (Spearman >= 0.6)
# ===========================================================================


def test_readme_exit_criterion_score_correlates_with_correctness() -> None:
    """Synthetic fixture: correct items have higher scores than wrong ones.

    Spearman(correctness, score) must be >= 0.6 to satisfy README 3.4's
    exit criterion for confidence-scoring correlation with correctness.
    """
    correctness = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    score = [0.9, 0.85, 0.3, 0.4, 0.95, 0.2, 0.8, 0.5, 0.92, 0.25]
    rho = spearman(correctness, score)
    assert rho >= 0.6, f"README 3.4 exit criterion failed: Spearman = {rho:.3f}"


# ===========================================================================
# Group E — Settings defaults
# ===========================================================================


def test_settings_default_weights_sum_to_one() -> None:
    """Settings default weights (README 3.4) sum to exactly 1.0."""
    s = get_settings()
    total = (
        float(s.weight_syntax)
        + float(s.weight_back_translation)
        + float(s.weight_sanity)
        + float(s.weight_multi_query)
        + float(s.weight_schema_coverage)
    )
    assert total == pytest.approx(1.0)


# ===========================================================================
# Extra coverage — defensive behaviours (not in the 18, but cheap to add)
# ===========================================================================


def test_spearman_length_mismatch_raises() -> None:
    """spearman raises ValueError on mismatched lengths."""
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_rank_empty_input_raises() -> None:
    """rank raises ValueError on empty input."""
    with pytest.raises(ValueError):
        rank([])


def test_scorer_rejects_inverted_thresholds() -> None:
    """pass_threshold < warn_threshold is rejected at construction time."""
    with pytest.raises(ValueError):
        HallucinationScorer(pass_threshold=10.0, warn_threshold=90.0)
