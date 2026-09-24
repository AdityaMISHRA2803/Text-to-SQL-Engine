"""Evaluation suite for the text2sql pipeline (Phase 5).

Public submodules:
    metrics        — SQL normalization + result-set comparison primitives
    completion_fns — stub (offline) and live (LLM) completion_fn factories
    runner         — CLI entry: `python -m evals.run`
    report         — JSON + Markdown report writers
"""
