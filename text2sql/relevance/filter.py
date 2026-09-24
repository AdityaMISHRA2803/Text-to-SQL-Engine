"""Schema relevance filter — narrows a full `Schema` to only the tables
likely needed to answer the user's question.

Pipeline:
  1. Build a one-line description per table: name + columns + sample
     categorical values (so the embedder sees both structure and content).
  2. Embed the user question + all table descriptions in ONE batched
     call (cheaper than N+1 calls and ensures all vectors share a
     vocabulary).
  3. Compute cosine similarity between the question vector and each
     table-description vector.
  4. Keep tables whose similarity is at or above the configurable
     threshold (default 0.3).
  5. Defensive fallback: if NO tables meet the threshold (e.g. an
     off-topic question, an OOV question), return the full schema
     unchanged. Returning an empty schema would break downstream prompt
     construction; over-inclusion is safer than under-inclusion.

The embedder is dependency-injected via the constructor so unit tests
can substitute a deterministic bag-of-words embedder and run without
network access or paid API keys.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from text2sql.llm.embedder import Embedder
from text2sql.schema import Schema, Table


#: Default similarity threshold. Tuned empirically for OpenAI
#: text-embedding-3-small against the e-commerce seed schema; works
#: equally well for the bag-of-words fake embedder used in tests
#: because we scale similarity to [0, 1].
DEFAULT_THRESHOLD = 0.3


def _describe_table(table: Table) -> str:
    """Render a table as a single descriptive string for embedding.

    Format: "Table <name> with columns <col1>, <col2>, ... . Sample values: <col>: [v1, v2, ...]; ..."
    """
    cols = ", ".join(c.name for c in table.columns)
    if table.sample_values:
        sample_parts = []
        for col_name, vals in table.sample_values.items():
            sample_parts.append(f"{col_name}: {vals}")
        samples = ". Sample values: " + "; ".join(sample_parts)
    else:
        samples = ""
    return f"Table {table.name} with columns {cols}{samples}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-stdlib cosine similarity. Returns 0.0 if either vector has zero norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class RelevanceFilter:
    """Filters a Schema to only the tables relevant to a user question.

    The embedder is dependency-injected. Default threshold is 0.3; the
    threshold is the cosine-similarity floor — tables strictly below
    the threshold are dropped.

    If no tables meet the threshold, the original Schema is returned
    unchanged (defensive — better to over-include than to hand an empty
    schema to the downstream prompt builder).
    """

    def __init__(self, embedder: Embedder, *, threshold: float = DEFAULT_THRESHOLD) -> None:
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [-1, 1]; got {threshold}")
        self.embedder = embedder
        self.threshold = threshold

    def filter(self, question: str, schema: Schema) -> Schema:
        """Return a Schema containing only tables whose description is similar enough to `question`.

        Args:
            question: The user's natural-language question.
            schema: The full Schema (e.g. from `extract_schema`).

        Returns:
            A Schema with a subset of the input tables (or the input
            Schema itself if no tables meet the threshold, or if the
            input Schema is empty).
        """
        if not schema.tables:
            return schema
        if not question or not question.strip():
            return schema

        descriptions = [_describe_table(t) for t in schema.tables]
        # One batched embed call — ensures all vectors share a vocabulary.
        vectors = self.embedder([question] + descriptions)
        if not vectors or len(vectors) != len(descriptions) + 1:
            # Embedder returned malformed output; be defensive and return full schema.
            return schema

        question_vec = vectors[0]
        table_vecs = vectors[1:]
        similarities = [_cosine_similarity(question_vec, tv) for tv in table_vecs]
        kept = [
            t for t, sim in zip(schema.tables, similarities)
            if sim >= self.threshold
        ]
        if not kept:
            return schema  # Defensive fallback: no tables met threshold.
        return Schema(tables=kept)

    def score(self, question: str, schema: Schema) -> dict[str, float]:
        """Return a {table_name: similarity_score} map. Useful for diagnostics and tests."""
        if not schema.tables:
            return {}
        descriptions = [_describe_table(t) for t in schema.tables]
        vectors = self.embedder([question] + descriptions)
        if not vectors or len(vectors) != len(descriptions) + 1:
            return {t.name: 0.0 for t in schema.tables}
        question_vec = vectors[0]
        table_vecs = vectors[1:]
        return {
            t.name: _cosine_similarity(question_vec, tv)
            for t, tv in zip(schema.tables, table_vecs)
        }


__all__ = ["DEFAULT_THRESHOLD", "RelevanceFilter", "_describe_table", "_cosine_similarity"]
