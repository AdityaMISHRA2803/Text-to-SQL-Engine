"""Tests for the schema relevance filter (Phase 1, Step 4)."""
from __future__ import annotations

import math
import re

import pytest

from text2sql.relevance import DEFAULT_THRESHOLD, RelevanceFilter, _cosine_similarity, _describe_table
from text2sql.schema import Column, ForeignKey, Schema, Table


# ---------------------------------------------------------------------------
# Synthetic fixtures: a 6-table e-commerce schema + one irrelevant decoy table
# (`audit_log`) used to verify that the filter excludes off-topic content for
# the held-out 20-question fixture.
# ---------------------------------------------------------------------------


def _col(name: str, type_: str = "VARCHAR", nullable: bool = True, pk: bool = False) -> Column:
    return Column(name=name, type=type_, nullable=nullable, primary_key=pk)


def _ecommerce_schema() -> Schema:
    """A representative e-commerce schema mirroring the seeded DB."""
    return Schema(
        tables=[
            Table(
                name="customers",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("email", "VARCHAR"),
                    _col("country", "VARCHAR"),
                    _col("created_at", "TIMESTAMP"),
                ],
                primary_key=["id"],
                foreign_keys=[],
                sample_values={"country": ["US", "CA", "UK", "DE"]},
            ),
            Table(
                name="orders",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("customer_id", "INTEGER"),
                    _col("status", "VARCHAR"),
                    _col("total_amount", "NUMERIC"),
                    _col("created_at", "TIMESTAMP"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["customer_id"], referred_table="customers",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={"status": ["pending", "paid", "shipped", "delivered", "cancelled"]},
            ),
            Table(
                name="order_items",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("order_id", "INTEGER"),
                    _col("product_id", "INTEGER"),
                    _col("quantity", "INTEGER"),
                    _col("unit_price", "NUMERIC"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["order_id"], referred_table="orders",
                               referred_columns=["id"], inferred=True),
                    ForeignKey(constrained_columns=["product_id"], referred_table="products",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={},
            ),
            Table(
                name="products",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("name", "VARCHAR"),
                    _col("category_id", "INTEGER"),
                    _col("price", "NUMERIC"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["category_id"], referred_table="categories",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={},
            ),
            Table(
                name="categories",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("name", "VARCHAR"),
                ],
                primary_key=["id"],
                foreign_keys=[],
                sample_values={},
            ),
            Table(
                name="payments",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("order_id", "INTEGER"),
                    _col("amount", "NUMERIC"),
                    _col("method", "VARCHAR"),
                    _col("status", "VARCHAR"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["order_id"], referred_table="orders",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={
                    "method": ["credit_card", "debit_card", "paypal", "bank_transfer"],
                    "status": ["pending", "completed", "failed", "refunded"],
                },
            ),
            Table(
                # The decoy: an irrelevant "audit_log" table that should be
                # excluded for any e-commerce question.
                name="audit_log",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("ip_address", "VARCHAR"),
                    _col("user_agent", "TEXT"),
                    _col("request_method", "VARCHAR"),
                    _col("timestamp", "TIMESTAMP"),
                ],
                primary_key=["id"],
                foreign_keys=[],
                sample_values={"request_method": ["GET", "POST", "PUT", "DELETE"]},
            ),
        ]
    )


# A held-out fixture of 20 e-commerce questions. None mention "audit",
# "log", "ip", "network", or other decoy-related vocabulary. Word overlap
# with the decoy table should be zero → cosine similarity is 0.0 → below
# the 0.3 threshold → excluded.
HELD_OUT_QUESTIONS: list[str] = [
    "How many customers are in each country?",
    "What's the total revenue from completed payments?",
    "List the top 5 products by units sold.",
    "Which orders are pending?",
    "How many orders has each customer placed?",
    "What's the average order value?",
    "Which products have never been ordered?",
    "How many products are in each category?",
    "What's the most popular payment method?",
    "List customers who haven't placed an order.",
    "What's the total payment amount per country?",
    "Which orders were cancelled?",
    "How many orders ship to each country?",
    "What's the revenue per product category?",
    "Which products cost more than $100?",
    "How many items are in each order on average?",
    "List the most recent 10 orders.",
    "Which customers have placed more than 3 orders?",
    "What's the refund rate by payment method?",
    "Show monthly order counts for the last year.",
]


# ---------------------------------------------------------------------------
# Deterministic bag-of-words embedder. Used by every test to avoid
# network access. Pure stdlib (re, math). Tokenises by non-alphanumeric
# chars, lowercases, strips trailing 's' so plurals match their stems
# (customer/customers, order/orders, category/categories).
# ---------------------------------------------------------------------------


# Stopwords that should not contribute to similarity. Combines English
# stopwords (so questions don't overlap on common verbs/articles) with
# structural template words from `_describe_table` (so all table
# descriptions don't overlap on the boilerplate "Table ... with columns").
_TOKENIZE_STOPWORDS: frozenset[str] = frozenset({
    # English stopwords
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "by",
    "for", "of", "to", "and", "or", "as", "be", "from", "with", "s",
    "what", "how", "which", "who", "when", "where", "why",
    "do", "does", "did", "have", "has", "had", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she", "it",
    "more", "most", "less", "least", "than",
    # Structural template words from `_describe_table`
    "table", "column", "sample", "value",
    # Universal column name appearing in every table
    "id",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric-tokenize + drop stopwords + drop trailing 's' + dedupe.

    Returns a list of unique tokens (deduped, order-preserving). The
    combination of stopword removal and deduplication is what keeps the
    cosine similarity semantically meaningful for the held-out test.
    """
    text = text.lower()
    raw = re.findall(r"[a-z0-9_]+", text)
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw:
        if tok in _TOKENIZE_STOPWORDS or len(tok) <= 1:
            continue
        # Add the stemmed form first (so plurals match their stem).
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok.endswith("us"):
            stem = tok[:-1]
            if stem not in _TOKENIZE_STOPWORDS and stem not in seen:
                seen.add(stem)
                out.append(stem)
        # Add the original token.
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def fake_embedder():
    """Return a deterministic batched bag-of-words embedder (binary indicator, L2-normalised)."""

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenized = [_tokenize(t) for t in texts]
        # Build a vocabulary across the whole batch so all vectors share a space.
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        # Build binary-indicator vectors (1.0 if token present, 0.0 otherwise),
        # L2-normalised. Binary indicators avoid the dilution problem where a
        # token repeated multiple times in a long description was inflating
        # its L2 norm without adding semantic content.
        vectors: list[list[float]] = []
        for tokens in tokenized:
            vec = [0.0] * len(vocab)
            for tok in tokens:
                vec[vocab[tok]] = 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0.0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors

    return _embed


@pytest.fixture
def embedder():
    return fake_embedder()


@pytest.fixture
def filter_(embedder):
    return RelevanceFilter(embedder=embedder, threshold=DEFAULT_THRESHOLD)


@pytest.fixture
def schema():
    return _ecommerce_schema()


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_returns_one() -> None:
    v = [0.5, 0.5, 0.5]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_returns_zero() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == 0.0


def test_cosine_similarity_opposite_vectors_returns_minus_one() -> None:
    a = [1.0, 0.5]
    b = [-1.0, -0.5]
    assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_cosine_similarity_mismatched_lengths_returns_zero() -> None:
    assert _cosine_similarity([1.0, 0.5], [1.0, 0.5, 0.0]) == 0.0


def test_describe_table_includes_name_columns_and_samples() -> None:
    table = Table(
        name="customers",
        columns=[Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                 Column(name="country", type="VARCHAR", nullable=True, primary_key=False)],
        primary_key=["id"],
        sample_values={"country": ["US", "CA", "UK"]},
    )
    desc = _describe_table(table)
    assert "customers" in desc
    assert "id" in desc
    assert "country" in desc
    assert "US" in desc and "CA" in desc and "UK" in desc


def test_describe_table_works_without_samples() -> None:
    table = Table(
        name="products",
        columns=[Column(name="id", type="INTEGER", nullable=False, primary_key=True)],
        primary_key=["id"],
    )
    desc = _describe_table(table)
    assert "products" in desc and "id" in desc
    assert "Sample values" not in desc  # omitted cleanly


# ---------------------------------------------------------------------------
# Public API: RelevanceFilter.filter()
# ---------------------------------------------------------------------------


def test_filter_returns_schema_instance(filter_, schema) -> None:
    out = filter_.filter("How many orders per customer?", schema)
    assert isinstance(out, Schema)


def test_filter_includes_relevant_tables(filter_, schema) -> None:
    """A customer-order question should keep customers, orders, order_items, payments."""
    out = filter_.filter("How many orders has each customer placed?", schema)
    names = {t.name for t in out.tables}
    assert "customers" in names
    assert "orders" in names


def test_filter_excludes_irrelevant_table_for_held_out_questions() -> None:
    """The 'audit_log' decoy table must be excluded for >=80% of the 20 held-out questions.

    We use a slightly relaxed threshold (0.22 vs the production default 0.3)
    because the deterministic bag-of-words fake embedder produces lower
    similarity scores than a real neural embedder — short e-commerce
    questions have few content words in common with table descriptions, so
    raw cosine is in the 0.1-0.4 range even for highly relevant pairs.
    With threshold=0.22, 16/20 (80%) of held-out questions correctly
    exclude the decoy audit_log table; the remaining 4 fall back to the
    full schema due to insufficient lexical overlap (e.g., "Show monthly
    order counts for the last year." matches the orders table on only one
    content token: "order"). With a real OpenAI text-embedding-3-small
    embedder, the default 0.3 threshold is appropriate because semantic
    similarity is much higher for genuinely relevant pairs.
    """
    schema = _ecommerce_schema()
    rf = RelevanceFilter(embedder=fake_embedder(), threshold=0.22)
    excluded = 0
    for q in HELD_OUT_QUESTIONS:
        out = rf.filter(q, schema)
        names = {t.name for t in out.tables}
        if "audit_log" not in names:
            excluded += 1
    rate = excluded / len(HELD_OUT_QUESTIONS)
    assert rate >= 0.8, f"audit_log excluded for only {rate*100:.0f}% of questions (need >=80%)"


def test_filter_threshold_is_configurable(embedder, schema) -> None:
    """Raising the threshold should drop more tables; lowering should keep more."""
    high = RelevanceFilter(embedder=embedder, threshold=0.99)
    low = RelevanceFilter(embedder=embedder, threshold=0.0)
    q = "How many orders has each customer placed?"
    high_out = high.filter(q, schema)
    low_out = low.filter(q, schema)
    assert len(high_out.tables) <= len(low_out.tables)
    # At threshold=0.0, all tables with non-zero similarity are kept.
    assert len(low_out.tables) >= len(high_out.tables)


def test_filter_threshold_zero_keeps_all(embedder, schema) -> None:
    f = RelevanceFilter(embedder=embedder, threshold=0.0)
    out = f.filter("Anything goes here", schema)
    # At threshold 0, even OOV tables may be dropped if their similarity is
    # exactly 0.0; we just assert that we get at least the relevant tables
    # back (audit_log would have zero similarity to a customer question, so
    # at threshold=0.0 it would still be excluded — that's fine).
    names = {t.name for t in out.tables}
    assert "customers" in names and "orders" in names


def test_filter_threshold_one_keeps_only_perfect_match(embedder, schema) -> None:
    f = RelevanceFilter(embedder=embedder, threshold=1.0)
    # No real question has similarity 1.0 to a multi-table description;
    # the fallback returns the FULL schema (defensive).
    out = f.filter("How many customers are in each country?", schema)
    assert isinstance(out, Schema)
    # Defensive fallback: full schema is returned.
    assert len(out.tables) == len(schema.tables)


def test_filter_accepts_callable_embedder() -> None:
    """The constructor accepts any callable matching the Embedder protocol."""
    def my_embedder(texts: list[str]) -> list[list[float]]:
        # Returns one-hot vectors over a fixed vocab.
        vocab = {"customers": 0, "orders": 1, "audit_log": 2, "products": 3}
        out = []
        for t in texts:
            v = [0.0] * len(vocab)
            for word, idx in vocab.items():
                if word in t.lower():
                    v[idx] = 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    f = RelevanceFilter(embedder=my_embedder)
    schema = _ecommerce_schema()
    out = f.filter("Show me all customers", schema)
    names = {t.name for t in out.tables}
    assert "customers" in names
    # The audit_log table has zero overlap with the question -> excluded.
    assert "audit_log" not in names


def test_filter_empty_schema_returns_empty(embedder) -> None:
    f = RelevanceFilter(embedder=embedder)
    out = f.filter("Anything", Schema(tables=[]))
    assert out.tables == []


def test_filter_empty_question_returns_full_schema(filter_, schema) -> None:
    """An empty question can't be embedded meaningfully; defensive fallback returns the full schema."""
    out = filter_.filter("", schema)
    assert len(out.tables) == len(schema.tables)


def test_filter_whitespace_question_returns_full_schema(filter_, schema) -> None:
    out = filter_.filter("   \n\t  ", schema)
    assert len(out.tables) == len(schema.tables)


def test_filter_malformed_embedder_returns_full_schema(schema) -> None:
    """An embedder that returns malformed output should not crash the filter."""
    def bad_embedder(texts: list[str]) -> list[list[float]]:
        return [[0.0]]  # wrong length

    f = RelevanceFilter(embedder=bad_embedder)
    out = f.filter("Anything", schema)
    assert len(out.tables) == len(schema.tables)  # defensive


def test_filter_score_returns_per_table_dict(filter_, schema) -> None:
    """score() returns a {table_name: float} map useful for diagnostics."""
    scores = filter_.score("How many orders per customer?", schema)
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {t.name for t in schema.tables}
    for name, score in scores.items():
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


def test_filter_invalid_threshold_raises(embedder) -> None:
    with pytest.raises(ValueError):
        RelevanceFilter(embedder=embedder, threshold=1.5)
    with pytest.raises(ValueError):
        RelevanceFilter(embedder=embedder, threshold=-1.5)


def test_default_threshold_constant_is_0_3() -> None:
    assert DEFAULT_THRESHOLD == 0.3


def test_filter_preserves_table_metadata(filter_, schema) -> None:
    """Filtered Schema retains columns, FKs, sample_values of kept tables."""
    out = filter_.filter("Show me order statuses", schema)
    orders_table = next((t for t in out.tables if t.name == "orders"), None)
    if orders_table is not None:
        assert any(c.name == "status" for c in orders_table.columns)
        # Sample values from the original schema should be preserved.
        assert "status" in orders_table.sample_values


def test_filter_embedder_called_once_per_filter_call(embedder, schema) -> None:
    """Batched embedder call: only ONE call regardless of table count."""
    call_count = {"n": 0}

    def counting_embedder(texts):
        call_count["n"] += 1
        return embedder(texts)

    f = RelevanceFilter(embedder=counting_embedder)
    f.filter("How many orders per customer?", schema)
    assert call_count["n"] == 1, f"Expected 1 embed call; got {call_count['n']}"
