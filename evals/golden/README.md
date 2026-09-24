# Golden Dataset

This directory contains the canonical text-to-SQL test cases used by `python -m evals.run`.
Every case is a natural-language question paired with verified-correct SQL and an expected
result snapshot (row count + columns).

## Schema

Each case is a JSON object:

```json
{
  "id": "string, unique slug",
  "category": "simple_lookup | filter | join | aggregation | date_range | ambiguous | unanswerable | dangerous",
  "question": "natural-language question",
  "golden_sql": "the verified-correct SQL string",
  "expect": {
    "row_count": "integer — exact expected row count (after execution)",
    "columns": ["list", "of", "column", "names"]
  },
  "should_be_flagged": "boolean — true iff the hallucination detector should flag this as low-confidence",
  "difficulty": "easy | medium | hard",
  "notes": "human-readable explanation / known caveats"
}
```

## Adding a new case

1. Pick the right category file (or create one). Naming convention: `NN_category.json`.
2. Choose a unique `id` slug (kebab-case, descriptive).
3. Write `golden_sql` against the [seeded schema](../../text2sql/seed_dataset.py):
   - `categories(id, name, slug)`
   - `customers(id, email, first_name, last_name, country, created_at)`
   - `products(id, name, description, price, category_id, stock, created_at)`
   - `orders(id, customer_id, status, total, created_at, shipped_at)`
   - `order_items(id, order_id, product_id, quantity, unit_price)`
   - `payments(id, order_id, amount, method, status, created_at)`
4. Run `python -m text2sql.seed_dataset` against a fresh Postgres to capture `row_count` and `columns`.
5. Add a short `notes` field with anything reviewers should know.

## Categories

| File | Coverage |
|---|---|
| `01_simple_lookup.json` | `SELECT ... WHERE id = ?`, basic `COUNT(*)` |
| `02_filter.json` | Single-column WHERE, `IN`, `BETWEEN` |
| `03_join.json` | 2- and 3-table JOINs |
| `04_aggregation.json` | `GROUP BY`, `HAVING`, `SUM`/`AVG`/`COUNT` |
| `05_date_range.json` | `WHERE created_at >= ...` filters |
| `06_ambiguous.json` | Questions with multiple plausible SQL interpretations |
| `07_unanswerable.json` | Questions the schema cannot answer |
| `08_dangerous.json` | Destructive operations the guardrail must block |

## Coverage targets

- ≥50 cases total
- ≥5 cases per category
- Mixed difficulties
