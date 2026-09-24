"""Prompt construction for SQL generation.

Composes a single prompt string from a schema (filtered tables + FK graph +
sample values), few-shot Q→SQL examples, an optional glossary of domain
terms, and the target user question. Hard-caps the prompt at a configurable
token estimate (default 6000) using pure-stdlib estimation (tiktoken if
available, otherwise a `len(text) // 4` fallback). No live LLM call.

Truncation policy (when token cap is exceeded):
  1. Drop few-shot examples (least load-bearing; the LLM can still answer).
  2. Drop glossary (rarely critical; only refines wording).
  3. Drop the FK-graph section (mostly redundant with the schema section).
  4. Drop sample-value lists from the schema section (reduce noise, keep
     column types so the LLM still knows the structure).
  5. Drop tables from the schema one by one (largest first) until the
     prompt fits. This handles absurdly small caps where the bare schema
     exceeds the budget.
  6. If the prompt is still over cap after all five trims, return what we
     have with a truncation marker `<!-- truncated: ... -->` so downstream
     callers can see what was cut.

The constructor never raises on token cap; it always returns a string that
fits within `token_cap` tokens (or as close as the schema allows).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel

from text2sql.schema import Schema


# Path to the bundled few-shot examples store (relative to this file).
DEFAULT_FEW_SHOTS_PATH = Path(__file__).parent / "few_shots.json"


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text`.

    Uses tiktoken's `cl100k_base` encoder if available (the same tokenizer
    used by OpenAI's gpt-3.5/gpt-4 models). Falls back to `len(text) // 4`
    when tiktoken is not installed. The fallback is conservative — actual
    tokenizers typically yield slightly fewer tokens per character, so the
    fallback may slightly under-budget; this is acceptable for our use case
    where the cap is a soft guard against runaway prompts.

    Note: Anthropic Claude uses a different tokenizer with ~10-15% variance
    from tiktoken's count. We accept this imprecision for budgeting.
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Conservative fallback: 4 characters per token. This is the upper
        # bound for most BPE tokenizers on natural-language text.
        return (len(text) + 3) // 4


class FewShot(BaseModel):
    """A single few-shot example: a natural-language question and its SQL answer."""

    question: str
    sql: str
    description: Optional[str] = None


class PromptBuilder:
    """Assembles a schema-aware prompt for an LLM to generate SQL.

    Defaults are tuned for the e-commerce seed schema (6 small tables,
    ~30 columns total) — a full prompt with 4 few-shots should sit around
    1.5-2.5K tokens, well under the 6000-token default cap.
    """

    DEFAULT_TOKEN_CAP = 6000
    DEFAULT_NUM_FEW_SHOTS = 4

    def __init__(
        self,
        few_shots: Optional[Sequence[FewShot]] = None,
        *,
        token_cap: int = DEFAULT_TOKEN_CAP,
        num_few_shots: int = DEFAULT_NUM_FEW_SHOTS,
        glossary: Optional[dict[str, str]] = None,
    ) -> None:
        # Track whether the caller passed an explicit `few_shots` value
        # (including an empty list) so we can distinguish "user wants no
        # examples" (render the `## Examples\n(none provided)` placeholder)
        # from "no examples available in the store" (skip the section).
        self._few_shots_explicit = few_shots is not None
        if few_shots is not None:
            self.few_shots: list[FewShot] = list(few_shots)
        else:
            self.few_shots = self._load_default_few_shots()
        self.token_cap = token_cap
        self.num_few_shots = num_few_shots
        self.glossary: dict[str, str] = dict(glossary or {})

    # ----- few-shot loading / selection -----

    @staticmethod
    def _load_default_few_shots() -> list[FewShot]:
        """Load the bundled few_shots.json if present; return empty list otherwise."""
        if not DEFAULT_FEW_SHOTS_PATH.exists():
            return []
        try:
            data = json.loads(DEFAULT_FEW_SHOTS_PATH.read_text(encoding="utf-8"))
            return [FewShot(**item) for item in data]
        except Exception:
            return []

    def _select_few_shots(self, question: str) -> list[FewShot]:
        """Pick up to `num_few_shots` examples, preferring ones whose question shares keywords with the user's question.

        Scoring: count of unique lowercase word overlap. Ties broken by
        original order. If the question is empty, the first `num_few_shots`
        examples are returned in their stored order.
        """
        if not self.few_shots:
            return []
        if not question:
            return self.few_shots[: self.num_few_shots]
        q_words = {w for w in question.lower().split() if len(w) > 2}

        def score(fs: FewShot) -> int:
            fs_words = {w for w in fs.question.lower().split() if len(w) > 2}
            return len(q_words & fs_words)

        scored = sorted(
            enumerate(self.few_shots),
            key=lambda pair: (-score(pair[1]), pair[0]),  # higher overlap first; ties: original order
        )
        return [fs for _, fs in scored[: self.num_few_shots]]

    # ----- section renderers -----

    def _render_glossary(self, glossary: dict[str, str]) -> str:
        if not glossary:
            return ""
        parts = ["## Glossary", ""]
        for term, defn in glossary.items():
            parts.append(f"- **{term}**: {defn}")
        return "\n".join(parts) + "\n"

    def _render_schema(self, schema: Schema, *, include_samples: bool = True) -> str:
        parts = ["## Schema", ""]
        for table in schema.tables:
            parts.append(self._render_schema_single(table, include_samples=include_samples))
        return "\n".join(parts).rstrip() + "\n"

    def _render_schema_single(self, table, *, include_samples: bool = True) -> str:
        """Render a single Table to markdown-ish text (used by `_render_schema` and by the truncation logic to size tables)."""
        parts = [f"### {table.name}"]
        for col in table.columns:
            null = " NULL" if col.nullable else " NOT NULL"
            pk = " PRIMARY KEY" if col.primary_key else ""
            parts.append(f"  - {col.name}: {col.type}{null}{pk}")
        if table.foreign_keys:
            for fk in table.foreign_keys:
                src = ",".join(fk.constrained_columns)
                dst = f"{fk.referred_table}({','.join(fk.referred_columns)})"
                tag = " (inferred)" if fk.inferred else ""
                parts.append(f"  - FK: {src} -> {dst}{tag}")
        if include_samples and table.sample_values:
            for col_name, vals in table.sample_values.items():
                parts.append(f"  - samples {col_name}: {vals}")
        parts.append("")
        return "\n".join(parts)

    def _render_fk_graph(self, schema: Schema) -> str:
        has_any = any(table.foreign_keys for table in schema.tables)
        if not has_any:
            return "## Foreign Key Graph\n(none)\n"
        parts = ["## Foreign Key Graph", ""]
        for table in schema.tables:
            for fk in table.foreign_keys:
                src = f"{table.name}.{','.join(fk.constrained_columns)}"
                dst = f"{fk.referred_table}.{','.join(fk.referred_columns)}"
                tag = " (inferred)" if fk.inferred else ""
                parts.append(f"- {src} -> {dst}{tag}")
        return "\n".join(parts) + "\n"

    def _render_few_shots(self, few_shots: Sequence[FewShot]) -> str:
        if not few_shots:
            return "## Examples\n(none provided)\n"
        parts = ["## Examples", ""]
        for i, fs in enumerate(few_shots, start=1):
            parts.append(f"### Example {i}")
            parts.append(f"Q: {fs.question}")
            parts.append("SQL:")
            parts.append("```sql")
            parts.append(fs.sql)
            parts.append("```")
            if fs.description:
                parts.append(f"({fs.description})")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    # ----- assembly + truncation -----

    def _assemble(
        self,
        question: str,
        schema: Schema,
        selected: Sequence[FewShot],
        glossary: dict[str, str],
        *,
        include_few_shots: bool,
        include_glossary: bool,
        include_fk_graph: bool,
        include_samples: bool,
    ) -> str:
        system = (
            "You are an expert PostgreSQL generator. Given the schema below and "
            "the user's question, produce exactly one SQL query that answers the "
            "question. Output ONLY the SQL inside a single ```sql``` block; do "
            "not include prose, comments, or multiple statements."
        )
        parts: list[str] = [system, ""]
        g = self._render_glossary(glossary)
        if include_glossary and g:
            parts.append(g.rstrip())
            parts.append("")
        parts.append(self._render_schema(schema, include_samples=include_samples).rstrip())
        parts.append("")
        if include_fk_graph:
            parts.append(self._render_fk_graph(schema).rstrip())
            parts.append("")
        if include_few_shots:
            fs_text = self._render_few_shots(selected).rstrip()
            parts.append(fs_text)
            parts.append("")
        parts.append(f"## Question\n{question}")
        return "\n".join(parts).rstrip() + "\n"

    def _with_marker(self, prompt: str, dropped: list[str]) -> str:
        if not dropped:
            return prompt
        marker = f"\n<!-- truncated: {', '.join(dropped)} -->\n"
        return prompt.rstrip() + "\n" + marker

    def build(
        self,
        question: str,
        schema: Schema,
        few_shots: Optional[Sequence[FewShot]] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> str:
        """Assemble a prompt string from schema + few-shots + question.

        Args:
            question: The natural-language question from the user.
            schema: A `Schema` model (from `text2sql.schema.extract_schema`).
            few_shots: Optional override list of few-shot examples. If None,
                picks up to `num_few_shots` from the constructor's store,
                scored by keyword overlap with the question.
            glossary: Optional override {term: definition} map. If None,
                uses the constructor's glossary.

        Returns:
            A single prompt string at or under `token_cap` tokens. If the
            full prompt exceeds the cap, optional sections are dropped in
            priority order (see module docstring); a truncation marker is
            appended listing what was dropped.
        """
        if glossary is None:
            glossary = self.glossary
        if few_shots is not None:
            selected = list(few_shots)
            user_wants_few_shots_section = True  # explicit (even if empty)
        elif self._few_shots_explicit:
            selected = list(self.few_shots)
            user_wants_few_shots_section = True  # explicit (even if empty)
        else:
            selected = self._select_few_shots(question)
            user_wants_few_shots_section = bool(selected)

        # Track which sections are currently included. We mutate these as
        # the truncation cascade progresses so that later re-assemblies don't
        # accidentally re-introduce a section we already dropped.
        include_few_shots = user_wants_few_shots_section
        include_glossary = bool(glossary)
        include_fk_graph = True
        include_samples = True
        current_schema = schema
        dropped: list[str] = []

        # 1) Try the full prompt with everything.
        prompt = self._assemble(
            question, current_schema, selected, glossary,
            include_few_shots=include_few_shots,
            include_glossary=include_glossary,
            include_fk_graph=include_fk_graph,
            include_samples=include_samples,
        )
        if estimate_tokens(prompt) <= self.token_cap:
            return prompt

        # 2) Drop few-shots.
        if include_few_shots:
            include_few_shots = False
            dropped.append("few_shots")
            prompt = self._assemble(
                question, current_schema, selected, glossary,
                include_few_shots=include_few_shots,
                include_glossary=include_glossary,
                include_fk_graph=include_fk_graph,
                include_samples=include_samples,
            )
            if estimate_tokens(prompt) <= self.token_cap:
                return self._with_marker(prompt, dropped)

        # 3) Drop glossary.
        if include_glossary:
            include_glossary = False
            dropped.append("glossary")
            prompt = self._assemble(
                question, current_schema, selected, {},
                include_few_shots=include_few_shots,
                include_glossary=include_glossary,
                include_fk_graph=include_fk_graph,
                include_samples=include_samples,
            )
            if estimate_tokens(prompt) <= self.token_cap:
                return self._with_marker(prompt, dropped)

        # 4) Drop FK graph.
        if include_fk_graph:
            include_fk_graph = False
            dropped.append("fk_graph")
            prompt = self._assemble(
                question, current_schema, selected, {},
                include_few_shots=include_few_shots,
                include_glossary=include_glossary,
                include_fk_graph=include_fk_graph,
                include_samples=include_samples,
            )
            if estimate_tokens(prompt) <= self.token_cap:
                return self._with_marker(prompt, dropped)

        # 5) Drop sample values from the schema section.
        if include_samples:
            include_samples = False
            dropped.append("sample_values")
            prompt = self._assemble(
                question, current_schema, selected, {},
                include_few_shots=include_few_shots,
                include_glossary=include_glossary,
                include_fk_graph=include_fk_graph,
                include_samples=include_samples,
            )
            if estimate_tokens(prompt) <= self.token_cap:
                return self._with_marker(prompt, dropped)

        # 6) Last resort: drop tables from the schema one by one (largest
        # first) until the prompt fits. This handles absurdly small caps
        # where even the bare schema exceeds the budget. If we end up with
        # an empty schema we still emit a `## Schema` header so the LLM
        # can see that no schema was provided.
        if current_schema.tables:
            dropped.append("tables")
            # Build a list sorted largest-first so we drop the biggest
            # contributors first; the cap will be reached in fewer steps.
            remaining = sorted(
                list(current_schema.tables),
                key=lambda t: len(self._render_schema_single(t)),
                reverse=True,
            )
            current_schema = Schema(tables=remaining)
            prompt = self._assemble(
                question, current_schema, selected, {},
                include_few_shots=include_few_shots,
                include_glossary=include_glossary,
                include_fk_graph=include_fk_graph,
                include_samples=include_samples,
            )
            while current_schema.tables and estimate_tokens(prompt) > self.token_cap:
                remaining = remaining[:-1]
                current_schema = Schema(tables=remaining)
                prompt = self._assemble(
                    question, current_schema, selected, {},
                    include_few_shots=include_few_shots,
                    include_glossary=include_glossary,
                    include_fk_graph=include_fk_graph,
                    include_samples=include_samples,
                )
        return self._with_marker(prompt, dropped)


__all__ = ["FewShot", "PromptBuilder", "estimate_tokens", "DEFAULT_FEW_SHOTS_PATH"]
