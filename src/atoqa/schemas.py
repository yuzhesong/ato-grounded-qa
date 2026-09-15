"""Structured-output contracts.

The generation step is constrained to `DraftAnswer`: a list of atomic claims,
each of which MUST carry at least one citation. Everything downstream (the
guards, the evaluation metrics, the audit trace) is defined against this shape,
so a model that returns prose instead of claims fails validation rather than
silently producing an uncheckable answer.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Intent(StrEnum):
    CALCULATION = "calculation"
    GUIDANCE = "guidance"
    ADVICE_SEEKING = "advice_seeking"
    OUT_OF_SCOPE = "out_of_scope"


class Verdict(StrEnum):
    ANSWER = "answer"
    ANSWER_WITH_CAVEAT = "answer_with_caveat"
    ABSTAIN = "abstain"
    REFUSE = "refuse"


class Chunk(BaseModel):
    """A retrievable unit of source text with the provenance needed to cite it."""

    id: str
    text: str
    title: str
    source_url: str
    qc_ref: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    # Income years this chunk is stated to apply to, e.g. ["2025-26"]. Empty
    # means year-agnostic. This is what makes the superseded-rate guard possible.
    income_years: list[str] = Field(default_factory=list)
    retrieved_at: str | None = None

    @property
    def citation(self) -> str:
        bits = [self.title]
        if self.heading_path:
            bits.append(" > ".join(self.heading_path))
        if self.qc_ref:
            bits.append(self.qc_ref)
        return " | ".join(bits)


class Retrieved(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    dense_rank: int | None = None
    lexical_rank: int | None = None


class Claim(BaseModel):
    """One checkable assertion plus the chunk ids that support it."""

    text: str
    citations: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim text must not be empty")
        return v.strip()


class DraftAnswer(BaseModel):
    """What the generator is required to return."""

    claims: list[Claim] = Field(default_factory=list)
    income_year: str | None = None
    # Set by the generator when the retrieved context does not support an answer.
    insufficient_context: bool = False


class GuardResult(BaseModel):
    name: str
    passed: bool
    score: float | None = None
    detail: str = ""


class Answer(BaseModel):
    """The final, audited response."""

    question: str
    verdict: Verdict
    intent: Intent
    income_year: str | None = None
    income_year_assumed: bool = False
    text: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    guards: list[GuardResult] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    trace_id: str = ""
    latency_ms: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @property
    def answered(self) -> bool:
        return self.verdict in (Verdict.ANSWER, Verdict.ANSWER_WITH_CAVEAT)
