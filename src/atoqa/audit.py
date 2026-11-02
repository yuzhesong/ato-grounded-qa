"""Append-only decision trace.

One JSON object per request, written to ``data/traces/``. It records what was
asked, what was retrieved and with what scores, which prompt version and model
produced the draft, every guard verdict, and the final disposition -- enough to
reconstruct after the fact why the system said what it said.

Records go to PostgreSQL when ``ATOQA_DATABASE_URL`` is set (see store.py),
otherwise to the JSONL file. One sink, not both: two copies of an audit trail
are two places for it to disagree. A failed write raises rather than being
swallowed, because an answer that leaves no trace breaks the invariant this
module exists for.

This is not logging for debugging. In a regulated setting the ability to answer
"why did it say that, on that date, from which source" is part of the product,
and it has to be designed in rather than recovered from stdout. The question
text is hashed as well as stored so traces can be correlated after the raw text
is redacted under a retention policy.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import settings
from .schemas import Answer, Retrieved

PROMPT_VERSION = "2026-09-15.a"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def _record(
    answer: Answer,
    retrieved: list[Retrieved],
    model_name: str,
    *,
    index_build_id: int | None = None,
) -> dict:
    return {
        "trace_id": answer.trace_id,
        "index_build_id": index_build_id,
        "at": datetime.now(UTC).isoformat(),
        "question": answer.question,
        "question_sha256": hashlib.sha256(answer.question.encode()).hexdigest(),
        "intent": answer.intent.value,
        "income_year": answer.income_year,
        "income_year_assumed": answer.income_year_assumed,
        "provider": settings.provider,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "thresholds": {
            "min_retrieval_score": settings.min_retrieval_score,
            "min_groundedness": settings.min_groundedness,
        },
        "retrieved": [
            {
                "id": r.chunk.id,
                "score": round(r.score, 4),
                "rank": r.rank,
                "dense_rank": r.dense_rank,
                "lexical_rank": r.lexical_rank,
                "source_url": r.chunk.source_url,
                "income_years": r.chunk.income_years,
                # Chunk ids are positional and get reused by the next re-index;
                # the hash pins down which text this id meant at the time.
                "text_sha256": hashlib.sha256(r.chunk.text.encode()).hexdigest(),
            }
            for r in retrieved
        ],
        "claims": [c.model_dump() for c in answer.claims],
        "guards": [g.model_dump() for g in answer.guards],
        "verdict": answer.verdict.value,
        "caveats": answer.caveats,
        "latency_ms": round(answer.latency_ms, 2),
    }


def write(
    answer: Answer,
    retrieved: list[Retrieved],
    model_name: str,
    directory: Path | None = None,
    *,
    index_build_id: int | None = None,
) -> Path | None:
    """Append the trace; return the JSONL path, or None for Postgres."""
    record = _record(
        answer, retrieved, model_name, index_build_id=index_build_id
    )
    if settings.database_url:
        from .store import get_store  # imported lazily so file-based runs need no driver

        get_store().write_trace(record)
        return None
    directory = Path(directory or settings.trace_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
