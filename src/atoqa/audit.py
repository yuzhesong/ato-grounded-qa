"""Append-only decision trace.

One JSON object per request, written to ``data/traces/``. It records what was
asked, what was retrieved and with what scores, which prompt version and model
produced the draft, every guard verdict, and the final disposition -- enough to
reconstruct after the fact why the system said what it said.

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


def write(
    answer: Answer,
    retrieved: list[Retrieved],
    model_name: str,
    directory: Path | None = None,
) -> Path:
    directory = Path(directory or settings.trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "trace_id": answer.trace_id,
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
            }
            for r in retrieved
        ],
        "claims": [c.model_dump() for c in answer.claims],
        "guards": [g.model_dump() for g in answer.guards],
        "verdict": answer.verdict.value,
        "caveats": answer.caveats,
        "latency_ms": round(answer.latency_ms, 2),
    }
    path = directory / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
