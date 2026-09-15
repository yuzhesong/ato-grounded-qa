"""HTTP surface.

Kept deliberately thin: the interesting behaviour is in `pipeline.ask`, and an
API that duplicated any of the control logic would be a second place for it to
drift. `/ask` returns the verdict and the guard results alongside the text, so
a caller can distinguish "answered" from "declined" programmatically instead of
parsing prose -- which is what any UI or downstream automation actually needs.

Not production-ready, and the README says so: there is no authentication, rate
limiting, tenancy or PII handling here. docs/governance.md lists what would
have to exist before this faced a real user.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import REPO_ROOT, settings
from .index import HybridIndex
from .ingest import load_markdown_corpus, load_raw
from .pipeline import ask
from .schemas import Answer

app = FastAPI(
    title="ATO grounded QA",
    version="0.1.0",
    description="Reference tool over ATO public guidance. Not tax advice.",
)


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    if (Path(settings.index_dir) / "chunks.json").exists():
        return HybridIndex.load()
    try:
        return HybridIndex(load_raw())
    except FileNotFoundError:
        # Fall back to the fixture corpus so the service starts without a fetch.
        return HybridIndex(load_markdown_corpus(REPO_ROOT / "tests" / "fixtures" / "corpus"))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health() -> dict:
    idx = get_index()
    return {"status": "ok", "chunks": len(idx.chunks), "provider": settings.provider}


@app.post("/ask", response_model=Answer)
def ask_endpoint(req: AskRequest) -> Answer:
    try:
        return ask(req.question, index=get_index())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
