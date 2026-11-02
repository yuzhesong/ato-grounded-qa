from pathlib import Path

import pytest

from atoqa.config import settings
from atoqa.index import HybridIndex
from atoqa.ingest import load_markdown_corpus

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def chunks():
    return load_markdown_corpus(FIXTURE_CORPUS)


@pytest.fixture(scope="session")
def index(chunks):
    return HybridIndex(chunks)


@pytest.fixture(autouse=True)
def _file_storage(monkeypatch):
    """Tests use files unless they opt into Postgres explicitly, so a developer's
    ATOQA_DATABASE_URL in .env cannot redirect the suite into their database."""
    monkeypatch.setattr(settings, "database_url", None)
