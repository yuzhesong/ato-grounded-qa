from pathlib import Path

import pytest

from atoqa.index import HybridIndex
from atoqa.ingest import load_markdown_corpus

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def chunks():
    return load_markdown_corpus(FIXTURE_CORPUS)


@pytest.fixture(scope="session")
def index(chunks):
    return HybridIndex(chunks)
