"""The HTTP surface.

Thin by design, so these tests check the two things a caller depends on: that
the service starts without a fetched corpus, and that a declined answer is
distinguishable from an accepted one *programmatically* rather than by parsing
prose. Any UI or downstream automation needs that distinction.
"""

import pytest
from fastapi.testclient import TestClient

from atoqa import api


@pytest.fixture
def client():
    api.get_index.cache_clear()
    with TestClient(api.app) as c:
        yield c
    api.get_index.cache_clear()


def test_health_reports_a_loaded_corpus(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0, "service must start from the fixture corpus with no fetch"


def test_ask_returns_a_cited_answer(client):
    r = client.post("/ask", json={"question": "What is the GST registration turnover threshold?"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in ("answer", "answer_with_caveat")
    assert "$75,000" in body["text"]
    assert body["sources"]
    assert body["trace_id"]


def test_a_declined_answer_is_machine_readable(client):
    """A caller must not have to read the prose to know it was declined."""
    r = client.post("/ask", json={"question": "Should I set up a trust to reduce my tax?"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "refuse"
    assert body["intent"] == "advice_seeking"


def test_guard_results_are_exposed_to_the_caller(client):
    r = client.post("/ask", json={"question": "What is the instant asset write-off threshold?"})
    body = r.json()
    assert body["verdict"] == "abstain"
    failed = [g["name"] for g in body["guards"] if not g["passed"]]
    assert "question_covered" in failed


@pytest.mark.parametrize("question", ["", "ab", "x" * 501])
def test_malformed_questions_are_rejected(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422
