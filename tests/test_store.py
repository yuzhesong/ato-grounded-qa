"""PostgreSQL storage.

Skipped unless ATOQA_TEST_DATABASE_URL names a database the suite may write
to, so `make test` stays offline. Each test gets its own throwaway schema;
nothing outside it is touched. CI runs these against a service container.
"""

import os
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

URL = os.environ.get("ATOQA_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="ATOQA_TEST_DATABASE_URL not set")
psycopg = pytest.importorskip("psycopg")

from psycopg.conninfo import make_conninfo  # noqa: E402

from atoqa import api  # noqa: E402
from atoqa import store as store_mod  # noqa: E402
from atoqa.cli import main  # noqa: E402
from atoqa.config import settings  # noqa: E402
from atoqa.embed import HashingEmbedder  # noqa: E402
from atoqa.index import HybridIndex  # noqa: E402
from atoqa.llm import ExtractiveStub  # noqa: E402
from atoqa.pipeline import ask  # noqa: E402

QUESTIONS = ("What is the GST registration turnover threshold?",
             "Who won the 1998 AFL grand final?",
             "What is the instant asset write-off threshold?")


@pytest.fixture
def conninfo():
    schema = f"atoqa_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    yield make_conninfo(URL, options=f"-c search_path={schema}")
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture
def store(conninfo):
    s = store_mod.Store(conninfo)
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def use_postgres(store, conninfo, monkeypatch):
    """Route the application's own storage calls to the test schema."""
    monkeypatch.setattr(settings, "database_url", conninfo)
    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    return store


@pytest.fixture
def pg_index(chunks, use_postgres):
    use_postgres.save_index(HybridIndex(chunks))
    return use_postgres.load_index()


@pytest.fixture
def db_client(use_postgres):
    api.get_index.cache_clear()
    try:
        with TestClient(api.app) as client:
            yield client
    finally:
        api.get_index.cache_clear()


def sql(conninfo, query, params=()):
    with psycopg.connect(conninfo) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else []


# --- index -------------------------------------------------------------------

def test_index_round_trips_without_changing_a_score(index, store):
    """The abstention gate is a threshold on the calibrated score, so storage
    must return identical scores, not merely the same ordering."""
    store.save_index(index)
    loaded = store.load_index()
    assert [c.model_dump() for c in loaded.chunks] == [c.model_dump() for c in index.chunks]
    assert np.array_equal(loaded.vectors, index.vectors)
    for q in ("GST threshold", "super guarantee rate", "how do I bake sourdough bread"):
        before, after = index.search(q), loaded.search(q)
        assert [(r.chunk.id, r.score) for r in before] == [(r.chunk.id, r.score) for r in after]


def test_loading_takes_the_latest_build_and_keeps_earlier_ones(chunks, store, conninfo):
    store.save_index(HybridIndex(chunks))
    second = store.save_index(HybridIndex(chunks[:3]))
    assert len(store.load_index().chunks) == 3
    assert sql(conninfo, "SELECT max(id), count(*) FROM index_builds") == [(second, 2)]


def test_vectors_from_another_embedder_are_refused(chunks, store):
    store.save_index(HybridIndex(chunks, embedder=HashingEmbedder(dim=128)))
    with pytest.raises(RuntimeError, match="Re-run `atoqa index`"):
        store.load_index()


def test_an_empty_database_says_how_to_fix_it(store):
    with pytest.raises(LookupError, match="atoqa index"):
        store.load_index()


# --- audit trace -------------------------------------------------------------

def test_every_decision_writes_a_trace_row(pg_index, use_postgres, conninfo):
    for q in QUESTIONS:
        ask(q, index=pg_index, llm=ExtractiveStub(), trace=True)

    rows = sql(
        conninfo,
        "SELECT verdict, question, question_sha256, record FROM traces",
    )
    assert {r[0] for r in rows} == {"answer", "refuse", "abstain"}

    for _, question, sha, record in rows:
        assert question and sha and record["thresholds"]
        assert "question" not in record, "redaction must have only one column to clear"


def test_a_trace_identifies_the_exact_text_it_was_answered_from(chunks, use_postgres, conninfo):
    """Chunk ids are positional, so after a re-index the same id can name
    different text, and the trace's text hash tells the two apart. The hash
    cannot separate builds with identical text but different metadata; the
    trace's index_build_id does that (see
    test_trace_pins_the_loaded_build_after_metadata_changes)."""
    use_postgres.save_index(HybridIndex(chunks))
    a = ask(QUESTIONS[0], index=use_postgres.load_index(), llm=ExtractiveStub(), trace=True)
    ids = [c.id for c in chunks]
    shifted = [c.model_copy(update={"id": ids[(i + 1) % len(ids)]}) for i, c in enumerate(chunks)]
    use_postgres.save_index(HybridIndex(shifted))            # same ids, different text

    (record,) = sql(conninfo, "SELECT record FROM traces WHERE trace_id = %s", (a.trace_id,))[0]
    # calc: passages are computations, not stored chunks.
    passages = [r for r in record["retrieved"] if not r["id"].startswith("calc:")]
    assert passages
    for p in passages:
        builds = sql(conninfo, "SELECT build_id FROM chunks WHERE id = %s AND text_sha256 = %s",
                     (p["id"], p["text_sha256"]))
        assert builds == [(1,)], f"{p['id']} should resolve to the build it was answered from"


def test_traces_cannot_be_deleted_rewritten_or_truncated(pg_index, use_postgres, conninfo):
    ask(QUESTIONS[0], index=pg_index, llm=ExtractiveStub(), trace=True)

    for stmt in (
        "DELETE FROM traces",
        "UPDATE traces SET verdict = 'answer'",
        "UPDATE traces SET question = NULL, verdict = 'refuse'",
        "UPDATE traces SET question = 'something else'",
        "TRUNCATE traces",
    ):
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            sql(conninfo, stmt)


def test_redaction_is_the_one_permitted_change(pg_index, use_postgres, conninfo):
    ask(QUESTIONS[0], index=pg_index, llm=ExtractiveStub(), trace=True)

    (before,) = sql(conninfo, "SELECT question_sha256, record FROM traces")
    sql(conninfo, "UPDATE traces SET question = NULL")

    assert sql(conninfo, "SELECT question, question_sha256, record FROM traces") == [
        (None, *before)
    ]


# --- wiring ------------------------------------------------------------------

def test_cli_indexes_into_and_answers_from_postgres(use_postgres, conninfo, capsys):
    assert main(["db-init"]) == 0
    assert main(["index", "--corpus", "tests/fixtures/corpus"]) == 0
    assert "postgres build 1" in capsys.readouterr().out
    assert main(["ask", "What is the GST registration turnover threshold?"]) == 0
    assert "$75,000" in capsys.readouterr().out
    assert sql(conninfo, "SELECT verdict FROM traces") == [("answer",)]


def test_health_rechecks_storage_after_caching(pg_index, use_postgres, db_client):
    response = db_client.get("/health")
    assert response.status_code == 200
    assert response.json()["index_build_id"] == pg_index.build_id

    use_postgres.close()

    response = db_client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Database storage is not ready."


def test_health_rejects_an_empty_database(db_client):
    assert db_client.get("/health").status_code == 503


def test_trace_pins_the_loaded_build_after_metadata_changes(pg_index, use_postgres, conninfo):
    changed = [
        c.model_copy(update={"heading_path": [*c.heading_path, "Revised heading"]})
        for c in pg_index.chunks
    ]
    new_build = use_postgres.save_index(HybridIndex(changed))
    assert new_build != pg_index.build_id

    # Exercise both a cached older index and the newly loaded build.
    for idx in (pg_index, use_postgres.load_index()):
        answer = ask(QUESTIONS[0], index=idx, llm=ExtractiveStub())
        (record,) = sql(
            conninfo, "SELECT record FROM traces WHERE trace_id = %s", (answer.trace_id,)
        )[0]
        assert record["index_build_id"] == idx.build_id

        passages = [p for p in record["retrieved"] if not p["id"].startswith("calc:")]
        assert passages
        expected = {c.id: c.heading_path for c in idx.chunks}

        for p in passages:
            assert sql(
                conninfo,
                "SELECT heading_path FROM chunks "
                "WHERE build_id = %s AND id = %s AND text_sha256 = %s",
                (record["index_build_id"], p["id"], p["text_sha256"]),
            ) == [(expected[p["id"]],)]


def test_postgres_audit_rejects_an_unsaved_index(chunks, use_postgres, conninfo):
    with pytest.raises(ValueError, match="database-loaded index"):
        ask(QUESTIONS[0], index=HybridIndex(chunks), llm=ExtractiveStub())

    assert sql(conninfo, "SELECT count(*) FROM traces") == [(0,)]
