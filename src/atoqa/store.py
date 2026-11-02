"""PostgreSQL storage for the index and the audit trace.

Optional. With ``ATOQA_DATABASE_URL`` unset, everything stays as files under
``data/`` and this module is never imported, so tests, CI and the offline
quickstart need neither a database nor a driver. Set it, and:

* **The index** is written by ``atoqa index`` as a new *build* -- chunks,
  vectors, and the embedder that produced them -- and loading takes the latest.
  Builds are kept rather than overwritten because chunk ids are positional.
  Each trace records the loaded index_build_id and each retrieved passage's
  id and text_sha256. The build ID and chunk ID identify the stored passage,
  including its metadata; the hash identifies its text.
* **The audit trace** is one row per decision instead of one JSONL line.

Retrieval stays in memory. This module stores vectors; it does not search
them. Moving search into the database would move where the cosine is computed,
and the calibrated score that drives abstention is the one number that must not
differ between storage backends. ``tests/test_store.py`` asserts that it does
not.

``traces`` is append-only by enforcement rather than convention: a trigger
rejects DELETE, TRUNCATE, and every UPDATE except the one a retention policy
needs -- clearing ``question`` to NULL. The hash survives, so a redacted trace
stays correlatable. The trigger guards against mistakes, not adversaries (the
table owner can drop it), so a deployment should also give the application
role only SELECT and INSERT on ``traces``.

Requires PostgreSQL 14+ for ``CREATE OR REPLACE TRIGGER``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import settings
from .embed import Embedder, get_embedder
from .index import HybridIndex
from .schemas import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS index_builds (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    built_at     timestamptz NOT NULL DEFAULT now(),
    embedder     text NOT NULL,
    dim          integer NOT NULL,
    chunk_count  integer NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    build_id      bigint NOT NULL REFERENCES index_builds (id) ON DELETE CASCADE,
    ord           integer NOT NULL,
    id            text NOT NULL,
    text_sha256   text NOT NULL,
    text          text NOT NULL,
    title         text NOT NULL,
    source_url    text NOT NULL,
    qc_ref        text,
    heading_path  text[] NOT NULL,
    income_years  text[] NOT NULL,
    retrieved_at  text,
    vector        real[] NOT NULL,
    PRIMARY KEY (build_id, ord),
    UNIQUE (build_id, id)
);
CREATE INDEX IF NOT EXISTS chunks_id_text_sha256 ON chunks (id, text_sha256);

CREATE TABLE IF NOT EXISTS traces (
    trace_id         text PRIMARY KEY,
    at               timestamptz NOT NULL,
    question         text,
    question_sha256  text NOT NULL,
    verdict          text NOT NULL,
    record           jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_verdict_at ON traces (verdict, at);

CREATE OR REPLACE FUNCTION traces_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    -- Nested rather than ANDed: NEW is NULL for DELETE and TRUNCATE.
    IF TG_OP = 'UPDATE' THEN
        IF NEW.question IS NULL
           AND to_jsonb(NEW) - 'question' = to_jsonb(OLD) - 'question' THEN
            RETURN NEW;
        END IF;
    END IF;
    RAISE EXCEPTION 'traces is append-only: % rejected', TG_OP;
END
$$;

CREATE OR REPLACE TRIGGER traces_append_only_rows
    BEFORE UPDATE OR DELETE ON traces
    FOR EACH ROW EXECUTE FUNCTION traces_append_only();

CREATE OR REPLACE TRIGGER traces_append_only_truncate
    BEFORE TRUNCATE ON traces
    FOR EACH STATEMENT EXECUTE FUNCTION traces_append_only();
"""


def _embedder_id(e: Embedder) -> str:
    """Recorded with each build so its vectors are never compared against query
    vectors from a different model. The scores would still land in [0, 1]; they
    would just mean nothing, and the abstention gate would not notice."""
    return f"{type(e).__name__}:{getattr(e, '_model', '-')}:{e.dim}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Store:
    def __init__(self, conninfo: str) -> None:
        # A mistyped URL would otherwise surface as a pool timeout on first use.
        conninfo_to_dict(conninfo)
        self._pool = ConnectionPool(conninfo, min_size=1, max_size=4, open=True, timeout=10.0)

    def close(self) -> None:
        self._pool.close()

    def check_ready(self) -> None:
        """Check connectivity and the trace table without writing a record."""
        with self._pool.connection(timeout=2.0) as conn:
            conn.execute("SET LOCAL statement_timeout = '2s'")
            conn.execute("SELECT 1 FROM traces LIMIT 0")

    def init_schema(self) -> None:
        """Idempotent. Run once by `atoqa db-init`, not on every connect, so the
        application role never needs DDL rights."""
        with self._pool.connection() as conn:
            conn.execute(SCHEMA)

    # --- index ---------------------------------------------------------------

    def save_index(self, index: HybridIndex) -> int:
        """Write the index as a new build and return its id.

        One transaction: readers keep seeing the previous build until this one
        commits, never half of it.
        """
        vectors = index.vectors
        with self._pool.connection() as conn:
            (build_id,) = conn.execute(
                "INSERT INTO index_builds (embedder, dim, chunk_count)"
                " VALUES (%s, %s, %s) RETURNING id",
                (_embedder_id(index.embedder), vectors.shape[1], len(index.chunks)),
            ).fetchone()
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (build_id, ord, id, text_sha256, text, title,"
                    " source_url, qc_ref, heading_path, income_years, retrieved_at, vector)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (build_id, i, c.id, _sha256(c.text), c.text, c.title, c.source_url,
                         c.qc_ref, c.heading_path, c.income_years, c.retrieved_at,
                         vectors[i].tolist())
                        for i, c in enumerate(index.chunks)
                    ],
                )
        return build_id

    def load_index(self) -> HybridIndex:
        embedder = get_embedder()
        with self._pool.connection() as conn:
            build = conn.execute(
                "SELECT id, embedder FROM index_builds ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if build is None:
                raise LookupError("No index in the database. Run `atoqa index` first.")
            build_id, built_with = build
            if built_with != _embedder_id(embedder):
                raise RuntimeError(
                    f"Index build {build_id} was embedded with {built_with}, but the "
                    f"configured embedder is {_embedder_id(embedder)}. Re-run `atoqa index`."
                )
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    "SELECT id, text, title, source_url, qc_ref, heading_path, income_years,"
                    " retrieved_at, vector FROM chunks WHERE build_id = %s ORDER BY ord",
                    (build_id,),
                ).fetchall()
        vectors = np.asarray([r.pop("vector") for r in rows], dtype=np.float32)
        return HybridIndex.from_vectors(
            [Chunk(**r) for r in rows], vectors, embedder, build_id=build_id
        )

    # --- audit trace ---------------------------------------------------------

    def write_trace(self, record: dict) -> None:
        """Insert one audit record. The question is kept out of the JSON body so
        that redaction has exactly one column to clear."""
        body = {k: v for k, v in record.items() if k != "question"}
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO traces (trace_id, at, question, question_sha256, verdict, record)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (record["trace_id"], record["at"], record["question"],
                 record["question_sha256"], record["verdict"], Jsonb(body)),
            )


@lru_cache(maxsize=1)
def get_store() -> Store:
    if not settings.database_url:
        raise RuntimeError("ATOQA_DATABASE_URL is not set.")
    return Store(settings.database_url)
