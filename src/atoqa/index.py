"""Hybrid retrieval: dense vectors + BM25, fused by reciprocal rank.

Two scores are kept deliberately separate:

* ``rank`` comes from reciprocal rank fusion and decides *ordering*. RRF is
  robust when the two retrievers disagree and needs no score calibration.
* ``score`` is a calibrated blend in [0, 1] and decides *whether to answer at
  all*. RRF scores are not comparable across queries, so they make a poor
  abstention threshold; a normalised blend does.

Conflating the two is how a system ends up confidently answering from the
best of six irrelevant passages.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embed import Embedder, get_embedder, tokenize
from .schemas import Chunk, Retrieved

# BM25 score at which a lexical match is treated as half-convincing. Raising it
# makes the system more willing to abstain.
BM25_SATURATION = 6.0


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        self._vectors = self.embedder.encode([self._embed_text(c) for c in chunks])
        self._bm25 = BM25Okapi([tokenize(self._embed_text(c)) for c in chunks])

    @staticmethod
    def _embed_text(c: Chunk) -> str:
        """Prepend the heading path so section context is part of the signal."""
        head = " > ".join([c.title, *c.heading_path])
        return f"{head}\n{c.text}"

    def search(self, query: str, k: int | None = None) -> list[Retrieved]:
        k = k or settings.top_k
        n = len(self.chunks)

        qv = self.embedder.encode([query])[0]
        dense = self._vectors @ qv                       # cosine, vectors are L2-normalised
        lexical = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

        dense_order = np.argsort(-dense)
        lex_order = np.argsort(-lexical)
        dense_rank = {int(idx): r for r, idx in enumerate(dense_order)}
        lex_rank = {int(idx): r for r, idx in enumerate(lex_order)}

        # Calibrated relevance for the abstention gate.
        #
        # Both components must be ABSOLUTE, not per-query relative. Dividing
        # BM25 by the best score in the current result set would hand the top
        # hit 1.0 for every query, including one with no relevant document in
        # the corpus at all -- which is exactly the case the gate exists to
        # catch. So: cosine clipped at zero, and BM25 put through a saturating
        # transform whose midpoint (BM25_SATURATION) is the score at which a
        # lexical match counts as half-convincing.
        dense_norm = np.clip(dense, 0.0, 1.0)
        lexical_pos = np.clip(lexical, 0.0, None)
        lex_norm = lexical_pos / (lexical_pos + BM25_SATURATION)
        blended = settings.dense_weight * dense_norm + (1 - settings.dense_weight) * lex_norm

        rrf = np.array(
            [
                1.0 / (settings.rrf_k + dense_rank[i] + 1)
                + 1.0 / (settings.rrf_k + lex_rank[i] + 1)
                for i in range(n)
            ]
        )

        top = np.argsort(-rrf)[:k]
        return [
            Retrieved(
                chunk=self.chunks[int(i)],
                score=float(blended[int(i)]),
                rank=r,
                dense_rank=dense_rank[int(i)],
                lexical_rank=lex_rank[int(i)],
            )
            for r, i in enumerate(top)
        ]

    @property
    def vocabulary(self) -> set[str]:
        """Every content word anywhere in the corpus.

        Used by the `question_covered` guard: a question containing a
        distinctive term the corpus has never used cannot be answered from
        this corpus, however well the generic words match.
        """
        if getattr(self, "_vocab", None) is None:
            from .guards import _content_words

            vocab: set[str] = set()
            for c in self.chunks:
                vocab |= _content_words(f"{c.title} {' '.join(c.heading_path)} {c.text}")
            self._vocab = vocab
        return self._vocab

    # --- persistence ---------------------------------------------------------

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or settings.index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self._vectors)
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> HybridIndex:
        directory = Path(directory or settings.index_dir)
        chunks = [Chunk(**d) for d in json.loads((directory / "chunks.json").read_text("utf-8"))]
        inst = cls.__new__(cls)
        inst.chunks = chunks
        inst.embedder = get_embedder()
        inst._vectors = np.load(directory / "vectors.npz")["vectors"]
        inst._bm25 = BM25Okapi([tokenize(cls._embed_text(c)) for c in chunks])
        return inst
