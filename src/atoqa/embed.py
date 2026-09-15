"""Embedding backends.

The default is a deterministic hashing embedder that needs no credentials and
no network. That is not a toy choice: it means the retrieval layer, the guard
stack, the tests and the evaluation harness all run identically on a laptop
and in CI, and a reviewer can clone this repository and get the same numbers
without an API key. Swapping in a real embedding model changes retrieval
quality, not the contract.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import settings

_TOKEN = re.compile(r"[a-z0-9$%\.]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Random-indexing / hashing-trick embedder.

    Each token (plus each 4-character shingle, which gives some robustness to
    morphology and to numbers written slightly differently) is hashed to a
    dimension with a signed weight. Sub-linear term weighting and L2
    normalisation make the dot product behave like a cosine over a bag of
    features. Deterministic across runs and machines.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _slot(feature: str, dim: int) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(h, "big")
        return n % dim, 1.0 if (n >> 63) & 1 else -1.0

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        for t in toks:
            if len(t) > 5:
                feats.extend(t[i : i + 4] for i in range(len(t) - 3))
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for f in self._features(text):
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                j, sign = self._slot(f, self.dim)
                out[i, j] += sign * (1.0 + np.log(c))
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, dim: int = 1536) -> None:
        from openai import OpenAI  # imported lazily so offline runs need no dependency

        self._client = OpenAI()
        self._model = model or settings.embed_model
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


class AzureOpenAIEmbedder(OpenAIEmbedder):
    def __init__(self, deployment: str | None = None, dim: int = 1536) -> None:
        import os

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
        self._model = deployment or settings.embed_deployment
        self.dim = dim


def get_embedder() -> Embedder:
    if settings.provider == "openai":
        return OpenAIEmbedder()
    if settings.provider == "azure":
        return AzureOpenAIEmbedder()
    return HashingEmbedder()
