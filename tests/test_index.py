import numpy as np
import pytest

from atoqa.config import settings
from atoqa.embed import HashingEmbedder
from atoqa.index import HybridIndex
from atoqa.schemas import Chunk


def test_hashing_embedder_is_deterministic_and_normalised():
    """Determinism is what lets CI reproduce eval numbers without an API key."""
    e = HashingEmbedder(dim=128)
    a = e.encode(["the GST turnover threshold is $75,000"])
    b = e.encode(["the GST turnover threshold is $75,000"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)


def test_related_text_scores_above_unrelated_text():
    e = HashingEmbedder(dim=512)
    v = e.encode(
        [
            "you must register for GST once turnover reaches $75,000",
            "GST registration turnover threshold",
            "the cat sat on the mat in the afternoon sun",
        ]
    )
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_retrieval_finds_the_right_document(index):
    top = index.search("what is the GST registration turnover threshold", k=3)
    assert any("gst" in r.chunk.source_url.lower() for r in top[:2])


def test_retrieval_confidence_separates_on_topic_from_off_topic(index):
    """The abstention gate depends on this separation, so it is asserted here
    rather than assumed. Scores must be absolute, not relative to the current
    result set -- normalising by the best score in the set would hand every
    query a 1.0, including a query with no relevant document at all."""
    on_topic = max(
        index.search(q, k=1)[0].score
        for q in ("GST registration turnover threshold", "super guarantee rate")
    )
    off_topic = max(
        index.search(q, k=1)[0].score
        for q in ("who won the 1998 AFL grand final", "how do I bake sourdough bread")
    )
    assert off_topic < settings.min_retrieval_score < on_topic, (
        f"off_topic={off_topic:.3f} gate={settings.min_retrieval_score} on_topic={on_topic:.3f}"
    )


def test_vocabulary_covers_the_corpus(index):
    vocab = index.vocabulary
    assert "turnover" in vocab and "guarantee" in vocab
    assert "surcharge" not in vocab, "fixture corpus must not discuss the surcharge"


def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    reloaded = HybridIndex.load(tmp_path)
    assert len(reloaded.chunks) == len(index.chunks)
    before = index.search("GST threshold", k=3)
    after = reloaded.search("GST threshold", k=3)
    assert [r.chunk.id for r in before] == [r.chunk.id for r in after]
    assert [round(r.score, 6) for r in before] == [round(r.score, 6) for r in after]


def test_empty_index_is_rejected():
    with pytest.raises(ValueError):
        HybridIndex([])


def test_heading_path_contributes_to_retrieval():
    """A chunk whose body omits the query term should still be findable via its
    heading, because ATO pages carry meaning in their section titles."""
    chunks = [
        Chunk(id="a#0", text="The amount is $150,000 per year or more.", title="Registering",
              source_url="u", heading_path=["Non-profit organisations"]),
        Chunk(id="b#0", text="Something entirely unrelated about vehicles.", title="Other",
              source_url="u2"),
    ]
    idx = HybridIndex(chunks)
    assert idx.search("non-profit organisations", k=1)[0].chunk.id == "a#0"
