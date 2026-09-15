from pathlib import Path

import pytest

from atoqa.ingest import load_markdown_corpus, parse_html

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


def test_fixture_corpus_loads_with_provenance(chunks):
    assert chunks
    for c in chunks:
        assert c.source_url
        assert c.title
        assert c.text.strip()


def test_readme_is_not_ingested_as_a_document(chunks):
    """The fixture directory contains a README explaining the corpus; treating
    it as tax guidance would put non-source text into the retrievable set."""
    assert not any("fixture corpus" in c.text.lower() for c in chunks)
    assert not any(c.title.lower() == "readme" for c in chunks)


def test_chunks_are_stamped_with_the_income_years_they_mention(chunks):
    wfh = [c for c in chunks if "70 cents" in c.text]
    assert wfh, "expected a chunk stating the fixed rate"
    assert "2025-26" in wfh[0].income_years
    assert "2024-25" in wfh[0].income_years


def test_headings_become_the_citation_path(chunks):
    paths = {tuple(c.heading_path) for c in chunks}
    assert ("Non-profit organisations",) in paths
    assert ("Records you need",) in paths


def test_citation_string_is_human_readable(chunks):
    c = next(c for c in chunks if c.heading_path and c.qc_ref)
    citation = c.citation
    assert c.title in citation and c.qc_ref in citation


def test_chunks_do_not_exceed_the_window(chunks):
    from atoqa.ingest import MAX_WORDS

    assert all(len(c.text.split()) <= MAX_WORDS for c in chunks)


def test_parse_html_extracts_title_and_qc_reference():
    html = """
    <html><head><title>ignored</title></head><body><main>
      <h1>Registering for GST</h1>
      <h2>When you must register</h2>
      <p>You must register once your GST turnover reaches $75,000 or more per year.</p>
      <p>QC22412</p>
      <script>var x = 1;</script>
    </main></body></html>
    """
    chunks = parse_html(html, "https://example.invalid/gst", "2026-09-15")
    assert chunks
    assert chunks[0].title == "Registering for GST"
    assert chunks[0].qc_ref == "QC22412"
    assert "var x" not in " ".join(c.text for c in chunks), "script content leaked into corpus"


def test_missing_corpus_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No fixture documents"):
        load_markdown_corpus(tmp_path)
