"""Each guard is tested twice: it must pass clean output and fire on the exact
fault it exists for. A guard only ever observed passing is not evidence."""

import pytest

from atoqa.guards import (
    citations_resolve,
    claims_grounded,
    extract_numbers,
    no_personal_advice,
    no_unsupported_claim,
    numbers_supported,
    question_covered,
    stem,
    year_consistent,
)
from atoqa.schemas import Chunk, Claim, DraftAnswer, Retrieved

SRC = Chunk(
    id="src#000",
    text="You must register for GST if your GST turnover is $75,000 or more per year. "
    "The fixed rate is 70 cents per work hour for 2024-25 and 2025-26.",
    title="Registering for GST",
    source_url="https://example.invalid/gst",
    qc_ref="QC22412",
    income_years=["2024-25", "2025-26"],
)
OLD = Chunk(
    id="old#000",
    text="The fixed rate was 67 cents per work hour for 2022-23 and 2023-24.",
    title="Fixed rate method",
    source_url="https://example.invalid/wfh",
    income_years=["2022-23", "2023-24"],
)
RETRIEVED = [Retrieved(chunk=SRC, score=0.9, rank=0), Retrieved(chunk=OLD, score=0.4, rank=1)]


def draft(text: str, citations=("src#000",)) -> DraftAnswer:
    return DraftAnswer(claims=[Claim(text=text, citations=list(citations))])


# --- numeric extraction ------------------------------------------------------

def test_extract_numbers_reads_units_and_skips_income_years():
    found = extract_numbers("70 cents per hour for 2024-25, threshold $75,000, rate 12%")
    assert (70.0, "cents", "70 cents") in found
    assert (75_000.0, "aud", "$75,000") in found
    assert (12.0, "percent", "12%") in found
    assert not any(v in (2024.0, 25.0) for v, _, _ in found), "income year leaked in as a figure"


@pytest.mark.parametrize(
    "text,raw",
    [
        ("threshold $75,000, then more", "$75,000"),
        ("the rate is 12%.", "12%"),
        ("costs $1,234.50; plus fees", "$1,234.50"),
    ],
)
def test_extract_numbers_does_not_capture_trailing_punctuation(text, raw):
    """Stray punctuation in the raw match leaks into guard detail messages,
    which a human reads when deciding whether an answer was wrongly blocked."""
    assert raw in [r for _v, _u, r in extract_numbers(text)]


def test_cents_and_dollars_are_the_same_fact():
    good = draft("The rate is $0.70 per work hour.")
    assert numbers_supported(good, RETRIEVED).passed


# --- one test per guard: clean passes, fault fires ---------------------------

def test_citations_resolve():
    assert citations_resolve(draft("GST turnover of $75,000 or more."), RETRIEVED).passed
    bad = citations_resolve(draft("x", citations=("nope#000",)), RETRIEVED)
    assert not bad.passed and "nope#000" in bad.detail


def test_numbers_supported_catches_a_fabricated_figure():
    assert numbers_supported(draft("register for GST at $75,000 or more"), RETRIEVED).passed
    bad = numbers_supported(draft("register for GST at $82,000 or more"), RETRIEVED)
    assert not bad.passed and "$82,000" in bad.detail


def test_no_unsupported_claim_blocks_an_invented_assertion():
    clean = draft("You must register for GST if your GST turnover is $75,000 or more per year.")
    assert no_unsupported_claim(clean, RETRIEVED).passed
    invented = draft(
        "Eligibility rests on the commissioner's discretionary apportionment assessment."
    )
    assert not no_unsupported_claim(invented, RETRIEVED).passed


def test_claims_grounded_scores_paraphrase_without_blocking_it():
    result = claims_grounded(draft("Registration is required at turnover of $75,000."), RETRIEVED)
    assert result.score is not None and 0.0 <= result.score <= 1.0


def test_year_consistent_catches_a_superseded_rate():
    current = draft("The fixed rate is 70 cents per work hour.", citations=("src#000",))
    assert year_consistent(current, RETRIEVED, "2025-26").passed
    stale = draft("The fixed rate was 67 cents per work hour.", citations=("old#000",))
    bad = year_consistent(stale, RETRIEVED, "2025-26")
    assert not bad.passed and "2022-23" in bad.detail


def test_year_consistent_is_not_applicable_without_a_year():
    assert year_consistent(draft("anything"), RETRIEVED, None).passed


def test_no_personal_advice():
    assert no_personal_advice(draft("The threshold is $75,000.")).passed
    assert not no_personal_advice(draft("You should register for GST now.")).passed


def test_question_covered_rejects_an_answer_to_a_different_question():
    """The canonical case: shared generic vocabulary, different subject."""
    d = draft("You must register for GST if your GST turnover is $75,000 or more per year.")
    assert question_covered(d, RETRIEVED, "What GST turnover requires registering?").passed
    off_topic = question_covered(d, RETRIEVED, "What is the instant asset write-off threshold?")
    assert not off_topic.passed
    assert "instant" in off_topic.detail


@pytest.mark.xfail(
    reason="stem() handles inflection but not derivation: register != registration. "
    "Recorded rather than hidden; see the note in guards.stem.",
    strict=True,
)
def test_question_covered_handles_derivational_morphology():
    d = draft("You must register for GST if your GST turnover is $75,000 or more per year.")
    assert question_covered(d, RETRIEVED, "What is the GST registration threshold?").passed


def test_question_covered_reports_terms_absent_from_the_whole_corpus():
    d = draft("You must register for GST if your GST turnover is $75,000 or more per year.")
    result = question_covered(
        d,
        RETRIEVED,
        "What is the Division 7A benchmark rate?",
        corpus_vocabulary={"gst", "turnover"},
    )
    assert not result.passed and "absent from the whole corpus" in result.detail


def test_question_covered_tolerates_incidental_vocabulary():
    """'drivers' never appears in the source, but the question is answerable."""
    ride = Chunk(
        id="ride#000",
        text="A business providing taxi, limousine or ride-sourcing services must register for "
        "GST regardless of its GST turnover.",
        title="Registering for GST",
        source_url="https://example.invalid/gst",
    )
    result = question_covered(
        draft(ride.text, citations=("ride#000",)),
        [Retrieved(chunk=ride, score=0.9, rank=0)],
        "Do ride-sourcing drivers have to register for GST?",
    )
    assert result.passed, result.detail


@pytest.mark.parametrize(
    "singular,plural",
    [("cover", "covers"), ("desk", "desks"), ("expense", "expenses"),
     ("record", "records"), ("service", "services"), ("threshold", "thresholds")],
)
def test_stemming_collapses_inflected_variants(singular, plural):
    """The property that matters is that the two forms agree, not what they
    agree on. Asserting a particular stem string tests the implementation;
    asserting agreement tests the thing the coverage guard relies on."""
    assert stem(singular) == stem(plural)


def test_stemming_leaves_short_and_unsuffixed_words_alone():
    assert stem("gst") == "gst"
    assert stem("tax-free") == "taxfree"
