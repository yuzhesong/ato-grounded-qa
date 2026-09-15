"""End-to-end behaviour of every branch through `pipeline.ask`."""

import pytest

from atoqa.llm import AdversarialStub, ExtractiveStub
from atoqa.pipeline import ask
from atoqa.schemas import Intent, Verdict

ANSWERED = (Verdict.ANSWER, Verdict.ANSWER_WITH_CAVEAT)


@pytest.fixture
def llm():
    return ExtractiveStub()


def run(question, index, llm):
    return ask(question, index=index, llm=llm, trace=False)


# --- deterministic branch ----------------------------------------------------

def test_arithmetic_is_computed_not_generated(index, llm):
    a = run("How much tax on $90,000 for 2025-26?", index, llm)
    assert a.verdict is Verdict.ANSWER
    assert a.intent is Intent.CALCULATION
    assert "$17,788" in a.text
    assert all(c.citations == ["calc:resident_income_tax"] for c in a.claims)


def test_the_arithmetic_is_shown(index, llm):
    a = run("How much tax on $90,000 for 2025-26?", index, llm)
    assert len(a.claims) > 1, "the working should be part of the answer, not hidden"
    assert any("Excess" in c.text for c in a.claims)


def test_a_computation_passes_the_same_guards_as_retrieved_text(index, llm):
    """The deterministic result is wrapped as a citable source deliberately, so
    there is one verification path rather than two."""
    a = run("How much tax on $90,000 for 2025-26?", index, llm)
    names = {g.name for g in a.guards}
    assert {"numbers_supported", "citations_resolve", "year_consistent"} <= names
    assert all(g.passed for g in a.guards)


def test_the_stated_year_changes_the_answer(index, llm):
    later = run("How much tax on $90,000 for 2025-26?", index, llm)
    earlier = run("How much tax on $90,000 for 2023-24?", index, llm)
    assert "$17,788" in later.text
    assert "$19,717" in earlier.text
    assert "$17,788" not in earlier.text


def test_a_year_without_verified_rates_is_refused(index, llm):
    a = run("How much tax on $90,000 for 2019-20?", index, llm)
    assert a.verdict is Verdict.ABSTAIN
    assert "verified rate table" in a.text


# --- retrieval branch --------------------------------------------------------

def test_guidance_answer_is_cited_and_grounded(index, llm):
    a = run("What is the GST registration turnover threshold?", index, llm)
    assert a.verdict in ANSWERED
    assert "$75,000" in a.text
    assert a.sources and all(s["url"] for s in a.sources)
    assert all(g.passed for g in a.guards), [g for g in a.guards if not g.passed]


def test_an_unstated_year_is_disclosed_rather_than_assumed_silently(index, llm):
    a = run("What is the working from home fixed rate?", index, llm)
    assert a.income_year_assumed
    assert any("No income year was specified" in c for c in a.caveats)


def test_a_superseded_year_abstains_instead_of_adapting_the_number(index, llm):
    a = run("What is the working from home fixed rate for 2022-23?", index, llm)
    assert a.verdict is Verdict.ABSTAIN
    assert "2022-23" in a.text
    assert "67" not in a.text


def test_a_topic_outside_the_corpus_abstains(index, llm):
    a = run("What is the instant asset write-off threshold?", index, llm)
    assert a.verdict is Verdict.ABSTAIN
    assert any(g.name == "question_covered" and not g.passed for g in a.guards)


# --- policy branches ---------------------------------------------------------

def test_personal_advice_is_refused_and_routed_to_a_human(index, llm):
    a = run("Should I set up a family trust to reduce my tax?", index, llm)
    assert a.verdict is Verdict.REFUSE
    assert a.intent is Intent.ADVICE_SEEKING
    assert "registered tax agent" in a.text


def test_off_topic_is_refused(index, llm):
    a = run("Who won the 1998 AFL grand final?", index, llm)
    assert a.verdict is Verdict.REFUSE
    assert a.intent is Intent.OUT_OF_SCOPE


def test_a_tax_question_with_no_keyword_is_rescued_by_retrieval(index, llm):
    """The keyword pre-filter is imprecise; retrieval gets the final say.

    "written evidence for work-related expenses" contains no word on the tax
    vocabulary list but is squarely covered by the corpus, and was being
    refused as out of scope before the rescue path existed."""
    a = run("When do I need written evidence for work-related expenses?", index, llm)
    assert a.verdict in ANSWERED
    assert "300" in a.text


# --- guards have real authority ---------------------------------------------

@pytest.mark.parametrize("mode", AdversarialStub.MODES)
def test_every_injected_fault_is_blocked(index, mode):
    """A guard that has never been observed to fire is not evidence of control."""
    question = "What is the GST registration turnover threshold?"
    stub = AdversarialStub(mode)
    a = ask(question, index=index, llm=stub, trace=False)
    if not stub.injected:
        pytest.skip(f"{mode} had no fault to inject on this question")
    assert a.verdict is Verdict.ABSTAIN, f"{mode} escaped verification"


def test_guidance_without_an_index_is_an_error(llm):
    with pytest.raises(ValueError, match="retrieval index is required"):
        ask("What is the GST threshold?", index=None, llm=llm, trace=False)


# --- audit -------------------------------------------------------------------

def test_a_trace_is_written_for_every_decision(index, llm, tmp_path, monkeypatch):
    from atoqa import audit
    from atoqa.config import settings

    monkeypatch.setattr(settings, "trace_dir", tmp_path)
    for q in ("What is the GST registration turnover threshold?",
              "Who won the 1998 AFL grand final?",
              "What is the instant asset write-off threshold?"):
        ask(q, index=index, llm=llm, trace=True)
    lines = [ln for f in tmp_path.glob("*.jsonl") for ln in f.read_text().splitlines()]
    assert len(lines) == 3, "every disposition must leave a trace, including refusals"

    import json

    records = [json.loads(ln) for ln in lines]
    assert {r["verdict"] for r in records} == {"answer", "refuse", "abstain"}
    for r in records:
        assert r["trace_id"] and r["prompt_version"] and r["question_sha256"]
        assert "thresholds" in r
    assert audit.PROMPT_VERSION
