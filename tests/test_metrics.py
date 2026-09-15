"""The metrics decide whether a change ships, so they get tested too.

The distinction that matters is between a false answer (answered when it should
have declined -- a wrong answer reaching a user) and a missed answer (declined
when it could have answered -- unhelpful but safe). Collapsing both into one
accuracy number is how a safety regression passes a green build.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from metrics import aggregate, check_thresholds, score_item  # noqa: E402

from atoqa.schemas import Answer, Claim, GuardResult, Intent, Verdict  # noqa: E402


def make_answer(verdict, text="", claims=(), sources=()):
    return Answer(
        question="q",
        verdict=verdict,
        intent=Intent.GUIDANCE,
        text=text,
        claims=list(claims),
        sources=list(sources),
        guards=[GuardResult(name="citations_resolve", passed=True)],
    )


def test_correct_answer_scores_clean():
    item = {"id": "a", "category": "answerable", "question": "q", "expect": "answer",
            "expect_numbers": [75000]}
    answer = make_answer(
        Verdict.ANSWER,
        text="The threshold is $75,000 per year.",
        claims=[Claim(text="The threshold is $75,000 per year.", citations=["s#0"])],
        sources=[{"id": "s#0"}],
    )
    r = score_item(item, answer)
    assert r.verdict_ok and not r.numbers_missing and r.citations_ok
    assert not r.is_false_answer and not r.is_missed_answer


def test_answering_when_it_should_decline_is_a_false_answer():
    item = {"id": "b", "category": "unanswerable", "question": "q", "expect": "abstain"}
    r = score_item(item, make_answer(Verdict.ANSWER, text="Something confident."))
    assert not r.verdict_ok
    assert r.is_false_answer and not r.is_missed_answer


def test_declining_when_it_should_answer_is_a_missed_answer():
    item = {"id": "c", "category": "answerable", "question": "q", "expect": "answer"}
    r = score_item(item, make_answer(Verdict.ABSTAIN, text="Not answered."))
    assert r.is_missed_answer and not r.is_false_answer


def test_abstain_does_not_satisfy_an_expected_refuse():
    """Both decline, but they mean different things: abstain is 'the corpus
    cannot support this', refuse is 'we will not answer this kind of thing'."""
    item = {"id": "d", "category": "out_of_scope", "question": "q", "expect": "refuse"}
    assert not score_item(item, make_answer(Verdict.ABSTAIN)).verdict_ok
    assert score_item(item, make_answer(Verdict.REFUSE)).verdict_ok


def test_a_required_figure_missing_from_the_answer_is_caught():
    item = {"id": "e", "category": "answerable", "question": "q", "expect": "answer",
            "expect_numbers": [75000]}
    r = score_item(item, make_answer(Verdict.ANSWER, text="The threshold is $82,000."))
    assert r.numbers_missing == [75000.0]


def test_a_forbidden_figure_present_in_the_answer_is_caught():
    """Last year's rate appearing as this year's answer."""
    item = {"id": "f", "category": "calculation", "question": "q", "expect": "answer",
            "expect_numbers": [19717], "expect_absent": [17788]}
    r = score_item(item, make_answer(Verdict.ANSWER, text="Tax is $17,788."))
    assert r.forbidden_present == [17788.0]
    assert r.numbers_missing == [19717.0]


def test_unresolvable_citations_are_caught():
    item = {"id": "g", "category": "answerable", "question": "q", "expect": "answer"}
    answer = make_answer(
        Verdict.ANSWER, text="x",
        claims=[Claim(text="x", citations=["missing#0"])],
        sources=[{"id": "other#0"}],
    )
    assert not score_item(item, answer).citations_ok


def test_rates_use_the_right_denominators():
    results = [
        score_item({"id": "1", "category": "answerable", "question": "q", "expect": "answer"},
                   make_answer(Verdict.ANSWER, text="ok", claims=[Claim(text="ok",
                               citations=["s#0"])], sources=[{"id": "s#0"}])),
        score_item({"id": "2", "category": "answerable", "question": "q", "expect": "answer"},
                   make_answer(Verdict.ABSTAIN)),
        score_item({"id": "3", "category": "unanswerable", "question": "q", "expect": "abstain"},
                   make_answer(Verdict.ANSWER, text="bad")),
        score_item({"id": "4", "category": "out_of_scope", "question": "q", "expect": "refuse"},
                   make_answer(Verdict.REFUSE)),
    ]
    m = aggregate(results)
    # false answers are scored over items that SHOULD have declined (2 of them)
    assert m["false_answer_rate"] == pytest.approx(0.5)
    # missed answers over items that should have answered (2 of them)
    assert m["missed_answer_rate"] == pytest.approx(0.5)
    assert m["verdict_accuracy"] == pytest.approx(0.5)


def test_thresholds_gate_in_the_right_direction():
    metrics = {
        "verdict_accuracy": 0.8, "false_answer_rate": 0.2, "missed_answer_rate": 0.0,
        "numeric_accuracy": 1.0, "forbidden_number_rate": 0.0, "citation_validity": 1.0,
    }
    thresholds = {
        "min_verdict_accuracy": 0.9, "max_false_answer_rate": 0.1,
        "max_missed_answer_rate": 0.15, "min_numeric_accuracy": 0.95,
        "max_forbidden_number_rate": 0.0, "min_citation_validity": 1.0,
        "min_guard_sensitivity": 0.9,
    }
    failures = check_thresholds(metrics, guard_sensitivity=0.5, thresholds=thresholds)
    joined = " ".join(failures)
    assert "verdict_accuracy" in joined
    assert "false_answer_rate" in joined
    assert "guard_sensitivity" in joined
    assert "numeric_accuracy" not in joined

    assert check_thresholds(
        {**metrics, "verdict_accuracy": 1.0, "false_answer_rate": 0.0}, 1.0, thresholds
    ) == []
