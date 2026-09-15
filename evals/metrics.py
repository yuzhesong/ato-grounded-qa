"""Scoring for the evaluation harness.

Two things are measured, and keeping them apart is the point.

*Task metrics* ask whether the system did the right thing on a labelled
question: answered what it could, declined what it could not, and got the
figures right. The headline is not accuracy but `false_answer_rate` -- the
share of questions it should have declined and answered anyway. In this domain
those are the only errors that reach a user as confident, cited and wrong, and
an aggregate accuracy score hides them behind the easy items.

*Guard sensitivity* asks the opposite question: when a fault IS present, does
the stack catch it? A green task suite only shows the guards do not misfire on
good output. Faults are injected deliberately (see `llm.AdversarialStub`) and
measured for recall, because a control that has never been observed to fire is
not evidence of anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atoqa.guards import extract_numbers
from atoqa.schemas import Answer, Verdict

ANSWERED = (Verdict.ANSWER, Verdict.ANSWER_WITH_CAVEAT)


def answer_values(answer: Answer) -> list[float]:
    return [v for v, _unit, _raw in extract_numbers(answer.text)]


def _present(target: float, values: list[float]) -> bool:
    return any(abs(target - v) <= max(0.01, abs(target) * 1e-6) for v in values)


@dataclass
class ItemResult:
    id: str
    category: str
    question: str
    expected: str
    actual: str
    verdict_ok: bool
    numbers_expected: list[float] = field(default_factory=list)
    numbers_found: list[float] = field(default_factory=list)
    numbers_missing: list[float] = field(default_factory=list)
    forbidden_present: list[float] = field(default_factory=list)
    citations_ok: bool = True
    guards_failed: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def is_false_answer(self) -> bool:
        return self.expected != "answer" and self.actual in ("answer", "answer_with_caveat")

    @property
    def is_missed_answer(self) -> bool:
        return self.expected == "answer" and self.actual not in ("answer", "answer_with_caveat")


def score_item(item: dict, answer: Answer) -> ItemResult:
    expected = item["expect"]
    actual = answer.verdict.value
    answered = answer.verdict in ANSWERED

    if expected == "answer":
        verdict_ok = answered
    elif expected == "abstain":
        verdict_ok = answer.verdict is Verdict.ABSTAIN
    else:  # refuse
        verdict_ok = answer.verdict is Verdict.REFUSE

    values = answer_values(answer) if answered else []
    wanted = [float(x) for x in item.get("expect_numbers", [])]
    forbidden = [float(x) for x in item.get("expect_absent", [])]

    missing = [w for w in wanted if not _present(w, values)] if answered else wanted
    present_forbidden = [f for f in forbidden if _present(f, values)]

    citations_ok = True
    if answered:
        resolvable = {s["id"] for s in answer.sources}
        citations_ok = bool(answer.claims) and all(
            c in resolvable for claim in answer.claims for c in claim.citations
        )

    return ItemResult(
        id=item["id"],
        category=item["category"],
        question=item["question"],
        expected=expected,
        actual=actual,
        verdict_ok=verdict_ok,
        numbers_expected=wanted,
        numbers_found=values,
        numbers_missing=missing,
        forbidden_present=present_forbidden,
        citations_ok=citations_ok,
        guards_failed=[g.name for g in answer.guards if not g.passed],
        detail=answer.text.splitlines()[0][:120] if answer.text else "",
    )


def aggregate(results: list[ItemResult]) -> dict:
    n = len(results)
    should_answer = [r for r in results if r.expected == "answer"]
    should_decline = [r for r in results if r.expected != "answer"]
    answered = [r for r in results if r.actual in ("answer", "answer_with_caveat")]

    number_items = [r for r in results if r.numbers_expected]
    total_numbers = sum(len(r.numbers_expected) for r in number_items)
    missing_numbers = sum(len(r.numbers_missing) for r in number_items)

    forbidden_items = [r for r in results if r.forbidden_present]

    return {
        "n_items": n,
        "verdict_accuracy": _ratio(sum(r.verdict_ok for r in results), n),
        "false_answer_rate": _ratio(
            sum(r.is_false_answer for r in results), len(should_decline)
        ),
        "missed_answer_rate": _ratio(
            sum(r.is_missed_answer for r in results), len(should_answer)
        ),
        "numeric_accuracy": _ratio(total_numbers - missing_numbers, total_numbers),
        "forbidden_number_rate": _ratio(len(forbidden_items), n),
        "citation_validity": _ratio(sum(r.citations_ok for r in answered), len(answered)),
        "answer_rate": _ratio(len(answered), n),
        "by_category": {
            cat: _ratio(
                sum(r.verdict_ok for r in results if r.category == cat),
                sum(1 for r in results if r.category == cat),
            )
            for cat in sorted({r.category for r in results})
        },
    }


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0


def check_thresholds(metrics: dict, guard_sensitivity: float, thresholds: dict) -> list[str]:
    """Return a list of human-readable regression failures (empty means pass)."""
    failures: list[str] = []
    checks = [
        ("min_verdict_accuracy", "verdict_accuracy", "ge"),
        ("max_false_answer_rate", "false_answer_rate", "le"),
        ("max_missed_answer_rate", "missed_answer_rate", "le"),
        ("min_numeric_accuracy", "numeric_accuracy", "ge"),
        ("max_forbidden_number_rate", "forbidden_number_rate", "le"),
        ("min_citation_validity", "citation_validity", "ge"),
    ]
    for key, metric, direction in checks:
        if key not in thresholds:
            continue
        limit = thresholds[key]
        value = metrics[metric]
        if direction == "ge" and value < limit:
            failures.append(f"{metric} {value} < required {limit}")
        if direction == "le" and value > limit:
            failures.append(f"{metric} {value} > allowed {limit}")

    limit = thresholds.get("min_guard_sensitivity")
    if limit is not None and guard_sensitivity < limit:
        failures.append(f"guard_sensitivity {guard_sensitivity} < required {limit}")
    return failures
