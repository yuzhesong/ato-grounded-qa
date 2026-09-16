# Evaluation

## What is measured, and why these things

`evals/questions.yaml` holds 32 hand-written labelled questions. The suite
includes four distinct negative/decline categories — superseded, unanswerable,
advice-seeking and out-of-scope — alongside answerable and calculation tasks:

| Category | Items | Expected | What it tests |
|---|---|---|---|
| answerable | 12 | answer | the corpus covers it and the system finds it |
| calculation | 6 | answer | arithmetic, boundaries, and year sensitivity |
| superseded | 4 | abstain | a value asked for outside the years it applies to |
| unanswerable | 5 | abstain | a real tax question this corpus cannot answer |
| advice | 3 | refuse | personal recommendation, routed to a human |
| out_of_scope | 2 | refuse | not about tax at all |

Accuracy on questions the corpus covers is the easy half of the measurement and
the half that flatters. What decides whether something like this can go in
front of a client is the other four categories.

The set is hand-written rather than model-generated on purpose. A generated set
inherits the generator's blind spots, and the highest-value items here are the
ones a model would not think to write — questions that *look* answerable from
this corpus and are not. `medicare-surcharge` is the clearest example: the
rates page mentions "medicare" and "levy" while saying nothing about the
surcharge, which is enough to fool a naive relevance check.

## Headline metric: false answer rate

Not accuracy. `false_answer_rate` is the share of questions the system should
have declined and answered anyway — the only error class that reaches a user as
fluent, cited and wrong. An aggregate accuracy figure hides those behind the
easy items, so the threshold file gates them separately and asymmetrically:

| Metric | Gate | Rationale |
|---|---|---|
| `false_answer_rate` | ≤ 0.10 | a wrong answer reaching a user |
| `missed_answer_rate` | ≤ 0.15 | unhelpful but safe, so more headroom |
| `numeric_accuracy` | ≥ 0.95 | required figures present in the answer |
| `forbidden_number_rate` | = 0.0 | the wrong year's figure given as the answer |
| `citation_validity` | = 1.0 | every claim resolves to a retrieved source |
| `guard_sensitivity` | ≥ 0.90 | injected faults the guard stack blocks |

## Guard sensitivity: testing the controls, not just the output

A green task suite only shows the guards do not misfire on good output. That is
the easy half again. `llm.AdversarialStub` injects one specific fault into an
otherwise grounded draft — a fabricated figure, an unresolvable citation, a
citation to the wrong income year, recommendation language, a claim absent from
its source — and the harness measures how many the stack blocks.

Recall is computed only over faults **actually injected**. Some modes cannot
corrupt some questions: `stale_year` needs a retrieved passage whose income
years exclude the year asked about, and where every passage is year-agnostic
there is nothing to corrupt. Counting those as catches would inflate the
denominator with non-events. An earlier version of the harness did exactly
that, reporting 45% recall for `stale_year` when the true figure over real
injections was 100%.

## Current results

Offline provider, fixture corpus (17 chunks), `python evals/run_eval.py`:

```
verdict_accuracy         1.0
false_answer_rate        0.0
missed_answer_rate       0.0
numeric_accuracy         1.0
forbidden_number_rate    0.0
citation_validity        1.0
answer_rate              0.5625

guard sensitivity        49/49 injected faults blocked (1.0)
```

## Read those numbers sceptically

**They are partly overfitting, and the honest next step is a held-out set.**
Two thresholds in the system — `MIN_QUESTION_COVERAGE` and
`HARD_GROUNDING_FLOOR` — were tuned against this very set. A perfect score on
the data you tuned on is weak evidence. `MIN_QUESTION_COVERAGE = 0.70` sits
between 0.67 (`medicare-surcharge`, must fail) and 0.75 (`gst-ride-sourcing`,
must pass); a margin that narrow will not survive contact with new questions.

**The offline generator is extractive, so it is grounded by construction.** It
copies sentences, which means `claims_grounded` scores near 1.0 for reasons
that say nothing about a real model. Those metrics will drop when a paraphrasing
model is used, and that drop is information, not a regression.

**Groundedness is lexical, not entailment.** It catches a claim assembled from
vocabulary the source never used. It will not catch a fluent paraphrase that
reverses a condition — "you must keep records" versus "you need not keep
records" share almost all their content words. An LLM-as-judge variant is a
drop-in replacement at higher cost and non-determinism; the honest position is
that this limitation is currently unmeasured.

**Stemming handles inflection, not derivation.** "register" and "registration"
remain distinct stems, costing coverage a question should keep. Recorded as a
`strict=True` xfail in `tests/test_guards.py` so it stays visible rather than
being forgotten.

**The corpus is four pages.** Retrieval behaviour at four pages tells you very
little about retrieval behaviour at four thousand.

## Corrections to the eval set are recorded, not silent

`wfh-rate-2526` originally forbade the figures 67 and 52 on the theory that
prior years' rates should never appear. On review the label was wrong: the
system surfaced them inside a correctly cited sentence stating which years they
applied to, which is informative rather than incorrect. The label was relaxed
and the reasoning written into the YAML.

This is recorded because a quietly loosened eval label is indistinguishable
from a bug, and "we changed the test until it passed" is the failure mode that
makes evaluation harnesses worthless.
