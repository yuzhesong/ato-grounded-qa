# Architecture

## The decision this system is built around

Most of the engineering in a tax assistant is not the retrieval or the prompt.
It is deciding, per question, **which of three mechanisms should answer it**:

| Mechanism | Used for | Why |
|---|---|---|
| Deterministic code | arithmetic, published rates and thresholds | These are facts. A model asked to recall them will sometimes be confidently wrong, and there is no upside to the risk — the correct answer is a table lookup. |
| Retrieval + generation | explanation, conditions, eligibility, "what does this cover" | These are language. The source says it better than a rule engine could, and the model's job is selection and phrasing, not recall. |
| Human referral | anything about the user's own circumstances | Not a capability gap. Advising on a person's affairs is regulated work, and a tool that does it anyway is a liability whether or not it is accurate. |

Getting that split right matters more than the quality of any single component.
A system that routes a rate lookup to the model has already lost; a system that
routes a "should I" question to retrieval has lost differently.

## Flow

```
question
  │
  ├─ router.route()                      deterministic: regex + keyword tables
  │    ├─ ADVICE_SEEKING ───────────────► REFUSE, refer to a registered tax agent
  │    ├─ OUT_OF_SCOPE ─────────────────► probe retrieval; refuse unless clearly covered
  │    ├─ CALCULATION ──────────────────► calculator.income_tax / marginal_rate
  │    │                                    from rates.py (verified tables only)
  │    └─ GUIDANCE
  │         ├─ threshold cue? ──────────► calculator.lookup(), year-scoped
  │         └─ index.search()             hybrid BM25 + dense, RRF ordering
  │              │
  │              └─ retrieval score < gate ──► ABSTAIN
  │
  ├─ llm.draft()                          structured output: claims[] each cited
  │     └─ insufficient_context ─────────► ABSTAIN
  │
  ├─ guards.run_all()                     six deterministic checks
  │     ├─ blocking failure ─────────────► ABSTAIN with the reason
  │     └─ advisory failure ─────────────► ANSWER_WITH_CAVEAT
  │
  └─ audit.write()                        JSONL or PostgreSQL trace, including refusals
```

## Why a computation is wrapped as a citable source

`pipeline._computation_chunk` turns a deterministic result into a `Chunk` with
the rate table's URL and QC reference, and feeds it through retrieval's own
guard stack. This looks like indirection and is load-bearing: it means there is
**one** answer to "can this claim be trusted", not one for generated text and a
separate, softer one for computed text. Computed answers are checked for
citation validity, numeric support and year consistency exactly like any other.

## Routing before retrieval, with retrieval as the backstop

The router is keyword-based, which makes it fast, free and imprecise. A real
tax question can dodge every keyword — "when do I need written evidence for
work-related expenses?" contains none. Refusing on the keyword list alone was
turning good questions away, so an apparent out-of-scope question is probed
against retrieval and re-routed as guidance if the corpus clearly covers it
(`min_rescue_score`, deliberately set above `min_retrieval_score`: a marginal
match is not evidence of coverage).

The reverse order — retrieve first, classify after — would be cleaner but
wastes a retrieval on every "should I…" question and loses the ability to
refuse advice-seeking questions before any model sees them.

## Two scores, two jobs

`index.search` returns both an RRF-based ordering and a separately calibrated
`score` in [0, 1]. Conflating them is a real trap: RRF values are not
comparable across queries, so using them as an abstention threshold means the
best of six irrelevant passages looks as good as a genuine hit. The calibrated
score uses absolute components — cosine clipped at zero, BM25 through a
saturating transform — precisely so that a query with no relevant document in
the corpus scores low instead of scoring 1.0 by normalisation.

This was not theoretical. The first implementation normalised BM25 by the best
score in the result set, and an off-topic query scored 0.77.

## Where the provider boundary sits

`embed.get_embedder()` and `llm.get_llm()` are the only places that know which
provider is in use. Offline (deterministic hash embedder plus extractive stub),
OpenAI, and Azure OpenAI are interchangeable; everything downstream is defined
against `DraftAnswer`, so swapping the model changes answer quality and cost,
never the contract or the verification path.

The offline default is the reason the evaluation harness can be a CI gate.

## What is deliberately absent

No agent loop, no tool-calling beyond the calculator, no multi-turn state. For
this task they would be decoration: the question decomposes into route →
retrieve → verify, and adding a planner would make the failure modes harder to
reason about without answering a single additional question correctly. The
structured-output contract and the guard stack are the parts that would carry
over unchanged into an agentic version.
