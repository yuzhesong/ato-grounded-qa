# ato-grounded-qa

Grounded question answering over ATO public guidance — built to demonstrate the
part of applied AI engineering that decides whether a system is safe to put in
front of someone: **knowing when not to answer.**

Every answer is decomposed into atomic claims, each carrying a citation. Six
deterministic guards then verify the draft before it is shown, and can downgrade
it to a caveat or refuse it outright. Arithmetic and published rates are computed
from verified reference tables, never recalled by a model. Requests for personal
advice are refused and routed to a human. Every decision — including every
refusal — writes an audit trace.

Runs offline with no API key: `make install && make eval`.

> Not tax advice. It reports what published ATO guidance says, with citations.
> ATO material is not redistributed in this repository — see
> [docs/governance.md](docs/governance.md).

## Quickstart

```bash
pip install -e ".[dev]"

# 60 seconds, no credentials, no network:
make test          # offline; the PostgreSQL tests skip without a database
make eval          # evaluation harness against regression gates

# ask something
python -m atoqa.cli ask "What is the GST registration turnover threshold?"
python -m atoqa.cli ask "How much tax on \$90,000 for 2025-26?"
python -m atoqa.cli ask "What was the working from home rate for 2022-23?"

# against the live corpus instead of the fixtures
make fetch && make index && make eval

# HTTP
make serve   # POST /ask
```

Set `ATOQA_PROVIDER=openai` or `azure` in `.env` to use a real model; see
`.env.example`.

## PostgreSQL storage (optional)

PostgreSQL stores versioned index builds and audit records. Retrieval still
runs in memory using the existing hybrid index. PostGIS is not implemented.

```bash
pip install -e ".[dev,postgres]"

export ATOQA_PROVIDER=offline
export ATOQA_DATABASE_URL=postgresql://atoqa:atoqa@127.0.0.1:5432/atoqa

make db-up
```

Wait until `docker compose exec -T db pg_isready -U atoqa -d atoqa` reports
that PostgreSQL is accepting connections, then run:

```bash
make db-init

# Reproducible demonstration using the synthetic fixture corpus:
python -m atoqa.cli index --corpus tests/fixtures/corpus
python -m atoqa.cli ask "What is the GST registration turnover threshold?"

# Run the PostgreSQL integration tests as well as the regular suite:
ATOQA_TEST_DATABASE_URL="$ATOQA_DATABASE_URL" pytest

make serve
```

For fetched ATO content, run `make fetch && make index` instead of indexing
the fixtures.

With PostgreSQL enabled, `ask --corpus` is rejected. Persist the corpus
with `index --corpus` first, then ask against the stored index.

The API caches its loaded index. After re-indexing, stop and restart all API
workers to load the new build. Uvicorn's `--reload` does not detect database
changes. `/health` reports the loaded `index_build_id` and returns HTTP 503
when its database readiness check fails.

To use file storage, leave `ATOQA_DATABASE_URL` unset in both the shell and
`.env`.

## What it does that a basic RAG pipeline does not

**Refuses to recall facts.** `rates.py` holds income tax brackets and
thresholds as data with provenance — source URL, QC reference, the date it was
checked, and a `verified` flag the calculator treats as a hard gate. Tax
arithmetic is a table lookup with the working shown. Asking a model to recall a
marginal rate is the single cheapest way to produce a confident, expensive
error, and there is no upside to the risk.

```
$ python -m atoqa.cli ask "How much tax on \$90,000 for 2025-26?"

ANSWER  (intent: calculation, income year: 2025-26)

Tax payable on $90,000 for 2025-26 is $17,788.00. [calc:resident_income_tax]
Income year 2025-26; bracket $45,001-$135,000 at 0.3 (30c per $1 over $45,000). [...]
Base amount at $45,000: $4,288. [...]
Excess: $90,000 - $45,000 = $45,000. [...]
Tax on excess: $45,000 x 0.3 = $13,500. [...]
Tax payable: $4,288 + $13,500 = $17,788. [...]

Note: Excludes Medicare levy (2%).
Note: Rates checked against QC73320 on 2026-09-15.
```

**Treats the income year as part of the question.** Tax facts are year-scoped,
and the classic failure is a fluent, correctly cited answer carrying last
year's number. Chunks are stamped at ingest with the income years they mention;
the `year_consistent` guard blocks a claim citing a source that does not cover
the year asked about; and a value requested outside its stated years is refused
rather than adapted. If no year is stated, the assumption is disclosed in the
answer rather than made silently.

```
$ python -m atoqa.cli ask "What was the working from home rate for 2022-23?"

ABSTAIN  (intent: guidance, income year: 2022-23)

Not answered. Working from home fixed rate on record applies to 2024-25,
2025-26, not 2022-23. Refusing to apply a value outside its stated years.
```

**Knows the difference between "no answer" and "an answer to a nearby
question".** `question_covered` strips vocabulary common to all tax guidance,
then requires the question's *distinctive* terms to appear in the cited
passages. The case that forced it: "what is the instant asset write-off
threshold?" retrieves the GST registration page on the strength of the word
"threshold", and a well-behaved generator then writes a fluent, correctly cited
answer about something else entirely.

**Separates ordering from confidence.** Retrieval returns an RRF-fused ranking
*and* a separately calibrated score in [0,1] used for abstention. RRF values are
not comparable across queries, so using them to gate answers means the best of
six irrelevant passages looks like a genuine hit. Both calibration components
are absolute rather than relative to the result set — the first implementation
normalised BM25 by the best score in the set and gave an off-topic query 0.77.

**Refuses advice before a model sees it.** "Should I set up a trust to reduce
my tax?" is refused by the router and referred to a registered tax agent, with
`no_personal_advice` re-checking generated output as a second, independent
control. That boundary is a regulatory question, not a quality one, so it gets
two.

## The guards

| Guard | Blocks? | Catches |
|---|---|---|
| `citations_resolve` | yes | a claim citing something that was never retrieved |
| `numbers_supported` | yes | a figure absent from every cited source |
| `year_consistent` | yes | a source that does not cover the income year asked about |
| `question_covered` | yes | a fluent answer to a different question |
| `no_unsupported_claim` | yes | a claim sharing almost no vocabulary with its source |
| `no_personal_advice` | yes | recommendation language in generated output |
| `claims_grounded` | caveat | loose paraphrase — flagged, not blocked |

Number checking is unit-aware, so "70 cents" and "$0.70" are recognised as the
same fact and a correct answer is not flagged for rephrasing it.

## Evaluation

The harness is a **build gate**, not a report: `python evals/run_eval.py` exits
non-zero if any metric in `evals/thresholds.yaml` regresses, and CI runs it on
every push.

32 hand-written labelled questions. The suite includes four distinct
negative/decline categories — superseded, unanswerable, advice-seeking and
out-of-scope — alongside answerable and calculation tasks. Accuracy on
questions the corpus covers is the easy half of the measurement and the half
that flatters. The headline metric is `false_answer_rate`, not accuracy:
answers reaching a user as fluent, cited and wrong are the only error class
that actually costs anything.

Guard sensitivity is measured separately by **injecting faults on purpose**
(`llm.AdversarialStub`) — a fabricated figure, a bogus citation, a stale-year
citation, advice language, an unsupported claim — and measuring how many the
stack blocks. A green test suite only proves the guards do not misfire on good
output; a control never observed firing is not evidence of anything.

```
verdict_accuracy    1.0     false_answer_rate      0.0
numeric_accuracy    1.0     missed_answer_rate     0.0
citation_validity   1.0     forbidden_number_rate  0.0

guard sensitivity   49/49 injected faults blocked
```

## What the evaluation caught during development

During development the evaluation surfaced six concrete defects. They're
documented in the README, tests and evaluation notes, with later examples also
visible in the commit history:

- Keyword-only scope routing rejected an in-scope question about written
  evidence, leading to the retrieval-based rescue path.
- Normalising BM25 by the best result made an off-topic query look confident
  with a score of 0.77, leading to absolute confidence calibration.
- An instant asset write-off question retrieved GST guidance because both used
  "threshold", prompting the `question_covered` guard.
- A suffix-ordering bug stemmed "expenses" and "expense" differently, causing
  the coverage guard to reject valid evidence.
- The extractive generator padded a super guarantee answer with a
  working-from-home claim because both mentioned a "rate", prompting a relative
  claim cutoff.
- Passing every `top_k` result to the generator still exposed the same weak
  working-from-home passage, prompting provider-independent context pruning in
  the pipeline.

**Read those sceptically — and [docs/evaluation.md](docs/evaluation.md) says
why.** Two thresholds were tuned against this same 32-item set, so a perfect
score is partly overfitting and a held-out set is the honest next step. The
offline generator is extractive and therefore grounded by construction, which
flatters the groundedness metrics. Groundedness is lexical, not entailment: it
will not catch a fluent paraphrase that reverses a condition. The corpus is
four pages.

## Layout

```
src/atoqa/
  router.py       deterministic intent + income-year extraction
  rates.py        year-scoped reference tables with provenance and a verified gate
  calculator.py   tax arithmetic, with the working shown
  ingest.py       fetch, normalise, heading-aware chunking, income-year stamping
  index.py        hybrid BM25 + dense retrieval; RRF ordering, calibrated confidence
  embed.py        deterministic offline embedder | OpenAI | Azure OpenAI
  llm.py          extractive stub | adversarial stub | OpenAI | Azure OpenAI
  prompts.py      versioned prompt text
  guards.py       the six post-generation controls
  pipeline.py     the orchestrator — one decision path, all controls visible
  audit.py        decision traces in JSONL or PostgreSQL
  store.py        PostgreSQL index builds and append-only audit records
  api.py          FastAPI surface
evals/            labelled set, metrics, regression gates, harness
tests/            tests incl. reference-data self-validation; fixture corpus
docs/             architecture, evaluation methodology, governance
```

## Reading order

If you have ten minutes: [`docs/architecture.md`](docs/architecture.md) for the
deterministic/model/human split, then `src/atoqa/guards.py`, then
[`docs/evaluation.md`](docs/evaluation.md) for what the numbers do and do not
show.

## Known limitations

- Thresholds tuned on the evaluation set; no held-out set yet.
- Groundedness is lexical overlap, not entailment.
- `stem()` handles inflection but not derivation ("register" ≠ "registration").
  Recorded as a strict xfail in `tests/test_guards.py` rather than hidden.
- Four-page corpus; nothing here demonstrates retrieval at scale.
- No auth, rate limiting, tenancy or PII handling — see
  [docs/governance.md](docs/governance.md) for the full list of what production
  would require.
