# Governance, data handling and what is missing

## ATO material is not redistributed

No ATO page content is committed to this repository. `atoqa.ingest.fetch`
downloads a fixed list of four public pages into `data/raw/`, which is
git-ignored, and records a manifest with each page's URL, fetch timestamp,
HTTP status and SHA-256 so a given index can be traced to what was actually
retrieved.

The fixture corpus in `tests/fixtures/corpus/` is **original wording written
for this test suite**. Rates and thresholds in it are factual values checked
against the ATO pages cited in each file's front matter; the prose around them
is not ATO text. This keeps the repository free of third-party content while
letting tests, the guard-sensitivity suite and CI run offline.

Anyone reusing this should check the ATO's current terms of use before
redistributing fetched content. Not redistributing it was the cheaper decision
and it made the test suite better, since tests now run without a network.

The seed list is explicit rather than crawled. A portfolio demo has no business
spidering a government site, and `fetch` is serial with a delay and a
descriptive user agent.

## Audit trail

Every request writes one audit record, including refusals and abstentions.
With `ATOQA_DATABASE_URL` unset, records are appended to
`data/traces/YYYY-MM-DD.jsonl`. With it set, records are inserted into
PostgreSQL's `traces` table. Each record carries:

- `trace_id`, timestamp, and the question plus its SHA-256
- intent, income year, and whether the year was **assumed** rather than stated
- provider, model name and `prompt_version`
- the threshold values in force at the time
- every retrieved chunk: id, calibrated score, dense and lexical ranks, source
  URL, applicable income years
- every claim and its citations
- every guard verdict with score and detail
- the final verdict and latency
- `index_build_id`: the database build loaded for the request, or null when no database index was supplied
- each retrieved passage's text SHA-256; together with the build ID and chunk ID, this identifies the stored passage and its metadata

This is not debug logging. In a regulated setting, "why did it say that, on
that date, from which source, under which thresholds" is part of the product,
and it has to be designed in rather than reconstructed from stdout later. The
question is hashed as well as stored so traces stay correlatable after the raw
text is redacted under a retention policy.

Recording the thresholds in force matters: an answer that was reasonable under
last month's abstention gate needs to be interpretable against that gate, not
today's.

## What is deliberately not built

This is a portfolio demonstration. It would need all of the following before
facing a real user, and the absence of each is a decision rather than an
oversight:

**Identity and access.** No authentication, authorisation, tenancy or rate
limiting on the API. `atoqa.api` is a thin wrapper over `pipeline.ask` so that
none of the control logic would have to move when those are added.

**PII.** Questions are stored in traces in plaintext. A real deployment needs
classification at the boundary, redaction before persistence, a retention
period, and a defensible answer to whether question text may be used to improve
the system. The SHA-256 alongside each question exists so traces survive
redaction of the text.

**Secrets and network.** Credentials come from environment variables, which is
right for local use and wrong for production; that becomes a managed secret
store with rotation. Egress needs pinning to the provider endpoint.

**Content freshness.** Rate tables in `rates.py` carry `checked_on` dates and a
`verified` flag, and the calculator refuses to compute for an unverified year —
but nothing *enforces* re-checking. A real system needs scheduled revalidation
against source, alerting when a page's content hash changes, and a hard expiry
after which a table stops being usable rather than merely stale. Tax content
changes on a legislated calendar, so this is predictable work, not monitoring
for surprises.

**Human review loop.** Abstentions and refusals are recorded but go nowhere. In
practice they are the most valuable output the system produces: a queue of
questions the tool could not answer is a prioritised list of what to add to the
corpus and where the guards are too tight.

**Cost and latency controls.** No caching, batching, token budget or timeout
policy. The offline provider makes these invisible during development, which is
convenient and misleading.

## Standing limitation

This tool reports what published ATO guidance says, with citations. It does not
give tax advice, and `Intent.ADVICE_SEEKING` refuses and refers rather than
attempting to. That boundary is enforced in the router before any model is
called, and re-checked after generation by the `no_personal_advice` guard —
two independent controls, because it is the one boundary whose breach is a
regulatory problem rather than a quality problem.
