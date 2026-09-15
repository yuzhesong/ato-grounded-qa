# Fixture corpus

These files are **paraphrased summaries written for this test suite**. They are
not ATO text and must not be treated as a source of tax information.

Each file records the ATO page its facts were checked against (`source_url`,
`qc_ref`, `checked_on`). Rates and thresholds are factual values; the wording
around them is original, so no ATO material is redistributed in this
repository. See `docs/governance.md`.

They exist so that unit tests, the guard-sensitivity suite and CI run offline
and deterministically. The evaluation harness can run against either this
corpus (`--corpus tests/fixtures/corpus`) or the live corpus fetched into
`data/raw` (`make fetch && make eval`); the labelled answers in
`evals/questions.yaml` are written to hold for both.
