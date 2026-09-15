"""Post-generation controls.

A language model asked to answer from retrieved context will usually do so.
"Usually" is not a control. Each guard below is a deterministic check that runs
after generation and can downgrade the verdict to a caveat or an abstention.
They are cheap, they need no second model call, and they fail closed.

  question_covered    the cited sources actually address what was asked
  citations_resolve   every cited id is a chunk that was actually retrieved
  numbers_supported   every figure in the answer appears in a cited chunk
  claims_grounded     each claim's wording overlaps the source it cites (advisory)
  no_unsupported_claim  no single claim is almost entirely absent from its source
  year_consistent     cited sources apply to the income year that was asked about
  no_personal_advice  the answer states the rules; it does not recommend actions

`numbers_supported` and `year_consistent` are the two that matter most in a tax
setting. Almost every dangerous answer here is a right-shaped sentence carrying
a number that is either invented or correct for the wrong income year.
"""

from __future__ import annotations

import re

from .config import settings
from .schemas import Chunk, Claim, DraftAnswer, GuardResult, Retrieved

# --- numeric extraction ------------------------------------------------------

_YEARISH = re.compile(r"\b20\d{2}\s*[-/–]\s*\d{2,4}\b")
# The digit group is `\d(?:[\d,]*\d)?` rather than `\d[\d,]*` so that it cannot
# end on a comma. "$75,000, threshold" was otherwise captured as "$75,000,"
# -- harmless for the parsed value, but it put stray punctuation into the
# guard's detail messages, which are read by a human deciding whether an answer
# was wrongly blocked.
_NUMERIC = re.compile(
    r"""
    (?P<dollar>\$\s?\d(?:[\d,]*\d)?(?:\.\d+)?)
  | (?P<percent>\d(?:[\d,]*\d)?(?:\.\d+)?\s?(?:%|per\s?cent|percent))
  | (?P<cents>\d(?:[\d,]*\d)?(?:\.\d+)?\s?(?:cents?|c)\b)
  | (?P<plain>\b\d(?:[\d,]*\d)?(?:\.\d+)?\b)
    """,
    re.X | re.I,
)

STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from had has have how i if in into is it
    its may must not of on or over per shall should so than that the their then there these this
    those to under up was were what when where which who why will with would you your""".split()
)


def _to_float(raw: str) -> float | None:
    digits = re.sub(r"[^\d.]", "", raw)
    if not digits or digits.count(".") > 1:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[float, str, str]]:
    """Return (value, unit, raw) for each figure. Income years are excluded.

    Units are tracked because '70 cents' and '$0.70' are the same fact, and a
    guard that treated them as different would flag correct answers.
    """
    cleaned = _YEARISH.sub(" ", text)
    out: list[tuple[float, str, str]] = []
    for m in _NUMERIC.finditer(cleaned):
        kind = m.lastgroup or "plain"
        raw = m.group(0)
        value = _to_float(raw)
        if value is None:
            continue
        unit = {"dollar": "aud", "percent": "percent", "cents": "cents"}.get(kind, "plain")
        out.append((value, unit, raw.strip()))
    return out


def _values_match(a: tuple[float, str, str], b: tuple[float, str, str]) -> bool:
    av, au, _ = a
    bv, bu, _ = b
    if au == bu or "plain" in (au, bu):
        return abs(av - bv) <= max(1e-6, abs(av) * 1e-9)
    if {au, bu} == {"cents", "aud"}:
        cents = av if au == "cents" else bv
        aud = bv if au == "cents" else av
        return abs(cents / 100.0 - aud) <= 1e-6
    return False


# --- individual guards -------------------------------------------------------

def citations_resolve(draft: DraftAnswer, retrieved: list[Retrieved]) -> GuardResult:
    valid = {r.chunk.id for r in retrieved}
    bad = sorted({c for claim in draft.claims for c in claim.citations if c not in valid})
    return GuardResult(
        name="citations_resolve",
        passed=not bad,
        score=0.0 if bad else 1.0,
        detail="" if not bad else f"cited ids not in retrieved set: {', '.join(bad)}",
    )


def _cited_text(claim: Claim, by_id: dict[str, Chunk]) -> str:
    return " ".join(by_id[c].text for c in claim.citations if c in by_id)


def numbers_supported(draft: DraftAnswer, retrieved: list[Retrieved]) -> GuardResult:
    by_id = {r.chunk.id: r.chunk for r in retrieved}
    unsupported: list[str] = []
    checked = 0
    for claim in draft.claims:
        source_nums = extract_numbers(_cited_text(claim, by_id))
        for num in extract_numbers(claim.text):
            checked += 1
            if not any(_values_match(num, s) for s in source_nums):
                unsupported.append(num[2])
    score = 1.0 if not checked else 1.0 - len(unsupported) / checked
    return GuardResult(
        name="numbers_supported",
        passed=not unsupported,
        score=round(score, 4),
        detail="" if not unsupported
        else f"figures absent from cited sources: {', '.join(sorted(set(unsupported)))}",
    )


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def claims_grounded(draft: DraftAnswer, retrieved: list[Retrieved]) -> GuardResult:
    """Lexical support per claim.

    This is a coarse proxy for entailment and it is honest about that: it
    catches a claim assembled from vocabulary the source never used, and it
    does not catch a fluent paraphrase that reverses a condition. An
    LLM-as-judge variant is a drop-in replacement at higher cost -- see
    docs/evaluation.md for the measured disagreement between the two.
    """
    by_id = {r.chunk.id: r.chunk for r in retrieved}
    scores: list[float] = []
    weak: list[str] = []
    for claim in draft.claims:
        claim_words = _content_words(claim.text)
        if not claim_words:
            continue
        source_words = _content_words(_cited_text(claim, by_id))
        overlap = len(claim_words & source_words) / len(claim_words)
        scores.append(overlap)
        if overlap < settings.min_groundedness:
            weak.append(f"{overlap:.2f}: {claim.text[:70]}")
    mean = sum(scores) / len(scores) if scores else 0.0
    return GuardResult(
        name="claims_grounded",
        passed=not weak,
        score=round(mean, 4),
        detail="" if not weak else "under-supported claims -> " + " | ".join(weak),
    )


# A claim scoring below this shares almost no vocabulary with the passage it
# cites -- not a loose paraphrase but a different statement. Separated from
# `claims_grounded` because the two failures deserve different responses: loose
# wording earns a caveat, an unsupported assertion has to block. One threshold
# doing both jobs meant a wholly fabricated claim was shipped with a footnote.
HARD_GROUNDING_FLOOR = 0.35


def no_unsupported_claim(draft: DraftAnswer, retrieved: list[Retrieved]) -> GuardResult:
    by_id = {r.chunk.id: r.chunk for r in retrieved}
    worst = 1.0
    offenders: list[str] = []
    for claim in draft.claims:
        claim_words = _content_words(claim.text)
        if not claim_words:
            continue
        source_words = _content_words(_cited_text(claim, by_id))
        overlap = len(claim_words & source_words) / len(claim_words)
        worst = min(worst, overlap)
        if overlap < HARD_GROUNDING_FLOOR:
            offenders.append(f"{overlap:.2f}: {claim.text[:70]}")
    return GuardResult(
        name="no_unsupported_claim",
        passed=not offenders,
        score=round(worst, 4),
        detail="" if not offenders else "claims not present in cited source -> "
        + " | ".join(offenders),
    )


def year_consistent(
    draft: DraftAnswer, retrieved: list[Retrieved], asked_year: str | None
) -> GuardResult:
    """Catch the superseded-rate failure.

    Tax facts are year-scoped. A chunk stamped with income years that do not
    include the year asked about is the single most common way to produce an
    answer that is fluent, cited, and wrong.
    """
    if not asked_year:
        return GuardResult(name="year_consistent", passed=True, score=None,
                           detail="no income year in scope")
    by_id = {r.chunk.id: r.chunk for r in retrieved}
    mismatches: list[str] = []
    for claim in draft.claims:
        years: set[str] = set()
        for cid in claim.citations:
            chunk = by_id.get(cid)
            if chunk:
                years.update(chunk.income_years)
        if years and asked_year not in years:
            mismatches.append(f"cites {sorted(years)} for a {asked_year} question")
    return GuardResult(
        name="year_consistent",
        passed=not mismatches,
        score=0.0 if mismatches else 1.0,
        detail="" if not mismatches else "; ".join(mismatches),
    )


# Words that appear across most tax guidance and therefore carry almost no
# information about WHICH page answers a question. Left out of coverage scoring.
GENERIC_TAX_WORDS = frozenset(
    """amount ato australia australian claim claimed claiming deduction deductions
    expense expenses financial income individual individuals liable pay payable paid
    rate rates records return returns tax taxable taxation threshold thresholds year
    years need needs long much many what which when where how does apply applies
    business businesses employee employer""".split()
)

# Tuned on the 32-item evaluation set. This is the least principled threshold
# in the system and the docs say so: 0.70 sits between "Medicare levy surcharge
# thresholds" (0.67, must fail -- the corpus discusses the levy but not the
# surcharge) and "do ride-sourcing drivers have to register for GST?" (0.75,
# must pass -- "drivers" is incidental). A margin that narrow is a heuristic
# holding a gap that a learned relevance model should eventually hold instead.
MIN_QUESTION_COVERAGE = 0.70

# Order matters, and the property that matters is CONSISTENCY, not linguistic
# correctness: "expense" and "expenses" must reach the same stem. An earlier
# ordering stripped "es" first, sending "expenses" to "expens" while "expense"
# stayed put -- so the two forms no longer matched and the coverage guard
# misfired on exactly the words it was added to handle.
_SUFFIXES = ("ing", "ed", "ly", "s")


def stem(word: str) -> str:
    """Deliberately crude suffix stripping.

    Only needs to collapse the variation that made the coverage check misfire:
    "cover"/"covers", "desk"/"desks", "expense"/"expenses".

    Known limitation, measured rather than assumed: this handles inflection but
    not derivation, so "register" and "registration" remain distinct stems and
    a question phrased with one against a source phrased with the other loses
    coverage it should keep. `tests/test_guards.py` records the case as an
    xfail so it stays visible. The fix is a real stemmer (Snowball) or an
    embedding-based coverage check; neither is vendored here because the
    measured effect on this corpus is one borderline item.
    """
    w = word.replace("-", "")
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _stems(words: set[str]) -> set[str]:
    out: set[str] = set()
    for w in words:
        out.add(stem(w))
        # "tax-free" should match a source that writes "tax free".
        if "-" in w:
            out.update(stem(part) for part in w.split("-") if len(part) > 2)
    return out


def _covered(term: str, source_stems: set[str]) -> bool:
    if stem(term) in source_stems:
        return True
    if "-" in term:
        parts = [p for p in term.split("-") if len(p) > 2]
        return bool(parts) and all(stem(p) in source_stems for p in parts)
    return False


def question_covered(
    draft: DraftAnswer,
    retrieved: list[Retrieved],
    question: str = "",
    corpus_vocabulary: set[str] | None = None,
) -> GuardResult:
    """Did the sources answer THIS question, or a nearby one?

    The failure this exists for: a question whose subject is absent from the
    corpus still retrieves the passages that share its generic vocabulary, and
    a well-behaved generator then writes a fluent, correctly cited answer about
    something else. "What is the instant asset write-off threshold?" retrieving
    the GST registration page on the strength of the word "threshold" is the
    real example that prompted this guard.

    Method: drop the vocabulary common to all tax guidance, then measure what
    share of the question's distinctive terms appear in the cited passages.
    Terms absent from the entire corpus are reported separately, because they
    are the strongest signal available that the corpus simply does not cover
    the topic -- though not, on measurement, a safe veto on their own: an
    incidental unknown word ("drivers") should not sink an answerable question.
    """
    if not question:
        return GuardResult(name="question_covered", passed=True, detail="no question supplied")
    salient = {w for w in _content_words(question) if w not in GENERIC_TAX_WORDS}
    if not salient:
        return GuardResult(
            name="question_covered", passed=True, score=None,
            detail="question has no distinctive terms to check",
        )

    by_id = {r.chunk.id: r.chunk for r in retrieved}
    cited = " ".join(
        f"{by_id[cid].text} {' '.join(by_id[cid].heading_path)} {by_id[cid].title}"
        for claim in draft.claims
        for cid in claim.citations
        if cid in by_id
    )
    cited_stems = _stems(_content_words(cited))
    covered = {w for w in salient if _covered(w, cited_stems)}
    ratio = len(covered) / len(salient)
    missing = sorted(salient - covered)

    note = ""
    if corpus_vocabulary is not None:
        corpus_stems = _stems(corpus_vocabulary)
        unknown = sorted(w for w in missing if not _covered(w, corpus_stems))
        if unknown:
            note = f"; absent from the whole corpus: {', '.join(unknown)}"

    passed = ratio >= MIN_QUESTION_COVERAGE
    return GuardResult(
        name="question_covered",
        passed=passed,
        score=round(ratio, 4),
        detail="" if passed else f"cited sources do not mention: {', '.join(missing)}{note}",
    )


_ADVICE = re.compile(
    r"\b(you should|i recommend|i'd recommend|i would recommend|i'd suggest|"
    r"my advice|the best option|you ought to|you must choose)\b",
    re.I,
)


def no_personal_advice(draft: DraftAnswer, *_: object) -> GuardResult:
    hits = sorted(
        {m.group(0).lower() for claim in draft.claims for m in _ADVICE.finditer(claim.text)}
    )
    return GuardResult(
        name="no_personal_advice",
        passed=not hits,
        score=0.0 if hits else 1.0,
        detail="" if not hits else f"recommendation language: {', '.join(hits)}",
    )


# Guards that must hold for the system to answer at all, versus guards whose
# failure only attaches a caveat. Splitting these is a product decision, so it
# lives in one visible place rather than scattered through the pipeline.
BLOCKING = (
    "citations_resolve",
    "numbers_supported",
    "year_consistent",
    "no_personal_advice",
    "question_covered",
    "no_unsupported_claim",
)
ADVISORY = ("claims_grounded",)


def run_all(
    draft: DraftAnswer,
    retrieved: list[Retrieved],
    asked_year: str | None,
    question: str = "",
    corpus_vocabulary: set[str] | None = None,
) -> list[GuardResult]:
    return [
        citations_resolve(draft, retrieved),
        numbers_supported(draft, retrieved),
        claims_grounded(draft, retrieved),
        no_unsupported_claim(draft, retrieved),
        year_consistent(draft, retrieved, asked_year),
        no_personal_advice(draft),
        question_covered(draft, retrieved, question, corpus_vocabulary),
    ]
