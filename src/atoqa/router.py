"""Deterministic routing: decide what the model is allowed to be used for.

This is the first control in the system and it is intentionally boring --
regex and keyword tables, no model call. It answers three questions:

  1. Which income year is the user asking about? (and did they say, or are we
     assuming? tax facts are year-scoped, so an unstated year is surfaced,
     never silently guessed)
  2. Is this arithmetic we should do ourselves rather than ask a model to do?
  3. Is this a request for personal advice, which the system must refuse and
     route to a human rather than answer?
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .rates import THRESHOLDS_BY_KEY
from .schemas import Intent

# --- income year -------------------------------------------------------------

_YEAR_PATTERNS = [
    re.compile(r"\b(20\d{2})\s*[-/–]\s*(\d{2})\b"),      # 2025-26, 2025/26, 2025–26
    re.compile(r"\b(20\d{2})\s*[-/–]\s*(20\d{2})\b"),    # 2025-2026
    re.compile(r"\bFY\s*(20\d{2})\b", re.I),                   # FY2026
    re.compile(r"\bFY\s*(\d{2})\b", re.I),                     # FY26
]


def extract_income_year(text: str) -> str | None:
    """Return a normalised 'YYYY-YY' income year, or None if not stated."""
    for pat in _YEAR_PATTERNS[:2]:
        m = pat.search(text)
        if m:
            start = int(m.group(1))
            tail = m.group(2)
            end = int(tail) if len(tail) == 4 else 2000 + int(tail)
            if end == start + 1:
                return f"{start}-{str(end)[-2:]}"
            if len(tail) == 2 and int(tail) == (start + 1) % 100:
                return f"{start}-{tail}"
    m = _YEAR_PATTERNS[2].search(text)
    if m:                                   # FY2026 means the year ending 2026
        end = int(m.group(1))
        return f"{end - 1}-{str(end)[-2:]}"
    m = _YEAR_PATTERNS[3].search(text)
    if m:
        end = 2000 + int(m.group(1))
        return f"{end - 1}-{str(end)[-2:]}"
    return None


# --- amounts -----------------------------------------------------------------

_AMOUNT = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*([km])?\b", re.I)


def extract_amount(text: str) -> float | None:
    m = _AMOUNT.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    suffix = (m.group(2) or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


# --- intent ------------------------------------------------------------------

_CALC_CUES = (
    "how much tax", "tax on", "tax payable", "tax would i pay", "tax do i pay",
    "marginal rate", "marginal tax rate", "what bracket", "which bracket",
    "how much income tax",
)

_ADVICE_CUES = (
    "should i", "would you recommend", "what would you do", "is it worth",
    "advise me", "your advice", "best way to minimise", "best way to minimize",
    "how do i avoid paying", "how can i avoid paying", "loophole",
    "structure my affairs", "tell me what to do",
)

# Cheap pre-filter for "is this even about tax". It is deliberately generous:
# a false negative here is a refusal on a legitimate question, which is the
# worse error, and the pipeline re-checks any apparent out-of-scope question
# against retrieval before refusing. Its recall is measured by the
# out_of_scope and answerable categories of the evaluation set.
_TAX_VOCAB = (
    "tax", "ato", "gst", "super", "superannuation", "deduction", "deductible",
    "income year", "financial year", "bas", "abn", "tfn", "medicare",
    "capital gains", "cgt", "fbt", "payg", "withholding", "offset", "rebate",
    "threshold", "levy", "depreciation", "depreciating", "write-off", "write off",
    "claim", "claiming", "lodge", "lodging", "return", "invoice", "turnover",
    "bracket", "rate", "expense", "expenses", "evidence", "receipt", "receipts",
    "record", "records", "substantiate", "substantiation", "work-related",
    "working from home", "employer", "employee", "contribution", "assessable",
    "exemption", "concession", "commissioner", "audit",
)

_THRESHOLD_CUES: dict[str, tuple[str, ...]] = {
    "super_guarantee_rate": ("super guarantee", "sg rate", "how much super", "superannuation rate"),
    "super_maximum_contribution_base": ("maximum contribution base", "contribution base"),
    "gst_registration_turnover_nonprofit": ("non-profit", "not-for-profit", "nonprofit"),
    "gst_registration_turnover": ("gst threshold", "register for gst", "gst registration"),
    "wfh_fixed_rate": ("working from home", "work from home", "wfh", "fixed rate method"),
    "tax_free_threshold": ("tax-free threshold", "tax free threshold"),
}


def detect_threshold(text: str) -> str | None:
    low = text.lower()
    # Non-profit GST must win over the general GST cue, hence ordering.
    for key, cues in _THRESHOLD_CUES.items():
        if key not in THRESHOLDS_BY_KEY:
            continue
        if any(c in low for c in cues):
            if key == "gst_registration_turnover_nonprofit" and "gst" not in low:
                continue
            return key
    return None


@dataclass
class Route:
    intent: Intent
    income_year: str | None
    income_year_assumed: bool
    amount: float | None = None
    threshold_key: str | None = None
    reason: str = ""


def route(question: str) -> Route:
    low = question.lower().strip()
    stated_year = extract_income_year(question)
    year = stated_year or settings.default_income_year
    assumed = stated_year is None

    if any(cue in low for cue in _ADVICE_CUES):
        return Route(
            Intent.ADVICE_SEEKING, year, assumed,
            reason="Question asks for a recommendation about the user's own affairs.",
        )

    if not any(v in low for v in _TAX_VOCAB):
        return Route(
            Intent.OUT_OF_SCOPE, None, False,
            reason="No Australian tax subject matter detected.",
        )

    amount = extract_amount(question)
    if any(cue in low for cue in _CALC_CUES) and amount is not None:
        return Route(
            Intent.CALCULATION, year, assumed, amount=amount,
            reason="Arithmetic over published rates; computed deterministically.",
        )
    if "marginal" in low and amount is not None:
        return Route(Intent.CALCULATION, year, assumed, amount=amount,
                     reason="Marginal rate lookup.")

    return Route(
        Intent.GUIDANCE, year, assumed, amount=amount,
        threshold_key=detect_threshold(question),
        reason="Retrieval-grounded explanation required.",
    )


def extract_all_income_years(text: str) -> list[str]:
    """Every income year mentioned in a passage, in order of first appearance.

    Used at ingest time to stamp chunks with the years they apply to, which is
    what lets the superseded-rate guard compare a chunk's years against the
    year the user asked about.
    """
    found: list[str] = []
    for m in re.finditer(r"\b(20\d{2})\s*[-/–]\s*(\d{2,4})\b", text):
        start = int(m.group(1))
        tail = m.group(2)
        end = int(tail) if len(tail) == 4 else 2000 + int(tail)
        if end != start + 1:
            continue
        y = f"{start}-{str(end)[-2:]}"
        if y not in found:
            found.append(y)
    return found
