"""Year-scoped reference data, transcribed from ATO public pages.

Why this module exists
----------------------
Rates, thresholds and brackets are *facts*, not language. Asking a language
model to recall them is the single most likely way to produce a confident,
wrong, expensive answer. So they live here as data with provenance, and the
calculator below is the only thing allowed to compute with them.

Every table carries the URL and QC reference it was transcribed from and the
date it was checked. `verified` is a hard gate: the calculator refuses to
compute for a year whose table has not been checked against the source, rather
than silently using stale numbers.

The bracket tables are also internally redundant -- each bracket states both
its marginal rate and its cumulative base amount -- which lets
`tests/test_rates.py` recompute every base from the bracket below it and fail
on a transcription error. Reference data gets validated before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ATO_RESIDENT_RATES_URL = "https://www.ato.gov.au/tax-rates-and-codes/tax-rates-australian-residents"


@dataclass(frozen=True)
class Bracket:
    lower: int          # inclusive, in whole dollars
    upper: int | None   # inclusive; None means "and over"
    rate: float         # marginal rate applying above `lower - 1`
    base: float         # cumulative tax at `lower - 1`


@dataclass(frozen=True)
class RateTable:
    income_year: str
    brackets: tuple[Bracket, ...]
    source_url: str
    qc_ref: str
    checked_on: str
    verified: bool = False
    notes: str = ""
    excludes: tuple[str, ...] = field(default_factory=tuple)


# --- Resident individual income tax, excluding Medicare levy -----------------
# Transcribed from ATO "Tax rates - Australian resident" (QC73320).

RESIDENT_RATES: dict[str, RateTable] = {
    "2026-27": RateTable(
        income_year="2026-27",
        brackets=(
            Bracket(0, 18_200, 0.00, 0.0),
            Bracket(18_201, 45_000, 0.15, 0.0),
            Bracket(45_001, 135_000, 0.30, 4_020.0),
            Bracket(135_001, 190_000, 0.37, 31_020.0),
            Bracket(190_001, None, 0.45, 51_370.0),
        ),
        source_url=ATO_RESIDENT_RATES_URL,
        qc_ref="QC73320",
        checked_on="2026-09-15",
        verified=True,
        excludes=("Medicare levy (2%)",),
    ),
    "2025-26": RateTable(
        income_year="2025-26",
        brackets=(
            Bracket(0, 18_200, 0.00, 0.0),
            Bracket(18_201, 45_000, 0.16, 0.0),
            Bracket(45_001, 135_000, 0.30, 4_288.0),
            Bracket(135_001, 190_000, 0.37, 31_288.0),
            Bracket(190_001, None, 0.45, 51_638.0),
        ),
        source_url=ATO_RESIDENT_RATES_URL,
        qc_ref="QC73320",
        checked_on="2026-09-15",
        verified=True,
        excludes=("Medicare levy (2%)",),
    ),
    "2024-25": RateTable(
        income_year="2024-25",
        brackets=(
            Bracket(0, 18_200, 0.00, 0.0),
            Bracket(18_201, 45_000, 0.16, 0.0),
            Bracket(45_001, 135_000, 0.30, 4_288.0),
            Bracket(135_001, 190_000, 0.37, 31_288.0),
            Bracket(190_001, None, 0.45, 51_638.0),
        ),
        source_url=ATO_RESIDENT_RATES_URL,
        qc_ref="QC73320",
        checked_on="2026-09-15",
        verified=True,
        excludes=("Medicare levy (2%)",),
    ),
    "2023-24": RateTable(
        income_year="2023-24",
        brackets=(
            Bracket(0, 18_200, 0.00, 0.0),
            Bracket(18_201, 45_000, 0.19, 0.0),
            Bracket(45_001, 120_000, 0.325, 5_092.0),
            Bracket(120_001, 180_000, 0.37, 29_467.0),
            Bracket(180_001, None, 0.45, 51_667.0),
        ),
        source_url=ATO_RESIDENT_RATES_URL,
        qc_ref="QC73320",
        checked_on="2026-09-15",
        verified=True,
        excludes=("Medicare levy (2%)",),
    ),
}


# --- Scalar thresholds -------------------------------------------------------
# Kept separate from the bracket tables because they are looked up, not summed.

@dataclass(frozen=True)
class Threshold:
    key: str
    label: str
    value: float
    unit: str
    income_years: tuple[str, ...]
    source_url: str
    qc_ref: str
    checked_on: str
    verified: bool = False
    notes: str = ""


THRESHOLDS: tuple[Threshold, ...] = (
    Threshold(
        key="super_guarantee_rate",
        label="Super guarantee rate",
        value=12.0,
        unit="percent_of_ote",
        income_years=("2025-26", "2026-27"),
        source_url=(
            "https://www.ato.gov.au/businesses-and-organisations/super-for-employers/"
            "paying-super-contributions/how-much-super-to-pay"
        ),
        qc_ref="QC33745",
        checked_on="2026-09-15",
        verified=True,
        notes="12% of ordinary time earnings from 1 July 2025 (11.5% for 2024-25).",
    ),
    Threshold(
        key="super_maximum_contribution_base",
        label="Maximum super contribution base (per quarter)",
        value=62_500.0,
        unit="aud_per_quarter",
        income_years=("2025-26",),
        source_url=(
            "https://www.ato.gov.au/businesses-and-organisations/super-for-employers/"
            "paying-super-contributions/how-much-super-to-pay"
        ),
        qc_ref="QC33745",
        checked_on="2026-09-15",
        verified=True,
    ),
    Threshold(
        key="gst_registration_turnover",
        label="GST registration turnover threshold",
        value=75_000.0,
        unit="aud_per_year",
        income_years=(),
        source_url=(
            "https://www.ato.gov.au/businesses-and-organisations/"
            "gst-excise-and-indirect-taxes/gst/registering-for-gst"
        ),
        qc_ref="QC22412",
        checked_on="2026-09-15",
        verified=True,
        notes="$150,000 for non-profit bodies; taxi/ride-sourcing must register regardless.",
    ),
    Threshold(
        key="gst_registration_turnover_nonprofit",
        label="GST registration turnover threshold (non-profit)",
        value=150_000.0,
        unit="aud_per_year",
        income_years=(),
        source_url=(
            "https://www.ato.gov.au/businesses-and-organisations/"
            "gst-excise-and-indirect-taxes/gst/registering-for-gst"
        ),
        qc_ref="QC22412",
        checked_on="2026-09-15",
        verified=True,
    ),
    Threshold(
        key="wfh_fixed_rate",
        label="Working from home fixed rate",
        value=70.0,
        unit="cents_per_hour",
        income_years=("2024-25", "2025-26"),
        source_url=(
            "https://www.ato.gov.au/individuals-and-families/"
            "income-deductions-offsets-and-records/deductions-you-can-claim/"
            "working-from-home-expenses/fixed-rate-method-67-cents"
        ),
        qc_ref="QC72159",
        checked_on="2026-09-15",
        verified=True,
        notes="67 cents applied for 2022-23 and 2023-24; 52 cents for 2020-21 and 2021-22.",
    ),
    Threshold(
        key="tax_free_threshold",
        label="Tax-free threshold",
        value=18_200.0,
        unit="aud_per_year",
        income_years=("2023-24", "2024-25", "2025-26", "2026-27"),
        source_url=ATO_RESIDENT_RATES_URL,
        qc_ref="QC73320",
        checked_on="2026-09-15",
        verified=True,
    ),
)

THRESHOLDS_BY_KEY: dict[str, Threshold] = {t.key: t for t in THRESHOLDS}

SUPPORTED_INCOME_YEARS: tuple[str, ...] = tuple(sorted(RESIDENT_RATES, reverse=True))
