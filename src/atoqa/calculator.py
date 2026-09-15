"""Deterministic computations over the reference tables in `rates.py`.

No model is involved here. Given a taxable income and an income year this
returns the tax payable, the bracket applied, and the arithmetic that produced
it, so the working can be shown and checked. The language model's only job
downstream is to phrase the result -- it is never asked to do the sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rates import RESIDENT_RATES, THRESHOLDS_BY_KEY, Bracket, Threshold


class CalculationRefused(Exception):
    """Raised when the reference data cannot support a trustworthy answer."""


@dataclass
class Computation:
    kind: str
    income_year: str | None
    value: float
    unit: str
    working: list[str] = field(default_factory=list)
    source_url: str = ""
    qc_ref: str = ""
    checked_on: str = ""
    excludes: tuple[str, ...] = ()
    notes: str = ""

    def as_numbers(self) -> list[str]:
        """Figures this computation is allowed to put in an answer."""
        return [f"{self.value:,.2f}", f"{self.value:,.0f}"]


def _money(x: float) -> str:
    return f"${x:,.2f}" if x % 1 else f"${x:,.0f}"


def format_value(value: float, unit: str) -> str:
    """Render a reference value the way the source states it.

    This matters more than it looks: the `numbers_supported` guard compares the
    figures in an answer against the figures in the cited source, so a value
    rendered as "7.5e+04" would fail verification against a source that says
    "$75,000". Formatting is part of the control, not cosmetics.
    """
    n = f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"
    return {
        "aud": f"${n}",
        "aud_per_year": f"${n} per year",
        "aud_per_quarter": f"${n} per quarter",
        "cents_per_hour": f"{n} cents per hour",
        "percent": f"{n}%",
        "percent_of_ote": f"{n}% of ordinary time earnings",
    }.get(unit, f"{n} ({unit})")


def bracket_for(taxable_income: float, income_year: str) -> Bracket:
    table = RESIDENT_RATES[income_year]
    for b in table.brackets:
        if b.upper is None or taxable_income <= b.upper:
            return b
    return table.brackets[-1]


def income_tax(taxable_income: float, income_year: str) -> Computation:
    """Resident individual income tax, excluding Medicare levy."""
    if taxable_income < 0:
        raise CalculationRefused("Taxable income cannot be negative.")

    table = RESIDENT_RATES.get(income_year)
    if table is None:
        raise CalculationRefused(
            f"No verified rate table for {income_year}. "
            f"Available: {', '.join(sorted(RESIDENT_RATES, reverse=True))}."
        )
    if not table.verified:
        # A deliberate hard stop: unverified reference data is treated as absent.
        raise CalculationRefused(
            f"The {income_year} rate table has not been verified against its source."
        )

    b = bracket_for(taxable_income, income_year)
    if b.rate == 0:
        tax = 0.0
        working = [
            f"Taxable income {_money(taxable_income)} is at or below the "
            f"tax-free threshold of {_money(18_200)}.",
            "Tax payable: $0.",
        ]
    else:
        excess = taxable_income - (b.lower - 1)
        tax = b.base + excess * b.rate
        working = [
            f"Income year {income_year}; bracket "
            f"{_money(b.lower)}-{_money(b.upper) if b.upper else 'and over'} "
            f"at {b.rate:.4g} ({b.rate * 100:.4g}c per $1 over {_money(b.lower - 1)}).",
            f"Base amount at {_money(b.lower - 1)}: {_money(b.base)}.",
            f"Excess: {_money(taxable_income)} - {_money(b.lower - 1)} = {_money(excess)}.",
            f"Tax on excess: {_money(excess)} x {b.rate:.4g} = {_money(excess * b.rate)}.",
            f"Tax payable: {_money(b.base)} + {_money(excess * b.rate)} = {_money(tax)}.",
        ]

    return Computation(
        kind="resident_income_tax",
        income_year=income_year,
        value=round(tax, 2),
        unit="aud",
        working=working,
        source_url=table.source_url,
        qc_ref=table.qc_ref,
        checked_on=table.checked_on,
        excludes=table.excludes,
        notes=table.notes,
    )


def marginal_rate(taxable_income: float, income_year: str) -> Computation:
    table = RESIDENT_RATES.get(income_year)
    if table is None or not table.verified:
        raise CalculationRefused(f"No verified rate table for {income_year}.")
    b = bracket_for(taxable_income, income_year)
    return Computation(
        kind="marginal_rate",
        income_year=income_year,
        value=round(b.rate * 100, 4),
        unit="percent",
        working=[
            f"Taxable income {_money(taxable_income)} falls in the "
            f"{_money(b.lower)}-{_money(b.upper) if b.upper else 'and over'} bracket "
            f"for {income_year}.",
            f"Marginal rate: {b.rate * 100:.4g}%.",
        ],
        source_url=table.source_url,
        qc_ref=table.qc_ref,
        checked_on=table.checked_on,
        excludes=table.excludes,
    )


def lookup(key: str, income_year: str | None = None) -> Computation:
    """Look up a scalar threshold, enforcing year applicability."""
    t: Threshold | None = THRESHOLDS_BY_KEY.get(key)
    if t is None:
        raise CalculationRefused(f"Unknown threshold '{key}'.")
    if not t.verified:
        raise CalculationRefused(f"Threshold '{key}' has not been verified against its source.")
    if income_year and t.income_years and income_year not in t.income_years:
        raise CalculationRefused(
            f"{t.label} on record applies to {', '.join(t.income_years)}, "
            f"not {income_year}. Refusing to apply a value outside its stated years."
        )
    return Computation(
        kind=f"threshold:{key}",
        income_year=income_year if t.income_years else None,
        value=t.value,
        unit=t.unit,
        working=[f"{t.label}: {format_value(t.value, t.unit)}."],
        source_url=t.source_url,
        qc_ref=t.qc_ref,
        checked_on=t.checked_on,
        notes=t.notes,
    )
