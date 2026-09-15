"""Validation of the reference data itself.

The rate tables are hand-transcribed from ATO pages, which means the most
likely defect in this repository is a typo in a number -- and a typo there is
invisible at runtime and wrong in every answer. So the tables are stored
redundantly (each bracket carries both its marginal rate and its cumulative
base) and this module recomputes one from the other.
"""

import pytest

from atoqa.rates import RESIDENT_RATES, SUPPORTED_INCOME_YEARS, THRESHOLDS


@pytest.mark.parametrize("year", sorted(RESIDENT_RATES))
def test_bracket_bases_are_internally_consistent(year):
    """Each bracket's base must equal tax accumulated through the bracket below.

    This is an independent check, not a restatement: the bases were transcribed
    from the source, and here they are derived from the widths and rates.
    """
    brackets = RESIDENT_RATES[year].brackets
    for lower, upper in zip(brackets, brackets[1:], strict=False):
        width = lower.upper - max(lower.lower - 1, 0)
        derived = lower.base + width * lower.rate
        assert derived == pytest.approx(upper.base, abs=0.01), (
            f"{year}: bracket starting {upper.lower} states base {upper.base} "
            f"but the bracket below derives {derived}"
        )


@pytest.mark.parametrize("year", sorted(RESIDENT_RATES))
def test_brackets_are_contiguous_and_ascending(year):
    brackets = RESIDENT_RATES[year].brackets
    assert brackets[0].lower == 0
    for lower, upper in zip(brackets, brackets[1:], strict=False):
        assert upper.lower == lower.upper + 1, f"{year}: gap or overlap at {lower.upper}"
        assert upper.rate >= lower.rate, f"{year}: marginal rate decreases at {upper.lower}"
    assert brackets[-1].upper is None


@pytest.mark.parametrize("year", sorted(RESIDENT_RATES))
def test_every_table_carries_provenance(year):
    t = RESIDENT_RATES[year]
    assert t.source_url.startswith("https://")
    assert t.qc_ref.startswith("QC")
    assert t.checked_on
    assert t.verified, "an unverified table must not ship"


@pytest.mark.parametrize("threshold", THRESHOLDS, ids=lambda t: t.key)
def test_every_threshold_carries_provenance(threshold):
    assert threshold.source_url.startswith("https://")
    assert threshold.qc_ref.startswith("QC")
    assert threshold.checked_on
    assert threshold.value > 0


def test_supported_years_are_newest_first():
    assert list(SUPPORTED_INCOME_YEARS) == sorted(SUPPORTED_INCOME_YEARS, reverse=True)
