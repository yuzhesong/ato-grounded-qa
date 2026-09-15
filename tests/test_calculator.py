import pytest

from atoqa.calculator import (
    CalculationRefused,
    format_value,
    income_tax,
    lookup,
    marginal_rate,
)


@pytest.mark.parametrize(
    "income,year,expected",
    [
        (0, "2025-26", 0.0),
        (18_200, "2025-26", 0.0),          # top of the tax-free threshold
        (18_201, "2025-26", 0.16),         # first taxable dollar
        (45_000, "2025-26", 4_288.0),      # exact bracket boundary
        (90_000, "2025-26", 17_788.0),
        (135_000, "2025-26", 31_288.0),
        (190_000, "2025-26", 51_638.0),
        (200_000, "2025-26", 56_138.0),
        (90_000, "2023-24", 19_717.0),     # same income, different year
        (90_000, "2026-27", 17_520.0),
    ],
)
def test_income_tax_known_values(income, year, expected):
    assert income_tax(income, year).value == pytest.approx(expected, abs=0.01)


def test_boundary_is_continuous_across_brackets():
    """No jump in tax payable for one extra dollar of income."""
    for boundary in (45_000, 135_000, 190_000):
        below = income_tax(boundary, "2025-26").value
        above = income_tax(boundary + 1, "2025-26").value
        assert above - below < 1.0


def test_working_is_shown():
    comp = income_tax(90_000, "2025-26")
    assert len(comp.working) >= 4
    assert any("17,788" in line for line in comp.working)
    assert comp.qc_ref == "QC73320"


def test_unsupported_year_is_refused_not_guessed():
    with pytest.raises(CalculationRefused, match="No verified rate table"):
        income_tax(90_000, "2019-20")


def test_negative_income_is_refused():
    with pytest.raises(CalculationRefused):
        income_tax(-1, "2025-26")


@pytest.mark.parametrize(
    "income,expected", [(30_000, 16.0), (90_000, 30.0), (150_000, 37.0), (500_000, 45.0)]
)
def test_marginal_rate(income, expected):
    assert marginal_rate(income, "2025-26").value == pytest.approx(expected)


def test_threshold_lookup_inside_stated_years():
    comp = lookup("wfh_fixed_rate", "2025-26")
    assert comp.value == 70.0
    assert "70 cents per hour" in comp.working[0]


def test_threshold_lookup_outside_stated_years_is_refused():
    """The superseded-rate case: a value must not be applied to a year it does
    not cover, even though the number is correct for some other year."""
    with pytest.raises(CalculationRefused, match="not 2022-23"):
        lookup("wfh_fixed_rate", "2022-23")


def test_year_agnostic_threshold_accepts_any_year():
    assert lookup("gst_registration_turnover", "2023-24").value == 75_000.0


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        (75_000, "aud_per_year", "$75,000 per year"),
        (62_500, "aud_per_quarter", "$62,500 per quarter"),
        (70, "cents_per_hour", "70 cents per hour"),
        (12, "percent_of_ote", "12% of ordinary time earnings"),
        (18_200, "aud", "$18,200"),
    ],
)
def test_format_value_matches_how_sources_state_figures(value, unit, expected):
    """Formatting is load-bearing: the numbers_supported guard compares the
    answer's figures against the source's, so '7.5e+04' would fail verification
    against a page that says '$75,000'."""
    assert format_value(value, unit) == expected
