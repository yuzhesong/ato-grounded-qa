import pytest

from atoqa.router import (
    detect_threshold,
    extract_all_income_years,
    extract_amount,
    extract_income_year,
    route,
)
from atoqa.schemas import Intent


@pytest.mark.parametrize(
    "text,expected",
    [
        ("rates for 2025-26", "2025-26"),
        ("rates for 2025/26", "2025-26"),
        ("rates for 2025–26", "2025-26"),
        ("rates for 2025-2026", "2025-26"),
        ("what about FY2026", "2025-26"),
        ("what about FY26", "2025-26"),
        ("no year mentioned here", None),
        ("2025-27 is not an income year", None),
    ],
)
def test_extract_income_year(text, expected):
    assert extract_income_year(text) == expected


def test_extract_all_income_years_preserves_order_and_dedupes():
    text = "70c for 2024-25 and 2025-26; 67c for 2022-23 and 2023-24; also 2025-26 again"
    assert extract_all_income_years(text) == ["2024-25", "2025-26", "2022-23", "2023-24"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$90,000", 90_000.0),
        ("$90000", 90_000.0),
        ("$90k", 90_000.0),
        ("$1.5m", 1_500_000.0),
        ("$18,200.50", 18_200.50),
        ("no amount", None),
    ],
)
def test_extract_amount(text, expected):
    assert extract_amount(text) == expected


@pytest.mark.parametrize(
    "question,intent",
    [
        ("How much tax on $90,000 for 2025-26?", Intent.CALCULATION),
        ("What is my marginal tax rate on $150,000?", Intent.CALCULATION),
        ("What is the GST registration threshold?", Intent.GUIDANCE),
        ("Should I set up a trust to reduce tax?", Intent.ADVICE_SEEKING),
        ("What is the best way to minimise my tax?", Intent.ADVICE_SEEKING),
        ("Who won the 1998 AFL grand final?", Intent.OUT_OF_SCOPE),
    ],
)
def test_intent_classification(question, intent):
    assert route(question).intent is intent


def test_advice_beats_calculation():
    """A question that is both arithmetic and advice-seeking must refuse.

    Ordering matters here: routing to the calculator first would answer the
    sum and silently drop the part that needed a human."""
    r = route("I earn $90,000 -- how much tax should I be paying and should I salary sacrifice?")
    assert r.intent is Intent.ADVICE_SEEKING


def test_unstated_year_is_flagged_as_assumed_not_silently_defaulted():
    stated = route("what is the wfh rate for 2024-25?")
    assert stated.income_year == "2024-25" and stated.income_year_assumed is False
    unstated = route("what is the wfh rate?")
    assert unstated.income_year is not None and unstated.income_year_assumed is True


@pytest.mark.parametrize(
    "question,key",
    [
        ("what is the gst threshold", "gst_registration_turnover"),
        ("gst threshold for a non-profit", "gst_registration_turnover_nonprofit"),
        ("super guarantee rate", "super_guarantee_rate"),
        ("working from home fixed rate", "wfh_fixed_rate"),
        ("what is the tax-free threshold", "tax_free_threshold"),
        ("how do I lodge my return", None),
    ],
)
def test_detect_threshold(question, key):
    assert detect_threshold(question) == key
