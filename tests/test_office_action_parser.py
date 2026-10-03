import pytest

from patent_ai.extraction.office_action_parser import _extract_citations, parse_office_action
from tests.fixtures.office_actions import ALL_FIXTURES, MULTI_REJECTION_MIXED


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_parse_extracts_expected_statutes(fixture):
    result = parse_office_action(fixture.text)
    assert result.rejection_types == fixture.expected_statutes


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_parse_extracts_expected_claims(fixture):
    result = parse_office_action(fixture.text)
    assert result.all_rejected_claims == fixture.expected_claims


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_parse_extracts_minimum_references(fixture):
    result = parse_office_action(fixture.text)
    assert len(result.all_cited_references) >= fixture.expected_min_references


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_parse_extracts_application_number(fixture):
    result = parse_office_action(fixture.text)
    assert result.application_number is not None
    assert "/" in result.application_number or result.application_number.isdigit()


def test_multi_rejection_splits_into_separate_records():
    result = parse_office_action(MULTI_REJECTION_MIXED.text)
    assert len(result.rejections) == 2
    first, second = result.rejections
    assert first.statute == "102"
    assert first.claims_rejected == [1, 2]
    assert "Chen" in first.text
    assert "Patel" not in first.text  # block should stop before next opener

    assert second.statute == "103"
    assert second.claims_rejected == [6, 7, 8]
    assert set(second.cited_references) >= {"US 7,888,999", "US 8,999,000"}


def test_office_action_type_detection():
    non_final = parse_office_action(MULTI_REJECTION_MIXED.text)
    assert non_final.office_action_type == "non-final"

    from tests.fixtures.office_actions import STRONG_103_WITH_MOTIVATION

    final = parse_office_action(STRONG_103_WITH_MOTIVATION.text)
    assert final.office_action_type == "final"


def test_citation_normalization_handles_variant_formats():
    result = parse_office_action(MULTI_REJECTION_MIXED.text)
    for ref in result.all_cited_references:
        assert ref.startswith("US ")
        assert "," in ref


def test_empty_text_yields_no_rejections():
    result = parse_office_action("This document contains no rejections at all.")
    assert result.rejections == []
    assert result.rejection_types == []
    assert result.all_rejected_claims == []


# --- Regression tests from real data: querying the real USPTO OCE office
# action dataset (patents-public-data.uspto_oce_office_actions, accessed via
# Kaggle's BigQuery integration) showed the original citation regex, which
# required a literal "US"/"U.S. Patent No." prefix, matched only 0.4%
# (21/5000) of real cited-reference identifiers -- most real citations are
# either a bare, parenthesized number (a later reference cited by name only,
# without repeating "US") or a foreign patent/publication (JP, WO, EP, ...).
# See docs/DATA_SOURCES.md for how this dataset was obtained and what it
# does/doesn't contain (structured fields, not office-action prose).


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Smith (US 5,111,222) in view of Jones (5,987,654).", ["US 5,111,222", "US 5,987,654"]),
        ("Tanaka (JP 2004-159476) discloses a similar mechanism.", ["JP 2004-159476"]),
        ("The combination further relies on the WO 2012136906 publication.", ["WO 2012136906"]),
        ("See also Chen (EP 1234567) and Osei (6,999,000).", ["EP 1234567", "US 6,999,000"]),
    ],
)
def test_extract_citations_catches_bare_and_foreign_references(text, expected):
    assert sorted(_extract_citations(text)) == sorted(expected)


@pytest.mark.parametrize(
    "text",
    [
        "Applicant's attorney is registered in CA and DE.",  # US state abbreviations, no number follows
        "The invention relates to a device (see Fig. 3).",  # parenthesized non-number
        "Applicant amended claim 5, 123 words long.",  # comma-separated numbers, not patent-number shaped
    ],
)
def test_extract_citations_avoids_false_positives(text):
    assert _extract_citations(text) == []
