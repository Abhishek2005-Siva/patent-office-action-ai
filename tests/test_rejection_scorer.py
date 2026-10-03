import pytest

from patent_ai.extraction.office_action_parser import parse_office_action
from patent_ai.scoring.rejection_scorer import score_office_action
from tests.fixtures.office_actions import (
    ALL_FIXTURES,
    MODERATE_102_MISSING_ELEMENT,
    MULTI_REJECTION_MIXED,
    STRONG_101_ABSTRACT_IDEA,
    STRONG_102_ALL_ELEMENTS,
    STRONG_103_WITH_MOTIVATION,
    WEAK_103_NO_MOTIVATION,
    WEAK_112_ELEMENT_IN_SPEC,
)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_overall_strength_in_valid_range(fixture):
    extraction = parse_office_action(fixture.text)
    result = score_office_action(extraction)
    assert 0 <= result.overall_strength <= 100
    for rs in result.rejection_scores:
        assert 0 <= rs.strength <= 100
        assert len(rs.reasoning) > 0


def test_103_with_explicit_motivation_scores_higher_than_without():
    weak = score_office_action(parse_office_action(WEAK_103_NO_MOTIVATION.text))
    strong = score_office_action(parse_office_action(STRONG_103_WITH_MOTIVATION.text))
    assert strong.overall_strength > weak.overall_strength
    assert strong.overall_strength - weak.overall_strength >= 20


def test_102_all_elements_scores_higher_than_partial_disclosure():
    partial = score_office_action(parse_office_action(MODERATE_102_MISSING_ELEMENT.text))
    full = score_office_action(parse_office_action(STRONG_102_ALL_ELEMENTS.text))
    assert full.overall_strength > partial.overall_strength


def test_112_heavy_hedging_scores_low():
    result = score_office_action(parse_office_action(WEAK_112_ELEMENT_IN_SPEC.text))
    assert result.overall_strength < 50


def test_101_confident_alice_analysis_scores_high():
    result = score_office_action(parse_office_action(STRONG_101_ABSTRACT_IDEA.text))
    assert result.overall_strength >= 50


def test_multi_rejection_reports_per_rejection_breakdown():
    result = score_office_action(parse_office_action(MULTI_REJECTION_MIXED.text))
    assert len(result.rejection_scores) == 2
    statutes = {rs.statute for rs in result.rejection_scores}
    assert statutes == {"102", "103"}
    assert result.weakest is not None
    assert result.strongest is not None


def test_no_rejections_yields_zero_score():
    result = score_office_action(parse_office_action("No rejections here."))
    assert result.overall_strength == 0
    assert result.rejection_scores == []
    assert result.weakest is None
    assert result.strongest is None


def test_claims_weighting_favors_rejection_covering_more_claims():
    # multi_rejection_mixed: 102 covers claims [1,2] (weight 2), 103 covers [6,7,8] (weight 3)
    result = score_office_action(parse_office_action(MULTI_REJECTION_MIXED.text))
    by_statute = {rs.statute: rs.strength for rs in result.rejection_scores}
    manual_weighted = (by_statute["102"] * 2 + by_statute["103"] * 3) / 5
    assert result.overall_strength == round(manual_weighted)
