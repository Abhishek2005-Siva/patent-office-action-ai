import pytest

from patent_ai.evaluation.metrics import rank_correlation, roc_auc


def test_rank_correlation_perfect_positive():
    corr, p = rank_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert corr == pytest.approx(1.0)


def test_rank_correlation_perfect_negative():
    corr, p = rank_correlation([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert corr == pytest.approx(-1.0)


def test_rank_correlation_no_relationship_near_zero():
    # a symmetric, non-monotonic pattern -> ~0 rank correlation
    corr, p = rank_correlation([1, 2, 3, 4, 5], [1, 5, 1, 5, 1])
    assert abs(corr) < 0.6


def test_rank_correlation_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        rank_correlation([1, 2, 3], [1, 2])


def test_rank_correlation_needs_at_least_two_points():
    with pytest.raises(ValueError):
        rank_correlation([1], [1])


def test_roc_auc_perfect_separation():
    scores = [10, 20, 30, 80, 90, 95]
    outcomes = [0, 0, 0, 1, 1, 1]
    assert roc_auc(outcomes, scores) == pytest.approx(1.0)


def test_roc_auc_inverted_scores_gives_low_auc():
    scores = [90, 80, 70, 20, 10, 5]
    outcomes = [0, 0, 0, 1, 1, 1]
    assert roc_auc(outcomes, scores) == pytest.approx(0.0)


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc([1, 1, 1], [10, 20, 30])
