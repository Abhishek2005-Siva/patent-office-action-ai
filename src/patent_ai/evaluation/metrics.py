"""Evaluation metrics from the build guide's Part 3: rank correlation and
ROC-AUC, as thin, testable wrappers around scipy/sklearn."""

from __future__ import annotations

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def rank_correlation(predicted_scores: list[float], actual_outcomes: list[float]) -> tuple[float, float]:
    """Returns (correlation, p_value). Guide's targets: >0.6 acceptable, >0.75 good."""
    if len(predicted_scores) != len(actual_outcomes):
        raise ValueError("predicted_scores and actual_outcomes must be the same length")
    if len(predicted_scores) < 2:
        raise ValueError("need at least 2 data points for a correlation")
    correlation, p_value = spearmanr(predicted_scores, actual_outcomes)
    return float(correlation), float(p_value)


def roc_auc(actual_outcomes: list[int], predicted_scores: list[float]) -> float:
    """Guide's targets: >0.75 good, >0.85 excellent. actual_outcomes must be
    binary (0/1) and predicted_scores must not be constant."""
    if len(set(actual_outcomes)) < 2:
        raise ValueError("roc_auc requires both classes (0 and 1) to be present")
    return float(roc_auc_score(actual_outcomes, predicted_scores))
