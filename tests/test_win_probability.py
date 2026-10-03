import pytest

from patent_ai.scoring.win_probability import predict_win_probability, recommendation_for_strength


@pytest.mark.parametrize(
    "strength,expected_base",
    [(0, 0.75), (29, 0.75), (30, 0.60), (49, 0.60), (50, 0.40), (69, 0.40), (70, 0.20), (100, 0.20)],
)
def test_base_probability_tiers(strength, expected_base):
    prob = predict_win_probability(strength)
    # with no bonuses/penalties, amendment_bonus=0.05, spec_bonus=-0.05 -> factor (1.05)(0.95)=0.9975
    assert prob == pytest.approx(expected_base * 0.9975, abs=0.01)


def test_probability_is_monotonically_non_increasing_in_strength():
    probs = [predict_win_probability(s) for s in range(0, 101, 5)]
    assert all(a >= b - 1e-9 for a, b in zip(probs, probs[1:]))


def test_narrow_amendment_beats_no_amendment():
    base = predict_win_probability(50)
    narrowed = predict_win_probability(50, amendment_narrows_scope=True)
    assert narrowed > base


def test_explicit_spec_support_beats_no_support():
    base = predict_win_probability(50)
    supported = predict_win_probability(50, has_explicit_spec_support=True)
    assert supported > base


def test_examiner_allowance_rate_scales_probability():
    lenient = predict_win_probability(50, examiner_allowance_rate=0.80)
    strict = predict_win_probability(50, examiner_allowance_rate=0.10)
    assert lenient > strict


def test_probability_is_clamped_to_valid_range():
    assert 0.05 <= predict_win_probability(0, amendment_narrows_scope=True, has_explicit_spec_support=True) <= 0.95
    assert 0.05 <= predict_win_probability(100, examiner_allowance_rate=0.01) <= 0.95


def test_invalid_strength_raises():
    with pytest.raises(ValueError):
        predict_win_probability(-1)
    with pytest.raises(ValueError):
        predict_win_probability(101)


def test_invalid_examiner_rate_raises():
    with pytest.raises(ValueError):
        predict_win_probability(50, examiner_allowance_rate=1.5)


@pytest.mark.parametrize("strength", [0, 29, 30, 49, 50, 69, 70, 100])
def test_recommendation_always_returns_nonempty_string(strength):
    rec = recommendation_for_strength(strength)
    assert isinstance(rec, str) and len(rec) > 10
