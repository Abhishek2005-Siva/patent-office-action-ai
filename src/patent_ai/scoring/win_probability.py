"""Win probability estimate from a rejection strength score.

Base tiers follow the build guide. `examiner_allowance_rate` is intentionally
optional: it would require a real historical per-examiner statistics dataset
(e.g. LexisNexis/Juristat/PatentsView examiner stats), which is not something
this project has scriptable, free access to -- see docs/DATA_SOURCES.md. When
omitted, the examiner factor is neutral (1.0) rather than silently guessing.
"""

from __future__ import annotations

AVERAGE_EXAMINER_ALLOWANCE_RATE = 0.40


def _base_probability(strength: int) -> float:
    if strength < 30:
        return 0.75
    if strength < 50:
        return 0.60
    if strength < 70:
        return 0.40
    return 0.20


def predict_win_probability(
    strength: int,
    *,
    examiner_allowance_rate: float | None = None,
    amendment_narrows_scope: bool = False,
    amendment_clarifies: bool = False,
    has_explicit_spec_support: bool = False,
    has_implicit_spec_support: bool = False,
) -> float:
    if not 0 <= strength <= 100:
        raise ValueError(f"strength must be in [0, 100], got {strength}")

    base = _base_probability(strength)

    examiner_factor = 1.0
    if examiner_allowance_rate is not None:
        if not 0 <= examiner_allowance_rate <= 1:
            raise ValueError("examiner_allowance_rate must be in [0, 1]")
        examiner_factor = examiner_allowance_rate / AVERAGE_EXAMINER_ALLOWANCE_RATE

    if amendment_narrows_scope:
        amendment_bonus = 0.15
    elif amendment_clarifies:
        amendment_bonus = 0.10
    else:
        amendment_bonus = 0.05

    if has_explicit_spec_support:
        spec_bonus = 0.10
    elif has_implicit_spec_support:
        spec_bonus = 0.05
    else:
        spec_bonus = -0.05

    probability = base * examiner_factor * (1 + amendment_bonus) * (1 + spec_bonus)
    return round(min(0.95, max(0.05, probability)), 3)


def recommendation_for_strength(strength: int) -> str:
    if strength < 30:
        return "WEAK REJECTION: strong candidate to fight with amendments and argument; high chance of overcoming."
    if strength < 50:
        return "MODERATE REJECTION: respond with narrow, targeted amendments; reasonable chance of overcoming."
    if strength < 70:
        return "STRONG REJECTION: consider an RCE with substantive amendments, or narrow the claims significantly."
    return "VERY STRONG REJECTION: consider examiner interview, significant claim narrowing, or appeal strategy."
