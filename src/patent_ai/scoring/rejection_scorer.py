"""Rejection strength scoring: 0-100, where higher means the examiner's
rejection is harder to overcome (a *stronger* rejection), following the
factor analysis in the build guide, refined per statute.

This is a transparent, rule-based scorer (not an ML model) -- there is no
scriptable source of real, labeled office-action outcome data to train a
model on (see docs/DATA_SOURCES.md), so the design goal here is a
well-reasoned, auditable heuristic an attorney can sanity-check, in the
spirit of the guide's "start simple, 3-4 factors" advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from patent_ai.extraction.office_action_parser import OfficeActionExtraction, RejectionRecord
from patent_ai.scoring.feature_extraction import RejectionFeatures, extract_rejection_features


def _clamp(value: float, lo: float = 0, hi: float = 100) -> int:
    return int(round(max(lo, min(hi, value))))


@dataclass(frozen=True)
class RejectionScore:
    statute: str
    claims_rejected: list[int]
    strength: int  # 0-100
    reasoning: list[str]


def score_102(f: RejectionFeatures) -> RejectionScore:
    reasoning = []
    score = 30.0
    if f.all_elements_present_language:
        score += 30
        reasoning.append("Examiner asserts the reference discloses every claimed element (strong for a 102).")
    else:
        reasoning.append("Examiner does not clearly assert element-by-element anticipation (weaker for a 102).")
    if f.num_references == 0:
        score -= 15
        reasoning.append("No reference is actually cited for this rejection.")
    elif f.num_references > 1:
        score -= 10
        reasoning.append(
            f"{f.num_references} references cited for what should be a single-reference (102) rejection "
            "-- may indicate the examiner is really making a 103-style combination under a 102 heading."
        )
    score += f.confidence_score * 3
    score -= f.hedging_score * 4
    if f.hedging_score:
        reasoning.append(f"Hedging language detected ({f.hedging_score} instance(s)), weakening the rejection.")
    if f.confidence_score:
        reasoning.append(f"Confident/explicit language detected ({f.confidence_score} instance(s)).")
    return RejectionScore("102", [], _clamp(score), reasoning)


def score_103(f: RejectionFeatures) -> RejectionScore:
    reasoning = []
    score = 25.0
    if f.has_explicit_motivation:
        score += 30
        reasoning.append("Explicit motivation-to-combine language is present (strong for a 103).")
    else:
        score -= 20
        reasoning.append("No explicit motivation-to-combine language found -- a classic weakness to attack in a 103 response.")
    score += min(f.num_references * 5, 15)
    score += f.confidence_score * 3
    score -= f.hedging_score * 4
    if f.hedging_score:
        reasoning.append(f"Hedging language detected ({f.hedging_score} instance(s)), weakening the rejection.")
    if f.confidence_score:
        reasoning.append(f"Confident/explicit language detected ({f.confidence_score} instance(s)).")
    return RejectionScore("103", [], _clamp(score), reasoning)


def score_112(f: RejectionFeatures) -> RejectionScore:
    reasoning = []
    score = 50.0
    score -= f.hedging_score * 8
    score += f.confidence_score * 5
    if f.hedging_score:
        reasoning.append(
            f"Examiner's own language hedges {f.hedging_score} time(s) (e.g. 'may', 'appears') -- "
            "112 rejections resting on uncertain language are usually weak."
        )
    if f.confidence_score:
        reasoning.append(f"Confident/explicit language detected ({f.confidence_score} instance(s)).")
    if not f.hedging_score and not f.confidence_score:
        reasoning.append("No strong signal either way; strength defaults to neutral.")
    return RejectionScore("112", [], _clamp(score), reasoning)


def score_101(f: RejectionFeatures) -> RejectionScore:
    reasoning = []
    score = 45.0
    score += f.confidence_score * 4
    score -= f.hedging_score * 6
    if f.sentence_count >= 4:
        score += 5
        reasoning.append("Examiner provides a fuller Alice/Mayo two-step analysis (more sentences of reasoning).")
    if f.hedging_score:
        reasoning.append(f"Hedging language detected ({f.hedging_score} instance(s)), weakening the rejection.")
    if f.confidence_score:
        reasoning.append(f"Confident/explicit language detected ({f.confidence_score} instance(s)).")
    return RejectionScore("101", [], _clamp(score), reasoning)


_SCORERS = {"102": score_102, "103": score_103, "112": score_112, "101": score_101}


def score_single_rejection(rejection: RejectionRecord) -> RejectionScore:
    features = extract_rejection_features(rejection)
    scorer = _SCORERS.get(rejection.statute)
    if scorer is None:
        return RejectionScore(rejection.statute, rejection.claims_rejected, 50, ["Unrecognized statute; neutral default."])
    result = scorer(features)
    return RejectionScore(result.statute, rejection.claims_rejected, result.strength, result.reasoning)


@dataclass(frozen=True)
class OfficeActionScore:
    rejection_scores: list[RejectionScore]
    overall_strength: int
    weakest: RejectionScore | None
    strongest: RejectionScore | None


def score_office_action(extraction: OfficeActionExtraction) -> OfficeActionScore:
    """Score every rejection in the office action and roll up an overall,
    claims-weighted strength score.

    Rationale for claims-weighting: a rejection covering more claims has a
    proportionally bigger impact on the case as a whole.
    """
    if not extraction.rejections:
        return OfficeActionScore([], 0, None, None)

    scored: list[RejectionScore] = []
    for rejection in extraction.rejections:
        scored.append(score_single_rejection(rejection))

    total_weight = sum(max(1, len(s.claims_rejected)) for s in scored)
    weighted_sum = sum(s.strength * max(1, len(s.claims_rejected)) for s in scored)
    overall = _clamp(weighted_sum / total_weight) if total_weight else 0

    weakest = min(scored, key=lambda s: s.strength)
    strongest = max(scored, key=lambda s: s.strength)

    return OfficeActionScore(scored, overall, weakest, strongest)
