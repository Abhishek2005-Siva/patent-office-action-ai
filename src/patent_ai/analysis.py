"""One entry point for the whole pipeline: parse -> score -> win probability -> response letter.

The Flask app and the Streamlit app both call this, so they cannot drift apart.
"""

from __future__ import annotations

from patent_ai.extraction.office_action_parser import parse_office_action
from patent_ai.generation.llm_client import LLMClient
from patent_ai.generation.response_generator import LLMResponseGenerator, TemplateResponseGenerator
from patent_ai.scoring.rejection_scorer import score_office_action
from patent_ai.scoring.win_probability import predict_win_probability, recommendation_for_strength


def analyze_office_action(
    office_action_text: str,
    *,
    claims_text: str = "",
    spec_text: str = "",
    applicant_name: str = "Applicant",
    llm_client: LLMClient | None = None,
) -> dict:
    """Analyze an office action and draft a response.

    With `llm_client` the argument prose is model-written (hybrid mode, with the generator's
    guardrail against invented citations); without it the offline template generator is used.
    """
    extraction = parse_office_action(office_action_text)
    score = score_office_action(extraction)
    win_probability = predict_win_probability(score.overall_strength)
    recommendation = recommendation_for_strength(score.overall_strength)

    if llm_client is not None:
        letter = LLMResponseGenerator(llm_client).generate(
            extraction=extraction,
            office_action_score=score,
            applicant_name=applicant_name,
            claims_text=claims_text,
            spec_text=spec_text,
        )
    else:
        letter = TemplateResponseGenerator().generate(
            extraction=extraction, office_action_score=score, applicant_name=applicant_name
        )

    return {
        "application_number": extraction.application_number,
        "office_action_type": extraction.office_action_type,
        "rejection_types": extraction.rejection_types,
        "rejected_claims": extraction.all_rejected_claims,
        "cited_references": extraction.all_cited_references,
        "overall_strength": score.overall_strength,
        "win_probability": win_probability,
        "recommendation": recommendation,
        "per_rejection": [
            {
                "statute": rs.statute,
                "claims_rejected": rs.claims_rejected,
                "strength": rs.strength,
                "reasoning": rs.reasoning,
            }
            for rs in score.rejection_scores
        ],
        "response_letter": letter.full_text,
        "used_llm": letter.used_llm,
    }
