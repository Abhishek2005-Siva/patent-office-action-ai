"""Response letter generation.

TemplateResponseGenerator (Method 1 in the build guide) is fully offline and
deterministic: every fact in its output was already computed by the parser /
scorer, so it cannot hallucinate. LLMResponseGenerator (Method 2/3, "hybrid")
asks an LLMClient to turn the same structured facts into more natural prose,
then runs a verification pass that rejects any patent-number citation the
model invents that wasn't actually in the office action -- falling back to
the template sentence for that rejection if verification fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from patent_ai.extraction.office_action_parser import OfficeActionExtraction, PATENT_CITATION
from patent_ai.generation.llm_client import LLMClient
from patent_ai.generation.templates import STATUTE_NAMES, render_rejection_argument
from patent_ai.scoring.rejection_scorer import OfficeActionScore
from patent_ai.scoring.win_probability import predict_win_probability, recommendation_for_strength


@dataclass(frozen=True)
class ResponseLetter:
    full_text: str
    used_llm: bool
    llm_fallback_count: int  # number of sections where LLM output was rejected and template used instead


def _header(applicant_name: str, application_number: str | None, claims: list[int]) -> str:
    claims_str = ", ".join(str(c) for c in claims) if claims else "N/A"
    return (
        "RESPONSE TO OFFICE ACTION UNDER 37 CFR 1.111\n"
        f"Applicant: {applicant_name}\n"
        f"Application No.: {application_number or 'N/A'}\n"
        f"Rejected Claim(s): {claims_str}\n"
        f"Date: {date.today().strftime('%B %d, %Y')}\n"
    )


def _remarks_section(office_action_score: OfficeActionScore) -> str:
    lines = ["REMARKS", ""]
    lines.append(
        f"Overall rejection strength assessment: {office_action_score.overall_strength}/100. "
        f"{recommendation_for_strength(office_action_score.overall_strength)}"
    )
    win_prob = predict_win_probability(office_action_score.overall_strength)
    lines.append(f"Estimated probability of overcoming the rejection(s) as filed: {win_prob:.0%}.")
    return "\n".join(lines)


class TemplateResponseGenerator:
    def generate(
        self,
        *,
        extraction: OfficeActionExtraction,
        office_action_score: OfficeActionScore,
        applicant_name: str = "Applicant",
    ) -> ResponseLetter:
        parts = [_header(applicant_name, extraction.application_number, extraction.all_rejected_claims)]
        parts.append(_remarks_section(office_action_score))
        parts.append("")

        for rejection, score in zip(extraction.rejections, office_action_score.rejection_scores):
            statute_name = STATUTE_NAMES.get(rejection.statute, rejection.statute)
            parts.append(f"I. Rejection of Claim(s) under 35 U.S.C. {rejection.statute} ({statute_name})")
            parts.append(f"   Strength: {score.strength}/100")
            argument = render_rejection_argument(
                statute=rejection.statute,
                claims=rejection.claims_rejected,
                references=rejection.cited_references,
                reasoning_bullets=score.reasoning,
            )
            parts.append(argument)
            parts.append("")

        parts.append("Applicant respectfully requests reconsideration and allowance of the pending claims.")
        return ResponseLetter(full_text="\n".join(parts), used_llm=False, llm_fallback_count=0)


def _citations_in_text(text: str) -> set[str]:
    return {"US " + m.group(1).replace("/", ",") for m in PATENT_CITATION.finditer(text)}


class LLMResponseGenerator:
    """Hybrid generator: template structure + LLM-written argument prose,
    with a citation-fabrication guardrail."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._template_gen = TemplateResponseGenerator()

    def _llm_argument(self, *, rejection, score, claims_text: str, spec_text: str) -> str | None:
        system = (
            "You are a U.S. patent prosecution attorney drafting arguments for a response to "
            "an office action. Use ONLY the reference numbers given to you below -- never invent "
            "or cite any patent number that is not explicitly listed. Be specific and concise."
        )
        prompt = (
            f"Rejection under 35 U.S.C. {rejection.statute}, claim(s) {rejection.claims_rejected}.\n"
            f"Cited reference(s) (do not cite any other patent numbers): {rejection.cited_references}\n"
            f"Examiner's rejection text:\n{rejection.text}\n\n"
            f"Applicant's claim text:\n{claims_text}\n\n"
            f"Specification excerpt:\n{spec_text}\n\n"
            f"Automated strength assessment ({score.strength}/100) reasoning:\n"
            + "\n".join(f"- {r}" for r in score.reasoning)
            + "\n\nWrite one persuasive paragraph arguing against this rejection."
        )
        return self._llm.generate(system=system, prompt=prompt, max_tokens=600)

    def generate(
        self,
        *,
        extraction: OfficeActionExtraction,
        office_action_score: OfficeActionScore,
        applicant_name: str = "Applicant",
        claims_text: str = "",
        spec_text: str = "",
    ) -> ResponseLetter:
        parts = [_header(applicant_name, extraction.application_number, extraction.all_rejected_claims)]
        parts.append(_remarks_section(office_action_score))
        parts.append("")

        fallback_count = 0
        for rejection, score in zip(extraction.rejections, office_action_score.rejection_scores):
            statute_name = STATUTE_NAMES.get(rejection.statute, rejection.statute)
            parts.append(f"I. Rejection of Claim(s) under 35 U.S.C. {rejection.statute} ({statute_name})")
            parts.append(f"   Strength: {score.strength}/100")

            llm_text = self._llm_argument(
                rejection=rejection, score=score, claims_text=claims_text, spec_text=spec_text
            )
            allowed_citations = set(rejection.cited_references)
            cited_in_output = _citations_in_text(llm_text or "")

            if llm_text and cited_in_output.issubset(allowed_citations):
                parts.append(llm_text)
            else:
                fallback_count += 1
                parts.append(
                    render_rejection_argument(
                        statute=rejection.statute,
                        claims=rejection.claims_rejected,
                        references=rejection.cited_references,
                        reasoning_bullets=score.reasoning,
                    )
                )
            parts.append("")

        parts.append("Applicant respectfully requests reconsideration and allowance of the pending claims.")
        return ResponseLetter(
            full_text="\n".join(parts), used_llm=True, llm_fallback_count=fallback_count
        )
