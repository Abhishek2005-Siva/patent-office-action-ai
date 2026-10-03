"""Per-statute argument templates for the offline, template-based generator.

Each template is filled only with facts already extracted/computed elsewhere
in the pipeline (claims, citations, scorer reasoning) -- it never invents
facts, so its output is safe to produce with no LLM involved.
"""

from __future__ import annotations

STATUTE_NAMES = {
    "101": "lack of patent-eligible subject matter",
    "102": "lack of novelty",
    "103": "obviousness",
    "112": "written description / definiteness",
}

_TEMPLATES = {
    "102": (
        "Claim(s) {claims} stand rejected under 35 U.S.C. 102 over {references}. "
        "{reasoning} "
        "Applicant respectfully submits that {references} does not disclose every element of "
        "claim {first_claim}, and therefore the rejection should be withdrawn or, in the "
        "alternative, the claim should be allowed as amended to more clearly recite the "
        "distinguishing feature(s) discussed above."
    ),
    "103": (
        "Claim(s) {claims} stand rejected under 35 U.S.C. 103 over {references}. "
        "{reasoning} "
        "Applicant respectfully submits that the Office has not established the requisite "
        "motivation to combine the cited references in the manner proposed, nor a reasonable "
        "expectation of success in doing so, as required under KSR Int'l Co. v. Teleflex Inc. "
        "and MPEP 2143."
    ),
    "112": (
        "Claim(s) {claims} stand rejected under 35 U.S.C. 112 for the reasons set forth in the "
        "Office Action. {reasoning} "
        "Applicant respectfully submits that the specification, as originally filed, provides "
        "adequate written description and enablement for the recited claim language, per MPEP "
        "2163."
    ),
    "101": (
        "Claim(s) {claims} stand rejected under 35 U.S.C. 101 as directed to a judicial "
        "exception. {reasoning} "
        "Applicant respectfully submits that the claims, considered as an ordered combination, "
        "recite a specific technical improvement that integrates any abstract idea into a "
        "practical application under Step 2A, Prong Two of the Office's Section 101 guidance "
        "(MPEP 2106)."
    ),
}


def render_rejection_argument(
    *,
    statute: str,
    claims: list[int],
    references: list[str],
    reasoning_bullets: list[str],
) -> str:
    template = _TEMPLATES.get(statute)
    if template is None:
        return f"Claim(s) {', '.join(map(str, claims))} stand rejected under 35 U.S.C. {statute}."

    claims_str = ", ".join(str(c) for c in claims) if claims else "(unspecified)"
    references_str = " and ".join(references) if references else "the cited reference(s)"
    reasoning_str = " ".join(reasoning_bullets)
    first_claim = claims[0] if claims else "1"

    return template.format(
        claims=claims_str,
        references=references_str,
        reasoning=reasoning_str,
        first_claim=first_claim,
    )
