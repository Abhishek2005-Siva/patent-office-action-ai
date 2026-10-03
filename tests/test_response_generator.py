from pathlib import Path

import pytest

from patent_ai.data.hupd_loader import HUPDDataset
from patent_ai.extraction.office_action_parser import parse_office_action
from patent_ai.generation.llm_client import FakeLLMClient
from patent_ai.generation.response_generator import LLMResponseGenerator, TemplateResponseGenerator
from patent_ai.scoring.rejection_scorer import score_office_action
from tests.fixtures.office_actions import ALL_FIXTURES, MULTI_REJECTION_MIXED, STRONG_103_WITH_MOTIVATION

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "hupd_jan2016" / "sample" / "2016"


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_template_generator_produces_grounded_letter(fixture):
    extraction = parse_office_action(fixture.text)
    score = score_office_action(extraction)
    letter = TemplateResponseGenerator().generate(
        extraction=extraction, office_action_score=score, applicant_name="Jane Doe"
    )

    assert not letter.used_llm
    assert "Jane Doe" in letter.full_text
    for statute in fixture.expected_statutes:
        assert f"35 U.S.C. {statute}" in letter.full_text
    for claim in fixture.expected_claims:
        assert str(claim) in letter.full_text
    for ref in extraction.all_cited_references:
        assert ref in letter.full_text


def test_template_generator_is_deterministic():
    extraction = parse_office_action(STRONG_103_WITH_MOTIVATION.text)
    score = score_office_action(extraction)
    gen = TemplateResponseGenerator()
    a = gen.generate(extraction=extraction, office_action_score=score)
    b = gen.generate(extraction=extraction, office_action_score=score)
    assert a.full_text == b.full_text


def test_llm_generator_uses_clean_llm_output_when_citations_are_valid():
    extraction = parse_office_action(STRONG_103_WITH_MOTIVATION.text)
    score = score_office_action(extraction)
    real_refs = extraction.rejections[0].cited_references
    fake = FakeLLMClient(
        responses=[f"The combination of {real_refs[0]} and {real_refs[1]} lacks a valid rationale."]
    )
    letter = LLMResponseGenerator(fake).generate(extraction=extraction, office_action_score=score)

    assert letter.used_llm
    assert letter.llm_fallback_count == 0
    assert "lacks a valid rationale" in letter.full_text
    assert len(fake.calls) == 1
    assert "103" in fake.calls[0]["prompt"]


def test_llm_generator_falls_back_when_llm_fabricates_a_citation():
    extraction = parse_office_action(STRONG_103_WITH_MOTIVATION.text)
    score = score_office_action(extraction)
    fake = FakeLLMClient(responses=["This is anticipated by US 1,000,000 which was never cited."])
    letter = LLMResponseGenerator(fake).generate(extraction=extraction, office_action_score=score)

    assert letter.llm_fallback_count == 1
    assert "US 1,000,000" not in letter.full_text
    # falls back to the deterministic template argument instead
    assert "motivation to combine" in letter.full_text.lower()


def test_llm_generator_handles_multi_rejection_document():
    extraction = parse_office_action(MULTI_REJECTION_MIXED.text)
    score = score_office_action(extraction)
    fake = FakeLLMClient(
        responses=[
            "Chen (US 6,777,888) does not disclose every claimed element.",
            "No motivation is shown to combine US 7,888,999 and US 8,999,000 with Chen.",
        ]
    )
    letter = LLMResponseGenerator(fake).generate(extraction=extraction, office_action_score=score)
    assert letter.llm_fallback_count == 0
    assert len(fake.calls) == 2


# --- Real-data integration: real applicant claim/spec text from HUPD fed
# through the generator alongside the synthetic office-action fixtures. ---

pytestmark_real = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="HUPD sample not downloaded (see scripts/download_hupd_sample.py)"
)


@pytestmark_real
def test_generators_run_end_to_end_on_real_applicant_claims():
    dataset = HUPDDataset(DATA_ROOT)
    apps = dataset.sample(5, seed=11)

    for app in apps:
        for fixture in (STRONG_103_WITH_MOTIVATION, MULTI_REJECTION_MIXED):
            extraction = parse_office_action(fixture.text)
            score = score_office_action(extraction)

            template_letter = TemplateResponseGenerator().generate(
                extraction=extraction, office_action_score=score, applicant_name=app.title[:40]
            )
            assert len(template_letter.full_text) > 100

            fake = FakeLLMClient()  # unscripted -> generic fake responses, exercises the full path
            llm_letter = LLMResponseGenerator(fake).generate(
                extraction=extraction,
                office_action_score=score,
                claims_text=app.claims[:2000],
                spec_text=app.background[:1000],
            )
            assert len(llm_letter.full_text) > 100
            assert len(fake.calls) == len(extraction.rejections)
