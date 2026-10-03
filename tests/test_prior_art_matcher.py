from collections import defaultdict
from pathlib import Path

import pytest

from patent_ai.data.hupd_loader import HUPDDataset
from patent_ai.scoring.prior_art_matcher import extract_claim_elements, score_prior_art_relevance

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "hupd_jan2016" / "sample" / "2016"


def test_extract_claim_elements_splits_on_wherein():
    claim = (
        "1. A water bottle comprising a body; a cap removably attached to the body, "
        "wherein the cap includes a magnetic locking mechanism configured to engage a "
        "corresponding magnet on the body."
    )
    elements = extract_claim_elements(claim)
    assert len(elements) >= 2
    assert any("magnetic locking mechanism" in e for e in elements)


def test_extract_claim_elements_handles_empty_string():
    assert extract_claim_elements("") == []


def test_identical_text_scores_near_maximum_overlap():
    claim = "1. A widget comprising a housing; a fastener; and a spring wherein the spring biases the fastener."
    result = score_prior_art_relevance(claim, claim)
    assert result.overlap_score >= 80


def test_unrelated_text_scores_low_overlap():
    claim = "1. A water bottle comprising a body; a cap; and a magnetic locking mechanism."
    unrelated_reference = (
        "A method for baking bread comprising mixing flour and yeast, allowing the dough "
        "to rise for several hours, and baking the dough in an oven at high temperature."
    )
    result = score_prior_art_relevance(claim, unrelated_reference)
    assert result.overlap_score <= 30


def test_partial_overlap_scores_between_identical_and_unrelated():
    claim = (
        "1. A water bottle comprising a body; a cap removably attached to the body; "
        "wherein the cap includes a magnetic locking mechanism."
    )
    reference_same_field_no_lock = (
        "A drinking container comprising a body and a screw-on cap for sealing the body, "
        "the container being suitable for carrying beverages during outdoor activities."
    )
    partial = score_prior_art_relevance(claim, reference_same_field_no_lock).overlap_score
    identical = score_prior_art_relevance(claim, claim).overlap_score
    unrelated = score_prior_art_relevance(
        claim,
        "A method for baking bread comprising mixing flour and yeast and baking in an oven.",
    ).overlap_score
    assert unrelated < partial < identical


def test_empty_reference_yields_zero_overlap():
    result = score_prior_art_relevance("1. A widget comprising a housing.", "")
    assert result.overlap_score == 0


# --- Real-data validation: same-CPC-class patent pairs should overlap more
# than random cross-class pairs, using actual HUPD claim text. ---

pytestmark_real = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="HUPD sample not downloaded (see scripts/download_hupd_sample.py)"
)


@pytestmark_real
def test_same_cpc_class_claims_overlap_more_than_cross_class_real_data():
    dataset = HUPDDataset(DATA_ROOT)
    apps = dataset.sample(400, seed=7)

    by_class: dict[str, list] = defaultdict(list)
    for app in apps:
        if app.main_cpc_label and app.first_claim:
            by_class[app.main_cpc_label[:3]].append(app)

    # need classes with >=2 members to form same-class pairs
    same_class_scores = []
    for cls, members in by_class.items():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            result = score_prior_art_relevance(members[i].first_claim, members[i + 1].first_claim)
            same_class_scores.append(result.overlap_score)

    cross_class_scores = []
    classes = [c for c, members in by_class.items() if members]
    for i in range(min(len(classes) - 1, 60)):
        a = by_class[classes[i]][0]
        b = by_class[classes[i + 1]][0]
        result = score_prior_art_relevance(a.first_claim, b.first_claim)
        cross_class_scores.append(result.overlap_score)

    assert len(same_class_scores) >= 10, "need enough same-class pairs for a meaningful comparison"
    assert len(cross_class_scores) >= 10

    same_avg = sum(same_class_scores) / len(same_class_scores)
    cross_avg = sum(cross_class_scores) / len(cross_class_scores)

    assert same_avg > cross_avg, (
        f"expected same-CPC-class claims to overlap more on average "
        f"(same={same_avg:.1f}, cross={cross_avg:.1f})"
    )
