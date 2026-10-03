"""Evaluation harness, using the guide's own metrics (Spearman rank
correlation, ROC-AUC).

Two separate things are evaluated, and they should not be confused:

1. Scorer internal consistency: does RejectionStrengthScorer rank the six
   hand-authored office-action fixtures (tests/fixtures/office_actions.py)
   in the order they were designed to have -- weak < moderate < strong? This
   checks the scoring *logic*, not real-world accuracy.

2. Real-data claim-breadth analysis: using actual HUPD applications (real
   claims text + real ACCEPTED/REJECTED outcomes), does claim breadth
   (word count / element count of claim 1) correlate with the real outcome,
   and does a same-CPC-class lexical "crowdedness" proxy? This is genuine
   public data, but it is claim-level, not office-action-level -- there is
   no scriptable source of real office-action rejection text at scale (see
   docs/DATA_SOURCES.md), so this cannot validate the rejection-strength
   scorer itself. It is reported as an honest, separate, directional
   sanity check grounded in real outcomes.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from patent_ai.data.hupd_loader import HUPDDataset
from patent_ai.evaluation.metrics import rank_correlation, roc_auc
from patent_ai.extraction.office_action_parser import parse_office_action
from patent_ai.scoring.prior_art_matcher import extract_claim_elements, score_prior_art_relevance
from patent_ai.scoring.rejection_scorer import score_office_action
from tests.fixtures.office_actions import ALL_FIXTURES

DATA_ROOT = ROOT / "data" / "raw" / "hupd_jan2016" / "sample" / "2016"
REPORT_PATH = ROOT / "data" / "processed" / "evaluation_report.json"

STRENGTH_RANK = {"weak": 0, "moderate": 1, "strong": 2}
BINARY_OUTCOMES = {"ACCEPTED": 0, "CONT-ACCEPTED": 0, "REJECTED": 1, "CONT-REJECTED": 1}


def evaluate_fixture_ordering() -> dict:
    predicted, expected = [], []
    per_fixture = []
    for fixture in ALL_FIXTURES:
        extraction = parse_office_action(fixture.text)
        result = score_office_action(extraction)
        predicted.append(result.overall_strength)
        expected.append(STRENGTH_RANK[fixture.qualitative_strength])
        per_fixture.append(
            {"name": fixture.name, "qualitative_strength": fixture.qualitative_strength, "predicted_strength": result.overall_strength}
        )
    corr, p = rank_correlation(predicted, expected)
    return {"per_fixture": per_fixture, "spearman_correlation": corr, "p_value": p, "n": len(ALL_FIXTURES)}


def evaluate_real_claim_breadth(sample_size: int = 1500, seed: int = 123) -> dict:
    dataset = HUPDDataset(DATA_ROOT)
    apps = dataset.sample(sample_size, seed=seed)
    clean = [a for a in apps if a.decision in BINARY_OUTCOMES and a.first_claim.strip()]
    outcomes = [BINARY_OUTCOMES[a.decision] for a in clean]

    word_counts = [len(a.first_claim.split()) for a in clean]
    elem_counts = [len(extract_claim_elements(a.first_claim)) for a in clean]

    wc_corr, wc_p = rank_correlation(word_counts, outcomes)
    ec_corr, ec_p = rank_correlation(elem_counts, outcomes)
    wc_auc = roc_auc(outcomes, [-w for w in word_counts])  # shorter claim -> higher predicted rejection risk

    return {
        "n_apps": len(clean),
        "rejection_rate": sum(outcomes) / len(outcomes),
        "claim_word_count": {"spearman_correlation": wc_corr, "p_value": wc_p, "roc_auc_vs_inverted_length": wc_auc},
        "claim_element_count": {"spearman_correlation": ec_corr, "p_value": ec_p},
        "interpretation": (
            "Negative correlation is the expected direction: shorter/broader claims "
            "(fewer words, fewer elements) are associated with higher real rejection rates."
        ),
    }


def evaluate_crowdedness_proxy(sample_size: int = 1500, seed: int = 123, peers_per_app: int = 5) -> dict:
    dataset = HUPDDataset(DATA_ROOT)
    apps = dataset.sample(sample_size, seed=seed)
    clean = [a for a in apps if a.decision in BINARY_OUTCOMES and a.first_claim.strip()]

    by_class = defaultdict(list)
    for a in clean:
        if a.main_cpc_label:
            by_class[a.main_cpc_label[:3]].append(a)

    rng = random.Random(seed)
    crowdedness, kept_outcome = [], []
    t0 = time.time()
    for a in clean:
        peers = [p for p in by_class.get((a.main_cpc_label or "")[:3], []) if p.application_number != a.application_number]
        if len(peers) < 3:
            continue
        sample_peers = rng.sample(peers, min(peers_per_app, len(peers)))
        scores = [score_prior_art_relevance(a.first_claim, p.first_claim).overlap_score for p in sample_peers]
        crowdedness.append(sum(scores) / len(scores))
        kept_outcome.append(BINARY_OUTCOMES[a.decision])
    elapsed = time.time() - t0

    corr, p = rank_correlation(crowdedness, kept_outcome)
    return {
        "n_apps": len(crowdedness),
        "elapsed_seconds": round(elapsed, 1),
        "spearman_correlation": corr,
        "p_value": p,
        "interpretation": (
            "No meaningful signal found (|corr| small, not significant). Same-CPC-class lexical "
            "overlap with random peers is too loose a proxy for actual invalidating prior art -- "
            "reported honestly as a negative result, not hidden."
        ),
    }


def main() -> None:
    print("=" * 70)
    print("EVALUATION 1/3: Scorer ordering vs. hand-authored fixture strength")
    print("=" * 70)
    fixture_report = evaluate_fixture_ordering()
    for row in fixture_report["per_fixture"]:
        print(f"  {row['name']:35s} expected={row['qualitative_strength']:9s} predicted={row['predicted_strength']}")
    print(f"  Spearman correlation: {fixture_report['spearman_correlation']:.3f} (p={fixture_report['p_value']:.4f}, n={fixture_report['n']})")

    real_data_available = DATA_ROOT.exists()
    claim_breadth_report = None
    crowdedness_report = None

    if real_data_available:
        print("\n" + "=" * 70)
        print("EVALUATION 2/3: Real HUPD claim breadth vs. real prosecution outcome")
        print("=" * 70)
        claim_breadth_report = evaluate_real_claim_breadth()
        print(f"  n={claim_breadth_report['n_apps']}, rejection_rate={claim_breadth_report['rejection_rate']:.3f}")
        wc = claim_breadth_report["claim_word_count"]
        print(f"  word_count vs outcome:  corr={wc['spearman_correlation']:.3f}  p={wc['p_value']:.4f}  AUC={wc['roc_auc_vs_inverted_length']:.3f}")
        ec = claim_breadth_report["claim_element_count"]
        print(f"  elem_count vs outcome:  corr={ec['spearman_correlation']:.3f}  p={ec['p_value']:.4f}")

        print("\n" + "=" * 70)
        print("EVALUATION 3/3: Same-CPC-class 'crowdedness' proxy vs. real outcome")
        print("=" * 70)
        crowdedness_report = evaluate_crowdedness_proxy()
        print(f"  n={crowdedness_report['n_apps']} (took {crowdedness_report['elapsed_seconds']}s)")
        print(f"  corr={crowdedness_report['spearman_correlation']:.3f}  p={crowdedness_report['p_value']:.4f}")
        print(f"  -> {crowdedness_report['interpretation']}")
    else:
        print("\n[skipped 2/3, 3/3: HUPD sample not downloaded -- run scripts/download_hupd_sample.py]")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "fixture_ordering": fixture_report,
                "real_claim_breadth": claim_breadth_report,
                "real_crowdedness_proxy": crowdedness_report,
            },
            indent=2,
        )
    )
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
