import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "hupd_jan2016" / "sample" / "2016"

spec = importlib.util.spec_from_file_location("run_evaluation", ROOT / "scripts" / "run_evaluation.py")
run_evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evaluation)


def test_fixture_ordering_returns_well_formed_report():
    report = run_evaluation.evaluate_fixture_ordering()
    assert report["n"] == 7
    assert -1.0 <= report["spearman_correlation"] <= 1.0
    assert len(report["per_fixture"]) == 7
    for row in report["per_fixture"]:
        assert 0 <= row["predicted_strength"] <= 100


def test_fixture_ordering_correlation_is_reasonably_strong():
    # not a hard business requirement, but regressions here should be noticed
    report = run_evaluation.evaluate_fixture_ordering()
    assert report["spearman_correlation"] > 0.5


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="HUPD sample not downloaded")
def test_real_claim_breadth_report_well_formed_small_sample():
    report = run_evaluation.evaluate_real_claim_breadth(sample_size=200, seed=1)
    assert report["n_apps"] > 0
    assert 0 <= report["rejection_rate"] <= 1
    assert -1 <= report["claim_word_count"]["spearman_correlation"] <= 1


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="HUPD sample not downloaded")
def test_crowdedness_report_well_formed_small_sample():
    report = run_evaluation.evaluate_crowdedness_proxy(sample_size=150, seed=1, peers_per_app=3)
    assert report["n_apps"] >= 0
    assert -1 <= report["spearman_correlation"] <= 1
