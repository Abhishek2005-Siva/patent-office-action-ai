from pathlib import Path

import pytest

from patent_ai.data.hupd_loader import HUPDDataset

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "hupd_jan2016" / "sample" / "2016"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="HUPD sample not downloaded (see scripts/download_hupd_sample.py)"
)


@pytest.fixture(scope="module")
def dataset() -> HUPDDataset:
    return HUPDDataset(DATA_ROOT)


def test_dataset_loads_expected_count(dataset: HUPDDataset):
    assert len(dataset) == 26808


def test_sample_returns_real_applications(dataset: HUPDDataset):
    apps = dataset.sample(10, seed=1)
    assert len(apps) == 10
    for app in apps:
        assert app.application_number
        assert app.decision in {
            "ACCEPTED",
            "REJECTED",
            "PENDING",
            "CONT-ACCEPTED",
            "CONT-REJECTED",
            "CONT-PENDING",
        }
        assert len(app.claims) > 0


def test_sample_is_deterministic_with_seed(dataset: HUPDDataset):
    a = dataset.sample(5, seed=42)
    b = dataset.sample(5, seed=42)
    assert [x.application_number for x in a] == [x.application_number for x in b]


def test_first_claim_extracts_claim_one(dataset: HUPDDataset):
    apps = dataset.sample(25, seed=2)
    found_multi_claim = False
    for app in apps:
        if not app.claims:
            continue
        first = app.first_claim
        assert len(first) > 0
        assert len(first) <= len(app.claims)
        if first != app.claims.strip():
            found_multi_claim = True
    assert found_multi_claim, "expected at least one multi-claim application in sample"


def test_allowed_and_rejected_flags_are_consistent(dataset: HUPDDataset):
    apps = dataset.sample(50, seed=3)
    for app in apps:
        assert not (app.is_allowed and app.is_rejected)


def test_find_by_application_number_roundtrip(dataset: HUPDDataset):
    known = dataset.sample(1, seed=5)[0]
    found = dataset.find_by_application_number(known.application_number)
    assert found is not None
    assert found.application_number == known.application_number
    assert found.title == known.title
