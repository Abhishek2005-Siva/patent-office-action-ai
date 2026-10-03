import io

import pytest

from patent_ai.webapp.app import create_app
from tests.fixtures.office_actions import ALL_FIXTURES, MULTI_REJECTION_MIXED, STRONG_103_WITH_MOTIVATION
from tests.pdf_helper import build_minimal_pdf


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Patent Office Action Analyzer" in resp.data


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_analyze_endpoint_with_text_for_every_fixture(client, fixture):
    resp = client.post("/analyze", data={"office_action_text": fixture.text, "applicant_name": "Acme Inc."})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rejection_types"] == fixture.expected_statutes
    assert data["rejected_claims"] == fixture.expected_claims
    assert 0 <= data["overall_strength"] <= 100
    assert 0.05 <= data["win_probability"] <= 0.95
    assert "Acme Inc." in data["response_letter"]
    assert len(data["per_rejection"]) == len(fixture.expected_statutes) or len(data["per_rejection"]) >= 1


def test_analyze_endpoint_rejects_missing_text(client):
    resp = client.post("/analyze", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_endpoint_accepts_json_body(client):
    resp = client.post(
        "/analyze",
        json={"office_action_text": STRONG_103_WITH_MOTIVATION.text, "applicant_name": "JSON Corp"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rejection_types"] == ["103"]
    assert "JSON Corp" in data["response_letter"]


def test_analyze_endpoint_with_pdf_upload(client):
    pdf_bytes = build_minimal_pdf("Claims 1-5 are rejected under 35 U.S.C. 103.")
    resp = client.post(
        "/analyze",
        data={
            "office_action_file": (io.BytesIO(pdf_bytes), "office_action.pdf"),
            "applicant_name": "PDF Corp",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["rejection_types"] == ["103"]
    assert data["rejected_claims"] == [1, 2, 3, 4, 5]


def test_analyze_endpoint_rejects_non_pdf_upload(client):
    resp = client.post(
        "/analyze",
        data={"office_action_file": (io.BytesIO(b"not a pdf"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_analyze_endpoint_use_llm_without_api_key_falls_back_to_template(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post(
        "/analyze",
        data={
            "office_action_text": MULTI_REJECTION_MIXED.text,
            "use_llm": "on",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["used_llm"] is False  # no API key -> silently falls back, not an error
