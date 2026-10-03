"""analyze_office_action is the single pipeline entry point shared by the Flask and Streamlit apps,
and OpenAICompatibleLLMClient is how non-Claude models plug into it."""

import sys
import types

import pytest

from patent_ai.analysis import analyze_office_action
from patent_ai.generation.llm_client import FakeLLMClient, OpenAICompatibleLLMClient
from tests.fixtures.office_actions import ALL_FIXTURES

EXPECTED_KEYS = {
    "application_number", "office_action_type", "rejection_types", "rejected_claims",
    "cited_references", "overall_strength", "win_probability", "recommendation",
    "per_rejection", "response_letter", "used_llm",
}


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=[f.name for f in ALL_FIXTURES])
def test_template_mode_returns_complete_result_for_every_fixture(fixture):
    result = analyze_office_action(fixture.text)
    assert set(result) == EXPECTED_KEYS
    assert result["used_llm"] is False
    assert 0 <= result["overall_strength"] <= 100
    assert result["response_letter"].strip()


def test_llm_mode_uses_the_given_client():
    fake = FakeLLMClient()
    result = analyze_office_action(ALL_FIXTURES[0].text, llm_client=fake)
    assert fake.calls, "the LLM client should have been called"
    assert set(result) == EXPECTED_KEYS


def test_empty_text_does_not_crash():
    result = analyze_office_action("   ")
    assert result["rejection_types"] == []


def _install_fake_openai(monkeypatch):
    seen = {}

    class _Completions:
        def create(self, **kwargs):
            seen["request"] = kwargs
            message = types.SimpleNamespace(content="drafted argument")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _OpenAI:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    return seen


def test_openai_compatible_client_passes_endpoint_model_and_messages(monkeypatch):
    seen = _install_fake_openai(monkeypatch)
    client = OpenAICompatibleLLMClient(
        api_key="k", model="some/model", base_url="https://integrate.api.nvidia.com/v1")
    text = client.generate(system="be brief", prompt="argue", max_tokens=123)
    assert text == "drafted argument"
    assert seen["client"] == {"api_key": "k", "base_url": "https://integrate.api.nvidia.com/v1"}
    assert seen["request"]["model"] == "some/model"
    assert seen["request"]["max_tokens"] == 123
    assert [m["role"] for m in seen["request"]["messages"]] == ["system", "user"]


def test_openai_compatible_client_requires_a_key():
    with pytest.raises(RuntimeError):
        OpenAICompatibleLLMClient(api_key="", model="m")
