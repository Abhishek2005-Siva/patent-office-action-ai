"""LLM client abstraction so the response generator can be tested fully
offline (FakeLLMClient) and, when a key is configured, call a real model through
the exact same interface: Claude (AnthropicLLMClient, ANTHROPIC_API_KEY) or any
OpenAI-compatible endpoint such as OpenAI or NVIDIA's free hosted models
(OpenAICompatibleLLMClient).
"""

from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, *, system: str, prompt: str, max_tokens: int = 2000) -> str: ...


class AnthropicLLMClient:
    """Wraps the Anthropic Python SDK. Uses `api_key` if given, otherwise
    ANTHROPIC_API_KEY from the environment; raises at construction time if
    neither is set, so callers fail fast instead of failing deep inside a
    generation call."""

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key. Pass api_key or set ANTHROPIC_API_KEY to enable real "
                "LLM-generated arguments, or use TemplateResponseGenerator / FakeLLMClient instead."
            )
        from anthropic import Anthropic

        self._client = Anthropic(api_key=key)
        self._model = model

    def generate(self, *, system: str, prompt: str, max_tokens: int = 2000) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class OpenAICompatibleLLMClient:
    """Any OpenAI-compatible chat endpoint: OpenAI itself (base_url=None) or, for
    example, NVIDIA's free hosted models (base_url="https://integrate.api.nvidia.com/v1")."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise RuntimeError("An API key is required for OpenAICompatibleLLMClient.")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(self, *, system: str, prompt: str, max_tokens: int = 2000) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


class FakeLLMClient:
    """Deterministic stand-in for tests: records every call it receives and
    returns a scripted response (or an echo of the prompt if none is left)."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    def generate(self, *, system: str, prompt: str, max_tokens: int = 2000) -> str:
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        if self._responses:
            return self._responses.pop(0)
        return f"[FAKE RESPONSE for prompt of length {len(prompt)}]"


def is_llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
