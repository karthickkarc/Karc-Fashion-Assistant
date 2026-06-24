"""
Pluggable LLM client.

`LLMClient.complete(system, user)` is the single method the rest of the
codebase depends on (used by nlu.py for intent parsing and
explainability.py for polishing template reasoning). This keeps every
provider's SDK quirks in exactly one file -- swapping Gemini for OpenAI for
Claude is a config change, not a code change anywhere else.

If no provider is configured (the default), `get_llm_client()` returns
`None` and every caller in this codebase already has a deterministic,
template-based fallback path -- the system is fully functional with zero
API keys.
"""
from __future__ import annotations

from . import config


class LLMClient:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai  # lazy import
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model, system_instruction=system_default())
        self._genai = genai

    def complete(self, system: str, user: str) -> str:
        model = self._genai.GenerativeModel("gemini-1.5-flash", system_instruction=system)
        response = model.generate_content(user)
        return response.text


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # lazy import
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
        )
        return resp.choices[0].message.content


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic  # lazy import
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


def system_default() -> str:
    return "You are a helpful assistant."


def get_llm_client() -> LLMClient | None:
    provider = config.LLM_PROVIDER
    try:
        if provider == "gemini" and config.GEMINI_API_KEY:
            return GeminiClient(config.GEMINI_API_KEY)
        if provider == "openai" and config.OPENAI_API_KEY:
            return OpenAIClient(config.OPENAI_API_KEY)
        if provider == "anthropic" and config.ANTHROPIC_API_KEY:
            return AnthropicClient(config.ANTHROPIC_API_KEY)
    except ImportError:
        # SDK not installed for the configured provider -- fall back to
        # template-based behaviour rather than crashing the app.
        return None
    return None
