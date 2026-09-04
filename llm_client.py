"""Small Intern-S2-Preview chat client used by the harness."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from config import AppConfig


class LLMClientError(RuntimeError):
    pass


class InternS2Client:
    """OpenAI-compatible client for Intern-S2-Preview style chat APIs."""

    def __init__(
        self,
        config: AppConfig,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        min_interval: float = 2.1,
        thinking_effort: str | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key if api_key else config.intern_api_key
        self.base_url = (base_url or config.intern_api_base_url).rstrip("/")
        self.model_name = model_name or config.intern_model_name
        self.thinking_effort = thinking_effort
        self._last_call: float = 0.0
        self._min_interval: float = min_interval

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
        thinking_mode: bool = False,
    ) -> str:
        if not self.is_configured():
            raise LLMClientError("INTERN_API_KEY is empty; cannot call Intern-S2-Preview.")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if thinking_mode:
            payload["thinking_mode"] = thinking_mode
        if self.thinking_effort:
            # GLM-style reasoning endpoints: cap hidden thinking so the budget
            # survives to visible content (empty replies when reasoning eats it).
            payload["thinking"] = {"type": "enabled", "effort": self.thinking_effort}
        if response_format:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for _ in range(self.config.max_llm_retries + 1):
            # Rate limiting: Intern-S2 allows 1 request per 2s
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            try:
                with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
                    self._last_call = time.monotonic()
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc

        raise LLMClientError(f"Intern-S2-Preview call failed: {last_error}") from last_error

    def json_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        thinking_mode: bool = False,
    ) -> dict[str, Any]:
        content = self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            extracted = _extract_json_object(content)
            if extracted is not None:
                return extracted
            raise LLMClientError(f"Model did not return valid JSON: {content[:500]}") from exc


def heavy_llm_client(config: AppConfig) -> InternS2Client:
    """Client for the hard generative stages (survey writer, repair agent).

    HEAVY_LLM_BASE_URL / HEAVY_LLM_API_KEY / HEAVY_LLM_MODEL point at a stronger
    OpenAI-compatible endpoint (e.g. GLM); when unset, the Intern-S2-Preview
    config is used unchanged. Simple judging tasks keep using InternS2Client(cfg).
    """
    base_url = (os.getenv("HEAVY_LLM_BASE_URL") or "").strip()
    api_key = (os.getenv("HEAVY_LLM_API_KEY") or "").strip()
    if not (base_url and api_key):
        return InternS2Client(config)
    return InternS2Client(
        config,
        api_key=api_key,
        base_url=base_url,
        model_name=(os.getenv("HEAVY_LLM_MODEL") or "").strip() or None,
        min_interval=float(os.getenv("HEAVY_LLM_MIN_INTERVAL") or 2.1),
        thinking_effort=(os.getenv("HEAVY_LLM_THINKING_EFFORT") or "").strip() or None,
    )


def _extract_json_object(content: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    parsed_objects: list[tuple[int, int, dict[str, Any]]] = []
    for start, char in enumerate(content):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_objects.append((end, start, parsed))
    if not parsed_objects:
        return None
    return max(parsed_objects, key=lambda item: (item[0], item[1]))[2]
