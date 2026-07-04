"""Small Intern-S2-Preview chat client used by the harness."""

from __future__ import annotations

import json
from typing import Any

import httpx

from config import AppConfig


class LLMClientError(RuntimeError):
    pass


class InternS2Client:
    """OpenAI-compatible client for Intern-S2-Preview style chat APIs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def is_configured(self) -> bool:
        return bool(self.config.intern_api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
    ) -> str:
        if not self.is_configured():
            raise LLMClientError("INTERN_API_KEY is empty; cannot call Intern-S2-Preview.")

        payload: dict[str, Any] = {
            "model": self.config.intern_model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.config.intern_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.intern_api_base_url}/chat/completions"

        last_error: Exception | None = None
        for _ in range(self.config.max_llm_retries + 1):
            try:
                with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc

        raise LLMClientError(f"Intern-S2-Preview call failed: {last_error}") from last_error

    def json_chat(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> dict[str, Any]:
        content = self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Model did not return valid JSON: {content[:500]}") from exc
