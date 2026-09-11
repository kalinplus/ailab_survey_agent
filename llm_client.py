"""Small Intern-S2-Preview chat client used by the harness."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from config import AppConfig

logger = logging.getLogger(__name__)


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
        request_timeout: float | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key if api_key else config.intern_api_key
        self.base_url = (base_url or config.intern_api_base_url).rstrip("/")
        self.model_name = model_name or config.intern_model_name
        self.thinking_effort = thinking_effort
        self._last_call: float = 0.0
        self._min_interval: float = min_interval
        # Output discipline: empty/whitespace replies observed so far; callers
        # can read this into their metrics.
        self.empty_reply_count = 0
        self.request_timeout: float = (
            request_timeout if request_timeout is not None else config.request_timeout_seconds
        )

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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
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
            # "disabled" turns thinking off entirely (DeepSeek-style reasoning
            # models burn the whole completion budget on reasoning_content
            # otherwise — grounded survey prose does not need it).
            if self.thinking_effort == "disabled":
                payload["thinking"] = {"type": "disabled"}
            else:
                payload["thinking"] = {"type": "enabled", "effort": self.thinking_effort}
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        message = self._post_chat(url, headers, payload)
        content = message.get("content")
        if _is_empty_reply(content):
            self._note_empty_reply()
            # Retry once after a fresh interval; the rate limiter inside
            # _post_chat enforces the min_interval gap between the two calls.
            message = self._post_chat(url, headers, payload)
            content = message.get("content")
            if _is_empty_reply(content):
                self._note_empty_reply()
                raise LLMClientError(
                    "empty reply after retry "
                    f"(model={self.model_name}, n_th={self.empty_reply_count})"
                )
        return content

    def tool_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Native tool-calling turn: returns the full assistant message.

        Unlike chat(), empty content is NOT an error here — a tool-calling
        turn legitimately carries its payload in ``tool_calls`` with null
        content (verified live on Intern-S2-Preview and DeepSeek by
        scripts/probe_tool_calling.py: structured tool_calls, valid args,
        role=tool round trip).
        """
        if not self.is_configured():
            raise LLMClientError("INTERN_API_KEY is empty; cannot call Intern-S2-Preview.")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self._post_chat(f"{self.base_url}/chat/completions", headers, payload)

    def _post_chat(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        """One logical chat request; retries transport-level errors only.

        Returns the full assistant message dict (content / tool_calls /
        reasoning_content), not just the content string — tool-calling turns
        carry their payload outside content. Empty-reply discipline lives in
        chat(), so an empty content returned here is a caller-visible outcome,
        not a transport failure.
        """
        last_error: Exception | None = None
        for _ in range(self.config.max_llm_retries + 1):
            # Rate limiting: Intern-S2 allows 1 request per 2s
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            try:
                with httpx.Client(timeout=self.request_timeout) as client:
                    self._last_call = time.monotonic()
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc

        raise LLMClientError(f"Intern-S2-Preview call failed: {last_error}") from last_error

    def _note_empty_reply(self) -> None:
        self.empty_reply_count += 1
        logger.warning(
            "[llm] empty reply (model=%s, n_th=%d)", self.model_name, self.empty_reply_count
        )

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
    Heavy calls get their own timeout budget: HEAVY_LLM_TIMEOUT_SECONDS
    (default 300) replaces the shared REQUEST_TIMEOUT_SECONDS so a long
    generation is not cut at the short default nor allowed to hang for hours.
    """
    heavy_timeout = float(os.getenv("HEAVY_LLM_TIMEOUT_SECONDS") or 300)
    base_url = (os.getenv("HEAVY_LLM_BASE_URL") or "").strip()
    api_key = (os.getenv("HEAVY_LLM_API_KEY") or "").strip()
    if not (base_url and api_key):
        return InternS2Client(config, request_timeout=heavy_timeout)
    return InternS2Client(
        config,
        api_key=api_key,
        base_url=base_url,
        model_name=(os.getenv("HEAVY_LLM_MODEL") or "").strip() or None,
        min_interval=float(os.getenv("HEAVY_LLM_MIN_INTERVAL") or 2.1),
        thinking_effort=(os.getenv("HEAVY_LLM_THINKING_EFFORT") or "").strip() or None,
        request_timeout=heavy_timeout,
    )


def _is_empty_reply(content: Any) -> bool:
    # Reasoning-heavy endpoints can return null or whitespace-only content when
    # hidden thinking eats the whole token budget; both count as empty.
    return not (isinstance(content, str) and content.strip())


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
