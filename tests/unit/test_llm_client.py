"""Unit tests for llm_client output discipline (GLM 输出纪律, batch 5 T11).

Spec: specs/GLM输出纪律.md — chat/json_chat retry an empty (or whitespace-only)
reply once after min_interval, then raise LLMClientError; empty_reply_count
tracks observed empties; heavy_llm_client applies the HEAVY_LLM_TIMEOUT_SECONDS
budget (default 300) instead of the shared REQUEST_TIMEOUT_SECONDS.

No network: httpx.Client is monkeypatched with a scripted fake that records
POST requests and constructor timeouts, mirroring the real contract shape.
"""

import logging
from pathlib import Path

import httpx
import pytest

import llm_client
from config import AppConfig
from llm_client import InternS2Client, LLMClientError, heavy_llm_client

HEAVY_ENV_VARS = (
    "HEAVY_LLM_BASE_URL",
    "HEAVY_LLM_API_KEY",
    "HEAVY_LLM_MODEL",
    "HEAVY_LLM_MIN_INTERVAL",
    "HEAVY_LLM_THINKING_EFFORT",
    "HEAVY_LLM_TIMEOUT_SECONDS",
)


def _config() -> AppConfig:
    return AppConfig(
        root_dir=Path("/tmp/evisurvey-test"),
        intern_api_base_url="http://intern.test/v1",
        intern_api_key="test-key",
        intern_model_name="intern-s2-preview",
        sciverse_api_token="",
        sciverse_api_base_url="https://api.sciverse.space",
        strategy_probing_enabled=False,
        strategy_probe_limit=0,
        strategy_cluster_count=4,
        strategy_memory_enabled=True,
        strategy_memory_max_chars=4000,
        request_timeout_seconds=60.0,
        max_llm_retries=1,
        tool_timeout_seconds=30.0,
        tool_result_max_chars=2000,
        default_language="zh",
        default_mode="demo",
    )


def _chat_body(content):
    return {"choices": [{"message": {"content": content}}]}


def _install_fake_http(monkeypatch, script):
    """Replace httpx.Client with a scripted fake (zero production intrusion).

    script entries: dict -> 200 response with that JSON body; Exception
    instance -> raised as a transport failure. One entry is consumed per POST.
    Returns a recorder with `requests` (one entry per POST) and `timeouts`
    (one entry per httpx.Client construction).
    """
    recorder = {"requests": [], "timeouts": []}
    steps = list(script)

    class FakeClient:
        def __init__(self, timeout=None):
            recorder["timeouts"].append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, url, headers=None, json=None):
            recorder["requests"].append({"url": url, "headers": headers, "payload": json})
            step = steps.pop(0)
            if isinstance(step, Exception):
                raise step
            # raise_for_status() requires a request attached to the response.
            return httpx.Response(200, json=step, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "Client", FakeClient)
    return recorder


def _clear_heavy_env(monkeypatch, **set_env):
    # heavy_llm_client reads os.environ directly; isolate from .env leakage
    # (load_dotenv in other tests persists HEAVY_LLM_* across the process).
    for var in HEAVY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in set_env.items():
        monkeypatch.setenv(var, value)


def test_empty_reply_retries_once_and_returns_content(monkeypatch):
    recorder = _install_fake_http(monkeypatch, [_chat_body(""), _chat_body("solid reply")])
    client = InternS2Client(_config(), min_interval=0)

    assert client.chat([{"role": "user", "content": "hi"}]) == "solid reply"
    assert client.empty_reply_count == 1
    assert len(recorder["requests"]) == 2
    # The retry resends the exact same payload (structure untouched).
    assert recorder["requests"][0]["payload"] == recorder["requests"][1]["payload"]
    assert recorder["requests"][1]["payload"]["model"] == "intern-s2-preview"


def test_whitespace_only_reply_counts_as_empty(monkeypatch):
    _install_fake_http(monkeypatch, [_chat_body("   \n\t"), _chat_body("ok")])
    client = InternS2Client(_config(), min_interval=0)

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert client.empty_reply_count == 1


def test_two_empty_replies_raise_llm_client_error(monkeypatch):
    recorder = _install_fake_http(monkeypatch, [_chat_body(""), _chat_body(" ")])
    client = InternS2Client(_config(), min_interval=0)

    with pytest.raises(LLMClientError, match="empty reply"):
        client.chat([{"role": "user", "content": "hi"}])
    assert client.empty_reply_count == 2
    # One retry, then give up — no third request.
    assert len(recorder["requests"]) == 2


def test_empty_reply_retry_waits_min_interval(monkeypatch):
    _install_fake_http(monkeypatch, [_chat_body(""), _chat_body("ok")])
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = InternS2Client(_config(), min_interval=2.1)

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 2.1


def test_empty_reply_logs_warning_with_model_and_count(monkeypatch, caplog):
    _install_fake_http(monkeypatch, [_chat_body(""), _chat_body("ok")])
    client = InternS2Client(_config(), min_interval=0)

    with caplog.at_level(logging.WARNING, logger="llm_client"):
        client.chat([{"role": "user", "content": "hi"}])

    warnings = [r for r in caplog.records if "empty reply" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "model=intern-s2-preview" in message
    assert "n_th=1" in message


def test_json_chat_inherits_empty_reply_discipline(monkeypatch):
    _install_fake_http(monkeypatch, [_chat_body(""), _chat_body('{"items": [1, 2]}')])
    client = InternS2Client(_config(), min_interval=0)

    assert client.json_chat([{"role": "user", "content": "hi"}]) == {"items": [1, 2]}
    assert client.empty_reply_count == 1


def test_transport_error_retry_still_works(monkeypatch):
    _install_fake_http(monkeypatch, [httpx.ConnectError("boom"), _chat_body("ok")])
    client = InternS2Client(_config(), min_interval=0)

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert client.empty_reply_count == 0


def test_heavy_client_timeout_defaults_to_300(monkeypatch):
    _clear_heavy_env(monkeypatch)
    config = _config()

    heavy = heavy_llm_client(config)

    assert heavy.request_timeout == 300.0
    # The shared config default stays untouched (Intern simple tasks).
    assert config.request_timeout_seconds == 60.0


def test_heavy_client_timeout_reads_env(monkeypatch):
    _clear_heavy_env(
        monkeypatch,
        HEAVY_LLM_BASE_URL="http://glm.test/v4",
        HEAVY_LLM_API_KEY="heavy-key",
        HEAVY_LLM_TIMEOUT_SECONDS="123.5",
    )

    heavy = heavy_llm_client(_config())

    assert heavy.request_timeout == 123.5
    assert heavy.base_url == "http://glm.test/v4"


def test_heavy_timeout_reaches_http_request(monkeypatch):
    _clear_heavy_env(
        monkeypatch,
        HEAVY_LLM_BASE_URL="http://glm.test/v4",
        HEAVY_LLM_API_KEY="heavy-key",
    )
    recorder = _install_fake_http(monkeypatch, [_chat_body("ok")])
    heavy = heavy_llm_client(_config())

    heavy.chat([{"role": "user", "content": "hi"}])

    assert recorder["timeouts"] == [300.0]
    assert recorder["requests"][0]["url"] == "http://glm.test/v4/chat/completions"


def test_plain_client_keeps_config_timeout(monkeypatch):
    recorder = _install_fake_http(monkeypatch, [_chat_body("ok")])
    client = InternS2Client(_config(), min_interval=0)

    client.chat([{"role": "user", "content": "hi"}])

    assert client.request_timeout == 60.0
    assert recorder["timeouts"] == [60.0]


def test_thinking_effort_disabled_sends_disabled_payload(monkeypatch):
    _clear_heavy_env(
        monkeypatch,
        HEAVY_LLM_BASE_URL="https://api.deepseek.com",
        HEAVY_LLM_API_KEY="sk-test",
        HEAVY_LLM_MODEL="deepseek-v4-flash",
        HEAVY_LLM_THINKING_EFFORT="disabled",
    )
    recorder = _install_fake_http(
        monkeypatch,
        [{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}],
    )
    cfg = AppConfig(
        intern_api_key="k", intern_api_base_url="https://x", root_dir=Path("."),
        intern_model_name="m", sciverse_api_token="s", sciverse_api_base_url="https://s",
        strategy_probing_enabled=False, strategy_probe_limit=1, strategy_cluster_count=2,
        strategy_memory_enabled=False, strategy_memory_max_chars=10,
        request_timeout_seconds=5.0, max_llm_retries=1, tool_timeout_seconds=5.0,
        tool_result_max_chars=100, default_language="en", default_mode="demo",
    )
    reply = llm_client.heavy_llm_client(cfg).chat([{"role": "user", "content": "hi"}])
    assert reply == "ok"
    assert recorder["requests"][0]["payload"]["thinking"] == {"type": "disabled"}
