import httpx
import pytest
import respx

from tools.clients.mineru_client import MinerUClient, mock_parse


# -- mock_parse shape --

def test_mock_parse_shape():
    out = mock_parse("http://x/a.pdf", "T")
    assert out["title"] == "T"
    assert out["parse_status"] == "mock"
    assert isinstance(out["paragraphs"], list)
    assert out["paragraphs"][0]["page"] == 1
    assert out["paragraphs"][0]["index"] == 0
    assert out["paragraphs"][0]["text"] == "Mock intro paragraph."
    assert isinstance(out["sections"], list)
    assert isinstance(out["figures"], list)
    assert isinstance(out["tables"], list)


# -- parse_url: success path --

@respx.mock
def test_parse_url_success():
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(
        json={"title": "Real Paper", "parse_status": "done"})
    c = MinerUClient(use_mock=False, api_key="k")
    out = c.parse_url("http://x/a.pdf")
    assert out["title"] == "Real Paper"
    assert out["parse_status"] == "done"


# -- parse_url: use_mock=True, server error -> fallback to mock --

@respx.mock
def test_parse_url_falls_back_to_mock_on_error():
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(status_code=500)
    c = MinerUClient(use_mock=True)
    out = c.parse_url("http://x/a.pdf")
    assert out["parse_status"] == "mock"


# -- parse_url: use_mock=False, server error -> raises --

@respx.mock
def test_parse_url_raises_when_mock_disabled():
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(status_code=500)
    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(httpx.HTTPStatusError):
        c.parse_url("http://x/a.pdf")


# -- extract_task: submit -> poll -> succeeded path (no real sleep) --

@respx.mock
def test_extract_task_succeeded(monkeypatch):
    monkeypatch.setattr("tools.clients.mineru_client.time.sleep", lambda _: None)
    c = MinerUClient(use_mock=False, api_key="k", poll_interval=3, max_wait=60)

    submit_route = respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"task_id": "t-123"})
    # First poll returns "processing", second returns "succeeded"
    poll_route = respx.get("https://mineru.net/api/v4/extract/task/t-123").mock(
        side_effect=[
            httpx.Response(200, json={"task_id": "t-123", "status": "processing"}),
            httpx.Response(200, json={"task_id": "t-123", "status": "succeeded", "result": {"text": "parsed"}}),
        ])

    out = c.extract_task("http://x/paper.pdf")
    assert out["status"] == "succeeded"
    assert out["result"]["text"] == "parsed"
    assert submit_route.call_count == 1
    assert poll_route.call_count == 2


# -- extract_task: submit -> poll -> max_wait exceeded -> mock fallback --

@respx.mock
def test_extract_task_max_wait_exceeded_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr("tools.clients.mineru_client.time.sleep", lambda _: None)
    c = MinerUClient(use_mock=True, api_key="k", poll_interval=3, max_wait=6)

    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"task_id": "t-456"})
    # Always return "processing" so max_wait is exceeded after 2 polls (0+3=3, 3+3=6 >= 6)
    respx.get("https://mineru.net/api/v4/extract/task/t-456").respond(
        json={"task_id": "t-456", "status": "processing"})

    out = c.extract_task("http://x/paper.pdf")
    assert out["parse_status"] == "mock"


# -- extract_task: submit raises -> mock fallback --

@respx.mock
def test_extract_task_submit_error_falls_back_to_mock():
    respx.post("https://mineru.net/api/v4/extract/task").respond(status_code=503)
    c = MinerUClient(use_mock=True)
    out = c.extract_task("http://x/paper.pdf")
    assert out["parse_status"] == "mock"


# -- extract_task: poll raises -> mock fallback --

@respx.mock
def test_extract_task_poll_error_falls_back_to_mock():
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"task_id": "t-789"})
    respx.get("https://mineru.net/api/v4/extract/task/t-789").respond(status_code=500)
    c = MinerUClient(use_mock=True, api_key="k")

    out = c.extract_task("http://x/paper.pdf")
    assert out["parse_status"] == "mock"


# -- extract_task: use_mock=False, submit error -> raises --

@respx.mock
def test_extract_task_raises_when_mock_disabled_submit_error():
    respx.post("https://mineru.net/api/v4/extract/task").respond(status_code=503)
    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(httpx.HTTPStatusError):
        c.extract_task("http://x/paper.pdf")
