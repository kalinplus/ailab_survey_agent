import json

import httpx
import pytest
import respx

from tools.clients.sciverse_client import SciVerseClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SCIVERSE_API_KEY", "k")
    return SciVerseClient(base_url="https://sv.test", min_interval=0, max_retries=0)


@respx.mock
def test_meta_search(client):
    route = respx.post("https://sv.test/meta-search").respond(json={"results": [{"unique_id": "paper:1"}]})
    out = client.meta_search("world models")
    assert out["results"][0]["unique_id"] == "paper:1"
    sent = json.loads(route.calls[0].request.content)
    assert sent["query"] == "world models"
    assert sent["page"] == 1 and sent["page_size"] == 25
    # boosts default to None -> omitted from body
    assert "freshness_boost" not in sent and "impact_boost" not in sent


@respx.mock
def test_meta_search_sends_boosts_when_provided(client):
    route = respx.post("https://sv.test/meta-search").respond(json={"results": []})
    client.meta_search("wm", impact_boost="MILD", freshness_boost="MILD")
    sent = json.loads(route.calls[0].request.content)
    assert sent["impact_boost"] == "MILD" and sent["freshness_boost"] == "MILD"


def test_default_base_url_is_api_host(monkeypatch):
    monkeypatch.setenv("SCIVERSE_API_KEY", "k")
    c = SciVerseClient()
    assert c.base_url == "https://api.sciverse.space"


@respx.mock
def test_agentic_search(client):
    respx.post("https://sv.test/agentic-search").respond(json={"results": [{"id": "p2"}]})
    out = client.agentic_search("world models", top_k=5)
    assert out["results"][0]["id"] == "p2"


@respx.mock
def test_get_content(client):
    route = respx.get("https://sv.test/content").respond(json={"doc_id": "d1", "text": "hello"})
    out = client.get_content("d1")
    assert out["text"] == "hello"
    # no offset passed -> offset/limit omitted (one call returns the whole text)
    sent = route.calls[0].request.url.params
    assert "offset" not in sent and "limit" not in sent


@respx.mock
def test_get_content_sends_offset_when_provided(client):
    route = respx.get("https://sv.test/content").respond(json={"text": "chunk", "more": False})
    client.get_content("d1", offset=700, limit=100)
    sent = route.calls[0].request.url.params
    assert sent["offset"] == "700" and sent["limit"] == "100"


@respx.mock
def test_read_full_text_single_call_when_not_paginated(client):
    route = respx.get("https://sv.test/content").respond(
        json={"text": "whole paper", "more": False, "next_offset": 11})
    assert client.read_full_text("d1") == "whole paper"
    assert route.calls.call_count == 1


@respx.mock
def test_read_full_text_follows_more_next_offset(client):
    route = respx.get("https://sv.test/content").mock(side_effect=[
        httpx.Response(200, json={"text": "part1-", "more": True, "next_offset": 6}),
        httpx.Response(200, json={"text": "part2", "more": False}),
    ])
    assert client.read_full_text("d1") == "part1-part2"
    assert route.calls.call_count == 2
    assert route.calls[1].request.url.params["offset"] == "6"


@respx.mock
def test_read_full_text_stops_at_max_pages(client):
    page = {"text": "x.", "more": True, "next_offset": 2}
    route = respx.get("https://sv.test/content").respond(json=page)
    out = client.read_full_text("d1", max_pages=3)
    assert route.calls.call_count == 3
    assert out == "x.x.x."


@respx.mock
def test_get_resource(client):
    respx.get("https://sv.test/resource").respond(content=b"PDF-BYTES")
    out = client.get_resource("paper.pdf")
    assert out == b"PDF-BYTES"


@respx.mock
def test_meta_paper_relations(client):
    route = respx.post("https://sv.test/meta-paper-relations").respond(
        json={"data": {"items": [{"id": "r1"}], "total_count": 1}})
    out = client.meta_paper_relations("paper:10.1/x", relation="REFERENCES", page=2, page_size=50)
    sent = json.loads(route.calls[0].request.content)
    # live contract: the key is unique_id (paper:<doi>), never paper_id
    assert sent["unique_id"] == "paper:10.1/x"
    assert sent["relation"] == "REFERENCES"
    assert sent["page"] == 2 and sent["page_size"] == 50
    assert out["data"]["items"][0]["id"] == "r1"


@respx.mock
def test_raises_on_error(client):
    respx.post("https://sv.test/meta-search").respond(status_code=500)
    with pytest.raises(httpx.HTTPStatusError):
        client.meta_search("x")


@respx.mock
def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("SCIVERSE_API_KEY", "k")
    c = SciVerseClient(base_url="https://sv.test", min_interval=0, max_retries=1,
                       backoff_base=0, rate_limit_backoff_base=0)
    route = respx.post("https://sv.test/agentic-search").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"hits": [{"chunk": "x"}]}),
        ]
    )
    out = c.agentic_search("wm")
    assert route.calls.call_count == 2
    assert out["hits"][0]["chunk"] == "x"


@respx.mock
def test_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("SCIVERSE_API_KEY", "k")
    c = SciVerseClient(base_url="https://sv.test", min_interval=0, max_retries=1,
                       backoff_base=0, rate_limit_backoff_base=0)
    route = respx.post("https://sv.test/agentic-search").respond(status_code=429)
    with pytest.raises(httpx.HTTPStatusError):
        c.agentic_search("wm")
    assert route.calls.call_count == 2
