import httpx
import pytest
import respx

from tools.clients.sciverse_client import SciVerseClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SCIVERSE_API_KEY", "k")
    return SciVerseClient(base_url="https://sv.test")


@respx.mock
def test_meta_search(client):
    respx.post("https://sv.test/meta-search").respond(json={"hits": [{"unique_id": "paper:1"}]})
    out = client.meta_search("world models")
    assert out["hits"][0]["unique_id"] == "paper:1"


@respx.mock
def test_agentic_search(client):
    respx.post("https://sv.test/agentic-search").respond(json={"results": [{"id": "p2"}]})
    out = client.agentic_search("world models", top_k=5)
    assert out["results"][0]["id"] == "p2"


@respx.mock
def test_get_content(client):
    respx.get("https://sv.test/content").respond(json={"doc_id": "d1", "text": "hello"})
    out = client.get_content("d1")
    assert out["text"] == "hello"


@respx.mock
def test_get_resource(client):
    respx.get("https://sv.test/resource").respond(content=b"PDF-BYTES")
    out = client.get_resource("paper.pdf")
    assert out == b"PDF-BYTES"


@respx.mock
def test_meta_paper_relations(client):
    respx.post("https://sv.test/meta-paper-relations").respond(json={"relations": [{"id": "r1"}]})
    out = client.meta_paper_relations("paper:1")
    assert out["relations"][0]["id"] == "r1"


@respx.mock
def test_raises_on_error(client):
    respx.post("https://sv.test/meta-search").respond(status_code=500)
    with pytest.raises(httpx.HTTPStatusError):
        client.meta_search("x")
