from tools.indexer.vector_store import Hit, VectorStore
from tools.clients.embedding_client import FakeEmbeddingClient


def test_add_and_query_roundtrip(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v"))
    vs.add(["a", "b"], ["world model dreamer", "reinforcement learning"], [{"paper_id": "p1"}, {"paper_id": "p2"}])
    hits = vs.query("world model dreamer", n=2)
    assert hits[0].id == "a"


def test_query_with_where_filter(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v2"))
    vs.add(["a", "b"], ["x", "y"], [{"paper_id": "p1"}, {"paper_id": "p2"}])
    hits = vs.query("x", n=5, where={"paper_id": "p2"})
    assert all(h.metadata["paper_id"] == "p2" for h in hits)


def test_add_upsert_updates_existing_document(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v3"))
    vs.add(["a"], ["original text"], [{"paper_id": "p1"}])
    vs.add(["a"], ["updated text"], [{"paper_id": "p1"}])
    hits = vs.query("updated text", n=1)
    assert len(hits) == 1
    assert hits[0].id == "a"


def test_query_respects_n(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v4"))
    vs.add(["a", "b", "c"], ["doc one", "doc two", "doc three"], [{"i": "1"}, {"i": "2"}, {"i": "3"}])
    hits = vs.query("doc one", n=2)
    assert len(hits) <= 2


def test_metadata_roundtrip(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v5"))
    vs.add(["a"], ["some text"], [{"paper_id": "p99", "year": "2024"}])
    hits = vs.query("some text", n=1)
    assert hits[0].metadata["paper_id"] == "p99"
    assert hits[0].metadata["year"] == "2024"


def test_empty_store_query_returns_empty(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v6"))
    hits = vs.query("anything", n=5)
    assert hits == []


def test_where_filter_no_match_returns_empty(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v7"))
    vs.add(["a", "b"], ["x", "y"], [{"paper_id": "p1"}, {"paper_id": "p2"}])
    hits = vs.query("x", n=5, where={"paper_id": "p_nonexistent"})
    assert hits == []
