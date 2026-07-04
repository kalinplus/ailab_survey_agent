from unittest.mock import MagicMock

from tools.clients.embedding_client import EmbeddingClient, FakeEmbeddingClient


# --- FakeEmbeddingClient tests ---


def test_fake_embedding_deterministic():
    e = FakeEmbeddingClient(dim=16)
    a = e.embed(["hello"])[0]
    b = e.embed(["hello"])[0]
    assert a == b and len(a) == 16


def test_different_text_different_vec():
    e = FakeEmbeddingClient()
    assert e.embed(["a"])[0] != e.embed(["b"])[0]


def test_fake_default_dim():
    e = FakeEmbeddingClient()
    vec = e.embed(["hello"])[0]
    assert len(vec) == 16


def test_fake_custom_dim():
    e = FakeEmbeddingClient(dim=8)
    vec = e.embed(["hello"])[0]
    assert len(vec) == 8


def test_fake_batch_preserves_order():
    e = FakeEmbeddingClient()
    texts = ["first", "second", "third"]
    results = e.embed(texts)
    assert len(results) == 3
    for i, t in enumerate(texts):
        assert results[i] == e.embed([t])[0]


# --- EmbeddingClient tests (mocked, no network) ---


def test_embedding_client_embed_extracts_vectors():
    fake_vec_a = [0.1, 0.2, 0.3]
    fake_vec_b = [0.4, 0.5, 0.6]
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_vec_a), MagicMock(embedding=fake_vec_b)]

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = mock_resp

    client = EmbeddingClient.__new__(EmbeddingClient)
    client.client = mock_openai
    client.model = "text-embedding-3-small"

    result = client.embed(["hello", "world"])
    assert result == [fake_vec_a, fake_vec_b]
    mock_openai.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input=["hello", "world"])


def test_embedding_client_single_text():
    fake_vec = [0.0, 1.0]
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_vec)]

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = mock_resp

    client = EmbeddingClient.__new__(EmbeddingClient)
    client.client = mock_openai
    client.model = "text-embedding-3-small"

    result = client.embed(["solo"])
    assert result == [[0.0, 1.0]]


def test_embedding_client_empty_list():
    mock_resp = MagicMock()
    mock_resp.data = []

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = mock_resp

    client = EmbeddingClient.__new__(EmbeddingClient)
    client.client = mock_openai
    client.model = "text-embedding-3-small"

    result = client.embed([])
    assert result == []
