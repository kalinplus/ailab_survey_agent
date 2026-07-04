import hashlib
import os

from openai import OpenAI


class EmbeddingClient:
    def __init__(self, model="text-embedding-3-small", api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class FakeEmbeddingClient:
    """Deterministic 16-d vectors for tests (no network)."""

    def __init__(self, dim=16):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([(b / 255.0 - 0.5) for b in (h * ((self.dim // len(h)) + 1))][:self.dim])
        return out
