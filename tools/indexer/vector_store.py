from dataclasses import dataclass

import chromadb


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, embedding_client, path=None, collection="papers"):
        self.embed = embedding_client
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        vecs = self.embed.embed(texts)
        self.collection.upsert(ids=ids, embeddings=vecs, documents=texts, metadatas=metadatas)

    def query(self, text: str, n: int = 10, where: dict | None = None) -> list[Hit]:
        qv = self.embed.embed([text])[0]
        res = self.collection.query(query_embeddings=[qv], n_results=n, where=where)
        hits = []
        for i, _id in enumerate(res["ids"][0]):
            dist = res["distances"][0][i]
            hits.append(Hit(id=_id, score=1.0 - dist, metadata=res["metadatas"][0][i]))
        return hits
