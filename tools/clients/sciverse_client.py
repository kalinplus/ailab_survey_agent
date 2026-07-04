import os

import httpx


class SciVerseClient:
    def __init__(self, base_url: str = "https://sciverse.space", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key or os.getenv('SCIVERSE_API_KEY', '')}"}

    def _post(self, path, payload):
        r = httpx.post(f"{self.base_url}{path}", json=payload, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def _get(self, path, params=None):
        r = httpx.get(f"{self.base_url}{path}", params=params, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def meta_search(self, query, filters=None, sort=None, freshness_boost="MILD",
                    impact_boost="MILD", page_size=25):
        return self._post("/meta-search", {"query": query, "filters": filters or [],
            "sort": sort or [], "freshness_boost": freshness_boost,
            "impact_boost": impact_boost, "page_size": page_size})

    def agentic_search(self, query, top_k=10, filters=None):
        return self._post("/agentic-search", {"query": query, "top_k": top_k, "filters": filters or {}})

    def get_content(self, doc_id, offset=0, limit=10):
        return self._get("/content", {"doc_id": doc_id, "offset": offset, "limit": limit})

    def get_resource(self, file_name):
        r = httpx.get(f"{self.base_url}/resource", params={"file_name": file_name},
                      headers=self.headers, timeout=120)
        r.raise_for_status()
        return r.content

    def meta_paper_relations(self, paper_id, relation="REFERENCES", page=1):
        return self._post("/meta-paper-relations",
                          {"paper_id": paper_id, "relation": relation, "page": page})
