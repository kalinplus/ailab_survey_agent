import os

import httpx


class SciVerseClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.getenv("SCIVERSE_API_BASE_URL", "https://api.sciverse.space")).rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key or os.getenv('SCIVERSE_API_KEY', '')}"}

    def _post(self, path, payload):
        r = httpx.post(f"{self.base_url}{path}", json=payload, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def _get(self, path, params=None):
        r = httpx.get(f"{self.base_url}{path}", params=params, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def meta_search(self, query, filters=None, sort=None, fields=None, page=1,
                    page_size=25, freshness_boost=None, impact_boost=None):
        # Verified SciVerse contract. freshness_boost / impact_boost bias toward
        # recent / highly-cited results and apply when sort is NOT set (valid: "MILD";
        # "HIGH" is rejected). Send each optional key only when provided.
        payload = {"query": query, "page": page, "page_size": page_size}
        if filters:
            payload["filters"] = filters
        if sort:
            payload["sort"] = sort
        if fields:
            payload["fields"] = fields
        if freshness_boost:
            payload["freshness_boost"] = freshness_boost
        if impact_boost:
            payload["impact_boost"] = impact_boost
        return self._post("/meta-search", payload)

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
