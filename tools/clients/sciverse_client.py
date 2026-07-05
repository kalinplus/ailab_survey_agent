import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)


class SciVerseClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 min_interval: float | None = None, max_retries: int | None = None,
                 backoff_base: float | None = None, rate_limit_backoff_base: float | None = None):
        self.base_url = (base_url or os.getenv("SCIVERSE_API_BASE_URL", "https://api.sciverse.space")).rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key or os.getenv('SCIVERSE_API_KEY', '')}"}
        # Rate limit + retry backoff at the external API boundary. Env-tunable so the
        # spacing can be widened without code changes when SciVerse starts 429-ing.
        self._last_call = 0.0
        self.min_interval = float(min_interval if min_interval is not None
                                  else os.getenv("SCIVERSE_MIN_INTERVAL", "1.0"))
        self.max_retries = int(max_retries if max_retries is not None
                               else os.getenv("SCIVERSE_MAX_RETRIES", "3"))
        self.backoff_base = float(backoff_base if backoff_base is not None
                                  else os.getenv("SCIVERSE_BACKOFF_BASE", "1.0"))
        # 429 gets a longer base than 5xx: a rate-limit window takes seconds-to-tens
        # to clear, and with max_retries=3 the series (5s,10s,20s) gives it time to
        # recover before we give up. Env-tunable like the other knobs.
        self.rate_limit_backoff_base = float(
            rate_limit_backoff_base if rate_limit_backoff_base is not None
            else os.getenv("SCIVERSE_RATE_LIMIT_BACKOFF_BASE", "5.0"))

    def _request(self, method, path, *, params=None, json=None, timeout=60):
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            t0 = time.monotonic()
            try:
                r = httpx.request(method, url, params=params, json=json,
                                  headers=self.headers, timeout=timeout)
            except httpx.TransportError as exc:  # timeout / network errors -> retry
                if attempt == self.max_retries:
                    raise
                wait = self.backoff_base * (2 ** attempt)
                logger.warning(f"[sciverse] {method} {path} {type(exc).__name__}, "
                               f"retry in {wait:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                time.sleep(wait)
                continue
            if r.status_code == 429 or r.status_code >= 500:  # transient -> retry
                if attempt == self.max_retries:
                    logger.warning(f"[sciverse] {method} {path} {r.status_code} "
                                   f"after {attempt + 1} attempts")
                    r.raise_for_status()
                wait = self._backoff_seconds(r, attempt)
                logger.warning(f"[sciverse] {method} {path} {r.status_code}, "
                               f"retry in {wait:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                time.sleep(wait)
                continue
            logger.info(f"[sciverse] {method} {path} {r.status_code} {len(r.content)}B "
                        f"{1000 * (time.monotonic() - t0):.0f}ms")
            return r
        raise RuntimeError("unreachable")  # loop always returns or raises on the final attempt

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def _backoff_seconds(self, response, attempt):
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        base = self.rate_limit_backoff_base if response.status_code == 429 else self.backoff_base
        return base * (2 ** attempt)

    def _post(self, path, payload):
        return self._request("POST", path, json=payload).json()

    def _get(self, path, params=None):
        return self._request("GET", path, params=params).json()

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
        return self._request("GET", "/resource", params={"file_name": file_name}, timeout=120).content

    def meta_paper_relations(self, paper_id, relation="REFERENCES", page=1):
        return self._post("/meta-paper-relations",
                          {"paper_id": paper_id, "relation": relation, "page": page})
