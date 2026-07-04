import os
import time

import httpx


def mock_parse(url: str, title: str = "Unknown") -> dict:
    return {
        "title": title,
        "abstract": "Mock abstract for offline parsing.",
        "sections": [{"name": "Introduction", "paragraphs": [{"page": 1, "index": 0, "text": "Mock intro paragraph."}]}],
        "paragraphs": [{"page": 1, "index": 0, "text": "Mock intro paragraph."}],
        "figures": [{"num": 1, "caption": "Mock figure.", "page": 1}],
        "tables": [],
        "parse_status": "mock",
    }


class MinerUClient:
    def __init__(self, base_url="https://mineru.net", api_key=None, use_mock=True, poll_interval=3, max_wait=180):
        self.base_url = base_url.rstrip("/")
        self.token = api_key or os.getenv("MINERU_API_KEY", "")
        self.use_mock = use_mock
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    def parse_url(self, url: str, light: bool = True) -> dict:
        try:
            r = httpx.post(f"{self.base_url}/api/v1/agent/parse/url",
                           json={"url": url, "light": light},
                           headers={"Authorization": f"Bearer {self.token}"}, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception:
            if self.use_mock:
                return mock_parse(url)
            raise

    def extract_task(self, pdf_url: str) -> dict:
        try:
            submit = httpx.post(f"{self.base_url}/api/v4/extract/task",
                                json={"url": pdf_url},
                                headers={"Authorization": f"Bearer {self.token}"}, timeout=60)
            submit.raise_for_status()
            task_id = submit.json()["task_id"]
            waited = 0
            while waited < self.max_wait:
                res = httpx.get(f"{self.base_url}/api/v4/extract/task/{task_id}",
                                headers={"Authorization": f"Bearer {self.token}"}, timeout=60)
                res.raise_for_status()
                data = res.json()
                if data.get("status") in ("succeeded", "failed"):
                    return data
                time.sleep(self.poll_interval)
                waited += self.poll_interval
            return mock_parse(pdf_url)
        except Exception:
            if self.use_mock:
                return mock_parse(pdf_url)
            raise
