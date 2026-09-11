import hashlib
import io
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
# Headings that open a bibliography section; optional numeric prefix ("6. References").
_REFERENCE_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[\s.\-)]*)?(references?|bibliography|works\s*cited|参考文献|引用文献)\s*:?\s*$",
    re.IGNORECASE,
)


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


def _unwrap(payload) -> dict:
    """Extract the payload dict from a {"code":0,"data":{...}} envelope.

    Bare dicts (older probe contract) pass through. A non-zero code is an API
    error even when the HTTP status is 200.
    """
    if isinstance(payload, dict):
        code = payload.get("code")
        if code not in (None, 0):
            msg = payload.get("msg") or payload.get("err_msg")
            raise RuntimeError(f"mineru api error code={code} msg={msg}")
        data = payload.get("data")
        if isinstance(data, dict):
            return data
    return payload


def _is_image_block(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(_IMAGE_LINE_RE.match(ln) for ln in lines)


def _split_markdown(md: str):
    """Split full.md into the parsed-paper shape: (sections, flat paragraphs, title).

    Level-2+ headers open sections; the first level-1 header is the title and does
    not open one. Markdown carries no page numbers, so paragraphs get page=None
    (explicitly unknown, never a fabricated page) and index is a paper-global block
    number that never resets per section. Image-only blocks are dropped: figures
    come from content_list.json, which carries page and caption. Paragraphs
    under a References-style heading get role="reference" — same semantics as
    the content_list path.
    """
    sections: list[dict] = [{"name": "", "paragraphs": []}]
    title = ""
    block: list[str] = []
    counter = [0]
    in_references = [False]

    def flush() -> None:
        nonlocal block
        text = "\n".join(block).strip()
        block = []
        if not text or _is_image_block(text):
            return
        sec = sections[-1]
        sec["paragraphs"].append({"page": None, "index": counter[0], "text": text,
                                  "role": "reference" if in_references[0] else ""})
        counter[0] += 1

    for line in md.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            flush()
            level, text = len(header.group(1)), header.group(2).strip()
            if level == 1 and not title:
                title = text
            else:
                in_references[0] = bool(_REFERENCE_HEADING_RE.match(text))
                sections.append({"name": text, "paragraphs": []})
        elif not line.strip():
            flush()
        else:
            block.append(line.rstrip())
    flush()
    sections = [s for s in sections if s["paragraphs"]]
    flat = [p for s in sections for p in s["paragraphs"]]
    return sections, flat, title


def _markdown_parsed(md: str, parse_status: str) -> dict:
    sections, flat, title = _split_markdown(md)
    # "full_text" lets DataCleaner.complete_abstract recover the abstract.
    return {
        "title": title,
        "abstract": "",
        "sections": sections,
        "paragraphs": flat,
        "figures": [],
        "tables": [],
        "full_text": md,
        "parse_status": parse_status,
    }


def _asset_key(pdf_url: str) -> str:
    return hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()[:16]


def _extract_image_assets(archive: zipfile.ZipFile, blocks, pdf_url: str) -> dict:
    """Write content_list image files from the zip to cache/mineru_assets/<key>/.

    Returns a map from the block's zip-relative img_path to the written
    repo-relative path. An image missing from the zip degrades that figure
    (img_path stays None), not the whole parse.
    """
    wanted = set()
    for block in blocks if isinstance(blocks, list) else []:
        # real zips type charts as "chart" (verified live 2026-09-09: they carry
        # img_path entries present in the zip, just like image blocks)
        if isinstance(block, dict) and block.get("type") in ("image", "chart"):
            img = str(block.get("img_path") or "").strip().replace("\\", "/")
            if img:
                wanted.add(img)
    if not wanted:
        return {}
    names = [n for n in archive.namelist() if not n.endswith("/")]
    out_dir = Path("cache") / "mineru_assets" / _asset_key(pdf_url)
    resolved: dict = {}
    for img in wanted:
        match = next((n for n in names if n == img or n.endswith("/" + img)), None)
        if match is None:
            logger.warning(f"[mineru] image {img} referenced by content_list not found in zip")
            continue
        dest = out_dir / Path(img).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read(match))
        resolved[img] = dest.as_posix()
    return resolved


def _structured_from_content_list(blocks, assets: dict):
    """Build (sections, flat paragraphs, title, figures, tables) from content_list.json.

    Text blocks become paragraphs with page=page_idx+1 and a paper-global block
    index; text_level marks headings (first level-1 = title, others open sections,
    same shape as the markdown splitter). Paragraphs under a References-style
    heading get role="reference" — marked, not removed. Image blocks become
    figure dicts carrying the extracted img_path; table blocks become table
    dicts with caption and body.
    """
    sections: list[dict] = [{"name": "", "paragraphs": []}]
    figures: list[dict] = []
    tables: list[dict] = []
    title = ""
    in_references = False
    counter = 0

    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        page = int(block.get("page_idx", 0)) + 1
        if btype == "text":
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            level = block.get("text_level")
            if level:
                if int(level) == 1 and not title:
                    title = text
                else:
                    in_references = bool(_REFERENCE_HEADING_RE.match(text))
                    sections.append({"name": text, "paragraphs": []})
                continue
            sections[-1]["paragraphs"].append({
                "page": page,
                "index": counter,
                "text": text,
                "role": "reference" if in_references else "",
            })
            counter += 1
        elif btype in ("image", "chart"):
            caption = " ".join(str(c) for c in (block.get("image_caption") or []) if str(c).strip())
            img = str(block.get("img_path") or "").strip().replace("\\", "/")
            figures.append({
                "num": len(figures) + 1,
                "page": page,
                "caption": caption,
                "img_path": assets.get(img),
            })
        elif btype == "table":
            caption = " ".join(str(c) for c in (block.get("table_caption") or []) if str(c).strip())
            tables.append({
                "num": len(tables) + 1,
                "page": page,
                "caption": caption,
                "body": str(block.get("table_body") or ""),
            })

    sections = [s for s in sections if s["paragraphs"]]
    flat = [p for s in sections for p in s["paragraphs"]]
    return sections, flat, title, figures, tables


def _parse_full_zip(zip_bytes: bytes, pdf_url: str) -> dict:
    """Unpack a v4 full.zip.

    content_list.json is the structured primary source for sections/paragraphs
    (real pages, paper-global block index), figures (with extracted img_path)
    and tables. full.md remains the full_text/title/abstract recovery source.
    A zip without content_list.json falls back to the markdown split (page=None).
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"full_zip is not a valid zip: {e}") from e
    names = archive.namelist()
    md_name = next((n for n in names if n.endswith("full.md")), None)
    if md_name is None:
        raise RuntimeError(f"full_zip has no full.md (entries={names})")
    md = archive.read(md_name).decode("utf-8", errors="replace")
    if not md.strip():
        raise RuntimeError("full.md is empty")
    out = _markdown_parsed(md, parse_status="fulltext")
    cl_name = next((n for n in names if n.endswith("content_list.json")), None)
    if cl_name is None:
        return out
    blocks = json.loads(archive.read(cl_name).decode("utf-8", errors="replace"))
    assets = _extract_image_assets(archive, blocks, pdf_url)
    sections, flat, title, figures, tables = _structured_from_content_list(blocks, assets)
    if title:
        out["title"] = title
    out["sections"] = sections
    out["paragraphs"] = flat
    out["figures"] = figures
    out["tables"] = tables
    return out


class MinerUError(RuntimeError):
    """Stage- and kind-classified MinerU failure (wave-2 recovery contract).

    stage: precheck | submit | poll | download | unpack
    kind: read_file | ssl | transport | timeout | http_429 | http_5xx |
          http_<code> | api_error
    """

    def __init__(self, stage: str, kind: str, detail: str = ""):
        super().__init__(f"mineru {stage} failed ({kind}): {detail}")
        self.stage = stage
        self.kind = kind


class MinerUTimeout(MinerUError, TimeoutError):
    """Poll-budget exhaustion — the ledger entry stays resumable, never resubmitted."""


_TRANSIENT_KINDS = {"ssl", "transport", "http_429", "http_5xx"}
_LEDGER_PATH = Path("cache") / "mineru_tasks.json"
_RESULTS_DIR = Path("cache") / "mineru_results"
_TASK_TTL_SECONDS = 86400.0  # a ledger entry older than this is not resumed


def _kind_of(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return "http_429" if code == 429 else ("http_5xx" if code >= 500 else f"http_{code}")
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "ssl" if "ssl" in str(exc).lower() else "transport"
    return "api_error"


def _retry_transient(fn, attempts: int, wait: float, what: str):
    for attempt in range(attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if _kind_of(exc) not in _TRANSIENT_KINDS or attempt == attempts:
                raise
            logger.warning(f"[mineru] {what} {_kind_of(exc)}, retry in {wait}s ({attempt + 1}/{attempts})")
            time.sleep(wait)


def _load_ledger() -> dict:
    try:
        data = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _ledger_set(key: str, base: dict | None, **fields) -> None:
    ledger = _load_ledger()
    if base is None:
        ledger.pop(key, None)
    else:
        entry = ledger.get(key) or {}
        entry.update(base)
        entry.update(fields)
        ledger[key] = entry
    try:
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LEDGER_PATH.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:  # the ledger is a recovery optimization, never a blocker
        logger.warning(f"[mineru] ledger write failed: {exc}")


def _save_result(key: str, parsed: dict) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / f"{key}.json").write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")


def _load_result(key: str):
    try:
        return json.loads((_RESULTS_DIR / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class MinerUClient:
    def __init__(self, base_url="https://mineru.net", api_key=None, use_mock=True, poll_interval=3, max_wait=180,
                 max_retries=2, backoff_base=2.0):
        self.base_url = base_url.rstrip("/")
        self.token = api_key or os.getenv("MINERU_API_KEY", "")
        self.use_mock = use_mock
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        # per-stage transient retry (ssl/transport/429/5xx) knobs
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def _download(self, url: str) -> bytes:
        r = httpx.get(url, headers=self._headers(), timeout=300)
        r.raise_for_status()
        return r.content

    def parse_url(self, url: str, light: bool = True) -> dict:
        """Agent parse endpoint — the fallback channel behind extract_fulltext.

        The endpoint is task-based: submit returns a task_id, then
        GET /api/v1/agent/parse/{task_id} must be polled until state=done
        (download markdown_url) or state=failed (err_code/err_msg). A 200
        without a task_id / markdown_url / non-empty markdown is never a success.
        """
        t0 = time.monotonic()
        try:
            r = httpx.post(f"{self.base_url}/api/v1/agent/parse/url",
                           json={"url": url, "light": light},
                           headers=self._headers(), timeout=120)
            r.raise_for_status()
            task_id = _unwrap(r.json()).get("task_id")
            if not task_id:
                raise RuntimeError("agent parse submit returned no task_id")
            waited = 0
            while waited < self.max_wait:
                res = httpx.get(f"{self.base_url}/api/v1/agent/parse/{task_id}",
                                headers=self._headers(), timeout=60)
                res.raise_for_status()
                data = _unwrap(res.json())
                state = data.get("state")
                if state == "done":
                    md_url = data.get("markdown_url")
                    if not md_url:
                        raise RuntimeError(f"agent parse done without markdown_url: {data}")
                    md = self._download(md_url).decode("utf-8", errors="replace")
                    if not md.strip():
                        raise RuntimeError("agent parse markdown is empty")
                    out = _markdown_parsed(md, parse_status="light")
                    logger.info(f"[mineru] parse_url task={task_id} done "
                                f"{1000 * (time.monotonic() - t0):.0f}ms paragraphs={len(out['paragraphs'])}")
                    return out
                if state == "failed":
                    raise RuntimeError(
                        f"agent parse failed: err_code={data.get('err_code')} err_msg={data.get('err_msg')}")
                time.sleep(self.poll_interval)
                waited += self.poll_interval
            raise TimeoutError(f"agent parse task={task_id} not done after {self.max_wait}s")
        except Exception as e:
            logger.warning(f"[mineru] parse_url failed ({type(e).__name__}: {e}), use_mock={self.use_mock}")
            if self.use_mock:
                return mock_parse(url)
            raise

    def _precheck_pdf(self, url: str) -> None:
        """Reject non-PDF targets before spending a submit (landing pages, html,
        dead links). Advisory only: when the probe itself cannot reach the URL,
        submit proceeds — MinerU's network may still fetch it fine."""
        try:
            with httpx.stream("GET", url, timeout=30, follow_redirects=True) as r:
                if r.status_code != 200:
                    raise MinerUError("precheck", "read_file", f"status={r.status_code}")
                magic = next(r.iter_bytes(8), b"") or b""
                if magic[:4] != b"%PDF":
                    raise MinerUError(
                        "precheck", "read_file",
                        f"not a pdf: content-type={r.headers.get('content-type', '')!r} magic={magic!r}")
        except MinerUError:
            raise
        except Exception as exc:
            logger.info(f"[mineru] precheck inconclusive ({type(exc).__name__}: {exc}); submitting anyway")

    def _submit(self, pdf_url: str) -> str:
        def call() -> str:
            r = httpx.post(f"{self.base_url}/api/v4/extract/task", json={"url": pdf_url},
                           headers=self._headers(), timeout=60)
            r.raise_for_status()
            task_id = _unwrap(r.json()).get("task_id")
            if not task_id:
                raise RuntimeError("v4 extract submit returned no task_id")
            return task_id
        try:
            return _retry_transient(call, self.max_retries, self.backoff_base, "submit")
        except Exception as exc:
            raise MinerUError("submit", _kind_of(exc), str(exc)) from exc

    def _poll_once(self, task_id: str) -> dict:
        def call() -> dict:
            r = httpx.get(f"{self.base_url}/api/v4/extract/task/{task_id}",
                          headers=self._headers(), timeout=60)
            r.raise_for_status()
            return _unwrap(r.json())
        return _retry_transient(call, self.max_retries, self.backoff_base, f"poll {task_id}")

    def _poll_until_zip(self, task_id: str, key: str, pdf_url: str) -> str:
        """Poll to done/failed under a monotonic deadline that counts HTTP time.
        A timeout leaves a resumable ledger entry — this never resubmits."""
        deadline = time.monotonic() + self.max_wait
        while True:
            try:
                data = self._poll_once(task_id)
            except Exception as exc:
                _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url),
                            state="failed", last_error=f"{_kind_of(exc)}: {exc}")
                raise MinerUError("poll", _kind_of(exc), str(exc)) from exc
            # "done" is the probed v4 state; "succeeded" is the documented variant.
            state = data.get("state") or data.get("status")
            if state in ("done", "succeeded"):
                zip_url = data.get("full_zip_url")
                if not zip_url:
                    _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url), state="failed",
                                last_error="done without full_zip_url")
                    raise MinerUError("poll", "api_error",
                                      f"done without full_zip_url: err_msg={data.get('err_msg')}")
                return zip_url
            if state == "failed":
                _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url), state="failed",
                            last_error=str(data.get("err_msg")))
                raise MinerUError("poll", "api_error", f"v4 extract failed: {data.get('err_msg')}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url), state="timeout",
                            last_error=f"not done after {self.max_wait}s")
                raise MinerUTimeout("poll", "timeout", f"task={task_id} not done after {self.max_wait}s")
            time.sleep(min(self.poll_interval, remaining))

    def _download_zip(self, zip_url: str) -> bytes:
        def call() -> bytes:
            r = httpx.get(zip_url, headers=self._headers(), timeout=300)
            r.raise_for_status()
            return r.content
        try:
            # a download failure only re-downloads — never a resubmit
            return _retry_transient(call, self.max_retries, self.backoff_base, "download")
        except Exception as exc:
            raise MinerUError("download", _kind_of(exc), str(exc)) from exc

    def _resumable_task(self, key: str, pdf_url: str):
        entry = _load_ledger().get(key)
        if not entry or not entry.get("task_id") or entry.get("pdf_url") != pdf_url:
            return None
        if entry.get("state") not in ("running", "timeout"):
            return None
        age = time.time() - float(entry.get("submitted_at") or 0)
        if age > _TASK_TTL_SECONDS:
            return None
        logger.info(f"[mineru] resuming task {entry['task_id']} "
                    f"(state={entry.get('state')}, age={age:.0f}s)")
        return entry["task_id"]

    def extract_fulltext(self, pdf_url: str) -> dict:
        """v4 extract endpoint — the fulltext main channel.

        Order: result cache -> ledger resume -> precheck -> submit -> poll ->
        download -> unpack. A submitted task is never resubmitted: poll-budget
        exhaustion leaves a resumable ledger entry, download failures only
        re-download, and a stale resumed task (server 404) gets exactly one
        fresh submit. Never reports success on a 200 with no content. All
        failures raise MinerUError(stage, kind) unless use_mock degrades.
        """
        t0 = time.monotonic()
        key = _asset_key(pdf_url)
        try:
            cached = _load_result(key)
            if cached is not None:
                logger.info(f"[mineru] result cache hit key={key} "
                            f"paragraphs={len(cached.get('paragraphs', []))}")
                return cached
            resumed = False
            task_id = self._resumable_task(key, pdf_url)
            if task_id:
                resumed = True
            else:
                self._precheck_pdf(pdf_url)
                task_id = self._submit(pdf_url)
                _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url,
                                      submitted_at=time.time()), state="running")
                logger.info(f"[mineru] extract_fulltext submitted task_id={task_id}")
            try:
                zip_url = self._poll_until_zip(task_id, key, pdf_url)
            except MinerUError as exc:
                if resumed and exc.kind.startswith("http_4"):
                    # stale ledger task the service no longer knows about
                    logger.warning(f"[mineru] resumed task {task_id} gone ({exc.kind}); resubmitting once")
                    task_id = self._submit(pdf_url)
                    _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url,
                                          submitted_at=time.time()), state="running")
                    zip_url = self._poll_until_zip(task_id, key, pdf_url)
                else:
                    raise
            try:
                out = _parse_full_zip(self._download_zip(zip_url), pdf_url)
            except MinerUError:
                raise
            except Exception as exc:
                raise MinerUError("unpack", _kind_of(exc), str(exc)) from exc
            _save_result(key, out)
            _ledger_set(key, dict(task_id=task_id, pdf_url=pdf_url), state="done")
            logger.info(f"[mineru] extract_fulltext task={task_id} done "
                        f"{1000 * (time.monotonic() - t0):.0f}ms paragraphs={len(out['paragraphs'])} "
                        f"figures={len(out['figures'])} tables={len(out['tables'])}")
            return out
        except MinerUError as exc:
            logger.warning(f"[mineru] extract_fulltext failed stage={exc.stage} kind={exc.kind}: "
                           f"{exc}, use_mock={self.use_mock}")
            if self.use_mock:
                return mock_parse(pdf_url)
            raise
        except Exception as exc:
            logger.warning(f"[mineru] extract_fulltext failed ({type(exc).__name__}: {exc}), "
                           f"use_mock={self.use_mock}")
            if self.use_mock:
                return mock_parse(pdf_url)
            raise
