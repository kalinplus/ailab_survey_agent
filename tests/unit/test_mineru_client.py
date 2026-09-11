import hashlib
import io
import json
import zipfile

import httpx
import pytest
import respx

from tools.clients.mineru_client import MinerUClient, MinerUError, _markdown_parsed, mock_parse

FULL_MD = (
    "# DreamerV3\n"
    "\n"
    "We study world models.\n"
    "\n"
    "## Method\n"
    "\n"
    "The agent learns a latent dynamics model.\n"
    "It plans in imagination.\n"
    "\n"
    "![Figure 1](images/fig1.jpg)\n"
    "\n"
    "## Results\n"
    "\n"
    "Scores improve across domains.\n"
)

PDF_URL = "http://x/paper.pdf"
ASSET_KEY = hashlib.sha256(PDF_URL.encode("utf-8")).hexdigest()[:16]

CONTENT_LIST = [
    {"type": "text", "text": "DreamerV3", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "We study world models.", "page_idx": 0},
    {"type": "text", "text": "Method", "text_level": 1, "page_idx": 1},
    {"type": "text", "text": "The agent learns a latent dynamics model.", "page_idx": 1},
    {"type": "image", "img_path": "images/fig1.jpg",
     "image_caption": ["Figure 1: World model overview."], "page_idx": 1},
    {"type": "chart", "img_path": "images/chart1.png", "page_idx": 1},  # uncaptioned, real zips carry these
    {"type": "text", "text": "It plans in imagination.", "page_idx": 2},
    {"type": "table", "table_caption": ["Table 1: Benchmarks."],
     "table_body": "<html><body><table><tr><td>game</td><td>score</td></tr></table></body></html>",
     "page_idx": 2},
    {"type": "text", "text": "Results", "text_level": 2, "page_idx": 3},
    {"type": "text", "text": "Scores improve across domains.", "page_idx": 3},
    {"type": "text", "text": "References", "text_level": 1, "page_idx": 4},
    {"type": "text", "text": "Hafner et al. Mastering diverse domains through world models.", "page_idx": 4},
]

FIG1_PNG = b"\x89PNG\r\n\x1a\nfig1-bytes"
CHART_PNG = b"\x89PNG\r\n\x1a\nchart1-bytes"


def _zip_bytes(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


def _no_sleep(monkeypatch):
    monkeypatch.setattr("tools.clients.mineru_client.time.sleep", lambda _: None)


# -- mock_parse shape --

def test_mock_parse_shape():
    out = mock_parse("http://x/a.pdf", "T")
    assert out["title"] == "T"
    assert out["parse_status"] == "mock"
    assert isinstance(out["paragraphs"], list)
    assert out["paragraphs"][0]["page"] == 1
    assert out["paragraphs"][0]["index"] == 0
    assert out["paragraphs"][0]["text"] == "Mock intro paragraph."
    assert isinstance(out["sections"], list)
    assert isinstance(out["figures"], list)
    assert isinstance(out["tables"], list)


# -- parse_url (agent endpoint, fallback channel): submit -> poll -> markdown --

@respx.mock
def test_parse_url_polls_to_done_and_downloads_markdown(monkeypatch):
    _no_sleep(monkeypatch)
    submit_route = respx.post("https://mineru.net/api/v1/agent/parse/url").respond(
        json={"code": 0, "data": {"task_id": "a-1"}})
    poll_route = respx.get("https://mineru.net/api/v1/agent/parse/a-1").mock(side_effect=[
        httpx.Response(200, json={"code": 0, "data": {"state": "running"}}),
        httpx.Response(200, json={"code": 0, "data": {"state": "done",
                                                      "markdown_url": "https://files.test/a-1.md"}}),
    ])
    respx.get("https://files.test/a-1.md").respond(content=FULL_MD.encode())

    out = MinerUClient(use_mock=False, api_key="k").parse_url("http://x/a.pdf")

    assert submit_route.call_count == 1
    assert poll_route.call_count == 2
    assert out["parse_status"] == "light"
    assert out["title"] == "DreamerV3"
    assert [s["name"] for s in out["sections"]] == ["", "Method", "Results"]
    # markdown channel: page explicitly unknown, index paper-global (never
    # per-section); role key present ("" outside a References-style section)
    assert out["paragraphs"][0] == {"page": None, "index": 0, "text": "We study world models.",
                                    "role": ""}
    assert [p["index"] for p in out["paragraphs"]] == [0, 1, 2]
    assert out["paragraphs"][1]["text"] == "The agent learns a latent dynamics model.\nIt plans in imagination."
    assert all(p["role"] == "" for p in out["paragraphs"])
    assert out["figures"] == []  # agent channel has no content_list figure metadata
    assert out["full_text"] == FULL_MD  # feeds DataCleaner.complete_abstract


def test_markdown_references_section_gets_reference_role():
    """Markdown with a References heading marks that section's paragraphs
    role='reference' — same semantics as the content_list path (the /content
    fulltext channel relies on this to keep bibliography text out of evidence)."""
    md = "# T\n\nown text\n\n## References\n\nSmith et al. other work [12].\n"
    out = _markdown_parsed(md, "light")
    roles = {p["text"]: p["role"] for p in out["paragraphs"]}
    assert roles["own text"] == ""
    assert roles["Smith et al. other work [12]."] == "reference"


@respx.mock
def test_parse_url_failed_state_raises(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(
        json={"code": 0, "data": {"task_id": "a-2"}})
    respx.get("https://mineru.net/api/v1/agent/parse/a-2").respond(
        json={"code": 0, "data": {"state": "failed", "err_code": 400, "err_msg": "not a pdf"}})

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="not a pdf"):
        c.parse_url("http://x/a.pdf")


@respx.mock
def test_parse_url_done_without_markdown_url_raises(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(
        json={"code": 0, "data": {"task_id": "a-3"}})
    respx.get("https://mineru.net/api/v1/agent/parse/a-3").respond(
        json={"code": 0, "data": {"state": "done"}})  # 200 with no content

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="markdown_url"):
        c.parse_url("http://x/a.pdf")


@respx.mock
def test_parse_url_empty_markdown_raises(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(
        json={"code": 0, "data": {"task_id": "a-4"}})
    respx.get("https://mineru.net/api/v1/agent/parse/a-4").respond(
        json={"code": 0, "data": {"state": "done", "markdown_url": "https://files.test/empty.md"}})
    respx.get("https://files.test/empty.md").respond(content=b"   \n")

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="empty"):
        c.parse_url("http://x/a.pdf")


@respx.mock
def test_parse_url_submit_200_without_task_id_raises():
    # The old bug: a 200 with no task payload was treated as a successful parse.
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(json={"code": 0, "data": {}})

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="no task_id"):
        c.parse_url("http://x/a.pdf")


@respx.mock
def test_parse_url_timeout(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(
        json={"code": 0, "data": {"task_id": "a-5"}})
    respx.get("https://mineru.net/api/v1/agent/parse/a-5").respond(
        json={"code": 0, "data": {"state": "running"}})

    out = MinerUClient(use_mock=True, api_key="k", poll_interval=3, max_wait=6).parse_url("http://x/a.pdf")
    assert out["parse_status"] == "mock"

    with pytest.raises(TimeoutError):
        MinerUClient(use_mock=False, api_key="k", poll_interval=3, max_wait=6).parse_url("http://x/a.pdf")


@respx.mock
def test_parse_url_falls_back_to_mock_on_error():
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(status_code=500)
    c = MinerUClient(use_mock=True)
    out = c.parse_url("http://x/a.pdf")
    assert out["parse_status"] == "mock"


@respx.mock
def test_parse_url_raises_when_mock_disabled():
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(status_code=500)
    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(httpx.HTTPStatusError):
        c.parse_url("http://x/a.pdf")


# -- extract_fulltext (v4 endpoint, main channel): submit -> poll -> zip -> unzip --

def _mock_v4_routes(task_id, zip_name, zip_entries):
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": task_id}})
    respx.get(f"https://mineru.net/api/v4/extract/task/{task_id}").mock(side_effect=[
        httpx.Response(200, json={"code": 0, "data": {"state": "waiting"}}),
        httpx.Response(200, json={"code": 0, "data": {"state": "done",
                                                      "full_zip_url": f"https://files.test/{zip_name}"}}),
    ])
    return respx.get(f"https://files.test/{zip_name}").respond(content=_zip_bytes(zip_entries))


@respx.mock
def test_extract_fulltext_submits_polls_downloads_and_unzips(monkeypatch, tmp_path):
    # asset writes go to the tmp cwd, not the repo's cache/
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    zip_route = _mock_v4_routes("t-1", "full.zip", {
        "job/full.md": FULL_MD,  # nested name: lookup must match by suffix
        "job/content_list.json": json.dumps(CONTENT_LIST),
        "job/images/fig1.jpg": FIG1_PNG,
        "job/images/chart1.png": CHART_PNG,
    })

    out = MinerUClient(use_mock=False, api_key="k").extract_fulltext(PDF_URL)

    assert zip_route.call_count == 1
    assert out["parse_status"] == "fulltext"
    assert out["title"] == "DreamerV3"  # first level-1 heading in content_list
    assert [s["name"] for s in out["sections"]] == ["", "Method", "Results", "References"]
    paras = out["paragraphs"]
    # real pages (page_idx+1) and paper-global strictly increasing block index
    assert [p["page"] for p in paras] == [1, 2, 3, 4, 5]
    assert [p["index"] for p in paras] == [0, 1, 2, 3, 4]
    assert len({(p["page"], p["index"]) for p in paras}) == len(paras)
    # References-section paragraph is marked, not removed
    assert paras[4]["role"] == "reference"
    assert all(p["role"] == "" for p in paras[:4])
    # figure assets extracted: image with caption + uncaptioned chart, both on disk
    expected_img = f"cache/mineru_assets/{ASSET_KEY}/fig1.jpg"
    expected_chart = f"cache/mineru_assets/{ASSET_KEY}/chart1.png"
    assert out["figures"] == [{"num": 1, "page": 2, "caption": "Figure 1: World model overview.",
                               "img_path": expected_img},
                              {"num": 2, "page": 2, "caption": "", "img_path": expected_chart}]
    assert (tmp_path / expected_img).read_bytes() == FIG1_PNG
    assert (tmp_path / expected_chart).read_bytes() == CHART_PNG
    # table block extracted with caption and body
    assert out["tables"] == [{"num": 1, "page": 3, "caption": "Table 1: Benchmarks.",
                              "body": "<html><body><table><tr><td>game</td><td>score</td></tr></table></body></html>"}]
    assert out["full_text"] == FULL_MD  # md still feeds DataCleaner.complete_abstract


@respx.mock
def test_extract_fulltext_without_content_list_falls_back_to_markdown(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # ledger + result cache writes stay in tmp
    _no_sleep(monkeypatch)
    _mock_v4_routes("t-md", "md-only.zip", {"job/full.md": FULL_MD})

    out = MinerUClient(use_mock=False, api_key="k").extract_fulltext(PDF_URL)

    assert out["parse_status"] == "fulltext"
    # no structured source: page explicitly unknown, index still paper-global
    assert all(p["page"] is None for p in out["paragraphs"])
    indexes = [p["index"] for p in out["paragraphs"]]
    assert indexes == sorted(set(indexes))
    assert [s["name"] for s in out["sections"]] == ["", "Method", "Results"]
    assert out["figures"] == []
    assert out["tables"] == []


@respx.mock
def test_extract_fulltext_image_missing_from_zip_keeps_record_without_asset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    _mock_v4_routes("t-noimg", "no-img.zip", {
        "job/full.md": FULL_MD,
        "job/content_list.json": json.dumps(CONTENT_LIST),  # references images/fig1.jpg
    })

    out = MinerUClient(use_mock=False, api_key="k").extract_fulltext(PDF_URL)

    # degraded figures: caption/page known, no fabricated asset paths
    assert out["figures"] == [{"num": 1, "page": 2, "caption": "Figure 1: World model overview.",
                               "img_path": None},
                              {"num": 2, "page": 2, "caption": "", "img_path": None}]
    assert not (tmp_path / "cache" / "mineru_assets").exists()


def test_markdown_same_text_across_sections_gets_distinct_coords():
    # old bug fixture: identical paragraph text in two sections used to collide on
    # (page=1, per-section index=0); the global index must separate them now
    md = "# T\n\nsame para\n\n## A\n\nsame para\n\n## B\n\nsame para\n"
    out = _markdown_parsed(md, "light")
    assert [(p["page"], p["index"]) for p in out["paragraphs"]] == [(None, 0), (None, 1), (None, 2)]


@respx.mock
def test_extract_fulltext_failed_state_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-2"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-2").respond(
        json={"code": 0, "data": {"state": "failed", "err_msg": "unsupported file"}})

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="unsupported file"):
        c.extract_fulltext("http://x/paper.pdf")


@respx.mock
def test_extract_fulltext_done_without_zip_url_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-3"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-3").respond(
        json={"code": 0, "data": {"state": "done"}})  # 200 with no content

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="full_zip_url"):
        c.extract_fulltext("http://x/paper.pdf")


@respx.mock
def test_extract_fulltext_zip_without_md_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-4"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-4").respond(
        json={"code": 0, "data": {"state": "done", "full_zip_url": "https://files.test/no-md.zip"}})
    respx.get("https://files.test/no-md.zip").respond(
        content=_zip_bytes({"job/content_list.json": json.dumps(CONTENT_LIST)}))

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="no full.md"):
        c.extract_fulltext("http://x/paper.pdf")


@respx.mock
def test_extract_fulltext_empty_zip_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-5"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-5").respond(
        json={"code": 0, "data": {"state": "done", "full_zip_url": "https://files.test/empty.zip"}})
    respx.get("https://files.test/empty.zip").respond(content=_zip_bytes({}))

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="no full.md"):
        c.extract_fulltext("http://x/paper.pdf")


@respx.mock
def test_extract_fulltext_blank_md_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-8"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-8").respond(
        json={"code": 0, "data": {"state": "done", "full_zip_url": "https://files.test/blank-md.zip"}})
    respx.get("https://files.test/blank-md.zip").respond(content=_zip_bytes({"full.md": "  \n"}))

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="empty"):
        c.extract_fulltext("http://x/paper.pdf")


@respx.mock
def test_extract_fulltext_bad_zip_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-6"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-6").respond(
        json={"code": 0, "data": {"state": "done", "full_zip_url": "https://files.test/bad.zip"}})
    respx.get("https://files.test/bad.zip").respond(content=b"not a zip at all")

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(RuntimeError, match="not a valid zip"):
        c.extract_fulltext("http://x/paper.pdf")


@respx.mock
def test_extract_fulltext_timeout_leaves_resumable_ledger(monkeypatch, tmp_path):
    """Poll-budget exhaustion raises TimeoutError (MinerUTimeout) and leaves a
    resumable ledger entry — the next call must resume, never resubmit."""
    monkeypatch.chdir(tmp_path)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-7"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-7").respond(
        json={"code": 0, "data": {"state": "running"}})

    out = MinerUClient(use_mock=True, api_key="k", poll_interval=0.02, max_wait=0.1).extract_fulltext(
        "http://x/paper.pdf")
    assert out["parse_status"] == "mock"

    with pytest.raises(TimeoutError):
        MinerUClient(use_mock=False, api_key="k", poll_interval=0.02, max_wait=0.1).extract_fulltext(
            "http://x/paper.pdf")

    from tools.clients.mineru_client import _asset_key, _load_ledger
    entry = _load_ledger()[_asset_key("http://x/paper.pdf")]
    assert entry["state"] == "timeout" and entry["task_id"] == "t-7"


@respx.mock
def test_extract_fulltext_mock_semantics_on_submit_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    route = respx.post("https://mineru.net/api/v4/extract/task").respond(status_code=503)

    out = MinerUClient(use_mock=True, api_key="k", max_retries=1,
                       backoff_base=0).extract_fulltext("http://x/paper.pdf")
    assert out["parse_status"] == "mock"
    assert route.calls.call_count == 2  # one transient retry, then degrade

    with pytest.raises(MinerUError) as ei:
        MinerUClient(use_mock=False, api_key="k", max_retries=0).extract_fulltext("http://x/paper.pdf")
    assert ei.value.stage == "submit" and ei.value.kind == "http_5xx"


@respx.mock
def test_extract_fulltext_nonzero_code_envelope_raises(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 401, "msg": "invalid token"})

    c = MinerUClient(use_mock=False, api_key="bad")
    with pytest.raises(RuntimeError, match="code=401"):
        c.extract_fulltext("http://x/paper.pdf")


# --- wave 2: recovery contract (ledger / cache / precheck / staged retry) ---


@respx.mock
def test_precheck_rejects_landing_page_without_submit(monkeypatch, tmp_path):
    """An html landing page (doi.org style) is rejected at precheck — the submit
    route is never hit, so no quota is spent."""
    monkeypatch.chdir(tmp_path)
    pdf = respx.get(PDF_URL).respond(
        content=b"<html>journal landing page</html>",
        headers={"content-type": "text/html"})
    submit = respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-x"}})

    c = MinerUClient(use_mock=False, api_key="k")
    with pytest.raises(MinerUError) as ei:
        c.extract_fulltext(PDF_URL)
    assert ei.value.stage == "precheck" and ei.value.kind == "read_file"
    assert pdf.calls.call_count == 1
    assert submit.calls.call_count == 0


@respx.mock
def test_precheck_passes_real_pdf_magic_bytes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    respx.get(PDF_URL).respond(content=b"%PDF-1.7 fake", headers={"content-type": "application/pdf"})
    _mock_v4_routes("t-pc", "pc.zip", {"job/full.md": FULL_MD})

    out = MinerUClient(use_mock=False, api_key="k").extract_fulltext(PDF_URL)
    assert out["parse_status"] == "fulltext"


@respx.mock
def test_result_cache_serves_second_call_without_http(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _mock_v4_routes("t-cache", "cache.zip", {"job/full.md": FULL_MD})

    first = MinerUClient(use_mock=False, api_key="k").extract_fulltext(PDF_URL)
    n_calls_after_first = len(respx.calls)
    second = MinerUClient(use_mock=False, api_key="k").extract_fulltext(PDF_URL)

    assert second == first
    assert len(respx.calls) == n_calls_after_first  # zero new HTTP


@respx.mock
def test_ledger_resumes_submitted_task_after_timeout(monkeypatch, tmp_path):
    """The wave-2 core: a poll-budget timeout leaves the task resumable — the
    next call polls the SAME task_id and the submit route is hit exactly once."""
    monkeypatch.chdir(tmp_path)
    submit = respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-r"}})
    poll = respx.get("https://mineru.net/api/v4/extract/task/t-r")
    respx.get("https://files.test/r.zip").respond(content=_zip_bytes({"job/full.md": FULL_MD}))

    # phase 1: the task stays running forever -> deterministic budget exhaustion
    poll.respond(json={"code": 0, "data": {"state": "running"}})
    c = MinerUClient(use_mock=False, api_key="k", poll_interval=0.02, max_wait=0.1)
    with pytest.raises(TimeoutError):
        c.extract_fulltext(PDF_URL)

    # phase 2: the same task completes -> the resumed call polls it home
    poll.mock(side_effect=[httpx.Response(200, json={"code": 0, "data": {
        "state": "done", "full_zip_url": "https://files.test/r.zip"}})])
    c2 = MinerUClient(use_mock=False, api_key="k", poll_interval=0.02, max_wait=30)
    out = c2.extract_fulltext(PDF_URL)  # resumes t-r, sees done, downloads

    assert submit.calls.call_count == 1
    assert out["parse_status"] == "fulltext"


@respx.mock
def test_download_failure_retries_download_only(monkeypatch, tmp_path):
    """A 500 on the zip download re-downloads; submit is never repeated."""
    monkeypatch.chdir(tmp_path)
    submit = respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-d"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-d").respond(
        json={"code": 0, "data": {"state": "done", "full_zip_url": "https://files.test/d.zip"}})
    respx.get("https://files.test/d.zip").mock(side_effect=[
        httpx.Response(500),
        httpx.Response(200, content=_zip_bytes({"job/full.md": FULL_MD})),
    ])

    out = MinerUClient(use_mock=False, api_key="k", max_retries=1,
                       backoff_base=0).extract_fulltext(PDF_URL)
    assert out["parse_status"] == "fulltext"
    assert submit.calls.call_count == 1


@respx.mock
def test_transient_poll_ssl_error_recovered_within_budget(monkeypatch, tmp_path):
    """One SSL blip mid-poll is retried in place; no resubmit, parse succeeds."""
    monkeypatch.chdir(tmp_path)
    submit = respx.post("https://mineru.net/api/v4/extract/task").respond(
        json={"code": 0, "data": {"task_id": "t-s"}})
    respx.get("https://mineru.net/api/v4/extract/task/t-s").mock(side_effect=[
        httpx.Response(200, json={"code": 0, "data": {"state": "running"}}),
        httpx.ConnectError("ssl: UNEXPECTED_EOF_WHILE_READING"),
        httpx.Response(200, json={"code": 0, "data": {"state": "done",
                                                      "full_zip_url": "https://files.test/s.zip"}}),
    ])
    respx.get("https://files.test/s.zip").respond(content=_zip_bytes({"job/full.md": FULL_MD}))

    out = MinerUClient(use_mock=False, api_key="k", max_retries=1,
                       backoff_base=0).extract_fulltext(PDF_URL)
    assert out["parse_status"] == "fulltext"
    assert submit.calls.call_count == 1
