# Project: Academic Paper Survey Generation Harness (学术论文综述生成 Harness)

## Context

Hackathon project for 上海人工智能实验室 2026 暑期夏令营. Task 4 (团队挑战题, 3人团队).

**Topic**: 世界模型 (World Models) — 生成一篇图文并茂的综述 (HTML/PDF).

**Core constraint**: Harness 的核心大模型调用必须使用 Intern-S2-Preview API. 评审时主办方会替换 API Key 重新运行.

## Key Requirements

- No hallucinated citations, no fact-less descriptions
- Output must be rich with figures and text (图文并茂)
- Must support `API_BASE_URL` and `API_KEY` via environment variables
- Code must be original (semantic plagiarism check will be performed)

## Available Resources

- **Intern-S2-Preview API**: https://chat.intern-ai.org.cn
- **MinerU 文档解析 API**: https://mineru.net/ (for parsing academic PDFs)
- **Sciverse 科学文献库 API**: https://sciverse.space/ (for paper search/retrieval)

## Architecture

```
User Query ("世界模型 综述")
    ↓
Agent Loop (Intern-S2-Preview as core LLM)
    ↓
Tool Calls:
  - Sciverse API → search & retrieve papers
  - MinerU API → parse PDF content
  - File I/O → read/write drafts, manage references
  - Web search → supplementary info (optional)
    ↓
Iterative: summarize → classify → organize → write → verify citations
    ↓
Output: HTML survey with figures
```

### Agent Loop Design

1. **Task Decomposition**: Break "world models survey" into sub-tasks (literature search, categorization, timeline, future directions)
2. **Tool Orchestration**: Call Sciverse to find papers → MinerU to parse → LLM to summarize/classify
3. **Citation Verification**: Cross-check every citation against actual paper metadata from Sciverse
4. **Iterative Refinement**: Review drafts, check for hallucinations, regenerate if needed
5. **Output Generation**: Render final survey as HTML with embedded figures

### Anti-Hallucination Strategy

- All paper metadata (title, authors, year, venue) must come from Sciverse API results, never LLM-generated
- Every claim must be traced back to a specific paper reference
- Verification pass: extract all citations from draft → validate each against source
- Figures must come from parsed papers (via MinerU) or be generated from verified content

## Deliverables Checklist

- [ ] Working Harness prototype (source code + run instructions)
- [ ] API config via environment variables (`INTERN_API_BASE_URL`, `INTERN_API_KEY`)
- [ ] Output: world models survey (HTML/PDF, 图文并茂)
- [ ] Intern-S2-Preview capability analysis table
- [ ] Presentation slides

## Bonus Features (prioritized)

1. **Memory system**: persist paper metadata across sessions to avoid re-searching
2. **Skill system**: packaged workflow for "generate survey from topic X"
3. **Sub-agent parallelism**: concurrent paper parsing and summarization
4. **Context compression**: handle long paper collections within model context limits

## Language

- Code and comments: English
- Comments and explanations to user: 中文
- Commit messages: English
- Survey output: 中文 (academic style)

## Preferences

- Fail fast, let exceptions bubble up
- Only add error handling at system boundaries (API calls, file I/O)
- Minimal abstraction — prefer flat, readable code over over-engineering
- Edit existing files over creating new ones
- **Real-API-first, no production mocks**: the harness MUST work under the real
  SciVerse / MinerU / Intern-S2 APIs. Never rely on mock fallbacks to make the
  pipeline produce output — mock fallbacks hide real failures and silently
  inject fake content (anti-hallucination risk). Production clients default to
  `use_mock=False`; when a real parse fails, degrade to the real SciVerse
  abstract (`parse_status="abstract_only"`) rather than emit mock data. Validate
  every integration change against the real APIs (stronger than mock-based unit
  tests, which can match a wrong contract and give false confidence). Test
  doubles are fine for fast unit tests but must mirror the real contract shape.
- **Test logging**: when running `pytest`, always pass `--log-cli-level=WARNING`
  by default — it surfaces process problems (MinerU degradation → mock,
  SciVerse / Intern-S2 failures, timeouts) without flooding the INFO progress
  logs. Add `-s` and bump to `--log-cli-level=INFO` only when you need the full
  pipeline trace (`[worker] P1/P2/P3 done`, per-call summaries). Why a flag is
  needed: stdlib root logger defaults to WARNING and `setup_logging()` (which
  raises it to INFO) only runs inside `worker.run()` / `verify_citations.run()`,
  so most tests stay silent without `--log-cli-level`.

## Test & Run Commands

Unit tests (fake LLM, no network): fast contract checks. Default log level surfaces
degradation/failures without flooding progress logs.

```
pytest tests/unit/ --log-cli-level=WARNING
```

Integration / e2e (real SciVerse + fake LLM + real MinerU degrading per-paper):
exercises A→B phases 1–6 end-to-end. Needs `SCIVERSE_API_KEY` (or a `.env`).

```
pytest tests/integration/ -v
```

Full real API (adds real Intern-S2, all three APIs live): opt-in marker, expensive.

```
pytest tests/integration/ -v -m real_api
```

CLI full flow (A planner → B/C tools): real run of the harness.

```
python main.py --topic "世界模型综述"            # full run
python main.py --topic "..." --max-papers 5 --max-core-papers 3   # fast smoke (caps P3 retrieval + parse)
python main.py --topic "..." --prepare-only      # A-owned requests only, no B/C
```

Add `-s --log-cli-level=INFO` when you need the full pipeline trace (`[worker] P1/P2/P3 done`, per-call summaries).

## External API Contracts (verified against live APIs 2026-07-05)

- **Intern-S2-Preview** (core LLM, OpenAI-compatible): base `INTERN_API_BASE_URL`,
  key `INTERN_API_KEY`, model `intern-s2-preview`. Duck-typed surface
  `.chat(messages, *, temperature) -> str` / `.json_chat(...) -> dict`.
- **SciVerse meta-search** (metadata list / filtering): base
  `https://api.sciverse.space` (NOT the `sciverse.space` website host — that
  404s). `POST /meta-search`; body keys `query` / `filters` / `fields` / `page`
  / `page_size` / `sort`, plus `freshness_boost` / `impact_boost` (send each
  only when set; they bias toward recent / highly-cited results and apply when
  `sort` is NOT set; verified valid value `"MILD"`, `"HIGH"` is rejected).
  Response `{"results": [...]}`. Each result: `unique_id`, `title`, `author`
  (list of `{orcid, name}`), `publication_published_year` (float),
  `publication_venue_name_unified`, `abstract`, `keywords`, `citation_count`
  (float), `doi`, `access_oa_url` (list), `locations` (list of
  `{type, is_oa, url, license}`), `is_content_accessible`. Filter entry shape:
  `{"field": "...", "operator": "FILTER_OP_GTE"|"FILTER_OP_LTE"|..., "value": ...}`.
- **SciVerse agentic-search** (RAG / citable chunks): `POST /agentic-search`
  `{query, top_k}`; response `{"hits": [...]}` (NOTE: `hits`, not `results`).
  Each hit carries a citable fragment: `chunk` (text), `doc_id`, `chunk_id`,
  `page_no`, `offset`, `score`, `title`, `publication_published_year`,
  `publication_venue_name_unified`. Use to ground claims/evidence. Caveat: the
  `author` field comes back mangled (ASCII codepoint lists) — do not use it;
  fetch authors via meta-search instead. `GET /content?doc_id=...` reads full
  text; `GET /resource?file_name=...` downloads figures/attachments (the
  `file_name` mapping is unverified — `/resource` returned 507 in probes).
- **MinerU** (PDF parsing): base `https://mineru.net`. `POST /api/v1/agent/parse/url`
  `{url, light}`; `POST /api/v4/extract/task` `{url}` → `task_id`, poll
  `GET /api/v4/extract/task/{task_id}` until `status` ∈ {succeeded, failed}.
  Needs a real PDF URL; doi.org landing pages are not direct PDFs.
