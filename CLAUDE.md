# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: EviSurvey — Evidence-grounded Survey Harness

Hackathon project for 上海人工智能实验室 2026 暑期夏令营, Task 4 (3人团队). Topic: World Models / GameCraft — generate a verified, figure-rich academic survey (HTML/PDF). **综述标题正在修改中**；核心约束不变：核心 LLM 必须是 Intern-S2-Preview（`INTERN_API_BASE_URL` / `INTERN_API_KEY`），评审时主办方会替换 API Key 重跑.

**Status (2026-08)**: Public submission delivered (final report + showcase demo + README). v2 improvement phase — bounded agent loops at three DAG-shaped holes (strategy / repair / goal gate). **关节1（策略 Agent）、关节2（修复 Agent + Goal Gate）、关节3（coverage 闸补全）均已实现**（`harness/agents/` + `harness/goal_gate.py`）。Plans and audit live in `docs/RealAgent/`; read `docs/RealAgent/Agent关节改造-审计与计划.md` (总计划) and per-joint module docs before touching the strategy/verify/revise paths.

## Architecture: Three Modules + Linear Pipeline

```
User topic ─▶ Module A (harness/) — Planner / AgentLoop / tool registry / state / memory / skills
                    │ emits request files to requests/
                    ▼
             Module B (tools/knowledge_pipeline_worker.py + tools/phases/) — P1–P6 linear
                    │ produces cache/knowledge_bundle.json (8 typed artifacts)
                    ▼
             Module C (write_survey → verify_citations → revise_survey → render_report)
                    ▼
             output/final_report.html / .pdf + verification + evaluation reports
```

- **A** (`harness/`): `AgentLoop.run()` (agent_loop.py:115-189) is a hardcoded sequence, not LLM-driven. `search_strategy_builder.py` builds sub-queries (3-level fallback: LLM → cluster → template; LLM only in `--mode full`). **`harness/agents/` = 关节1 策略 Agent** (spec `docs/RealAgent/关节1-策略Agent-方案与测试.md`): `loop.py` shared bounded typed-action chassis (budget breakers + trajectory jsonl; joint 2 will reuse it), `relevance.py` report card (bge embedding, keyword fallback), `strategy_agent.py` probe→cluster→naming→sample→edit-loop(≤3 rounds)→deterministic gate→best-so-far rollback. Enabled when `--mode full` (disable via `EVISURVEY_STRATEGY_AGENT=0`); toolbox = permission boundary; output schema unchanged (P1–P6 untouched). `tool_registry.py` dispatches B/C tools in-process with timeout + result truncation. `--prepare-only` stops after A emits requests.
- **B** phases (`tools/phases/`): P1 decompose → P2 survey analyzer → P3 retriever (real meta-search per aspect; MinerU off unless `use_mineru`) → P5 cards/evidence/synthesis (`paper_cards`, `evidence_store`, `figure_bank`, `table_bank`, `taxonomy`, `citation_index`) → P6 assembler. **No P4** (superseded by SciVerse `agentic-search`).
- **C**: `write_survey.py` drafts from the bundle (template-based, not free LLM writing) → `verify_citations.py` (regex whitelist + claim_map NLI) → `revise_survey.py` → `render_report.py` (HTML/PDF + visual assets + `evaluation_report.json` + `review_report.json` readiness gates). **关节2 修复回路**：`agent_loop._verify_repair_loop` = verify → 确定性 Goal Gate（`harness/goal_gate.py`，三道闸：invalid=0 + unsupported 清零或预算尽 + coverage 保留率 ≥ `EVISURVEY_COVERAGE_MIN`（默认 0.7，关节3：以 round 0 为基线的正文/图表引用保留率，拦删除掏空；破线即停修 + `stop_reason=coverage_fail`）；不动点 + best-so-far 回滚）→ revise (≤2 轮，每轮结果都被消费；goal_gate 写入 final_state.json)。`revise_survey` 双内核：默认删除式（交付行为不变）；`EVISURVEY_REPAIR_AGENT=1` 走 `harness/agents/repair_agent.py`（A–E 失败分组 → LLM 批量选动作 remap/rewrite/swap/backfill/delete-最后手段 → 改写句增量重过 NLI → `output/repair_log.json` + trajectory）。引用非法 id 的 claim 是 A 类连带，不进 B/C 组（一次 remap 修全部出现）。
- Typed models in `tools/models/`; clients in `tools/clients/`; NLI + cleaner in `tools/nlp/`; claim mapper + structural checks in `tools/verify/`.
- Runtime dirs: `requests/` (A→B/C request files), `cache/` (artifacts), `output/`, `logs/` (run.jsonl + state.json), `memory/` (strategy memory persisted across runs).

**Known traps (verified, see audit §1 in docs/RealAgent/总计划):** NLI defaults to a keyword FakeNLIModel (real model needs `EVISURVEY_REAL_NLI=1` — hard precondition for repair experiments); `use_mineru` now opt-in via `--use-mineru` (default False → demo-mode evidence store stays thin, masked only by FINAL_SEED_PAPERS); the old discarded-second-verify half-loop is fixed by `_verify_repair_loop` + Goal Gate; full-mode runs with repair agent can OOM MPS during full verify (agentic backfill grows evidence lists × claim `best_match` — v2.1 backlog), use `EVISURVEY_NLI_DEVICE=cpu` for such runs (E2E 2026-08-21 verified green on cpu); plotting modules pin `matplotlib.use("Agg")` because ToolRegistry worker threads cannot use the macosx GUI backend. (Dead `harness/agent_tool_loop.py` was removed when the shared chassis landed in `harness/agents/loop.py`.)

### Reproducible final run

Prebuilt topic-aligned seed data in `cache/final_*.json`. `FINAL_SEED_PAPERS=1` makes Module C load these instead of a live bundle — reproducibility mechanism for the delivered run, **not a mock**; regenerate with `scripts/build_final_seed_papers.py`. `scripts/run_showcase_demo.py` is the 8-stage demo runner (sets `FINAL_SEED_PAPERS=1`, `SHOWCASE_LOG=1`, `GENERATE_IMAGE_QUALITY=low` itself).

## Anti-Hallucination (enforced, keep it that way)

- Paper metadata (title/authors/year/venue) only from SciVerse API results, never LLM-generated.
- Claims bind to evidence via claim map (`tools/verify/claim_mapper.py`); SciVerse `agentic-search` chunks ground claims.
- Every citation cross-checked against `citation_index` by the NLI verifier (`tools/nlp/nli_verifier.py`).
- Figures/tables mined into `figure_bank`/`table_bank`; generated visuals derived from verified content only.

## Commands

```bash
# Unit tests (fake LLM, no network)
pytest tests/unit/ --log-cli-level=WARNING
pytest tests/unit/test_phase3.py::test_name -v          # single test

# Integration (real SciVerse + fake LLM; needs .env)
pytest tests/integration/ -v
pytest tests/integration/ -v -m real_api                # all three APIs live, expensive
RUN_NLI_REAL=1 pytest -m nli_real tests/unit/test_nli_verifier.py -v   # real NLI model

# CLI full flow
python main.py --topic "世界模型综述" --max-papers 5 --max-core-papers 3 --mode full  # fast smoke
python main.py --topic "..." --prepare-only --mode full     # A-owned requests only

python scripts/run_strategy_ab_test.py                  # 策略关节 A/B/C 对照 (real APIs, ~10min)
python scripts/run_repair_ab_test.py                    # 修复关节 A/B/C 对照 (real APIs + real NLI, ~1h)
RUN_NLI_REAL=1 python -m pytest -m nli_real tests/integration/test_repair_agent.py -v  # 真 NLI 修复集成

python scripts/run_showcase_demo.py                     # final delivery demo
```

Add `-s --log-cli-level=INFO` for the full pipeline trace; default WARNING surfaces degradations (MinerU → abstract_only, API errors, timeouts).

## Environment

- **Conda env: `base`** (`/Users/kalin/miniconda3`, Python 3.13.12). No dedicated project env; all deps installed there (pytest, httpx, pydantic, python-dotenv, sentence-transformers 5.6.0, torch 2.11.0). Unit suite verified green on it.
- `.env` keys (present): `INTERN_API_BASE_URL`, `INTERN_API_KEY`, `SCIVERSE_API_KEY`, `MINERU_API_KEY`, `SCIVERSE_MIN_INTERVAL`. Optional: `INTERN_MODEL_NAME` (default `intern-s2-preview`), `HF_ENDPOINT`, `EVISURVEY_LANGUAGE` (zh) / `EVISURVEY_MODE` (demo), `EVISURVEY_REAL_NLI`, `EVISURVEY_STRATEGY_AGENT` (default on in full mode), `EVISURVEY_REPAIR_AGENT` (default off — on = joint-2 action repair in revise), `EVISURVEY_COVERAGE_MIN` (default 0.7 — joint-3 coverage gate threshold), `EVISURVEY_EMBED_MODEL` (default `BAAI/bge-small-en-v1.5`), `STRATEGY_*`, `REQUEST_TIMEOUT_SECONDS` / `MAX_LLM_RETRIES` / `TOOL_TIMEOUT_SECONDS` / `TOOL_RESULT_MAX_CHARS`. `API_BASE_URL`/`API_KEY` are aliases. `config.py` loads `.env` and strips `all_proxy`.
- NLI/embedding models download from HuggingFace on first use; if slow set `HF_ENDPOINT=https://hf-mirror.com`. NLI inference auto-picks MPS on Apple Silicon (~2.7x faster than the 4 CPU cores); override with `EVISURVEY_NLI_DEVICE=cpu|mps|cuda`.

## Language

Code/comments English; explanations to user 中文; commit messages English; survey output 中文 academic style (showcase used `--language en`).

## Preferences

- Fail fast, let exceptions bubble up; error handling only at system boundaries (API calls, file I/O).
- Minimal abstraction — flat, readable code; edit existing files over creating new ones.
- **Real-API-first, no production mocks**: harness MUST work under real SciVerse / MinerU / Intern-S2. Clients default `use_mock=False`; on real parse failure degrade to real SciVerse abstract (`parse_status="abstract_only"`), never emit mock data. Validate integration changes against real APIs. Test doubles are fine for unit tests but must mirror the real contract shape.

## External API Contracts (verified 2026-07-05; full field lists in `docs/外部服务接口/`)

- **Intern-S2-Preview** (core LLM, OpenAI-compatible): `INTERN_API_BASE_URL`/`INTERN_API_KEY`, model `intern-s2-preview`. Surface `.chat(...)`/`.json_chat(...)`; ~1 req / 2s (client enforces). `config.py` normalizes bare hosts to `.../api/v1`.
- **SciVerse** (`https://api.sciverse.space` — NOT the website host): `POST /meta-search` returns `{"results": [...]}`; `POST /agentic-search {query, top_k}` returns `{"hits": [...]}` (different key!). Boosts `freshness_boost`/`impact_boost` accept `"MILD"` only (`"HIGH"` rejected), apply only when `sort` unset. agentic-search `author` field is mangled — get authors via meta-search. `GET /content?doc_id=` reads full text; `/resource` unverified (507 in probes).
- **MinerU** (`https://mineru.net`): `POST /api/v1/agent/parse/url {url, light}`; `POST /api/v4/extract/task {url}` → poll `GET /api/v4/extract/task/{id}` until succeeded/failed. Needs a direct PDF URL (doi.org landing pages are not PDFs).

## Reference Repos

`ref/` (SurGE, SurveyX, LiRA) — design references only. Do not modify; do not copy code (semantic plagiarism check — code must be original).
