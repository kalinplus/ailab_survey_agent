# AGENTS.md

Project-level rules for EviSurvey. This file **supplements** the personal principles in `~/.pi/agent/AGENTS.md` (global defaults are not repeated here and always apply; within this repo this file overrides them where they differ). Keep this file minimal on purpose — details live one hop away (progressive disclosure), and live status lives in `PROGRESS.md`, never here.

**EviSurvey** — evidence-grounded survey harness (originated as 上海人工智能实验室 2026 暑期夏令营 Task 4, topic: World Models / GameCraft; 比赛已结束，2026-09 起转入自主升级阶段): given a topic, retrieve real papers → build an evidence store → write a figure-rich survey → NLI-verify every claim → repair → render HTML/PDF. Core LLM default: Intern-S2-Preview (`INTERN_API_BASE_URL` / `INTERN_API_KEY`). Judgment/classification tasks may be offloaded to TypeSafe JEV (`TYPESAFE_API_KEY`, `EVISURVEY_JEV=auto|on|off`) — must degrade gracefully to the Intern-S2 path when its key is absent.

## Every-task rules

- Read `PROGRESS.md` at session start (Current / Next / Log); write back when a bounded unit of work completes. `PROGRESS.md` and `specs/` are committed to git.
- Real-API-first, no production mocks: SciVerse / MinerU / Intern-S2 must work live; on real parse failure degrade to the real SciVerse abstract (`parse_status="abstract_only"`), never fabricate. Unit-test doubles must mirror the real contract shape.
- The anti-hallucination chain is enforced — never break it (paper metadata only from API results; claims bind to evidence via the claim map and are re-checked verbatim; citations must pass the whitelist; figures/visuals only from parsed or verified content). Full chain: `docs/系统架构与流程.md` §7.
- Acceptance = mechanism tests pass AND a document-level read of the deliverable. Never sign off a batch on numbers alone — rules in `docs/验收纪律.md` (includes the L3 judge requirement).
- `ref/` (SurGE, SurveyX, LiRA) is design-reference only: do not modify, do not copy code (semantic plagiarism check).
- Survey output defaults to 中文 academic style (showcase used `--language en`).

## Environment

Conda `base` (`/Users/kalin/miniconda3`, Python 3.13) — no dedicated project env; `.env` holds the three external API keys. Full env-var inventory and HF/NLI notes: `docs/运行与测试.md`.

## Core commands

```bash
pytest tests/unit/ --log-cli-level=WARNING                        # unit tests (fake LLM, no network)
python main.py --topic "世界模型综述" --max-papers 5 --mode full    # full-flow CLI smoke
python scripts/run_showcase_demo.py                               # delivered showcase demo
```

All other commands (integration, real-NLI, three-layer eval, A/B scripts) and run traps: `docs/运行与测试.md`.

## Where to look

- System architecture, end-to-end pipeline, agent joints, run-mode switches — `docs/系统架构与流程.md` (总览；§10 文档地图)
- v2 per-joint specs & audit — `docs/RealAgent/`
- Running / testing / evaluation / traps — `docs/运行与测试.md`
- Acceptance discipline — `docs/验收纪律.md`
- Parallel worktree dispatch protocol — `docs/并行派发协议.md`
- External API contracts — `docs/外部服务接口/`
- Hackathon-phase docs, superseded (team division, module B/C work lists, old interface spec) — `archive/docs-hackathon-2026-07/`
