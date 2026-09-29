# EviSurvey — Evidence-grounded Survey Harness

**[English](README.md)** | **[简体中文](README.zh-CN.md)**

> An agentic harness that turns a research topic into a verified, figure-rich
> academic survey — built for the **World Models** topic at the Shanghai AI Lab
> 2026 Summer Camp Hackathon (Team Task 4).

EviSurvey converts one-shot LLM survey writing into a **multi-stage, traceable,
verifiable** research workflow. Every claim is grounded in a real paper, every
citation is checked against source metadata, and the final report ships as an
illustrated HTML/PDF. Generation runs on **DeepSeek Flash**; verification and
judging run on **Intern-S2-Preview**. Paper metadata comes only from the
SciVerse API — the LLM organizes and phrases, it never invents facts.

---

## ✨ Highlights

- **Anti-hallucination by construction** — paper metadata comes only from the
  SciVerse API; before a single sentence is drafted, a citation **whitelist**
  locks down which papers may be cited. Legality is converged at the source,
  not scrubbed afterwards.
- **Deterministic skeleton, bounded LLM decisions** — the pipeline order is
  hardcoded (reproducible runs, pinpointable failures); the LLM only makes
  budget-bounded decisions inside three "joints" (retrieval strategy, repair
  actions), choosing from an explicit action menu. A pure-code Goal Gate
  decides "passed" — **the LLM never grades its own homework**.
- **Verify–repair loop that cannot hollow out the survey** — every claim is
  checked three ways (whitelist membership, NLI against the cited evidence,
  attribution: someone else's work cannot be credited to the cited paper);
  repair may rewrite or re-bind claims instead of deleting them, and a
  coverage gate (≥ 70% text/figure retention vs. round 0) stops
  delete-everything "fixes". Failed gates end as `quality_failed` with a
  nonzero exit code — honest, not decorated.
- **Structured knowledge, not raw text** — papers become **Paper Cards**;
  claims become a **Claim-to-Evidence Map**; references become a **Citation
  Index**. Eight typed artifacts flow through the pipeline, connected by files
  (openable, replayable, auditable).
- **Figure-rich output** — figures and tables are mined from parsed papers
  into a `figure_bank` / `table_bank` and embedded in the final report.

---

## 🏗️ Architecture

The system is introduced here by the three capabilities that make it trustworthy — exactly the three stages data flows through: **Retrieval Strategy** (where the material comes from) → **Citation Verification** (whether every sentence holds up) → **Quality Assessment** (whether the deliverable is acceptable as a whole).

```
 Research topic "World-model survey"
      │
      ▼
┌────────────────────────────────────────────────┐
│ ① Retrieval Strategy — where material comes from  │
│    pick search directions → real-API search &     │
│    parse → build cards / collect evidence →       │
│    lock the citation whitelist                    │
└───────────────────┬────────────────────────────┘
                    │ whitelist + evidence store + figure bank
                    ▼
          [ Writing: template skeleton + optional LLM prose ]
          (written for verification: claims come first)
                    │ output/survey.md
                    ▼
┌────────────────────────────────────────────────┐
│ ② Citation Verification — does every sentence hold up │
│    structural + per-claim NLI + attribution checks │
│    → repair (repair agent) → re-verify             │
│    (loop adjudicated by pure code, see below)      │
└───────────────────┬────────────────────────────┘
                    │ repaired survey.md
                    ▼
┌────────────────────────────────────────────────┐
│ ③ Quality Assessment — is the deliverable acceptable │
│    in-run scores + baseline comparison + render    │
│    → post-hoc three-layer eval (redundancy / NLI / │
│      LLM judge) → honest status: completed or      │
│      quality_failed                               │
└───────────────────┬────────────────────────────┘
                    ▼
     output/final_report.html / .pdf + quality reports
```

### ① Retrieval Strategy — where the material comes from

*What to search* is decided by a bounded **strategy agent** (joint 1, `--mode
full`): probe → cluster → name directions → sample → show the LLM a "report
card" (hits, overlap, relevance, year span per direction) → it proposes edits
→ retry, ≤ 3 rounds, accepted by a deterministic gate with best-so-far
rollback.

*How to search*: per-direction SciVerse `meta-search` (the **only** source of
paper metadata) with LLM zh→en query translation; optional MinerU PDF parsing
(off by default; on failure it degrades to the real abstract, never fabricates).

*What becomes usable*: papers get cards (deep with full text, shallow with
abstract only); evidence chunks are collected and bound to papers; figures and
tables are mined. Then — before any writing — a **citation whitelist**
(`cache/citation_ready_set.json`) is locked, ranking recency × influence 50/50
so pure recency can't squeeze out the classics. Everything downstream (write /
verify / repair / render) honors this whitelist.

### ② Citation Verification — does every sentence hold up

The draft (`write_survey.py`) is **written for verification**: deterministic
section skeleton, LLM (optional, `EVISURVEY_WRITER_LLM=1`) plans claims as
data (text / citation / scope) before prose, and internal jargon or raw paper
ids can never leak into the text.

`verify_citations.py` splits the text into claims and checks each three ways:
**structural** (is the cited id whitelisted; does the figure exist), **semantic
NLI** (does the cited evidence actually say this), and **attribution** (a
claim labeled "this paper's own contribution" may not rest on a quote that is
actually about prior work).

Failures are repaired by the **repair agent** (joint 2,
`EVISURVEY_REPAIR_AGENT=1`): failures are grouped mechanically into six
classes, the LLM picks typed actions (re-map citation / rewrite / swap or
backfill evidence / delete as last resort), and every rewritten claim is
re-checked by NLI before it counts as repaired.

The verify–repair loop (≤ 2 repair rounds) is adjudicated by the **Goal Gate**
(`harness/goal_gate.py`) — pure code, LLM has no say in "passed". Gate order:
coverage broke 70%? → attribution violations or evidence gaps? → invalid
citations and unsupported claims both zero? → budget exhausted? → no progress
(fixpoint)? A best-so-far rollback keeps the strongest round (fewer problems
first; longer text wins ties — you can't farm metrics by deleting), and the
reported numbers always describe the **delivered** text.

### ③ Quality Assessment — is the deliverable acceptable

`render_report.py` computes `evaluation_report.json` (overall / citation
validity / evidence grounding / visualization) and `review_report.json`
(including a `topic_relevance` gate that must be answered at acceptance when
below 0.7), renders the HTML/PDF, and compares against a baseline survey.
Post-hoc, `scripts/run_survey_eval.py` runs a three-layer evaluation — L0
redundancy, L1 NLI sampling, and an **L3 LLM judge that actually reads the
document** (the only layer that does). A failed gate never blocks rendering,
but the run state and exit code honestly say `quality_failed`.

> **Where the code lives**: `harness/` is the controller (agent loop, planner,
> tool registry, goal gate, `agents/` joints); `tools/` does the work
> (`phases/` knowledge pipeline, write / verify / revise / render, `clients/`
> external APIs, `models/` schemas). Steps communicate through files
> (`requests/`, `cache/`, `output/`) — the files are the interface.
> Full walkthrough: `docs/系统架构与流程.md`.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone <repo-url> && cd AILabMacroHard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires **Python ≥ 3.11**.

### 2. Configure environment

Copy `.env.example` → `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `INTERN_API_BASE_URL` | ✅ | Intern-S2-Preview endpoint |
| `INTERN_API_KEY` | ✅ | Intern-S2-Preview API key |
| `INTERN_MODEL_NAME` | — | Defaults to `intern-s2-preview` |
| `SCIVERSE_API_KEY` | ✅ | SciVerse paper search / RAG |
| `MINERU_API_KEY` | ✅ | MinerU PDF parsing |
| `HF_ENDPOINT` | — | HuggingFace mirror, e.g. `https://hf-mirror.com` |
| `HEAVY_LLM_*` | — | Optional generation model (e.g. DeepSeek Flash); `HEAVY_LLM_BASE_URL` / `HEAVY_LLM_API_KEY` / `HEAVY_LLM_MODEL`. Falls back to Intern-S2 |

> `API_BASE_URL` / `API_KEY` are accepted as aliases.

### 3. Run the harness

```bash
# Full run
python main.py --topic "世界模型综述"

# Fast smoke (cap retrieval + parsing)
python main.py --topic "世界模型综述" --max-papers 5 --max-core-papers 3 --mode full

# Only emit request files, skip the pipeline
python main.py --topic "世界模型综述" --prepare-only --mode full
```

CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | — *(required)* | Survey topic |
| `--language` | `zh` | Survey language (`zh` / `en`) |
| `--mode` | `demo` | `demo` / `full` (`full` enables the strategy agent) |
| `--max-papers` | `40` | Cap on retrieval |
| `--max-core-papers` | `15` | Cap on parsed core papers |
| `--use-mineru` | off | Enable MinerU PDF parsing |
| `--prepare-only` | off | Emit request files only |

---

## 📦 Outputs

Artifacts land in `output/`:

- `final_report.html` / `final_report.pdf` — the figure-rich survey.
- `survey.md` / `survey_revised.md` — drafted / revised Markdown.
- `citation_result.json` — citation verification results.
- `evaluation_report.json` / `review_report.json` — quality evaluation.
- `repair_log.json` — what the repair agent did and why.
- `final_state.json` — full run state (`completed` or `quality_failed`) for traceability.

Intermediate request files are written to `requests/`; the parsed knowledge
cache lives in `cache/`.

---

## 🧪 Testing

```bash
# Unit tests (fake LLM, no network)
pytest tests/unit/ --log-cli-level=WARNING

# Integration (real SciVerse + fake LLM + real MinerU degrading per-paper)
pytest tests/integration/ -v

# Full real API (all three APIs live)
pytest tests/integration/ -v -m real_api
```

---

## 🗂️ Project Structure

```
.
├── main.py                      # CLI entry → AgentLoop
├── config.py                    # env-backed AppConfig
├── llm_client.py                # Intern-S2 client (+ optional heavy model)
├── harness/                     # controller: agent loop, planner, registry,
│   │                            #   goal gate, state, memory
│   └── agents/                  # bounded agent joints: strategy / repair
├── tools/
│   ├── clients/                 # SciVerse, MinerU, fake LLM
│   ├── phases/                  # P1–P6 knowledge pipeline
│   ├── models/                  # typed request/artifact schemas
│   ├── nlp/                     # NLI verifier, data cleaner
│   ├── verify/                  # claim mapper, source contract, structural checks
│   ├── knowledge_pipeline_worker.py   # knowledge pipeline entry (P1–P6)
│   ├── write_survey.py          # draft (written for verification)
│   ├── verify_citations.py      # three-layer citation verification
│   ├── revise_survey.py         # repair (deletion / repair agent)
│   └── render_report.py         # evaluation + HTML/PDF render
├── tests/                       # unit + integration
├── docs/                        # design docs (系统架构与流程.md is the map)
├── requests/  output/  cache/   # runtime I/O
└── .env.example
```

---

## 🔌 External API Contracts

Verified against the live APIs on 2026-07-05 (full field-level contracts in `docs/外部服务接口/`).

- **Intern-S2-Preview** — verification / decision LLM, OpenAI-compatible.
  ~1 request / 2s (throttled in the client). Generation tasks route to the
  optional heavy model (DeepSeek Flash, thinking disabled — Intern reasons by
  default and burns small token budgets on thinking) with Intern as fallback.
- **SciVerse** — `https://api.sciverse.space`. `POST /meta-search` (metadata;
  the only source of paper metadata) and `POST /agentic-search` (citable
  chunks with page numbers).
- **MinerU** — `https://mineru.net`. PDF → structured text + figures.

---

## 🛡️ Anti-Hallucination (five enforced locks)

1. **Metadata from SciVerse only** — titles, authors, years, venues are never LLM-generated.
2. **Claim-to-evidence binding** — every claim points at a specific, page-numbered evidence chunk (`claim_map`).
3. **Whitelist + NLI verification** — every citation passes the whitelist and a per-claim NLI check.
4. **Attribution check** — someone else's work can never be credited to the cited paper.
5. **Figures from real sources only** — `figure_bank` / `table_bank` mined from parsed papers, or visuals derived from verified content.

---

## 📄 License

Hackathon project for the Shanghai AI Lab 2026 Summer Camp. See project files
for contributor and asset attributions.
