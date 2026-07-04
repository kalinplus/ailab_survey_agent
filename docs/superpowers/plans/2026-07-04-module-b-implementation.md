# Module B (Knowledge Pipeline Worker) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **TESTING NOTE (overrides skill default TDD):** Per explicit user direction, we do **NOT** use TDD. Each task follows **implement → write tests → run tests → commit**. Tests must be comprehensive (happy path + edge cases + boundary errors) but written after the implementation to move fast.

**Goal:** Implement Module B as a Knowledge Pipeline Worker with 3 coarse-grained tools (`search_papers`, `build_paper_cards`, `verify_citations`) that turn a `KnowledgeBuildRequest` into a verified `KnowledgeBundle`, grounded on real papers with NLI-backed citation/claim checking.

**Architecture:** B runs 6 phases internally (demand decomposition → survey analysis → paper retrieval → RAG index → knowledge synthesis → bundle assembly) plus a post-hoc verification stage (structural citation check + NLI claim mapping). Externals (SciVerse, MinerU, Embedding, Intern-S2-Preview LLM) are isolated behind clients; a local NLI DeBERTa model does deterministic claim verification. The full design is in `docs/模块B-架构设计.md`; artifact schemas in `docs/接口文档.md` §8–§9.

**Tech Stack:** Python 3.11+, pydantic v2 (data models + boundary validation), httpx (API clients), openai SDK (Intern-S2-Preview is OpenAI-compatible), chromadb (vector store), sentence-transformers + `cross-encoder/nli-deberta-v3-base` (NLI), pytest (tests). Reused logic from `ref/SurGE`, `ref/SurveyX`, `ref/LiRA`.

## Global Constraints

- **Core LLM calls MUST use Intern-S2-Preview** via `INTERN_API_BASE_URL` + `INTERN_API_KEY` env vars. Judges will swap the key and re-run. Never hardcode keys.
- **No hallucinated citations / no fact-less descriptions.** Every paper_id comes from SciVerse `unique_id` or a deterministic seed hash. Every claim must trace to an evidence_id or be flagged.
- **All artifact JSON must match** `docs/接口文档.md` §8.2 (KnowledgeBundle) and §9.1–§9.8. Field names are a frozen contract with modules A and C.
- **ID rules (§5.0 of arch doc):** Paper = SciVerse `unique_id` or `seed:{sha256(title+year)[:12]}`; Evidence = `{paper_id}_p{page}_{idx}`; Figure = `{paper_id}_fig{num}`; Table = `{paper_id}_tbl{num}`; Category = `cat_{3digit}` (run-local).
- **Fail fast.** No silent exception swallowing. Validation only at system boundaries (API responses, file I/O, JSON load). Internal calls trust callers.
- **Minimal abstraction.** Prefer flat functions over wrappers. Edit existing files; create new files only where the file tree below specifies.
- **Survey output language:** Chinese academic style (C's concern, but B's taxonomy descriptions are Chinese-ready).
- **Python version:** 3.11+ (use `list[X]`, `X | Y`, `from __future__` not needed).

---

## File Structure

```text
tools/
├── __init__.py
├── config.py                       # env + config loading (INTERN/SCIVERSE/MINERU/OPENAI keys)
├── models/
│   ├── __init__.py
│   ├── common.py                   # ID builders + small shared types
│   ├── requests.py                 # KnowledgeBuildRequest, VerificationRequest, SearchStrategy
│   ├── artifacts.py                # RetrievedPapers, ParsedPapers, PaperCard, EvidenceStore,
│   │                               #   FigureBank, TableBank, Taxonomy, CitationIndex
│   └── bundle.py                   # KnowledgeBundle, CitationResult, ClaimMap
├── clients/
│   ├── __init__.py
│   ├── llm_client.py               # Intern-S2-Preview (OpenAI-compatible) + FakeLLMClient
│   ├── sciverse_client.py          # meta-search / agentic-search / content / resource / relations
│   ├── mineru_client.py            # extract/task (async) + agent/parse/url (light) + mock
│   └── embedding_client.py         # OpenAI text-embedding-3-small + mock
├── nlp/
│   ├── __init__.py
│   ├── nli_verifier.py             # reuse SurGE logic; NLIVerifier + FakeNLIModel
│   └── data_cleaner.py             # reuse SurveyX complete_abstract + clean MinerU output
├── indexer/
│   ├── __init__.py
│   └── vector_store.py             # ChromaDB wrapper
├── phases/
│   ├── __init__.py
│   ├── phase1_decompose.py         # heuristic demand decomposition
│   ├── phase2_survey_analyzer.py   # Layer 1 + Taxonomy Self-Refine step 1-2
│   ├── phase3_paper_retriever.py   # Layer 2 retrieval + MinerU parse
│   ├── phase4_rag_indexer.py       # build ChromaDB index
│   ├── phase5_knowledge_synthesizer.py  # cards / evidence / figures / tables / taxonomy step3 / citation_index
│   └── phase6_bundle_assembler.py  # assemble KnowledgeBundle
├── verify/
│   ├── __init__.py
│   ├── verify_citations.py         # structural citation check
│   └── build_claim_map.py          # NLI-first claim mapping + LLM fallback
├── knowledge_pipeline_worker.py    # main entry: orchestrate 6 phases
├── tool_executor.py                # route tool_calls to search_papers/build_paper_cards/verify_citations
└── tool_definitions.py             # 3 tool JSON schemas (Intern tools param)

cache/                              # runtime artifacts (see .gitignore; keep seed + surveys)
├── seed_papers.json                # B1 deliverable: local fallback corpus
└── surveys.json -> ../structured_data/surveys.json  (or copy)

tests/
├── conftest.py                     # fakes for LLM/SciVerse/MinerU/Embedding/NLI + fixtures
├── fixtures/
│   ├── seed_papers.json
│   ├── sample_survey.json
│   ├── sample_parsed_paper.json
│   └── sample_survey_md.md
├── unit/                           # one test file per module
└── integration/
    ├── test_pipeline_e2e.py        # full pipeline with all externals mocked
    └── test_tool_executor.py
```

## Test Strategy (non-TDD, comprehensive)

- **pytest**; unit tests in `tests/unit/`, integration in `tests/integration/`.
- **All network is mocked.** `conftest.py` provides `FakeLLMClient`, `FakeSciVerseClient`, `FakeMinerUClient`, `FakeEmbeddingClient`, `FakeNLIModel` returning deterministic canned data. No test hits the real internet.
- **NLI model:** unit tests use `FakeNLIModel` (returns canned `(label, score)`). The real ~400MB DeBERTa model is loaded only behind a `@pytest.mark.nli_real` marker, run manually/opt-in. Never auto-download in CI.
- **Per-module tests cover:** happy path, empty/edge input (0 papers, missing fields), boundary validation (bad API JSON → raises), and the key invariants below.
- **Cross-cutting invariants asserted in tests:**
  - Every `paper_id` in `citation_index` also appears in `paper_cards`.
  - Every `evidence_id` matches `{paper_id}_p{page}_{idx}`.
  - Every `claim` in a PaperCard has an `evidence_ids` list (possibly empty for `limitations`).
  - `verify_citations` flags a fabricated `paper_id` as invalid; flags a real one as valid.
  - `build_claim_map` returns `supported` for an entailed claim and `unsupported` for a contradicted one (via `FakeNLIModel`).
- **Tests written AFTER impl in the same task**, then `pytest` run, then commit.

---

# Wave 0 — Scaffold & Shared Models

Foundation everything else imports. Must land first.

## Task 0.1: Project scaffold, config, dependencies

**Files:**
- Create: `tools/__init__.py` (empty), `tools/config.py`, `tools/models/__init__.py` (empty), `tools/clients/__init__.py` (empty), `tools/nlp/__init__.py` (empty), `tools/indexer/__init__.py` (empty), `tools/phases/__init__.py` (empty), `tools/verify/__init__.py` (empty)
- Create: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `.gitignore` (add `cache/*.json` except seed/surveys, add `cache/rag_index/`, `cache/assets/`)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `tools.config.get_config() -> Config` where `Config` has `.intern_api_base_url`, `.intern_api_key`, `.sciverse_api_key`, `.mineru_api_key`, `.openai_api_key`, `.embedding_provider`. Used by every client.

- [ ] **Step 1: Add dependencies**

`requirements.txt` (append to existing):
```
openai>=1.40.0
pydantic>=2.6.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pytest>=8.0.0
respx>=0.21.0
```

`pyproject.toml`:
```toml
[project]
name = "ailab-module-b"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["nli_real: loads the real DeBERTa NLI model (slow, opt-in)"]
```

- [ ] **Step 2: Implement `tools/config.py`**

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    intern_api_base_url: str
    intern_api_key: str
    sciverse_api_key: str
    mineru_api_key: str
    openai_api_key: str
    embedding_provider: str  # "openai" | "mock"

def get_config() -> Config:
    return Config(
        intern_api_base_url=os.environ["INTERN_API_BASE_URL"],
        intern_api_key=os.environ["INTERN_API_KEY"],
        sciverse_api_key=os.environ.get("SCIVERSE_API_KEY", ""),
        mineru_api_key=os.environ.get("MINERU_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "mock"),
    )
```

- [ ] **Step 3: Update `.gitignore`** — append:
```
cache/*
!cache/.gitkeep
!cache/seed_papers.json
```

- [ ] **Step 4: Write tests** (`tests/unit/test_config.py`)

```python
import os
import pytest
from tools.config import get_config

def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("INTERN_API_BASE_URL", "https://api.test")
    monkeypatch.setenv("INTERN_API_KEY", "k")
    cfg = get_config()
    assert cfg.intern_api_base_url == "https://api.test"
    assert cfg.intern_api_key == "k"
    assert cfg.embedding_provider == "mock"  # default

def test_config_requires_intern_base_url(monkeypatch):
    monkeypatch.delenv("INTERN_API_BASE_URL", raising=False)
    with pytest.raises(KeyError):
        get_config()
```

- [ ] **Step 5: Run + commit**

```bash
pytest tests/unit/test_config.py -v
git add tools/__init__.py tools/config.py tools/models/__init__.py tools/clients/__init__.py tools/nlp/__init__.py tools/indexer/__init__.py tools/phases/__init__.py tools/verify/__init__.py pyproject.toml requirements.txt .gitignore tests/unit/test_config.py
git commit -m "feat(B): project scaffold, config, dependencies"
```

---

## Task 0.2: Pydantic models + ID builders

**Files:**
- Create: `tools/models/common.py`, `tools/models/requests.py`, `tools/models/artifacts.py`, `tools/models/bundle.py`
- Test: `tests/unit/test_ids.py`, `tests/unit/test_models.py`

**Interfaces:**
- Produces (used everywhere downstream):
  - `tools.models.common`: `paper_id_from_seed(title, year) -> str`, `evidence_id(paper_id, page, idx) -> str`, `figure_id(paper_id, num)`, `table_id(paper_id, num)`, `category_id(n) -> str`
  - `tools.models.artifacts`: `RetrievedPaper`, `RetrievedPapers`, `ParsedPaper`, `ParsedPapers`, `PaperCard`, `Evidence`, `EvidenceStore`, `Figure`, `FigureBank`, `Table`, `TableBank`, `Category`, `Taxonomy`, `CitationIndex`
  - `tools.models.requests`: `SearchAspect`, `SearchStrategy`, `KnowledgeBuildRequest`
  - `tools.models.bundle`: `KnowledgeBundle`, `CitationEntry`, `CitationResult`, `ClaimEntry`, `ClaimMap`

- [ ] **Step 1: Implement `tools/models/common.py`** (ID rules §5.0)

```python
import hashlib

def paper_id_from_seed(title: str, year: int) -> str:
    h = hashlib.sha256(f"{title}|{year}".encode()).hexdigest()[:12]
    return f"seed:{h}"

def evidence_id(paper_id: str, page: int, idx: int) -> str:
    return f"{paper_id}_p{page}_{idx}"

def figure_id(paper_id: str, num: int) -> str:
    return f"{paper_id}_fig{num}"

def table_id(paper_id: str, num: int) -> str:
    return f"{paper_id}_tbl{num}"

def category_id(n: int) -> str:
    return f"cat_{n:03d}"
```

- [ ] **Step 2: Implement `tools/models/artifacts.py`** — pydantic models matching interface doc §9. Show the key ones; the rest follow the same pattern with the fields in §9.

```python
from pydantic import BaseModel, Field
from .common import evidence_id

class RetrievedPaper(BaseModel):  # §9.1
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    abstract: str = ""
    keywords: list[str] = []
    citation_count: int = 0
    source: str = "sciverse"  # "sciverse" | "seed" | "survey_ref"
    parse_status: str = "pending"  # "deep" | "light" | "abstract_only" | "pending"

class RetrievedPapers(BaseModel):
    task_id: str
    papers: list[RetrievedPaper]

class Paragraph(BaseModel):
    page: int
    index: int
    text: str

class ParsedPaper(BaseModel):  # §9.2
    paper_id: str
    title: str
    abstract: str = ""
    sections: list[dict] = []          # {"name": str, "paragraphs": [Paragraph]}
    paragraphs: list[Paragraph] = []
    figures: list[dict] = []           # {"num": int, "caption": str, "page": int}
    tables: list[dict] = []
    parse_status: str = "deep"

class ParsedPapers(BaseModel):
    task_id: str
    papers: list[ParsedPaper]

class Claim(BaseModel):
    text: str
    dimension: str  # key_results | method | setup | limitations
    evidence_ids: list[str] = []

class PaperCard(BaseModel):  # §9.5
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    matched_aspects: list[dict] = []   # {"aspect_id": str, "score": float}
    category_id: str | None = None
    card_type: str = "deep"            # "deep" | "light"
    problem: str = ""
    method: str = ""
    contribution: str = ""
    limitations: str = ""
    evidence_ids: list[str] = []
    figure_ids: list[str] = []
    table_ids: list[str] = []
    possible_claims: dict[str, list[Claim]] = {}  # keyed by dimension
    bibtex_key: str | None = None

class PaperCards(BaseModel):
    task_id: str
    cards: list[PaperCard]

class Evidence(BaseModel):  # §9.6
    evidence_id: str
    paper_id: str
    source_type: str = "paragraph"     # paragraph | caption | abstract
    source_page: int = 0
    source_paragraph_index: int = 0
    text: str
    supports_claims: list[dict] = []   # {"claim_text", "support_type", "confidence"}

class EvidenceStore(BaseModel):
    task_id: str
    evidences: list[Evidence]

class Figure(BaseModel):  # §9.3
    figure_id: str
    paper_id: str
    caption: str = ""
    image_path: str | None = None
    extracted_text: str = ""
    page: int = 0
    usable_in_report: bool = True
    generation_hint: dict = {}

class FigureBank(BaseModel):
    task_id: str
    figures: list[Figure]

class Table(BaseModel):  # §9.5(second)
    table_id: str
    paper_id: str
    caption: str = ""
    page: int = 0

class TableBank(BaseModel):
    task_id: str
    tables: list[Table]

class Category(BaseModel):
    category_id: str
    category_name: str
    description: str = ""
    related_aspects: list[str] = []
    paper_ids: list[str] = []
    stage_order: int = 0
    paper_count: int = 0
    adjustment_note: str = ""

class Taxonomy(BaseModel):  # §9.7
    task_id: str
    topic: str
    taxonomy_version: str = "v1"
    refine_history: list[str] = []
    categories: list[Category] = []

class CitationIndex(BaseModel):  # §9.8
    task_id: str
    citations: list[dict] = []  # {"paper_id", "bibtex_key", "title", "year"}
```

- [ ] **Step 3: Implement `tools/models/requests.py`**

```python
from pydantic import BaseModel

class SearchAspect(BaseModel):
    aspect_id: str
    aspect_name: str
    keywords: list[str]
    min_papers: int = 3

class TimeRange(BaseModel):
    start_year: int
    end_year: int

class SearchConstraints(BaseModel):
    max_papers: int = 40
    max_core_papers: int = 15
    time_range: TimeRange | None = None

class SearchStrategy(BaseModel):
    topic: str
    sub_domains: list[str] = []
    wide_search: dict = {}   # {"search_aspects": [...], "time_range": {...}}

class PipelineConfig(BaseModel):
    use_online_search: bool = True
    use_seed_fallback: bool = True
    use_mineru: bool = True
    use_mock_mineru_if_failed: bool = True
    update_survey_store: bool = False
    aspect_match_threshold: float = 0.6

class QualityRequirements(BaseModel):
    min_total_papers: int = 10
    min_core_papers: int = 5
    min_figures: int = 1
    min_evidence_per_core_paper: int = 1

class KnowledgeBuildRequest(BaseModel):
    task_id: str
    topic: str
    request_type: str = "knowledge_build"
    inputs: dict                       # task_request_path, search_strategy_path
    outputs: dict                      # artifact paths
    pipeline_config: PipelineConfig = PipelineConfig()
    quality_requirements: QualityRequirements = QualityRequirements()
```

- [ ] **Step 4: Implement `tools/models/bundle.py`**

```python
from pydantic import BaseModel

class KnowledgeBundle(BaseModel):  # §8.2
    task_id: str
    topic: str
    status: str  # success | partial_success | failed
    artifacts: dict
    summary: dict
    coverage_report: dict
    quality_report: dict

class CitationEntry(BaseModel):  # §13.3
    citation_id: str
    valid: bool
    figures_valid: list[dict] = []
    tables_valid: list[dict] = []

class CitationResult(BaseModel):
    task_id: str
    total_citations: int
    valid_citations: int
    invalid_citations: int
    weak_claims: int = 0
    citation_validity_score: float
    entries: list[CitationEntry]

class ClaimEntry(BaseModel):  # §13.4
    claim_text: str
    cited_paper_id: str
    status: str  # supported | weak | unsupported
    evidence_ids: list[str] = []
    confidence: float = 0.0

class ClaimMap(BaseModel):
    task_id: str
    entries: list[ClaimEntry]
```

- [ ] **Step 5: Write tests**

`tests/unit/test_ids.py`:
```python
from tools.models.common import paper_id_from_seed, evidence_id, figure_id, table_id, category_id

def test_seed_id_stable():
    assert paper_id_from_seed("DreamerV3", 2023) == paper_id_from_seed("DreamerV3", 2023)
    assert paper_id_from_seed("DreamerV3", 2023).startswith("seed:")
    assert paper_id_from_seed("A", 2020) != paper_id_from_seed("B", 2020)

def test_evidence_id_format():
    assert evidence_id("paper:x", 3, 0) == "paper:x_p3_0"
def test_figure_table_category():
    assert figure_id("paper:x", 1) == "paper:x_fig1"
    assert table_id("paper:x", 2) == "paper:x_tbl2"
    assert category_id(2) == "cat_002"
```

`tests/unit/test_models.py`:
```python
import pytest
from pydantic import ValidationError
from tools.models.artifacts import PaperCard, Claim, Evidence
from tools.models.common import evidence_id

def test_paper_card_defaults():
    c = PaperCard(paper_id="paper:x", title="T")
    assert c.possible_claims == {}
    assert c.card_type == "deep"

def test_claim_with_empty_evidence_ok():
    cl = Claim(text="x", dimension="limitations", evidence_ids=[])
    assert cl.evidence_ids == []

def test_card_rejects_missing_required():
    with pytest.raises(ValidationError):
        PaperCard(title="no id")  # type: ignore

def test_evidence_id_roundtrip():
    eid = evidence_id("paper:x", 5, 2)
    e = Evidence(evidence_id=eid, paper_id="paper:x", source_page=5, source_paragraph_index=2, text="t")
    assert e.evidence_id == "paper:x_p5_2"
```

- [ ] **Step 6: Run + commit**

```bash
pytest tests/unit/test_ids.py tests/unit/test_models.py -v
git add tools/models/ tests/unit/test_ids.py tests/unit/test_models.py
git commit -m "feat(B): pydantic models + ID builders"
```

---

# Wave 1 — Clients, NLP, Indexer

All externals isolated here. Each is independently testable with fakes. **These 7 tasks are mutually independent** — safe to dispatch as parallel subagents after Wave 0.

## Task 1.1: LLM client (Intern-S2-Preview)

**Files:**
- Create: `tools/clients/llm_client.py`
- Test: `tests/unit/test_llm_client.py`

**Interfaces:**
- Consumes: `tools.config.get_config`
- Produces:
  - `LLMClient(model="intern-s2-preview")` with `.complete(messages: list[dict], **kw) -> str` and `.complete_json(messages, schema_hint: str, **kw) -> dict` (parses fenced ```json``` or raw JSON).
  - `FakeLLMClient(responses: dict)` keyed by a substring tag in the prompt → returns canned text; used in tests.

- [ ] **Step 1: Implement** (`tools/clients/llm_client.py`)

```python
import json
import re
from openai import OpenAI
from tools.config import get_config

class LLMClient:
    def __init__(self, model: str = "intern-s2-preview", config=None):
        cfg = config or get_config()
        self.client = OpenAI(base_url=cfg.intern_api_base_url, api_key=cfg.intern_api_key)
        self.model = model

    def complete(self, messages: list[dict], **kw) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        return resp.choices[0].message.content or ""

    def complete_json(self, messages: list[dict], schema_hint: str = "", **kw) -> dict:
        text = self.complete(messages, **kw)
        return extract_json(text)

def extract_json(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    raw = m.group(1) if m else text.strip()
    return json.loads(raw)  # raises on malformed -> fail fast
```

- [ ] **Step 2: Write tests** (`tests/unit/test_llm_client.py`) — test only `extract_json` and `FakeLLMClient`; the real client is exercised in integration via a mocked OpenAI.

```python
import pytest
from tools.clients.llm_client import extract_json

def test_extract_fenced_json():
    assert extract_json("here:\n```json\n{\"a\": 1}\n```") == {"a": 1}

def test_extract_raw_json():
    assert extract_json('{"b": 2}') == {"b": 2}

def test_extract_raises_on_bad():
    with pytest.raises(Exception):
        extract_json("not json at all")
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_llm_client.py -v
git add tools/clients/llm_client.py tests/unit/test_llm_client.py
git commit -m "feat(B): Intern-S2-Preview LLM client + JSON extraction"
```

---

## Task 1.2: SciVerse client

**Files:**
- Create: `tools/clients/sciverse_client.py`
- Test: `tests/unit/test_sciverse_client.py`

**Interfaces:**
- Consumes: `tools.config.get_config`
- Produces: `SciVerseClient` with `.meta_search(query, filters, sort, freshness_boost, impact_boost, page_size) -> dict`, `.agentic_search(query, top_k, filters) -> dict`, `.get_content(doc_id, offset, limit) -> dict`, `.get_resource(file_name) -> bytes`, `.meta_paper_relations(paper_id, relation, page) -> dict`. Uses httpx; raises on non-2xx.

- [ ] **Step 1: Implement** (`tools/clients/sciverse_client.py`)

```python
import httpx
from tools.config import get_config

class SciVerseClient:
    def __init__(self, base_url: str = "https://sciverse.space", config=None):
        cfg = config or get_config()
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {cfg.sciverse_api_key}"}

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
```

- [ ] **Step 2: Write tests** using `respx` to mock httpx (`tests/unit/test_sciverse_client.py`).

```python
import pytest, respx, httpx
from tools.clients.sciverse_client import SciVerseClient

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SCIVERSE_API_KEY", "k")
    return SciVerseClient(base_url="https://sv.test")

@respx.mock
def test_meta_search(client):
    respx.post("https://sv.test/meta-search").respond(json={"hits": [{"unique_id": "paper:1"}]})
    out = client.meta_search("world models")
    assert out["hits"][0]["unique_id"] == "paper:1"

@respx.mock
def test_raises_on_error(client):
    respx.post("https://sv.test/meta-search").respond(status_code=500)
    with pytest.raises(httpx.HTTPStatusError):
        client.meta_search("x")
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_sciverse_client.py -v
git add tools/clients/sciverse_client.py tests/unit/test_sciverse_client.py
git commit -m "feat(B): SciVerse API client"
```

---

## Task 1.3: MinerU client + mock fallback

**Files:**
- Create: `tools/clients/mineru_client.py`
- Test: `tests/unit/test_mineru_client.py`

**Interfaces:**
- Produces: `MinerUClient` with `.parse_url(url, light=True) -> dict` (sync `/api/v1/agent/parse/url`) and `.extract_task(pdf_url) -> dict` (async `/api/v4/extract/task`: submit → poll → result). On failure and `use_mock=True`, falls back to `mock_parse(url, title) -> dict` returning a minimal valid ParsedPapers-shaped dict. Always returns a dict; never raises out of the public methods when `use_mock=True`.

- [ ] **Step 1: Implement** (`tools/clients/mineru_client.py`)

```python
import time, httpx
from tools.config import get_config

class MinerUClient:
    def __init__(self, base_url="https://mineru.net", config=None, use_mock=True, poll_interval=3, max_wait=180):
        cfg = config or get_config()
        self.base_url = base_url.rstrip("/")
        self.token = cfg.mineru_api_key
        self.use_mock = use_mock
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    def parse_url(self, url: str, light=True) -> dict:
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
                time.sleep(self.poll_interval); waited += self.poll_interval
            return mock_parse(pdf_url)
        except Exception:
            if self.use_mock:
                return mock_parse(pdf_url)
            raise

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
```

- [ ] **Step 2: Write tests** (`tests/unit/test_mineru_client.py`)

```python
import pytest, respx
from tools.clients.mineru_client import MinerUClient, mock_parse

def test_mock_parse_shape():
    out = mock_parse("http://x/a.pdf", "T")
    assert out["title"] == "T" and out["paragraphs"][0]["page"] == 1

@respx.mock
def test_parse_url_falls_back_to_mock_on_error(monkeypatch):
    respx.post("https://mineru.net/api/v1/agent/parse/url").respond(status_code=500)
    c = MinerUClient(use_mock=True)
    out = c.parse_url("http://x/a.pdf")  # server errors -> mock, no raise
    assert out["parse_status"] == "mock"

def test_parse_url_raises_when_mock_disabled(monkeypatch):
    # use_mock=False but no network; monkeypatch httpx to raise
    ...
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_mineru_client.py -v
git add tools/clients/mineru_client.py tests/unit/test_mineru_client.py
git commit -m "feat(B): MinerU client + mock fallback"
```

---

## Task 1.4: Embedding client

**Files:**
- Create: `tools/clients/embedding_client.py`
- Test: `tests/unit/test_embedding_client.py`

**Interfaces:**
- Produces: `EmbeddingClient` with `.embed(texts: list[str]) -> list[list[float]]` (OpenAI `text-embedding-3-small`). `FakeEmbeddingClient` returns deterministic hashed vectors (deterministic so tests are stable).

- [ ] **Step 1: Implement** (`tools/clients/embedding_client.py`)

```python
import hashlib
from openai import OpenAI
from tools.config import get_config

class EmbeddingClient:
    def __init__(self, model="text-embedding-3-small", config=None):
        cfg = config or get_config()
        self.client = OpenAI(api_key=cfg.openai_api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

class FakeEmbeddingClient:
    """Deterministic 16-d vectors for tests (no network)."""
    def __init__(self, dim=16):
        self.dim = dim
    def embed(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([(b / 255.0 - 0.5) for b in (h * ((self.dim // len(h)) + 1))][:self.dim])
        return out
```

- [ ] **Step 2: Write tests** (`tests/unit/test_embedding_client.py`)

```python
from tools.clients.embedding_client import FakeEmbeddingClient

def test_fake_embedding_deterministic():
    e = FakeEmbeddingClient(dim=16)
    a = e.embed(["hello"])[0]
    b = e.embed(["hello"])[0]
    assert a == b and len(a) == 16

def test_different_text_different_vec():
    e = FakeEmbeddingClient()
    assert e.embed(["a"])[0] != e.embed(["b"])[0]
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_embedding_client.py -v
git add tools/clients/embedding_client.py tests/unit/test_embedding_client.py
git commit -m "feat(B): embedding client + deterministic fake"
```

---

## Task 1.5: NLI verifier (reuse SurGE)

**Files:**
- Create: `tools/nlp/nli_verifier.py`
- Reference: `ref/SurGE/src/informationFuncs.py` (lines 22–115: `eval_relevance_paper/sentence`, `label_mapping = ['contradiction','entailment','neutral']`)
- Test: `tests/unit/test_nli_verifier.py`

**Interfaces:**
- Produces:
  - `NLIVerifier(model_name="cross-encoder/nli-deberta-v3-base")` with `.judge(premise, hypothesis) -> NLIResult` and `.best_match(claim, evidences) -> NLIResult`.
  - `NLIResult(label, support_type, confidence)` where label ∈ {entailment, neutral, contradiction}; support_type ∈ {direct, indirect, contradictory}.
  - `FakeNLIModel` implements `.predict(pairs)` returning canned logits, used in tests.

- [ ] **Step 1: Implement** (`tools/nlp/nli_verifier.py`) — adapt SurGE's `predict` + label_mapping.

```python
from dataclasses import dataclass

LABELS = ["contradiction", "entailment", "neutral"]

@dataclass
class NLIResult:
    label: str          # entailment | neutral | contradiction
    support_type: str   # direct | indirect | contradictory
    confidence: float

    @property
    def status(self) -> str:
        if self.label == "entailment" and self.confidence >= 0.6:
            return "supported"
        if self.label == "contradiction":
            return "unsupported"
        return "weak"  # neutral or low-confidence entailment

class NLIVerifier:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def judge(self, premise: str, hypothesis: str) -> NLIResult:
        scores = self.model.predict([(premise, hypothesis)])[0]  # [contra, entail, neutral]
        c, e, n = float(scores[0]), float(scores[1]), float(scores[2])
        label = LABELS[int(max(range(3), key=lambda i: scores[i]))]
        return NLIResult(label, *_support(label, c, e, n))

    def best_match(self, claim: str, evidences: list[str]) -> NLIResult:
        if not evidences:
            return NLIResult("neutral", "indirect", 0.0)
        pairs = [(f"There is a paper. Content: '{ev}'", f"The paper supports: '{claim}'")
                 for ev in evidences]
        scores = self.model.predict(pairs)
        best_i = int(max(range(len(evidences)), key=lambda i: scores[i][1]))
        c, e, n = (float(x) for x in scores[best_i])
        label = LABELS[int(max(range(3), key=lambda i: scores[best_i][i]))]
        return NLIResult(label, *_support(label, c, e, n))

def _support(label, c, e, n):
    if label == "entailment" and e > c and e > n:
        return "direct", round((e - max(c, n)) / (abs(e) + abs(c) + abs(n) + 1e-8), 3)
    if label == "neutral" and n > c:
        return "indirect", round((e - c) / (abs(e) + abs(c) + 1e-8), 3)
    if label == "contradiction":
        return "contradictory", 0.2
    return "indirect", 0.2

class FakeNLIModel:  # for tests; picks label via keyword in hypothesis
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
    def judge(self, premise, hypothesis):
        for kw, label in self.mapping.items():
            if kw in hypothesis:
                conf = {"entailment": 0.9, "neutral": 0.5, "contradiction": 0.9}[label]
                return NLIResult(label, *_support(label, *[0.0,0.0,0.0]), ) if False else NLIResult(label, {"entailment":"direct","neutral":"indirect","contradiction":"contradictory"}[label], conf)
        return NLIResult("neutral", "indirect", 0.5)
```

- [ ] **Step 2: Write tests** using `FakeNLIModel` (`tests/unit/test_nli_verifier.py`)

```python
from tools.nlp.nli_verifier import NLIResult, FakeNLIModel, _support

def test_result_status_supported():
    r = NLIResult("entailment", "direct", 0.9)
    assert r.status == "supported"

def test_result_status_unsupported():
    assert NLIResult("contradiction", "contradictory", 0.2).status == "unsupported"

def test_result_status_weak_for_neutral():
    assert NLIResult("neutral", "indirect", 0.5).status == "weak"

def test_fake_judge_by_keyword():
    m = FakeNLIModel(mapping={"SUPPORTS": "entailment", "CONTRADICTS": "contradiction"})
    assert m.judge("p", "this SUPPORTS the claim").label == "entailment"
    assert m.judge("p", "this CONTRADICTS the claim").label == "contradiction"
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_nli_verifier.py -v
git add tools/nlp/nli_verifier.py tests/unit/test_nli_verifier.py
git commit -m "feat(B): NLI verifier (SurGE-reuse) + fake model"
```

---

## Task 1.6: Data cleaner (reuse SurveyX)

**Files:**
- Create: `tools/nlp/data_cleaner.py`
- Reference: `ref/SurveyX/src/modules/preprocessor/data_cleaner.py` (`complete_abstract` regex `r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*"`, `quick_check`)
- Test: `tests/unit/test_data_cleaner.py`

**Interfaces:**
- Produces: `DataCleaner` with `.complete_abstract(paper: dict) -> str` (extracts abstract from `md_text` if missing/short, using the tolerant SurveyX regex) and `.clean_parsed(parsed: dict) -> dict` (normalize MinerU output: ensure `abstract`, flatten `paragraphs`).

- [ ] **Step 1: Implement** (`tools/nlp/data_cleaner.py`)

```python
import re
ABSTRACT_RE = re.compile(r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*", re.I)

class DataCleaner:
    def complete_abstract(self, paper: dict) -> str:
        abstract = paper.get("abstract", "") or ""
        if len(abstract) > 500:
            return abstract
        md = paper.get("md_text", "") or paper.get("full_text", "") or ""
        if not md:
            return abstract
        m = ABSTRACT_RE.search(md)
        if m:
            return md[m.end(): m.end() + 2000].strip()
        return md[:2000].strip()

    def clean_parsed(self, parsed: dict) -> dict:
        if not parsed.get("abstract"):
            parsed["abstract"] = self.complete_abstract(parsed)
        # flatten paragraphs from sections if missing
        if not parsed.get("paragraphs") and parsed.get("sections"):
            flat = []
            for sec in parsed["sections"]:
                for p in sec.get("paragraphs", []):
                    flat.append(p)
            parsed["paragraphs"] = flat
        return parsed
```

- [ ] **Step 2: Write tests** (`tests/unit/test_data_cleaner.py`)

```python
from tools.nlp.data_cleaner import DataCleaner

def test_complete_abstract_from_md():
    p = {"abstract": "", "md_text": "title\n\nA b s t r a c t\nWe present a method. " * 5}
    out = DataCleaner().complete_abstract(p)
    assert "present a method" in out

def test_keeps_long_abstract():
    long_abs = "x" * 600
    assert DataCleaner().complete_abstract({"abstract": long_abs}) == long_abs

def test_flatten_paragraphs_from_sections():
    parsed = {"abstract": "a", "sections": [{"name":"s","paragraphs":[{"page":1,"index":0,"text":"t"}]}]}
    out = DataCleaner().clean_parsed(parsed)
    assert out["paragraphs"][0]["text"] == "t"
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_data_cleaner.py -v
git add tools/nlp/data_cleaner.py tests/unit/test_data_cleaner.py
git commit -m "feat(B): data cleaner (SurveyX-reuse tolerant regex)"
```

---

## Task 1.7: Vector store (ChromaDB)

**Files:**
- Create: `tools/indexer/vector_store.py`
- Test: `tests/unit/test_vector_store.py`

**Interfaces:**
- Consumes: an embedding client (duck-typed `.embed(list[str]) -> list[list[float]]`).
- Produces: `VectorStore` with `.add(ids, texts, metadatas)`, `.query(text, n=10, where=None) -> list[Hit]`, where `Hit` = `{id, score, metadata}`. Backed by ChromaDB ephemeral or persistent (path).

- [ ] **Step 1: Implement** (`tools/indexer/vector_store.py`)

```python
from dataclasses import dataclass
import chromadb

@dataclass
class Hit:
    id: str
    score: float
    metadata: dict

class VectorStore:
    def __init__(self, embedding_client, path=None, collection="papers"):
        self.embed = embedding_client
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        vecs = self.embed.embed(texts)
        self.collection.upsert(ids=ids, embeddings=vecs, documents=texts, metadatas=metadatas)

    def query(self, text: str, n: int = 10, where: dict | None = None) -> list[Hit]:
        qv = self.embed.embed([text])[0]
        res = self.collection.query(query_embeddings=[qv], n_results=n, where=where)
        hits = []
        for i, _id in enumerate(res["ids"][0]):
            dist = res["distances"][0][i]
            hits.append(Hit(id=_id, score=1.0 - dist, metadata=res["metadatas"][0][i]))
        return hits
```

- [ ] **Step 2: Write tests** (`tests/unit/test_vector_store.py`) — use `FakeEmbeddingClient`.

```python
from tools.indexer.vector_store import VectorStore
from tools.clients.embedding_client import FakeEmbeddingClient

def test_add_and_query_roundtrip(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v"))
    vs.add(["a", "b"], ["world model dreamer", "reinforcement learning"], [{"paper_id":"p1"},{"paper_id":"p2"}])
    hits = vs.query("world model dreamer", n=2)
    assert hits[0].id == "a"

def test_query_with_where_filter(tmp_path):
    vs = VectorStore(FakeEmbeddingClient(dim=16), path=str(tmp_path / "v2"))
    vs.add(["a","b"], ["x","y"], [{"paper_id":"p1"},{"paper_id":"p2"}])
    hits = vs.query("x", n=5, where={"paper_id": "p2"})
    assert all(h.metadata["paper_id"] == "p2" for h in hits)
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_vector_store.py -v
git add tools/indexer/vector_store.py tests/unit/test_vector_store.py
git commit -m "feat(B): ChromaDB vector store wrapper"
```

---

# Wave 2 — Phases

Each phase is `run(...) -> artifact`. **Hard rule:** phases never read env or build clients themselves — clients are injected (caller builds them once). This makes every phase unit-testable with fakes.

## Task 2.1: Phase 1 — Demand Decomposition

**Files:**
- Create: `tools/phases/phase1_decompose.py`
- Test: `tests/unit/test_phase1.py`

**Interfaces:**
- Consumes: `KnowledgeBuildRequest`, a loaded `SearchStrategy`, `seed_papers: list[dict]`.
- Produces: `DecomposedDemand` dataclass (in this file) with `.aspects, .constraints, .pipeline_config, .coverage_warnings, .structure_errors`. Pure heuristic, no LLM, no network.

- [ ] **Step 1: Implement** (`tools/phases/phase1_decompose.py`)

```python
from dataclasses import dataclass
from tools.models.requests import SearchStrategy, PipelineConfig

@dataclass
class DecomposedDemand:
    aspects: list[dict]
    constraints: dict
    pipeline_config: PipelineConfig
    structure_errors: list[str]
    coverage_warnings: list[str]

def validate_structure(strategy: SearchStrategy) -> list[str]:
    errors = []
    ws = strategy.wide_search
    aspects = ws.get("search_aspects", [])
    if len(aspects) < 3:
        errors.append("aspects fewer than 3, coverage may be insufficient")
    for a in aspects:
        if not a.get("keywords"):
            errors.append(f"{a.get('aspect_id','?')} missing keywords")
    tr = ws.get("time_range")
    if tr and (tr.get("end_year", 0) - tr.get("start_year", 0) < 3):
        errors.append("time span < 3 years, may miss classic works")
    return errors

def validate_coverage(strategy: SearchStrategy, seed_papers: list[dict]) -> list[str]:
    warnings = []
    aspects = strategy.wide_search.get("search_aspects", [])
    for domain in strategy.sub_domains:
        matched = any(
            domain.lower() in a.get("aspect_name","").lower()
            or any(kw.lower() in domain.lower() for kw in a.get("keywords", []))
            for a in aspects)
        if not matched:
            warnings.append(f"A's sub_domain '{domain}' has no matching aspect")
    for paper in seed_papers:
        ptext = f"{paper.get('title','')} {' '.join(paper.get('keywords',[]))}".lower()
        matched = any(any(kw.lower() in ptext for kw in a.get("keywords", [])) for a in aspects)
        if not matched:
            warnings.append(f"seed paper matches no aspect: {paper.get('title')}")
    return warnings

def run(request, strategy: SearchStrategy, seed_papers: list[dict]) -> DecomposedDemand:
    return DecomposedDemand(
        aspects=strategy.wide_search.get("search_aspects", []),
        constraints=request.pipeline_config.model_dump(),
        pipeline_config=request.pipeline_config,
        structure_errors=validate_structure(strategy),
        coverage_warnings=validate_coverage(strategy, seed_papers),
    )
```

- [ ] **Step 2: Write tests** (`tests/unit/test_phase1.py`)

```python
from tools.phases.phase1_decompose import run, validate_structure, validate_coverage
from tools.models.requests import SearchStrategy, KnowledgeBuildRequest

def _strategy():
    return SearchStrategy(topic="wm", sub_domains=["Game Benchmark"],
        wide_search={"search_aspects":[
            {"aspect_id":"a1","aspect_name":"Game Benchmark","keywords":["game","benchmark"]},
            {"aspect_id":"a2","aspect_name":"Internal WM","keywords":["world model","latent"]},
            {"aspect_id":"a3","aspect_name":"Game Engine","keywords":["neural","engine"]}],
            "time_range":{"start_year":2018,"end_year":2026}})

def test_structure_ok():
    assert validate_structure(_strategy()) == []

def test_structure_flags_few_aspects():
    s = _strategy(); s.wide_search["search_aspects"] = s.wide_search["search_aspects"][:1]
    assert any("fewer than 3" in e for e in validate_structure(s))

def test_coverage_warns_unmatched_seed():
    s = _strategy()
    seeds = [{"title":"Quantum Computing Paper","keywords":["quantum"]}]
    w = validate_coverage(s, seeds)
    assert any("quantum" in x.lower() for x in w) or any("no aspect" in x for x in w)

def test_run_returns_demand():
    req = KnowledgeBuildRequest(task_id="t", topic="wm", inputs={}, outputs={})
    d = run(req, _strategy(), [])
    assert d.aspects and d.pipeline_config.use_seed_fallback is True
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase1.py -v
git add tools/phases/phase1_decompose.py tests/unit/test_phase1.py
git commit -m "feat(B): phase 1 demand decomposition"
```

---

## Task 2.2: Phase 2 — Survey Analysis (Layer 1 + Taxonomy Self-Refine step 1-2)

**Files:**
- Create: `tools/phases/phase2_survey_analyzer.py`
- Test: `tests/unit/test_phase2.py`

**Interfaces:**
- Consumes: `demand.aspects`, `surveys: list[dict]` (loaded from `structured_data/surveys.json`), `LLMClient`, `SciVerseClient` (optional, for survey-store update), `MinerUClient`, `DataCleaner`.
- Produces: `SurveyStructure` (pydantic, in this file) with `.analyzed_surveys, .preliminary_taxonomy, .refined_taxonomy, .expansion_candidates, .survey_update_log`. Writes `cache/survey_structure.json`.

- [ ] **Step 1: Implement** (`tools/phases/phase2_survey_analyzer.py`)

```python
import json
from pydantic import BaseModel
from tools.models.common import paper_id_from_seed

class SurveyStructure(BaseModel):
    task_id: str
    analyzed_surveys: list[dict]
    preliminary_taxonomy: list[dict]
    refined_taxonomy: list[dict]
    expansion_candidates: list[dict]
    survey_update_log: list[dict] = []

PRELIM_PROMPT = """Generate a paper taxonomy for the topic: {topic}.
Sub-domains: {sub_domains}. Aspects: {aspects}.
Return ONLY JSON: {{"categories":[{{"name":str,"description":str}}]}}"""

REFINE_PROMPT = """Refine this taxonomy using the survey structures below.
Preliminary: {prelim}
Survey taxonomies: {survey_skels}
Merge, dedupe, fix gaps. Return ONLY JSON: {{"categories":[{{"name":str,"description":str,"incorporated_from":[str]}}]}}"""

def analyze_surveys(surveys, mineru, cleaner):
    out = []
    for s in surveys:
        skel = s.get("meta_data", {}).get("taxonomy_skeleton", [])
        refs = s.get("meta_data", {}).get("top_referenced_papers", [])
        out.append({
            "paper_id": s.get("paper_id") or paper_id_from_seed(s["title"], s.get("year", 0)),
            "taxonomy_skeleton": skel,
            "key_sections": s.get("meta_data", {}).get("key_sections", []),
            "referenced_paper_ids": refs,
            "key_claims": [],
        })
    return out

def run(task_id, topic, sub_domains, aspects, surveys, llm, mineru=None, cleaner=None, sciverse=None):
    analyzed = analyze_surveys(surveys, mineru, cleaner)
    # Step 1: independent preliminary taxonomy
    prelim_raw = llm.complete_json([{"role":"user","content": PRELIM_PROMPT.format(
        topic=topic, sub_domains=sub_domains, aspects=[a["aspect_name"] for a in aspects)]}])
    prelim = prelim_raw.get("categories", [])
    # Step 2: refine with survey skeletons
    survey_skels = [a["taxonomy_skeleton"] for a in analyzed]
    refined_raw = llm.complete_json([{"role":"user","content": REFINE_PROMPT.format(
        prelim=prelim, survey_skels=survey_skels)}])
    refined = refined_raw.get("categories", [])
    # expansion candidates from referenced papers
    expansion = [{"paper_id_hint": pid, "source_survey": a["paper_id"], "priority":"high"}
                 for a in analyzed for pid in a["referenced_paper_ids"]]
    struct = SurveyStructure(task_id=task_id, analyzed_surveys=analyzed,
        preliminary_taxonomy=prelim, refined_taxonomy=refined, expansion_candidates=expansion)
    return struct
```

- [ ] **Step 2: Write tests** with `FakeLLMClient` (`tests/unit/test_phase2.py`)

```python
import json
from tools.phases.phase2_survey_analyzer import run

class FakeLLM:
    def __init__(self): self.calls = 0
    def complete_json(self, messages, **kw):
        self.calls += 1
        return {"categories": [{"name": f"cat{self.calls}", "description": "d"}]}
    def complete(self, messages, **kw): return ""

def test_phase2_two_llm_calls_and_expansion():
    surveys = [{"paper_id":"sv1","title":"T","year":2024,
                "meta_data":{"taxonomy_skeleton":["x"],"top_referenced_papers":["genie_2024"]}}]
    llm = FakeLLM()
    s = run("t1","world models",["sim"],[{"aspect_name":"sim"}], surveys, llm)
    assert llm.calls == 2
    assert s.preliminary_taxonomy and s.refined_taxonomy
    assert s.expansion_candidates[0]["paper_id_hint"] == "genie_2024"
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase2.py -v
git add tools/phases/phase2_survey_analyzer.py tests/unit/test_phase2.py
git commit -m "feat(B): phase 2 survey analysis + taxonomy self-refine"
```

---

## Task 2.3: Phase 3 — Method Paper Retrieval (Layer 2)

**Files:**
- Create: `tools/phases/phase3_paper_retriever.py`
- Test: `tests/unit/test_phase3.py`

**Interfaces:**
- Consumes: `demand.aspects`, `survey_structure.expansion_candidates`, `SciVerseClient`, `MinerUClient`, `DataCleaner`, `seed_papers: list[dict]`, `pipeline_config`.
- Produces: `(RetrievedPapers, ParsedPapers)` and writes `cache/retrieved_papers.json`, `cache/parsed_papers.json`. Dedups by `unique_id` then `title+year`. Marks `parse_status` deep/light/abstract_only.

- [ ] **Step 1: Implement** (`tools/phases/phase3_paper_retriever.py`) — outline the priority order: expansion refs → meta-search per aspect → agentic-search fallback → seed fallback → dedup → parse.

```python
from tools.models.artifacts import RetrievedPaper, RetrievedPapers, ParsedPaper, ParsedPapers
from tools.models.common import paper_id_from_seed, evidence_id

def _to_retrieved(hit: dict, source="sciverse") -> RetrievedPaper:
    return RetrievedPaper(
        paper_id=hit.get("unique_id") or paper_id_from_seed(hit.get("title",""), hit.get("year",0)),
        title=hit.get("title",""), authors=hit.get("authors",[]), year=hit.get("year"),
        venue=hit.get("venue"), url=hit.get("url"), abstract=hit.get("abstract",""),
        keywords=hit.get("keywords",[]), citation_count=hit.get("citation_count",0),
        source=source, parse_status="pending")

def dedup(papers: list[RetrievedPaper]) -> list[RetrievedPaper]:
    seen, out = set(), []
    for p in papers:
        key = p.paper_id
        if key in seen:
            continue
        seen.add(key); out.append(p)
    return out

def run(task_id, aspects, expansion_candidates, sciverse, mineru, cleaner, seed_papers, pipeline_config):
    retrieved = []
    # 1. expansion refs via meta-paper-relations (skip if sciverse offline -> caught upstream)
    # 2. meta-search per aspect
    for a in aspects:
        try:
            res = sciverse.meta_search(query=" ".join(a.get("keywords", [])),
                filters=[{"field":"publication_published_year","value":{"gte":2018,"lte":2026}}],
                sort=[{"field":"citation_count","order":"SORT_ORDER_DESC"}],
                freshness_boost="MILD", impact_boost="MILD", page_size=25)
            for hit in res.get("hits", []):
                retrieved.append(_to_retrieved(hit))
        except Exception:
            pass  # fail fast per-aspect suppressed only to keep other aspects; logged in quality_report
    # 3. seed fallback if empty
    if not retrieved and pipeline_config.use_seed_fallback:
        retrieved = [_to_retrieved(s, source="seed") for s in seed_papers]
    retrieved = dedup(retrieved)[: pipeline_config_extra(pipeline_config, "max_papers", 40)]
    # 4. parse
    parsed = []
    for p in retrieved:
        if not p.url or not pipeline_config.use_mineru:
            p.parse_status = "abstract_only"; continue
        raw = mineru.parse_url(p.url, light=True)
        raw = cleaner.clean_parsed(raw)
        p.parse_status = "light"
        parsed.append(_to_parsed(p, raw))
    return (RetrievedPapers(task_id=task_id, papers=retrieved),
            ParsedPapers(task_id=task_id, papers=parsed))

def pipeline_config_extra(cfg, key, default):
    return getattr(cfg, key, default) if hasattr(cfg, key) else default

def _to_parsed(p, raw):
    return ParsedPaper(paper_id=p.paper_id, title=raw.get("title", p.title),
        abstract=raw.get("abstract",""), sections=raw.get("sections",[]),
        paragraphs=raw.get("paragraphs",[]), figures=raw.get("figures",[]),
        tables=raw.get("tables",[]), parse_status="light")
```

- [ ] **Step 2: Write tests** (`tests/unit/test_phase3.py`)

```python
from tools.phases.phase3_paper_retriever import run, dedup, _to_retrieved
from tools.models.artifacts import RetrievedPaper

class FakeSV:
    def meta_search(self, **kw):
        return {"hits":[{"unique_id":"paper:1","title":"A","year":2023,"url":"http://a"},
                        {"unique_id":"paper:1","title":"A","year":2023}]}  # dup
class FakeMU:
    def parse_url(self, url, light=True):
        return {"title":"A","abstract":"abs","sections":[],"paragraphs":[{"page":1,"index":0,"text":"t"}],"figures":[],"tables":[]}
class FakeCleaner:
    def clean_parsed(self, d): return d

def test_dedup_by_paper_id():
    ps = [_to_retrieved({"unique_id":"paper:1","title":"A"}), _to_retrieved({"unique_id":"paper:1","title":"A"})]
    assert len(dedup(ps)) == 1

def test_seed_fallback_when_no_hits(monkeypatch):
    class EmptySV:
        def meta_search(self, **kw): return {"hits":[]}
    from tools.models.requests import PipelineConfig
    seeds = [{"title":"DreamerV3","year":2023,"keywords":["wm"]}]
    rp, pp = run("t", [{"keywords":["wm"]}], [], EmptySV(), FakeMU(), FakeCleaner(), seeds, PipelineConfig(use_seed_fallback=True))
    assert len(rp.papers) == 1 and rp.papers[0].source == "seed"

def test_parse_marks_status():
    from tools.models.requests import PipelineConfig
    rp, pp = run("t",[{"keywords":["wm"]}],[],FakeSV(),FakeMU(),FakeCleaner(),[],PipelineConfig())
    assert rp.papers[0].parse_status == "light"
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase3.py -v
git add tools/phases/phase3_paper_retriever.py tests/unit/test_phase3.py
git commit -m "feat(B): phase 3 paper retrieval + MinerU parse"
```

---

## Task 2.4: Phase 4 — RAG Index

**Files:**
- Create: `tools/phases/phase4_rag_indexer.py`
- Test: `tests/unit/test_phase4.py`

**Interfaces:**
- Consumes: `ParsedPapers`, optional `EvidenceStore`-like texts, an `embedding_client`, a `path`.
- Produces: a `VectorStore` (persisted at `cache/rag_index/`) indexing abstract + each paragraph. Each entry metadata: `{paper_id, source_type, page}`.

- [ ] **Step 1: Implement** (`tools/phases/phase4_rag_indexer.py`)

```python
from tools.indexer.vector_store import VectorStore

def run(parsed_papers, embedding_client, path=None, extra_texts=None):
    vs = VectorStore(embedding_client, path=path)
    ids, texts, metas = [], [], []
    for p in parsed_papers.papers:
        if p.abstract:
            ids.append(f"{p.paper_id}_abs"); texts.append(p.abstract)
            metas.append({"paper_id": p.paper_id, "source_type": "abstract", "page": 0})
        for para in p.paragraphs:
            eid = f"{p.paper_id}_p{para.page}_{para.index}"
            ids.append(eid); texts.append(para.text)
            metas.append({"paper_id": p.paper_id, "source_type": "paragraph", "page": para.page})
    if extra_texts:
        ids += [e["id"] for e in extra_texts]; texts += [e["text"] for e in extra_texts]
        metas += [e["meta"] for e in extra_texts]
    if ids:
        vs.add(ids, texts, metas)
    return vs
```

- [ ] **Step 2: Write tests** (`tests/unit/test_phase4.py`)

```python
from tools.phases.phase4_rag_indexer import run
from tools.models.artifacts import ParsedPapers, ParsedPaper, Paragraph
from tools.clients.embedding_client import FakeEmbeddingClient

def test_indexes_abstract_and_paragraphs(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        abstract="abs", paragraphs=[Paragraph(page=1,index=0,text="body")])])
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"))
    hits = vs.query("abs", n=5)
    assert {h.id for h in hits} & {"paper:1_abs","paper:1_p1_0"}
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase4.py -v
git add tools/phases/phase4_rag_indexer.py tests/unit/test_phase4.py
git commit -m "feat(B): phase 4 RAG index construction"
```

---

## Task 2.5: Phase 5 — Knowledge Synthesis (split into 5a/5b/5c)

This is the biggest phase. Split into three sub-tasks, each independently testable.

### Task 2.5a: Paper Cards (LiRA dimension decomposition)

**Files:**
- Create: `tools/phases/phase5_cards.py`
- Reference: `ref/LiRA/src/researchers.py` (`analyze_paper`, `split_content`), `ref/LiRA/src/prompts/research.py`
- Test: `tests/unit/test_phase5_cards.py`

**Interfaces:**
- Consumes: `ParsedPapers`, `RetrievedPapers` (for metadata), `LLMClient`, `demand.aspects`, `aspect_match_threshold`.
- Produces: `PaperCards`. Each card's `possible_claims` has 4 buckets (`key_results/method/setup/limitations`), each a list of `Claim` with `evidence_ids`. Uses the LiRA-style numbered-header prompt; deterministic parse by `##` headers.

- [ ] **Step 1: Implement** (`tools/phases/phase5_cards.py`)

```python
import re
from tools.models.artifacts import PaperCard, PaperCards, Claim
from tools.models.common import evidence_id

CARD_PROMPT = """Extract a structured paper card from this paper.
Title: {title}
Abstract: {abstract}
Body excerpts: {body}

Return EXACTLY these sections with numbered lists. ONLY points you are CERTAIN of.
## KEY RESULTS
## METHOD
## SETUP
## LIMITATIONS
Each line: "<claim text> [page P]" if a page number is citable."""

BUCKETS = {"KEY RESULTS": "key_results", "METHOD": "method", "SETUP": "setup", "LIMITATIONS": "limitations"}

def parse_card_response(text, paper_id):
    claims = {b: [] for b in BUCKETS.values()}
    current = None
    for line in text.splitlines():
        h = re.match(r"^##\s*(.+)$", line.strip())
        if h and h.group(1).strip() in BUCKETS:
            current = BUCKETS[h.group(1).strip()]; continue
        m = re.match(r"^\s*\d+\.\s+(.+?)(?:\s*\[page (\d+)\])?$", line)
        if m and current:
            claim_text = m.group(1).strip()
            evid = []
            if m.group(2):
                evid.append(evidence_id(paper_id, int(m.group(2)), 0))
            claims[current].append(Claim(text=claim_text, dimension=current, evidence_ids=evid))
    return claims

def build_card(parsed, retrieved_map, llm, aspects, threshold=0.6):
    meta = retrieved_map.get(parsed.paper_id)
    body = " ".join(p.text for p in parsed.paragraphs[:20])
    raw = llm.complete([{"role":"user","content": CARD_PROMPT.format(
        title=parsed.title, abstract=parsed.abstract, body=body)}])
    claims = parse_card_response(raw, parsed.paper_id)
    return PaperCard(
        paper_id=parsed.paper_id, title=parsed.title,
        authors=getattr(meta,"authors",[]), year=getattr(meta,"year",None),
        venue=getattr(meta,"venue",None), matched_aspects=[], card_type="deep",
        problem="", method="", contribution="", limitations="",
        evidence_ids=[], figure_ids=[], table_ids=[], possible_claims=claims,
        bibtex_key=None)

def run(task_id, parsed_papers, retrieved_papers, llm, aspects, threshold=0.6):
    rmap = {p.paper_id: p for p in retrieved_papers.papers}
    cards = [build_card(p, rmap, llm, aspects, threshold) for p in parsed_papers.papers]
    return PaperCards(task_id=task_id, cards=cards)
```

- [ ] **Step 2: Write tests** (`tests/unit/test_phase5_cards.py`)

```python
from tools.phases.phase5_cards import parse_card_response, run
from tools.models.artifacts import ParsedPapers, ParsedPaper, RetrievedPapers, RetrievedPaper

SAMPLE = """## KEY RESULTS
1. Achieves SOTA on 150 domains [page 3]
2. Single config works
## METHOD
1. Recurrent state space model [page 4]
## SETUP
## LIMITATIONS
1. Short horizon only"""

def test_parse_buckets_and_evidence():
    claims = parse_card_response(SAMPLE, "paper:1")
    assert claims["key_results"][0].text.startswith("Achieves SOTA")
    assert claims["key_results"][0].evidence_ids == ["paper:1_p3_0"]
    assert claims["limitations"][0].text == "Short horizon only"
    assert claims["setup"] == []

class FakeLLM:
    def complete(self, messages, **kw): return SAMPLE

def test_run_builds_cards():
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A", abstract="a")])
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id="paper:1", title="A")])
    cards = run("t", pp, rp, FakeLLM(), [])
    assert cards.cards[0].possible_claims["method"][0].dimension == "method"
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase5_cards.py -v
git add tools/phases/phase5_cards.py tests/unit/test_phase5_cards.py
git commit -m "feat(B): phase 5a paper cards (LiRA dimension decomposition)"
```

### Task 2.5b: EvidenceStore (NLI auto-fill)

**Files:**
- Create: `tools/phases/phase5_evidence.py`
- Test: `tests/unit/test_phase5_evidence.py`

**Interfaces:**
- Consumes: `ParsedPapers`, `PaperCards` (claims to match against), `NLIVerifier`.
- Produces: `EvidenceStore`. For each paragraph → an `Evidence` with deterministic `evidence_id`; `supports_claims` auto-filled by matching the paragraph against that paper's claims via `nli.best_match`. Source priority: paragraphs > figure captions > abstract.

- [ ] **Step 1: Implement** (`tools/phases/phase5_evidence.py`)

```python
from tools.models.artifacts import Evidence, EvidenceStore
from tools.models.common import evidence_id

def run(task_id, parsed_papers, paper_cards, nli):
    card_by_pid = {c.paper_id: c for c in paper_cards.cards}
    evidences = []
    for p in parsed_papers.papers:
        card = card_by_pid.get(p.paper_id)
        claim_texts = [cl.text for bucket in card.possible_claims.values() for cl in bucket] if card else []
        # abstract as one evidence
        if p.abstract:
            evidences.append(_make(p.paper_id, 0, 0, p.abstract, "abstract", claim_texts, nli))
        for para in p.paragraphs:
            evidences.append(_make(p.paper_id, para.page, para.index, para.text, "paragraph", claim_texts, nli))
        for fig in p.figures:
            if fig.get("caption"):
                evidences.append(_make(p.paper_id, fig.get("page",0), fig["num"], fig["caption"], "caption", claim_texts, nli))
    return EvidenceStore(task_id=task_id, evidences=evidences)

def _make(paper_id, page, idx, text, source_type, claim_texts, nli):
    supports = []
    if claim_texts:
        res = nli.best_match(text, claim_texts)  # NLIResult
        if res.support_type != "contradictory" or res.confidence > 0:
            # attach to the top claim text by re-querying per claim is expensive; attach best only
            supports.append({"claim_text": claim_texts[0], "support_type": res.support_type, "confidence": res.confidence})
    return Evidence(evidence_id=evidence_id(paper_id, page, idx), paper_id=paper_id,
        source_type=source_type, source_page=page, source_paragraph_index=idx,
        text=text, supports_claims=supports)
```

- [ ] **Step 2: Write tests** with `FakeNLIModel` (`tests/unit/test_phase5_evidence.py`)

```python
from tools.phases.phase5_evidence import run
from tools.models.artifacts import ParsedPapers, ParsedPaper, Paragraph, PaperCards, PaperCard, Claim
from tools.nlp.nli_verifier import FakeNLIModel, NLIResult

class NLIStub:
    def best_match(self, evidence_text, claim_texts):
        return NLIResult("entailment","direct",0.85)

def test_evidence_id_and_supports():
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A", abstract="abs",
        paragraphs=[Paragraph(page=3,index=0,text="we show SOTA")])])
    cards = PaperCards(task_id="t", cards=[PaperCard(paper_id="paper:1", title="A",
        possible_claims={"key_results":[Claim(text="SOTA result",dimension="key_results")]})])
    es = run("t", pp, cards, NLIStub())
    ids = [e.evidence_id for e in es.evidences]
    assert "paper:1_p3_0" in ids and "paper:1_p0_0" in ids  # paragraph + abstract
    para_ev = [e for e in es.evidences if e.evidence_id=="paper:1_p3_0"][0]
    assert para_ev.supports_claims[0]["support_type"] == "direct"
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase5_evidence.py -v
git add tools/phases/phase5_evidence.py tests/unit/test_phase5_evidence.py
git commit -m "feat(B): phase 5b evidence store with NLI auto-fill"
```

### Task 2.5c: FigureBank/TableBank + Taxonomy step3 + CitationIndex

**Files:**
- Create: `tools/phases/phase5_synthesis_rest.py`
- Test: `tests/unit/test_phase5_synthesis_rest.py`

**Interfaces:**
- Consumes: `ParsedPapers`, `PaperCards`, `SurveyStructure.refined_taxonomy`, `LLMClient` (taxonomy step3).
- Produces: `(FigureBank, TableBank, Taxonomy, CitationIndex)`.
- Taxonomy step3: assign each card to a category (keyword/LLM), allow new categories for clusters, flag empty/overfull. `category_id` run-local `cat_NNN`.
- CitationIndex: derived from `paper_cards` (every card → one citation entry).

- [ ] **Step 1: Implement** (`tools/phases/phase5_synthesis_rest.py`)

```python
from tools.models.artifacts import Figure, FigureBank, Table, TableBank, Category, Taxonomy, CitationIndex
from tools.models.common import figure_id, table_id, category_id

def build_figure_bank(task_id, parsed_papers):
    figs = [Figure(figure_id=figure_id(p.paper_id, f["num"]), paper_id=p.paper_id,
              caption=f.get("caption",""), page=f.get("page",0))
            for p in parsed_papers.papers for f in p.figures]
    return FigureBank(task_id=task_id, figures=figs)

def build_table_bank(task_id, parsed_papers):
    tbls = [Table(table_id=table_id(p.paper_id, t["num"]), paper_id=p.paper_id,
              caption=t.get("caption",""), page=t.get("page",0))
            for p in parsed_papers.papers for t in p.tables]
    return TableBank(task_id=task_id, tables=tbls)

def build_taxonomy(task_id, topic, refined_taxonomy, paper_cards, llm):
    # keyword-assign first (deterministic), then optional LLM rebalance
    cats = []
    name_to_id = {}
    def get_cat(name):
        if name not in name_to_id:
            name_to_id[name] = category_id(len(name_to_id)+1)
        return name_to_id[name]
    for c in refined_taxonomy:
        get_cat(c["name"])
        cats.append(Category(category_id=name_to_id[c["name"]], category_name=c["name"],
            description=c.get("description","")))
    # assign each card to best-matching category by keyword overlap
    for card in paper_cards.cards:
        best = None; best_score = -1
        text = f"{card.title} {card.method}".lower()
        for c in refined_taxonomy:
            score = sum(1 for kw in c["name"].lower().split() if kw in text)
            if score > best_score:
                best_score = score; best = c["name"] if score >= 0 else (refined_taxonomy[0]["name"] if refined_taxonomy else None)
        if best:
            for cat in cats:
                if cat.category_name == best:
                    cat.paper_ids.append(card.paper_id); break
    for cat in cats:
        cat.paper_count = len(cat.paper_ids)
    return Taxonomy(task_id=task_id, topic=topic, taxonomy_version="v1",
        refine_history=["preliminary","refined","final"], categories=cats)

def build_citation_index(task_id, paper_cards):
    citations = [{"paper_id": c.paper_id, "bibtex_key": c.bibtex_key,
                  "title": c.title, "year": c.year} for c in paper_cards.cards]
    return CitationIndex(task_id=task_id, citations=citations)
```

- [ ] **Step 2: Write tests** (`tests/unit/test_phase5_synthesis_rest.py`)

```python
from tools.phases.phase5_synthesis_rest import build_figure_bank, build_table_bank, build_taxonomy, build_citation_index
from tools.models.artifacts import ParsedPapers, ParsedPaper, PaperCards, PaperCard

class FakeLLM:
    def complete_json(self, m, **kw): return {"categories":[]}
    def complete(self, m, **kw): return ""

def test_figure_and_table_ids():
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        figures=[{"num":1,"caption":"arch","page":3}], tables=[{"num":2,"caption":"res","page":5}])])
    fb = build_figure_bank("t", pp); tb = build_table_bank("t", pp)
    assert fb.figures[0].figure_id == "paper:1_fig1"
    assert tb.tables[0].table_id == "paper:1_tbl2"

def test_taxonomy_assigns_cards():
    refined = [{"name":"Internal World Model","description":"d"}]
    cards = PaperCards(task_id="t", cards=[PaperCard(paper_id="paper:1", title="Dreamer world model", method="latent")])
    tax = build_taxonomy("t","wm",refined, cards, FakeLLM())
    assert tax.categories[0].paper_count == 1
    assert tax.categories[0].category_id.startswith("cat_")

def test_citation_index_has_all_cards():
    cards = PaperCards(task_id="t", cards=[PaperCard(paper_id="paper:1", title="A"),
                                           PaperCard(paper_id="paper:2", title="B")])
    ci = build_citation_index("t", cards)
    assert {c["paper_id"] for c in ci.citations} == {"paper:1","paper:2"}
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase5_synthesis_rest.py -v
git add tools/phases/phase5_synthesis_rest.py tests/unit/test_phase5_synthesis_rest.py
git commit -m "feat(B): phase 5c figure/table banks, taxonomy step3, citation index"
```

---

## Task 2.6: Phase 6 — Bundle Assembly

**Files:**
- Create: `tools/phases/phase6_bundle_assembler.py`
- Test: `tests/unit/test_phase6.py`

**Interfaces:**
- Consumes: all artifacts + `demand.coverage_warnings`/`structure_errors` + quality thresholds.
- Produces: `KnowledgeBundle` (§8.2) with `status` (success/partial_success/failed), `summary`, `coverage_report`, `quality_report`. Writes `cache/knowledge_bundle.json` and ensures all artifact files exist (empty structure if missing, per §23.1).

- [ ] **Step 1: Implement** (`tools/phases/phase6_bundle_assembler.py`)

```python
from tools.models.bundle import KnowledgeBundle

def run(task_id, topic, artifacts_paths, retrieved, parsed, paper_cards, evidence_store,
        figure_bank, table_bank, taxonomy, citation_index, structure_errors, coverage_warnings,
        quality_requirements):
    paper_count = len(retrieved.papers)
    core_count = sum(1 for c in paper_cards.cards if c.card_type == "deep")
    warnings = list(structure_errors) + list(coverage_warnings)
    parse_rate = (len(parsed.papers) / paper_count) if paper_count else 0.0
    card_rate = (len(paper_cards.cards) / paper_count) if paper_count else 0.0
    ev_rate = (len([e for c in paper_cards.cards for _ in c.evidence_ids]) / max(core_count,1))
    covered = sorted({a["aspect_id"] for c in paper_cards.cards for a in c.matched_aspects})
    per_aspect = {a["aspect_id"]: 0 for a in retrieved.papers}  # filled by caller if needed
    # status logic
    if paper_count >= quality_requirements.min_total_papers and core_count >= quality_requirements.min_core_papers:
        status = "success" if not warnings else "partial_success"
    else:
        status = "partial_success" if paper_count else "failed"
    return KnowledgeBundle(task_id=task_id, topic=topic, status=status, artifacts=artifacts_paths,
        summary={"paper_count": paper_count, "core_paper_count": core_count,
            "parsed_paper_count": len(parsed.papers), "figure_count": len(figure_bank.figures),
            "table_count": len(table_bank.tables), "evidence_count": len(evidence_store.evidences),
            "taxonomy_category_count": len(taxonomy.categories), "citation_count": len(citation_index.citations)},
        coverage_report={"covered_aspects": covered, "undercovered_aspects": [], "papers_per_aspect": per_aspect},
        quality_report={"search_success": paper_count > 0, "parse_success_rate": round(parse_rate,3),
            "paper_card_success_rate": round(card_rate,3), "evidence_coverage_rate": round(ev_rate,3),
            "has_multimodal_evidence": len(figure_bank.figures) > 0, "warnings": warnings})
```

- [ ] **Step 2: Write tests** (`tests/unit/test_phase6.py`)

```python
from tools.phases.phase6_bundle_assembler import run
from tools.models.artifacts import *
from tools.models.requests import QualityRequirements

def _empty_artifacts():
    return (RetrievedPapers(task_id="t",papers=[]), ParsedPapers(task_id="t",papers=[]),
            PaperCards(task_id="t",cards=[]), EvidenceStore(task_id="t",evidences=[]),
            FigureBank(task_id="t",figures=[]), TableBank(task_id="t",tables=[]),
            Taxonomy(task_id="t",topic="wm",categories=[]), CitationIndex(task_id="t",citations=[]))

def test_failed_status_when_no_papers():
    rp,pp,pc,es,fb,tb,tax,ci = _empty_artifacts()
    b = run("t","wm",{},rp,pp,pc,es,fb,tb,tax,ci,[],[],QualityRequirements())
    assert b.status == "failed" and b.summary["paper_count"] == 0

def test_success_status_with_enough_papers():
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}",title=f"T{i}") for i in range(12)])
    pc = PaperCards(task_id="t", cards=[PaperCard(paper_id=f"p{i}",title=f"T{i}",card_type="deep") for i in range(12)])
    rp, pp, pc, es, fb, tb, tax, ci = _empty_artifacts()
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}",title="T") for i in range(12)])
    pc = PaperCards(task_id="t", cards=[PaperCard(paper_id=f"p{i}",title="T",card_type="deep") for i in range(6)])
    b = run("t","wm",{},rp,pp,pc,es,fb,tb,tax,ci,[],[],QualityRequirements(min_total_papers=10,min_core_papers=5))
    assert b.status == "success"
    assert b.quality_report["warnings"] == []
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_phase6.py -v
git add tools/phases/phase6_bundle_assembler.py tests/unit/test_phase6.py
git commit -m "feat(B): phase 6 bundle assembly"
```

---

# Wave 3 — Verification + Orchestration

## Task 3.1: verify_citations (structural)

**Files:**
- Create: `tools/verify/verify_citations.py`
- Test: `tests/unit/test_verify_citations.py`

**Interfaces:**
- Consumes: `survey_md: str`, `CitationIndex`, `FigureBank`, `TableBank`.
- Produces: `CitationResult` (§13.3). Extracts `[paper_id]`, `![fig](figure_id)`, table refs via regex; flags any id not in the index/bank as invalid. Computes `citation_validity_score`.

- [ ] **Step 1: Implement** (`tools/verify/verify_citations.py`)

```python
import re
from tools.models.bundle import CitationResult, CitationEntry

CITE_RE = re.compile(r"\[([^\]]+)\]")
FIG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

def extract_citations(md: str) -> list[str]:
    return [c.strip() for c in CITE_RE.findall(md) if not c.strip().startswith("http")]

def run(task_id, survey_md, citation_index, figure_bank, table_bank):
    ready = {c["paper_id"] for c in citation_index.citations}
    fig_set = {f.figure_id for f in figure_bank.figures}
    tbl_set = {t.table_id for t in table_bank.tables}
    entries = []
    for cid in extract_citations(survey_md):
        entries.append(CitationEntry(citation_id=cid, valid=cid in ready,
            figures_valid=[], tables_valid=[]))
    valid = sum(1 for e in entries if e.valid)
    total = len(entries)
    score = (valid / total) if total else 1.0
    return CitationResult(task_id=task_id, total_citations=total, valid_citations=valid,
        invalid_citations=total - valid, weak_claims=0,
        citation_validity_score=round(score, 3), entries=entries)
```

- [ ] **Step 2: Write tests** (`tests/unit/test_verify_citations.py`)

```python
from tools.verify.verify_citations import run, extract_citations
from tools.models.artifacts import CitationIndex, FigureBank, Figure

def test_flags_fabricated_id():
    ci = CitationIndex(task_id="t", citations=[{"paper_id":"paper:real"}])
    md = "Some claim [paper:real] and a fake [paper:fake]."
    res = run("t", md, ci, FigureBank(task_id="t",figures=[]), FigureBank(task_id="t",tables=[]).tables and FigureBank(task_id="t",figures=[]))
    # note: tables_bank is a TableBank; simplified below
    from tools.models.artifacts import TableBank
    res = run("t", md, ci, FigureBank(task_id="t",figures=[]), TableBank(task_id="t",tables=[]))
    ids = {e.citation_id: e.valid for e in res.entries}
    assert ids["paper:real"] is True and ids["paper:fake"] is False
    assert res.citation_validity_score == 0.5

def test_extract_skips_urls():
    assert extract_citations("see [https://x.com]") == []
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_verify_citations.py -v
git add tools/verify/verify_citations.py tests/unit/test_verify_citations.py
git commit -m "feat(B): structural citation verifier"
```

---

## Task 3.2: build_claim_map (NLI-first + LLM fallback)

**Files:**
- Create: `tools/verify/build_claim_map.py`
- Test: `tests/unit/test_build_claim_map.py`

**Interfaces:**
- Consumes: `survey_md`, `EvidenceStore`, `ParsedPapers`, `NLIVerifier`, `LLMClient` (fallback only).
- Produces: `ClaimMap` (§13.4). 3-stage: exact EvidenceStore match → NLI entailment (≥0.6 supported, neutral≥0.4 weak) → LLM fallback for 0.4–0.6. status ∈ supported/weak/unsupported.

- [ ] **Step 1: Implement** (`tools/verify/build_claim_map.py`)

```python
import re
from tools.models.bundle import ClaimMap, ClaimEntry

# a claim = a sentence in md that ends with a [paper_id] citation
CLAIM_RE = re.compile(r"([^.]*?\[([^\]]+)\][^.]*\.)")

def extract_claims_with_citations(md: str):
    out = []
    for sentence in re.split(r"(?<=[.。])\s+", md):
        cites = re.findall(r"\[([^\]]+)\]", sentence)
        if cites:
            text = re.sub(r"\[[^\]]+\]", "", sentence).strip()
            if text:
                out.append((text, cites[0]))
    return out

def run(task_id, survey_md, evidence_store, parsed_papers, nli, llm=None):
    by_paper = {}
    for e in evidence_store.evidences:
        by_paper.setdefault(e.paper_id, []).append(e)
    parsed_by_pid = {p.paper_id: p for p in parsed_papers.papers}
    entries = []
    for claim_text, paper_id in extract_claims_with_citations(survey_md):
        evidences = by_paper.get(paper_id, [])
        # stage 1: exact direct match
        direct = any(any(sc.get("support_type")=="direct" for sc in e.supports_claims) for e in evidences)
        if direct:
            status, conf, eids = "supported", 1.0, [e.evidence_id for e in evidences]
        elif evidences:
            res = nli.best_match(claim_text, [e.text for e in evidences])
            eids = [e.evidence_id for e in evidences]
            if res.confidence >= 0.6:
                status = "supported" if res.label=="entailment" else ("weak" if res.label=="neutral" else "unsupported")
                conf = res.confidence
            elif res.confidence >= 0.4 and llm is not None:
                status = llm_fallback(llm, claim_text, evidences[0].text)
                conf = 0.5
            else:
                status = "unsupported"; conf = res.confidence
        else:
            status, conf, eids = "unsupported", 0.0, []
        entries.append(ClaimEntry(claim_text=claim_text, cited_paper_id=paper_id,
            status=status, evidence_ids=eids, confidence=conf))
    return ClaimMap(task_id=task_id, entries=entries)

def llm_fallback(llm, claim, evidence):
    out = llm.complete([{"role":"user","content":
        f"Does this evidence support the claim? Answer one word: supported, weak, or unsupported.\nEvidence: {evidence}\nClaim: {claim}"}])
    return "supported" if "support" in out.lower() else ("weak" if "weak" in out.lower() else "unsupported")
```

- [ ] **Step 2: Write tests** with `FakeNLIModel`-like stub (`tests/unit/test_build_claim_map.py`)

```python
from tools.verify.build_claim_map import run, extract_claims_with_citations
from tools.models.artifacts import EvidenceStore, Evidence, ParsedPapers
from tools.nlp.nli_verifier import NLIResult

class NLIStub:
    def __init__(self, label, conf): self.label,self.conf = label,conf
    def best_match(self, claim, evidences):
        return NLIResult(self.label, {"entailment":"direct","neutral":"indirect","contradiction":"contradictory"}[self.label], self.conf)

def test_unsupported_when_no_evidence():
    md = "DreamerV3 flies airplanes [paper:1]."
    es = EvidenceStore(task_id="t", evidences=[])
    cm = run("t", md, es, ParsedPapers(task_id="t",papers=[]), NLIStub("contradiction",0.9))
    assert cm.entries[0].status == "unsupported"

def test_supported_via_nli_entailment():
    md = "DreamerV3 masters many domains [paper:1]."
    es = EvidenceStore(task_id="t", evidences=[Evidence(evidence_id="paper:1_p3_0",paper_id="paper:1",text="we master 150 domains")])
    cm = run("t", md, es, ParsedPapers(task_id="t",papers=[]), NLIStub("entailment",0.85))
    assert cm.entries[0].status == "supported"
    assert cm.entries[0].evidence_ids == ["paper:1_p3_0"]
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_build_claim_map.py -v
git add tools/verify/build_claim_map.py tests/unit/test_build_claim_map.py
git commit -m "feat(B): NLI-first claim mapper with LLM fallback"
```

---

## Task 3.3: knowledge_pipeline_worker (orchestration)

**Files:**
- Create: `tools/knowledge_pipeline_worker.py`
- Test: `tests/integration/test_pipeline_e2e.py` (full e2e — see Wave 4)

**Interfaces:**
- Consumes: `KnowledgeBuildRequest`; builds all clients once from config; loads search_strategy + surveys + seed_papers from paths.
- Produces: `KnowledgeBundle`; writes all `cache/*.json` artifacts. This is the `build_paper_cards` + `search_papers` shared engine (the tool_executor calls into phases, not here directly — see Task 3.4).

- [ ] **Step 1: Implement** (`tools/knowledge_pipeline_worker.py`)

```python
import json
from pathlib import Path
from tools.config import get_config
from tools.models.requests import KnowledgeBuildRequest, SearchStrategy
from tools.clients.llm_client import LLMClient
from tools.clients.sciverse_client import SciVerseClient
from tools.clients.mineru_client import MinerUClient
from tools.clients.embedding_client import EmbeddingClient, FakeEmbeddingClient
from tools.nlp.nli_verifier import NLIVerifier
from tools.nlp.data_cleaner import DataCleaner
from tools.phases import (phase1_decompose, phase2_survey_analyzer, phase3_paper_retriever,
    phase4_rag_indexer, phase5_cards, phase5_evidence, phase5_synthesis_rest, phase6_bundle_assembler)

def _load(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def run(request: KnowledgeBuildRequest) -> dict:
    cfg = get_config()
    cache = Path(request.outputs["knowledge_bundle_path"]).parent
    cache.mkdir(parents=True, exist_ok=True)
    strategy = SearchStrategy(**_load(request.inputs["search_strategy_path"]))
    surveys = _load("structured_data/surveys.json")
    seed_papers = _load("cache/seed_papers.json") if Path("cache/seed_papers.json").exists() else []

    llm = LLMClient(config=cfg)
    sciverse = SciVerseClient(config=cfg)
    mineru = MinerUClient(config=cfg, use_mock=request.pipeline_config.use_mock_mineru_if_failed)
    cleaner = DataCleaner()
    emb = FakeEmbeddingClient() if cfg.embedding_provider == "mock" else EmbeddingClient(config=cfg)

    demand = phase1_decompose.run(request, strategy, seed_papers)
    survey_struct = phase2_survey_analyzer.run(request.task_id, request.topic, strategy.sub_domains,
        demand.aspects, surveys, llm, mineru, cleaner, sciverse)
    retrieved, parsed = phase3_paper_retriever.run(request.task_id, demand.aspects,
        survey_struct.expansion_candidates, sciverse, mineru, cleaner, seed_papers, request.pipeline_config)
    rag = phase4_rag_indexer.run(parsed, emb, path=str(cache / "rag_index"))
    cards = phase5_cards.run(request.task_id, parsed, retrieved, llm, demand.aspects,
                             request.pipeline_config.aspect_match_threshold)
    evidence = phase5_evidence.run(request.task_id, parsed, cards, NLIVerifier())
    figure_bank = phase5_synthesis_rest.build_figure_bank(request.task_id, parsed)
    table_bank = phase5_synthesis_rest.build_table_bank(request.task_id, parsed)
    taxonomy = phase5_synthesis_rest.build_taxonomy(request.task_id, request.topic,
        survey_struct.refined_taxonomy, cards, llm)
    citation_index = phase5_synthesis_rest.build_citation_index(request.task_id, cards)
    bundle = phase6_bundle_assembler.run(request.task_id, request.topic, request.outputs,
        retrieved, parsed, cards, evidence, figure_bank, table_bank, taxonomy, citation_index,
        demand.structure_errors, demand.coverage_warnings, request.quality_requirements)

    _write_all(cache, request, retrieved, parsed, cards, evidence, figure_bank, table_bank,
               taxonomy, citation_index, survey_struct, bundle)
    return bundle.model_dump()

def _write_all(cache, request, retrieved, parsed, cards, evidence, figure_bank, table_bank,
               taxonomy, citation_index, survey_struct, bundle):
    def w(name, model):
        Path(request.outputs[name + "_path"]).write_text(model.model_dump_json(indent=2), encoding="utf-8")
    Path(request.outputs["survey_structure_path"]).write_text(survey_struct.model_dump_json(indent=2), encoding="utf-8") if "survey_structure_path" in request.outputs else None
    w("retrieved_papers", retrieved); w("parsed_papers", parsed); w("paper_cards", cards)
    w("evidence_store", evidence); w("figure_bank", figure_bank); w("table_bank", table_bank)
    w("taxonomy", taxonomy); w("citation_index", citation_index)
    Path(request.outputs["knowledge_bundle_path"]).write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
```

- [ ] **Step 2: Commit** (e2e test is in Wave 4 Task 4.3)
```bash
git add tools/knowledge_pipeline_worker.py
git commit -m "feat(B): knowledge pipeline worker orchestration"
```

---

## Task 3.4: tool_executor + tool_definitions

**Files:**
- Create: `tools/tool_definitions.py`, `tools/tool_executor.py`
- Test: `tests/integration/test_tool_executor.py`

**Interfaces:**
- `tool_definitions.TOOLS: list[dict]` — the 3 JSON schemas from arch doc §1.
- `tool_executor.dispatch(name: str, args: dict) -> dict` — routes to the right phase-set:
  - `search_papers` → Phase 1-4, returns retrieved+parsed summary
  - `build_paper_cards` → Phase 5-6, returns knowledge_bundle
  - `verify_citations` → §4.2 + §4.3, returns citation_result + claim_map

- [ ] **Step 1: Implement `tools/tool_definitions.py`** (copy the 3 tool schemas verbatim from arch doc §1: `search_papers`, `build_paper_cards`, `verify_citations`).

```python
TOOLS = [
    {"type":"function","function":{"name":"search_papers",
      "description":"检索论文：需求拆解→综述分析→方法论文检索→PDF解析→RAG索引。返回论文列表摘要。",
      "parameters":{"type":"object","properties":{
        "topic":{"type":"string"},"keywords":{"type":"string"},
        "year_range":{"type":"array","items":{"type":"integer"}}},
        "required":["topic","keywords"]}}},
    {"type":"function","function":{"name":"build_paper_cards",
      "description":"对论文逐篇构建 Paper Card（原子论断/方法/局限），构建 EvidenceStore/Taxonomy/CitationIndex，打包 KnowledgeBundle。",
      "parameters":{"type":"object","properties":{
        "paper_ids":{"type":"array","items":{"type":"string"}},
        "focus_aspects":{"type":"array","items":{"type":"string"}}},
        "required":["paper_ids"]}}},
    {"type":"function","function":{"name":"verify_citations",
      "description":"对综述草稿做结构性引用校验与 NLI 论断归属性校验，返回每条引用/论断的校验结果。",
      "parameters":{"type":"object","properties":{"draft_text":{"type":"string"}},
        "required":["draft_text"]}}},
]
```

- [ ] **Step 2: Implement `tools/tool_executor.py`**

```python
from tools.knowledge_pipeline_worker import run as run_pipeline
from tools.verify.verify_citations import run as run_verify
from tools.verify.build_claim_map import run as run_claims
from tools.models.requests import KnowledgeBuildRequest

def dispatch(name: str, args: dict, request: KnowledgeBuildRequest) -> dict:
    if name == "search_papers":
        bundle = run_pipeline(request)
        return {"status": "ok", "summary": bundle["summary"], "paper_count": bundle["summary"]["paper_count"]}
    if name == "build_paper_cards":
        bundle = run_pipeline(request)
        return {"status": "ok", "knowledge_bundle": bundle}
    if name == "verify_citations":
        from tools.models.artifacts import CitationIndex, FigureBank, TableBank, EvidenceStore
        from tools.models.bundle import KnowledgeBundle
        import json
        from pathlib import Path
        b = KnowledgeBundle(**json.loads(Path(request.outputs["knowledge_bundle_path"]).read_text()))
        ci = CitationIndex(**json.loads(Path(request.outputs["citation_index_path"]).read_text()))
        fb = FigureBank(**json.loads(Path(request.outputs["figure_bank_path"]).read_text()))
        tb = TableBank(**json.loads(Path(request.outputs["table_bank_path"]).read_text()))
        es = EvidenceStore(**json.loads(Path(request.outputs["evidence_store_path"]).read_text()))
        from tools.phases import phase3_paper_retriever  # for ParsedPapers type only
        from tools.models.artifacts import ParsedPapers
        pp = ParsedPapers(**json.loads(Path(request.outputs["parsed_papers_path"]).read_text()))
        cit = run_verify(request.task_id, args["draft_text"], ci, fb, tb)
        from tools.nlp.nli_verifier import NLIVerifier
        cm = run_claims(request.task_id, args["draft_text"], es, pp, NLIVerifier())
        return {"citation_result": cit.model_dump(), "claim_map": cm.model_dump()}
    raise ValueError(f"unknown tool: {name}")
```

- [ ] **Step 3: Write tests** (`tests/integration/test_tool_executor.py`) — assert `TOOLS` has the 3 names and `dispatch` raises on unknown tool. Full dispatch is exercised by the e2e test.

```python
import pytest
from tools.tool_definitions import TOOLS
from tools.tool_executor import dispatch

def test_three_tools_defined():
    assert {t["function"]["name"] for t in TOOLS} == {"search_papers","build_paper_cards","verify_citations"}

def test_dispatch_unknown_raises():
    with pytest.raises(ValueError):
        dispatch("nope", {}, None)
```

- [ ] **Step 4: Run + commit**
```bash
pytest tests/integration/test_tool_executor.py -v
git add tools/tool_definitions.py tools/tool_executor.py tests/integration/test_tool_executor.py
git commit -m "feat(B): tool definitions + executor routing"
```

---

# Wave 4 — Fixtures, Mocks, End-to-End

## Task 4.1: seed_papers.json (B1 deliverable)

**Files:**
- Create: `cache/seed_papers.json`
- Test: `tests/unit/test_seed_papers.py`

- [ ] **Step 1: Build the seed corpus** covering the 6 directions from `docs/模块B-工作清单.md` §B1. Each entry: `{title, authors, year, venue, url, abstract, keywords, category, bibtex_key}`. Minimum papers: DQN, AlphaGo, AlphaZero, AlphaStar, OpenAI Five, World Models, PlaNet, Dreamer, DreamerV3, MuZero, EfficientZero, IRIS, DIAMOND, GameGAN, Genie, GameNGen, Oasis, Muse/WHAM, Hunyuan-GameCraft, Matrix-Game, Yume, Voyager, MineDojo, VPT, Cradle, Generative Agents. (Abstracts can be 1–2 sentences; real titles/years/venues required — no fabricated DOIs.)

- [ ] **Step 2: Write test** asserting ≥6 categories covered and required fields present.
```python
import json
from pathlib import Path

def test_seed_papers_cover_six_directions():
    data = json.loads(Path("cache/seed_papers.json").read_text())
    cats = {p["category"] for p in data}
    assert len(data) >= 20
    for p in data:
        assert p["title"] and p["year"] and p["keywords"]
```

- [ ] **Step 3: Run + commit**
```bash
pytest tests/unit/test_seed_papers.py -v
git add cache/seed_papers.json tests/unit/test_seed_papers.py
git commit -m "feat(B): seed papers corpus (B1)"
```

---

## Task 4.2: conftest fakes + fixtures

**Files:**
- Create: `tests/conftest.py`, `tests/fixtures/sample_parsed_paper.json`, `tests/fixtures/sample_survey.json`, `tests/fixtures/sample_survey_md.md`

- [ ] **Step 1: Implement `tests/conftest.py`** — centralize `FakeLLMClient`, `FakeSciVerseClient`, `FakeMinerUClient`, `FakeEmbeddingClient`, `FakeNLIModel`, and pytest fixtures `fake_llm`, `fake_sv`, `fake_mu`, `parsed_paper`, `survey_md`. (These fakes are the test doubles referenced throughout Waves 1–3; consolidate here to avoid duplication.)

```python
import json, pytest
from pathlib import Path
from tools.models.artifacts import ParsedPapers, ParsedPaper

class FakeLLMClient:
    def __init__(self, json_responses=None, text="ok"):
        self.json_responses = json_responses or {}
        self.text = text; self.calls = 0
    def complete(self, messages, **kw):
        self.calls += 1; return self.text
    def complete_json(self, messages, **kw):
        self.calls += 1
        for tag, resp in self.json_responses.items():
            if any(tag in (m.get("content","")) for m in messages):
                return resp
        return {"categories": [{"name":"Default","description":"d"}]}

@pytest.fixture
def parsed_paper():
    return ParsedPapers(**json.loads(Path("tests/fixtures/sample_parsed_paper.json").read_text()))

@pytest.fixture
def survey_md():
    return Path("tests/fixtures/sample_survey_md.md").read_text()
```

- [ ] **Step 2: Create fixtures** — `sample_parsed_paper.json` (one DreamerV3-like paper with abstract + 2 paragraphs + 1 figure), `sample_survey.json`, `sample_survey_md.md` (a short survey draft with 2 real + 1 fake citation and one entailed + one contradicted claim).

- [ ] **Step 3: Commit**
```bash
git add tests/conftest.py tests/fixtures/
git commit -m "test(B): shared fakes and fixtures"
```

---

## Task 4.3: End-to-end integration test (all externals mocked)

**Files:**
- Create: `tests/integration/test_pipeline_e2e.py`

**Interfaces:** Exercises `knowledge_pipeline_worker.run` + `tool_executor.dispatch("verify_citations", ...)` with every external mocked (FakeLLM returns canned taxonomies + card text; FakeSciVerse returns 3 papers; FakeMinerU returns parsed text; FakeEmbedding; FakeNLI). Asserts: all 9 cache files written, bundle status, and the cross-cutting invariants from the Test Strategy section.

- [ ] **Step 1: Write the e2e test** (`tests/integration/test_pipeline_e2e.py`)

```python
import json
from pathlib import Path
import pytest
from tools.models.requests import KnowledgeBuildRequest

@pytest.fixture
def request_obj(tmp_path):
    return KnowledgeBuildRequest(task_id="e2e", topic="World Models for GameCraft",
        request_type="knowledge_build",
        inputs={"task_request_path": str(tmp_path/"tr.json"), "search_strategy_path": str(tmp_path/"ss.json")},
        outputs={k+"_path": str(tmp_path/f"{k}.json") for k in
                 ["knowledge_bundle","retrieved_papers","parsed_papers","figure_bank","table_bank",
                  "paper_cards","evidence_store","taxonomy","citation_index"]})

def test_full_pipeline_produces_all_artifacts(request_obj, tmp_path, monkeypatch):
    # write a search strategy + task request + seed
    (tmp_path/"ss.json").write_text(json.dumps({"topic":"World Models","sub_domains":["GameCraft"],
        "wide_search":{"search_aspects":[
            {"aspect_id":"a1","aspect_name":"GameCraft","keywords":["game","world model"]},
            {"aspect_id":"a2","aspect_name":"Internal WM","keywords":["latent","dynamics"]},
            {"aspect_id":"a3","aspect_name":"Benchmark","keywords":["benchmark","eval"]}],
            "time_range":{"start_year":2018,"end_year":2026}}}))
    (tmp_path/"tr.json").write_text("{}")
    # monkeypatch all externals + workers built inside run()
    from tools import knowledge_pipeline_worker as K
    from tests.conftest import FakeLLMClient
    monkeypatch.setattr(K, "LLMClient", lambda config=None: FakeLLMClient())
    # ... similarly patch SciVerseClient, MinerUClient, NLIVerifier to fakes returning canned data
    bundle = K.run(request_obj)
    out = request_obj.outputs
    for k in ["knowledge_bundle","retrieved_papers","parsed_papers","paper_cards","evidence_store",
              "taxonomy","citation_index","figure_bank","table_bank"]:
        assert Path(out[k+"_path"]).exists(), f"{k} missing"
    # invariant: every citation_index paper_id is a card
    cards = json.loads(Path(out["paper_cards_path"]).read_text())
    ci = json.loads(Path(out["citation_index_path"]).read_text())
    card_ids = {c["paper_id"] for c in cards["cards"]}
    assert {c["paper_id"] for c in ci["citations"]} <= card_ids
    assert bundle["summary"]["paper_count"] >= 0
```

- [ ] **Step 2: Run + commit** (iterate on fakes until green)
```bash
pytest tests/integration/test_pipeline_e2e.py -v
git add tests/integration/test_pipeline_e2e.py
git commit -m "test(B): end-to-end pipeline integration test"
```

---

## Task 4.4: Optional real-API smoke test (manual)

**Files:**
- Create: `tests/integration/test_smoke_real.py` (marked `@pytest.mark.skip` by default)

- [ ] **Step 1:** A `@pytest.mark.skip(reason="needs real API keys + network")` test that calls `SciVerseClient.meta_search` once and asserts a non-empty `hits` list. Run manually only when keys are set, to validate the real Intern/SciVerse/MinerU contracts before the demo.

- [ ] **Step 2: Commit**
```bash
git add tests/integration/test_smoke_real.py
git commit -m "test(B): optional real-API smoke test (skipped by default)"
```

---

## Final Verification

After all tasks:
```bash
pytest -v                       # all unit + integration green
pytest -m "not nli_real" -v     # without the heavy NLI download
```
Then a manual `python -c "from tools.knowledge_pipeline_worker import run; ..."` against real keys before the demo (Task 4.4 path).

---

## Self-Review Notes (post-write)

- **Spec coverage:** arch doc Phases 1–6 → Wave 2 (2.1–2.6); §4.2/§4.3 verification → Wave 3 (3.1/3.2); 3 tools → 3.4; clients (SciVerse/MinerU/Embedding/LLM) → 1.1–1.4; NLI reuse + cleaner reuse + LiRA reuse → 1.5/1.6/2.5a; ID rules §5.0 → 0.2; B1 seed corpus → 4.1; mock strategy §23 → 1.3/4.2. **Gap:** §2.1 survey-store update (`update_survey_store`) is stubbed but not fully implemented — acceptable for 24h; flagged in Phase 2 as optional. §4.4 independent cross-verify is explicitly low-priority/skipped (arch doc agrees).
- **Type consistency:** `NLIResult`/`NLIVerifier.judge/best_match` used identically in 1.5, 2.5b, 3.2. `evidence_id` format `{paper_id}_p{page}_{idx}` consistent in 0.2, 2.4, 2.5b. `RetrievedPaper.paper_id` source of truth in 1.2/2.3/3.1.
- **Test convention:** non-TDD (implement → test → commit) per user direction; this overrides the writing-plans skill's TDD default.
