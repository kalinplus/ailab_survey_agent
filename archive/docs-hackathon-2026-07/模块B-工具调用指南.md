# Module B 工具调用指南

> 面向 A/C 组开发者。描述 Module B 对外暴露的 2 个工具及其 I/O 合约。

## 概览

Module B 通过 `harness/tool_registry.py` 注册 2 个粗粒度 tool：

| Tool Name | 功能 | 调用时机 |
|-----------|------|----------|
| `knowledge_pipeline_worker` | 论文检索 → 解析 → PaperCards → Evidence → Taxonomy → Bundle | A 发起知识构建 |
| `verify_citations` | 综述引用验证 + Claim→Evidence 映射 | C 生成 survey.md 后，A 发起后验 |

---

## 1. knowledge_pipeline_worker

### 调用方式

```python
from harness.tool_registry import ToolRegistry
result = tool_registry.run("knowledge_pipeline_worker", "requests/knowledge_build_request.json")
```

或直接调用：

```python
from tools.knowledge_pipeline_worker import run
result = run("requests/knowledge_build_request.json")
```

### 输入：KnowledgeBuildRequest

文件路径：`requests/knowledge_build_request.json`

```json
{
  "task_id": "task_world_model_001",
  "topic": "世界模型",
  "request_type": "knowledge_build",
  "inputs": {
    "task_request_path": "cache/task_request.json",
    "search_strategy_path": "cache/search_strategy.json"
  },
  "outputs": {
    "knowledge_bundle_path": "cache/knowledge_bundle.json",
    "retrieved_papers_path": "cache/retrieved_papers.json",
    "parsed_papers_path": "cache/parsed_papers.json",
    "figure_bank_path": "cache/figure_bank.json",
    "table_bank_path": "cache/table_bank.json",
    "paper_cards_path": "cache/paper_cards.json",
    "evidence_store_path": "cache/evidence_store.json",
    "taxonomy_path": "cache/taxonomy.json",
    "citation_index_path": "cache/citation_index.json"
  },
  "pipeline_config": {
    "use_online_search": true,
    "use_seed_fallback": true,
    "use_mineru": false,
    "use_mock_mineru_if_failed": false,
    "use_influence_score": true,
    "aspect_match_threshold": 0.6
  },
  "quality_requirements": {
    "min_total_papers": 10,
    "min_core_papers": 5,
    "min_figures": 1,
    "min_evidence_per_core_paper": 1
  }
}
```

### 前置依赖（A 负责生成）

| 文件 | 说明 |
|------|------|
| `cache/task_request.json` | 任务描述 |
| `cache/search_strategy.json` | 搜索策略（sub_domains、wide_search） |
| `cache/seed_papers.json` | 种子论文库（可选，fallback 用） |
| `cache/surveys.json` | 已知综述列表（可选） |

### 输出：9 个 artifact 文件

| 文件 | Model | 供谁消费 |
|------|-------|----------|
| `cache/knowledge_bundle.json` | KnowledgeBundle | A（验证） |
| `cache/retrieved_papers.json` | RetrievedPapers | 内部 |
| `cache/parsed_papers.json` | ParsedPapers | 内部 |
| `cache/figure_bank.json` | FigureBank | C（图文引用） |
| `cache/table_bank.json` | TableBank | C（表格引用） |
| `cache/paper_cards.json` | PaperCards | A（prelock）、C（写综述） |
| `cache/evidence_store.json` | EvidenceStore | C（evidence panel）、B（后验） |
| `cache/taxonomy.json` | Taxonomy | C（综述结构） |
| `cache/citation_index.json` | CitationIndex | A（prelock） |

### 返回值

```python
{
    "status": "success" | "partial_success" | "failed",
    "outputs": ["cache/retrieved_papers.json", ...],
    "metrics": {
        "paper_count": 40,
        "citation_count": 120,
    },
    "message": "knowledge_pipeline_worker: success (40 papers, 120 citations)"
}
```

---

## 2. verify_citations

### 调用方式

```python
result = tool_registry.run("verify_citations", "requests/verification_request.json")
```

### 输入：VerificationRequest

文件路径：`requests/verification_request.json`

```json
{
  "task_id": "task_world_model_001",
  "inputs": {
    "survey_markdown_path": "output/survey.md",
    "paper_cards_path": "cache/paper_cards.json",
    "citation_ready_set_path": "cache/citation_ready_set.json",
    "evidence_store_path": "cache/evidence_store.json",
    "figure_bank_path": "cache/figure_bank.json",
    "table_bank_path": "cache/table_bank.json",
    "generated_artifact_bank_path": "cache/generated_artifact_bank.json"
  },
  "outputs": {
    "citation_result_path": "output/citation_result.json",
    "claim_map_path": "cache/claim_map.json"
  }
}
```

### 前置依赖

| 文件 | 由谁产出 |
|------|----------|
| `output/survey.md` | C（write_survey） |
| `cache/citation_ready_set.json` | A（citation prelock） |
| `cache/evidence_store.json` | B（knowledge_pipeline_worker） |
| `cache/figure_bank.json` | B |
| `cache/table_bank.json` | B |
| `cache/generated_artifact_bank.json` | C（write_survey） |

### 输出

| 文件 | 说明 |
|------|------|
| `output/citation_result.json` | 引用合法性检查结果 |
| `cache/claim_map.json` | Claim→Evidence 映射 |

### 返回值

```python
{
    "status": "success" | "partial_success",
    "outputs": ["output/citation_result.json", "cache/claim_map.json"],
    "metrics": {
        "total_citations": 38,
        "valid_citations": 36,
        "invalid_citations": 2,
        "citation_validity_score": 0.947,
        "total_claims": 25,
        "supported_claims": 20,
        "weak_claims": 3,
        "unsupported_claims": 2,
    },
    "message": "verify_citations: partial_success (citation_score=0.947, 2 unsupported claims)"
}
```

---

## 3. 注意事项

### Rate Limit

Intern-S2-Preview 限制 1 req / 2s。`InternS2Client` 内置 2.1s 节流，无需调用方处理。40 篇 paper 的 card 生成约需 ~80s 纯等待。

### MinerU（默认关闭）

`use_mineru: false` 是默认值。当前 arxiv PDF 有防爬限制，MinerU 100% 返回空壳。Pipeline 通过 SciVerse agentic-search backfill 确保 evidence 完整性，不依赖 MinerU。

若需开启（未来有可靠 PDF 源时）：

```json
"pipeline_config": {"use_mineru": true}
```

### Seed Fallback

若 SciVerse API 无法访问，`use_seed_fallback: true` 会使用 `cache/seed_papers.json`（102 篇预检索论文）作为兜底。

### Paper Influence Score

`use_influence_score: true` 是默认值。P3 会按年份分段设置 citation 门槛召回，并在本地按 SciVerse 返回顺序、引用数、年份和元数据完整度 rerank 后再截断 `max_papers`。

### 超时

默认 tool timeout = 300s（5min）。全真实 API 下 knowledge_pipeline_worker 约需 2-9 分钟（取决于论文数量和 LLM 响应速度）。可通过 `ToolRegistry` 构造参数调整。

### Anti-Hallucination 保障

1. 所有论文元数据来自 SciVerse API，不由 LLM 生成
2. Shallow cards 从 abstract 提取 claims（LLM 辅助结构化）
3. Agentic-search backfill: 无 parsed-paper evidence 的 claim 通过 SciVerse agentic-search 补充可引用 chunk
4. NLI verifier 确认 chunk 与 claim 的语义对齐
5. 后验 verify_citations 做双重检查
