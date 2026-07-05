# Module B 架构设计：Knowledge Pipeline Worker

> 日期：2026-07-05（v3：同步实现状态，移除 Phase 4 RAG，加入 agentic-search backfill）
> 状态：**已实现，集成测试通过**

## 0. 设计决策汇总

| 决策点 | 选择 | 理由 |
|---|---|---|
| **交付形式** | `tool_registry.run(name, request_path) -> dict` | 确定性 agent loop 编排，无 tool_calls 往返延迟 |
| 对外接口 | 2 个 tool（`knowledge_pipeline_worker` + `verify_citations`） | 粗粒度、单请求进单结果出，harness 可复现 |
| 文献结构 | 两层（Layer 1 综述 + Layer 2 方法论文） | Layer 1 提供"地图"，Layer 2 填充"肉" |
| 向量检索（Phase 4 RAG） | **不采用**（用 SciVerse agentic-search 替代） | 本地 ChromaDB RAG 为死代码：建索引但下游 Phase 5 从未查询；agentic-search 直接返回带 page_no/doc_id 的可引用 chunk，无需自建向量库即可完成 claim grounding |
| MinerU 策略 | **默认关闭**（`use_mineru: false`） | arxiv PDF 防爬 → 100% 空壳；abstract + agentic-search chunk 已足够 |
| SciVerse 策略 | meta-search 主检索 + agentic-search evidence backfill | 前者给论文列表，后者补 claim grounding |
| 关键词翻译 | Phase 3 LLM 翻译（中→英学术查询） | SciVerse 对英文查询效果远优于中文 |
| Paper Cards | deep（全文）+ **shallow（仅 abstract）** | MinerU 关闭时仍能产出 cards → evidence chain 不断 |
| Evidence 来源 | parsed paragraphs + **agentic-search chunk backfill** | 无 MinerU 时 backfill 是唯一 evidence 来源 |
| 引用校验 | NLI 确定性判断 + LLM 边缘 case 回退 | 零成本、可复现；仅模糊区间回退 LLM |
| 数据清洗 | SurveyX Cleaner 逻辑（容错正则） | MinerU 输出格式杂乱 |
| 论断生成 | LiRA 维度分解（4 bucket + 编号列表） | 原子化论断 + NLI = 确定性匹配 |
| Rate Limit | InternS2Client 内置 2.1s 节流 | Intern-S2 限制 1 req/2s |
| Request Timeout | 120s（可通过 `REQUEST_TIMEOUT_SECONDS` env 配置） | Intern-S2 偶发慢响应 |

---

## 1. 外部接口

B 通过 `harness/tool_registry.py` 注册 **2 个粗粒度 tool**，由 `harness/agent_loop.py` 确定性编排调用。

### Tool 1：knowledge_pipeline_worker

```python
# 签名
def run(request_path: str) -> dict:
    """Build complete KnowledgeBundle (Phases 1-6)."""
```

**输入**：`requests/knowledge_build_request.json`（KnowledgeBuildRequest schema）
**输出**：9 个 artifact 文件 + 返回 `{status, outputs, metrics, message}`

### Tool 2：verify_citations

```python
def run(request_path: str) -> dict:
    """Verify survey citations (structural) + map claims to evidence (NLI)."""
```

**输入**：`requests/verification_request.json`
**输出**：`output/citation_result.json` + `cache/claim_map.json`

详细 I/O schema 见 `docs/模块B-工具调用指南.md`。

---

## 2. 内部架构总览

```
knowledge_pipeline_worker.run(request_path)
│
├─ Phase 1: Demand Decomposition（需求拆解）
│   search_strategy → aspects + constraints + coverage validation
│   纯启发式，不调 LLM
│
├─ Phase 2: Survey Analysis（综述分析层）
│   surveys.json + search_aspects
│   → Taxonomy Self-Refine Step 1-2（LLM）
│   → expansion_candidates
│
├─ Phase 3: Method Paper Retrieval（方法论文检索）
│   SciVerse meta-search（per aspect）
│   → LLM 关键词翻译（中文 → 英文学术查询）
│   → PDF URL 过滤（仅实际 PDF 送 MinerU，默认跳过）
│   → 空壳检测（MinerU 返回空内容 → degrade to abstract_only）
│   → seed fallback（API 不可用时）
│   输出：retrieved_papers + parsed_papers
│
│
├─ Phase 5: Knowledge Synthesis（知识整合）
│   ├─ 5.1 Paper Cards（deep + shallow）
│   │   deep: parsed papers 全文 + LLM 提取
│   │   shallow: retrieved papers abstract + LLM 提取
│   ├─ 5.2 EvidenceStore
│   │   parsed paragraphs + agentic-search backfill
│   │   NLI 验证 chunk↔claim 语义对齐
│   ├─ 5.3 FigureBank / TableBank（从 parsed papers 提取）
│   ├─ 5.4 Taxonomy Self-Refine Step 3（LLM + 论文分布调整）
│   └─ 5.5 CitationIndex
│
└─ Phase 6: Bundle Assembly（打包交付）
    knowledge_bundle.json → 返回给 A
```

## 2.1 各阶段输入输出

每个 phase 是 `tools/phases/` 下的纯函数；编排顺序见 `tools/knowledge_pipeline_worker.py:run`。

**总体（worker.run）**

- 入口：`run(request_path) -> dict`
- 输入：`knowledge_build_request.json` + `search_strategy.json`（A 产出）+ `cache/surveys.json`（Layer 1 综述库）+ `cache/seed_papers.json`（102 篇，API 不可用时 fallback）
- 输出：9 个 artifact JSON + 返回 `{status, outputs, metrics, message}`
- 下游：`knowledge_bundle.json` → A（校验）+ C（写综述）

**逐阶段**

| Phase | 模块 | 输入 | 做什么 | 输出 |
|---|---|---|---|---|
| P1 需求拆解 | `phase1_decompose` | request, strategy, seed_papers | 纯启发式，**不调 LLM**：校验 aspects 结构 + 覆盖 | `DecomposedDemand`（aspects/constraints/structure_errors/coverage_warnings） |
| P2 综述分析 | `phase2_survey_analyzer` | topic, sub_domains, aspects(P1), surveys, llm | 分析 Layer 1 综述骨架 + **Taxonomy Self-Refine Step 1-2**（LLM：prelim→refined） | `SurveyStructure`（refined_taxonomy + expansion_candidates） |
| P3 论文检索 | `phase3_paper_retriever` | aspects(P1), expansion_candidates(P2), sciverse/mineru/cleaner/seed, pipeline_config, llm | **LLM 关键词翻译（中→英）** + SciVerse meta-search per aspect + PDF URL 过滤 + MinerU 解析 + 空壳检测 + seed fallback | `RetrievedPapers` + `ParsedPapers` |
| P5.1 Cards | `phase5_cards` | parsed/retrieved(P3), aspects, llm | deep（parsed 全文）+ shallow（retrieved abstract）→ LLM 提取 4 维度 claims | `PaperCards` |
| P5.2 Evidence | `phase5_evidence` | parsed(P3), cards(P5.1), nli, sciverse | parsed 段落/abstract/caption → evidence + **agentic-search backfill**（无 parsed evidence 的 claim 用 SciVerse chunk 补，NLI 验证 chunk↔claim） | `EvidenceStore` |
| P5.3-5 Synthesis | `phase5_synthesis_rest` | parsed(P3), refined_taxonomy(P2), cards(P5.1), llm | FigureBank / TableBank / Taxonomy（归并 + 按 keyword overlap 把 card 挂到 category）/ CitationIndex | `FigureBank`/`TableBank`/`Taxonomy`/`CitationIndex` |
| P6 打包 | `phase6_bundle_assembler` | 全部上游 artifact + structure_errors/coverage_warnings + quality_requirements | 聚合 + 状态判定（success/partial_success/failed）+ quality_report | `KnowledgeBundle` |

依赖链：`strategy → P1 → P2 → P3 → P5.* → P6`。P5.1–P5.5 在 P3 之后相互基本独立（见接口文档 §7.2 并行说明）。Phase 编号沿用代码模块名；无 Phase 4（向量检索 RAG 不采用，见 §0 决策表）。

---

## 3. 关键实现细节

### 3.1 LLM 关键词翻译（Phase 3）

SciVerse 对英文学术查询效果好，中文 topic 直接搜索会返回不相关结果。

```python
# tools/phases/phase3_paper_retriever.py
def _generate_search_queries(keywords: list[str], llm) -> list[str]:
    """中文关键词 → 2-3 条英文学术查询"""
    prompt = f"Translate keywords into English academic search queries: {' '.join(keywords)}"
    result = llm.chat([{"role": "user", "content": prompt}])
    return [line.strip() for line in result.splitlines() if line.strip()][:3]
```

### 3.2 Shallow Cards（Phase 5.1）

MinerU 关闭时 parsed_papers 为空，但 retrieved papers 有 abstract。Shallow cards 从 abstract 提取 claims，确保 agentic-search backfill 有 claims 可接。

```python
# tools/phases/phase5_cards.py
def build_shallow_card(retrieved, llm, aspects, threshold=0.6):
    """Card from abstract only (no parsed body)."""
    raw = llm.chat([...CARD_PROMPT with title + abstract...])
    claims = parse_card_response(raw, retrieved.paper_id)
    return PaperCard(..., card_type="shallow", possible_claims=claims)
```

### 3.3 Agentic-Search Backfill（Phase 5.2）

对未被 parsed-paper evidence 覆盖的 claims，调用 SciVerse agentic-search 获取可引用 chunk。

```python
# tools/phases/phase5_evidence.py
def _agentic_backfill(paper_cards, evidences, sciverse, nli):
    supported = {claim_text for e in evidences for s in e.supports_claims}
    for card in paper_cards:
        for claim in card.all_claims():
            if claim.text in supported:
                continue
            hits = sciverse.agentic_search(claim.text, top_k=3)["hits"]
            for hit in hits:
                # NLI 验证 chunk↔claim
                if nli.best_match(hit["chunk"], [claim.text]).support_type != "contradictory":
                    evidences.append(Evidence(source_type="agentic_chunk", ...))
```

### 3.4 Rate Limiting（LLM Client）

```python
# llm_client.py
class InternS2Client:
    _min_interval = 2.1  # Intern-S2: 1 req per 2s
    # 每次调用前检查 elapsed，不足则 sleep
```

### 3.5 Thinking Mode（LLM Client，全部关闭）

`InternS2Client.chat()` 暴露 `thinking_mode` 参数（默认 `False`），作为 payload 字段发给
Intern-S2。模块 B 是确定性流水线，所有 LLM 调用都是结构化抽取/翻译/分类这类"快"任务，
**不需要 thinking**，且 Intern-S2 受速率限制（1 req / 2s），thinking 会显著拖慢链路却无收益。

| 调用点 | 用途 | thinking_mode |
|---|---|---|
| `phase2_survey_analyzer.py`（prelim + refine） | 构建 taxonomy | 默认 False |
| `phase3_paper_retriever.py` | 中文关键词 → 英文 query | 默认 False |
| `phase5_cards.py` | 抽取 paper card | 显式 `CARD_THINKING_MODE = False` |
| `verify/claim_mapper.py` | claim 映射辅助 | 默认 False |

新增 B 内 LLM 调用时默认保持 `thinking_mode=False`；确需推理（如复杂的综合改写）再显式打开。

---

## 4. 后验验证阶段

C 写完 `survey.md` 后，A 调用 `verify_citations` 做双重检查：

### 4.1 Structural Verification

检查 `[paper_id]` ∈ citation_ready_set，`![fig](id)` ∈ figure_bank。

### 4.2 Claim Mapping（NLI-first）

```
阶段 1：精确匹配 EvidenceStore 预存的 supports_claims
阶段 2：NLI DeBERTa 蕴含判断（confidence ≥ 0.6 → supported）
阶段 3：LLM 回退（仅 confidence 0.4-0.6 的边缘 case）
```

---

## 5. 文件结构（当前实现）

```text
tools/
├── knowledge_pipeline_worker.py   # Tool 1 入口（Phase 1-6 编排）
├── verify_citations.py            # Tool 2 入口（后验验证）
├── phases/
│   ├── phase1_decompose.py
│   ├── phase2_survey_analyzer.py
│   ├── phase3_paper_retriever.py  # LLM 翻译 + PDF 过滤 + 空壳检测
│   ├── phase5_cards.py            # deep + shallow cards
│   ├── phase5_evidence.py         # parsed evidence + agentic backfill
│   ├── phase5_synthesis_rest.py   # FigureBank / TableBank / Taxonomy / CitationIndex
│   └── phase6_bundle_assembler.py
├── verify/
│   ├── structural.py              # §4.1 结构性验证
│   └── claim_mapper.py            # §4.2 NLI claim mapping
├── clients/
│   ├── sciverse_client.py         # SciVerse API (meta-search + agentic-search)
│   ├── mineru_client.py           # MinerU PDF 解析（默认关闭）
│   └── llm_fake.py                # 测试用 fake LLM
├── nlp/
│   ├── nli_verifier.py            # NLI 校验
│   └── data_cleaner.py            # MinerU 输出清洗
└── models/
    ├── requests.py                # KnowledgeBuildRequest, PipelineConfig
    ├── artifacts.py               # RetrievedPapers, ParsedPapers, PaperCards, EvidenceStore, ...
    ├── bundle.py                  # KnowledgeBundle
    └── common.py                  # ID 生成工具

cache/
├── seed_papers.json               # 102 篇预检索种子论文
├── surveys.json                   # 综述库
├── retrieved_papers.json          # Phase 3 输出
├── parsed_papers.json             # Phase 3 输出
├── paper_cards.json               # Phase 5 输出
├── evidence_store.json            # Phase 5 输出
├── figure_bank.json               # Phase 5 输出
├── table_bank.json                # Phase 5 输出
├── taxonomy.json                  # Phase 5 输出
├── citation_index.json            # Phase 5 输出
└── knowledge_bundle.json          # Phase 6 输出
```

---

## 6. Anti-Hallucination 保障链

```
SciVerse meta-search → 论文元数据（title/authors/year/venue）
                        ↓ 不由 LLM 生成
LLM → 从 abstract 提取 claims（结构化，4 维度 bucket）
                        ↓
SciVerse agentic-search → 可引用 text chunk（page_no + doc_id）
                        ↓
NLI DeBERTa → 验证 chunk↔claim 语义对齐
                        ↓
verify_citations → 综述引用 ∈ citation_ready_set
                 → claim→evidence NLI 蕴含判断
```

每层独立验证，任一层失败 → 标记 unsupported → C 必须修改或删除。

---

## 7. 测试状态

| 测试 | 状态 | 耗时 |
|------|------|------|
| 单元测试 (149) | ✅ passed | 0.3s |
| 集成测试 (real SciVerse + fake LLM) | ✅ passed | 69s |
| 全真实 API (Intern-S2 + SciVerse) | ⚠️ timeout 120s 应修复 | ~9min |

---

## 8. 待办优化

- [ ] MinerU：找到绕过 arxiv 防爬的 PDF 源后开启（`use_mineru: true`）
- [ ] 并行化：多 aspect 并发调用 SciVerse（当前串行）
- [ ] NLI 本地模型：cross-encoder/nli-deberta-v3-base 替代 FakeNLI
- [ ] Paper Influence Score：综合引用 + 时效 + 开源影响力排序
