# Module B 架构设计：Knowledge Pipeline Worker

> 日期：2026-07-04（v2：加入 tool call 交付形式 + SurGE/SurveyX/LiRA 复用）
> 状态：设计稿（待实现）

## 0. 设计决策汇总

| 决策点 | 选择 | 理由 |
|---|---|---|
| **交付形式** | **书生 API tool call 工具** | Intern-S2-Preview 原生支持 tool_calls，B 封装为粗粒度 tool，模型做编排大脑 |
| 对外接口 | 3 个 tool（build_knowledge / verify_citations / search_papers） | 粗粒度避免 tool call 往返延迟（tools 不支持流式） |
| 文献结构 | 两层（Layer 1 综述 + Layer 2 方法论文） | Layer 1 提供"地图"，Layer 2 填充"肉" |
| 向量检索 | Phase 4 构建，专供 Phase 5 语义检索 | evidence-to-claim 匹配、aspect 内排序 |
| Embedding | 外部 API（如 OpenAI），可配置 | 质量高，通过接口可切换 |
| MinerU 策略 | Layer 1 🎯全文 / 核心 ⚡轻量(前2-3页) / 扩展 ⚡轻量(可选) | 平衡质量和 API 调用量 |
| SciVerse 策略 | meta-search 主检索 + agentic-search 补充 | freshness_boost/impact_boost 替代本地分段 top-k |
| B→C 产出 | Paper Cards + Taxonomy + Claim Map + Figure 描述/数据 | 遵循接口文档的 artifact 规范 |
| Phase 1 验证 | 基于 A 的 sub_domains，不硬编码固定方向 | 用户 topic 可能只涉及 1-2 个方向，硬编码 6 方向会过约束 |
| 综述库更新 | 静态 + 按需扩展，渐进增长 | 每次运行可搜索新综述并评估价值，但不强制修改本地库 |
| Taxonomy 骨架 | Self-Refine（先独立生成，再参考综述修正） | 避免完全跟随已有综述的分类偏差，借鉴 SurveyForge 的启发式大纲学习 |
| **引用校验** | **NLI DeBERTa 确定性判断（复用 SurGE）** | 替代 LLM 仲裁，零成本、可复现；仅边缘 case 回退 LLM |
| **数据清洗** | **SurveyX Cleaner 逻辑（容错正则提取）** | MinerU 输出格式杂乱，容错正则省去手工调试 |
| **论断生成** | **LiRA 维度分解（固定 bucket + 编号列表）** | 原子化论断 + NLI = Claim Mapper 变确定性匹配 |

---

## 1. 外部接口（书生 API Tool Call）

B 封装为 **3 个粗粒度 tool**，通过书生 Chat API 的 `tools` 参数暴露给 Intern-S2-Preview 模型。模型在 agent loop 中按需调用，B 本地执行后通过 `role: tool` 回传结果。

> **为什么粗粒度**：书生 API 使用 `tools` 时不支持流式输出，细粒度工具会导致大量非流式往返。每个 tool 内部封装完整的 Phase 逻辑，模型只需做"什么时候调"的决策。

### Tool 1：search_papers（论文检索）

对应 Phase 1-4 的检索+解析流程。

```python
{
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "根据关键词和年份范围检索相关论文，返回论文元数据列表。内部完成：需求拆解 → 综述分析 → 方法论文检索 → PDF解析 → RAG索引构建。",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "综述主题"},
                "keywords": {"type": "string", "description": "检索关键词，逗号分隔"},
                "year_range": {"type": "array", "items": {"type": "integer"}, "description": "年份范围 [start, end]"}
            },
            "required": ["topic", "keywords"]
        }
    }
}
```

**内部执行**：Phase 1 拆解 → Phase 2 综述分析 → Phase 3 检索+MinerU解析 → Phase 4 RAG索引
**返回**：`retrieved_papers.json` + `parsed_papers.json` 的摘要（top-N 论文列表 + 统计信息）

### Tool 2：build_paper_cards（知识整合）

对应 Phase 5-6 的知识合成+打包流程。

```python
{
    "type": "function",
    "function": {
        "name": "build_paper_cards",
        "description": "对已检索的论文列表逐篇解析，提取结构化 Paper Card（含原子化论断、方法摘要、局限性），构建 EvidenceStore、Taxonomy、CitationIndex，打包为 KnowledgeBundle。内部使用 NLI 模型做论断校验。",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"}, "description": "需要处理的论文 ID 列表（来自 search_papers 返回结果）"},
                "focus_aspects": {"type": "array", "items": {"type": "string"}, "description": "重点关注的方向（可选，用于 taxonomy 分类引导）"}
            },
            "required": ["paper_ids"]
        }
    }
}
```

**内部执行**：Phase 5.1-5.5 知识合成 → Phase 6 打包
**返回**：`knowledge_bundle.json`（含 paper_cards + taxonomy + citation_index + quality_report）

### Tool 3：verify_citations（引用校验）

对应后验验证阶段（C 写完综述后）。

```python
{
    "type": "function",
    "function": {
        "name": "verify_citations",
        "description": "对综述草稿中的引用声明进行结构性和归属性校验。先用确定性检查验证 paper_id 是否存在于知识库，再用 NLI 模型验证论断与论文内容的一致性。返回每条引用的校验结果和置信分数。",
        "parameters": {
            "type": "object",
            "properties": {
                "draft_text": {"type": "string", "description": "待校验的综述草稿文本（markdown）"}
            },
            "required": ["draft_text"]
        }
    }
}
```

**内部执行**：§4.2 Citation Verifier（结构性）→ §4.3 Claim Mapper（NLI 为主，LLM 仅边缘 case）
**返回**：`citation_result.json` + `claim_map.json`

### Tool Call 编排模式

```python
# 书生 API 编排示例（简化）
tools = [search_papers_tool, build_paper_cards_tool, verify_citations_tool]

# 第一轮：模型决定调用 search_papers
res = client.chat.completions.create(
    model="intern-s2-preview",
    thinking_mode=True,    # 智能体场景必须开启
    messages=messages,
    tools=tools,
    stream=False,           # tools 时不支持流式
)

if res.choices[0].finish_reason == "tool_calls":
    for call in res.choices[0].message.tool_calls:
        args = json.loads(call.function.arguments)
        result = tool_executor.dispatch(call.function.name, args)  # 本地执行 B 的逻辑
        messages.append({"role": "tool", "content": json.dumps(result), "tool_call_id": call.id})

    # 第二轮：模型基于工具结果继续规划
    res2 = client.chat.completions.create(model="intern-s2-preview", messages=messages, tools=tools, ...)
```

### tool_choice 流控

| 阶段 | tool_choice | 说明 |
|---|---|---|
| 检索阶段 | `required` 指向 `search_papers` | 强制模型先检索再生成 |
| 整合阶段 | `required` 指向 `build_paper_cards` | 强制构建知识库 |
| 校验阶段 | `required` 指向 `verify_citations` | 强制引用校验 |
| 生成阶段 | `auto` 或 `none` | 模型自由生成综述文本，不再触发 tool |

**所有中间 artifact 的 schema 严格遵循接口文档 §9。**

---

## 2. B 内部架构总览

```
knowledge_pipeline_worker.run(request)
│
├─ Phase 1: Demand Decomposition（需求拆解）
│   解析 search_strategy → aspects + 约束 + 覆盖度验证
│   不调 LLM，纯启发式
│
├─ Phase 2: Layer 1 — Survey Analysis（综述分析层）
│   输入：data/surveys.json + search_aspects
│   ├─ 2.1 综述库更新（按需搜索 + 价值评估 + 渐进增长）
│   ├─ 2.2 Taxonomy Self-Refine Step 1-2
│   │     Step 1: LLM 独立生成 preliminary_taxonomy（不看综述）
│   │     Step 2: LLM 参考综述修正 → refined_taxonomy
│   └─ 2.3 提取引用链 → expansion_candidates
│   输出：survey_structure.json（含 preliminary + refined taxonomy）
│
├─ Phase 3: Layer 2 — Method Paper Retrieval（方法论文检索层）
│   输入：expansion_candidates + search_aspects + SciVerse + seed fallback
│   处理：扩展检索 → MinerU 按需解析（deep 全文 / light abstract）
│   输出：retrieved_papers.json + parsed_papers.json
│          │
│          └──→ Phase 5 Taxonomy Step 3 的论文分布输入
│
├─ Phase 4: RAG Index Construction（向量索引构建）
│   输入：Layer 1 + Layer 2 的全部论文内容
│   处理：Embedding API → 向量库 + 关键词索引
│   输出：rag_index/
│
├─ Phase 5: Knowledge Synthesis（知识整合）
│   输入：survey_structure + retrieved_papers + parsed_papers + RAG 索引
│   ├─ 5.1 Paper Cards 构建（LLM 从论文内容提取）
│   ├─ 5.2 EvidenceStore 构建
│   ├─ 5.3 FigureBank / TableBank 构建（元数据 + generation_hint）
│   ├─ 5.4 Taxonomy Self-Refine Step 3
│   │     LLM 结合论文分布调整 → final_taxonomy → taxonomy.json
│   └─ 5.5 CitationIndex 构建
│   输出：paper_cards.json + evidence_store.json + figure_bank.json +
│         table_bank.json + taxonomy.json + citation_index.json
│
└─ Phase 6: Bundle Assembly（打包交付）
    输出：knowledge_bundle.json → 返回给 A
```

### 数据流全景

```
                    Intern-S2-Preview（编排大脑）
                          │
              tool_calls: search_papers
                          │
                          ▼
              ┌─── B 模块本地执行 ─────────────────────────────────────────┐
              │                                                           │
              │  Phase 1: 需求拆解 + 覆盖度验证（纯启发式）                │
              │    search_strategy → aspects + constraints                │
              │    验证 sub_domains ↔ aspects 一致性                       │
              │                       │                                   │
              │                       ▼                                   │
              │  Phase 2: Layer 1 — 综述分析                             │
              │    data/surveys.json（按需搜索扩展，渐进增长）               │
              │    → MinerU 全文解析 → SurveyX Cleaner 清洗               │
              │    → Taxonomy Self-Refine Step 1-2                        │
              │      Step 1: LLM 独立生成 preliminary_taxonomy           │
              │      Step 2: LLM + 综述修正 → refined_taxonomy           │
              │    → 引用链 → expansion_candidates                       │
              │    → survey_structure.json                               │
              │                       │                                   │
              │                       ▼                                   │
              │  Phase 3: Layer 2 — 方法论文检索                          │
              │    引用链扩展 + SciVerse + seed fallback                  │
              │    → MinerU 按需解析（deep/light）                        │
              │    → SurveyX Cleaner 清洗 → retrieved + parsed           │
              │                       │                                   │
              │          ┌────────────┘                                   │
              │          ▼                                                │
              │  Phase 4: RAG Index                                       │
              │    Embedding API → ChromaDB 向量库 + 关键词索引           │
              │    → rag_index/                                           │
              └───────────────────────────────────────────────────────────┘
                          │
              role: tool → 返回论文列表摘要
                          │
                          ▼
                    Intern-S2-Preview
                          │
              tool_calls: build_paper_cards
                          │
                          ▼
              ┌─── B 模块本地执行 ─────────────────────────────────────────┐
              │  Phase 5: Knowledge Synthesis                              │
              │    → paper_cards.json     (LiRA 维度分解原子论断)          │
              │    → evidence_store.json  (NLI 自动填写 supports_claims)   │
              │    → figure_bank.json     (图表元数据+generation_hint)     │
              │    → table_bank.json      (表格元数据)                     │
              │    → Taxonomy Self-Refine Step 3                          │
              │      LLM + 论文分布调整 → final_taxonomy                  │
              │      → taxonomy.json                                      │
              │    → citation_index.json  (引用索引)                      │
              │                       │                                   │
              │                       ▼                                   │
              │  Phase 6: Bundle Assembly                                  │
              │    → knowledge_bundle.json                                 │
              └───────────────────────────────────────────────────────────┘
                          │
              role: tool → 返回 knowledge_bundle
                          │
                          ▼
                    Intern-S2-Preview
              （模型基于 knowledge_bundle 生成综述 → C 写 survey.md）
                          │
              tool_calls: verify_citations
                          │
                          ▼
              ┌─── B 模块本地执行 ───────────────────────────┐
              │  §4.2 Citation Verifier（结构性，确定性检查）  │
              │  §4.3 Claim Mapper（NLI DeBERTa 为主）       │
              │    → citation_result.json + claim_map.json     │
              └──────────────────────────────────────────────┘
                          │
              role: tool → 返回校验结果
                          │
                          ▼
                    Intern-S2-Preview
              （通过 → 输出最终综述 / 不通过 → 要求 C 修订 → 再次 verify）
```

---

## 3. 各 Phase 详解

### Phase 1: Demand Decomposition（需求拆解）

**输入**：`KnowledgeBuildRequest`（内含 `search_strategy.json` 路径）

**做什么**：
- 加载 `search_strategy.json`
- 提取 `search_aspects`（A 已生成的搜索方向列表）
- 提取 `wide_search` 和 `deep_search` 的约束（max_papers、时间范围等）
- 提取 `pipeline_config`（是否使用在线搜索、seed fallback、MinerU 等）
- 检查本地已有资源（surveys.json、seed_papers.json）与 aspects 的匹配度

**输出**：内部数据结构 `DecomposedDemand`（不写文件，直接传入 Phase 2/3）

```python
@dataclass
class DecomposedDemand:
    aspects: list[SearchAspect]       # 搜索方向 + keywords + min_papers
    constraints: SearchConstraints   # max_papers, time_range, etc.
    pipeline_config: PipelineConfig   # use_online, use_seed, use_mineru, ...
    local_coverage: CoverageReport   # 本地资源与 aspects 的匹配情况
    phase1_validation: ValidationReport  # 验证结果
```

**验证逻辑**（分两层）：

**第一层：结构完整性（确定性，必须通过）**

```python
def validate_structure(search_strategy):
    errors = []
    if len(search_strategy.wide_search.search_aspects) < 3:
        errors.append("aspects 少于 3 个，覆盖可能不足")
    for aspect in search_strategy.wide_search.search_aspects:
        if not aspect.keywords:
            errors.append(f"{aspect.aspect_id} 缺少 keywords")
    tr = search_strategy.wide_search.time_range
    if tr.end_year - tr.start_year < 3:
        errors.append("时间跨度小于 3 年，可能遗漏经典工作")
    return errors
```

**第二层：覆盖度检查（启发式，借鉴 SurGE Comprehensiveness）**

不硬编码固定方向（如 6 大类），而是**基于 A 自己声明的 sub_domains** 验证一致性：

```python
def validate_coverage(search_strategy, seed_papers, topic_understanding):
    warnings = []

    # 核心逻辑：A 声明了哪些 sub_domains，检查是否都有对应 aspect
    sub_domains = topic_understanding.sub_domains
    aspects = search_strategy.wide_search.search_aspects

    for domain in sub_domains:
        matched = any(
            domain.lower() in a.aspect_name.lower()
            or any(kw in domain.lower() for kw in a.keywords)
            for a in aspects
        )
        if not matched:
            warnings.append(f"A 声明的子领域 '{domain}' 没有对应 aspect")

    # seed_papers 是否都能匹配到至少一个 aspect
    for paper in seed_papers:
        paper_text = f"{paper['title']} {' '.join(paper.get('keywords', []))}".lower()
        matched = any(
            any(kw.lower() in paper_text for kw in a.keywords)
            for a in aspects
        )
        if not matched:
            warnings.append(f"seed paper 无法匹配任何 aspect: {paper['title']}")

    return warnings
```

> **设计原则**：验证的是"A 自己的分析是否自洽"，而不是"是否覆盖了某个固定的领域分类"。用户 topic 可能只涉及 1-2 个方向，硬编码 6 方向会过约束。
>
> **后续增强**：可用引用关系知识库（论文间的 citation graph）来验证 aspect 之间的相关性，检查是否有重要论文因为 aspect 划分不当而落在缝隙中。24h 内优先级不高，但架构上预留接口。

**要点**：
- Phase 1 是纯启发式的，不需要 LLM 调用（A 已经用 LLM 生成了 search_strategy）
- LLM 重活在 Phase 2（综述分析）和 Phase 5（知识整合）
- 验证失败产生 warnings，不阻断流程，但记录到 knowledge_bundle 的 quality_report 中

---

### Phase 2: Layer 1 — Survey Analysis（综述分析层）

**目的**：从已有综述论文中提取"地图"——分类法骨架、发展脉络、引用链

**输入**：
- `data/surveys.json`（本地综述库，初始 7 篇）
- search_aspects（从 Phase 1）
- `DecomposedDemand.local_coverage`（Phase 1 的覆盖度报告）

#### 2.1 综述库更新策略（静态 + 按需扩展）

本地综述库 `data/surveys.json` 采用**渐进增长**模式，不每次运行都重新搜索：

```
默认行为（pipeline_config.update_survey_store = false）：
  → 只读 data/surveys.json，不修改
  → 使用已有综述作为 Layer 1 输入

按需扩展（pipeline_config.update_survey_store = true）：
  → 搜索新综述 → 评估价值 → 追加到 data/surveys.json
```

**触发条件**（满足任一即搜索新综述）：

```text
1. Phase 1 验证发现覆盖度 warning（本地综述和当前 topic 匹配度低）
2. pipeline_config.use_online_search = true（默认行为，且 update_survey_store = true）
3. 本地综述中没有任何一篇匹配当前 topic 的 sub_domains
```

**新综述价值评估维度**：

```text
价值判断（满足任一即采纳）：
1. 相关性：综述标题/abstract 是否匹配当前 topic 的 sub_domains
2. 新颖性：是否覆盖了本地综述没有的分类维度或时间跨度
3. 时效性：是否比本地已有综述更新（近 1 年内优先）
4. 影响力：是否有一定引用量或来自顶级会议/期刊

评估方式：先用关键词过滤 + embedding 相似度粗筛，再用 LLM 判断是否纳入
```

**更新流程**：

```
SciVerse 搜索（topic + "survey" / "review" / "roadmap"）
  → 候选综述列表（title + abstract + year + venue）
  → 价值评估（4 维打分）
  → 通过阈值的候选 → MinerU 解析 → 结构化 → 追加到 surveys.json
  → 同时记录更新日志（新增了哪些，为什么）
```

#### 2.2 Taxonomy 骨架：Self-Refine 方法

不直接综合已有综述的分类建议，而是采用 **Self-Refine** 思路（借鉴 SurveyForge 的"启发式大纲学习"）：

```
Step 1: LLM 独立生成初步 Taxonomy（不看综述）
  输入：topic + search_aspects + sub_domains
  输出：preliminary_taxonomy（类别列表 + 描述 + 逻辑关系）
  目的：不受已有综述偏差影响，生成适配当前 topic 的分类框架

Step 2: LLM 参考综述修正
  输入：preliminary_taxonomy + survey_structure（多篇综述的 taxonomy_skeleton）
  输出：refined_taxonomy（合并 + 修正 + 去重）
  目的：用综述的领域知识弥补 Step 1 的不足，但不盲从

Step 3: LLM 结合实际论文分布调整
  输入：refined_taxonomy + retrieved_papers（Phase 3 初步检索结果）
  输出：final_taxonomy（检查空类别/过密类别，最终调整）
  目的：确保分类和实际论文分布一致
```

**对比直接综合**：

| | 直接综合 | Self-Refine |
|---|---|---|
| LLM 调用 | 1 次 | 2-3 次 |
| 偏差风险 | 高（完全跟随综述分类方式） | 低（先独立思考，再参考修正） |
| 对已有综述的依赖 | 强（综述是"权威"） | 弱（综述是"顾问"） |
| 24h 可行性 | ✅ | ✅（额外成本可接受） |

**为什么要先独立生成**：已有综述的分类法不一定适合当前 topic。比如 World Models 综述按"起承转合"分，但用户可能想要按"方法类型"分。先让 LLM 独立思考，再用综述知识修正，能产生更适合当前 topic 的分类框架。

#### 2.3 处理步骤

```
1. 加载 surveys.json + 按需搜索扩展
2. 对每篇综述：
   a. 如果有 PDF URL → 调 MinerU 全文解析
   b. 提取关键信息：
      - taxonomy_skeleton（已有的分类建议）
      - key_sections（章节结构）
      - top_referenced_papers（引用的核心论文列表）
      - 关键观点和论断
3. Self-Refine 生成 Taxonomy 骨架：
   a. LLM 独立生成 preliminary_taxonomy
   b. LLM 参考综述修正 → refined_taxonomy
   c. （Phase 3 后）LLM 结合论文分布调整 → final_taxonomy
4. 提取所有 top_referenced_papers → 形成扩展检索候选列表（或用 SciVerse `meta-paper-relations` 直接查询 REFERENCES）
```

**输出**：`survey_structure.json`

```json
{
  "task_id": "string",
  "survey_update_log": [
    {
      "action": "added",
      "paper_id": "survey_new_topic_2026",
      "reason": "覆盖了本地综述缺少的 'Benchmark' 维度，2026 年发表",
      "value_scores": { "relevance": 0.9, "novelty": 0.8, "recency": 0.9, "influence": 0.7 }
    }
  ],
  "analyzed_surveys": [
    {
      "paper_id": "survey_visual_world_roadmap_2025",
      "taxonomy_skeleton": ["Video generation foundations", "Interactive control mechanisms", "Game-oriented world models"],
      "key_sections": ["Video Generation Roadmap", "Interactive World Models", "GameCraft Applications"],
      "referenced_paper_ids": ["genie2_2024", "gamenegen_2024", "hunyuan_gamecraft_2025"],
      "key_claims": ["Interactive world models shift from internal planning to real-time simulators"]
    }
  ],
  "preliminary_taxonomy": [
    { "name": "...", "description": "...", "source": "llm_independent" }
  ],
  "refined_taxonomy": [
    { "name": "...", "description": "...", "source": "llm_refined", "incorporated_from": ["survey_xxx"] }
  ],
  "expansion_candidates": [
    { "paper_id_hint": "genie_2024", "source_survey": "survey_visual_world_roadmap_2025", "priority": "high" }
  ]
}
```

**要点**：
- MinerU 全文解析综述是为了提取比 abstract 更丰富的结构信息
- 综述库渐进增长，不每次运行都重新搜索——`update_survey_store` 控制
- Self-Refine 保证 taxonomy 适配当前 topic 而非简单复制已有分类
- `expansion_candidates` 是给 Phase 3 的检索起点
- `preliminary_taxonomy` 和 `refined_taxonomy` 都保留，便于调试和追溯

---

### Phase 3: Layer 2 — Method Paper Retrieval（方法论文检索层）

**目的**：获取具体的方法论文，填充 Layer 1 的骨架

**输入**：
- `expansion_candidates`（从 Phase 2）
- `search_aspects`（从 Phase 1）
- `cache/seed_papers.json`（本地兜底）
- `run_config`（是否在线搜索）

**检索策略**（按优先级）：

```
1. Layer 1 引用链扩展（最高优先）
   用 SciVerse meta-paper-relations 查询综述 REFERENCES
   批量通过 meta-search 获取候选论文 metadata

2. SciVerse meta-search 主检索
   每个 aspect 独立：query=keywords, filters=year/citation
   freshness_boost=MILD + impact_boost=MILD → 时序感知 + 高被引上浮
   详见下方 SciVerse API 调用模式

3. SciVerse agentic-search 补充
   meta-search 结果不足时，用自然语言提问补充
   返回片段可直接作为 evidence

4. Seed fallback
   如果在线搜索失败，使用本地 seed_papers.json

5. 去重合并
   按 unique_id / title+year 去重
   保留最高优先级的来源
```

**PDF 获取与解析**（分层策略）：

| 论文类型 | 解析深度 | MinerU API | 说明 |
|---|---|---|---|
| 综述论文（Layer 1） | 🎯 精准全文 | `/api/v4/extract/task`（异步） | 需要 Token，≤200MB，≤200页 |
| 核心方法论文（deep） | ⚡ 轻量前2-3页 | `/api/v1/agent/parse/url` | 无需 Token，IP 限频，≤10MB，≤20页 |
| 扩展论文（light） | ⚡ 轻量(可选) | `/api/v1/agent/parse/url` | IP 限频允许则解析 Intro，否则 abstract_only |

#### SciVerse API 调用模式

SciVerse 提供两套互补的检索 API，在 Pipeline 中的角色：

| 端点 | 用途 | Pipeline 步骤 |
|---|---|---|
| `POST /meta-search` | 结构化元数据检索（filter/sort/boost） | Phase 2 引用链 / Phase 3 主检索 |
| `POST /agentic-search` | 自然语言检索，返回片段 + doc_id | Phase 3 补充 / Phase 5 evidence |
| `GET /content?doc_id=xxx` | 按 doc_id 拉全文（offset/limit 分段） | Phase 3 补充（可选，减少 MinerU 依赖） |
| `GET /resource?file_name=xxx` | 下载图片等附件 | Phase 5 FigureBank |
| `POST /meta-paper-relations` | 查引用/被引/相关工作（分页） | Phase 2 引用链扩展 |

**meta-search 调用模式**（Phase 3 每个 aspect 独立执行）：

```python
response = sciverse_client.meta_search(
    query=" ".join(aspect.keywords),
    filters=[
        {"field": "publication_published_year", "value": {"gte": 2020, "lte": 2026}},
        {"field": "citation_count", "value": {"gte": 5}},
    ],
    sort=[{"field": "citation_count", "order": "SORT_ORDER_DESC"}],
    freshness_boost="MILD",   # 近 10 年加权，兼顾经典和新兴
    impact_boost="MILD",       # 高被引上浮，相关性仍主导
    page_size=25,
)
```

**时序感知排序**：`freshness_boost` + `impact_boost` 原生替代 SurveyForge 的本地时间段 top-k：
- `MILD`（推荐）：近 10 年高斯衰减 + 高被引轻度上浮
- `STRONG`：近 3 年 / 强偏向高被引，适合追踪前沿
- 两参数可叠加（都非 NONE = 相关 + 偏新 + 高被引），rescore 两阶段实现

**agentic-search 调用模式**（Phase 3 补充 / Phase 5 evidence）：

```python
response = sciverse_client.agentic_search(
    query=f"{aspect.aspect_name}: key methods in {topic}",
    top_k=10,
    filters={"publication_published_year": {"gte": 2020, "lte": 2026}, "lang": "en"},
)
# hits[].chunk → 可直接作为 evidence_text
# hits[].doc_id → 可传给 /content 拉取全文
```

**要点**：
- Phase 3 用 meta-search 做结构化检索（主），agentic-search 做语义补充（辅）
- `freshness_boost` / `impact_boost` 原生支持时序感知，无需本地分段排序
- agentic-search 返回的 `chunk` 可直接用于 EvidenceStore，减少 MinerU 依赖
- `/resource` 端点可直接获取论文图片，减少手动扒图
- `meta-paper-relations` 可自动化引用链提取，替代手工从综述中提取 referenced_papers
- `/content` 的 offset/limit 分段拉取适配长上下文，可替代部分 MinerU 全文解析

**输出**：
- `retrieved_papers.json`（接口文档 §9.1 schema）
- `parsed_papers.json`（接口文档 §9.2 schema）

**要点**：
- 在线搜索和 seed fallback 可以并行进行
- MinerU 解析也可以并行（不同 PDF）
- `abstract_only` 的论文标记 `parse_status: "abstract_only"`，后续 Evidence 用 abstract 作为 source

---

### Phase 4: RAG Index Construction（向量索引构建）

**目的**：构建统一的向量索引，支撑后续 Knowledge Synthesis 中的语义检索

**注意**：RAG 索引专供 Phase 5 语义检索使用（evidence-to-claim 匹配、aspect 内论文排序），不参与 Phase 2/3 的论文筛选流程。

**方案选择**：ChromaDB

> **决策理由**：
> 1. 我们的语料规模很小（50-200 篇论文 × abstract + 关键段落），ChromaDB 轻松覆盖
> 2. 论文发现层已被 SciVerse（meta-search + agentic-search）覆盖，本地 RAG 只需对已检索论文做语义匹配
> 3. AutoSurvey2 的 530K arXiv 向量库是 **abstract only**（[GitHub](https://github.com/AutoSurveys/AutoSurvey)），且 AutoSurvey2 未开源数据格式，接入成本不确定；SciVerse 的 `agentic-search` 返回**片段级**结果（带 page_no），比 abstract-only 向量库更精确，无需额外引入
> 4. ChromaDB Python native、API 简单、支持元数据过滤，24h hackathon 上手最快
> 5. ChromaDB 底层基于 chromadb-hnswlib，十万级向量检索性能无压力，演示机大语料也够用

**索引内容**：

```text
被索引的单元：
1. 论文 abstract（每篇论文一个向量）
2. 论文关键章节段落（ParsedPapers 中的 paragraphs）
3. EvidenceStore 中的 evidence_text
4. Survey 关键论断（Phase 2 中的 key_claims）

每个索引条目的 metadata：
  paper_id, source_type, aspect_ids, category_id, year
```

**Embedding 策略**：

```python
class EmbeddingConfig:
    provider: str = "openai"  # "openai" | "local" | "mock"
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = 100
    fallback_to_local: bool = True
```

**检索策略**（全部使用 ChromaDB 默认值，不做自定义优化）：

| 策略 | 选择 | 说明 |
|---|---|---|
| 向量维度 | 跟随 embedding 模型 | `text-embedding-3-small` → 1536d |
| 距离度量 | cosine（ChromaDB 默认） | 适合文本语义相似度 |
| 切分粒度 | abstract 整段 + 关键段落（不切句） | 语料小，无需细粒度切分 |
| 重叠 | 不设 | 粒度已是段落级，不需要 overlap |
| 索引类型 | HNSW（ChromaDB 默认） | 十万级无压力 |
| 元数据过滤 | paper_id, source_type, aspect_ids, category_id, year | 支持按 aspect/category 缩小检索范围 |

**输出**：`rag_index/` 目录（向量库文件 + 元数据）

**要点**：
- 向量库是 B 的内部工具，不暴露给 A/C
- 索引的粒度要平衡：太粗（整篇论文）检索不准，太细（每句话）噪声多
- 推荐粒度：abstract + 关键段落 + evidence text

---

### Phase 5: Knowledge Synthesis（知识整合）

**目的**：将所有原始数据合成为结构化的知识 artifact

这是 B 的核心产出阶段，输出接口文档规定的全部 artifact。

#### 5.0 ID 构造规则

> **核心原则**：所有 ID 必须从稳定外部来源派生，绝不由 LLM 生成，确保跨运行一致性。

| 实体 | ID 格式 | 来源 | 跨运行一致 |
|---|---|---|---|
| Paper | SciVerse `unique_id`（如 `paper:10.1038/xxx`） | SciVerse API 返回 | ✅ 稳定 |
| Paper（seed 无外部 ID） | `seed:{sha256(title+year)[:12]}` | 确定性哈希 | ✅ 稳定 |
| Evidence | `{paper_id}_p{page}_{chunk_idx}` | 从 paper_id + 源位置派生 | ✅ 稳定 |
| Figure | `{paper_id}_fig{num}` | 从 paper_id + 图编号派生 | ✅ 稳定 |
| Table | `{paper_id}_tbl{num}` | 从 paper_id + 表编号派生 | ✅ 稳定 |
| Category | `cat_{3位序号}` | 运行内递增计数器 | ❌ 单次运行内一致 |

**Category ID 跨运行不一致的应对**：`category_name` 是人类可读的稳定标签，跨运行可通过名称模糊匹配；`category_id` 仅用于单次运行内的内部引用。

#### 5.0.1 Matched Aspects 边界规则

Aspects 来自 A 的 `search_strategy`，**B 不自行新增 aspects**（这是 A 的职责边界）：

```python
def match_aspects(paper, aspects, threshold=0.6):
    matched = []
    for aspect in aspects:
        score = compute_similarity(paper.abstract, aspect.keywords)
        if score >= threshold:
            matched.append({"aspect_id": aspect.aspect_id, "score": round(score, 3)})
    return matched  # 可能为空
```

- 匹配度低于阈值 → `matched_aspects: []`，`match_scores` 全部记录（供 quality_report 参考）
- 不自动新增 aspect → 在 `quality_report` 中记录"X 篇论文未匹配任何 aspect"，由 A 决定是否补充
- 阈值可配置（`pipeline_config.aspect_match_threshold`，默认 0.6）

#### 5.0.2 多语言 Embedding 说明

- `text-embedding-3-small` 原生支持多语言（包括中文），向量空间统一
- 我们的场景：论文原文几乎全英文，Embedding 用于英文论文间语义匹配，**不涉及跨语言检索**，无需特殊处理
- 如后续需要中文 query 检索英文论文，可切换 `text-embedding-3-large` 或 `multilingual-e5-large`

#### 5.1 Paper Cards 构建

对每篇论文生成结构化卡片，**`possible_claims` 采用 LiRA 式维度分解**（借鉴 `ref/LiRA/src/researchers.py`）：

```json
{
  "paper_id": "paper:10.48550/arXiv.2305.1xxxxx",
  "title": "DreamerV3: Mastering Diverse Domains through World Models",
  "authors": ["Dan Hafner", "..."],
  "year": 2023,
  "venue": "arXiv",
  "matched_aspects": [
    {"aspect_id": "aspect_002", "score": 0.85}
  ],
  "category_id": "cat_002",
  "card_type": "deep",
  "problem": "How to build a single world model that generalizes across diverse domains without task-specific tuning.",
  "method": "Recurrent state-space model with autoencoder for perception, transformer for dynamics, and critic for imagination-based policy learning.",
  "contribution": "First world model to master 150+ diverse domains with a single configuration.",
  "limitations": "Still limited to relatively short-horizon tasks; visual fidelity lower than video generation models.",
  "evidence_ids": ["paper:10.48550/arXiv.2305.1xxxxx_p3_0", "paper:10.48550/arXiv.2305.1xxxxx_p3_1"],
  "figure_ids": ["paper:10.48550/arXiv.2305.1xxxxx_fig1"],
  "table_ids": [],
  "possible_claims": {
    "key_results": [
      {"text": "DreamerV3 achieves SOTA on 150+ diverse domains with a single configuration.", "dimension": "key_results", "evidence_ids": ["paper:..._p3_0"]},
      {"text": "Imagination-based RL remains competitive with model-free approaches at scale.", "dimension": "key_results", "evidence_ids": ["paper:..._p5_2"]}
    ],
    "method": [
      {"text": "Uses recurrent state-space model with autoencoder for perception.", "dimension": "method", "evidence_ids": ["paper:..._p3_1"]}
    ],
    "setup": [
      {"text": "Evaluated on Atari 100k, Procgen, DMLab, and Minecraft.", "dimension": "setup", "evidence_ids": ["paper:..._p6_0"]}
    ],
    "limitations": [
      {"text": "Limited to relatively short-horizon tasks.", "dimension": "limitations", "evidence_ids": []}
    ]
  },
  "bibtex_key": "hafner2023dreamerv3"
}
```

**构建方式**：
- deep 类型：用 ParsedPapers 全文 + LLM 按维度提取
- light 类型：用 abstract + metadata + LLM 按维度提取
- `possible_claims` 结构化为 4 个维度 bucket（`key_results` / `method` / `setup` / `limitations`），每项为原子化论断
- 每个 claim 附带 `evidence_ids` 指向 EvidenceStore 中的具体段落
- 长论文（>10k token）先按 section 分段分析再合并（LiRA 的 two-tier 策略）

**LLM 提取 Prompt 设计要点**（借鉴 LiRA `ref/LiRA/src/prompts/research.py`）：
- 固定 4 个维度 bucket，每个输出编号列表（`## KEY RESULTS` / `## METHOD` / `## SETUP` / `## LIMITATIONS`）
- 校准指令：`"ONLY including points that you are absolutely certain of"` — 确保每个论断可验证
- 解析方式：`##` header 分桶 → 编号列表逐条提取，确定性解析，不需要 LLM 二次处理

#### 5.2 EvidenceStore 构建

从 ParsedPapers 中提取证据碎片：

```json
{
  "evidence_id": "paper:10.48550/arXiv.2305.1xxxxx_p3_0",
  "paper_id": "paper:10.48550/arXiv.2305.1xxxxx",
  "source_type": "paragraph",
  "source_page": 3,
  "source_paragraph_index": 0,
  "text": "We present DreamerV3, a single algorithm that masters diverse domains...",
  "supports_claims": [
    {
      "claim_text": "DreamerV3 demonstrates that a single world model architecture can generalize across diverse domains.",
      "support_type": "direct",
      "confidence": 0.9
    }
  ]
}
```

**ID 构造**：`{paper_id}_p{page}_{paragraph_index}`，从源文件位置确定性派生。

**来源优先级**：全文段落 > 图表 caption > abstract

**supports_claims 字段说明**：
- `support_type`：`direct`（直接证明）/ `indirect`（提供背景上下文）/ `contradictory`（与 claim 矛盾）
- `confidence`：**NLI DeBERTa 确定性判断**（0-1），替代原来的 LLM 仲裁
- **向量相似度 ≠ 相关性**：向量检索只用于候选缩小（1000 → top-20），evidence-to-claim 匹配由 NLI 模型判断

**NLI 自动填写机制**（借鉴 `ref/SurGE/src/informationFuncs.py`）：

```python
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

def nli_fill_supports(evidence_text, claim_text):
    """用 NLI 判断 evidence 是否支撑 claim，返回 (support_type, confidence)"""
    premise = f"There is a paper. Content: '{evidence_text}'"
    hypothesis = f"The paper content supports the claim: '{claim_text}'"
    scores = nli_model.predict([(premise, hypothesis)])
    # scores[0] = [contradiction_logit, entailment_logit, neutral_logit]
    c, e, n = scores[0]
    if e > c and e > n:
        return "direct", round(float(e - max(c, n)) / (abs(e) + abs(c) + abs(n) + 1e-8), 3)
    elif n > c and e > c:
        return "indirect", round(float(e - c) / (abs(e) + abs(c) + 1e-8), 3)
    else:
        return "contradictory" if c > e else "indirect", 0.2
```

**NLI 依赖**：`sentence-transformers` + `cross-encoder/nli-deberta-v3-base`（~400MB，本地推理，零 API 成本）

#### 5.3 FigureBank / TableBank 构建

**不是扒图**，而是记录图表元数据和描述：

```json
{
  "figure_id": "paper:10.48550/arXiv.2305.1xxxxx_fig1",
  "paper_id": "paper:10.48550/arXiv.2305.1xxxxx",
  "caption": "DreamerV3 architecture overview.",
  "image_path": "cache/assets/figures/paper:10.48550/arXiv.2305.1xxxxx_fig1.png",
  "extracted_text": "World model → Imagination → Critic → Actor",
  "page": 3,
  "related_sections": ["sec_method"],
  "usable_in_report": true,
  "generation_hint": {
    "type": "architecture_diagram",
    "description": "Three-component world model: autoencoder, recurrent model, critic",
    "suggested_chart": "flowchart",
    "data_points": ["Perception", "Dynamics", "Value", "Action"]
  }
}
```

`generation_hint` 是给 C 的提示：如果原图不可用，可以根据描述生成。

#### 5.4 Taxonomy 构建（Self-Refine Step 3）

Phase 2 产生了 `preliminary_taxonomy`（独立生成）和 `refined_taxonomy`（综述修正），Phase 5 在 Phase 3 检索到实际论文后执行 Step 3：

```
Phase 2 Step 1: LLM 独立生成 preliminary_taxonomy
Phase 2 Step 2: LLM 参考综述修正 → refined_taxonomy
Phase 3: 检索到实际论文分布
Phase 5 Step 3: LLM 结合论文分布调整 → final_taxonomy → taxonomy.json
```

```json
{
  "task_id": "string",
  "topic": "string",
  "taxonomy_version": "v1",
  "refine_history": ["preliminary", "refined", "final"],
  "categories": [
    {
      "category_id": "cat_001",
      "category_name": "Game as AI Benchmark",
      "description": "Games as testbeds for AI capabilities.",
      "related_aspects": ["aspect_001"],
      "paper_ids": ["paper:10.xxx/dqn2015", "paper:10.xxx/alphago2016", "paper:10.xxx/alphazero2017", "paper:10.xxx/alphastar2019"],
      "stage_order": 1,
      "paper_count": 4,
      "adjustment_note": "Phase 3 检索到 6 篇匹配论文，2 篇归入 cat_002（更合适的类别）"
    }
  ]
}
```

**分类方法**：
1. 以 Phase 2 的 refined_taxonomy 为基础
2. Phase 3 检索到的每篇论文按关键词/LLM 匹配到最合适的 category
3. LLM 检查空类别（有 category 但无论文）和过密类别（论文过多），必要时合并或拆分
4. **允许新增 category**：如果多篇论文形成新的聚类方向，LLM 可以创建新 category（ID 运行内递增）
5. 输出 final_taxonomy

#### 5.5 CitationIndex 构建

从 paper_cards 自动生成，确保所有可引用的论文都在索引中。

---

### Phase 6: Bundle Assembly（打包交付）

按接口文档 §8 schema 打包 `knowledge_bundle.json`。

检查所有 artifact 文件是否存在，填写 summary / coverage_report / quality_report。

---

## 4. 后验验证阶段（C 写完综述后）

验证回答三个递进的问题：
1. **结构性**：C 引用的 paper_id/figure_id 是否在 B 提供的知识库中？（有没有编造引用？）
2. **归属性**：C 把某个论断归给某篇论文，这篇论文真的支持这个论断吗？（有没有张冠李戴？）
3. **充分性**：论断是否有足够的证据支撑，还是只是 C 自己的推断？（有没有无中生有？）

### 4.1 验证数据来源

**`citation_ready_set` 从哪来、准确性如何保证？**

`citation_ready_set` 不是外部 ground truth，而是 **B 自己构建的 `citation_index`**（Phase 5 §5.5 产出）。它包含 B 实际检索并解析过的所有论文。所以这个 set 的准确性是**构造性保证的**——B 只会放入自己确认存在的论文。

```
B Phase 3: SciVerse 检索 → unique_id + metadata（外部来源，稳定）
B Phase 3: MinerU 解析 → 确认 PDF 内容存在
B Phase 5: 构建 citation_index = {所有检索到的 paper_id}
    ↓
A 收到 knowledge_bundle → 提取 citation_index → 形成 citation_ready_set
    ↓
C 写 survey.md 时，只能引用 citation_ready_set 中的 paper_id
    ↓
B 验证：survey.md 中的 [paper_id] ∈ citation_ready_set？
```

**准确性边界**：set 本身不会"错"（不会包含不存在的论文），但可能**不完整**（B 检索遗漏了相关论文）。这是检索覆盖度问题，不是验证问题——由 Phase 1 覆盖度检查 + A 的 search_strategy 质量来保障。

### 4.2 Citation Verifier（结构性验证）

检查 `survey.md` 中的每个 `[paper_id]`：

```python
def verify_citations(survey_md, citation_ready_set, figure_bank, table_bank):
    results = []
    for citation_id in extract_citations(survey_md):
        entry = {
            "citation_id": citation_id,
            "valid": citation_id in citation_ready_set,
            "figures_valid": [],
            "tables_valid": [],
        }
        for fig_id in extract_figure_refs(survey_md, citation_id):
            entry["figures_valid"].append({
                "figure_id": fig_id,
                "exists": fig_id in figure_bank,
            })
        for tbl_id in extract_table_refs(survey_md, citation_id):
            entry["tables_valid"].append({
                "table_id": tbl_id,
                "exists": tbl_id in table_bank,
            })
        results.append(entry)
    return results  → citation_result.json
```

**能捕获的问题**：
- ✅ 幻觉引用：C 编造了一个不存在的 paper_id
- ✅ 幻觉图表：引用了不存在的 figure/table
- ❌ 无法捕获：paper_id 真实存在但论断与论文内容不符（需要 §4.3）

### 4.3 Claim Mapper（归属性 + 充分性验证）

这是验证的**核心**，回答"这篇论文真的说了这话吗？"。**采用 NLI-first 策略，LLM 仅作为边缘 case 的回退**。

**工作流程**（三阶段，前两阶段为确定性）：

```
阶段 1：提取 + 精确匹配
  从 survey.md 提取每个带引用的论断
  → 论断文本 + 引用的 paper_id
  → 用 paper_id 查 EvidenceStore
  → 检查 evidence 的 supports_claims 是否有 direct match

阶段 2：NLI 蕴含判断（核心，替代原 LLM 仲裁）
  对阶段 1 未匹配的论断：
  premise = evidence 原文（从 parsed_papers 获取）
  hypothesis = "该论文内容支撑论断: '{claim_text}'"
  → cross-encoder/nli-deberta-v3-base 给出 (support_type, confidence)
  entailment argmax → supported
  neutral argmax 且 entailment > contradiction → weak
  contradiction argmax → unsupported

阶段 3：LLM 回退（仅对 NLI 置信度低的边缘 case）
  NLI confidence 在 0.4-0.6 区间的模糊论断
  → 用 Intern-S2-Preview 做最终判断
```

```python
def map_claims(survey_md, evidence_store, parsed_papers, nli_model):
    claims = extract_claims_with_citations(survey_md)

    results = []
    for claim in claims:
        # 阶段 1：精确匹配 EvidenceStore 预存的 supports_claims
        evidences = evidence_store.query(paper_id=claim.paper_id)
        direct_match = any(
            e.supports_claims for e in evidences
            if any(sc["claim_text"] == claim.text and sc["support_type"] == "direct"
                   for sc in e.supports_claims)
        )

        if direct_match:
            status = "supported"
        else:
            # 阶段 2：NLI 蕴含判断
            best_support = nli_best_match(claim.text, evidences, parsed_papers, nli_model)
            if best_support["confidence"] >= 0.6:
                status = best_support["support_type"]  # "supported" | "weak"
            elif best_support["confidence"] >= 0.4:
                # 阶段 3：边缘 case → LLM 回退
                status = llm_judge_claim(claim.text, best_support["evidence_text"])
            else:
                status = "unsupported"

        results.append({
            "claim_text": claim.text,
            "cited_paper_id": claim.paper_id,
            "status": status,
            "evidence_ids": [e.evidence_id for e in evidences],
        })

    return results  → claim_map.json
```

**验证结果**：

| status | 判定方式 | 处理 |
|---|---|---|
| `supported` | EvidenceStore direct match 或 NLI entailment ≥0.6 | 通过 |
| `weak` | NLI neutral ≥0.4 或 LLM 部分支撑 | 标记，C 可选择性修改 |
| `unsupported` | NLI contradiction 或无匹配且 NLI <0.4 | **阻断**，C 必须修改或删除 |

**相比原设计的改进**：
- 原设计：Phase 5 LLM 填 supports_claims → Phase 6 LLM 仲裁 = **两个 LLM 互相猜**，存在自洽性偏差
- 新设计：Phase 5 LLM 提取论断 → NLI DeBERTa 独立校验 = **确定性模型打破循环依赖**
- NLI 模型与生成模型完全独立，不存在自洽性偏差
- LLM 仅在边缘 case（confidence 0.4-0.6）介入，大幅减少 LLM 调用量

### 4.4 独立交叉验证（可选增强）

用 SciVerse 做独立的第三方验证，打破 B→C→B 的循环依赖：

```python
def independent_verify(unsupported_claims, sciverse_client):
    """对 Claim Mapper 判定为 unsupported 的论断做独立验证"""
    for claim in unsupported_claims:
        # 用论断文本去 SciVerse 搜索
        hits = sciverse_client.agentic_search(
            query=claim.claim_text,
            top_k=5,
            filters={"publication_published_year": {"gte": 2020}},
        )
        # 如果 SciVerse 返回的片段中有支撑该论断的内容，
        # 说明 C 的论断可能正确但引用错了论文 → 建议修正引用
        # 如果 SciVerse 也找不到支撑 → 论断本身可能有问题
```

**价值**：SciVerse 的索引独立于 B 的检索结果，能发现 B 遗漏的论文或 C 引用错误的情况。

**24h hackathon 优先级**：低。§4.2 + §4.3 已覆盖核心需求，§4.4 是锦上添花。

---

## 5. 文件结构

```text
tools/
├── knowledge_pipeline_worker.py   # 主入口，编排 6 个 Phase
├── tool_executor.py               # Tool Call 路由：分发 search_papers / build_paper_cards / verify_citations
├── tool_definitions.py             # 3 个 tool 的 JSON Schema 定义（书生 API tools 参数）
├── phase1_decompose.py             # 需求拆解
├── phase2_survey_analyzer.py       # Layer 1 综述分析
├── phase3_paper_retriever.py      # Layer 2 方法论文检索
├── phase4_rag_indexer.py           # RAG 索引构建
├── phase5_knowledge_synthesizer.py # Paper Cards / Evidence / Taxonomy 等
├── phase6_bundle_assembler.py     # 打包 KnowledgeBundle
├── verify_citations.py            # 后验：引用校验（结构性）
├── build_claim_map.py             # 后验：Claim Mapper（NLI + LLM 回退）
│
├── clients/
│   ├── sciverse_client.py         # SciVerse API
│   ├── mineru_client.py           # MinerU PDF 解析
│   └── embedding_client.py        # Embedding API（可配置）
│
├── nlp/
│   ├── nli_verifier.py            # NLI DeBERTa 校验（复用 SurGE 逻辑）
│   └── data_cleaner.py            # MinerU 输出清洗（复用 SurveyX 逻辑）
│
├── indexer/
│   └── vector_store.py            # 向量库封装（ChromaDB）

cache/
├── survey_structure.json          # Phase 2 输出
├── retrieved_papers.json           # Phase 3 输出
├── parsed_papers.json             # Phase 3 输出
├── rag_index/                     # Phase 4 输出（向量库）
├── paper_cards.json               # Phase 5 输出
├── evidence_store.json            # Phase 5 输出
├── figure_bank.json               # Phase 5 输出
├── table_bank.json                # Phase 5 输出
├── taxonomy.json                  # Phase 5 输出
├── citation_index.json            # Phase 5 输出
└── knowledge_bundle.json           # Phase 6 输出
```

---

## 6. RAG 索引方案（已定稿）

> **结论**：向量库选型 ChromaDB，不做额外调研。SciVerse `agentic-search` 已覆盖论文发现层，本地 RAG 仅服务 Phase 5（evidence-to-claim 匹配、aspect 内排序）。

**已解决的设计决策**：

| 问题 | 决策 | 理由 |
|---|---|---|
| 向量库选型 | **ChromaDB** | Python native、API 简单、元数据过滤支持好、24h 上手最快 |
| AutoSurvey2 向量库 | **不引入** | 公开版仅 530K abstracts（无全文），AutoSurvey2 未开源格式；SciVerse `agentic-search` 提供片段级检索，更精确 |
| Embedding 模型 | **OpenAI `text-embedding-3-small`** | 1536d，质量高，成本可控（语料小） |
| 索引粒度 | **abstract 整段 + 关键段落** | 语料小（50-200 篇），无需细粒度切分 |
| 检索策略 | **ChromaDB 默认（cosine + HNSW）** | 无需自定义，默认值就是最优 |
| 本地/演示缩放 | **ChromaDB 统一方案** | 本地轻量开发，演示机十万级向量无压力 |

参考系统（设计参考 + 代码复用）：
- AutoSurvey：530K arXiv CS abstracts + `nomic-embed-text-v1`（[GitHub](https://github.com/AutoSurveys/AutoSurvey)）— 启发了检索增强综述的范式，但数据不直接复用
- SurveyForge：SANA 时序感知重排序 — 已通过 SciVerse `freshness_boost` / `impact_boost` 实现
- HiReview：引文网络层次聚类 — 可参考其 Taxonomy 构建方法（后续增强）
- **SurGE**（`ref/SurGE`）：NLI DeBERTa 引用校验 — **直接复用** `src/informationFuncs.py` 的蕴含判断逻辑，用于 §4.3 Claim Mapper 和 §5.2 EvidenceStore
- **SurveyX**（`ref/SurveyX`）：数据清洗 — **直接复用** `src/modules/preprocessor/data_cleaner.py` 的容错正则提取，用于 Phase 2/3 MinerU 输出清洗
- **LiRA**（`ref/LiRA`）：原子论断提取 — **借鉴** `src/researchers.py` 的维度分解 + 编号列表模式，用于 §5.1 Paper Cards 的 `possible_claims`

---

## 7. 参考 baseline 启示

从参考 baseline 调研中提取的对 Module B 有价值的信息：

| 启示 | 对 B 的影响 | 复用状态 |
|---|---|---|
| SurGE 揭示"检索上限 68% vs 端到端召回不到 10%" | Phase 1 拆解质量直接决定下游天花板；Claim-to-Evidence Map 解决核心瓶颈 | — |
| **SurGE NLI DeBERTa 引用校验** | **§4.3 Claim Mapper + §5.2 EvidenceStore 的 supports_claims 改为 NLI 确定性判断**，替代 LLM 仲裁 | ✅ 已复用 |
| SurveyForge 的时序感知重排序 | 可直接落地为 Paper Influence Score 的排序公式 | ❌ 已由 SciVerse boost 替代 |
| SurveyForge 的"启发式大纲学习" | Phase 2 Taxonomy Self-Refine 的直接参考——先独立生成大纲，再参考综述和检索论文修正 | — |
| **SurveyX 数据清洗（容错正则）** | **Phase 2/3 MinerU 输出清洗**，用 `\s*a\s*b\s*s*t\s*r\s*a\s*c\s*t\s*` 容错正则提取 abstract | ✅ 已复用 |
| **LiRA 维度分解论断提取** | **§5.1 Paper Cards 的 `possible_claims` 改为 4 维度 bucket（key_results/method/setup/limitations）**，每项原子化 | ✅ 已借鉴 |
| InteractiveSurvey 的逐句引用插入 | 与 Claim Mapper 思路一致，可参考其自适应阈值 | — |
| FIKSurvey 的反馈驱动改写 | 验证了 B 的 Verification → C 的 Revision 这条闭环的价值 | — |
| HiReview 的引文网络层次聚类 | 可参考其 Taxonomy 构建方法；后续可用引用关系知识库做 Phase 1 覆盖度验证 | — |
| SurGE Comprehensiveness 指标 | Phase 1 覆盖度验证的设计参考——用已知 coverage 作为 ground truth 检查 aspect 召回率 | — |
