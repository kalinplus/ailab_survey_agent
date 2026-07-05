# 成员 B：论文知识库、Paper Card 与可信引用

> 角色：Knowledge Grounding Engineer（34%）
> 状态：**核心功能已实现，集成测试通过**

## 1. 已完成

### 基础模块

- [x] seed_papers.json（102 篇，6 方向，SciVerse 真实 API 检索）
- [x] knowledge_pipeline_worker.py（Phase 1-6 编排，tool_registry 注册）
- [x] SciVerse client（meta-search + agentic-search，合约已验证）
- [x] MinerU client（真实 API，含空壳检测 + PDF URL 过滤）
- [x] Phase 1: Demand Decomposition（启发式，不调 LLM）
- [x] Phase 2: Survey Analyzer（Taxonomy Self-Refine Step 1-2）
- [x] Phase 3: Paper Retriever（LLM 关键词翻译 + PDF 过滤 + 空壳检测 + seed fallback）
- [x] Phase 5: Paper Cards（deep + shallow，LiRA 4 维度 bucket）
- [x] Phase 5: EvidenceStore（parsed paragraphs + agentic-search backfill）
- [x] Phase 5: FigureBank / TableBank / Taxonomy / CitationIndex
- [x] Phase 6: Bundle Assembly

### 后验验证

- [x] verify_citations.py（tool_registry 注册）
- [x] Structural verification（citation ∈ ready_set, figure ∈ bank）
- [x] Claim Mapper（NLI-first + LLM 边缘 case 回退）

### 基础设施

- [x] InternS2Client（2.1s rate limit，120s timeout）
- [x] NLI Verifier（FakeNLI for tests + 接口准备 DeBERTa）
- [x] Data Cleaner（SurveyX 容错正则）
- [x] 合约修复：SciVerse base URL、Intern-S2 API path、native field mapping
- [x] Real-API-first 偏好写入 CLAUDE.md

### 测试

- [x] 单元测试 149 passed（0.3s）
- [x] 集成测试：A→B 全链路（real SciVerse + fake LLM，69s）
- [x] 全真实 API 测试框架（Intern-S2 + SciVerse，需 API key 有效）

### 文档

- [x] 模块B-架构设计.md（v3，同步实现状态）
- [x] 模块B-工具调用指南.md（面向 A/C 组）
- [x] CLAUDE.md 外部 API 合约

---

## 2. 待办

### 高优先（影响全链路）

- [ ] 全真实 API 测试通过（当前 timeout，已调到 120s 待验证）
- [ ] NLI 本地模型接入（cross-encoder/nli-deberta-v3-base 替代 FakeNLI）
- [ ] 配合 Module C 联调 write_survey → verify_citations 闭环

### 中优先（质量提升）

- [ ] MinerU：找到可靠 PDF 源后开启（绕过 arxiv 防爬）
- [ ] 多 aspect 并发 SciVerse 调用（当前串行，~2x 加速）
- [ ] Paper Influence Score（综合引用 + 时效 + 开源影响力）
- [ ] Taxonomy Step 3 验证空/密类别

### 低优先（锦上添花）

- [ ] 独立交叉验证（§4.4，SciVerse 独立验证 unsupported claims）
- [ ] SciVerse `/content` 全文拉取（替代部分 MinerU）
- [ ] 综述库自动更新（`update_survey_store: true`）
- [ ] Memory 系统（论文元数据跨 session 持久化）

---

## 3. 对外接口

| 对接 | 方向 | 数据 |
|------|------|------|
| A → B | agent_loop 调用 | `knowledge_pipeline_worker`（知识构建） |
| A → B | agent_loop 调用 | `verify_citations`（后验验证） |
| B → A | 文件交付 | knowledge_bundle.json → A 验证 + citation prelock |
| B → C | 文件交付 | paper_cards / evidence_store / figure_bank / table_bank / taxonomy |

---

## 4. 答辩要点

1. **Anti-Hallucination Chain**：SciVerse 元数据 → LLM 结构化提取 → agentic-search chunk → NLI 验证 → 后验 claim mapping
2. **Real-API-first**：不依赖 mock，评审替换 API key 可直接重跑
3. **Shallow Cards + Backfill**：MinerU 不可用时仍能完整产出 evidence chain
