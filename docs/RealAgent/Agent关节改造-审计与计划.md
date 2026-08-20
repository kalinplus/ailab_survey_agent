# Agent 关节改造：代码审计与实施计划

> 日期：2026-08-20 ｜ 审计基线：commit `d9d305e`（Public submission finalized）
> 性质：**计划文档，尚未实施**。审计部分为已验证的代码事实（含 file:line 锚点），设计部分为 v2 改造方案。

---

## 0. 背景与核心问题

项目文档（CLAUDE.md / README / 设计文档）将系统描述为"Agent 编排的综述生成 Harness"。核心问题：**文档宣称的确定性部分与 Agentic 部分是否如实实现？**

审计结论（一句话）：

> 这是一个确定性 DAG 流水线，LLM 是"阶段内单发调用"的组件；文档宣称的"Agent 动态决策"在主流程中基本不存在，唯一的真 Agent 循环是死代码。

改造方向不是"全面 Agent 化"，而是：**保留确定性骨架，只在三个"DAG 表达不了"的关节接入受预算约束的 Agent 循环。**

---

## 1. 代码审计：文档 vs 实现

### 1.1 确定性部分核对（基本属实）

| 宣称 | 现实 | 判定 |
|---|---|---|
| API 请求/数据清洗 | SciVerse/MinerU client + DataCleaner，真实 | ✅ |
| PDF 解析 + 降级 | `use_mock=False` 默认，失败降级真实 abstract（`parse_status="abstract_only"`） | ✅ |
| 论文元数据管理 | citation_index / paper_cards 均源自 SciVerse 返回 | ✅ |
| 引用抽取 | `tools/verify/structural.py` 纯正则 + 白名单集合匹配 | ✅ |
| HTML 渲染 / 测试分层 | 属实 | ✅ |
| NLI 验证 | 代码结构真实（3 阶段），**但默认是 FakeNLIModel** | ⚠️ 见 1.4 |

### 1.2 Agentic 部分核对（多数不实）

| 文档宣称 | 代码现实 | 证据 |
|---|---|---|
| 主题拆分子问题 | LLM 路径**仅在 `--mode full` 且配 key 时启用**；CLI 默认 `demo` 走硬编码模板 aspects；聚类路径默认关 | `planner.py:99`、`config.py:79` |
| 动态生成检索查询 | 真实存在三级 fallback（LLM→聚类→模板），但默认全落到模板 | `search_strategy_builder.py:60-99` |
| 根据检索结果调整分类 | P2 有 PRELIM+REFINE 两次 LLM 调用精炼 taxonomy——真实，但为**单轮批处理**，非迭代循环，结果不回流 | `phase2_survey_analyzer.py:87,101` |
| 根据缺失证据继续搜索 | `phase5_evidence._agentic_backfill` 真实（无证据 claim→agentic_search→入库），但是**固定执行的一次性补全**，非 Agent 决策"要不要再搜" | `phase5_evidence.py:34-66` |
| 验证失败后决定补证据/重写 | ❌ **完全不实**。`revise_survey.py` 是纯正则**删除**（删非法引用、删 unsupported 句子、重建 References），零 LLM，零决策 | `revise_survey.py:36-39` |
| 预算/完成条件决定停止 | ❌ 不存在 Goal Gate。停止 = 阶段跑完 | — |

### 1.3 六个关键问题的答案

1. **Agent 能基于 observation 选工具吗？** → 不能。`AgentLoop.run()`（`agent_loop.py:115-189`）是硬编码顺序：worker→prelock→write→verify→[revise]→render。唯一能观察结果选工具的 `AgentToolLoop`（有 observation、重试、max_turns=8）**全库零引用**，docstring 自认 "Optional… for demos"（`agent_tool_loop.py:1`）。
2. **失败后按错误类型改变策略？** → 仅策略生成层有（LLM→聚类→模板 fallback）。工具失败直接 `RuntimeError`（`agent_loop.py:216`）；验证失败的处理与失败类型无关。
3. **显式状态和调用预算？** → 状态真实（StateManager + `logs/state.json` + `run.jsonl`）；调用/token 预算不存在，只有数据量上限（max_papers）。
4. **Citation Verification 闭环？** → **半闭环、最多一轮、结果被丢弃**。`agent_loop.py:167-180`：verify→fail→revise→re-verify，第二次结果无代码再读，无条件进 render。
5. **Goal Gate 还是写死阶段？** → 写死。`_verification_passed` 的 0.95 阈值（`agent_loop.py:203-211`）只决定"是否 revise 一次"，从不阻止结束。
6. **日志能重放 trajectory？** → 部分。A 层 run.jsonl 记 tool_started/finished + metrics + 截断结果；工具内部 LLM 调用走标准 logging 不进 jsonl；无 token/耗时/停止原因的统一轨迹。

### 1.4 三个额外重大偏差（文档未如实呈现）

1. **综述正文不是 LLM 写的**。`write_survey.py` 全文 1394 行仅一处 `.chat`（`_plan_image_prompt`，为图片写 prompt）。`_render_survey`（`write_survey.py:332-431`）里 Abstract/Introduction/Challenges/Future/Conclusion 是**写死的中英文字面量**，只把 `[paper_id]` 和 evidence 句子拼进模板。
2. **NLI 默认是关键词假模型**。`verify_citations.py:112-135` 与 `knowledge_pipeline_worker.py:125-140` 的 `_nli_model()` 都默认 `FakeNLIModel`（"world"/"model"/"game" 等 7 个词→entailment）；真 NLI 需 `EVISURVEY_REAL_NLI=1`，LLM fallback 需 `EVISURVEY_CLAIM_LLM_FALLBACK=1` 且默认关。
3. **沙箱是声明式的**。`ToolSpec.allowed_read/write_paths` 只进 prompt 和 manifest 展示；`SandboxGuard` 只在入口检查 request_path 存在（`tool_registry.py:65`），工具内部读写无强制约束。

### 1.5 级联病理（比单项偏差更严重）

`planner.py:68` 硬编码 `"use_mineru": False` ⇒ `parsed_papers` 默认为空（`phase3_paper_retriever.py:387`）⇒ demo 模式下 agentic 回填也关（`use_online_search = mode=="full"`，`knowledge_pipeline_worker.py:82`）⇒ **evidence_store 为空** ⇒ `claim_mapper` 走 Stage 3，所有带引用句子全部判 unsupported ⇒ 触发 revise 后是**全文句子级删除**。

当前交付物未暴露此问题，纯因 showcase 用 `FINAL_SEED_PAPERS=1` 载入预置 `final_evidence_store.json`。**评审换 key 重跑（不带 final 种子）时，demo 模式的失败模式是灾难性的。**

### 1.6 真实落地、值得保留的部分

- memory 系统：真实持久化策略运行记录 + lessons，`load_for_strategy` 注入策略 prompt（`harness/memory_manager.py`）。
- agentic backfill 机制（P5.2）：对无 parsed 证据的 claim 用真实 SciVerse chunk 补证据。
- SciVerse/MinerU/Intern-S2 契约层：实测校准（`hits` vs `results`、author 字段损坏、`"HIGH"` boost 被拒等）。
- `FINAL_SEED_PAPERS` 可复现机制、三层测试体系、prelock 白名单、结构校验。

---

## 2. 理论框架：Workflow 与 Agent 的分界

判据（Anthropic *Building Effective Agents*：Workflow 是路径预先编排的系统，Agent 是模型根据中间结果动态决定路径的系统）。一个环节命中任意一条即存在 Agent 真实需求：

1. **决策依赖执行中才可见的数据**（第 N 步的正确动作取决于第 N-1 步返回了什么，且无法在写代码时枚举）；
2. **失败类型 × 修复动作的组合空间过大**（硬编码退化为一刀切）；
3. **停止条件是数据依赖的**（"做多少才算够"取决于做到现在看到了什么）。

任务的双相结构：

- **探索相（发散）**：领域分类法是**语料的函数**，检索前未知——命中判据 1；
- **生产相（收敛）**：大纲与证据池固定后的组织、渲染、索引——确定性工程，Agent 无增量。

代码中的三个"Agent 形状的洞"（每个补丁对应一条判据）：

| 洞 | 判据 | 代码证据 |
|---|---|---|
| 验证失败的一刀切退化 | 2 | `revise_survey.py` 全部失败一律正则删除 |
| 检索策略默认塌方到模板 | 1 | 三级 fallback 默认落到模板；世界模型是热门领域掩盖了覆盖率问题 |
| 半闭环验证 + 无停止条件 | 3 | 第二次 verify 结果被丢弃；0.95 阈值只决定修不修一次 |

---

## 3. 改造设计

### 3.0 总体形态

**两个 Agent 关节 + 一个确定性 Goal Gate**。关键设计决策：

- **不用自由工具循环，用类型化动作循环（typed-action loop）**。理由：Intern-S2 限速 1 req/2s（`llm_client.py:24`）；动作空间可枚举（split/merge/rewrite/backfill/weaken/delete）；两个专用小循环各约 50 行，符合 minimal abstraction 偏好。
- 现成 `AgentToolLoop` 不做底座（自由循环方差大），维持死代码现状或删除。
- demo 快速路径与 `FINAL_SEED_PAPERS` showcase 机制**零改动**，关节只活在真实路径（full 模式）上。

### 3.1 关节 1（探索相）：检索策略-分类法协同进化

**方案已定稿并拆分为独立模块文档 → [关节1-策略Agent-方案与测试.md](关节1-策略Agent-方案与测试.md)**（含两个定稿决策、救援阶梯、通用循环底盘设计、分层测试与 A/B 验收方案）。实施以该文档为准。

概要：把"先拆分、后搜索、不反馈"改为"探查聚类 → LLM 命名 → 全量试搜 → 成绩单反馈 → 编辑循环（≤3 轮，确定性 Gate 终止 + best-so-far 回滚）"。相关度信号定稿用 **Embedding**（`bge-small-en-v1.5`，复用已有 `sentence-transformers` 依赖，否决 NLI 零样本挪用）；0 结果方向走**四级救援阶梯**（R0 放宽年份过滤 → R1 LLM 同义改写+probe → R2 agentic-search 语义发现 → R3 邻近关键词池 → 仍死则删除并记 memory），否决随机搜索与第 4 方搜索 API（评审换 key 重跑约束）。

**架构约束（坦诚记录）**：mid-pipeline 回环（P2 精炼后回头改检索）目前做不起——B worker 单进程跑完 P1→P6，回头需拆 worker 支持按阶段重跑，跨 A/B 边界。**环前（策略循环）和环后（修复循环）便宜，环中贵**。v2 只做环前+环后；mid-pipeline 反馈留 v3。

### 3.2 关节 2（生产相）：按失败类型修复——最高优先级

位置：重写 `tools/revise_survey.py` + 修 `harness/agent_loop.py:167-180`。失败分类**已经算好**（claim_map 的 `status/confidence/evidence_ids` + citation_result 的 `valid`），缺的只是动作选择层。

| 失败类型（现成字段判定） | 合理动作 | 实现 |
|---|---|---|
| A. 引用 id ∉ 白名单（`valid=false`） | **remap**：LLM 看 claim 句 + citation_index 标题，返回正确 paper_id 或 null | 1 次 LLM |
| B. `unsupported` 且该论文**有**证据条目 | **换证据**（同论文挑蕴含更高 chunk，可加 1 次 agentic_search）或 **rewrite**（LLM 改写为证据支持的弱形式） | 每条 1 次 LLM |
| C. `unsupported` 且该论文**零**证据 | **backfill**：`agentic_search(claim_text)` → chunk 入 evidence_store → 单句重跑 NLI；仍败才删 | 复用 `_agentic_backfill` 调用模式 |
| D. `weak`（neutral, 0.4–0.6） | **weaken**：单句降断言强度（"证明"→"提示"），或换证据，confidence 够高则保留 | 每条 1 次 LLM |
| E. 图表引用无效 | remap 到 figure_bank caption 或删行 | 机械 |

工程细节：

- **批量调用对付限速**：按失败类型分组，1 次 LLM 处理 5 条（返回 JSON 数组）；20 条失败 ≈ 4 次调用。两轮全预算 ≈ 10–15 次 LLM ≈ 30s。
- **增量验证**：修复后只对改动句重跑 claim NLI + 全文重跑结构正则（正则免费），不重跑整个 verify。
- **修复日志即实验数据**：每条记 `{claim_id, failure_type, action, outcome}` 写入 `output/repair_log.json`。

`agent_loop.py` 接线（替代 167-180 的半闭环）：

```python
for round in range(2):                                  # MAX_REPAIR_ROUNDS
    result = self._run_tool("verify_citations", ...)
    if gate.passed(result): break
    if round > 0 and not improved(result, prev): break  # 不动点: 修复无进步 → 提前停
    self._run_tool("revise_survey", ...)                # 现在是修复动作, 不是删除
```

### 3.3 Goal Gate（确定性代码，非 LLM）

Gate 本身不 LLM 化——确定性策略作用于数据依赖信号，这正是它与 Agent 关节的区别。落 `agent_loop.py`，约半天：

```python
gates = {
    "structural_validity": invalid_citations == 0,        # 造假 id 零容忍
    "unsupported_claims":  unsupported == 0 or repair_budget_exhausted,
    "min_coverage":        sections >= N and figures >= M,
}
# passed / stop_reason 写入 final_state.json 和 run.jsonl
```

### 3.4 前提修正（不做则实验无效）

1. **`EVISURVEY_REAL_NLI=1` 必须开**：修复环跑在 FakeNLI 假标签上，实验测的是"修复关键词匹配的失败"，结论不可信。真模型首次下载需 `HF_ENDPOINT=https://hf-mirror.com`。
2. **`use_mineru` 显式可配**（`planner.py:68` 不再硬编码 False），否则 parsed=∅ 的级联病理（§1.5）依旧。

### 3.5 明确不 Agent 化的清单

P3 rerank 与 P5 cards（打分/抽取，动作空间为空）、NLI 本身（verifier 非决策者）、ToolRegistry 调度、沙箱、渲染、prelock。`write_survey` 模板→LLM 生成是**生成质量轴**（另一条轴），不属于 agentization，且威胁 FINAL_SEED_PAPERS 可复现叙事——后续按章节渐进，v2 不动。

---

## 4. 修之前 / 修之后流程对照

### 修之前（当前真实行为）

```
主题
 ▼
[A] task_request ──────────── demo 默认: use_online_search=False, use_mineru=False(硬编码)
 ▼
[A] search_strategy ───────── demo: 硬编码模板; full: LLM 单发, 失败→模板 (无探查/无反馈/K写死)
 ▼
[B] worker 单进程线性:
    P1 decompose ── LLM 批量判 seed→aspect, 失败跳过
    P2 analyzer ─── LLM×2 初分+精炼, 单发不回流
    P3 retrieve ─── 每 aspect 真实 meta-search + 确定性过滤/排序; LLM 仅中文→英文查询
    │  MinerU ───── use_mineru=False ⇒ parsed_papers 默认为空
    P5.1 cards ──── LLM 单发/篇 (从 abstract)
    P5.2 evidence ─ parsed=∅ ⇒ 空; agentic 回填仅 full; NLI 默认 FakeNLI
    P5.3-5/P6 ───── 确定性 banks/index/bundle
 ▼
[A] citation_prelock (确定性白名单)
 ▼
[C] write_survey ──────────── 模板拼接 (Abstract/Intro/Conclusion 写死字面量)
 ▼
[C] verify_citations ──────── 结构: 正则+白名单; 语义: claim_map 默认跑 FakeNLI
 ▼
[A] _verification_passed? ─── 0.95 阈值, 只决定"修不修一次"
 ├─ 通过 → render
 └─ 失败 → revise(正则一刀切删除) → 覆盖 survey.md → 再 verify(结果无人读) → render
```

级联病理：demo 默认 evidence=∅ ⇒ 全部 claim unsupported ⇒ 全文句子级删除；被 `FINAL_SEED_PAPERS` 掩盖。

### 修之后（同骨架，五个插入点）

```
主题
 ▼
[A] task_request (不变)
 ▼
╔═ 关节1: 策略循环 (full; demo 仍走模板快路径) ═╗
║ 探查 20 篇 → 聚类初始 K aspect                  ║
║ loop ≤3: 每 aspect 1 次 meta-search → 产出表    ║
║ (确定性) → LLM 选 split|merge|rewrite|add|accept║
║ 教训写 memory/                                  ║
╚═════════════════════════════════════════════════╝
 ▼
[B] worker P1–P6 (骨架不变; EVISURVEY_REAL_NLI=1; use_mineru 显式可配)
 ▼
[A] citation_prelock (不变)
 ▼
[C] write_survey (不变) → [C] verify (真 NLI)
 ▼
╔═ Goal Gate (确定性) ════════════════════════════╗
║ structural==1.0; unsupported==0 或预算尽; 覆盖≥N ║
║ 判定+stop_reason 落盘                            ║
╚═════════════════════════════════════════════════╝
 ├─ 达标 → render (附 stop_reason)
 └─ 未达标 → ╔═ 关节2: 修复循环 (round ≤2) ═══════╗
             ║ 按失败类型分派: remap/换证据/弱化/   ║
             ║ backfill→单句重验/删(兜底)          ║
             ║ 同类型批量 1次LLM≈5条; 修复日志落盘  ║
             ║ 不动点停止 → 回 Goal Gate           ║
             ╚════════════════════════════════════╝
```

### 插入点总表

| 插入点 | 文件 | 之前 | 之后 |
|---|---|---|---|
| 策略生成 | `search_strategy_builder.py` | 单发 LLM，默认落模板 | 探查→拆分→产出反馈→编辑循环 ≤3 轮 |
| 证据校验 | `verify_citations.py` / worker | 默认 FakeNLI | 真 NLI（环境变量前置） |
| MinerU 开关 | `planner.py:68` | 硬编码 False | 显式可配 |
| 修复策略 | `revise_survey.py` | 正则一刀切删除 | 失败类型→动作映射 |
| 循环控制 | `agent_loop.py:167-180` | 修一次、二次结果丢弃、无条件渲染 | Gate + ≤2 轮 + 不动点 + stop_reason |

完全不动的：P3 检索排序、P5 cards、prelock、write_survey、render、ToolRegistry、memory 持久化、FINAL_SEED_PAPERS。

LLM 调用量（full 一次运行，粗算）：修前 ≈ 策略 1 + P1 ~2 + P2×2 + P3 翻译 0–1 + cards ~N + 图片 prompt ~5；修后 +循环 ~3 + 修复 ≤15。2s/req 限速下总增幅约 +30–60s。

---

## 5. 优先级与工作量

| 优先级 | 事项 | 改动文件 | 量级 |
|---|---|---|---|
| P0 | 关节 2 修复循环 + Goal Gate | `revise_survey.py` 重写、`agent_loop.py` 接线 | 价值最可量化、半径最小 |
| P0 | `EVISURVEY_REAL_NLI=1` 前提 | 实验 runner / 文档 | 一个环境变量，决定实验有效性 |
| P1 | 关节 1 策略循环 | `search_strategy_builder.py` | 去双重门禁 + ~80 行循环 |
| P2 | `use_mineru` 显式化 | `planner.py` | 一行 + 文档 |
| v3 | mid-pipeline 反馈（拆 worker）、write_survey 渐进 LLM 化 | — | 跨 A/B 边界，另立项 |

每步独立可验证：P0 完成即可跑对照实验的前后两半，符合切片循环。

---

## 6. 实验设计（"要不要 Agent"的实证回答）

- **主题集**：固定 5–10 个主题，含 1–2 个冷僻主题（暴露模板策略覆盖率塌方）。
- **基线（修之前）指标**：
  1. 验证失败后被删除的句子占比（删除率）；
  2. 模板查询的子领域覆盖率（各 aspect 产出 ≥3 篇相关的比例）；
  3. 二次验证的真实通过率（现在这个数字根本没人看）。
- **对照（+关节 2）**：同主题集重跑，比较删除率、unsupported 率、citation_validity_score、成本（LLM 调用数/时长）。预期叙事：修复环把删除率从 ~40% 降到 ~10%、支持率显著上升。
- **数据采集自动化**：`repair_log.json` 的 `{failure_type, action, outcome}` 即对照实验的原始数据，无需额外埋点。
- **有效性前提**：`EVISURVEY_REAL_NLI=1`（否则修复的是假标签）。

60 秒面试答案（已论证，存档备查）：

> 这个任务分两相。生产相——解析、渲染、索引——我全部用确定性工作流，因为路径可枚举，确定性就是质量。但三个环节的输入分布在执行前无法枚举：领域分类法是语料的函数、检索是依赖中间结果的试错、引用失败类型乘修复动作的组合空间过大。硬编码在这些点会退化成一刀切策略——我们最初版本就是验证失败一律删除，内容损失很大。所以我保留确定性骨架，只在这三个关节接入受预算约束的 Agent loop，并用 Goal Gate 以覆盖率和证据支持率决定停止。判断依据不是"Agent 更先进"，而是每个接入点都有对照实验数据支撑。
