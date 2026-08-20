# 关节 1：检索策略 Agent — 落地方案与测试方案

> 日期：2026-08-20 ｜ 状态：**方案定稿，待实施**
> 上游：[Agent关节改造-审计与计划.md](Agent关节改造-审计与计划.md) §2（判据）、§3.1（本关节概要）
> 本文件是关节 1 的完整实施规格，含两个已定稿决策、落地方案、四层测试与 A/B 验收方案。

---

## 1. 问题回顾

现状（`harness/search_strategy_builder.py`）：拆分方案**一次性**产生（demo=硬编码模板；full=单发 LLM），交给 P3 检索后**无人回头看检索质量**。两类典型失败无人发现：

- 某方向查询词与领域行话错位 → 搜出 **0 篇**，该方向在综述中静默空缺；
- 两方向关键词重叠 → 搜回**同一批热门论文**，综述内容重复。

根源：**拆分方案的质量取决于搜出什么，但代码是先拆分、后搜索、不反馈**（总计划判据 1：决策依赖执行中才可见的数据）。

改造一句话：把"闭卷拆分"改成"开卷拆分"——粗搜探查 → 聚类定形 → LLM 命名 → 全量试搜 → 成绩单反馈 → 编辑循环（≤3 轮，确定性 Gate 终止 + best-so-far 回滚）。

---

## 2. 两个已定稿决策（2026-08-20 讨论）

### 决策 1：相关度信号用 Embedding，否决 NLI 零样本挪用

- **否决理由**：NLI 的训练目标是蕴含关系判定（成立/矛盾/中立），"这篇论文是否关于 X"是分布外任务。
- **采纳方案**：`sentence-transformers` 的 `SentenceTransformer("BAAI/bge-small-en-v1.5")`（~130MB，CPU 可跑，英文语料——P3 查询本就是英文）。
- **依赖事实（已验证）**：`requirements.txt` 已含 `sentence-transformers>=3.0.0`（NLIVerifier 的 CrossEncoder 出自同库），**零新增依赖**，lazy import + `HF_ENDPOINT` 镜像路径与 NLI 完全一致。
- **嵌入对象**：方向侧 = `aspect_name + description + keywords`；论文侧 = `title + 摘要前 1–2 句`。每轮 K×10 对，本地推理。
- **阈值策略**：硬 Gate 只压在客观量（`n_unique ≥ 3`）上；余弦相关度作为软信号列给决策 LLM，绝对阈值一次校准即可，避免脆阈值卡死循环。
- **降级链**：模型下载失败 → 退回关键词匹配列（成绩单保留两列，结构不变），日志记录降级。
- 附赠（可选，不阻塞）：probe 聚类可从关键词聚类升级为 bge 嵌入聚类。

### 决策 2：0 结果方向用四级救援阶梯，否决随机搜索

| 级 | 动作 | 成本 | 说明 |
|---|---|---|---|
| R0 | **放宽过滤器重试** | 零 | P3 硬编码 `2018–2026` 年份过滤（`_year_filters`），很多 0 结果纯粹是被它筛死的。`probe_query(query, filters=none)` |
| R1 | **LLM 同义改写 + probe** | 决策调用本身 | LLM 自带领域行话知识（"execution feedback" → "unit test guided synthesis"），是**知识引导的探索**，严格优于随机。连续 2 次改写失败后允许发散试邻近词——这就是"随机"，一句 prompt 的事，不用写代码 |
| R2 | **agentic-search 语义发现** | 现有 API 内 | meta-search 对关键词敏感；`agentic_search(方向的自然语言描述)` 是语义召回，hit 带真实 `title`。从命中标题/keywords **提词** → 回查 meta-search。即"容错搜索引擎"，但在已付费、已合规的 SciVerse 内 |
| R3 | **邻近关键词池** | 零 | SciVerse 结果原生带 `keywords` 字段（`_to_retrieved` 已在读）。用 embedding 找死方向最邻近的已检索论文，让其"捐"关键词。免解析 Related Work |
| R4 | **仍死 → 删除该方向** | — | 诚实结论："此子领域在语料中不存在"本身就是发现。记入 memory，下次同主题不再浪费时间。**R4 是特性不是失败** |

- **为什么否决随机**：随机 = 从词汇空间盲目采样；R1 = 从领域同义词分布采样，同成本高命中，且已包含"受控随机"（发散指令）。
- **为什么禁第 4 方搜索 API**（Google / Perplexity）：死因不是成本，是**硬约束**——评审替换 API Key 重跑，系统只允许三个 API。第 4 个外部依赖 = 重跑当天必挂。R2 已在系统内实现同等容错能力。

---

## 3. 落地方案

### 3.1 总体流程

```
t0 (零LLM):  probe SciVerse 20 篇 → 确定性聚类得 K 簇
LLM call 1:  给簇命名/配关键词 → 初始 aspects          # "聚类定形, LLM 命名"
框架 (零LLM): 全 aspect 采样 (meta-search page_size=8) → 成绩单 v1
turns:       Agent 见成绩单 → [probe_query×0~2] → commit_edits / accept
             → 全量重采样 → 成绩单 v2 → ...             # ≤3 轮
终点:        Gate 复核 → best-so-far → _validate_strategy → 输出
             on_finish: 教训写 memory (record_strategy_run 复用)
```

### 3.2 文件与组件

```
harness/agents/
├── loop.py            # 通用循环底盘 (~80 行), 关节2 修复 Agent 复用
├── relevance.py       # EmbeddingScorer(lazy) + 关键词降级 + 成绩单构建
└── strategy_agent.py  # AgentConfig(prompt/工具箱/预算) + Gate + 救援工具
```

接缝：

- `build_search_strategy` 保持入口：`mode=full` 且配置就绪时委托给策略 Agent；demo/模板快路径**原样保留**；
- 输出经现成 `_coerce_strategy` / `_validate_strategy` 校验（schema 契约复用，不另发明）；
- 死代码 `harness/agent_tool_loop.py` 被底盘吸收替换；
- 外层 `ToolRegistry`（file-based B/C 工具协议）**不动**——两层工具体系：外层跨模块重型工具，内层 Agent micro-tools（进程内函数）。

### 3.3 通用循环底盘（loop.py）

```python
@dataclass
class AgentConfig:
    name: str                      # 身份, 进日志和 prompt
    system_prompt: str             # 每关节独立的角色设定
    tools: dict[str, ToolFn]       # 能力即权限: 不在字典里 = 调不到
    max_turns: int
    max_llm_calls: int
    max_wallclock: float
    on_action: callable | None     # 观测埋点 (trajectory jsonl)
    on_finish: callable | None     # 收尾 hook (memory / 实验日志)

class BoundedAgentLoop:
    def run(self, task) -> AgentResult: ...
```

底盘职责（写一次，关节 1/2 共用）：

1. 每轮 `llm.json_chat` → 动作分发 → observation；
2. **状态自动注入**：成绩单等框架侧状态每轮自动附在 user message，不让模型浪费 turn 问状态；
3. **预算熔断**：max_turns / max_llm_calls / max_wallclock 任一触发 → 终止并带 reason；
4. **trajectory jsonl**：每轮一行 `{agent, turn, action, args, result_digest, llm_idx, elapsed}`——即实验设计的轨迹数据基础设施；
5. JSON 解析失败 → 一次修复重试 → 再失败冒泡（fail-fast）。

### 3.4 策略 Agent 配置

**工具箱（= 权限边界）**：

| 工具 | 参数 | 对应设计 |
|---|---|---|
| `probe_query` | `query, filters=none, page_size=8` | 测试查询不落盘；R0 由 `filters=none` 实现 |
| `discover_by_description` | `text` | R2：agentic-search 语义发现，返回命中标题/keywords 候选词 |
| `commit_edits` | `edits[]` | split/merge/rewrite/add/delete，提交即换轮（触发全量重采样+新成绩单） |
| `accept` | — | 终止提议，需过 Gate 复核 |

R3 关键词池由框架自动附在消息里（不占工具位）。

**Prompt 要点**：角色 = 检索策略师；动作枚举与 JSON 格式；救援 playbook（R0→R1→R2→R3 顺序，含"连续 2 次改写失败可发散"指令）；只输出一个 JSON 对象。

**预算**：`max_rounds=3`、`max_llm_calls=7`（含命名 1 次）、SciVerse 调用上限 20、wallclock 兜底。

### 3.5 成绩单 schema（每 aspect 一行，自动注入）

| 列 | 含义 | 发现什么 |
|---|---|---|
| `n_hits` | 原始返回数 | 0 = 查询词不对路 |
| `n_unique` | 全局去重后独有数 | 高 hits 低 unique = 重复搜热门论文 |
| `overlap` | 与其他方向的 Jaccard 重叠 | 两方向实为同一方向 → merge |
| `rel` | Embedding 余弦相关度（降级时为关键词分） | 搜到但跑题 = 查询太宽 |
| `top5_titles` | 原文标题样本 | 给 LLM 肉眼判断 |
| `year_span` | 年份分布 | 语料偏老/偏新 |

### 3.6 Gate 与终止

终止四条件（确定性代码持有，LLM 的 `accept` 需复核）：

1. **达标**：全 aspect `n_unique ≥ 3` 且 rel 达标；
2. **预算尽**：轮数 / LLM 调用 / wallclock；
3. **不动点**：全局分（Σ min(n_unique,10)/10 × rel）两轮改进 < ε → 提前停；
4. **回归保护**：框架始终保留 best-so-far；某轮改差 → 回滚最优版再停。**保证循环下界 = 单发水平，可形式化断言 `score(final) ≥ score(round_1)`**。

### 3.7 环境变量

- `EVISURVEY_EMBED_MODEL`（默认 `BAAI/bge-small-en-v1.5`）；`HF_ENDPOINT` 复用现有镜像机制；
- 启用开关：`mode=full` 即走 Agent（或加 `EVISURVEY_STRATEGY_AGENT` 显式开关，实现时定）；
- 真 NLI 前提（`EVISURVEY_REAL_NLI=1`）属于总计划 §3.4 前提修正，不在本模块范围。

### 3.8 实现切片（每片独立可验证）

| # | 切片 | 验收 |
|---|---|---|
| 1 | `loop.py` 底盘 + fake LLM 单测 | JSON 动作协议/熔断/trajectory 全绿 |
| 2 | `relevance.py` 成绩单 + embedding（含降级路径） | 单测绿，两列正确 |
| 3 | 真接入：probe/采样/naming/probe_query | 集成测试绿（真 SciVerse + fake LLM） |
| 4 | Gate + 回滚 + memory + trajectory 落盘 | 单测绿；同主题跑两次第二次 memory_context 非空 |
| 5 | 对照运行（§4.4） | 热主题 + 冷主题各 ≥1，A/B/C 报告产出 |

---

## 4. 测试方案

总原则：**单元/集成测试证明"代码对"，A/B 对照实验证明"设计值"**。沿用项目现有三层测试体系（unit fake LLM 无网络 / integration 真 SciVerse + fake LLM / `-m real_api` 全真）。

### 4.1 单元测试（fake LLM、fake SciVerse，无网络）

| 文件 | 测试点 |
|---|---|
| `tests/unit/test_agent_loop.py` | ① 动作分发：fake LLM 返回 `call_tool` → 正确调用工具并回填 observation；② 非法动作/JSON 解析失败 → 一次修复重试 → 再失败冒泡（fail-fast）；③ 工具不在工具箱 → 立即拒绝（权限边界）；④ 三种熔断各自触发（turns/llm_calls/wallclock）→ 带 reason 终止；⑤ trajectory jsonl 每轮一行且字段完整；⑥ 状态自动注入：每条 user message 含当前成绩单 |
| `tests/unit/test_relevance.py` | ① n_unique 全局去重正确（构造已知重叠输入）；② pairwise overlap/Jaccard 正确；③ EmbeddingScorer lazy 加载；模型加载异常 → 降级关键词列且日志记录；④ 余弦计算与归一化（小模型可 mock embedding 输出定值） |
| `tests/unit/test_strategy_gate.py` | ① 全达标 → pass；② LLM `accept` 但某 aspect `n_unique=1` → Gate 拒绝并强制继续（若预算剩）；③ 回滚：round_2 全局分 < round_1 → 输出 round_1 策略（断言 `score(final) ≥ score(round_1)`）；④ 不动点：改进 < ε → 提前停；⑤ R0：`filters=none` 正确透传；⑥ R2：`discover_by_description` 从 agentic hits 提取标题/keywords 生成候选查询 |

Fake SciVerse client **镜像真实契约**（`meta_search` 返回 `results`、`agentic_search` 返回 `hits`——注意两者字段名不同，这是实测过的坑）。

### 4.2 集成测试（真 SciVerse + fake LLM）

`tests/integration/test_strategy_agent.py`：

- 世界模型主题真 probe + 真采样：成绩单各列非空、schema 合法；
- fake LLM 按剧本决策（round_1 `commit_edits` → round_2 `accept`）：循环收敛、轮数正确、输出过 `_validate_strategy`；
- 救援路径真触发：构造一个必死查询（生僻词），验证 R0/R1 在真实 API 上的行为与轨迹记录。

### 4.3 真实 API 冒烟（opt-in `-m real_api`）

- 真 Intern-S2 跑完整策略 Agent（世界模型主题）：断言 LLM ≤7 次、轮 ≤3、策略合法、wallclock 上限内；
- 冷僻主题一条：验证救援阶梯真实触发（从 trajectory jsonl 检查 R0/R2 调用记录存在）。

### 4.4 效果验收：A/B/C 对照实验（证明"改了确实有效"）

**与总计划 §6 的关系**：总计划 §6 是修复关节的端到端实验；本节是**策略关节的局部对照**，共用 trajectory jsonl 基础设施。

**主题集**（固定，写进脚本）：

- 热 ×2：世界模型、具身智能（预期模板也够用——验证"不劣化"）；
- 中 ×1：自选；
- 冷 ×2：预期模板塌方的冷僻主题（如神经符号程序合成一类）。

**三配置**：

| 配置 | 说明 |
|---|---|
| A. template | 现有 demo 模板路径（基线，零 LLM） |
| B. single-shot | 现有 full 单发 LLM 策略（无循环） |
| C. agent-loop | 本方案 |

**指标**（每主题每配置，数据源 = trajectory jsonl + search_strategy.json + 各轮成绩单快照）：

| 指标 | 定义 | 证明什么 |
|---|---|---|
| 方向成活率 | `n_unique ≥ 3` 的 aspect 占比 | 核心效果 |
| 零方向率 | `n_unique = 0` 占比 | 模板塌方指标 |
| 平均相关度 | top10 结果的 embedding 余弦均值 | 查询质量 |
| 方向重叠度 | aspect 间 pairwise Jaccard 均值 | 重复检索 |
| 成本 | LLM 调用数 / SciVerse 调用数 / 墙钟 | 预算守规矩 |

**验收线**：

- 硬断言（自动化进测试）：C 配置所有运行 LLM ≤7、SciVerse ≤20、轮 ≤3；`score(final) ≥ score(round_1)`（回滚保证）；trajectory 完整可解析；
- 方向性结论（人工判读报告）：冷主题上 C > B ≥ A 的方向成活率；热主题上 C 不劣于 B；
- 预期叙事（待数据验证）：冷主题 C 把零方向率显著压低，代价是 +3~5 次 LLM 调用（≈10s）。

**汇总脚本**：`scripts/run_strategy_ab_test.py` —— 跑 3 配置 × 主题集，从 trajectory jsonl 聚合指标，输出 `output/strategy_ab_report.md` 对照表。此报告即面试实验数据的第一块。

### 4.5 回归保护（不破坏现有交付）

- demo 模板快路径行为不变（现有单测全绿）；
- **`FINAL_SEED_PAPERS` showcase 路径零影响**（可复现叙事不破坏——策略 Agent 只活在 full 真实路径）；
- `build_search_strategy` 输出 schema 不变（下游 P1–P6 零改动）。

### 4.6 验收清单（Definition of Done）

- [ ] §4.1 单测全绿（3 个新测试文件）
- [ ] §4.2 集成测试绿（真 SciVerse）
- [ ] §4.3 real_api 冒烟通过（预算内）
- [ ] 冷主题 ≥2 跑通，救援阶梯轨迹可查
- [ ] §4.4 A/B/C 报告产出且硬断言全过
- [ ] §4.5 回归三项确认
- [ ] 同主题二次运行 memory_context 非空（教训真实沉淀）
