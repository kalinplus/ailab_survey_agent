# 关节 2：修复 Agent + Goal Gate — 落地方案与测试方案

> 日期：2026-08-20 ｜ 状态：**已实施**（实现记录与实验结果见文末 §5–§6）
> 上游：[Agent关节改造-审计与计划.md](Agent关节改造-审计与计划.md) §2（判据）、§3.2/§3.3（本关节概要）
> 本文件是关节 2 的完整实施规格，含三个已定稿决策、落地方案、分层测试与对照验收方案。
> 前置：关节 1 已实现（`harness/agents/loop.py` 底盘 + trajectory 基础设施可复用）。

---

## 1. 问题回顾

综述写完后要过验证（`verify_citations`），挂了就进修复（`revise_survey`）。现状四个问题：

### 1.1 修复只会一招——删除

`tools/revise_survey.py` 的全部动作：

| 失败 | 现状动作 | 代码 |
|---|---|---|
| 引用 id ∉ 白名单 | 抹掉 `[paper_id]` 字符串 | `:79` |
| 图表引用无效 | **整行删除**（相近 caption 的正确图不找） | `:78` |
| claim 不被证据支持 | **整句删除** | `_remove_sentence_containing:138-148` |

修 N 处 = 少 N 句话，章节可能被删出空洞。而失败原因五花八门（id 写错 / 证据没匹配上 / 措辞过强 / 真没证据），对应的合理动作（换 id / 换证据 / 弱化 / 回填）全被堵死。

### 1.2 闭环是半截的

`harness/agent_loop.py:167-180`：verify ① → 不过 → revise → verify ②，**第二次的结果没人读**。修完有没有变好系统不知道；revise 删过头把文章修得更差也照单全收。"循环"只有一轮体，名不副实。

### 1.3 验证信号本身是假的（前提问题）

`verify_citations.py:112-115`：默认 `FakeNLIModel`，关键词命中 `world/model/game/agent` 即 0.9 置信 entailment——世界模型主题几乎句句"自动 supported"。修复环跑在这上面测的是"修关键词匹配"，结论不可信（总计划 §3.4）。

但直接开真 NLI 会暴露级联病理（总计划 §1.5）：demo 模式 `use_mineru` 硬编码 False（`planner.py:68`）→ 证据库近空 → 真模型判全部 unsupported → 修复全删。**真 NLI、use_mineru 可配、修复动作升级必须捆绑实施**。

### 1.4 失败分类的原料早已齐全，无人使用

claim_map 每条已带 `status / confidence / evidence_ids`，citation_result 已带 `valid`。五类失败用现有字段**机械可判**：

| 类型 | 判定字段 | 语义 |
|---|---|---|
| A | `entry.valid == false` 且非图表 | 引用 id 写错/不在白名单 |
| B | `status == "unsupported"` 且该论文证据条目 > 0 | 有证据但都不蕴含（多半是匹配/措辞问题） |
| C | `status == "unsupported"` 且该论文证据条目 == 0 | 零证据 |
| D | `status == "weak"`（neutral, conf 0.4–0.6） | 证据方向对但断言过强 |
| E | 图表 ref ∉ figure_bank/generated_bank | 图表引用错 |

这正是判据 2 的标准场景：失败×修复组合空间大（5×5），硬编码 if-else 会爆炸，LLM 批量选动作才划算。

---

## 2. 三个已定稿决策（2026-08-20 讨论）

### 决策 1：允许 LLM 改写句子（否决"只删不改"）

- LLM 可对失败句执行 `rewrite`：保持 `[paper_id]` 引用不变，把断言降到证据支持的弱形式（"证明了"→"提示了"），或贴合已有证据 chunk 重述。
- **风险兜底（硬约束）**：改写句必须**重过 NLI**（增量验证）才算修复成功；重跑仍 unsupported → 该句进入下一轮换动作或最终删除。防"改写引入新断言"。
- 删除仍是每个动作链的最后一环，且必须带 reason（进 repair_log）。

### 决策 2：`use_mineru` 显式可配，与关节 2 捆绑实施

- `planner.py:68` 的硬编码 `"use_mineru": False` 改为 `build_task_request(..., use_mineru: bool = False)` 参数透传；`main.py` 加 `--use-mineru` CLI flag。
- 不做则 C 类失败全是"零证据"假象（证据库空转），动作修复实验无效。
- 默认值仍为 False：showcase / demo 路径行为零变化；改进运行与实验显式开启。

### 决策 3：Goal Gate v2 只设两道闸，coverage 闸留 v3

```python
gates = {
    "structural_validity": invalid_citations == 0 and invalid_figures == 0 and invalid_tables == 0,
    "unsupported_claims":  unsupported == 0 or repair_budget_exhausted,
}
```

- coverage 下限（最少章节/图表数）涉及交付标准重定，v3 再议；v2 的 `stop_reason` 字段为其预留。
- Gate 是确定性代码持有，LLM 无权判定"通过"（与关节 1 Gate 同一原则）。

---

## 3. 落地方案

### 3.1 总体流程

```
write_survey
   ▼
verify_citations ① ── gate.passed? ──yes──▶ render_report
   │ no（失败清单 = citation_result + claim_map 分组为 A–E）
   ▼
修复轮 r=1..MAX_REPAIR_ROUNDS(2):
   失败分组 → 修复 Agent 批量决策（按类型分组，1 次 LLM 处理 ≤5 条，返回动作数组）
   → 框架执行动作（机械部分代码做：候选挑选/caption 匹配/证据入库）
   → 增量验证（改写句单句重跑 NLI；结构正则全文免费重跑）
   → verify_citations ②（全量，结果这次必须被读）
   → gate.passed? / improved(unsupported)? —— 无进步提前停（不动点）
   ▼
best-so-far（若某轮把文章修得更差 → 回滚最优版本）──▶ render_report
passed / stop_reason 写入 final_state.json + logs/run.jsonl
```

### 3.2 文件与组件

```
harness/goal_gate.py        # Goal Gate + improved() 判定（确定性，独立可测）
harness/agents/repair_agent.py  # 失败分组器 + 批量决策 + 动作执行 + 增量验证
tools/revise_survey.py      # 重写内核：入口/IO/请求协议不变（B/C 边界零改动），委托 repair_agent
harness/agent_loop.py       # 167-180 半闭环 → Gate 循环接线（读每轮 verify 结果）
harness/planner.py + main.py # use_mineru 透传 + --use-mineru flag
output/repair_log.json      # 每条失败 {claim, failure_type, action, outcome, round}
```

**与关节 1 的复用关系（坦诚记录）**：复用 `loop.py` 的 `trajectory_writer`（repair 决策逐条落 `logs/trajectory/{task_id}_repair.jsonl`）、JSON 修复重试模式、预算计数模式。**不复用 `BoundedAgentLoop` 本体**——修复的决策结构是"一次性批量分类"（失败清单有限、组内无顺序依赖，信息一次全给），不是多轮探索；强行套逐轮循环会让"轮"失去意义。外层 Gate 循环（`agent_loop.py`）承担轮次控制，对应关节 1 里框架层的角色。

### 3.3 Goal Gate（`harness/goal_gate.py`，确定性）

```python
@dataclass
class GateResult:
    passed: bool
    stop_reason: str        # passed | structural_fail | unsupported_cleared |
                            # repair_budget_exhausted | fixpoint | coverage_deferred_v3

def evaluate(citation_result: dict, *, repair_round: int, prev_unsupported: int | None) -> GateResult
def improved(current_unsupported: int, prev_unsupported: int) -> bool   # 严格变小才算进步
```

接线（替代 `agent_loop.py:167-180`）：

```python
prev = None
for repair_round in range(MAX_REPAIR_ROUNDS + 1):        # 2 轮修复 + 1 次终验
    citation_result = self._run_tool("verify_citations", ...)
    gate = goal_gate.evaluate(citation_result, repair_round=repair_round, prev_unsupported=prev)
    if gate.passed: break
    if repair_round == MAX_REPAIR_ROUNDS: break           # 预算尽
    if prev is not None and not goal_gate.improved(unsupported, prev): break   # 不动点
    self._run_tool("revise_survey", ...)
    prev = unsupported
final_state["goal_gate"] = {"passed": ..., "stop_reason": ...}   # 交付叙事直接受益
```

### 3.4 修复 Agent（`harness/agents/repair_agent.py`）

**输入**：survey.md + citation_result + claim_map + evidence_store + citation_ready_set + figure_bank + generated_artifact_bank。

**第一步（零 LLM）：失败分组器**。从 claim_map / citation_result 机械分出 A–E 五组（§1.4 表），每组附决策所需的上下文：

- A 组每条附：claim 句 + citation_ready_set 中标题含 claim 关键词的 ≤5 个候选 paper_id（代码挑）；
- B/D 组每条附：claim 句 + 该论文证据中 NLI 分最高的 ≤3 条 chunk 摘要（代码挑）；
- C 组每条附：claim 句 + cited_paper_id；
- E 组每条附：错误 ref + figure_bank 中 caption 词重叠 ≥2 的 ≤3 个候选 figure_id（代码挑）。

**第二步：批量决策（LLM）**。按类型分组，每组一次 `json_chat`（≤5 条/次，返回 JSON 数组，每个元素 `{claim_id, action, params, reason}`）。 Intern-S2 2s 限速下 20 条失败 ≈ 4–5 次调用/轮。

**第三步：动作执行（框架，机械）**：

| action | params | 执行 |
|---|---|---|
| `remap_citation` | `new_id`（必须是候选或白名单成员） | 替换 `[old_id]` → `[new_id]`，句子重过 NLI |
| `swap_evidence` | `evidence_id`（候选内） | 记录新绑定，句子重过 NLI |
| `rewrite_claim` | `new_text`（不得含新 `[id]`，长度 0.5–2× 原句） | 替换句子，重过 NLI |
| `backfill_evidence` | —（C 类） | `agentic_search(claim_text, top_k=3)` → chunk 以 `source_type="agentic_chunk"` 写入 `cache/evidence_store.json`（与 P5 同构，第二轮全量 verify 自然可见）→ 句子重过 NLI |
| `delete_claim` | `reason`（必填） | 删除句（每条动作链的最后手段） |
| `remap_figure`（E 类） | `new_figure_id`（候选内） | 替换 ref；无候选则删行 |

**参数校验**：`new_id/new_figure_id/evidence_id` ∉ 候选集 → 该条动作拒绝，记 `outcome=invalid_action`，下一轮重议。LLM 输出永远不直接信任（防幻觉 id）。

**第四步：增量验证**。改动句集合 → 逐句 `nli.best_match(claim, 该论文全部证据)` → 更新该条 status；结构正则（引用/图表白名单）全文重跑（免费）。注意：**进入下一大轮前仍跑全量 verify**——增量结果只用于动作反馈与 repair_log，最终判定以全量为准，避免增量/全量口径漂移。

**第五步：repair_log 落盘**（`output/repair_log.json` + trajectory jsonl），每条：

```json
{"round": 1, "claim_id": "...", "failure_type": "B", "action": "rewrite_claim",
 "outcome": "repaired | still_unsupported | invalid_action | deleted",
 "before/after 摘要": "..."}
```

### 3.5 预算与熔断

| 项 | 上限 | 触发后 |
|---|---|---|
| 修复大轮 `MAX_REPAIR_ROUNDS` | 2 | 停，best-so-far |
| LLM 调用 | 8/轮、16 全程 | 未决策的失败标 `unresolved`，进删除评估 |
| SciVerse agentic（backfill） | 10/轮、20 全程 | 剩余 C 类直接进删除评估 |
| 删除占比 | 软约束：全程删除数 < 修复失败数 50% 时才允许无理由批量删 | 超限仅告警 + repair_log 记录（不阻塞——诚实呈现比假修复好） |

best-so-far：以 `(unsupported 数, citation_validity_score, 保留字数)` 字典序比较各轮产出，回滚最优版本。

### 3.6 环境变量

- `EVISURVEY_REPAIR_AGENT`（默认 **0/off**）：开启后 revise 走修复 Agent；关闭保持现状删除式修复。与关节 1 不同（那个默认开）——revise 在 demo/showcase 也会跑，默认关保证交付叙事零风险，改进运行显式开。
- `EVISURVEY_REAL_NLI=1`：**实验与改进运行的硬前提**（不改默认值，避免破坏 demo；文档与实验脚本必须显式设置）。
- `--use-mineru` CLI flag（决策 2）。

### 3.7 实现切片（每片独立可验证）

| # | 切片 | 验收 |
|---|---|---|
| 1 | Goal Gate + agent_loop 循环接线 + use_mineru 透传 | 单测绿；二次 verify 结果确实被读（构造 round1 fail → round2 pass → 循环正确退出且 final_state 带 stop_reason） |
| 2 | 失败分组器 + 批量决策骨架（fake LLM） | 单测绿：五类分组正确；LLM 输出解析/参数校验/非法动作拒绝 |
| 3 | 动作执行器五类 + 增量验证（fake NLI/SciVerse） | 单测绿：每类动作执行正确；改写句重过 NLI；删除带 reason |
| 4 | 真接入：真 NLI 工厂复用 + agentic backfill + repair_log/trajectory 落盘 | 集成绿（真 NLI + fake LLM）；repair_log schema 完整 |
| 5 | 端到端对照实验（§4.4） | 三配置 × 主题集报告产出，硬断言全过 |

---

## 4. 测试方案

总原则同关节 1：**单元/集成证明"代码对"，对照实验证明"设计值"**。

### 4.1 单元测试（fake LLM / fake NLI / fake SciVerse，无网络）

| 文件 | 测试点 |
|---|---|
| `tests/unit/test_goal_gate.py` | ① 全过 → passed，stop_reason=passed；② structural 失败 → 不 passed；③ unsupported>0 且预算尽 → stop_reason=repair_budget_exhausted；④ improved：unsupported 严格下降才算进步（相等/上升 → 不动点停）；⑤ agent_loop 接线：round1 fail → revise → round2 pass → 循环退出、final_state.goal_gate 正确、**第二次 verify 结果被消费**（回归 §1.2）；⑥ round2 无进步 → 提前停不再跑第三轮 |
| `tests/unit/test_repair_agent.py` | ① 分组：构造混合 claim_map/citation_result → A–E 各归各位，候选附带正确；② 批量决策：fake LLM 返回动作数组 → 正确分发；③ 参数校验：`new_id` ∉ 候选 → invalid_action，不执行；④ remap/rewrite/swap 执行后句子确实变化且 `[paper_id]` 合法；⑤ 改写句重过 NLI（fake NLI 可剧本化翻案/不翻案两种）；⑥ backfill：fake SciVerse 返回 hits → chunk 入 evidence_store（`source_type="agentic_chunk"`）→ 单句重判；⑦ 删除必须带 reason，repair_log 记录；⑧ 预算熔断：LLM/SciVerse 上限触发 → 未处理失败标 unresolved；⑨ best-so-far：round2 更差 → 回滚 round1 版本；⑩ `EVISURVEY_REPAIR_AGENT=0` → 行为与旧删除式修复等价（回归）；⑪ E 类：caption 重叠候选匹配 |

Fake 契约对齐：fake NLI 镜像 `NLIVerifier.judge/best_match`；fake SciVerse 镜像 `agentic_search → {"hits": [...]}`（chunk/page_no/doc_id 字段与 P5 一致）。

### 4.2 集成测试（真 NLI + fake LLM）

`tests/integration/test_repair_agent.py`（`RUN_NLI_REAL=1` 风格 opt-in 或独立 marker，cross-encoder 首次下载 ~700MB）：

- 真模型判定一组真实构造的 supported/weak/unsupported 句 → 分组正确；
- 修复动作端到端：rewrite 后真 NLI 翻案；真 agentic backfill（真 SciVerse）chunk 可蕴含 claim。

### 4.3 真实 API 冒烟（opt-in）

真 Intern-S2 + 真 NLI + 真 SciVerse 全环（`EVISURVEY_REPAIR_AGENT=1 EVISURVEY_REAL_NLI=1`，full 模式小参数）：预算内完成、final_state.goal_gate 有 stop_reason、repair_log 可解析。

### 4.4 效果验收：三配置对照实验

**主题集**：世界模型（热，回归）+ 2 个中/冷主题（预期失败多，修复价值大）。

| 配置 | 说明 |
|---|---|
| A. no-repair | 跳过 revise（验证失败直接渲染，暴露原始失败量） |
| B. delete-only | 现状删除式修复 |
| C. action-repair | 本方案 |

**指标**（数据源 = repair_log + 各轮 citation_result/claim_map 快照）：

| 指标 | 定义 | 证明什么 |
|---|---|---|
| unsupported 残留 | 终轮 unsupported 数 | 核心效果（三配置应 C < B < A） |
| 保留率 | 修复后正文字数 / 修复前 | B 的删除代价 vs C 的保留 |
| citation_validity | 终轮结构分 | 不劣化 |
| 动作分布 | repair_log 按 action 统计 | 删除是少数（设计意图验证） |
| 成本 | LLM / SciVerse / 墙钟 | 预算守规矩 |

**硬断言（自动化进脚本）**：C 配置 LLM ≤16、SciVerse ≤20、轮 ≤2；`unsupported(final) ≤ unsupported(round_1)`（best-so-far 保证）；repair_log 完整可解析；每条 delete 有 reason。

**汇总脚本**：`scripts/run_repair_ab_test.py` → `output/repair_ab_report.md`（对齐关节 1 的 `run_strategy_ab_test.py` 结构）。

### 4.5 回归保护

- `EVISURVEY_REPAIR_AGENT` 默认 off → demo/showcase/交付路径行为与现状完全一致（现有单测全绿即证）；
- `use_mineru` 默认 False 不变，仅新增显式开启通道；
- revise_survey 请求协议（inputs/outputs 字段）零改动，B/C 边界不动；
- `FINAL_SEED_PAPERS` 路径零影响。

### 4.6 验收清单（Definition of Done）

- [ ] §4.1 单测全绿（2 个新测试文件 + agent_loop 接线测试）
- [ ] §4.2 集成测试绿（真 NLI）
- [ ] §4.3 real_api 冒烟通过（预算内，goal_gate 有 stop_reason）
- [ ] §4.4 三配置报告产出且硬断言全过
- [ ] C 配置修复后 unsupported 残留 < B（删除式）在 ≥2 个主题上成立
- [ ] §4.5 回归四项确认
- [ ] repair_log / repair trajectory 落盘可复盘

---

## 5. 明确不做（v2 范围外）

- coverage 闸（决策 3，v3）；
- 修复教训入 memory（关节 1 的 lessons 机制可后续扩展，v2 先靠 repair_log）；
- `write_survey` 模板 → LLM 生成（生成质量轴，威胁 FINAL_SEED_PAPERS 叙事，v3 渐进）；
- mid-pipeline 反馈（需拆 B worker，v3）。

---

## 6. 实现记录（2026-08-20）

代码落地与规格一致。实现层补充（代码注释有说明）：

1. **连带 claim 去重**（对 §3.4 的重要修正）：引用非法 id 的 claim 是 A 类失败的连带伤害——一个 id 被 survey 引用几十次时，逐条进 C 组决策既浪费预算又逼 LLM 编 id（首跑 30 条 invalid_action 的教训）。分组器现将其跳过，A 类一次 remap 修复全部出现，下一轮全量 verify 自然重新归类。
2. **修复 notes 禁用方括号**：`[B] action -> outcome` 格式的 notes 行会被验证器的引用提取器当成 paper id，凭空造出几十条假 claim（首跑 unsupported 55→72 的元凶）。
3. **rewrite 保留句末标点**：否则相邻句子会拼在一起，改变 claim 切分。
4. **MPS 默认**：`NLIVerifier` 自动选 mps（实测 ~2.7×），`EVISURVEY_NLI_DEVICE` 可覆盖。

**文件**：`harness/goal_gate.py`（新）、`harness/agents/repair_agent.py`（新）、`tools/revise_survey.py`（双内核，协议不变）、`harness/agent_loop.py`（`_verify_repair_loop` 替代半闭环 + best-so-far 回滚 + goal_gate 写入 final_state）、`planner.py`/`main.py`（use_mineru 透传 + `--use-mineru`）、`tools/nlp/nli_verifier.py`（设备选择）。

**测试**：单测 243 passed（新增 test_goal_gate.py 12 + test_repair_agent.py 11）；集成 `tests/integration/test_repair_agent.py` 2 项（真 NLI 翻案 + 真 SciVerse 回填，`RUN_NLI_REAL=1 -m nli_real`）；对照脚本 `scripts/run_repair_ab_test.py` → `output/repair_ab_report.md`。

**实验结果（热/中主题完整，冷主题 C 因进程 OOM 中断，报告如实标注）**：
- 结构失败两配置均清零；C 的 remap 对重复引用的非法 id 杠杆最大（一次修全部出现）。
- unsupported 残留 B(6,6) < C(7,34)——但 B 靠删句（claim 物理消失），C 修复可修复的（冒烟：rewrite 30 条 29 条真 NLI 翻案）并诚实暴露剩余。
- DoD 原指标"C 残留 < B"未达成，属指标设计偏乐观（删除口径天然占优）；v3 建议以 unsupported+正文保留率联合评分。

**遗留（v2.1）**：best_match 证据列表上限（OOM 根因：回填后大列表 × 全量 verify 的内存峰值）；对照脚本共享三配置的初始 verify（省 1/3 时间）；write_survey 引用纪律（初始 invalid 63–280 的根因，生成质量轴）。
