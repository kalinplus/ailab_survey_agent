# 关节 3：Global Gate（coverage 闸补全）— 落地方案与测试方案

> 日期：2026-08-21 ｜ 状态：**设计定稿，待实施**
> 上游：[Agent关节改造-审计与计划.md](Agent关节改造-审计与计划.md) §3.3（三闸设计）、[关节2-修复Agent-方案与测试.md](关节2-修复Agent-方案与测试.md) 决策3（coverage 留 v3）与 §6 实验教训（v2.1 遗留）
> 前置：关节 1、关节 2 已实现（`harness/agents/` + `harness/goal_gate.py` 两道闸运行中）。
> 本文件是关节 3 的完整实施规格。它是三件改造里最小的一件：**一个确定性闸的补全，不含任何 LLM 环节**。

---

## 1. 问题回顾

### 1.1 总计划的三闸设计，v2 只落了两道

总计划 §3.3 的 Goal Gate 原文：

```python
gates = {
    "structural_validity": invalid_citations == 0,
    "unsupported_claims":  unsupported == 0 or repair_budget_exhausted,
    "min_coverage":        sections >= N and figures >= M,
}
```

关节 2 决策 3 只实施前两道（`harness/goal_gate.py:34-51`），coverage 闸留 v3，`stop_reason` 字段与 docstring 均已预留。

### 1.2 关节 2 实验暴露的病态：删除掏空无人拦

对照实验（关节 2 §6）：删除式修复 B 的 unsupported 残留（6, 6）低于动作修复 C（7, 34）——B"获胜"纯因 claim 被物理删除。现有两道闸的判定口径**天然偏向删除**：删光 = unsupported 归零 = passed。交付物被掏空没有任何信号拦截。v2.1 遗留建议"以 unsupported + 正文保留率联合评分"。

### 1.3 代码事实：总计划原文的 N/M 绝对下限拦不住掏空

- 修复只删句子，不删章节标题——标题由 `write_survey.py:379-381` 模板生成，`sections >= N` 在修复循环里恒真；
- figures 数由上游 `figure_bank` / `generated_artifact_bank` 决定，修复动作不改变它；
- 删除式 revise 对无效图表引用**删整行**（关节 2 §1.1），所以图表引用数会随修复劣化；
- 真正随修复单调劣化、且直接度量"掏空"的量只有两个：**正文字数**与**图表引用数**。

结论：coverage 闸的信号必须是**相对基线的保留率**，而非绝对计数。

---

## 2. 三个已定稿决策（2026-08-21 讨论）

### 决策 1：范围取窄——只补闸，不动 render_report

`goal_gate.py` 第三道闸 + `_verify_repair_loop` 接线 + `stop_reason` 扩展 + 测试。`render_report` 的 `review_report.json` 10 道 hard_gates（`render_report.py:304-351`）维持"评审报告"定位不前置：其可前置子集（citation_validity / evidence_grounding / figure_table_legality / reference_integrity）与 verify 信号重叠，readiness 子集（文件存在性等）只能在 render 后算。**Global 的含义 = 判定对象从"验证指标"扩展到"交付物形态"**（保留率），并由它统一持有修复循环的停止条件。

### 决策 2：信号 = 相对基线保留率（否决绝对下限 N/M 与每节内容量）

以 round 0（write_survey 后首次 verify 时）的 survey.md 为基线：

- `text_retention = len(now) / len(baseline)`；
- `figure_ref_retention = refs(now) / refs(baseline)`（`![alt](path)` 计数，语义对齐 `tools/verify/structural.py:15` 的 `FIG_RE`）；
- 两比例均 ≥ 阈值 R（默认 0.7，`EVISURVEY_COVERAGE_MIN` 可配）才过闸。

理由：跨主题稳（冷主题基线小、下限也小）；直接拦"删除掏空"；与 `_repair_rank`（`agent_loop.py:307-317`）已有的 `-len` 保留字数一脉相承。否决项：绝对下限 N/M（§1.3 已证失效）；每节内容量下限（需解析章节结构、每节阈值单独定，复杂度高一档收益边际）。

### 决策 3：硬闸语义（否决软标注与半硬分支）

coverage 破线 → `passed=False` + **立即停修**（`stop_reason=coverage_fail`，不再进下一轮 revise——再修只会继续删）+ best-so-far 回滚照旧。render 仍无条件执行（交付不阻塞），`final_state.json` / `run.jsonl` 如实标注。与"诚实呈现优于假修复"原则一致；也否决了"仅在 unsupported 已清零时生效"的半硬分支——unsupported>0 时继续修同样会加深掏空，提前停 + 回滚（回滚目标天然兼顾保留率）更符合实验教训。

---

## 3. 落地方案

### 3.1 `harness/goal_gate.py` 接口扩展

```python
STOP_COVERAGE = "coverage_fail"
MIN_RETENTION_DEFAULT = 0.7

@dataclass
class GateResult:
    passed: bool
    stop_reason: str
    unsupported: int
    invalid_citations: int
    citation_validity_score: float
    coverage: dict[str, Any] | None = None   # {text_retention, figure_ref_retention, min_retention, ok}

def evaluate(metrics, *, repair_round, max_repair_rounds=MAX_REPAIR_ROUNDS,
             prev_unsupported=None,
             text_retention=1.0, figure_ref_retention=1.0,
             min_retention=MIN_RETENTION_DEFAULT) -> GateResult

def figure_ref_count(md: str) -> int   # 纯辅助：![...](...) 计数
```

判定顺序：**passed**（三闸全过：invalid==0 ∧ unsupported==0 ∧ coverage_ok）→ **coverage_fail**（新，失败诊断里最高优先级）→ budget → fixpoint → repairing。

设计要点：

- `figure_ref_count` 在 goal_gate 内自持 2 行正则，**不 import `tools/`**——A 层（harness）不新增对 B/C 层的直接依赖（现有 agent_loop 只经 tool_registry 触达工具）。
- goal_gate 保持**零 env、纯函数**：`EVISURVEY_COVERAGE_MIN` 由调用方（agent_loop）读取传参，可测性不变。

### 3.2 基线快照与信号计算（`agent_loop._verify_repair_loop`）

- round 0 首次 verify 后快照 `baseline_chars = len(survey_text)`、`baseline_figure_refs = figure_ref_count(survey_text)`；
- 每轮算两个 retention 传入 `evaluate`；**除零守卫**：基线为 0 → 该比例记 1.0（空产物不因这道闸误报）；
- round 0 自身 retention 恒 1.0 → 闸只在 round ≥ 1 可能触发，行为上天然只拦"修复造成的掏空"，与 write_survey 的产出质量无关。

### 3.3 循环接线（约 8 行改动，无新分支）

`coverage_fail != STOP_REPAIRING` → 现有 `break` 自然生效。best-so-far 排序与回滚逻辑零改动。闸对两种 revise 内核（默认删除式 / `EVISURVEY_REPAIR_AGENT=1` 修复 Agent）一视同仁——它正是拦"删除式靠掏空在旧闸下获胜"的那道闸。

### 3.4 配置

`EVISURVEY_COVERAGE_MIN`（默认 0.7 = 容忍删 30% 正文），agent_loop 读取传参。showcase / `FINAL_SEED_PAPERS` 路径 round 0 即 passed、coverage 恒 1.0，**零行为变化**。

### 3.5 实现切片（每片独立可验证）

| # | 切片 | 验收 |
|---|---|---|
| 1 | goal_gate 扩展（`STOP_COVERAGE` + `coverage` 字段 + `figure_ref_count` + 判定顺序） | 新单测绿（§4.1 前半） |
| 2 | agent_loop 接线（基线快照 + retention + env 传参） | 接线单测绿（§4.1 后半） |
| 3 | 全量回归 + 文档同步（CLAUDE.md、总计划、本文件实现记录） | 全量单测绿 + 三处文档更新 |

---

## 4. 测试方案

### 4.1 单元测试（`tests/unit/test_goal_gate.py` 新增，fake 一切，无网络）

evaluate 层：

| # | 测试点 |
|---|---|
| ① | 三闸全过（invalid=0、unsupported=0、retention≥R）→ passed=True，stop_reason=passed |
| ② | unsupported==0 但 text_retention<R → passed=False，stop_reason=coverage_fail（回归关节 2 B 配置病态：删光不算过） |
| ③ | figure_ref_retention<R 单独破线同样触发 coverage_fail |
| ④ | 优先级：coverage 破线且预算尽 → stop_reason=coverage_fail（诊断优先于 budget） |
| ⑤ | 默认参数：不传 retention 时旧行为完全不变（既有 12 个测试即回归证明） |
| ⑥ | `figure_ref_count`：带/不带图、多图计数正确 |

接线层（`_verify_repair_loop`）：

| # | 测试点 |
|---|---|
| ⑦ | round 0 恒过闸（retention=1.0） |
| ⑧ | 除零守卫：基线 refs=0 → 比例 1.0，不误报 |
| ⑨ | **核心场景**：round 1 revise 大幅删文（>30%）→ 循环退出、`final_state.goal_gate.stop_reason=coverage_fail`、survey.md 回滚为 best-so-far 版本 |
| ⑩ | `EVISURVEY_COVERAGE_MIN` 传参生效（调高阈值 → 同场景触发） |

### 4.2 兼容性风险与处置

既有单测若构造"revise 后正文大幅缩水"的场景，可能意外触发新闸：跑全量回归确认；个别冲突测试显式传 `min_retention=0.0` 或更新断言（预期改动量极小）。

### 4.3 验收清单（Definition of Done）

- [ ] §4.1 新单测全绿（evaluate 层 6 + 接线层 4）
- [ ] 全量单测回归绿（原 243+ 不破）
- [ ] 接线测试证明删除掏空被拦：coverage_fail + 回滚 + final_state 如实标注
- [ ] showcase / `FINAL_SEED_PAPERS` 路径零行为变化（coverage 恒 1.0）
- [ ] 文档同步：本文件实现记录、CLAUDE.md、总计划勾掉 v3 项

---

## 5. 明确不做（关节 3 范围外）

- render_report hard_gates 前置（决策 1 否决）；
- 每节内容量信号（决策 2 否决）；
- coverage 破线后的"补章节/补图表"修复动作——修复动作空间无此动作，章节数由上游 taxonomy 决定，属 write_survey / mid-pipeline 轴（总计划 v3 另立项项）；
- best-so-far 排序改造（`_repair_rank` 已含 `-len`，与闸方向一致，不重复加权）。

---

## 6. 实现记录

（实施后补）
