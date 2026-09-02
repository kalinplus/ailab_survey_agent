# 搜索模块 SubAgent 工具

状态：spec 已确认（2026-09-02），待实现。
定位：给关节1 策略 Agent（`harness/agents/strategy_agent.py`）加一个可委托的探索型 SubAgent 工具，让主线 Agent 上下文不被中间检索结果污染。

## 动机

策略 Agent 现在 自己 亲自调 `probe_query` / `discover_by_description`，原始命中（titles、keywords、样本摘要）直接进主循环上下文；方向确定后的深度探索（多查询试探、挖字段关键词）属于可外包的中间过程。把它委托给一个有界 SubAgent，主线只回收一份紧凑摘要（建议关键词 + 命题题目），上下文干净且预算仍受控。

## 设计（薄全局）

- 实现位置：`harness/agents/strategy_agent.py` 内部，不新建模块。工具以闭包方式拿到既有的 `budget`（`_CountingSciverse`）、`llm_json_chat`、trajectory writer、`end_year`。
- 新 action：`{"action": "delegate_search", "goal": "<自然语言探索目标>", "queries": ["...", ...]}`。加入主 toolbox（权限边界）。
- SubAgent = 一个独立 `BoundedAgentLoop`（复用 `harness/agents/loop.py`）：
  - toolbox：`search_papers`（走共享 budget 的 meta_search，filters=none，page_size≤8，返回挖掘后的 title/year/keywords 紧凑列表，不回传原始 hits）、`discover`（走共享 budget 的 agentic_search，复用 `_mine_discovery_hits`）、`report_findings`（`LoopFinished` 收尾，payload = 紧凑摘要）。
  - **无递归**：SubAgent toolbox 不含 `delegate_search`，也不含主线的 `commit_edits` / `accept`（策略编辑权只在主线）。
  - 自有预算：`max_turns=4`、`max_llm_calls=3`、wallclock ≤ 90s。SciVerse 调用走同一个 `_CountingSciverse`（全局预算天然受限）；`_BudgetExceeded` 在工具里转 error observation，SubAgent 据此收尾，主线循环不中断。
  - 调用次数上限：每次 run 最多委托 2 次（闭包计数器），超出返回 error observation。
  - trajectory：与主线同一个 jsonl writer，`agent: "strategy_subagent"`，可区分可回放。
- 回收摘要 schema（主线 Observation，经 `_digest` 截断）：`{"n_queries", "papers": [{title, year} ≤8], "field_keywords": ≤10, "notes"}`。
- 可观测：`meta["subagent"] = {"delegations": n, "llm_calls": n}`；SubAgent 的 sciverse 调用已并入 `meta["sciverse_calls"]`。
- prompt 接线：`STRATEGY_SYSTEM` 增加 `delegate_search` 说明与使用时机（R2/R3 失败后的深度探索、新方向试探），并注明它消耗委托次数、结果只含摘要。
- 不加 env 开关：工具恒在 toolbox，LLM 自主决定是否委托；预算由上限兜底。

## 验收条件

worktree 内判定（零网络，FakeSciverse / ScriptedLLM 模式，见 `tests/unit/test_strategy_gate.py`）：

1. `pytest tests/unit/` 全绿；`python -c "import harness.agents.strategy_agent"` 通过（编译/导入级验收）。
2. 新单测（加在 `tests/unit/test_strategy_gate.py`）：
   - delegate_search 正例：主线调用后收到紧凑摘要 payload；SubAgent 内部多轮 search 的原始输出不进主线 observation。
   - 无递归：断言 SubAgent config.tools 不含 `delegate_search`（实现暴露的工具名集合可检）。
   - 共享预算：SubAgent 的 sciverse 调用计入 `budget.calls`；预算耗尽时返回 error observation 且主线继续。
   - 委托上限：第 3 次委托被拒。
   - 既有工具（probe_query / discover_by_description / commit_edits / accept）存量断言保持绿。
3. 冒烟（合并后主工作区做，主会话执行）：`python main.py --topic "世界模型综述" --max-papers 5 --max-core-papers 3 --mode full` 全链路无错，trajectory 里可见 strategy_subagent 记录（LLM 若不委托则记录 zero-delegation，不算失败）。
4. 对比测试（合并后主工作区做，主会话执行）：`python scripts/run_strategy_ab_test.py` 跑通，SubAgent 开启不劣于基线（global_score 不降）。

## 边界 / 不做

- 不动 `loop.py` 底盘（SubAgent 是它的第二个消费者，不是底盘改动）。
- 不动 repair_agent、P3 检索、tool_registry——只在关节1 内部。
- 不做并行多 SubAgent、不做 SubAgent 嵌套 SubAgent。
- 真跑对比（验收 3/4）合并后主工作区一次，不在 worktree 内做（协议默认）。
