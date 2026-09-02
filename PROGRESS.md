# PROGRESS

工作状态文件（人可编辑）。会话开始先读，完成一个有边界的单元或收尾时更新。
协议见 `~/.claude/CLAUDE.md` 的 File-based Session State；非平凡需求的 spec 放 `specs/<topic>.md`。

## Current

### 文件为窗口的非阻塞工作流（本仓落地）

- 任务目标：协作从阻塞式对话框迁到"文件为窗口"——状态进 `PROGRESS.md` / `specs/`，会话边界读写，需求可异步累积。
- 验收条件：
  - 协议写入用户级 CLAUDE.md（已达成）
  - 本仓建立 `PROGRESS.md`，项目 CLAUDE.md 注明 solo dev 与提交约定（已达成）
  - 一个真实需求按 spec → 实现 → 回写跑通全流程

## Next

### 真实需求走通一轮工作流

- 任务目标：用下一个实际需求检验协议摩擦点（spec 粒度、回写时机）。
- 验收条件：
  - 全流程（spec 文件 → 实现 → PROGRESS 回写）走通一次
  - 摩擦点记入 Log

### 给搜索模块加上 SubAgent 工具

- 任务目标：现在的搜索 Agent 模块都是自己去搜索文件，聚合关键词，并且具体搜索。但是在确定要搜索的方向后，其实更好的方式是让一个 subagent 去搜索，这样就不会被中间的一些信息和工具输出影响到上下文。
- 验收条件：
  - 给搜索 Agent 加入 Subagent 工具，并且 Subagent 可用工具列表里不包含 Subagent 不能递归，代码编译要通过。
  - 搜索 Agent 的已有工具以及新增的 Subagent 工具都要能够正常使用。单独的工具单元测试要能够通过。
  - 整体流程的话，对比测试和冒烟测试没有问题。

## Log

### 2026-09-02

- 落地 worktree 并行派发协议：项目 CLAUDE.md 增 "Parallel Dispatch Protocol" 小节；用户级 CLAUDE.md 增通用 "Git Worktree 并行派发" 原则；`PROGRESS.md` 首次提交（worktree 只见已提交内容，此为派发前提）。下一个真实任务（搜索 SubAgent 工具）按协议派发，验证全链路。
- 建立工作流：`PROGRESS.md` + `specs/` + 用户级 CLAUDE.md 协议（含 PROGRESS 模板）；项目转个人开发，本文件与 `specs/` 提交进 git（项目 CLAUDE.md 已注明）。
- 未跟踪文件 `docs/相关工作对比.md` 待归档/提交。
- （承接此前状态）v2 三关节完成：关节1 策略 Agent（`harness/agents/`）、关节2 修复回路 + Goal Gate（`harness/goal_gate.py`）、关节3 coverage 闸；总计划与审计在 `docs/RealAgent/`。
