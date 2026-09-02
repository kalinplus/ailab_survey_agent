# PROGRESS

工作状态文件（人可编辑）。会话开始先读，完成一个有边界的单元或收尾时更新。
协议见 `~/.claude/CLAUDE.md` 的 File-based Session State；非平凡需求的 spec 放 `specs/<topic>.md`。

## Current

（无进行中任务）

## Next

### 交付件重跑与评测联动

- 任务目标：评测 v2 首跑暴露 delivered `output/survey.md` 工件真实缺陷（84 处引用仅 3 个唯一 id、小节相似度 max 0.952、纯描述无批判分析 → L3 informational/guidance 1/5）。用交付级配置重跑一次生成侧（showcase demo 或 full），使四层评测对象是自洽工件，验证 L3 分数回升、id 失配告警消失。
- 验收条件：
  - 新 run 的 `output/survey.md` 与 `cache/final_*` 同 id 空间，`id_space_mismatch=false`
  - 四层全量真跑通过，L3 三维 rationale 指向的缺陷不再是引用塌缩/复述
  - 重跑后的 showcase 产物（HTML/PDF）仍通过 readiness gates

## Log

### 2026-09-02（下午：双 worktree 并行派发落地）

- 双任务并行派发全流程走通：评测体系 v2 四层改造（spec `specs/评测体系v2-离线四层改造.md`）+ 搜索 SubAgent 工具（spec `specs/搜索SubAgent工具.md`，派发前由主会话补薄设计并提交）。触达文件集不相交，两 worktree 并行零冲突；各 cherry-pick 单 commit 合并（`eeefb94` SubAgent、`a444ee1` 评测 v2）。
- 评测 v2 真跑验收通过：`EVISURVEY_REAL_NLI=1` 全量四区块报告（`output/survey_eval_report.md`）。L1.5 缓存续跑在真实中断场景验证成功（上次被杀 run 留下 18 条缓存命中，本次仅新检索 19 条）；id 失配告警按设计触发（`10.1109/...` vs `alphastar_2019`）。L3 打分 1.667/5 且 rationale 与 L0/L1 数据交叉印证（84 引用 3 唯一 id、相似度 0.952、零批判）——judge 正确，暴露的是工件本身缺陷，评测体系价值首验。
- SubAgent 真跑验收通过：`main.py` full 冒烟 exit 0、`meta["subagent"]` 恒存在（delegations=0）；A/B/C 对比全硬断言 PASS，C 配置 survival 全面优于 A/B（1.0 vs 0.25–0.5）。Intern-S2 在 10 个真实场景零自发委托（aspects 健康时不需深挖；委托机制由 4 条单测覆盖：正例摘要/无递归/共享预算/上限拒绝）。
- 工作流验收："文件为窗口"三条全达成——协议写入用户级 CLAUDE.md、本仓建立 PROGRESS/specs 并提交、真实需求（本双任务）spec→实现→真跑→回写走通。
- 摩擦点（供协议迭代）：
  - worktree 基点落后 main（0b909d0），agent 需自行 ff-merge 才能看到派发前提交的 spec——派发前应确认 Agent tool 的 worktree 基点行为或派发说明里写明基点 commit。
  - 真跑验收有顺序依赖：评测先于 smoke（smoke 覆盖 `output/survey.md` 会破坏评测的工件配对）；多任务真跑要按数据依赖排序，不是简单串行。
  - 后台长命令输出经管道缓冲后中途不可观测，被杀时零诊断信息——长真跑一律 `python -u` 直出。
  - review 发现的问题（评测 id 失配裸 0.0）通过 SendMessage 回传 agent amend 单 commit 解决，保持一任务一 commit，闭环顺畅。
- 清理：删除已合入 main 的遗留分支 `feat/module-b-knowledge-pipeline`；两个 worktree 与 `worktree-agent-*` 分支已清理。`docs/相关工作对比.md` 仍未跟踪，待用户决定归档方式。

### 2026-09-02

- 评测体系 v2 立项：诊断确认现有三层过度集中于引用核查（L1 与线上 verify 同源、L2 依赖 gold、L3 单 judge 偏置），最大盲区是 citation_coverage≈0.4–0.5 的未引用正文无人核查。定稿四层方案（确定性画像 / corpus-grounded coverage / 未引用句验证 / DeepSurvey 三维 judge；AHA 明确不做），spec 在 `specs/评测体系v2-离线四层改造.md`，按并行派发协议出 worktree。
- 工作流迁移模板落地 `~/.claude/templates/workflow-bootstrap.md`：引导 prompt + 项目协议节填空模板，供新仓库快速迁移（协议本体在用户级 CLAUDE.md 全局生效，迁移仅项目级四件）。
- 落地 worktree 并行派发协议：项目 CLAUDE.md 增 "Parallel Dispatch Protocol" 小节；用户级 CLAUDE.md 增通用 "Git Worktree 并行派发" 原则；`PROGRESS.md` 首次提交（worktree 只见已提交内容，此为派发前提）。下一个真实任务（搜索 SubAgent 工具）按协议派发，验证全链路。
- 建立工作流：`PROGRESS.md` + `specs/` + 用户级 CLAUDE.md 协议（含 PROGRESS 模板）；项目转个人开发，本文件与 `specs/` 提交进 git（项目 CLAUDE.md 已注明）。
- 未跟踪文件 `docs/相关工作对比.md` 待归档/提交。
- （承接此前状态）v2 三关节完成：关节1 策略 Agent（`harness/agents/`）、关节2 修复回路 + Goal Gate（`harness/goal_gate.py`）、关节3 coverage 闸；总计划与审计在 `docs/RealAgent/`。
