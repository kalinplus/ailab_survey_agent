# PROGRESS

工作状态文件（人可编辑）。会话开始先读，完成一个有边界的单元或收尾时更新。
协议见 `~/.claude/CLAUDE.md` 的 File-based Session State；非平凡需求的 spec 放 `specs/<topic>.md`。

## Current

### S0 交付件重跑与四层复测（S1+S2 已合并，待真跑）

- 任务目标：S1/S2 合并后用交付级配置重跑一次生成侧（full，开 `EVISURVEY_WRITER_LLM=1` + 广度旋钮），使四层评测对象是自洽工件，与 2026-09-02 基线（L3 1.667）出 A/B。
- 建议配置：`EVISURVEY_MAX_QUERIES_PER_ASPECT=4 EVISURVEY_META_PAGE_SIZE=50 EVISURVEY_MAX_CORPUS=60 EVISURVEY_ASPECT_MIN_PAPERS=2 EVISURVEY_WRITER_LLM=1 EVISURVEY_REAL_NLI=1 --use-mineru`（真跑前先小步冒烟确认旋钮组合不炸）
- 验收条件：
  - 新 run 的 `output/survey.md` 与评测 evidence store 同 id 空间，`id_space_mismatch=false`
  - 语料 ≥30 / 5 类目 / gold recall ≥0.10 / coverage 维持 1.0
  - L0 max sim<0.8、零引用正文节=0、L1.5 support rate≥0.5、L1 precision≥0.8
  - L3 总分较基线回升且三维 rationale 不再指向引用塌缩/复述
  - 重跑后的 showcase 产物（HTML/PDF）仍通过 readiness gates

## Next

（无排队任务）

## Log

### 2026-09-04（下午：S0 真跑 4 轮迭代——LLM 写作 × NLI × 修复回路的三重冲突诊断与修复）

- S0 尝试 1 失败：P5.2 agentic backfill 在语料 60 下 692 次搜索 ~18min，超 `TOOL_TIMEOUT_SECONDS=900`，主进程死而 worker 线程活到写完 bundle——假象"跑完"。教训：长真跑命令里不要加 `; echo`（吞退出码）。
- S0 尝试 2 完成但产物回退（正文 4 节被修复回路吃光）。诊断三重根因：① writer LLM 把 id 裸写进句子且 DOI 内插空格（27 bare vs 8 合法括号），括号白名单过滤看不见 → 全成未引用 claim；② NLI 对 zh LLM 转述 12/15 判 unsupported；③ claim mapper 把 `![caption](id)` 的方括号当引用（cited_paper_id 竟是 caption 文本）→ 垃圾 claim → delete-repair 掏空正文。citation_ready_set 的 `[:max_core_papers]` 硬截是"84 引用 3 唯一 id"的上游机制，`--max-core-papers 15` 放开。
- 修复 commit `98ec405`：writer prompt 改 [Pn] 别名（模型永远见不到可改坏的 id，解析后映射回真实 id）+ 裸 id/DOI 残段句级丢弃 + `HEAVY_LLM_*` env 三元组（writer/修复走 GLM `glm-5.3-flash`，open.bigmodel.cn OpenAI 兼容端点；GLM 是常思考模型，max_tokens 要给足、`thinking` 不支持 disabled）+ claim mapper 剥 embed 行。Intern 留给简单判断任务。
- 尝试 3（en + GLM）：structural invalid=0（别名方案生效，零真泄漏）、gate 到 fixpoint、coverage 1.02——但 GLM 带引用句 4/4 被 NLI 判 unsupported，正文仍被吃到只剩 1 节。结论：NLI 杀转述、亲近引用；claim_map 只提出 4 条（GLM 段内引用稀疏）。
- 尝试 4 进行中：SUMMARY 段改"近乎逐字拼装 evidence snippet + 挂 tag"（评测陷阱 memory 的镜像：verbatim 句过 sentence-window NLI）+ 开 `EVISURVEY_REPAIR_AGENT=1`（关节2 remap/rewrite/backfill 替代 delete-only，正是为 unsupported 设计的机制；`EVISURVEY_NLI_DEVICE=cpu` 防 MPS OOM）。
- 遗留：P2 taxonomy Intern 随机性大（26/4/~2 类目跨 run）；MinerU `/agent/parse/url` 秒回 200 零轮询 → parsed=0（图表链路断，用户指示先放下）；P1 coverage LLM 偶发 JSON 前缀泄漏（有降级）。覆盖闸盲区：retention 以 round 0 为基线，测"相对变矮"测不了"绝对太矮"。

### 2026-09-04（S1+S2 双 worktree 并行派发第二轮：合并完成）

- 第二轮双任务并行：S1 检索广度与来源深度（`specs/检索广度与来源深度.md`）+ S2 写作端混合内核（`specs/写作端混合内核.md`，用户拍板混合内核：骨架模板+正文 LLM grounded+模板兜底）。spec 先行提交（`f9943a8`）规避上轮 worktree 基点坑，派发提示写明先 `git merge --ff-only main`——本轮零基点摩擦。
- S1 合并（`6ba5fe5`，ff）：广度旋钮 5 个 env（默认=交付行为，含 `EVISURVEY_MAX_CORPUS` 显式设置时切 aspect 均衡裁剪）+ MinerU 全文只解析 core 窗口（默认 15）。亮点：`phase5_evidence.py`/`search_strategy_builder.py` 零改动即达标（全文经 P3 parsed 输出自动入 P5）。
- S2 合并（`d3b7fb6`，cherry-pick，分支基点是 S1 合并前）：选文改相关性矩阵+等额配额+有界复用（REUSE_LIMIT=2 硬记账，防塌缩也防空节）；禁词表+节首句去引用+比较段轮转收尾+开放挑战/未来方向/结论改证据句带引用；`EVISURVEY_WRITER_LLM=1` 时正文三段走 Intern-S2（句子级白名单过滤，单节失败回退模板）。final seed 上 max 相似度 0.736→0.566。
- 偏差记录：S2 的 bge 默认关（`EVISURVEY_WRITER_EMBED=1` 开）——单测不得触发 HF 下载；`ASPECT_MIN_PAPERS` 默认 0（交付路径不回归，S0 显式开 2）；S1 的 aspect 下限救援只在 influence 模式生效（full 模式即 influence 模式，可接受）。
- 合并后主工作区 `pytest tests/unit/` 336 passed（318 基线 + S1 +S2 新测试）；两 worktree 与分支已清理。剩余：S0 交付级真跑 + 四层复测（Current 挂着）。

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
