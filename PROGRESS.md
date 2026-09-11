# PROGRESS

工作状态文件。常驻契约与 v2 改进总览见 `specs/EviSurvey-v2-契约与改进总览.md`；成果汇报见 `docs/改进成果汇报.md`；模块设计见 `docs/RealAgent/`。

## Current

（空；下一波候选见 Next，待批准。）

## Next

### 策略关节 native tool calling 真跑 A/B（切换默认的前置）
- 任务目标：`EVISURVEY_AGENT_TOOLS=1` 已落地（底盘双模式 + InternS2Client.tool_chat + 策略关节接线，默认仍 JSON 模式）；需一次真跑对照（native vs JSON 两臂，同 topic 同预算），确认 finish 率、report-card 分、LLM 调用效率不劣于现状后把默认切到 native。
- 验收条件：两臂 trajectory 均落盘可对比；native 臂 finish 率与 global_score 不低于 JSON 臂（或差距显式归因）；决策写入本文件后切换默认值并更新 AGENTS.md。

### 波9 候选1：writer 装配层落盘与失败归因
- 任务目标：`_assemble_claim_paragraphs` 的写侧防御（sanitize 白名单/跨节去重/单句重解析）整体丢弃句子导致回退时，把原始回复+逐句丢弃原因落盘——波8-final2 中验证级拒绝已清零，4/9 单元失败全部后移到这一层且无归因。
- 验收条件：装配层整体失败也写 writer_llm_rejects.jsonl；真跑 LLM 存活单元 ≥6/9 或落盘数据可归因。

### 波9 候选2：章节-论文分派精度与 P2 方差
- 任务目标：P2 分类数 3→6→4 纯方差摊薄正文；关键词弱匹配错位（AvalonBench 胜率句落 Neural Rendering 节等）是 topic_relevance 0.656<0.7 的主要剩余拖分。
- 验收条件：分派错位句显著减少或分派带相关性校验；真跑 topic_relevance ≥0.7 或失败原因显式改判。

### 波9 候选3：own_limitation 提取覆盖
- 任务目标：过闸 own_limitation 每轮仅 0-2 条，Future Directions 持续空转；瓶颈在提取端（P5.1）对 selected 论文的覆盖。
- 验收条件：selected 论文过闸 own_limitation ≥3 条；FD 节有带引用句或显式归因。

## Log

### 2026-09-11 策略关节原生 tool calling 验证与落地（双模式，默认关）
用户质疑底盘 JSON typed-action 太古早，要求验证原生 Chat Completions tool calling。`scripts/probe_tool_calling.py` 对两真实端点实测：Intern-S2 与 DeepSeek Flash 均完整支持结构化 `tool_calls`（参数精确合法）、紧 512 token 预算下思考不烧工具调用（Intern reasoning ~351 chars 走独立字段）、`tools`+`response_format=json_object` 共存（DeepSeek 报错仅因 json_object 要求 prompt 含 "json" 字样，与 tools 无关）、`role=tool` 回传闭环正确；`thinking:disabled` 仍被 Intern 忽略但不影响。据此落地：`loop.py` 双模式底盘（`tool_schemas`+`llm_tool_chat`，白名单/预算/轨迹/闸门全共用）、`llm_client.py` 增 `tool_chat()`（`_post_chat` 改返回完整 message）、策略关节接线（schemas 从 prompt 提炼 + `_NATIVE_TOOLS_ADDENDUM` 覆盖 JSON 教学文案，`EVISURVEY_AGENT_TOOLS=1` 开启，默认 off 不改交付行为）、FakeLLMClient 镜像 `tool_chat`。571 单测全绿（新增 7 个 native 用例；1 个旧断言文案随修复提示措辞更新，行为不变）；两端点 3 轮真 API 烟测 PASS（increment+3/+4/finish 精确执行）。probe 报告在 `output/tool_calling_probe_report.json`。

### 2026-09-11 波8 终验收通过（goal gate 全绿）
两次真跑：第一次诚实失败（entry 补收违反 supports_claims NLI 闸契约 → 9 条 invalid_source_binding 级联 → coverage_fail 拦截+回滚，波8b 撤销并测试钉住契约）；第二次全绿：**0 unsupported、0 角色违规、validity 1.0、coverage 1.0/1.0、grounding 0.574→0.9、L1 R/P 0.833、L3 3/3/3**。writer 拒绝前沿三轮演进（unknown_cite_alias→source_role_out_of_scope→装配层），前两层已清零。P2 占位描述治理连带检索去污（RAP/LLM 综述/SMART-LLM 退出引用，topic_relevance 0.62→0.656）。验收基准 `archive/wave8-final2-20260911-115240`。

### 2026-09-10 波1–波7：证据契约、Agent 关节、管线健壮性全部落地
claim 三件套契约（source_role/source_quote/evidence_ids + NLI 闸 + mapper 逐字复核）；策略/修复/Goal Gate 三关节（有界循环、A–F 失败分组、三道闸+回滚）；MinerU 台账恢复与 content 双通道；三道主题闸门 + 经典种子注入；agentic 死通道关闭（P5.2 12min→80s）；taxonomy/图表/PDF 渲染修复。各波详细取证在 git 历史（`backup/pre-polish-20260911` 保留改写前完整记录）与本地 archive/。

### 2026-09-08 批次 T1–T12：从 grounding 0.042 到 0.476
MinerU 异步轮询修复、SciVerse 精确标题检索、goal gate 诚实化（quality_failed+非零退出）、final-seed 测试污染修复、GLM/DeepSeek 空回复根因（推理烧尽预算）与 thinking disabled 路由定型。
