# EviSurvey v2：契约与改进总览

本文档是 v2 改进阶段（2026-09-08 至 2026-09-11）的**唯一总览 spec**，取代期间的分波过程 spec（过程细节保留在 git 历史与本地 `backup/pre-polish-20260911` 分支）。模块级设计文档见 `docs/RealAgent/`，外部 API 契约见 `docs/外部服务接口/`。

## 一、四项常驻契约（改动任何相关代码前先读）

1. **证据契约（反幻觉主干）**：正文每个事实句绑定到某篇论文的 claim；claim 携带 `source_role`（own_method/own_contribution/own_result/own_setup/own_limitation/background/related_work）、`source_quote`（逐字摘录）、`evidence_ids`（真实块 ID）。claim 只有通过 P5.2 的 NLI 闸（claim vs 自证原文，direct/entailment）才进入证据行的 `supports_claims`——**该列表是写作的唯一合法来源**；终验 claim mapper 会逐字复核绑定行确实存在于 supports_claims（四键全等），任何侧门喂入都会被判 `invalid_source_binding`（波8 教训，已有测试钉住）。论文元数据只来自 SciVerse API 结果，绝不 LLM 生成。
2. **写作契约**：writer 以结构化 JSON（text/cite/sources/scope）产出句子；多层校验——引用别名归一化（`[P1]`/`p1`→裸键）、scope↔role 映射（method←own_method/own_setup；contribution←own_contribution/own_result；comparison←前四者；limitation←own_limitation；background←background/related_work）、单句、禁语。任何一层整体失败回退到确定性模板（诚实降级），并落盘原始回复与拒绝原因到 `output/writer_llm_rejects.jsonl`（含一次别名类定向重试）。
3. **闸门契约（goal gate 三道闸）**：非法引用=0；unsupported 清零或预算耗尽；覆盖率（正文/图表引用相对第 0 轮保留率 ≥0.7）拦"修复越修越空"。修复循环 ≤2 轮 revise，带不动点检测与 best-so-far 回滚；失败以 `quality_failed` + 非零退出诚实收场，绝不谎报 completed。
4. **评测契约**：机制指标（grounding/validity/测试）只是必要条件；每轮真跑验收必须（a）冻结归档 cache/output/requests/memory 后跑 L0–L3（L3 judge 必跑——唯一通读全文的层），（b）人工全文通读终稿逐项核对（离题/重复/事实范围/泄漏/可读性），（c）不达标诊断显式回应（修复/豁免/降级），不得删段凑通过。计数≠可用（图记录数≠可用图数），结构合法≠内容正确。

## 二、改进程序（按波次；每项均为管线内改动，重跑自动生效）

| 波次 | 主题 | 关键改动 |
|---|---|---|
| 批次 T1–T12 | 修复基础 | MinerU 异步轮询/断点恢复、SciVerse 精确标题检索、goal gate 诚实化、final-seed 测试污染修复 |
| 波1–1.5 | 证据身份 | SciVerse `/content` 全文通道、doc_id 身份链、agentic 闸门（溯源校验）、六端点契约实测 |
| 波2 | MinerU 恢复 | 五段错误分类、任务台账（24h 续轮询绝不重复提交）、结果缓存、read-file 预检、content 预检省 1/4 MinerU 调用 |
| 波3 | 相关性闸门 | 负模式锚定短语、经典种子注入（被引 ≥2 次高被引论文主动补入）、content 图片资产落盘 |
| 波4 | 冷跑收敛 | NLI CPU 化 + MPS 缓存治理、remap 回退、全部机制修复真跑兑现 |
| 波5 | 写作清理 | Revision Notes 副文件化、作者块过滤、对比段拆句、meta-search primary_topic 域黑名单 |
| 波6 | 证据约束写作 | claim 三件套契约、freeform 绕过删除、quote_diagnostic advisory、**模型路由**（生成→DeepSeek Flash thinking disabled；轻量校验→Intern-S2） |
| 波7 | 死通道与成品 | agentic backfill 死通道关闭（P5.2 12min→80s）、主题相似度闸（bge，0.60 精确拐点+双条件+fail-open）、own_limitation 提取增强、taxonomy 赋值/图/矩阵修复、PDF 表格跨页 |
| 波8 | writer 契约 | 别名归一化+拒绝落盘+定向重试、P2 骨架占位描述治理（连带检索去污）、limitation 池 seen 过滤、scope↔role 显式映射；波8b 撤销契约违规的补收并以测试钉住 |

## 三、最终验收结果（archive/wave8-final2，2026-09-11）

goal gate 全绿：**unsupported 0、角色违规 0、引用合法性 1.0、覆盖率 1.0/1.0、grounding 0.9**（12 sup+3 weak/15）；L1 引用质量 R/P 0.833、0 unsupported pair；L3 judge 3/3/3；全流程真跑约 25 分钟（暖缓存）。与 v2 起点对比：grounding 0.042→0.9、unsupported 26/35→0、证据 ID 冲突 808 重号→0、figure_bank 0 真实图片→全带文件、gate 从谎报 completed 到全链诚实。

## 四、已知局限（下一波候选，见 PROGRESS.md Next）

正文 LLM prose 存活率受装配层防御静默回退限制（4/9 单元，待落盘归因）；P2 分类数方差（3–6）摊薄正文、章节-论文分派靠关键词弱匹配致错位（topic_relevance 0.656<0.7）；own_limitation 提取覆盖薄致 Future Directions 空转；白名单 cap 15 使 canonical/gold 覆盖有结构性上限。
