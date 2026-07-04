# 模块 C 工作清单与计划：Grounded Survey Writer & Report Renderer

## 1. 模块定位

模块 C 负责最终综述生成、派生图表生成、质量评测和交互式报告渲染。

核心目标不是简单把内容写成一篇综述，而是在 A 和 B 已经建立的可信边界内，生成一份：

```text
可引用、可追溯、可验证、可视化、低幻觉、低返修次数的综述报告。
```

最新分工中，B 负责论文源数据和证据资产：

```text
PaperCards
EvidenceStore
FigureBank
TableBank
CitationIndex
CitationReadySet
CitationResult
ClaimMap
```

C 负责消费这些资产，并生成：

```text
survey.md
timeline.json
generated_artifact_bank.json
evaluation_report.json
final_report.html
final_report.pdf
```

其中，综述中的图和表只能来自两类来源：

```text
1. B 提供的论文原始资产：
   - figure_bank.json 中的论文原图 / 论文证据图
   - table_bank.json 中的论文原始表格

2. C 基于论文源内容生成的派生资产：
   - 调用生图 API 生成的结构图、时间线图、分类图、对比图
   - Agent 基于论文内容制作的对比表、总结表、未来方向矩阵
   - 这些资产必须登记到 generated_artifact_bank.json
   - 每个派生资产必须声明 source_artifacts、supporting_papers 和 provenance
```

也就是说，C 可以灵活生成更适合综述表达的图表，但不能凭空生成。所有生成图表都必须能回溯到论文源内容。

---

## 2. C 的硬约束

C 只读取 A 给的 request 文件，不直接调用 B。

```text
requests/survey_generation_request.json
requests/revision_request.json
requests/evaluation_render_request.json
```

C 写作和渲染时必须遵守：

```text
1. 只能引用 citation_ready_set 中的 paper_id。
2. 不能自己新增 reference。
3. 使用论文原图时，只能来自 figure_bank。
4. 使用论文原始表格时，只能来自 table_bank。
5. 自己生成的图、表、矩阵必须登记到 generated_artifact_bank。
6. 每个 generated artifact 必须有论文来源，不能伪装成论文原图或论文原表。
7. 重大 claim 必须有 citation。
8. 如果验证失败，只能删除非法内容或重写为有证据的 claim，不能临时编新引用。
```

---

## 3. 输入数据与用途

### 3.1 SurveyGenerationRequest

文件：

```text
requests/survey_generation_request.json
```

用途：

```text
A 给 C 的综述生成请求。C 以它为唯一入口读取输入路径和输出路径。
```

C 需要读取其中的：

```text
inputs.knowledge_bundle_path
inputs.paper_cards_path
inputs.taxonomy_path
inputs.citation_ready_set_path
inputs.evidence_store_path
inputs.figure_bank_path
inputs.table_bank_path

outputs.survey_markdown_path
outputs.timeline_path
outputs.generated_artifact_bank_path

writing_rules
generated_artifact_policy
visual_requirements
```

### 3.2 PaperCards

文件：

```text
cache/paper_cards.json
```

用途：

```text
提供每篇论文的结构化知识，包括研究问题、方法、贡献、局限、所属类别、相关证据、相关图表。
```

C 的使用方式：

```text
1. 作为 survey.md 正文的主要知识来源。
2. 用于 timeline 按年份组织发展脉络。
3. 用于 comparison table / future matrix 的内容提取。
4. 用于检查每个章节是否覆盖核心论文。
```

### 3.3 Taxonomy

文件：

```text
cache/taxonomy.json
```

用途：

```text
提供论文分类体系和阶段结构。
```

C 的使用方式：

```text
1. 作为 survey.md 的章节骨架。
2. 生成 taxonomy graph。
3. 计算 coverage_score。
4. 检查是否有类别未被综述覆盖。
```

### 3.4 CitationReadySet

文件：

```text
cache/citation_ready_set.json
```

用途：

```text
写作前锁定可引用论文、可使用论文图、可使用论文表。
```

C 的使用方式：

```text
1. 所有 [paper_id] 必须来自 allowed_citations。
2. 论文图必须来自 allowed_figure_ids。
3. 论文表必须来自 allowed_table_ids。
4. 写作前先构建 allowed_paper_id 集合，生成时做硬过滤。
```

### 3.5 EvidenceStore

文件：

```text
cache/evidence_store.json
```

用途：

```text
提供论文中的证据片段，包括摘要、正文段落、图注、表格说明。
```

C 的使用方式：

```text
1. 为重大 claim 找证据来源。
2. 为生成图表提供源内容。
3. 为 Evidence Panel 提供展示材料。
4. 降低 B 后验验证时出现 unsupported claim 的概率。
```

### 3.6 FigureBank

文件：

```text
cache/figure_bank.json
```

用途：

```text
B 提供的论文原始图片资产，包括 image_path、caption、extracted_text、paper_id、page、related_sections。
```

C 的使用方式：

```text
1. 直接在 survey.md / final_report.html 中嵌入论文原图。
2. 根据 caption 和 extracted_text 判断该图适合放在哪个章节。
3. 如果原图质量不适合直接展示，可基于 generation_hint 或 extracted_text 生成派生图，但新图必须登记到 generated_artifact_bank。
```

### 3.7 TableBank

文件：

```text
cache/table_bank.json
```

用途：

```text
B 提供的论文原始表格资产，包括 content_markdown、caption、paper_id、page、related_sections。
```

C 的使用方式：

```text
1. 直接展示论文原始表格。
2. 用作 C 生成对比表或总结表的源数据。
3. 派生表格必须登记到 generated_artifact_bank，不能算作 table_bank 原始表。
```

### 3.8 CitationResult

文件：

```text
output/citation_result.json
```

用途：

```text
B 对 C 生成的 survey.md 做结构性校验，检查引用、论文图、论文表和 generated artifact 是否合法。
```

C 的使用方式：

```text
1. 如果 invalid_items 非空，进入 revise_survey。
2. 在 final_report 中展示 citation summary。
3. 在 evaluation_report 中填 citation_validity_score。
```

### 3.9 ClaimMap

文件：

```text
cache/claim_map.json
```

用途：

```text
B 对 C 生成的关键论断做证据映射，标记 supported / weak / unsupported。
```

C 的使用方式：

```text
1. 计算 evidence_grounding_score。
2. 生成 Evidence Panel。
3. 对 unsupported claim 做修订。
4. 对 weak claim 做降级表述或补充引用。
```

### 3.10 GeneratedArtifactBank

文件：

```text
cache/generated_artifact_bank.json
```

用途：

```text
记录 C 自己生成的派生图表。
```

必须包含：

```text
artifact_id
artifact_type
title
artifact_path
artifact_format
source_artifacts
supporting_papers
provenance
usable_in_report
```

关键要求：

```text
1. source_artifacts 必须指向 paper_cards、taxonomy、evidence_store、figure_bank 或 table_bank 等真实输入。
2. supporting_papers 必须来自 citation_ready_set。
3. provenance 固定为 generated_by_c_from_verified_artifacts。
4. 不能把 generated artifact 描述成论文原图或论文原始表格。
```

---

## 4. 输出文件与完成标准

### 4.1 survey.md

文件：

```text
output/survey.md
```

完成标准：

```text
1. Markdown 格式。
2. 章节结构完整。
3. 引用格式统一为 [paper_id]。
4. 所有 paper_id 均来自 citation_ready_set。
5. 重大 claim 均带引用。
6. 不新增 reference。
7. 图片和表格来源合法。
8. C 生成的图表在文中引用时，可在 caption 中标明 Generated analysis based on verified papers。
```

推荐章节：

```text
1. Abstract
2. Introduction
3. Games as AI Benchmarks
4. Internal World Models for Agents
5. Neural Game Engines
6. Foundation Interactive Game World Models / GameCraft
7. LLM / MLLM Game Agents and World Models
8. Benchmarks and Evaluation
9. Open Challenges
10. Future Directions
11. Conclusion
12. References
```

### 4.2 timeline.json

文件：

```text
cache/timeline.json
```

完成标准：

```text
1. 每个条目包含 year、category_id、category_name、paper_ids、summary。
2. paper_ids 必须能在 paper_cards 中找到。
3. paper_ids 最好也在 citation_ready_set 中。
4. summary 不能包含没有论文支撑的新 claim。
```

### 4.3 generated_artifact_bank.json

文件：

```text
cache/generated_artifact_bank.json
```

完成标准：

```text
1. 所有 C 生成图、表、矩阵都登记。
2. 每个 artifact 有来源、有支撑论文、有输出路径。
3. artifact_type 只能使用接口允许的类型。
4. 不与 FigureBank / TableBank 混淆。
```

优先生成：

```text
generated_timeline_001
generated_taxonomy_graph_001
generated_comparison_table_001
generated_future_matrix_001
generated_evaluation_chart_001
```

### 4.4 evaluation_report.json

文件：

```text
output/evaluation_report.json
```

完成标准：

```text
1. 输出 overall_score。
2. 输出 coverage_score。
3. 输出 citation_validity_score。
4. 输出 evidence_grounding_score。
5. 输出 visualization_score。
6. 输出 structure_score。
7. details 中给出每个分数的计算依据。
```

推荐评分公式：

```text
overall_score =
  0.25 * coverage_score
+ 0.25 * citation_validity_score
+ 0.25 * evidence_grounding_score
+ 0.15 * structure_score
+ 0.10 * visualization_score
```

### 4.5 final_report.html / final_report.pdf

文件：

```text
output/final_report.html
output/final_report.pdf
```

完成标准：

```text
1. 展示 survey 正文。
2. 展示 timeline。
3. 展示 taxonomy graph。
4. 展示 comparison table。
5. 展示 future matrix。
6. 展示 paper figures / paper tables。
7. 展示 citation summary。
8. 展示 evaluation summary。
9. 展示 interactive evidence report。
```

---

## 5. 工具实现清单

### 5.1 write_survey.py

功能：

```text
读取 survey_generation_request.json，生成 survey.md、timeline.json、generated_artifact_bank.json。
```

核心步骤：

```text
1. 加载 request。
2. 加载所有输入 artifact。
3. 构建 allowed_paper_ids、allowed_figure_ids、allowed_table_ids。
4. 按 taxonomy 组织章节。
5. 为每个章节选择 paper_cards。
6. 为每个关键 claim 绑定 evidence。
7. 选择可用 FigureBank / TableBank。
8. 生成 C 派生图表。
9. 写 generated_artifact_bank。
10. 写 survey.md。
11. 本地预检 survey.md。
```

### 5.2 build_timeline.py

功能：

```text
根据 paper_cards 和 taxonomy 生成 timeline.json，并可生成 timeline 图。
```

输出：

```text
cache/timeline.json
output/generated_assets/timeline.png 或 timeline.svg
generated_artifact_bank 中登记 timeline artifact
```

### 5.3 build_generated_artifacts.py

功能：

```text
统一生成 C 的派生图表和表格。
```

至少支持：

```text
taxonomy_graph
comparison_table
future_matrix
evaluation_chart
summary_table
```

如果调用生图 API：

```text
1. prompt 必须来自 evidence_store / paper_cards / figure_bank / table_bank。
2. prompt 中不能加入没有来源的新事实。
3. 生成结果必须登记到 generated_artifact_bank。
4. caption 必须说明 based on verified paper artifacts。
```

### 5.4 revise_survey.py

功能：

```text
根据 revision_request.json 修订 survey。
```

处理规则：

```text
1. invalid citation：删除或替换为 allowed paper_id。
2. invalid paper figure：删除或替换为 figure_bank 中合法图。
3. invalid paper table：删除或替换为 table_bank 中合法表。
4. invalid generated artifact：删除或补登记 generated_artifact_bank。
5. unsupported claim：删除、弱化或改写成 evidence 支持的论断。
6. weak claim：补充 evidence 或降低语气。
```

### 5.5 evaluate_survey.py

功能：

```text
生成 evaluation_report.json。
```

评分依据：

```text
coverage_score：taxonomy 覆盖率 + 核心论文覆盖率。
citation_validity_score：直接来自 citation_result。
evidence_grounding_score：claim_map 中 supported / weak / unsupported 比例。
structure_score：章节完整性、摘要、挑战、未来方向、参考文献完整性。
visualization_score：timeline、taxonomy graph、paper figures、paper tables、generated artifacts 是否齐全且合法。
```

### 5.6 render_report.py

功能：

```text
根据 evaluation_render_request.json 渲染 HTML/PDF。
```

页面结构：

```text
1. Header：主题、overall score、citation score、evidence score。
2. Sidebar：目录。
3. Main：survey 正文。
4. Visual Section：timeline、taxonomy graph、comparison table、future matrix。
5. Evidence Panel：claim -> evidence -> papers -> figures/tables。
6. Verification Panel：citation_result 摘要。
7. Evaluation Panel：evaluation_report 摘要。
8. References。
```

---

## 6. 降低幻觉和减少 revise 次数的策略

### 6.1 写作前预检

在调用模型或生成正文前，先建立四个白名单：

```text
allowed_paper_ids
allowed_figure_ids
allowed_table_ids
allowed_generated_artifact_types
```

任何不在白名单里的内容不得进入 survey。

### 6.2 Claim 先证据后写作

不要让模型自由发挥整段综述。先构建 section claim plan：

```json
{
  "section": "Neural Game Engines",
  "claims": [
    {
      "claim": "Neural game engines shift world models from latent planning modules to interactive visual simulators.",
      "supporting_papers": ["genie_2024", "gamengen_2024"],
      "supporting_evidence": ["evidence_genie_001", "evidence_gamengen_002"],
      "allowed_figures": ["figure_genie_001"]
    }
  ]
}
```

再把 claim plan 变成自然语言。

这样可以显著减少 unsupported claim。

### 6.3 每段最多引入少量新信息

建议每段遵守：

```text
1 个主 claim
1-3 个 supporting papers
0-1 个 figure/table
不超过 2 个没有直接证据的解释性句子
```

这比长段落更容易通过 B 的 ClaimMap 验证。

### 6.4 生成图表必须先生成数据表

如果要调用生图 API，不要直接让模型“画一张酷图”。

流程应该是：

```text
paper_cards / evidence_store
  ↓
生成 chart_data.json / table_data.md
  ↓
基于 chart_data 调用生图 API 或绘图工具
  ↓
登记 generated_artifact_bank
```

这样生成图表的每个元素都有源头。

### 6.5 对 generated artifact 做 provenance 检查

每个 generated artifact 生成后立即检查：

```text
1. supporting_papers 是否全在 citation_ready_set。
2. source_artifacts 是否真实存在。
3. artifact_path 是否真实存在。
4. caption 是否没有宣称它是论文原图。
5. usable_in_report 是否为 true。
```

### 6.6 本地自检再交给 B 验证

在输出 survey.md 后，C 先做本地自检：

```text
1. 正则提取所有 [paper_id]。
2. 检查 paper_id 是否都在 citation_ready_set。
3. 检查所有图片路径是否来自 figure_bank 或 generated_artifact_bank。
4. 检查所有表格是否来自 table_bank 或 generated_artifact_bank。
5. 检查 References 中没有 citation_ready_set 之外的条目。
6. 检查每个 section 至少有 1 个 citation。
7. 检查重大 claim 是否附近有 citation。
```

通过本地自检后再交给 B，可以减少 revise 次数。

### 6.7 对 unsupported claim 的处理优先级

如果 B 返回 unsupported：

```text
1. 优先删除没有必要的泛化论断。
2. 其次改写成更窄、更可证的表述。
3. 再其次换成 evidence_store 中已有的 claim。
4. 不要临时新增 paper_id。
5. 不要为了保留漂亮句子牺牲证据一致性。
```

---

## 7. 推荐开发顺序

### 第一阶段：最小闭环

目标：

```text
C 能读 survey_generation_request.json，并输出 survey.md + timeline.json + generated_artifact_bank.json。
```

任务：

```text
1. 实现 request loader。
2. 实现 citation_ready_set 白名单加载。
3. 实现 paper_cards / taxonomy 加载。
4. 实现 survey.md 初版生成。
5. 实现 timeline.json。
6. 实现 generated_artifact_bank 空结构 + timeline 登记。
```

### 第二阶段：低幻觉写作

目标：

```text
降低 citation invalid 和 unsupported claim。
```

任务：

```text
1. 实现 section claim plan。
2. 实现 evidence-first 写作策略。
3. 实现本地 citation / figure / table / generated artifact 预检。
4. 实现 References 自动生成，禁止模型自由写参考文献。
```

### 第三阶段：派生图表增强

目标：

```text
让报告图文并茂，但所有图表都有 provenance。
```

任务：

```text
1. 生成 taxonomy graph。
2. 生成 comparison table。
3. 生成 future matrix。
4. 如有时间，调用生图 API 生成更美观的 timeline / taxonomy / evaluation chart。
5. 所有派生制品登记 generated_artifact_bank。
```

### 第四阶段：验证修订闭环

目标：

```text
能根据 B 的 citation_result 和 claim_map 自动修订。
```

任务：

```text
1. 实现 revise_survey.py。
2. 对 invalid citation / invalid figure / invalid table / invalid artifact 做删除或替换。
3. 对 unsupported claim 做删除或改写。
4. 输出 survey_revised.md。
```

### 第五阶段：评测与报告

目标：

```text
输出 evaluation_report.json + final_report.html + final_report.pdf。
```

任务：

```text
1. 实现 evaluate_survey.py。
2. 实现 HTML 模板。
3. 实现 Evidence Panel。
4. 实现 Citation Summary。
5. 实现 Evaluation Summary。
6. 导出 PDF。
```

---

## 8. 答辩亮点

模块 C 可以讲三个亮点：

```text
1. Grounded generation：
   C 不是自由写作，而是在 CitationReadySet、EvidenceStore、FigureBank、TableBank 约束下生成综述。

2. Flexible but traceable visualization：
   报告中的图表既可以来自论文原图，也可以由 C 基于论文源内容生成，但所有生成制品都会登记到 GeneratedArtifactBank，记录来源和 provenance。

3. Self-check before verification：
   C 在交给 B 验证前先做本地白名单检查和 claim-evidence plan，尽量减少幻觉引用、非法图表和 revise 次数。
```

一句话总结：

```text
我负责模块 C：在锁定引用和论文源证据的约束下生成综述，基于 FigureBank / TableBank 和 GeneratedArtifactBank 构建可追溯图文报告，并通过本地预检、评测和修订闭环降低幻觉与返修次数。
```

