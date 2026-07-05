Let me first fetch the pages for all 6 Sciverse API interfaces

Let me fetch the meta-catalog page as well

以下是 Sciverse 六大 API 接口的全面总结，涵盖每个接口的概述、适用场景、不适用场景及核心能力。

---

### **1. agentic-search — 智能检索与片段返回**

**概述：** 面向 LLM Agent 与 RAG 场景设计，用自然语言提问，返回最相关的可引用文献段落。每条结果包含标题、正文片段、doc_id、页码/偏移位置等来源信息，适合快速找到可引用的上下文。

**适用场景：** RAG 应用中为 LLM 补充含引用的文献证据；Agent 工具调用中一屏拿到可回链的片段与原文位置；问答系统中结合文献原文与片段生成带出处的回答。

**不适用场景：** 精确 DOI、标题或字段化分页导出（应使用 meta-search）；读取完整原文上下文（应使用 content 接口）；下载图表或附件资源（应使用 resource）。

**核心能力：** 支持 query（最长 4096 字符）、top_k（1-100）、sub_queries 查询改写（0-4）；支持按语言、标题、作者、发表年份、期刊、被引次数、主题领域等维度做过滤器（filters）。注意该接口只返回语义证据片段，不生成最终答案。

---

### **2. meta-search — 按字段过滤与排序检索元数据**

**概述：** 按年份、期刊、DOI、语言等结构化条件筛选论文书目信息，返回标题、摘要、作者、发表年份等元数据，不返回段落正文。支持不传 query 仅以 filters/sort 精准检索，也支持传入 query 做全文模糊检索。

**适用场景：** 按年份、期刊、DOI、语言、开放获取状态等字段筛选论文，返回论文级元数据列表；论文列表的批量导出与分页。

**不适用场景：** 不返回全文片段（需要证据片段用 agentic-search，需要原文正文用 content）。

**核心能力：** 支持 papers / authors / sources 三种集合检索；丰富的 FilterItem 算子（EQ/NE/GT/GTE/LT/LTE/IN/NIN/CONTAINS/MATCH/MATCH_PHRASE）；SortItem 排序；分页支持 page/page_size 浅翻页和 cursor 深翻页；支持 freshness_boost（MILD/STRONG 新鲜度加权）和 impact_boost（MILD/STRONG 影响力加权），两者可叠加。字段可见性受 Token 权限影响。

---

### **3. content — 按 doc_id 读取原文**

**概述：** 用 doc_id 分段读取文献全文文本，doc_id 通常来自 agentic-search 或 meta-search 的返回结果。适合详情页展示、引用核对和长文分批加载。

**适用场景：** 从 agentic-search 的 evidence chunk 继续读取上下文；查看完整文献正文以进行引用核对或深度阅读。

**不适用场景：** 不适合直接在元数据检索或片段检索中使用（依赖上游接口先提供 doc_id）。

**核心能力：** 通过 doc_id 必填参数定位文献；支持 offset/limit 分段拉取（limit 默认 700 字符，仅在传入 offset 时生效）；响应返回 text（Markdown/纯文本）、chars_returned、next_offset 和 more 字段，推荐根据 more/next_offset 续读以降低超时风险。按 Unicode 字符计数，不按字节。

---

### **4. resource — 按相对路径下载附件**

**概述：** 用于拉取论文插图、实验图、解析图等文献相关二进制附件。file_name 通常来自检索结果、解析结果或正文中的图片路径，只传相对路径，不要传完整 URL。响应为二进制流。

**适用场景：** 下载论文中的图片（Figure、图表等）和其他二进制附件。

**不适用场景：** 不返回文本内容或解释图表含义（图表理解需要上层多模态模型）；不接受完整 URL（只接受平台返回的相对路径，避免 SSRF 和路径穿越风险）。

**核心能力：** 请求参数仅 file_name（相对路径，不得含 \、..，不得以 / 开头）；响应为二进制流，带 Content-Type、Content-Disposition 和 X-Request-ID。file_name 通常来自 content Markdown、解析结果、agentic-search 或 meta-search 返回的资源引用字段。

---

### **5. meta-catalog — 查看元数据字段目录**

**概述：** 返回 meta-search 可用字段的 schema，包括哪些字段可以筛选或排序、默认返回哪些列，以及过滤算子和枚举样本值。适合在搭建筛选器、生成查询表单或让 Agent 自动拼装 meta-search 请求前调用。

**适用场景：** 当 Agent 或前端需要动态生成 meta-search 筛选器、字段选择器或排序条件时，先读取 catalog 获取字段能力信息。

**不适用场景：** 不返回论文结果（论文结果由 meta-search 返回）；sample values 仅为样本值（不代表完整取值空间，会缓存 24 小时）。

**核心能力：** 支持 collection 参数（papers / authors / sources）；每个字段返回 name、type、filterable、sortable、searchable、default_returned、description、operators、sample_values；全局返回 default_fields 和 filter_operators 列表。不同账号权限看到的结果可能不同，调用方需动态构造请求。

---

### **6. meta-paper-relations — 分页查论文引用关系**

**概述：** citations / references / related_works 是无界关系数组（高被引文献可达数千条），meta-search 出于响应体积考虑只内联截断少量条目。本接口先用 meta-search 或 agentic-search 拿到目标论文 unique_id，再按 relation 类型分页获取完整列表。

**适用场景：** 查看某篇论文的完整被引列表（CITATIONS）、参考文献列表（REFERENCES）或相关工作列表（RELATED_WORKS）。

**不适用场景：** 无法通过 doc_id 查询引用关系（必须用 unique_id）；RELATED_WORKS 不能替代语义检索（开放问题的语义证据召回应使用 agentic-search）。

**核心能力：** 必填参数有 unique_id（如 paper:10.1038/xxx）和 relation（CITATIONS / REFERENCES / RELATED_WORKS）；支持 page/page_size 分页（1-200）；响应返回 items（每条含 id、id_type、title）、total_count、page、page_size、total_pages。CITATIONS 表示"谁引用了我"，REFERENCES 表示"我引用了谁"，两者方向相反。

---

以上六个接口覆盖了从语义检索（agentic-search）、结构化元数据检索（meta-search）、字段能力探查（meta-catalog）、原文读取（content）、附件下载（resource）到引用关系追踪（meta-paper-relations）的完整学术文献工作流。所有接口统一使用 API Key Bearer Token 鉴权，同一套 API Key 可用于已开通的 Sciverse、点石与 Skills 能力。