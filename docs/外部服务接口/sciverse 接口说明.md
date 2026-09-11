# SciVerse API 接口说明（2026-09-09 实测版）

Base URL：`https://api.sciverse.space`（注意不是网站域名）。鉴权：`Authorization: Bearer <SCIVERSE_API_KEY>`。官方文档站 <https://sciverse.space/docs>（概览），官方 SDK <https://github.com/opendatalab/Sciverse-Agent-Tools>。API host 不暴露 `/docs`/`/openapi.json`。同一把 Key 可用于 Sciverse / 点石 / Skills 生态。

本文档全部条目于 2026-09-09 用真实 Key 逐一验证过（样本：GameNGen, arXiv 2408.14837）。此前版本的"六接口总结"来自厂商页面转述，方法名/参数键/分页语义多处与实测不符，已废弃。

## 数据身份（先读这个）

- **`doc_id`：64 位不透明内部哈希**（如 `6442a8e2a2da…`），既不是 DOI 也不是 arXiv id。它是 `/content` 的钥匙，**只在 meta-search / agentic-search 的返回里出现**，必须检索时持久化，无法从 paper_id 推导。
- **`unique_id`：`paper:<doi>` 形态**，即本仓库的 paper_id。`meta-paper-relations` 用它查询。
- meta-search 记录同时携带 `doc_id` 与 `doi`，两者可互相对应但不相等。

## 1. POST /meta-search — 结构化元数据检索

请求：`{query, page, page_size, filters?, sort?, fields?, freshness_boost?, impact_boost?}`（可选键只在提供时发送；boosts 仅 `MILD`/`STRONG`，且只在未设 sort 时生效；`sort` 与 boost 同发会被拒）。

响应：`results[]`，单条含 `doc_id`、`unique_id`、`doi`、`title`、`abstract`、`author[]`、`publication_published_year`、`publication_venue_name_unified`、`keywords`、`citation_count`、`influential_citation_count`、`fwci`、`citation_normalized_percentile`、`access_*`、`locations` 等。

实测注意：`access_oa_url`/`locations` 在我们的 Key 下返回空（权限或覆盖未定，不要依赖它找 PDF URL）。

## 2. POST /agentic-search — 语义片段检索

请求：`{query, top_k, filters}`。响应键是 **`hits[]`**（不是 `results`），单条含 `chunk`（片段文本）、`chunk_id`、`doc_id`、`title`、`page_no`、`offset`、`score` 及出版元数据。`author` 字段损坏（要用作者走 meta-search）。

**关键实测结论（2026-09-09）**：这是"语义相关论文"检索器——即使拿论文自身的标题/断言作 query，**被查论文自己的 chunk 也不会出现在 top-8**。因此它不能用于为被引论文提供自身证据；历史上把它当作 grounding 通道并挂到论文名下，是跨论文冒名归属的根源（见诊断文档 §4.1）。正确用法：找相关文献线索；自身证据用 `/content`。

## 3. GET /content — 按 doc_id 读全文（自有证据通道）

请求：`?doc_id=<完整doc_id>`；**不带 offset 时一次返回全文**（实测 GameNGen 50,742 字符 Markdown，`more=false`）。`offset`/`limit` 仅显式传入才发送（limit 默认 700 字符只在传 offset 时生效）。

响应：`{text, more, next_offset}`（另有 `code/biz_code/message` 等信封字段；`bytes_returned`/`request_tokens` 计量）。超长文档按 `more=true` + `next_offset` 续读。

正文是 Markdown，标题行 `# ...`，**图片以相对路径内嵌**（`![...](dt=…/hash.jpg)`），这些相对路径是 `/resource` 的输入。全文含 References 节文本——本仓库用 `_split_markdown` 的 `role="reference"` 标记将其排除出证据。

## 4. GET /resource — 按相对路径下载二进制附件

请求：`?file_name=<content 返回的相对路径>`（不得含 `\`、`..`、不得以 `/` 开头）。响应：二进制流（实测 `image/jpeg`，合法 JPEG 头，11KB，2 秒）。

实测注意：遇到过一次瞬时 ReadTimeout，重试即成功——客户端 `_request` 已带传输错误退避重试。早期记录的"507 不可用"已过时。

## 5. GET /meta-catalog — 字段目录

请求：`?collection=papers|authors|sources`。响应：每字段 `name/type/filterable/sortable/default_returned/operators/sample_values`。papers 集合共 65 字段；`doc_id`、`doi`、年份、venue、被引数、`primary_topic.*`（domain/field/subfield）、`fwci` 等均可过滤，`citation_count`/`fwci`/年份可排序。字段可见性受 Key 权限影响。

## 6. POST /meta-paper-relations — 引用关系（必须 POST，键是 unique_id）

请求体：`{unique_id: "paper:10.48550/arxiv.2408.14837", relation: "CITATIONS"|"REFERENCES"|"RELATED_WORKS", page, page_size}`。**GET 返回 405；请求键是 `unique_id`，发 `paper_id` 无效**（客户端旧方法正是错的这个）。

响应：`{items[]（每条 id/id_type/title，id_type 如 semantic_scholar）, total_count, page, page_size, total_pages}`。实测 GameNGen REFERENCES=34 条。用途：经典覆盖的原料（拉核心论文的参考文献清单做真实 bib 扩展）。

## 客户端实现约定（tools/clients/sciverse_client.py）

- 全端点走 `_request`：429/5xx 指数退避（429 基数更长）、传输错误重试、请求间隔 `SCIVERSE_MIN_INTERVAL`（默认 1s）。
- `get_content(doc_id, offset=None, limit=None)`：可选参数不传就不发；`read_full_text(doc_id, max_pages=8)` 负责首呼全量 + 按 more 续读。
- `meta_paper_relations(unique_id, relation, page, page_size)`。
- `meta_search` 的 boosts 语义见上；`agentic_search` 返回 `hits`。

## 本仓库的接入位置

- 检索（P3）：`_to_retrieved` 从 hit 保留 `doc_id`（身份链起点）。
- 卡片（P5.1）：深卡/浅卡从 retrieved 传递 `doc_id` 到 `PaperCard`。
- 证据（P5.2）：`_content_backfill` 用 `read_full_text(card.doc_id)` 拉论文自身全文 → `content_chunk` 证据（provenance 三字段齐全）；`_agentic_backfill` 保留但受来源闸门约束（`tools/verify/provenance.py`：DOI 相等 OR 归一化标题相等，否则拒绝）。
