MinerU 提供两类文档解析 API，核心区别如下：

---

### **🎯 精准解析 API**

需 Token 鉴权（`Bearer {token}`），面向高精度、深度结构化提取场景。

**接口清单：**
- **单文件 URL 解析** — `POST /api/v4/extract/task`。传入远程文件 URL，支持 pipeline/vlm/MinerU-HTML 三种模型，可配置 OCR、公式/表格识别、callback 回调、额外导出格式等丰富参数。不支持本地文件上传。
- **本地文件批量上传** — `POST /api/v4/file-urls/batch` 申请上传链接（最多 50 个），PUT 上传后系统自动提交任务。适合多文件本地解析。
- **URL 批量提交** — `POST /api/v4/extract/task/batch`，一次最多 50 个 URL。适合远程文件批量处理。
- **查询结果**（单文件 `GET .../task/{task_id}` / 批量 `GET .../results/batch/{batch_id}`），返回 Zip 包（Markdown + JSON + 可选 docx/html/latex）。

**适用场景**：文件 ≤200MB/200 页，需表格/公式精准识别、多格式输出，或需要 callback 回调。支持 PDF、图片、Office 文档、HTML。

**不适合**：文件超限、需免登录、仅需 Markdown 的轻量场景、AI Agent 工作流。

---

### **⚡ Agent 轻量解析 API**

免登录，IP 限频防滥用，专为 AI Agent 工作流设计，追求最快速度。

**接口清单：**
- **URL 解析** — `POST /api/v1/agent/parse/url`，传入远程文件 URL，仅输出 Markdown（CDN 链接）。参数有限（PDF 可配表格/公式/OCR/页码范围）。
- **本地文件上传** — `POST /api/v1/agent/parse/file` 获取签名上传 URL，PUT 上传后自动解析。每次一个文件。
- **查询结果** — `GET /api/v1/agent/parse/{task_id}`，返回 state 和 `markdown_url`。

**适用场景**：文件 ≤10MB/20 页，快速获取 Markdown，零门槛接入。支持 PDF、图片、Docx、PPTx、Xlsx。

**不适合**：文件超限、需 HTML 解析、需批量/多格式/高精度结构化输出、需 callback 回调。