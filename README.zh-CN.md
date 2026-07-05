# EviSurvey —— 基于证据的综述生成 Harness

**[English](README.md)** | **[简体中文](README.zh-CN.md)**

> 一个把研究主题转化为**经过验证、图文并茂**的学术综述的 Agentic Harness
> —— 为上海人工智能实验室 2026 暑期夏令营 Hackathon（团队挑战题 4）
> **世界模型** 主题打造。

EviSurvey 把一次性的大模型综述写作，转化为一条**多阶段、可追踪、可验证**
的研究工作流：每一条论断都落在真实论文上，每一条引用都与来源元数据核对，
最终输出图文并茂的 HTML/PDF 报告。

---

## ✨ 核心特性

- **从机制上防止幻觉** —— 论文元数据只来自 SciVerse API；每条引用都交叉
  核对，每条论断都绑定到具体证据。
- **基于真实工具的 Agent Loop** —— Intern-S2-Preview 通过显式工具调用编排
  SciVerse 检索、MinerU PDF 解析、文件 I/O 与 NLI 校验器（而非自由发挥）。
- **结构化知识，而非裸文本** —— 论文变成 **Paper Card**；论断变成
  **Claim-to-Evidence Map**；参考文献变成 **Citation Index**。共 8 个带类型
  的产物在流水线中流转。
- **图文并茂的输出** —— 从解析后的论文中挖出图表，存入
  `figure_bank` / `table_bank`，并嵌入最终报告。

---

## 🏗️ 架构

Harness 由一个 Agent Loop 协调三个模块：

```
                 ┌──────────────────────────────────────────┐
   用户主题 ───▶ │  模块 A —— 规划器 / Agent Loop            │
   "世界模型综述" │  (Intern-S2-Preview 作为核心 LLM)        │
                 │  拆解 → 编排 → 汇总                      │
                 └───────────────┬──────────────────────────┘
                                 │ 工具调用
            ┌────────────────────┼─────────────────────────┐
            ▼                    ▼                         ▼
   ┌─────────────────┐  ┌─────────────────┐      ┌─────────────────┐
   │ 模块 B          │  │ 模块 C          │      │ 工具            │
   │ 知识流水线      │  │ 校验 / 撰写     │      │ SciVerse·MinerU │
   │ (阶段 1–6)      │  │ / 渲染          │      │ NLI·文件 I/O    │
   └─────────────────┘  └─────────────────┘      └─────────────────┘
                                 │
                                 ▼
              HTML / PDF 综述 + 引用校验 + 质量评测报告
```

### 模块 A —— 规划器（`harness/`）
负责 Agent Loop、任务拆解、检索策略构建、工具注册表、状态管理、记忆与技能
系统，并向 B、C 输出它们消费的 request 文件。

### 模块 B —— 知识流水线（`tools/knowledge_pipeline_worker.py`、`tools/phases/`）
通过六个阶段构建 `KnowledgeBundle`：

| 阶段 | 名称 | 产物 |
|------|------|------|
| **P1** | 检索策略分解 | 搜索策略 + 子查询 |
| **P2** | 已有综述分析 | 现有综述的结构图 |
| **P3** | 论文检索 | `retrieved_papers`、`parsed_papers` |
| **P5** | Paper Card / 证据 / 综合 | `paper_cards`、`evidence_store`、`taxonomy` |
| **P6** | Bundle 组装 | `knowledge_bundle`（8 产物 + 引用索引） |

### 模块 C —— 校验 / 撰写 / 渲染（`tools/`）
- `write_survey.py` —— 基于 bundle 起草综述。
- `verify_citations.py` —— 基于 NLI 的引用校验。
- `revise_survey.py` —— 基于评审反馈的修订。
- `render_report.py` —— 渲染最终的图文 HTML/PDF。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone <repo-url> && cd AILabMacroHard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

要求 **Python ≥ 3.11**。

### 2. 配置环境变量

复制 `.env.example` → `.env` 并填入密钥：

```bash
cp .env.example .env
```

| 变量 | 必填 | 说明 |
|------|------|------|
| `INTERN_API_BASE_URL` | ✅ | Intern-S2-Preview 端点 |
| `INTERN_API_KEY` | ✅ | Intern-S2-Preview API Key |
| `INTERN_MODEL_NAME` | — | 默认 `intern-s2-preview` |
| `SCIVERSE_API_KEY` | ✅ | SciVerse 论文检索 / RAG |
| `MINERU_API_KEY` | ✅ | MinerU PDF 解析 |
| `HF_ENDPOINT` | — | HuggingFace 镜像，如 `https://hf-mirror.com` |

> `API_BASE_URL` / `API_KEY` 可作为别名。

### 3. 运行 Harness

```bash
# 完整运行
python main.py --topic "世界模型综述"

# 快速冒烟（限制检索与解析规模）
python main.py --topic "世界模型综述" --max-papers 5 --max-core-papers 3 --mode full

# 仅生成模块 A 的 request 文件，跳过 B/C
python main.py --topic "世界模型综述" --prepare-only --mode full
```

CLI 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--topic` | — *(必填)* | 综述主题 |
| `--language` | `zh` | 综述语言（`zh` / `en`） |
| `--mode` | `demo` | `demo` / `full` |
| `--max-papers` | `40` | P3 检索上限 |
| `--max-core-papers` | `15` | 解析的核心论文上限 |
| `--prepare-only` | 关 | 仅输出 A 的 request 文件 |

---

## 📦 输出产物

产物落在 `output/`：

- `final_report.html` / `final_report.pdf` —— 图文并茂的综述。
- `survey.md` / `survey_revised.md` —— 起草 / 修订后的 Markdown。
- `citation_result.json` —— 引用校验结果。
- `evaluation_report.json` —— 质量评测。
- `final_state.json` —— 完整运行状态，便于追溯。

中间 request 文件写入 `requests/`；解析后的知识缓存位于 `cache/`。

---

## 🧪 测试

```bash
# 单元测试（fake LLM，无网络）
pytest tests/unit/ --log-cli-level=WARNING

# 集成测试（真实 SciVerse + fake LLM + 真实 MinerU 逐篇降级）
pytest tests/integration/ -v

# 全真实 API（三个 API 全开）
pytest tests/integration/ -v -m real_api

```

---

## 🗂️ 项目结构

```
.
├── main.py                      # CLI 入口 → AgentLoop
├── config.py                    # 基于环境变量的 AppConfig
├── llm_client.py                # Intern-S2-Preview 客户端
├── harness/                     # 模块 A：规划器、agent loop、注册表、状态
├── tools/
│   ├── clients/                 # SciVerse、MinerU、fake LLM
│   ├── phases/                  # P1–P6 知识流水线
│   ├── nlp/                     # NLI 校验器、数据清洗
│   ├── knowledge_pipeline_worker.py   # 模块 B 入口
│   ├── write_survey.py          # 模块 C：起草
│   ├── verify_citations.py      # 模块 C：引用校验
│   ├── revise_survey.py         # 模块 C：修订
│   └── render_report.py         # 模块 C：渲染
├── tests/                       # 单元 + 集成
├── docs/                        # 设计文档 + API 契约
├── requests/  output/  cache/   # 运行时 I/O
└── .env.example
```

---

## 🔌 外部 API 契约

经 2026-07-05 对线上 API 验证（详见 `CLAUDE.md`）：

- **Intern-S2-Preview** —— 核心 LLM，OpenAI 兼容。约 1 请求 / 2 秒。
- **SciVerse** —— `https://api.sciverse.space`。`POST /meta-search`（元数据）
  与 `POST /agentic-search`（可引用 RAG chunk）。
- **MinerU** —— `https://mineru.net`。PDF → 结构化文本 + 图表。

---

## 🛡️ 防幻觉策略

1. **元数据仅来自 SciVerse** —— 题目、作者、年份、会议绝不由 LLM 生成。
2. **Claim-to-Evidence 绑定** —— 每条论断指向某篇论文的具体 chunk。
3. **引用校验 pass** —— 每条引用都与来源 `citation_index` 核对。
4. **图表来自解析后的论文** —— `figure_bank` / `table_bank` 通过 MinerU
   挖掘，而非凭空合成。

---

## 📄 许可

上海人工智能实验室 2026 暑期夏令营 Hackathon 项目。贡献者与素材归属见
项目内文件。
