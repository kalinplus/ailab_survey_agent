# EviSurvey —— 基于证据的综述生成 Harness

**[English](README.md)** | **[简体中文](README.zh-CN.md)**

> 一个把研究主题转化为**经过验证、图文并茂**的学术综述的 Agentic Harness
> —— 为上海人工智能实验室 2026 暑期夏令营 Hackathon（团队挑战题 4）
> **世界模型** 主题打造。

EviSurvey 把一次性的大模型综述写作，转化为一条**多阶段、可追踪、可验证**
的研究工作流：每一条论断都落在真实论文上，每一条引用都与来源元数据核对，
最终输出图文并茂的 HTML/PDF 报告。生成用 **DeepSeek Flash**，核查与评审用
**Intern-S2-Preview**；论文元数据只来自 SciVerse API——LLM 只负责组织和措辞，
没有资格发明事实。

---

## ✨ 核心特性

- **从机制上防止幻觉** —— 论文元数据只来自 SciVerse API；动笔**之前**先用
  引用**白名单**锁死"允许引用哪些论文"，合法性从源头收敛，而非事后清洗。
- **确定性骨架 + 有界的 LLM 决策** —— 流程步骤顺序是写死的代码（运行可复
  现、问题可定位）；LLM 只在三个"关节"内部做有预算的决策（检索策略、修复
  动作），且只能从一个明确的动作清单里选。"过没过"由纯代码的 Goal Gate 裁
  决——**LLM 无权给自己的作业打分**。
- **修不空的查-修回路** —— 每条论断过三道检查（引用 id 白名单、与所引证据
  的 NLI 语义核对、归属检查——别人的成绩不能算成引用论文的）；修复可以改写、
  换证据而不是只会删；覆盖闸（正文/图表保留率 ≥ 70%）拦住"全删了指标就好了"
  的假修复。闸没过，状态和退出码老实写 `quality_failed`——诚实，不装好。
- **结构化知识，而非裸文本** —— 论文变成 **Paper Card**；论断变成
  **Claim-to-Evidence Map**；参考文献变成 **Citation Index**。共 8 个带类型
  的产物在流水线中流转，环节之间靠文件衔接（能打开看、能重放、能对账）。
- **图文并茂的输出** —— 从解析后的论文中挖出图表，存入
  `figure_bank` / `table_bank`，并嵌入最终报告。

---

## 🏗️ 架构

这里按系统能真正立住的三个能力来介绍——正好是数据流过的三个阶段：
**检索策略**（素材从哪来）→ **引用验证**（每句话站不站得住）→
**质量评估**（最终算不算合格）。

```
 研究主题 "世界模型综述"
      │
      ▼
┌──────────────────────────────────────────────┐
│ ① 检索策略 —— 素材从哪来                        │
│    定检索方向 → 真实 API 搜论文/解析 →           │
│    做卡片/收证据 → 圈定引用白名单                │
└───────────────────┬──────────────────────────┘
                    │ 白名单 + 证据库 + 图库
                    ▼
          [ 写作：模板骨架 + 可选 LLM 段落 ]
          （为验证而写：论断先行）
                    │ output/survey.md
                    ▼
┌──────────────────────────────────────────────┐
│ ② 引用验证 —— 每句话站不站得住                  │
│    结构校验 + 逐句 NLI + 归属检查                │
│    → 站不住就修（修复 Agent）→ 再查              │
│    （查-修循环由纯代码裁判把关，见下）            │
└───────────────────┬──────────────────────────┘
                    │ 修好的 survey.md
                    ▼
┌──────────────────────────────────────────────┐
│ ③ 质量评估 —— 最终算不算合格                    │
│    运行内评分 + 基线对比 + 渲染交付              │
│    → 事后三层评测（冗余 / NLI / LLM 评审）       │
│    → 诚实状态：completed 或 quality_failed      │
└───────────────────┬──────────────────────────┘
                    ▼
     output/final_report.html / .pdf + 质量报告
```

### ① 检索策略 —— 素材从哪来

**搜什么**由有界的**策略 Agent**（关节 1，`--mode full` 启用）决定：先零成本
探测 → 自动聚类 → 命名检索方向 → 试搜采样 → 给 LLM 看一张"成绩单"（每个方
向查到几篇、重合多少、相关度多高、年份分布）→ 它提修改 → 再试；最多 3 轮，
纯代码闸门验收，不达标回滚到最好的版本。

**怎么搜**：逐方向用 SciVerse `meta-search`（论文元数据的**唯一**来源），
中文关键词先由 LLM 翻译成英文学术查询；MinerU PDF 解析默认关（失败降级为真
实摘要，绝不造假）。

**圈定什么能用**：论文做卡片（有全文做深卡、只有摘要做浅卡）；证据片段逐条
和论文绑定；挖图表。然后——动笔**之前**——锁一份**引用白名单**
（`cache/citation_ready_set.json`，新近度 × 影响力各占一半，防止纯看新把经
典挤出去）。下游写作/校验/修复/渲染全都只认这份名单。

### ② 引用验证 —— 每句话站不站得住

初稿（`write_survey.py`）**为验证而写**：章节骨架是确定性模板，LLM（可选，
`EVISURVEY_WRITER_LLM=1`）先规划论断数据（说什么/引哪篇/什么性质）再落笔，
内部术语和裸论文 id 不允许出现在正文里。

`verify_citations.py` 把正文拆成一句句论断，逐句过三道检查：**结构性**
（引用 id 在不在白名单、图存不存在）、**语义 NLI**（所引的证据原文真的说了
这个吗）、**归属**（标了"这是论文自己的贡献"的论断，不能建立在其实讲别人
工作的引文上）。

查出问题由**修复 Agent**（关节 2，`EVISURVEY_REPAIR_AGENT=1`）修：失败按原
因机械分 6 组，LLM 按组选动作（换引用 / 改写 / 换或补证据 / 删——最后手段），
每句改写重新过 NLI 才算修好。

查-修循环（最多 2 轮修复）由 **Goal Gate**（`harness/goal_gate.py`）裁决——
纯代码，LLM 无权判"过"。裁决顺序：覆盖跌破 70% 了吗 → 有归属违规或证据缺
口吗 → 引用错误和不支持的论断清零了吗 → 修复预算到顶了吗 → 还有进步吗
（fixpoint）。全程记住"最好的版本"（问题少优先，问题一样多时字多的赢——防
止靠删句子刷指标），报告的数字永远描述**实际交付**的那版。

### ③ 质量评估 —— 最终算不算合格

`render_report.py` 算 `evaluation_report.json`（总分 / 引用合法性 / 证据支
撑度 / 可视化）和 `review_report.json`（含 **topic_relevance 主题相关度**闸
——低于 0.7 的必须在验收时显式回应），渲染 HTML/PDF，并与基线综述对比。事
后用 `scripts/run_survey_eval.py` 跑三层评测——L0 冗余、L1 NLI 抽查、L3 LLM
评审**真正通读全文**打分（唯一读文档的层）。闸没过不阻断渲染，但状态和退出
码老实写 `quality_failed`。

> **代码住哪**：`harness/` 是调度（agent loop、planner、tool registry、goal
> gate、`agents/` 关节）；`tools/` 干活（`phases/` 知识管线、write/verify/
> revise/render 四件套、`clients/` 外部服务、`models/` 数据格式）。环节之间
> 靠文件（`requests/`、`cache/`、`output/`）递话——文件就是接口。
> 完整走读见 `docs/系统架构与流程.md`。

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
| `HEAVY_LLM_*` | — | 可选的生成模型（如 DeepSeek Flash）：`HEAVY_LLM_BASE_URL` / `HEAVY_LLM_API_KEY` / `HEAVY_LLM_MODEL`；不配置则回退 Intern-S2 |

> `API_BASE_URL` / `API_KEY` 可作为别名。

### 3. 运行 Harness

```bash
# 完整运行
python main.py --topic "世界模型综述"

# 快速冒烟（限制检索与解析规模）
python main.py --topic "世界模型综述" --max-papers 5 --max-core-papers 3 --mode full

# 只生成请求文件，跳过流水线
python main.py --topic "世界模型综述" --prepare-only --mode full
```

CLI 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--topic` | — *(必填)* | 综述主题 |
| `--language` | `zh` | 综述语言（`zh` / `en`） |
| `--mode` | `demo` | `demo` / `full`（`full` 启用策略 Agent） |
| `--max-papers` | `40` | 检索上限 |
| `--max-core-papers` | `15` | 解析的核心论文上限 |
| `--use-mineru` | 关 | 开启 MinerU PDF 解析 |
| `--prepare-only` | 关 | 仅输出请求文件 |

---

## 📦 输出产物

产物落在 `output/`：

- `final_report.html` / `final_report.pdf` —— 图文并茂的综述。
- `survey.md` / `survey_revised.md` —— 起草 / 修订后的 Markdown。
- `citation_result.json` —— 引用校验结果。
- `evaluation_report.json` / `review_report.json` —— 质量评测与审阅。
- `repair_log.json` —— 修复 Agent 做了什么、为什么。
- `final_state.json` —— 完整运行状态（`completed` 或 `quality_failed`），便于追溯。

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
├── llm_client.py                # Intern-S2 客户端（+ 可选生成模型）
├── harness/                     # 调度：agent loop、planner、注册表、
│   │                            #   goal gate、状态、记忆
│   └── agents/                  # 有界 Agent 关节：策略 / 修复
├── tools/
│   ├── clients/                 # SciVerse、MinerU、fake LLM
│   ├── phases/                  # P1–P6 知识管线
│   ├── models/                  # 带类型的请求/产物 schema
│   ├── nlp/                     # NLI 校验器、数据清洗
│   ├── verify/                  # claim 映射、归属契约、结构检查
│   ├── knowledge_pipeline_worker.py   # 知识管线入口（P1–P6）
│   ├── write_survey.py          # 起草（为验证而写）
│   ├── verify_citations.py      # 三层引用校验
│   ├── revise_survey.py         # 修复（删除式 / 修复 Agent）
│   └── render_report.py         # 评测 + HTML/PDF 渲染
├── tests/                       # 单元 + 集成
├── docs/                        # 设计文档（入口：系统架构与流程.md）
├── requests/  output/  cache/   # 运行时 I/O
└── .env.example
```

---

## 🔌 外部 API 契约

经 2026-07-05 对线上 API 验证（字段级完整契约见 `docs/外部服务接口/`）：

- **Intern-S2-Preview** —— 核查/决策 LLM，OpenAI 兼容。约 1 请求 / 2 秒
  （客户端内置节流）。生成类任务路由到可选的生成模型（DeepSeek Flash，关思
  考——Intern 默认先推理，小预算下会把答案挤没），Intern 兜底。
- **SciVerse** —— `https://api.sciverse.space`。`POST /meta-search`（元数据；
  论文元数据唯一来源）与 `POST /agentic-search`（带页码的可引用 chunk）。
- **MinerU** —— `https://mineru.net`。PDF → 结构化文本 + 图表。

---

## 🛡️ 防幻觉（五道锁，代码强制）

1. **元数据仅来自 SciVerse** —— 题目、作者、年份、会议绝不由 LLM 生成。
2. **Claim-to-Evidence 绑定** —— 每条论断指向带页码的具体证据片段（`claim_map`）。
3. **白名单 + NLI 校验** —— 每条引用过白名单，且逐条论断过 NLI 语义核对。
4. **归属检查** —— 别人的工作不能算成引用论文自己的。
5. **图表只来自真实来源** —— `figure_bank` / `table_bank` 从解析论文挖掘，
   或由已验证内容生成。

---

## 📄 许可

上海人工智能实验室 2026 暑期夏令营 Hackathon 项目。贡献者与素材归属见
项目内文件。
