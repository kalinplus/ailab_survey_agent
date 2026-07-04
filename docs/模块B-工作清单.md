# 成员 B：论文知识库、Paper Card 与可信引用

> 角色：Knowledge Grounding Engineer（34%）
> 贡献定位：让系统生成的综述不是模型凭空写出来的，而是 grounded on verified papers。

## 1. 职责定位

B 负责整个系统的"知识来源"和"可信性"。核心解决：

```text
论文从哪里来？
引用是不是真的？
每个论断有没有证据？
论文怎么分类？
哪些论文是核心论文？
```

题目特别强调不能有幻觉引用，所以 B 的模块是项目最关键的技术亮点之一。

---

## 2. 基础任务

### B1：整理 seed_papers.json

现场网络和 API 可能不稳定，必须有本地论文种子库。

至少覆盖 6 个方向：

```text
1. Game as AI Benchmark
2. Internal World Model
3. Neural Game Engine
4. Foundation Interactive Game World Model / GameCraft
5. LLM / MLLM Game Agent
6. Benchmark / Evaluation / Toolkit
```

种子论文清单（至少包含）：

```text
DQN, AlphaGo, AlphaZero, AlphaStar, OpenAI Five
World Models, PlaNet, Dreamer, DreamerV3
MuZero, EfficientZero, IRIS, DIAMOND
GameGAN, Genie, GameNGen, Oasis, Muse / WHAM
Hunyuan-GameCraft, Hunyuan-GameCraft-2
Matrix-Game 2.0, Matrix-Game 3.0, Yume, HY-WorldPlay
Voyager, MineDojo, VPT, Cradle, Generative Agents, WorldMark
```

参考：https://github.com/tsinghua-fib-lab/World-Model

---

### B2：实现 Paper Card

每篇论文 → 结构化 JSON：

```json
{
  "paper_id": "dreamerv3_2023",
  "title": "DreamerV3: Mastering Diverse Domains through World Models",
  "authors": ["..."],
  "year": 2023,
  "venue": "arXiv",
  "url": "...",
  "category": "Internal World Model",
  "keywords": ["latent dynamics", "imagination", "model-based reinforcement learning"],
  "problem": "What problem does this paper solve?",
  "method": "What is the core method?",
  "contribution": "What is the main contribution?",
  "limitation": "What are the limitations?",
  "evidence": [{ "source": "abstract", "text": "..." }]
}
```

答辩话术：我们不直接让模型写综述，而是先把每篇论文压缩成结构化 Paper Card，再基于 Paper Card 生成综述。

---

### B3：实现论文分类 Taxonomy Builder

6 个分类类别：

```text
1. Game as AI Benchmark
2. Internal World Model for Agents
3. Neural Game Engine
4. Foundation Interactive Game World Model / GameCraft
5. LLM / MLLM Game Agent
6. Benchmark / Evaluation / Toolkit
```

分类依据：论文标题、摘要、关键词、研究目标、方法类型、应用场景

策略：先规则分类，再让模型辅助判断。

---

### B4：实现 Citation Verifier

5 条规则：

```text
1. 综述中每个 citation_key 必须存在于 citation_index.json
2. 参考文献只能从 paper_cards.json 自动生成
3. 不允许模型自己编造参考文献
4. 引用不存在则标记 invalid
5. 引用存在但 claim 不支持则标记 weak
```

输出示例：

```json
{
  "total_citations": 38,
  "valid_citations": 36,
  "invalid_citations": 2,
  "weak_claims": 4,
  "citation_validity_score": 0.947
}
```

---

### B5：实现 Claim-to-Evidence Map

每个关键论断绑定论文证据：

```json
{
  "claim_id": "claim_012",
  "claim": "Interactive world models shift the role of world models from internal planning modules to real-time controllable simulators.",
  "supporting_papers": ["genie_2024", "gamenegen_2024", "hunyuan_gamecraft_2025"],
  "evidence_status": "supported"
}
```

比普通 Citation Checker 更高级——不仅检查引用是否存在，还检查关键论断是否能映射到具体论文证据。

---

## 3. 创新模块

### 创新 B1：Paper Store

统一知识库：`cache/paper_cards.json` + `cache/citation_index.json` + `cache/paper_store.db`

支撑后续所有生成。

---

### 创新 B2：Claim-to-Evidence Map

把综述里的 claim 和 paper 绑定，直接对应题目要求的"没有幻觉引用、没有缺乏事实证据的幻觉描述"。

---

### 创新 B3：Paper Influence Score

新近 arXiv 工作 citation 往往严重滞后，因此不能只按引用数判断价值。综合评分公式：

```text
final_score =
  citation_score * 0.3
  + recency_score * 0.2
  + category_importance * 0.2
  + open_source_impact * 0.2
  + benchmark_value * 0.1
```

确保新兴 GameCraft 论文不会因为引用少而被漏掉。

---

## 4. 负责的文件

```text
tools/
├── search_papers.py
├── parse_papers.py
├── build_paper_cards.py
├── classify_papers.py
├── verify_citations.py
├── build_claim_map.py
└── paper_ranker.py

cache/
├── seed_papers.json
├── retrieved_papers.json
├── paper_cards.json
├── citation_index.json
└── claim_map.json
```

---

## 5. 答辩内容（第 4～7 分钟）

讲 3 个点：

```text
1. Paper Card 如何把论文变成结构化知识。
2. Citation Verifier 如何避免虚构引用。
3. Claim-to-Evidence Map 如何保证关键论断有证据。
```

重点句：

> 我们不允许模型自由生成参考文献。所有引用必须来自 Paper Store，所有关键论断必须能映射到 supporting papers。

一句话总结：

> 我负责让综述内容可验证、有证据、不幻觉。

---

## 6. 时间安排

### 第 0～2 小时：确定接口

- 确定 paper_cards.json 格式
- 确定 citation_index.json 格式
- 确定 claim_map.json 格式
- 确定与 A（工具注册）和 C（综述生成/评测）的接口

### 第 2～6 小时：基础模块

```text
整理 seed_papers.json
写 build_paper_cards.py
写 classify_papers.py
```

### 第 6～12 小时：打通主流程

确保 `python main.py --topic "World Models for GameCraft"` 能生成 `cache/paper_cards.json`、`cache/taxonomy.json`

### 第 12～18 小时：创新模块

```text
Citation Verifier
Claim-to-Evidence Map
Paper Influence Score
```

### 第 18～22 小时：打磨

- 检查引用
- 配合 C 准备 evaluation_report.json
- 准备答辩讲稿

---

## 7. 与其他成员的协作接口

| 对接 | 方向 | 数据 |
|------|------|------|
| A → B | Agent Loop 调用 B 的工具 | search_papers、build_paper_cards、classify_papers、verify_citations、build_claim_map |
| B → A | 工具注册 | 通过 Tool Registry 注册 |
| B → C | Paper Card + Taxonomy | paper_cards.json、taxonomy.json、claim_map.json → 综述生成和评测 |
| A + B | Memory / Cache | paper_cache.json、failed_cases.md |
