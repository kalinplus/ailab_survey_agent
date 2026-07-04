## **自动综述生成领域调研**

领域已经形成了相当成熟的"技术谱系"，代表性 SOTA 系统包括 AutoSurvey / AutoSurvey2、SurveyForge、SurveyX、LLM×MapReduce-V3、HiReview、InteractiveSurvey、LiRA 等，评测则从早期的 LLM-as-Judge 打分逐步收敛到以 SurGE、SurveyBench、SAM 为代表的、包含引用真实性（NLI 级别）、覆盖率、结构质量和内容质量四维度的标准化框架，且已有多套可直接复用的开源代码（SurveyX、SurveyForge、LLMxMapReduce 等）。**

结合你们「分工.md」里设计的 EviSurvey（Paper Card、Citation Verifier、Claim-to-Evidence Map、Survey Evaluator 等模块），下面把这个方向目前最值得参考的工作、方案设计思路、评测体系和现成工具做一次系统梳理，方便你们对照自己的架构判断哪些是已经做对的、哪些是可以直接"抄作业"的。

### 主线方案：从两阶段 RAG 到记忆驱动的智能体流水线

这个方向最早的严肃系统是 **AutoSurvey**（NeurIPS 2024），它确立了后续几乎所有工作都在沿用的基本范式：先做检索式大纲生成，再让多个"专职"LLM 并行撰写各小节，最后做整合与迭代精炼，用来应对长上下文和模型参数知识不足这两大瓶颈。[OpenReview](https://openreview.net/forum?id=FExX8pMrdT&referrer=%5Bthe+profile+of+Yidong_Wang1%5D%28%2Fprofile%3Fid%3D~Yidong_Wang1%29) 今年新出的 **AutoSurvey2** 在此基础上把检索层升级为基于完整 arXiv 元数据构建的向量数据库（几十万到百万级论文规模），并引入并行分节生成、迭代精炼与实时检索最新文献，用多 LLM 评审框架衡量覆盖度、结构和相关性。[arXiv](https://arxiv.org/html/2510.26012v1)

**SurveyForge**（ACL 2025，上海AI Lab 出品）是目前引用率最高、也最贴近你们分工设计的系统，它明确指出 AI 生成综述的两大短板——大纲缺乏逻辑深度、引用覆盖不到位——并给出了针对性方案：先用"启发式大纲学习"，让模型参考人类写的综述大纲结构和检索到的领域论文来生成大纲；再用一个带"记忆"和"时序感知重排序"的**学者导航智能体（SANA, Scholar Navigation Agent）**为每个小节检索并排序高质量文献，其中的时序感知重排序机制会按发表时间分组，在每个时间段内按引用量取 top-k，兼顾"经典论文"和"新兴论文"，这正好呼应了你们文档里提到的"新近 arXiv 工作引用滞后"的问题，可以直接作为 Paper Influence Score 的参考实现。[ACL Anthology](https://aclanthology.org/2025.acl-long.609.pdf)

**SurveyX**（IAAR-Shanghai）和 **LLM×MapReduce-V3**（清华 THUNLP）代表了另外两条工程化路线。SurveyX 已经做成了可以直接跑的开源系统，提供论文标题加关键词就能生成完整学术综述，并且区分了"完整线上版"和"离线开源版"（离线版依赖用户自己提供本地参考文献，功能上做了阉割）。[GitHub](https://github.com/IAAR-Shanghai/SurveyX) LLM×MapReduce 系列则专注解决"超长材料输入"这个更底层的问题：它借鉴 CNN 卷积思想，用"堆叠式卷积扩展层"逐层把局部信息汇聚成全局表征，让短上下文模型也能处理海量文献输入，V3 版本进一步做成了基于 MCP 协议驱动的分层模块化智能体系统，支持交互式深度综述生成。[GitHub](https://github.com/thunlp/LLMxMapReduce) [ACL Anthology](https://aclanthology.org/2025.emnlp-demos.51)

还有几支队伍专门针对"可信性"和"可读性"这两个和你们 B、C 模块高度相关的痛点：**HiReview** 不依赖纯文本相似度，而是在引文网络上做层次聚类，先构建论文间的分类树（Taxonomy Tree），再基于这个分类结构生成综述内容，本质上是把"论文关系图谱"当成一等公民而不是后处理产物。[arXiv](https://arxiv.org/html/2504.08762v1) **InteractiveSurvey** 则在生成综述正文的同时，对每句话去检索参考文献库中的语义相近片段，用自适应相似度阈值决定是否插入引用，这和你们 Claim-to-Evidence Map 的思路几乎一致，只是它是逐句做而不是逐 claim 做。[arXiv](https://arxiv.org/html/2504.08762v1) **LiRA**（多智能体文献综述框架）用大纲、分节撰写、编辑、审校四类专职智能体模拟人类写综述的完整流程，专门强调降低幻觉、提升可读性，在 SciReviewGen 和某医学数据集上超过了 AutoSurvey 和 MASS-Survey。[arXiv](https://arxiv.org/html/2510.05138v1) 今年 ACL Findings 的 **FIKSurvey**（Feedback Is The Key）则提出"反馈驱动"的迭代改写机制，在四系统的盲评对比中，其内容质量、引用召回率（97.33%）和精确率（92.85%）均超过 AutoSurvey、SurveyX、SurveyForge，接近甚至局部超过人类写手水平，这说明"生成后反馈校验再改写"这个闭环，本质上就是你们 Harness 里 Citation Verifier 反哺 Survey Writer 这条设计的价值所在。[ACL Anthology](https://aclanthology.org/2026.findings-acl.1904.pdf)

下表汇总了几个核心系统在同一基准上的对比数据，可以直观看出各家的技术侧重与短板：

| 系统 | 核心机制 | 优势维度 | 已知短板 |
|---|---|---|---|
| AutoSurvey | 检索+并行分节+精炼 | 结构质量（SQS 最优）、速度快 | 引用准确率、覆盖率偏低 |
| SurveyForge | 启发式大纲+记忆驱动 SANA+时序重排 | 参考文献覆盖率、文档级引用准确率最高 | 内容质量相对一般 |
| StepSurvey | 逐步规划再撰写 | 内容质量（CQS）最高、语言流畅度好 | 结构质量偏弱 |
| SurveyX | 端到端全流程自动化 | 综合表现均衡，人类评审接近 5 分档 | 依赖在线检索能力（离线版功能阉割） |
| FIKSurvey | 反馈驱动迭代改写 | 引用召回/精确率全面领先，逼近人类水平 | 需要多轮反馈，成本更高 |

数据来自 SurGE 基准与 FIK 盲评实验。[arXiv](https://arxiv.org/pdf/2508.15658) [ACL Anthology](https://aclanthology.org/2026.findings-acl.1904.pdf)

### 评测体系：从"整体打分"走向"四维度可分解、有人类对齐验证"的框架

这个方向的评测经历了明显的代际演化。早期方法（AutoSurvey、SurveyForge 自带的评测）都是"自证清白"型——每个系统设计自己的评测标准来验证自己的管线，天然带有偏向自身设计假设的系统性偏差，缺乏跨系统的公平比较基础。[arXiv](https://arxiv.org/pdf/2508.15658)

目前最值得借鉴的标准化框架是清华团队今年提出的 **SurGE**（Survey Generation Evaluation，SIGIR 2026），它把综述生成任务拆解为"检索"和"生成"两个可以独立诊断的阶段，配套一个超百万论文的检索语料库和 205 篇经过博士生标注、Cohen's Kappa 达到 0.792（高度一致）的专家验证真值综述。它的四维评测指标设计得非常精细：**Comprehensiveness（覆盖度）**直接算生成综述参考文献与真值综述参考文献的召回率；**Citation Accuracy（引用准确率）**用一个 NLI（自然语言推理）模型在文档级、章节级、句子级三个粒度上分别判断每条引用是否真的支持它所在位置的论断——如果引用的论文根本不在语料库中，直接判定为幻觉给零分；**Structural Quality**用 LLM-as-Judge 打分（SQS）加上基于语义嵌入的软标题召回率（SHR）双管齐下；**Content Quality**则用 GPT-4o 按流畅度、逻辑清晰度、冗余度等五项标准逐节打分（CQS）。这套框架最关键的贡献是做了严格的元评测：在有/无人类真值参考两种设定下，其指标与专家排序的 Kendall's τ 分别达到 0.805 和 0.610，证明这是一个和人类判断高度对齐、可以规模化替代人工评审的代理指标。[arXiv](https://arxiv.org/pdf/2508.15658)

SurGE 用这套指标测试了 RAG 基线和三大专门化智能体（AutoSurvey、StepSurvey、SurveyForge）后发现一个很扎心的结论：检索阶段召回率上限能到 68%，但端到端综述的实际引用召回率不到 10%，说明当前系统的瓶颈根本不在"找不到论文"，而在"模型不会用检索到的证据"——这也直接验证了你们设计 Claim-to-Evidence Map 这个模块的必要性，因为"检索到"和"真正被论断引用支撑"是两件完全不同的事。同时该研究还发现一个"流畅度陷阱"：朴素 RAG 在 ROUGE、BLEU 这类 n-gram 指标上表现尚可，但引用准确率和召回率全场最低，说明文本读起来通顺不代表内容可信，这也是为什么 SurGE 强调必须用事实中心指标而非传统文本相似度指标来评价综述质量。[arXiv](https://arxiv.org/pdf/2508.15658)

另外两套值得关注的评测体系是 **SurveyForge 自带的 SAM（Survey Assessment Metrics）系列**（拆成 SAM-R 参考文献质量、SAM-O 大纲质量、SAM-C 内容质量三部分，并用真实博士生评审做了 Kappa 一致性验证）[ACL Anthology](https://aclanthology.org/2025.acl-long.609.pdf)，以及 EMNLP 2025 的 **SurveyGen** 基准，它专门针对引用可信度设计了 precision/recall/F1 三件套，通过标题检索去 S2ORC 数据库核实每条生成引用是否真实存在，这个思路和你们 B 模块 Citation Verifier 的规则（"引用必须存在于 citation_index.json"）几乎完全一致，只是它做到了更大规模的自动化验证。[ACL Anthology](https://aclanthology.org/2025.emnlp-main.136.pdf) 更新的 **SurveyBench**（2510.03120，区别于 SurveyForge 自带的同名基准）走的是"读者需求对齐"路线，用 quiz 问答式评测检验生成综述能否真正回答读者关心的问题，同时兼顾图表等非文本元素的丰富度评估，这提示你们的 Interactive HTML Report 如果加入"读者能否通过综述回答关键问题"这个维度，会比单纯的可读性评分更有说服力。[arXiv](https://arxiv.org/html/2510.03120v1)

### 现成工具与可复用代码

如果你们想在 24 小时内少造轮子，几个仓库可以直接参考代码结构甚至复用部分模块：**SurveyX** 的 GitHub 仓库开源了完整离线处理流程，包含 LLM 配置文件（`src/configs/config.py`）和按任务 ID 分文件夹存结果的工作流设计，这个目录结构和你们「分工.md」里规划的 `cache/`、`output/` 分层几乎异曲同工，可以直接参考它的配置文件写法来加速 A 模块的开发。[GitHub](https://github.com/IAAR-Shanghai/SurveyX) **SurveyForge** 的仓库同时开源了代码和对应的 SurveyBench 数据集（Hugging Face 上可下载），如果时间紧张，B 模块的"引用覆盖率评测"可以直接拿它的 SAM-R 实现改造成你们的 Citation Verifier。[ACL Anthology](https://aclanthology.org/2025.acl-long.609.pdf) **LLMxMapReduce** 仓库由 OpenBMB、THUNLP 等联合维护，提供了处理超长文献输入的分治框架代码，如果你们后续论文数量增多导致上下文塞不下，可以直接借用它的"卷积式分层聚合"思路，而不必自己重新设计分块策略。[GitHub](https://github.com/thunlp/LLMxMapReduce) 另外，`worldbench/awesome-ai-auto-research` 这个 GitHub 列表专门按"检索-综述生成-创意生成-实验-写作"整理了全流程自动化研究的论文和工具，其中"Survey & Related Work Generation"板块基本收录了上面提到的所有系统，适合作为你们答辩前查漏补缺的索引。[GitHub](https://github.com/worldbench/awesome-ai-auto-research)

除了这些偏学术的开源系统，市面上也有面向普通研究者的商业化文献综述工具，比如集成了引用管理、PDF 对话、多来源检索的 ResearchPal 和 Paperguide，以及更早期的 Stanford STORM（多视角提问驱动的长文生成方法，最初面向维基百科式文章而非严格学术综述）。不过需要提醒的是，第三方评测（例如一些研究团队做的横评）普遍反映这类商业工具在"引用严谨性"上明显弱于专门针对学术场景设计的 AutoSurvey/SurveyForge 系列系统，Scite.ai 在引用质量上表现相对突出但覆盖面有限，这进一步印证了你们把 Citation Verifier 和 Claim-to-Evidence Map 作为核心创新点这个判断方向是对的——目前几乎所有成熟商业工具的短板恰恰就在这里。[Paperguide](https://paperguide.ai/ai-literature-review) [ResearchPal](https://researchpal.co)

结合以上调研，你们「分工.md」里 EviSurvey 的设计其实已经踩在了这个领域最新的研究前沿上：Paper Card 对应各家系统的结构化知识库构建（AutoSurvey2 的向量数据库、SurveyForge 的 Research Paper Database），Citation Verifier 和 Claim-to-Evidence Map 对应 SurGE 的句子级 NLI 引用准确率和 SurveyGen 的 precision/recall/F1 三件套，Survey Evaluator 对应 SAM 系列和 SurGE 的四维框架。真正值得补强的两处细节是：一是像 SurveyForge 那样引入"时序感知重排序"来解决新兴 GameCraft 论文引用滞后的问题（你们文档里已经提到但可以直接落地成排序公式）；二是可以考虑在答辩里明确引用 SurGE 揭示的"检索上限 68% vs. 端到端召回不到 10%"这个数据，用来论证你们的 Claim-to-Evidence Map 解决的正是当前 SOTA 系统尚未攻克的核心瓶颈，这会让评委更清楚地看到你们的创新不是重复劳动，而是精准打在了这个领域公认的痛点上。

https://github.com/oneal2000/SurGE
https://github.com/IAAR-Shanghai/SurveyX
https://github.com/InternScience/SurveyForge
https://github.com/annihi1ation/auto_research
https://github.com/lira-workflow/auto-review-writing