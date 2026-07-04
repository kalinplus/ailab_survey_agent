# Game Craft Agent / Model 调研：以 World Model 为主线

检索日期：2026-06-17  
范围：game agent、LLM/MLLM game agent、game world model、interactive world model、neural game engine、GameCraft 类模型、survey / awesome 仓库 / benchmark。  
说明：citation 主要来自 OpenAlex、论文页或检索结果的近似值；GitHub stars 来自 2026-06-17 附近 GitHub API 检索。新近 arXiv 工作 citation 往往严重滞后，因此不要只按引用数判断价值。

## 0. 总体判断

“game craft agent / model”现在最好分成两条相互汇合的线：

1. **Game Agent**：模型在已有游戏环境中感知、规划、行动。早期以 RL / search / self-play 为核心，后期引入 LLM/MLLM、记忆、工具调用、自然语言规划和通用电脑控制。
2. **Game World Model / GameCraft**：模型学习或生成游戏世界动力学。早期是 agent 内部的 latent dynamics model，后期变成可直接交互的视频/3D世界生成器，即 neural game engine 或 foundation world model。

world model 的“起承转合”可以概括为：

> 游戏作为 AI benchmark -> agent 在真实游戏引擎中学习 -> agent 学习内部世界模型用于规划 -> 生成模型直接模拟可交互游戏世界 -> foundation interactive world model 成为 agent 的训练场、模拟器和游戏创作工具。

核心矛盾也从“能不能赢游戏”变成了：

- 能不能学到可泛化的世界动力学？
- 能不能实时响应用户动作？
- 能不能保持长程一致性和空间/物理一致性？
- 能不能作为 agent 的训练环境，而不是只有 demo 视频？
- 能不能被可靠评测，而不是只靠主观观感？

## 1. 起：游戏作为通用智能 Benchmark

这一阶段，世界由真实游戏引擎给出，模型主要学习 policy/value/search。它们不一定是 world model 论文，但定义了“为什么游戏是通用智能试验场”。

| 工作 | 年份 | 近似影响力 | 为什么重要 |
|---|---:|---:|---|
| [DQN: Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) | 2015 | Nature 页 25k+ citations；OpenAlex 约 29k | Atari 49 games，从像素到动作，奠定 deep RL game agent 范式。 |
| [AlphaGo](https://www.nature.com/articles/nature16961) | 2016 | Nature 页 13k+ citations；OpenAlex 约 15k | policy/value network + MCTS，展示神经网络和搜索结合能达到超人水平。 |
| [AlphaGo Zero](https://www.nature.com/articles/nature24270) | 2017 | OpenAlex 约 9k | 去掉人类棋谱，纯 self-play。 |
| [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) | 2018 | OpenAlex 约 3.5k | 单一算法掌握 chess / shogi / Go。 |
| [AlphaStar: Grandmaster level in StarCraft II](https://www.nature.com/articles/s41586-019-1724-z) | 2019 | OpenAlex 约 3.4k | 实时战略、多智能体、长时序、部分可观测。 |
| [OpenAI Five: Dota 2 with Large Scale Deep RL](https://arxiv.org/abs/1912.06680) | 2019 | OpenAlex 约 1k | 大规模 self-play 和多智能体协作。 |
| [Human-level performance in 3D multiplayer games](https://www.science.org/doi/10.1126/science.aau6249) | 2019 | OpenAlex 约 680 | 3D first-person capture-the-flag，多智能体和 population-based training。 |
| [Agent57](https://arxiv.org/abs/2003.13350) | 2020 | OpenAlex 约 140 | Atari 全集 human benchmark 的重要节点。 |
| [CICERO / Diplomacy](https://www.science.org/doi/10.1126/science.ade9097) | 2022 | OpenAlex 约数十到百级，数据库差异大 | 语言沟通 + 策略推理，连接 LLM agent 与 game-theoretic reasoning。 |

这一阶段的问题是：真实游戏引擎昂贵、封闭、不可微、数据效率低；agent 很难在内部“想象未来”。这推动了 world model 线。

## 2. 承：Agent 内部的 World Model

这里的 world model 不是直接生成可玩的游戏画面，而是 agent 内部学习一个 dynamics model，用于 imagined rollout、planning 或 policy learning。

| 工作 | 年份 | 近似影响力 | 核心贡献 |
|---|---:|---:|---|
| [World Models](https://arxiv.org/abs/1803.10122) | 2018 | 高影响力，OpenAlex 检索需人工核对 | VAE + MDN-RNN + controller，在 learned dream environment 中训练 policy，再迁移到真实环境。 |
| [PlaNet: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) | 2018/2019 | OpenAlex 约 360+ | 从像素学习 latent dynamics，在 latent space 中 online planning。 |
| Dreamer 系列 | 2019-2023 | 高影响力 | 不只规划，而是在 latent imagination 中学习 actor-critic。 |
| [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) | 2023 | OpenAlex 约 90+，实际影响高 | 单一配置跨 150+ tasks，并在 Minecraft 从零收集 diamond。 |
| [MuZero](https://www.nature.com/articles/s41586-020-03051-4) | 2020 | OpenAlex 约 1.3k | 不需要已知规则，学习可用于 MCTS 的 reward/value/policy dynamics，在 Atari、Go、Chess、Shogi 上强。 |
| [EfficientZero](https://arxiv.org/abs/2111.00210) | 2021 | 中高影响 | 数据高效的 MuZero 类方法，Atari 100k setting 重要基线。 |
| [IRIS: Transformers are Sample-Efficient World Models](https://arxiv.org/abs/2209.00588) | 2022/2023 | repo `eloialonso/iris` 约 890 stars | 用 transformer/tokenized world model 做 Atari。 |
| [DIAMOND: Diffusion for World Modeling](https://arxiv.org/abs/2405.12399) | 2024 | repo `eloialonso/diamond` 约 2k stars | diffusion world model；视觉细节对 Atari world modeling 很重要；NeurIPS 2024 Spotlight。 |

这一阶段的关键词：

- latent dynamics
- imagination-based RL
- model-based planning
- sample efficiency
- policy/value/search 与模型学习耦合

局限：

- 主要服务 agent training/planning，不一定能直接作为用户可交互世界。
- 视觉世界通常低分辨率、短 horizon。
- learned model 的误差会在 rollout 中累积。

## 3. 转：World Model 变成 Neural Game Engine

这里开始从“agent 内部想象”转向“模型本身生成可交互游戏画面”。这是 GameCraft/interactive world model 的直接前身。

| 工作 | 年份 | 影响 | 核心意义 |
|---|---:|---:|---|
| [GameGAN: Learning to Simulate Dynamic Environments](https://arxiv.org/abs/2005.12126) | 2020 | OpenAlex 约个位到数十级，方向影响大 | 早期 neural game engine，从 gameplay video + action 学习 Pac-Man 类环境的生成模拟。 |
| [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) | 2024 | OpenAlex 约 10+，新但方向关键 | DeepMind 11B foundation world model，从无动作标注互联网视频学习 controllable latent action，生成可交互环境。 |
| [GameNGen: Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837) | 2024 | OpenAlex 低，但传播广 | 用 diffusion model 实时模拟 Doom，宣称 20+ FPS，代表“游戏引擎可由生成模型近似”。 |
| [Oasis / Decart & Etched](https://oasis.decart.ai/) | 2024 | 行业关注高 | Minecraft-like AI world model demo，展示“实时可玩生成世界”的大众化可能性。 |
| [Microsoft Muse / WHAM](https://www.microsoft.com/en-us/research/project/muse/) | 2025 | 新近 | World and Human Action Model，面向 gameplay ideation 和游戏开发辅助。 |

这一阶段的变化：

- 输入从 state/action 变成 video/action/history。
- 目标从 reward prediction / latent rollout 变成 high-fidelity next-frame or video prediction。
- 交互从 agent 内部 rollout 变成用户实时控制。

关键挑战：

- 动作控制是否真正生效，而不是只生成看起来像游戏的视频。
- 长程一致性：地形、物体、角色、任务状态是否保持。
- 3D 几何一致性和视角一致性。
- 延迟、帧率、分辨率。
- 是否能支持 agent training，而不是只能做人类 demo。

## 4. 合：Foundation Interactive Game World Model / GameCraft 线

这是最贴近“GameCraft model”的当前主线：模型不只是玩游戏，也不只是预测 latent dynamics，而是作为可交互、可指令跟随、可生成的游戏世界基础模型。

| 工作 | 年份 | 仓库 / 指标 | 核心贡献 |
|---|---:|---:|---|
| [Hunyuan-GameCraft](https://arxiv.org/abs/2506.17201) | 2025 | `Tencent-Hunyuan/Hunyuan-GameCraft-1.0` 约 724 stars | High-dynamic Interactive Game Video Generation with Hybrid History Condition；100+ AAA games、百万级 gameplay recordings；统一键鼠为 camera representation。 |
| [Hunyuan-GameCraft-2](https://arxiv.org/abs/2511.23429) | 2025/2026 | project page / arXiv | Instruction-following Interactive Game World Model；从纯动作控制扩展到自然语言 + 键鼠；提出 InterBench。 |
| [Matrix-Game 2.0](https://arxiv.org/abs/2508.13009) | 2025 | `SkyworkAI/Matrix-Game` 约 2.2k stars | Open-source, real-time, streaming interactive world model。 |
| [Matrix-Game 3.0](https://arxiv.org/abs/2604.08995) | 2026 | 同上 | 720p、40 FPS、长程记忆、实时流式交互。 |
| [Yume: An Interactive World Generation Model](https://arxiv.org/abs/2507.17744) | 2025 | 开源权重/数据/代码 | text/image/video 到 interactive world；强调 keyboard exploration、memory、MVDT。 |
| [HY-WorldPlay / HY-World 1.5](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | 2026 | 约 1.5k stars | 实时延迟和几何一致性的系统框架。 |
| [DreamX-World](https://github.com/AMAP-ML/DreamX-World) | 2026 | 约 357 stars | General-purpose interactive world model。 |
| [LIVE: Long-Horizon Interactive Video World Modeling](https://arxiv.org/abs/2602.03747) | 2026 | 新近 | 长 horizon interactive video world modeling，针对误差累积。 |
| [WorldMark](https://arxiv.org/abs/2604.21686) | 2026 | `alaya-studio/WorldMark` 新仓库 | Unified benchmark for interactive video world models，评价 visual quality、control、world consistency。 |
| [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) | 2024 | DeepMind blog | 单张 prompt image 生成可交互 3D-like environment。 |
| [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) | 2025 | DeepMind blog | 文本生成 720p、24 FPS、数分钟一致的 realtime interactive world。 |

这条线与传统 game agent 的关系：

- 过去：agent 在真实游戏中训练。
- 现在：world model 可以生成训练环境、模拟器、任务变体、罕见场景。
- 未来：agent 与 world model 共同进化，world model 负责生成/模拟世界，agent 负责探索/规划/完成任务，二者形成闭环。

## 5. LLM / MLLM Game Agent 线

这条线不局限于 Minecraft，但 Minecraft 因为开放世界、长任务链、可编程接口和大量互联网知识，成为高频环境。

| 工作 | 年份 | 指标 | 核心贡献 |
|---|---:|---:|---|
| [VPT: Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795) | 2022 | `openai/Video-Pre-Training` 约 1.7k stars | 从无标注视频学习行为先验，再用少量标注动作对齐；Minecraft 是案例，但方法对视频到行为模型很关键。 |
| [MineDojo](https://arxiv.org/abs/2206.08853) | 2022 | `MineDojo/MineDojo` 约 2.2k stars | Open-ended embodied agent 环境 + 互联网规模知识；NeurIPS Outstanding Paper。 |
| [Voyager](https://arxiv.org/abs/2305.16291) | 2023 | `MineDojo/Voyager` 约 7k stars；OpenAlex 约 190+ citations | LLM lifelong embodied agent；自动课程、技能库、迭代 prompting。 |
| [GITM: Ghost in the Minecraft](https://arxiv.org/abs/2305.17144) | 2023 | `OpenGVLab/GITM` 约 640 stars | LLM + text-based knowledge + memory，完成开放世界任务。 |
| [JARVIS-1](https://arxiv.org/abs/2311.05997) | 2023 | `CraftJarvis/JARVIS-1` 约 397 stars | Memory-augmented multimodal language model + controller。 |
| [Cradle](https://arxiv.org/abs/2403.03186) | 2024 | 新近 | Foundation agents for general computer control；覆盖 RDR2、Cities: Skylines、Stardew Valley 等 GUI/游戏。 |
| [Generative Agents](https://dl.acm.org/doi/10.1145/3586183.3606763) | 2023 | OpenAlex 约 1.4k citations | 游戏/模拟社会中的 LLM agent memory-reflection-planning 范式。 |
| SIMA / SIMA 2 | 2024-2026 | DeepMind 官方 | Generalist 3D virtual environment agent，与 Genie/world model 线高度相关。 |

这条线的核心模块：

- perception：截图、视频、状态文本、多模态输入
- planning：LLM decomposition、task graph、chain-of-thought / tree search
- memory：episodic memory、skill library、world knowledge
- action grounding：API 调用、键鼠、低层控制器
- reflection：失败分析、自我改进、curriculum

与 world model 结合的机会：

- 用 interactive world model 作为 simulator 给 LLM agent 做低成本训练。
- 用 world model 做 look-ahead planning，LLM 负责高层策略。
- world model 生成新任务/新地图，agent 做自动探索和评测。
- LLM agent 反过来给 world model 收集高价值交互轨迹。

## 6. Survey / Awesome / Benchmark / Toolkit

### 6.1 Survey

| 资源 | 类型 | 说明 |
|---|---|---|
| [A Survey on Large Language Model-Based Game Agents](https://arxiv.org/abs/2404.02039) | Survey | 覆盖 adventure、communication、competition、cooperation、simulation、crafting/exploration 等 game agent 类型；其配套 awesome repo 很有用。 |
| [Understanding World or Predicting Future? A Comprehensive Survey of World Models](https://arxiv.org/abs/2411.14499) | Survey | world model 总综述，适合搭建 world model 起承转合。 |
| [Procedural Content Generation via Machine Learning](https://ieeexplore.ieee.org/document/8382280) | Survey / position | PCGML 经典综述，连接 game generation 和 generative model。 |
| [Deep learning, reinforcement learning, and world models](https://www.sciencedirect.com/science/article/pii/S0893608022001150) | Survey | world model / deep RL 背景。 |

### 6.2 Awesome 仓库

| 仓库 | stars 近似 | 价值 |
|---|---:|---|
| [git-disl/awesome-LLM-game-agent-papers](https://github.com/git-disl/awesome-LLM-game-agent-papers) | 914 | LLM game agent 最直接入口，按游戏类型和 agent 机制分类。 |
| [knightnemo/Awesome-World-Models](https://github.com/knightnemo/Awesome-World-Models) | 3024 | world model 综合入口，覆盖 robotics、driving、video、interactive world 等。 |
| [datamllab/awesome-game-ai](https://github.com/datamllab/awesome-game-ai) | 970 | 偏 game AI / multi-agent RL。 |
| [simoninithomas/awesome-ai-tools-for-game-dev](https://github.com/simoninithomas/awesome-ai-tools-for-game-dev) | 993 | 偏游戏开发中的 AI 工具。 |
| [NegarMirgati/awesome-pcgml](https://github.com/NegarMirgati/awesome-pcgml) | 5 | PCGML 论文清单，star 少但主题相关。 |

### 6.3 Benchmark / Environment / Toolkit

| 平台 | stars 近似 | 说明 |
|---|---:|---|
| [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) | 19.5k | Unity 游戏/仿真训练智能体的主流工具。 |
| [OpenAI Gym](https://github.com/openai/gym) | 37k | RL benchmark 基础设施，虽已被 Gymnasium 继承但历史影响很大。 |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | 5.3k | 博弈、RL、search/planning 环境集合。 |
| [DeepMind Lab](https://github.com/google-deepmind/lab) | 7.4k | 3D first-person agent research platform。 |
| [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | 2.4k | Atari Learning Environment。 |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) | 2.0k | Doom-based RL environment。 |
| [TextWorld](https://github.com/microsoft/TextWorld) | 1.4k | Text-based game environment。 |
| [NLE](https://github.com/facebookresearch/nle) | 984 | NetHack Learning Environment，复杂长程 roguelike。 |
| [MiniHack](https://github.com/facebookresearch/minihack) | 518 | 基于 NetHack 的可控开放任务环境。 |
| [Melting Pot](https://github.com/google-deepmind/meltingpot) | 846 | 多智能体社会互动 benchmark。 |
| [MineDojo](https://github.com/MineDojo/MineDojo) | 2.2k | Minecraft open-ended embodied agent platform。 |
| [Crafter](https://github.com/danijar/crafter) | 560 | 类 Minecraft 的轻量 open-ended benchmark。 |

## 7. 推荐阅读顺序

如果目标是系统理解 game craft / world model，而不是只看最新 demo，建议按以下顺序：

1. **基础 RL game agent**
   - DQN
   - AlphaGo / AlphaZero
   - AlphaStar
   - MuZero

2. **内部 world model**
   - World Models
   - PlaNet
   - Dreamer 系列，尤其 DreamerV3
   - IRIS / DIAMOND

3. **world model 变 neural game engine**
   - GameGAN
   - Genie
   - GameNGen
   - Oasis
   - Microsoft Muse / WHAM

4. **GameCraft / interactive world model 最新线**
   - Hunyuan-GameCraft
   - Hunyuan-GameCraft-2
   - Matrix-Game 2.0 / 3.0
   - Yume
   - HY-WorldPlay
   - Genie 2 / Genie 3
   - WorldMark

5. **agent 与 world model 融合**
   - VPT
   - Voyager
   - Cradle
   - SIMA / SIMA 2
   - Generative Agents

## 8. 未来方向和研究空白

### 8.1 评测仍然落后于 demo

当前 interactive game world model 的 demo 很强，但标准化评测刚起步。WorldMark / InterBench 这类工作值得持续跟进。建议关注：

- control alignment：动作是否精确影响世界
- response latency：是否能真实交互
- temporal consistency：长时间运行是否崩坏
- spatial consistency：转身、回看、绕行后是否一致
- object permanence：物体是否保持存在和状态
- task completion：agent 能否在模型世界中完成目标
- transferability：模型世界中训练的 agent 能否迁移到真实游戏

### 8.2 长程记忆是核心瓶颈

短视频生成已经很强，但游戏世界需要状态持续存在：地图、NPC、库存、任务、伤害、资源、天气、物理变化。Matrix-Game 3.0、LIVE、HY-WorldPlay 都在试图解决这个问题。

### 8.3 生成式世界模型与 RL world model 尚未完全融合

RL world model 重 reward/value/planning，生成式 world model 重画质/交互/一致性。真正有价值的是把两者结合：

- high-fidelity visual dynamics
- symbolic / latent state tracking
- reward and affordance prediction
- planning-friendly abstraction
- controllable environment generation

### 8.4 LLM agent 需要 grounding

LLM 很擅长任务分解和反思，但弱在动作落地、空间理解和实时控制。解决路线包括：

- 低层 controller / skill library
- MLLM perception
- learned affordance model
- world model look-ahead
- imitation learning from gameplay video

### 8.5 游戏开发应用可能比通用 AGI 更早落地

GameCraft 类模型短期最可能落地在：

- gameplay ideation
- 快速原型
- 关卡生成
- QA / bug replay
- NPC 行为模拟
- 自动生成训练场景
- 玩家行为预测
- 游戏视频到可玩原型

## 9. 一句话总结

如果只看“game agent”，主线是从 DQN/AlphaGo/AlphaStar 到 LLM/MLLM agents；如果看“game craft model”，主线应从 World Models/PlaNet/Dreamer/MuZero 接到 Genie/GameNGen/DIAMOND，再进入 Hunyuan-GameCraft、Matrix-Game、Yume、HY-WorldPlay、Genie 3 这一代 interactive foundation world model。真正的下一步不是单纯生成更漂亮的视频，而是让 world model 成为 agent 可规划、可训练、可评测、可长期交互的游戏世界。

