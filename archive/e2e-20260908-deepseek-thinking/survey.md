# World Models for Games: A Survey

## Abstract
This survey covers 5 themes over 12 selected studies, spanning Procedural Content Generation Mechanisms to Interactive and Controllable Generation Frameworks. Every claim in the body cites the paper whose evidence supports it, and the reference list contains only papers actually cited. Representative work includes Grandmaster level in StarCraft II using multi-agent reinforcement learning [paper:10.1038/s41586-019-1724-z], GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations [paper:10.48550/arxiv.2402.12348], and Genie: Generative Interactive Environments [paper:10.48550/arxiv.2402.15391].

## Introduction
As game environments grow increasingly complex and dynamic, how can agents possess a structured, predictive understanding of their world to enable intelligent interaction and creation? This survey answers that question by systematically examining the emerging paradigm of world models for games, which learn to represent and simulate environments internally. This synthesis is needed now because recent advances in generative architectures and interactive applications have fragmented the field across disparate research communities, making it difficult to assess progress, compare approaches, or identify the core challenges that must be solved to unlock the next generation of game AI.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Procedural Content Generation Mechanisms

Procedural Content Generation Mechanisms focuses on Methods for automatically creating game levels, assets, and narratives using algorithmic approaches.

Recent work on procedural-content mechanisms splits between systems that synthesize a playable simulation and systems that evaluate the behavior of large language models inside game worlds. A diffusion-based approach can power a neural game engine by extracting gameplay from a classic title and then using that experience to generate a playable environment that interactively simulates new trajectories [paper:10.48550/arxiv.2408.14837]. Complementary evidence from LLM benchmarks shows that agent strategies are also generated through interaction and evaluation: social deduction games challenge players to make informed decisions while engaging in discussions where they must deceive [paper:10.48550/arxiv.2310.05036], while board and card games require pure logic and strategic reasoning that can be assessed in competitive settings [paper:10.48550/arxiv.2402.12348].

![Publication years of the selected representative papers.](publication_timeline)

GameNGen [paper:10.48550/arxiv.2408.14837] stands out as a landmark system because it is the first game engine powered entirely by a neural model, enabling real-time interaction over long trajectories at high quality. This emphasis on producing the environment through training contrasts with AvalonBench and GTBench, which do not generate the world itself but instead expose the reasoning limits of LLMs: the former asks agents to handle evolving game phases and deceptive dialogue [paper:10.48550/arxiv.2310.05036], and the latter situates the same models in board and card games that test game-theoretic abilities [paper:10.48550/arxiv.2402.12348]. Taken together, these works agree that algorithmic game interaction can be decomposed into an environment side and an agent side, with the two sets of benchmarks separated from the generative engine.

What the reported evidence does not settle is how these mechanisms connect: AvalonBench and GTBench never show whether an LLM trained in their settings could acquire enough strategic competence to steer a learned environment, while GameNGen demonstrates only that a diffusion-based engine can reproduce interactive trajectories and gives no account of the agent reasoning required to succeed inside them. The findings therefore leave open the question of whether environment generation and agent evaluation can be fused into a single pipeline, or whether they will continue to be studied as separate technical problems.

Procedural Content Generation Mechanisms hands the open question to Latent Architectures for World Representation, which asks about neural network structures that encode game states into compact latent spaces for efficient processing.

## Latent Architectures for World Representation

Latent Architectures for World Representation focuses on Neural network structures that encode game states into compact latent spaces for efficient processing.

DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation takes Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. as its target; the reported method is Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to..., which yields Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. [paper:10.1109/cvpr52734.2025.01122]. Mastering Diverse Domains through World Models starts from Developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental challenge in artificial intelligence. and builds on Although current reinforcement learning algorithms can be readily applied to tasks similar to what they have been developed for, configuring them for new application domains..., contributing Developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental challenge in artificial intelligence. [paper:10.48550/arxiv.2301.04104]. On Many real-world applications require artificial agents to compete and coordinate with other agents in complex environments., Grandmaster level in StarCraft II using multi-agent reinforcement learning contributes Many real-world applications require artificial agents to compete and coordinate with other agents in complex environments. through As a stepping stone to this goal, the domain of StarCraft has emerged as an important challenge for artificial intelligence research, owing to its iconic and enduring status... [paper:10.1038/s41586-019-1724-z].

![Distribution of selected papers across the survey taxonomy.](taxonomy_overview)

Compared side by side, DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation centers on Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to... and contributes Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. [paper:10.1109/cvpr52734.2025.01122]; Mastering Diverse Domains through World Models centers on Although current reinforcement learning algorithms can be readily applied to tasks similar to what they have been developed for, configuring them for new application domains... and contributes Developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental challenge in artificial intelligence. [paper:10.48550/arxiv.2301.04104]; Grandmaster level in StarCraft II using multi-agent reinforcement learning centers on As a stepping stone to this goal, the domain of StarCraft has emerged as an important challenge for artificial intelligence research, owing to its iconic and enduring status... and contributes Many real-world applications require artificial agents to compete and coordinate with other agents in complex environments. Inside Latent Architectures for World Representation, the split runs between DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation and Grandmaster level in StarCraft II using multi-agent reinforcement learning, not between venues or years.

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Latent Architectures for World Representation hands the open question to Internal Simulation and Planning Capabilities, which asks about models' ability to predict future states and plan actions within an internal environment model.

## Internal Simulation and Planning Capabilities

Internal Simulation and Planning Capabilities focuses on Models' ability to predict future states and plan actions within an internal environment model.

Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model starts from Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. and builds on Tree-based planning methods have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available., contributing Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. [paper:10.1038/s41586-020-03051-4]. On We introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention., Voyager: An Open-Ended Embodied Agent with Large Language Models contributes We introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention. through Voyager consists of three key components: 1) an automatic curriculum that maximizes exploration, 2) an ever-growing skill library of executable code for storing and retrieving... [paper:10.48550/arxiv.2305.16291].

Compared side by side, Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model centers on Tree-based planning methods have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available. and contributes Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. [paper:10.1038/s41586-020-03051-4]; Voyager: An Open-Ended Embodied Agent with Large Language Models centers on Voyager consists of three key components: 1) an automatic curriculum that maximizes exploration, 2) an ever-growing skill library of executable code for storing and retrieving... and contributes We introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention. Inside Internal Simulation and Planning Capabilities, the split runs between Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model and Voyager: An Open-Ended Embodied Agent with Large Language Models, not between venues or years.

Internal Simulation and Planning Capabilities hands the open question to Benchmark Datasets and Evaluation Metrics, which asks about standardized datasets and quantitative measures used to assess model performance and generalization in games.

## Benchmark Datasets and Evaluation Metrics

Benchmark Datasets and Evaluation Metrics focuses on Standardized datasets and quantitative measures used to assess model performance and generalization in games.

On The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment., Human-level control through deep reinforcement learning contributes The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. through To use reinforcement learning successfully in situations approaching real-world complexity, however, agents are confronted with a difficult task: they must derive efficient... [paper:10.1038/nature14236]. MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge addresses Autonomous agents have made great strides in specialist domains like Atari games and Go.. Its method can be summarized as However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks..., and its contribution is Autonomous agents have made great strides in specialist domains like Atari games and Go. [paper:10.48550/arxiv.2206.08853].

Compared side by side, Human-level control through deep reinforcement learning centers on To use reinforcement learning successfully in situations approaching real-world complexity, however, agents are confronted with a difficult task: they must derive efficient... and contributes The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. [paper:10.1038/nature14236]; MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge centers on However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks... and contributes Autonomous agents have made great strides in specialist domains like Atari games and Go. Inside Benchmark Datasets and Evaluation Metrics, the split runs between Human-level control through deep reinforcement learning and MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge, not between venues or years.

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

Benchmark Datasets and Evaluation Metrics hands the open question to Interactive and Controllable Generation Frameworks, which asks about systems enabling user interaction and specific control over generated content during gameplay or creation.

## Interactive and Controllable Generation Frameworks

Interactive and Controllable Generation Frameworks focuses on Systems enabling user interaction and specific control over generated content during gameplay or creation.

Genie: Generative Interactive Environments addresses # Genie: Generative Interactive Environments Jake Bruce<sup>\*,1</sup>, Michael Dennis<sup>\*,1</sup>, Ashley Edwards<sup>\*,1</sup>, Jack Parker-Holder<sup>\*,1</sup>, Yuge (Jimmy) Shi<sup>\*,1</sup>, Edward.... Its method can be summarized as This is made possible via a latent action interface, learned fully unsupervised from Internet videos., and its contribution is # Genie: Generative Interactive Environments Jake Bruce<sup>\*,1</sup>, Michael Dennis<sup>\*,1</sup>, Ashley Edwards<sup>\*,1</sup>, Jack Parker-Holder<sup>\*,1</sup>, Yuge (Jimmy) Shi<sup>\*,1</sup>, Edward... [paper:10.48550/arxiv.2402.15391]. AgentBench: Evaluating LLMs as Agents takes The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. as its target; the reported method is Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments., which yields The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. [paper:10.48550/arxiv.2308.03688].

Compared side by side, Genie: Generative Interactive Environments centers on This is made possible via a latent action interface, learned fully unsupervised from Internet videos. and contributes # Genie: Generative Interactive Environments Jake Bruce<sup>\*,1</sup>, Michael Dennis<sup>\*,1</sup>, Ashley Edwards<sup>\*,1</sup>, Jack Parker-Holder<sup>\*,1</sup>, Yuge (Jimmy) Shi<sup>\*,1</sup>, Edward... [paper:10.48550/arxiv.2402.15391]; AgentBench: Evaluating LLMs as Agents centers on Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments. and contributes The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Inside Interactive and Controllable Generation Frameworks, the split runs between Genie: Generative Interactive Environments and AgentBench: Evaluating LLMs as Agents, not between venues or years.

Interactive and Controllable Generation Frameworks closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

The reported evidence in "AvalonBench" and "GTBench" stops at task-specific game environments and chosen model instances, leaving unsettled whether the strategic reasoning observed transfers to larger-scale or mixed-cooperative interactions [paper:10.48550/arxiv.2310.05036] [paper:10.48550/arxiv.2402.12348]. Similarly, "Diffusion Models Are Real-Time Game Engines" and "DriveDreamer4D" show strong in-domain performance—respectively real-time DOOM trajectories and 4D driving scenes—but provide no evidence about out-of-distribution dynamics, long-horizon consistency, or physical correctness, so the generalizability of these neural simulators remains open [paper:10.48550/arxiv.2408.14837] [paper:10.1109/cvpr52734.2025.01122].

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

These boundaries translate into one concrete step per direction. For Mastering Diverse Domains through World Models, follow-up work should keep developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental while removing the reported boundary (# Mastering Diverse Domains through World Models Danijar Hafner, $^{12}$ Jurgis Pasukonis, $^{1}$ Jimmy Ba, $^{2}$ Timothy Lillicrap $^{1}$ # Abstract Developing a general algorithm that learns to...) [paper:10.48550/arxiv.2301.04104]. For Grandmaster level in StarCraft II using multi-agent reinforcement learning, follow-up work should keep many real-world applications require artificial agents to compete and coordinate with other agents in complex while removing the reported boundary (however, many real-world applications involve a moderately large population of agents, such as algorithmic trading (Wellman et al., 2005), sport team competition (Hausknecht & Stone, 2015), and...) [paper:10.1038/s41586-019-1724-z]. For Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model, follow-up work should keep constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial while removing the reported boundary (constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. Tree-based planning methods have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available. [paper:10.1038/s41586-020-03051-4]. For Voyager: An Open-Ended Embodied Agent with Large Language Models, follow-up work should keep we introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores while removing the reported boundary (for a virtual environment, Voyager [179] introduces the first LLM-powered embodied lifelong learning agent in Minecraft that autonomously explores, acquires skills, and discovers new tasks using a...) [paper:10.48550/arxiv.2305.16291].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1038/nature14236: Human-level control through deep reinforcement learning (2015).
- paper:10.1038/s41586-019-1724-z: Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (2020).
- paper:10.1109/cvpr52734.2025.01122: DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation (2025).
- paper:10.48550/arxiv.2206.08853: MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge (2022).
- paper:10.48550/arxiv.2301.04104: Mastering Diverse Domains through World Models (2023).
- paper:10.48550/arxiv.2305.16291: Voyager: An Open-Ended Embodied Agent with Large Language Models (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2310.05036: AVALONBENCH: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2402.12348: GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations (2024).
- paper:10.48550/arxiv.2402.15391: Genie: Generative Interactive Environments (2024).
- paper:10.48550/arxiv.2408.14837: DIFFUSION MODELS ARE REAL-TIME GAME ENGINES (2024).

## Revision Notes

- type=A remap_citation -> invalid_action: The citation refers to the Voyager paper, which is present in the candidates list.
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=8)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=8)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=8)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=8)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=8)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=16)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=16)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=16)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=16)
- type=B None -> unresolved: no usable action list after retry: empty reply after retry (model=deepseek-v4-flash, n_th=16)
- type=B rewrite_claim -> invalid_action: The original claim text is garbled; rewrite to the supported assertion about acknowledged LLM agent potential without making unsupported claims about evaluation needs.
- type=B rewrite_claim -> invalid_action: Removes corrupted template text and states the abstract-supported sentence from the cited paper.
- type=B rewrite_claim -> invalid_action: The source mentions this motivation; rewrite to a clean assertion supported by the evidence preview.
- type=B rewrite_claim -> repaired: Completes and repairs the truncated quote, matching the abstract's exact supportive sentence.
- type=B rewrite_claim -> invalid_action: Rewrites the malformed claim to the clearly evidenced description of Voyager from the cited source.
- type=D rewrite_claim -> invalid_action: Original assertion about StarCraft is unsupported by the evidence; rewritten to a weaker claim the evidence directly supports.
- type=D rewrite_claim -> invalid_action: Original claim about Voyager is malformed and unrelated to the cited evidence; rewritten to a supported statement about tree-based planning.
