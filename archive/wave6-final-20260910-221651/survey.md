# World Models for Games: A Survey

## Abstract
This survey covers 6 research themes and 13 selected papers, spanning Procedural Generation and Scripting to Game-Oriented and Cross-Domain Applications. The review separates reported findings from open questions and lists the papers cited in its discussion. Representative work includes Grandmaster level in StarCraft II using multi-agent reinforcement learning, Mastering Diverse Domains through World Models, and DIFFUSION MODELS ARE REAL-TIME GAME ENGINES.

## Introduction
What does it mean for an artificial agent to learn and use an internal model of a game world, and how can such models be built, evaluated, and exploited across the many roles games demand of them? As games become both a dominant testbed for learning-based control and a practical target for automated design, the field has accumulated a fragmented mix of procedural generators, latent dynamics models, learned simulators, and multi-agent approximations whose relationships and trade-offs remain unclear. This survey answers that question by organizing the area around its core themes and clarifying what is genuinely shared across them.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Procedural Generation and Scripting

Procedural Generation and Scripting focuses on Methods for procedurally generating game worlds and scripting dynamic events, including rule-based, grammar-based, and search-based approaches.

ChatGPT playing good achieved 22.2% win rate against evil rule-based bots, while a good-role bot achieved 38.2% in the same setting [paper:10.48550/arxiv.2310.05036]. GameNGen is the first game engine powered entirely by a neural model, enabling real-time interaction with complex environments over long trajectories at high quality [paper:10.48550/arxiv.2408.14837].

![Publication years of the selected representative papers.](publication_timeline)

These deep neural networks are trained by combining supervised learning from human expert games with reinforcement learning from self-play [paper:10.1038/nature16961]. AVALONBENCH is a game environment for evaluating multi-agent LLM agents, including rule-based baseline opponents and ReAct-style LLM agents with role-specific prompts [paper:10.48550/arxiv.2310.05036]. GameNGen is trained in two phases: an RL-agent learns to play and sessions are recorded, then a diffusion model generates the next frame conditioned on past frames and actions [paper:10.48550/arxiv.2408.14837].

Procedural Generation and Scripting hands the open question to Latent Representations and World Models, which asks about learning compact latent representations of game states and dynamics for prediction, planning, and control in world models, including internal representation and general world model definitions.

## Latent Representations and World Models

Latent Representations and World Models focuses on Learning compact latent representations of game states and dynamics for prediction, planning, and control in world models, including internal representation and general world model definitions.

On Go, chess and shogi, MuZero matched AlphaZero's superhuman performance without any knowledge of the game rules [paper:10.1038/s41586-020-03051-4].

![Distribution of selected papers across the survey taxonomy.](taxonomy_overview)

MuZero's learned model iteratively predicts the reward, action-selection policy, and value function, the quantities most directly relevant to planning [paper:10.1038/s41586-020-03051-4].

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Latent Representations and World Models hands the open question to Generative Simulation and Interactive Control, which asks about generative approaches for simulating futures and enabling interactive control, including video generation, diffusion-based control, and action-conditioned generation.

## Generative Simulation and Interactive Control

Generative Simulation and Interactive Control focuses on Generative approaches for simulating futures and enabling interactive control, including video generation, diffusion-based control, and action-conditioned generation.

At 11B parameters, Genie can be considered a foundation world model [paper:10.48550/arxiv.2402.15391]. AlphaStar achieved Grandmaster level for all three StarCraft races and ranked above 99.8% of officially ranked human players [paper:10.1038/s41586-019-1724-z].

Their method is a multi-agent reinforcement learning algorithm using human and agent game data within a diverse league of adapting strategies [paper:10.1038/s41586-019-1724-z].

Generative Simulation and Interactive Control hands the open question to Multi-Agent and Mean-Field Simulation, which asks about scalable simulation and planning techniques for large populations of agents using mean-field approximations and related multi-agent methods.

## Multi-Agent and Mean-Field Simulation

Multi-Agent and Mean-Field Simulation focuses on Scalable simulation and planning techniques for large populations of agents using mean-field approximations and related multi-agent methods.

Poor long-term reasoning, decision-making, and instruction following are the main obstacles for developing usable LLM agents [paper:10.48550/arxiv.2308.03688].

AgentBench is a multi-dimensional benchmark with 8 distinct environments to assess LLM-as-Agent reasoning and decision-making abilities [paper:10.48550/arxiv.2308.03688].

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

Multi-Agent and Mean-Field Simulation hands the open question to Benchmarks and Evaluation, which asks about datasets, benchmarks, and evaluation frameworks for assessing world models, simulators, and agents across diverse games and tasks, including simulator quality and benchmark dimensions.

## Benchmarks and Evaluation

Benchmarks and Evaluation focuses on Datasets, benchmarks, and evaluation frameworks for assessing world models, simulators, and agents across diverse games and tasks, including simulator quality and benchmark dimensions.

The survey reviews FER methods from the pro-deep learning era using handcrafted features like SVM and HOG through to the deep learning era [paper:10.3390/info15030135]. DriveDreamer is instantiated on the nuScenes benchmark and verified to enable precise, controllable video generation capturing real-world traffic structural constraints [paper:10.48550/arxiv.2309.09777].

DriveDreamer employs a two-stage training pipeline: first learning structured traffic constraints, then anticipating future states [paper:10.48550/arxiv.2309.09777].

Benchmarks and Evaluation hands the open question to Game-Oriented and Cross-Domain Applications, which asks about applications of world models to specific game genres and cross-domain extensions such as autonomous driving, robotics, and social simulacra.

## Game-Oriented and Cross-Domain Applications

Game-Oriented and Cross-Domain Applications focuses on Applications of world models to specific game genres and cross-domain extensions such as autonomous driving, robotics, and social simulacra.

Voyager is the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores, acquires diverse skills, and makes novel discoveries without human intervention [paper:10.48550/arxiv.2305.16291].

Voyager has three key components: an automatic curriculum maximizing exploration, an ever-growing skill library of executable code, and an iterative prompting mechanism using environment feedback, execution errors, and self-verification [paper:10.48550/arxiv.2305.16291]. Training ran 700,000 steps with mini-batches of 4,096 from random initialization, using 5,000 first-generation TPUs for self-play and 64 second-generation TPUs for training [paper:10.48550/arxiv.1712.01815].

Game-Oriented and Cross-Domain Applications closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

The captured evidence does not yet support sharper open-challenge statements.

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.


## Future Directions

The recorded limitations do not yet justify specific research proposals.

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The synthesis rests on the following reported findings. ChatGPT playing good achieved 22.2% win rate against evil rule-based bots, while a good-role bot achieved 38.2% in the same setting [paper:10.48550/arxiv.2310.05036]. GameNGen is the first game engine powered entirely by a neural model, enabling real-time interaction with complex environments over long trajectories at high quality [paper:10.48550/arxiv.2408.14837]. These deep neural networks are trained by combining supervised learning from human expert games with reinforcement learning from self-play [paper:10.1038/nature16961]. Their scopes remain tied to the respective study settings; they do not by themselves establish a shared performance ranking.

## References

- paper:10.1038/nature16961: Mastering the game of Go with deep neural networks and tree search (2016).
- paper:10.1038/s41586-019-1724-z: Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (2020).
- paper:10.3390/info15030135: Article Advances in Facial Expression Recognition: A Survey of Methods, Benchmarks, Models, and Datasets (2024).
- paper:10.48550/arxiv.1712.01815: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (2017).
- paper:10.48550/arxiv.2305.16291: Voyager: An Open-Ended Embodied Agent with Large Language Models (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2309.09777: DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving (2023).
- paper:10.48550/arxiv.2310.05036: AVALONBENCH: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2402.15391: Genie: Generative Interactive Environments (2024).
- paper:10.48550/arxiv.2408.14837: DIFFUSION MODELS ARE REAL-TIME GAME ENGINES (2024).
