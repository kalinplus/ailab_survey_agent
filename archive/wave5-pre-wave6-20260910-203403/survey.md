# World Models for Games: A Survey

## Abstract
This survey examines world models for games across four areas: procedural content generation mechanisms, standardized game world benchmarks, latent dynamics and simulation models, and interactive controllable generation frameworks. It covers approaches such as a general algorithm that learns environment models and improves behavior by imagining future scenarios, achieving diamond collection in Minecraft without human data or curricula [paper:10.48550/arxiv.2301.04104], a neural model that functions as a real-time game engine by predicting frames conditioned on past frames and actions [paper:10.48550/arxiv.2408.14837], and a planning algorithm that combines tree-based search with a learned model to achieve superhuman performance without knowledge of underlying dynamics [paper:10.1038/s41586-020-03051-4]. The survey also discusses a general reinforcement learning algorithm that achieves superhuman performance in chess, shogi, and Go through self-play without domain knowledge beyond game rules [paper:10.48550/arxiv.1712.01815]. These claims are grounded in reported evidence including performance across over 150 diverse tasks with a single configuration, real-time frame generation at 20 frames per second on a single TPU with human raters near random chance at distinguishing simulation from game clips, and superhuman results in Atari, Go, chess, and shogi.

## Introduction
What does it mean for a game world model to be *good*, and how can the field tell? Research on world models for games has grown quickly across procedural generation, learned simulation, and interactive generation, yet these strands evaluate their systems against different goals and assumptions, leaving no shared basis for comparing progress or judging whether a model truly captures a game world's dynamics. This survey answers that question by organizing the field around the mechanisms, benchmarks, and frameworks that define what a world model must do.

Standardized Game World Benchmarks collects the work on rule-based taxonomy fallback for Standardized Game World Benchmarks. Latent Dynamics and Simulation Models follows the evidence for rule-based taxonomy fallback for Latent Dynamics and Simulation Models. Interactive Controllable Generation Frameworks maps what is known about rule-based taxonomy fallback for Interactive Controllable Generation Frameworks.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Procedural Content Generation Mechanisms

Procedural Content Generation Mechanisms focuses on Rule-based taxonomy fallback for Procedural Content Generation Mechanisms.

Many real-world applications require artificial agents to compete and coordinate with other agents in complex environments, and the domain of StarCraft has emerged as an important challenge for artificial intelligence research [paper:10.1038/s41586-019-1724-z]. Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence [paper:10.1038/s41586-020-03051-4].

![Publication years of the selected representative papers.](publication_timeline)

Tree-based planning methods have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available [paper:10.1038/s41586-020-03051-4]. The work on Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model addresses planning capabilities by building on tree-based planning methods that have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available [paper:10.1038/s41586-020-03051-4].

The available evidence for Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model is limited to its abstract, so its mechanisms cannot be examined in detail here [paper:10.1038/s41586-020-03051-4].

Procedural Content Generation Mechanisms hands the open question to Standardized Game World Benchmarks, which asks about rule-based taxonomy fallback for Standardized Game World Benchmarks.

## Standardized Game World Benchmarks

Standardized Game World Benchmarks focuses on Rule-based taxonomy fallback for Standardized Game World Benchmarks.

Standardized game world benchmarks provide a rule-based taxonomy fallback for evaluating general algorithms across diverse domains. Mastering Diverse Domains through World Models addresses the fundamental challenge of developing a general algorithm that learns to solve tasks across a wide range of applications [paper:10.48550/arxiv.2301.04104]. AVALONBENCH: Evaluating LLMs Playing the Game of Avalon explores the potential of Large Language Models Agents in playing the strategic social deduction game, Resistance Avalon [paper:10.48550/arxiv.2310.05036].

![Distribution of selected papers across the survey taxonomy.](taxonomy_overview)

Mastering Diverse Domains through World Models notes that although current reinforcement learning algorithms can be readily applied to tasks similar to what they have been developed for, configuring them for new application domains requires significant human expertise and experimentation [paper:10.48550/arxiv.2301.04104]. AVALONBENCH: Evaluating LLMs Playing the Game of Avalon challenges players in Avalon not only to make informed decisions based on dynamically evolving game phases, but also to engage in discussions where they must deceive, deduce, and negotiate with other players [paper:10.48550/arxiv.2310.05036].

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Mastering Diverse Domains through World Models and AVALONBENCH: Evaluating LLMs Playing the Game of Avalon both address the difficulty of applying algorithms to new application domains [paper:10.48550/arxiv.2301.04104] [paper:10.48550/arxiv.2310.05036]. Mastering Diverse Domains through World Models focuses on general algorithm development across a wide range of applications, whereas AVALONBENCH: Evaluating LLMs Playing the Game of Avalon focuses on strategic social deduction gameplay [paper:10.48550/arxiv.2301.04104] [paper:10.48550/arxiv.2310.05036].

Configuring reinforcement learning algorithms for new application domains requires significant human expertise and experimentation [paper:10.48550/arxiv.2301.04104]. Players in Avalon must engage in discussions where they must deceive, deduce, and negotiate with other players, which presents a challenge for Large Language Models Agents [paper:10.48550/arxiv.2310.05036].

Standardized Game World Benchmarks hands the open question to Latent Dynamics and Simulation Models, which asks about rule-based taxonomy fallback for Latent Dynamics and Simulation Models.

## Latent Dynamics and Simulation Models

Latent Dynamics and Simulation Models focuses on Rule-based taxonomy fallback for Latent Dynamics and Simulation Models.

Diffusion Models Are Real-Time Game Engines presents GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality [paper:10.48550/arxiv.2408.14837]. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm addresses the game of chess, the most widely-studied domain in the history of artificial intelligence [paper:10.48550/arxiv.1712.01815].

When trained on the classic game DOOM, GameNGen extracts gameplay and uses it to generate a playable environment that can interactively simulate new trajectories [paper:10.48550/arxiv.2408.14837]. The strongest chess programs are based on a combination of sophisticated search techniques, domain-specific adaptations, and handcrafted evaluation functions that have been refined by human experts over several decades [paper:10.48550/arxiv.1712.01815].

GameNGen is powered entirely by a neural model and enables real-time interaction with a complex environment over long trajectories at high quality [paper:10.48550/arxiv.2408.14837]. In contrast to neural simulation, the strongest chess programs rely on sophisticated search techniques, domain-specific adaptations, and handcrafted evaluation functions refined by human experts over several decades [paper:10.48550/arxiv.1712.01815].

The evidence for Diffusion Models Are Real-Time Game Engines is drawn from full-text material, while the evidence for Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm is also full-text [paper:10.48550/arxiv.2408.14837] [paper:10.48550/arxiv.1712.01815].

Latent Dynamics and Simulation Models hands the open question to Interactive Controllable Generation Frameworks, which asks about rule-based taxonomy fallback for Interactive Controllable Generation Frameworks.

## Interactive Controllable Generation Frameworks

Interactive Controllable Generation Frameworks focuses on Rule-based taxonomy fallback for Interactive Controllable Generation Frameworks.

Genie: Generative Interactive Environments [paper:10.48550/arxiv.2402.15391] presents a generative interactive environment made possible via a latent action interface, learned fully unsupervised from Internet videos. AgentBench: Evaluating LLMs as Agents [paper:10.48550/arxiv.2308.03688] addresses the widely acknowledged potential of Large Language Model (LLM) as agents by providing a means to quantitatively evaluate LLMs as agents on challenging tasks in interactive environments.

Genie: Generative Interactive Environments [paper:10.48550/arxiv.2402.15391] realizes its interactive environment through a latent action interface that is learned fully unsupervised from Internet videos.

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

The available evidence for Genie: Generative Interactive Environments [paper:10.48550/arxiv.2402.15391] does not specify limitations of the latent action interface or of the unsupervised learning from Internet videos.

Interactive Controllable Generation Frameworks closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Across the surveyed work, the evidence stops at demonstrating planning with a learned model in classic board and video domains, learning to solve tasks across diverse applications, and evaluating large language model agents in a strategic social deduction game, leaving open how such capabilities transfer beyond those settings [paper:10.1038/s41586-020-03051-4][paper:10.48550/arxiv.2301.04104][paper:10.48550/arxiv.2310.05036]. What therefore stays unsettled is whether a single general algorithm can combine the planning strengths shown in the first setting, the broad domain coverage shown in the second, and the social reasoning demanded in the third, since no reported result establishes that integration [paper:10.1038/s41586-020-03051-4][paper:10.48550/arxiv.2301.04104][paper:10.48550/arxiv.2310.05036].

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

**Future Directions**

Extend the neural game engine line of work by testing whether the real-time, long-trajectory interaction demonstrated in "DIFFUSION MODELS ARE REAL-TIME GAME ENGINES" can be reproduced beyond the single classic game on which it was trained [paper:10.48550/arxiv.2408.14837]. Follow-up work should train and evaluate the same fully neural engine paradigm on additional complex environments and report whether real-time quality is preserved over long trajectories [paper:10.48550/arxiv.2408.14837]. This would clarify how far the claim of a general neural game engine extends [paper:10.48550/arxiv.2408.14837]. Build on the generative interactive environments of "Genie" by probing how well its generative interactive setup transfers to domains with different visual and control statistics than those used in its original training [paper:10.48550/arxiv.2402.15391]. Concretely, follow-up work should take the Genie approach and systematically vary the environment family, then measure whether interactive generation remains coherent [paper:10.48550/arxiv.2402.15391]. Such a study would test the generality of the generative interactive environment contribution [paper:10.48550/arxiv.2402.15391]. Connect the self-play reinforcement learning of "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" to the neural simulation paradigm by asking whether self-play agents can be trained inside learned generative environments rather than fixed simulators [paper:10.48550/arxiv.1712.01815]. Follow-up work should instantiate a general self-play algorithm within a neural generative environment and assess whether the combination yields competent play [paper:10.48550/arxiv.1712.01815]. This would test whether the general reinforcement learning approach retains its strength when the environment itself is model-generated [paper:10.48550/arxiv.1712.01815]. Pursue integration across these threads by using the generative interactive environments of "Genie" as training grounds for the general self-play reinforcement learning of "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" [paper:10.48550/arxiv.2402.15391][paper:10.48550/arxiv.1712.01815]. A concrete next step is to train a general self-play agent entirely within a Genie-style generative environment and evaluate the resulting policy [paper:10.48550/arxiv.2402.15391][paper:10.48550/arxiv.1712.01815]. This direction would reveal whether generative environments can substitute for handcrafted simulators in general reinforcement learning [paper:10.48550/arxiv.1712.01815][paper:10.48550/arxiv.2402.15391]. Finally, address the evaluation gap implied by the single-domain demonstration in "DIFFUSION MODELS ARE REAL-TIME GAME ENGINES" by defining shared protocols for real-time, long-horizon neural game engines [paper:10.48550/arxiv.2408.14837]. Follow-up work should specify standardized interaction-length and quality criteria and apply them across neural engine implementations [paper:10.48550/arxiv.2408.14837]. Comparable protocols would make claims about real-time neural game engines verifiable across studies [paper:10.48550/arxiv.2408.14837].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1038/s41586-019-1724-z: Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (2020).
- paper:10.48550/arxiv.1712.01815: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (2017).
- paper:10.48550/arxiv.2301.04104: Mastering Diverse Domains through World Models (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2310.05036: AVALONBENCH: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2402.15391: Genie: Generative Interactive Environments (2024).
- paper:10.48550/arxiv.2408.14837: DIFFUSION MODELS ARE REAL-TIME GAME ENGINES (2024).
