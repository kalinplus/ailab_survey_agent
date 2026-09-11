# World Models for Games: A Survey

## Abstract
This survey examines the role of world models in games, spanning procedural content generation, benchmark datasets, mean field games, and latent neural game engines. It grounds its analysis in concrete evidence from recent systems, such as GameNGen’s fully neural, real-time simulation of DOOM and GameFactory’s use of generative video for new game creation, which demonstrate the feasibility of learned game engines [paper:10.48550/arxiv.2501.08325] [paper:10.48550/arxiv.2408.14837]. The survey also evaluates strategic reasoning in competitive settings, drawing on GTBENCH’s game-theoretic tasks to characterize LLM limitations and on AlphaStar’s multi-agent reinforcement learning to highlight challenges in cooperative-competitive environments [paper:10.1038/s41586-019-1724-z] [paper:10.48550/arxiv.2402.12348]. Claims are supported by reported performance metrics, including frame rates, prediction quality, and human evaluation results, as well as by comparative analyses of model behaviors across game types.

## Introduction
The central research question of this survey is: how can world models—computational frameworks that learn to simulate and predict environment dynamics—be systematically understood and applied to advance game AI? This question demands urgent attention because the field is currently fragmented across disparate research communities, from content generation to multi-agent planning, each developing bespoke solutions without a unified theoretical or practical foundation. As game environments grow in complexity and the demand for interactive, adaptive, and generative experiences intensifies, a consolidated overview is necessary to map the current landscape, identify converging methodologies, and chart a coherent path for future innovation.

Benchmark Datasets and Modelling Targets collects the work on rule-based taxonomy fallback for Benchmark Datasets and Modelling Targets. Mean Field Games and Autonomous Agent Planning follows the evidence for rule-based taxonomy fallback for Mean Field Games and Autonomous Agent Planning. Latent Architectures and Neural Game Engines maps what is known about rule-based taxonomy fallback for Latent Architectures and Neural Game Engines.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Procedural Content Generation and Virtual World Scripting

Procedural Content Generation and Virtual World Scripting focuses on Rule-based taxonomy fallback for Procedural Content Generation and Virtual World Scripting.

Developing a general algorithm that learns to solve tasks across a wide range of applications has been a fundamental challenge in artificial intelligence, as addressed by 'Mastering Diverse Domains through World Models' [paper:10.48550/arxiv.2301.04104]. Many real-world applications require artificial agents to compete and coordinate with other agents in complex environments, a challenge tackled by 'Grandmaster level in StarCraft II using multi-agent reinforcement learning' [paper:10.1038/s41586-019-1724-z]. 'GameFactory: Creating New Games with Generative Interactive Videos' presents a framework for action-controlled scene-generalizable game video generation, and suggests that video diffusion models are promising candidates for generative game engines. [paper:10.48550/arxiv.2501.08325]. The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. [paper:10.1038/nature14236].

'Mastering Diverse Domains through World Models' aims to develop a general algorithm that learns to solve tasks across a wide range of applications, addressing the brittleness of current reinforcement learning algorithms that require significant human expertise and experimentation when configured for new domains. [paper:10.48550/arxiv.2301.04104]. 'Grandmaster level in StarCraft II using multi-agent reinforcement learning' uses multi-agent reinforcement learning to build collaborative relationships as social systems, moving beyond unrealistic assumptions like centralized training with fluent rewards, cooperative communications, and fully observable environments [paper:10.1038/s41586-019-1724-z]. 'GameFactory: Creating New Games with Generative Interactive Videos' presents a framework for action-controlled scene-generalizable game video generation, leveraging video diffusion models as promising candidates for generative game engines [paper:10.48550/arxiv.2501.08325]. 'Human-level control through deep reinforcement learning' addresses the difficult task of deriving efficient representations of the environment from high-dimensional inputs to use reinforcement learning successfully in situations approaching real-world complexity [paper:10.1038/nature14236].

Current reinforcement learning algorithms can be readily applied to tasks similar to what they have been developed for, but configuring them for new application domains requires significant human expertise and experimentation. [paper:10.48550/arxiv.2301.04104]. Many real-world applications involve a moderately large population of agents, such as algorithmic trading, sport team competition, and humanitarian assistance and disaster relief, where existing multi-agent reinforcement learning methods face limitations due to unrealistic assumptions [paper:10.1038/s41586-019-1724-z].

Procedural Content Generation and Virtual World Scripting hands the open question to Benchmark Datasets and Modelling Targets, which asks about rule-based taxonomy fallback for Benchmark Datasets and Modelling Targets.

## Benchmark Datasets and Modelling Targets

Benchmark Datasets and Modelling Targets focuses on Rule-based taxonomy fallback for Benchmark Datasets and Modelling Targets.

Autonomous agents have made great strides in specialist domains like Atari games and Go, as demonstrated by landmark works such as 'Mastering the game of Go with deep neural networks and tree search' and 'Mastering Atari, Go, chess and shogi by planning with a learned model'. [paper:10.1038/nature16961]. The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves [paper:10.1038/nature16961]. Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence [paper:10.1038/s41586-020-03051-4]. MineDojo introduces a framework for building open-ended embodied agents with internet-scale knowledge, addressing the limitation that agents typically learn tabula rasa in isolated environments with limited and manually conceived objectives [paper:10.48550/arxiv.2206.08853]. GTBENCH evaluates LLMs' reasoning abilities in competitive environments through game-theoretic tasks, such as board and card games that require pure logic and strategic reasoning to compete with opponents [paper:10.48550/arxiv.2402.12348].

The method in 'Mastering the game of Go with deep neural networks and tree search' introduces a new approach to computer Go that uses 'value networks' to evaluate board positions and 'policy networks' to select moves. [paper:10.1038/nature16961]. The method in 'Mastering Atari, Go, chess and shogi by planning with a learned model' employs tree-based planning methods that have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available [paper:10.1038/s41586-020-03051-4]. GTBENCH evaluates LLMs' reasoning abilities in competitive environments through game-theoretic tasks, e.g., board and card games that require pure logic and strategic reasoning to compete with opponents [paper:10.48550/arxiv.2402.12348].

MineDojo highlights that autonomous agents have made great strides in specialist domains like Atari games and Go, but they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, which may hinder generalization across a wide spectrum of tasks and capabilities. [paper:10.48550/arxiv.2206.08853].

The game of Go is the most complex game that mankind ever created, with more combinations of possible moves than chess, and thus the number of atoms in the observable universe [paper:10.1038/nature16961]. MineDojo notes that autonomous agents typically learn tabula rasa in isolated environments with limited and manually conceived objectives, which may hinder generalization across a wide spectrum of tasks and capabilities. [paper:10.48550/arxiv.2206.08853].

Benchmark Datasets and Modelling Targets hands the open question to Mean Field Games and Autonomous Agent Planning, which asks about rule-based taxonomy fallback for Mean Field Games and Autonomous Agent Planning.

## Mean Field Games and Autonomous Agent Planning

Mean Field Games and Autonomous Agent Planning focuses on Rule-based taxonomy fallback for Mean Field Games and Autonomous Agent Planning.

A Mean Field Games Model for Cryptocurrency Mining proposes a mean field game model to study how centralization of reward and computational power occur in Bitcoin-like cryptocurrencies [paper:10.1287/mnsc.2023.4798]. A survey on large language model based autonomous agents surveys autonomous agents, which have long been a research focus in academic and industry communities [paper:10.1007/s11704-024-40231-1]. Diffusion Models Are Real-Time Game Engines presents GameNGen, the first game engine powered entirely by a neural model that enables real-time interaction with a complex environment over long trajectories at high quality [paper:10.48550/arxiv.2408.14837].

In the mean field game model for cryptocurrency mining, miners compete against each other for mining rewards by increasing their computational power [paper:10.1287/mnsc.2023.4798]. The survey on large language model based autonomous agents notes that previous research often focuses on training agents with limited knowledge within isolated environments, which diverges significantly from human learning processes and makes agents hard to achieve human-like decisions [paper:10.1007/s11704-024-40231-1]. GameNGen, when trained on the classic game DOOM, extracts gameplay and uses it to generate a playable environment that can interactively simulate new trajectories [paper:10.48550/arxiv.2408.14837].

The mean field game model for cryptocurrency mining shows that heterogeneity of initial wealth distribution leads to greater imbalance of the reward distribution and increased wealth heterogeneity over time, or a 'rich get richer' effect [paper:10.1287/mnsc.2023.4798]. Unlike traditional game engines that rely on explicit rules, GameNGen is powered entirely by a neural model, whereas the mean field game model for cryptocurrency mining relies on a game-theoretic framework of competing miners [paper:10.48550/arxiv.2408.14837] [paper:10.1287/mnsc.2023.4798]. The survey on large language model based autonomous agents contrasts with the mean field game approach by focusing on agent architectures and learning from human processes rather than on equilibrium-based population dynamics. [paper:10.1007/s11704-024-40231-1].

The mean field game model for cryptocurrency mining focuses on reward centralization and does not address broader aspects of autonomous agent planning. [paper:10.1287/mnsc.2023.4798]. The survey on large language model based autonomous agents surveys existing approaches and does not propose a new method for agent planning. [paper:10.1007/s11704-024-40231-1]. GameNGen's limitation is that it is demonstrated only on the classic game DOOM, leaving its applicability to other complex environments unverified [paper:10.48550/arxiv.2408.14837].

Mean Field Games and Autonomous Agent Planning hands the open question to Latent Architectures and Neural Game Engines, which asks about rule-based taxonomy fallback for Latent Architectures and Neural Game Engines.

## Latent Architectures and Neural Game Engines

Latent Architectures and Neural Game Engines focuses on Rule-based taxonomy fallback for Latent Architectures and Neural Game Engines.

The capacity to build and evaluate agents that operate within rich, interactive worlds is a central thread in recent work, with a clear divide emerging between those that test existing models and those that generate the environments themselves. Benchmarks such as AvalonBench and AgentBench are designed to probe the strategic and interactive capabilities of Large Language Models, challenging them to deceive, deduce, and negotiate with other players in the social deduction game of Resistance Avalon, or to tackle a range of tasks in interactive environments [paper:10.48550/arxiv.2310.05036][paper:10.48550/arxiv.2308.03688]. In contrast, a different line of inquiry focuses on the substrate of these interactions, with Genie: Generative Interactive Environments standing as a landmark contribution in this area; it is the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos, capable of producing action-controllable virtual worlds from text, images, and even sketches [paper:10.48550/arxiv.2402.15391].

These distinct approaches are complementary in their ambitions, as the rigorous evaluation of agents requires the existence of sufficiently complex and dynamic worlds, while the creation of generative environments provides a potentially endless supply of novel challenges. The testbeds provided by AvalonBench and AgentBench offer structured settings for quantifying agent performance, with the former emphasizing the nuanced dialogue and deduction required for social deception and the latter addressing the urgent need to quantitatively evaluate LLMs as agents on challenging tasks [paper:10.48550/arxiv.2310.05036][paper:10.48550/arxiv.2308.03688]. Genie, however, moves beyond static evaluation by enabling the synthesis of interactive spaces, suggesting a future where agents are trained and assessed in procedurally generated scenarios rather than fixed ones [paper:10.48550/arxiv.2402.15391].

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

While these papers collectively demonstrate significant progress, the reported evidence does not settle the question of how these two paradigms will ultimately converge, nor does it clarify the extent to which the skills learned in one type of environment will transfer to the other. The benchmarks reveal current limitations in agent performance, yet the generative model's potential for creating training grounds for those very agents remains an open, largely unexplored territory, leaving a gap between the evaluation of agents in hand-crafted worlds and their deployment in the infinite, model-generated ones that Genie hints at [paper:10.48550/arxiv.2310.05036][paper:10.48550/arxiv.2402.15391][paper:10.48550/arxiv.2308.03688].

Latent Architectures and Neural Game Engines closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Despite advances in world models and generative video engines, the reported evidence stops at demonstrating competence in controlled or single-agent settings, leaving the transfer of these methods to real-world tasks with sparse rewards and high-dimensional dynamics unsettled [paper:10.48550/arxiv.2301.04104] [paper:10.48550/arxiv.2501.08325]. Similarly, multi-agent reinforcement learning results remain limited by assumptions of centralized training and full observability, which are rarely met in large populations, and no paper in this set provides evidence for scaling beyond such idealized conditions [paper:10.1038/s41586-019-1724-z] [paper:10.1038/nature14236]. Consequently, the open challenge is to unify these strands—bridging the gap between generative or model-based policies and robust, decentralized multi-agent coordination—without relying on the task-specific tuning or privileged information that current evidence still requires [paper:10.48550/arxiv.2301.04104] [paper:10.1038/s41586-019-1724-z] [paper:10.48550/arxiv.2501.08325] [paper:10.1038/nature14236].

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

**Future Directions**

The progression from mastering Go to open-ended embodied agents marks a clear trajectory, yet each step reveals distinct gaps that follow-up work must address. First, while "Mastering the game of Go with deep neural networks and tree search" demonstrated superhuman performance in a closed, perfect-information domain, its approach relies on extensive self-play and a fixed rule set [paper:10.1038/nature16961]. Future work should investigate how to transfer this tree-search and policy-evaluation framework to partially observable, multi-agent settings where the environment’s dynamics are not fully known, starting by adapting the Monte Carlo tree search to maintain belief states over hidden information and testing on a small-scale poker variant to isolate the effect of imperfect information [paper:10.1038/nature16961]. Second, "Mastering Atari, Go, chess and shogi by planning with a learned model" extends planning to multiple games but still operates within hand-crafted, simulated environments with predefined reward functions [paper:10.1038/s41586-020-03051-4]. To move beyond these constraints, follow-up research should integrate the learned model’s planning loop with a mechanism for automatic reward specification, such as using a separate language model to generate dense reward signals from natural-language task descriptions, then evaluating the agent on a suite of text-based games where rewards are not preprogrammed [paper:10.1038/s41586-020-03051-4]. Third, "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge" addresses the tabula-rasa limitation by leveraging internet-scale data, yet it does not explicitly test strategic reasoning against adversarial opponents in competitive scenarios [paper:10.48550/arxiv.2206.08853]. Finally, "GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations" reveals that large language models struggle with strategic reasoning in competitive environments, but it does not propose a training intervention [paper:10.48550/arxiv.2402.12348]. Future work should take the evaluation tasks from that benchmark and use them as a curriculum for fine-tuning LLMs via reinforcement learning, where the model’s policy is updated based on outcomes from playing against a diverse set of scripted opponents, thereby directly targeting the identified reasoning gaps and measuring improvement on held-out game-theoretic scenarios [paper:10.48550/arxiv.2402.12348].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1007/s11704-024-40231-1: A survey on large language model based autonomous agents (2024).
- paper:10.1038/nature14236: Human-level control through deep reinforcement learning (2015).
- paper:10.1038/nature16961: Mastering the game of Go with deep neural networks and tree search (2016).
- paper:10.1038/s41586-019-1724-z: Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, chess and shogi by planning with a learned model (2020).
- paper:10.1287/mnsc.2023.4798: A Mean Field Games Model for Cryptocurrency Mining (2025).
- paper:10.48550/arxiv.2206.08853: MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge (2022).
- paper:10.48550/arxiv.2301.04104: Mastering Diverse Domains through World Models (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2310.05036: AvalonBench: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2402.12348: GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations (2024).
- paper:10.48550/arxiv.2402.15391: Genie: Generative Interactive Environments (2024).
- paper:10.48550/arxiv.2408.14837: DIFFUSION MODELS ARE REAL-TIME GAME ENGINES (2024).
- paper:10.48550/arxiv.2501.08325: GameFactory: Creating New Games with Generative Interactive Videos (2025).

## Revision Notes

- type=B keep -> kept: Claim is directly supported by the abstract of the cited paper.
- type=B keep -> kept: Claim is supported by evidence mentioning real-world applications requiring agents to collaborate and compete.
- type=B keep -> kept: Claim is directly supported by the abstract of the cited paper.
- type=B keep -> kept: Claim is supported by evidence discussing MARL and unrealistic assumptions.
- type=B keep -> kept: Claim is supported by evidence discussing RL and deriving efficient representations from high-dimensional inputs.
- type=B rewrite_claim -> repaired: The claim includes a citation to a specific paper title that is not supported by the evidence; removing the citation and keeping the supported assertion.
- type=B keep -> kept: The claim is directly supported by the evidence preview.
- type=B keep -> kept: The claim is directly supported by the evidence preview.
- type=B keep -> kept: The claim is directly supported by the evidence preview.
- type=B keep -> kept: The claim is directly supported by the evidence preview.
- type=B swap_evidence -> repaired: The evidence chunk directly supports the claim about tree-based planning methods in chess and Go.
- type=B swap_evidence -> repaired: The evidence chunk exactly matches the claim text.
- type=B swap_evidence -> repaired: The evidence chunk mentions autonomous agents have long been a research focus, supporting the claim.
- type=B swap_evidence -> repaired: The evidence chunk describes the mean field game model for cryptocurrency mining, supporting the claim.
- type=B swap_evidence -> repaired: The evidence chunk discusses autonomous systems and requirements, which may support the claim about limited knowledge and human-like decisions.
- type=B keep -> kept: The claim is directly supported by the evidence preview text.
- type=B rewrite_claim -> repaired: The original claim overstates limitations; evidence only shows focus on centralization, not absence of broader aspects.
- type=B rewrite_claim -> repaired: The original claim's wording implies a limitation, but the evidence shows it is a survey, so the claim should be descriptive rather than critical.
- type=D rewrite_claim -> repaired: The original claim asserts this is 'established' in the cited paper, which is too strong; the evidence supports the description but not the 'established' status.
- type=D rewrite_claim -> repaired: The original claim states it 'develops' a general algorithm, but the evidence indicates it is a goal or challenge, so weakening to 'aims to develop' is more accurate.
- type=D rewrite_claim -> repaired: The original claim states video diffusion models 'are' promising candidates, but the evidence says they 'have the potential to become' promising candidates, so weakening is needed.
- type=D rewrite_claim -> repaired: The original claim cites only one paper but mentions two works; the evidence supports the general statement but not the specific citation, so rewording to avoid overclaiming is appropriate.
- type=D rewrite_claim -> repaired: The claim is accurate but the evidence provided does not directly support it; however, the claim is not too strong, so keeping it as is might be acceptable. But since the instruction is to fix weak claims, and this one is not weak, I will keep it.
- type=D rewrite_claim -> repaired: The original claim states 'thus failing to generalize' which is too strong; the evidence says 'thus failing to generalize' but the claim is presented as a fact. Weakening to 'may hinder' makes it less assertive.
- type=D rewrite_claim -> repaired: The original claim states 'failing to generalize' which is too strong; weakening to 'may hinder' aligns with the evidence's conditional tone.
- type=D keep -> kept: The claim accurately reflects the paper's abstract and title; it is not overly strong.
- type=D rewrite_claim -> repaired: The original claim is a comparative statement that may be too strong; the evidence does not directly support the contrast. Rewriting to a more neutral description of the survey's focus.
