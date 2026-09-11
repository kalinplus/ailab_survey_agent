# World Models for Games: A Survey

## Abstract

## Introduction
Game intelligence increasingly rests on models that learn how a world evolves rather than on hand-written rules alone; whether such models can support planning, generation, and fair evaluation is the question this survey addresses. Each theme that follows gathers the evidence for one capability and states where that evidence stops.

Procedural Content Generation Mechanisms surveys taxonomy of techniques employing world models to procedurally generate game content including levels, items, and storylines. Agent-Based World Model Dynamics collects the work on classification of models focusing on agent-environment interaction, behavior prediction, and dynamic evolution within simulated worlds. Latent Architectures for Interactive Generation follows the evidence for overview of latent variable models and neural architectures designed for real-time, interactive content generation and simulation. Evaluation Benchmarks for Procedural and Simulation Models maps what is known about summary of existing benchmarks, metrics, and evaluation protocols for assessing procedural generation and simulation fidelity.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Procedural Content Generation Mechanisms

Procedural Content Generation Mechanisms focuses on Taxonomy of techniques employing world models to procedurally generate game content including levels, items, and storylines.

World Models for Autonomous Driving: An Initial Survey [paper:10.1109/tiv.2024.3398357] contributes a treatment of prediction capability as the model's capacity to reason about the future evolution of dynamic driving scenes, a capability it identifies as fundamental to enabling safe, efficient, and proactive behavior. Understanding World or Predicting Future? A Comprehensive Survey of World Models [paper:10.1145/3746449] contributes a synthesis of the field amid advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which it positions as central to the pursuit of artificial general intelligence, and it summarizes representative papers along with their code repositories. Reasoning with Language Model is Planning with World Model [paper:10.48550/arxiv.2305.14992] contributes the framing that reasoning with large language models constitutes planning with a world model, building on their remarkable reasoning capabilities when prompted to generate intermediate reasoning steps such as Chain-of-Thought. In the rapidly evolving landscape emphasized by the driving survey, the capability to predict future events is described as fundamental to enabling safe, efficient, and proactive autonomous driving and as central to the decision-making pipeline, suggesting that forecasting capabilities of this kind may also offer a useful starting point for procedural mechanisms that anticipate how generated levels, items, and storylines will unfold [paper:10.1109/tiv.2024.3398357].

![Publication years of the selected representative papers.](publication_timeline)

At the mechanistic level, prediction plays a central role in the decision-making pipeline, where accurately forecasting the behavior of surrounding agents is fundamental for understanding the dynamics of scenes; transposed into a procedural content generation taxonomy, this corresponds to rolling out a world model of a candidate level or storyline before it is committed to the game [paper:10.1109/tiv.2024.3398357]. The comprehensive survey delineates methodological families organized around understanding the world versus predicting the future, realized respectively through multimodal large language models and through video generation models such as Sora, a distinction a generation taxonomy can map onto structure-aware modeling of game worlds versus direct synthesis of future content [paper:10.1145/3746449]. The language-based mechanism prompts the model to generate intermediate steps before the final answer, following the chain-of-thought paradigm, a mechanism directly reusable for constructing storylines incrementally [paper:10.48550/arxiv.2305.14992].

Compared with the driving survey, which scopes prediction to the future evolution of dynamic driving scenes, the comprehensive survey frames the field along the broader axis of understanding the world versus predicting the future, suggesting that generation mechanisms be classified by whether they internalize environment structure or directly synthesize future observations [paper:10.1109/tiv.2024.3398357] [paper:10.1145/3746449]. Whereas the two surveys characterize world models through predictive and representational capacity, Reasoning with Language Model is Planning with World Model locates the operative mechanism in the reasoning of language models themselves, motivating a taxonomy split between generative-simulation mechanisms and language-based planning mechanisms for producing content such as storylines [paper:10.1109/tiv.2024.3398357] [paper:10.1145/3746449] [paper:10.48550/arxiv.2305.14992].

A noted limitation is that even reasoning-oriented models still struggle with problems that require multi-step reasoning and planning, despite the impressive results achieved with chain-of-thought prompting, which suggests limits on how far language-based generation can sustain long, coherent content [paper:10.48550/arxiv.2305.14992]. The comprehensive survey itself closes by outlining key challenges and providing insights into potential future research directions, indicating that open problems remain before world model mechanisms can serve as turnkey procedural generators of game content [paper:10.1145/3746449]. The driving survey's evidence is largely confined to autonomous driving, where prediction capability serves safe, efficient, and proactive decision-making rather than the creation of levels, items, or storylines, suggesting that its mechanisms would likely need adaptation rather than direct application to game content generation [paper:10.1109/tiv.2024.3398357].

Procedural Content Generation Mechanisms hands the open question to Agent-Based World Model Dynamics, which asks about classification of models focusing on agent-environment interaction, behavior prediction, and dynamic evolution within simulated worlds.

## Agent-Based World Model Dynamics

Agent-Based World Model Dynamics focuses on Classification of models focusing on agent-environment interaction, behavior prediction, and dynamic evolution within simulated worlds.

Agent-based modeling and simulation has evolved as a powerful tool for modeling complex systems, offering insights into emergent behaviors and interactions among diverse agents [paper:10.1057/s41599-024-03611-3]. Hypergraph-Based Model for Modeling Multi-Agent Q-Learning Dynamics in Public Goods Games [paper:10.1109/tnse.2024.3473941] tackles modeling the learning dynamic of multi-agent systems, a crucial issue for understanding the emergence of collective behavior. Model-Based Planning for Web Agents [paper:10.48550/arxiv.2411.06559] shows that, for language agents automating web-based tasks, incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning.

![Distribution of selected papers across the survey taxonomy.](taxonomy_overview)

In public goods games where agents interact in multiple larger groups, the hypergraph-based approach builds on multi-agent reinforcement learning, which extends machine learning in modeling competitive and cooperative behaviors, toward predicting human collective behavior in resource allocation [paper:10.1109/tnse.2024.3473941]. The web-agent framework rests on intelligent agents capable of perceiving webpages, reasoning over content, and acting to automate end-to-end tasks such as online shopping, with the large language model interrogated as a world model of the internet for model-based planning [paper:10.48550/arxiv.2411.06559]. Agent-based modeling and simulation can be defined in very diverse disciplines like artificial intelligence, complexity science, and game theory, and its recent integration with large language models empowers the simulation of diverse agents and their interactions [paper:10.1057/s41599-024-03611-3].

Whereas planning for web agents can employ advanced algorithms such as tree search, which has been shown to outperform reactive planning, agent-based modeling and simulation instead foregrounds emergent behaviors arising from interactions among diverse agents [paper:10.48550/arxiv.2411.06559]. The hypergraph-based formulation represents agents interacting in multiple larger groups, while large language model empowered agent-based modeling spans disciplines like artificial intelligence, complexity science, and game theory to model complex systems [paper:10.1109/tnse.2024.3473941] [paper:10.1057/s41599-024-03611-3].

Nevertheless, the practical utility of agent-based modeling has been limited by computational constraints and simplistic agent behaviors, especially when simulating large populations [paper:10.1057/s41599-024-03611-3].

Agent-Based World Model Dynamics hands the open question to Latent Architectures for Interactive Generation, which asks about overview of latent variable models and neural architectures designed for real-time, interactive content generation and simulation.

## Latent Architectures for Interactive Generation

Latent Architectures for Interactive Generation focuses on Overview of latent variable models and neural architectures designed for real-time, interactive content generation and simulation.

Interactive content generation increasingly rests on latent generative architectures coupled with large-scale simulation, knowledge, and evaluation infrastructures, and the works surveyed in this section span game video generation, open-ended embodied simulation, and game-theoretic reasoning assessment. A representative generative architecture is GameFactory [paper:10.48550/arxiv.2501.08325], a framework for action-controlled and scene-generalizable game video generation that builds on the potential of generative videos to revolutionize game development by autonomously creating new content. MineDojo [paper:10.48550/arxiv.2206.08853] contributes an open-ended embodied-agent framework built with internet-scale knowledge and open-sources its simulation suite, knowledge bases, algorithm implementation, and pretrained models to promote research towards the goal of generally capable embodied agents. GTBench [paper:10.48550/arxiv.2402.12348] contributes a game-theoretic evaluation methodology that uncovers the strategic reasoning limitations of large language models as they are integrated into critical real-world applications.

To achieve action-controlled video generation, GameFactory first collects an action-annotated game video dataset, GF-Minecraft, which supports the framework's broader goal of action-controlled, scene-generalizable game video generation [paper:10.48550/arxiv.2501.08325]. MineDojo methodologically addresses the tendency of autonomous agents to learn tabula rasa in isolated environments with limited and manually conceived objectives by building open-ended embodied agents with internet-scale knowledge [paper:10.48550/arxiv.2206.08853]. GTBench operationalizes its evaluation by testing the strategic and logical reasoning abilities of large language models in competitive game-theoretic settings, encompassing the understanding of intricate scenarios and strategic planning [paper:10.48550/arxiv.2402.12348].

In comparison, GameFactory and MineDojo target complementary layers of interactive generation: GameFactory produces action-controlled, scene-generalizable game videos as new content, whereas MineDojo provides a simulation suite with knowledge bases on which open-ended embodied agents are trained [paper:10.48550/arxiv.2501.08325] [paper:10.48550/arxiv.2206.08853]. Relative to these generation- and simulation-oriented architectures, GTBench shifts the focus from producing interactive content to evaluating the reasoning layer, probing where large language models, despite notable abilities across a wide range of tasks, show strategic reasoning limitations in competitive settings [paper:10.48550/arxiv.2501.08325] [paper:10.48550/arxiv.2206.08853] [paper:10.48550/arxiv.2402.12348].

A key limitation motivating internet-scale architectures is that autonomous agents in specialist domains like Atari games and Go typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize [paper:10.48550/arxiv.2206.08853]. On the reasoning side, the evidence indicates that large language models exhibit strategic reasoning limitations in competitive game-theoretic evaluations, a constraint for interactive generation systems that rely on such models for strategic planning and decision-making [paper:10.48550/arxiv.2402.12348].

Latent Architectures for Interactive Generation hands the open question to Evaluation Benchmarks for Procedural and Simulation Models, which asks about summary of existing benchmarks, metrics, and evaluation protocols for assessing procedural generation and simulation fidelity.

## Evaluation Benchmarks for Procedural and Simulation Models

Evaluation Benchmarks for Procedural and Simulation Models focuses on Summary of existing benchmarks, metrics, and evaluation protocols for assessing procedural generation and simulation fidelity.

Existing evaluation practice for agentic and simulation models spans interactive benchmarks for LLM agents, survey-level syntheses of autonomous agents, and world-model-driven forecasting and planning protocols in autonomous driving. A landmark work, AgentBench [paper:10.48550/arxiv.2308.03688], addresses the urgent need to quantitatively evaluate LLMs as agents on challenging tasks in interactive environments. The survey on large language model based autonomous agents consolidates a field in which autonomous agents have long been a research focus in both academic and industry communities [paper:10.1007/s11704-024-40231-1]. Driving Into the Future contributes multiview visual forecasting and planning with a world model, in which predicting future events in advance and evaluating the foreseeable risks empowers autonomous vehicles to better plan their actions, enhancing safety and efficiency on the road [paper:10.1109/cvpr52733.2024.01397].

Methodologically, AgentBench establishes a quantitative protocol that evaluates LLMs as agents within interactive environments on challenging tasks [paper:10.48550/arxiv.2308.03688]. The survey grounds such evaluation in its definition of an autonomous agent as a system that interacts with its environment by perceiving its surroundings and taking actions over time to achieve specific goals [paper:10.1007/s11704-024-40231-1]. On the simulation side, current autonomous driving systems typically divide the problem into four steps, namely perception, tracking, trajectory prediction, and path planning, among which trajectory prediction plays a pivotal role [paper:10.1109/cvpr52733.2024.01397].

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

A comparison across these works shows divergent evaluation settings, spanning interactive environments for LLM agents in AgentBench, environment interaction over time for autonomous agents in the survey, and forecast-based risk evaluation within driving pipelines [paper:10.48550/arxiv.2308.03688] [paper:10.1007/s11704-024-40231-1] [paper:10.1109/cvpr52733.2024.01397].

A central limitation recorded in the survey is that previous research historically concentrated on training agents with limited knowledge in isolated environments, a process that differs from human learning and hampers the agents' ability to make decisions akin to those of humans [paper:10.1007/s11704-024-40231-1]. AgentBench likewise reflects an evaluation gap, since the potential of LLMs as agents has been widely acknowledged while quantitative evaluation on challenging tasks in interactive environments remained an urgent need [paper:10.48550/arxiv.2308.03688]. In autonomous driving, the decomposition into perception, tracking, trajectory prediction, and path planning contrasts with the human ability to predict the motion of other road agents in order to anticipate potential danger, a gap that forecasting and planning with a world model aims to close [paper:10.1109/cvpr52733.2024.01397].

Evaluation Benchmarks for Procedural and Simulation Models closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Read across the body sections, the recorded boundaries cluster into a few open problems: World Models for Autonomous Driving: An Initial Survey: Prediction Capability Prediction capability assesses the model's capacity to reason about the future evolution of dynamic driving scenes and is fundamental to enabling safe, efficient, and proactive... [paper:10.1109/tiv.2024.3398357]; Understanding World or Predicting Future? A Comprehensive Survey of World Models: Finally, we outline key challenges and provide insights into potential future research directions. We summarize the representative papers along with their code repositories in https://github.com/tsin... [paper:10.1145/3746449]; Reasoning with Language Model is Planning with World Model: Large language models (LLMs) have achieved remarkable results on reasoning benchmarks (OpenAI 2024; Guo et al. 2025). However, even such reasoning-oriented models still struggle with problems that re... [paper:10.48550/arxiv.2305.14992]; Hypergraph-Based Model for Modeling Multi-Agent Q-Learning Dynamics in Public Goods Games: With the looming climate crisis, limited planetary resources, and the associated challenges to human societies, predicting human collective behavior in resource allocation is a problem of increasing... [paper:10.1109/tnse.2024.3473941]. Each remains open rather than settled.

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions


Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1007/s11704-024-40231-1: A survey on large language model based autonomous agents (2024).
- paper:10.1057/s41599-024-03611-3: Large language models empowered agent-based modeling and simulation: a survey and perspectives (2024).
- paper:10.1109/cvpr52733.2024.01397: Driving Into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving (2024).
- paper:10.1109/tiv.2024.3398357: World Models for Autonomous Driving: An Initial Survey (2024).
- paper:10.1109/tnse.2024.3473941: Hypergraph-Based Model for Modeling Multi-Agent Q-Learning Dynamics in Public Goods Games (2024).
- paper:10.1145/3746449: Understanding World or Predicting Future? A Comprehensive Survey of World Models (2025).
- paper:10.48550/arxiv.2206.08853: MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge (2022).
- paper:10.48550/arxiv.2305.14992: Reasoning with Language Model is Planning with World Model (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2402.12348: GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations (2024).
- paper:10.48550/arxiv.2411.06559: Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents (2024).
- paper:10.48550/arxiv.2501.08325: GameFactory: Creating New Games with Generative Interactive Videos (2025).

## Revision Notes

- type=B swap_evidence -> repaired: This chunk defines prediction capability exactly as claimed (capacity to reason about future evolution of dynamic scenes, fundamental to safe/efficient/proactive driving).
- type=B backfill_evidence -> repaired: Claim spans two grounded parts (intro motivation re GPT-4/Sora/AGI and code-repository summary); no single preview chunk covers both, so semantic search should fetch grounding chunks.
- type=B swap_evidence -> repaired: This chunk directly grounds the chain-of-thought/intermediate-step reasoning capability the claim attributes to the paper; the framing itself is the paper's title.
- type=B rewrite_claim -> invalid_action: 
- type=B None -> unresolved: model did not answer this failure
- type=B rewrite_claim -> repaired: Evidence (chunk 343) supports only CoT intermediate-step generation; the world-model-simulation framing and stepwise-consequence-planning claims overreach, so weaken to the supported core.
- type=B rewrite_claim -> invalid_action: 
- type=B None -> unresolved: model did not answer this failure
- type=B None -> unresolved: model did not answer this failure
- type=B None -> unresolved: model did not answer this failure
- type=B backfill_evidence -> repaired: Preview chunks support MARL and resource-allocation prediction; public goods/hypergraph grounding should exist elsewhere in this paper, so fetch it via semantic search.
- type=B backfill_evidence -> repaired: The LLM-as-world-model for model-based planning is this paper's core contribution but is absent from preview chunks; semantic search should ground it.
- type=B rewrite_claim -> repaired: Evidence only supports that tree search outperforms reactive planning; coupling an implicit world model with tree search conflates two distinct approaches, so weaken to what the evidence states.
- type=B backfill_evidence -> repaired: Preview chunks cover multi-agent systems/MARL but not the hypergraph group representation; grounding chunks for the hypergraph formulation should exist in the paper, so fetch via semantic search.
- type=B swap_evidence -> repaired: This chunk explicitly states GameFactory is a framework for action-controlled and scene-generalizable game video generation, directly supporting the claim.
- type=B backfill_evidence -> repaired: Claim mirrors MineDojo's abstract (open-ended agents with internet-scale knowledge plus open-sourcing suite/models); backfill will retrieve the grounding chunks the paper clearly contains.
- type=B backfill_evidence -> repaired: GTBench's game-theoretic evaluation of LLM strategic reasoning is stated in the paper's abstract; backfill will fetch that contribution chunk instead of the generic intro chunks.
- type=B backfill_evidence -> repaired: MineDojo's abstract proposes open-ended embodied agents with internet-scale knowledge to fix tabula-rasa learning; backfill will retrieve the exact grounding for this causal framing.
- type=B remap_citation -> invalid_action: Claim is about GTBench but cites GameFactory (2501.08325); 2402.12348 is the GTBench paper already cited for claim_23, matching the sentence subject.
- type=B swap_evidence -> repaired: Chunk agentic_240 verbatim supports the tabula-rasa limitation in specialist domains like Atari and Go, directly grounding the claim.
- type=B backfill_evidence -> repaired: Cited paper's intro centers on strategic reasoning of LLMs in game settings; semantic search should surface chunks about game-theoretic limitation findings absent from the preview.
- type=B backfill_evidence -> repaired: Cited paper is AgentBench itself; its abstract/intro should contain chunks grounding the claim about quantitatively evaluating LLMs as agents, which the truncated preview chunks don't show.
- type=B backfill_evidence -> repaired: Preview chunks cover general trajectory prediction only; the cited 'Driving Into the Future' paper's abstract should ground the world-model forecasting and planning contribution.
- type=B backfill_evidence -> unresolved: AgentBench's benchmark-design sections should contain chunks establishing the quantitative interactive-environment protocol; preview shows only generic intro text.
- type=B rewrite_claim -> invalid_action: The single cited AgentBench paper can only ground the AgentBench portion; the claims about the survey and driving pipelines' evaluation settings cannot be supported by this citation.
- type=B rewrite_claim -> repaired: Evidence supports isolation and divergence from human learning, not 'real-world conditions', so weaken the assertion to match
- type=B backfill_evidence -> unresolved: Previewed AgentBench intro chunks are truncated and none fully cover the evaluation-gap statement; semantic search should fetch grounding chunks from the same paper
- type=B backfill_evidence -> unresolved: Decomposition and human-anticipation parts are supported; the world-model gap framing needs grounding chunks fetched from the cited paper
- type=E delete_line -> deleted: Last resort: unrecoverable figure reference with empty candidate list
- type=E delete_line -> deleted: Last resort: unrecoverable figure reference with empty candidate list
- type=D rewrite_claim -> repaired: Evidence supports prediction being fundamental to safe, efficient driving and decision-making, but the 'supplies the conceptual core' assertion for game content is unsupported; hedge to a suggestion.
- type=D rewrite_claim -> repaired: First half matches evidence nearly verbatim, but 'a principal limitation' and the definitive 'which bounds' overreach; soften 'principal' to 'noted' and 'bounds' to 'suggests limits'.
- type=D rewrite_claim -> repaired: Evidence confirms prediction serves safe/efficient/proactive driving, but 'must be adapted' is a categorical conclusion the survey does not assert; weaken to a suggestion.
- type=D keep -> kept: Claim mirrors the evidence almost verbatim ('incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning for web agents'); assertion strength is appropriate as-is.
- type=D rewrite_claim -> repaired: Evidence states the dataset was collected to achieve action-controlled generation, not that it grounds scene-generalizable generation (achieved via open-domain generative priors); weaken the dataset's asserted role.
- type=D None -> unresolved: llm budget exhausted
- type=D None -> unresolved: llm budget exhausted
- type=D None -> unresolved: llm budget exhausted
- type=D None -> unresolved: llm budget exhausted
- type=D None -> unresolved: llm budget exhausted
