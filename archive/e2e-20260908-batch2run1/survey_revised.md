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

World Models for Autonomous Driving: An Initial Survey [paper:10.1109/tiv.2024.3398357] contributes a treatment of prediction capability as the model's capacity to reason about the future evolution of dynamic driving scenes, a capability it identifies as fundamental to enabling safe, efficient, and proactive behavior. Understanding World or Predicting Future? A Comprehensive Survey of World Models contributes a comprehensive review of the world model literature, motivated by advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which it positions as central to the pursuit of artificial general intelligence [paper:10.1145/3746449]. Reasoning with Language Model is Planning with World Model [paper:10.48550/arxiv.2305.14992] contributes the framing that reasoning with large language models constitutes planning with a world model, building on their remarkable reasoning capabilities when prompted to generate intermediate reasoning steps such as Chain-of-Thought. The driving survey frames prediction capability as fundamental to enabling safe, efficient, and proactive autonomous driving and as central to its decision-making pipeline, suggesting that forecasting future events and their implications is broadly valuable; whether this forecasting capability can supply an analogous core for procedural mechanisms that anticipate how generated levels, items, and storylines will unfold remains a conceptual transfer rather than an established result [paper:10.1109/tiv.2024.3398357].

![Publication years of the selected representative papers.](publication_timeline)

At the mechanistic level, prediction plays a central role in the decision-making pipeline, where accurately forecasting the behavior of surrounding agents is fundamental for understanding the dynamics of scenes; transposed into a procedural content generation taxonomy, this corresponds to rolling out a world model of a candidate level or storyline before it is committed to the game [paper:10.1109/tiv.2024.3398357]. The comprehensive survey frames world models as computational frameworks serving two interconnected purposes: building implicit internal representations of the environment and predicting future states [paper:10.1145/3746449]. The language-based mechanism prompts the model to generate intermediate steps before the final answer through chain-of-thought reasoning, and the paper's framing of language-model reasoning as planning with a world model suggests that such simulated multi-step reasoning could be adapted for incrementally constructing content such as storylines [paper:10.48550/arxiv.2305.14992].

Whereas the driving survey scopes prediction to the future evolution of dynamic driving scenes, requiring models to go beyond momentary perception, this survey instead suggests classifying generation mechanisms by whether they internalize environment structure or directly synthesize future observations [paper:10.1109/tiv.2024.3398357]. Whereas the two surveys characterize world models through predictive and representational capacity, Reasoning with Language Model is Planning with World Model locates the operative mechanism in the reasoning of language models themselves, motivating a taxonomy split between generative-simulation mechanisms and language-based planning mechanisms for producing content such as storylines [paper:10.1109/tiv.2024.3398357] [paper:10.1145/3746449] [paper:10.48550/arxiv.2305.14992].

A noted limitation is that even reasoning-oriented models still struggle with problems that require multi-step reasoning and planning, despite the impressive results achieved with chain-of-thought prompting, which suggests limits on how far language-based generation can sustain long, coherent content [paper:10.48550/arxiv.2305.14992]. The comprehensive survey itself closes by outlining key challenges and providing insights into potential future research directions, indicating that open problems remain before world model mechanisms can serve as turnkey procedural generators of game content [paper:10.1145/3746449]. The driving survey's evidence is largely confined to autonomous driving, where prediction capability serves safe, efficient, and proactive decision-making rather than the creation of levels, items, or storylines, suggesting that its mechanisms would likely need adaptation rather than direct application to game content generation [paper:10.1109/tiv.2024.3398357].

Procedural Content Generation Mechanisms hands the open question to Agent-Based World Model Dynamics, which asks about classification of models focusing on agent-environment interaction, behavior prediction, and dynamic evolution within simulated worlds.

## Agent-Based World Model Dynamics

Agent-Based World Model Dynamics focuses on Classification of models focusing on agent-environment interaction, behavior prediction, and dynamic evolution within simulated worlds.

Agent-based modeling and simulation has evolved as a powerful tool for modeling complex systems, offering insights into emergent behaviors and interactions among diverse agents [paper:10.1057/s41599-024-03611-3]. Hypergraph-Based Model for Modeling Multi-Agent Q-Learning Dynamics in Public Goods Games [paper:10.1109/tnse.2024.3473941] tackles modeling the learning dynamic of multi-agent systems, a crucial issue for understanding the emergence of collective behavior. Model-Based Planning for Web Agents [paper:10.48550/arxiv.2411.06559] shows that, for language agents automating web-based tasks, incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning.

Multi-agent reinforcement learning extends traditional reinforcement learning frameworks and advances machine learning in modeling competitive and cooperative behaviors, while predicting human collective behavior in resource allocation calls for models of human economic interaction [paper:10.1109/tnse.2024.3473941]. The web-agent framework rests on intelligent agents capable of perceiving webpages, reasoning over content, and acting to automate end-to-end tasks such as online shopping, with the large language model interrogated as a world model of the internet for model-based planning [paper:10.48550/arxiv.2411.06559]. Agent-based modeling and simulation can be defined in very diverse disciplines like artificial intelligence, complexity science, and game theory, and its recent integration with large language models empowers the simulation of diverse agents and their interactions [paper:10.1057/s41599-024-03611-3].

Whereas planning for web agents can employ advanced algorithms such as tree search, which has been shown to outperform reactive planning, agent-based modeling and simulation instead foregrounds emergent behaviors arising from interactions among diverse agents [paper:10.48550/arxiv.2411.06559]. The hypergraph-based formulation captures settings in which agents interact in larger groups, and large language model empowered agent-based modeling is being explored to model complex systems such as human collective behavior in resource allocation, building on multi-agent systems research across areas like artificial intelligence and game theory [paper:10.1109/tnse.2024.3473941].

Nevertheless, the practical utility of agent-based modeling has been limited by computational constraints and simplistic agent behaviors, especially when simulating large populations [paper:10.1057/s41599-024-03611-3].

Agent-Based World Model Dynamics hands the open question to Latent Architectures for Interactive Generation, which asks about overview of latent variable models and neural architectures designed for real-time, interactive content generation and simulation.

## Latent Architectures for Interactive Generation

Latent Architectures for Interactive Generation focuses on Overview of latent variable models and neural architectures designed for real-time, interactive content generation and simulation.

Interactive content generation increasingly rests on latent generative architectures coupled with large-scale simulation, knowledge, and evaluation infrastructures, and the works surveyed in this section span game video generation, open-ended embodied simulation, and game-theoretic reasoning assessment. A representative generative architecture is GameFactory [paper:10.48550/arxiv.2501.08325], a framework for action-controlled and scene-generalizable game video generation that builds on the potential of generative videos to revolutionize game development by autonomously creating new content. MineDojo [paper:10.48550/arxiv.2206.08853] contributes an open-ended embodied-agent framework built with internet-scale knowledge and open-sources its simulation suite, knowledge bases, algorithm implementation, and pretrained models to promote research towards the goal of generally capable embodied agents. GTBench [paper:10.48550/arxiv.2402.12348] contributes a game-theoretic evaluation methodology that uncovers the strategic reasoning limitations of large language models as they are integrated into critical real-world applications.

To achieve action-controlled video generation, GameFactory firstly collects an action-annotated game video dataset, GF-Minecraft, within a framework designed for action-controlled and scene-generalizable game video generation [paper:10.48550/arxiv.2501.08325]. MineDojo methodologically addresses the tendency of autonomous agents to learn tabula rasa in isolated environments with limited and manually conceived objectives by building open-ended embodied agents with internet-scale knowledge [paper:10.48550/arxiv.2206.08853]. GTBench operationalizes its evaluation by testing the strategic reasoning abilities of large language models in competitive game-theoretic settings, where such reasoning involves understanding intricate scenarios and strategic planning [paper:10.48550/arxiv.2402.12348].

In comparison, GameFactory and MineDojo serve complementary purposes: GameFactory produces action-controlled, scene-generalizable game videos as new content, whereas MineDojo provides a simulation suite with knowledge bases for developing open-ended embodied agents [paper:10.48550/arxiv.2501.08325]. Relative to these generation- and simulation-oriented architectures, GTBench shifts the focus from producing interactive content to evaluating the reasoning layer, probing where large language models, despite notable abilities across a wide range of tasks, show strategic reasoning limitations in competitive settings [paper:10.48550/arxiv.2501.08325] [paper:10.48550/arxiv.2206.08853] [paper:10.48550/arxiv.2402.12348].

A key limitation motivating internet-scale architectures is that autonomous agents in specialist domains like Atari games and Go typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize [paper:10.48550/arxiv.2206.08853]. On the reasoning side, the evidence indicates that large language models exhibit strategic reasoning limitations in competitive game-theoretic evaluations, a constraint for interactive generation systems that rely on such models for strategic planning and decision-making [paper:10.48550/arxiv.2402.12348].

Latent Architectures for Interactive Generation hands the open question to Evaluation Benchmarks for Procedural and Simulation Models, which asks about summary of existing benchmarks, metrics, and evaluation protocols for assessing procedural generation and simulation fidelity.

## Evaluation Benchmarks for Procedural and Simulation Models

Evaluation Benchmarks for Procedural and Simulation Models focuses on Summary of existing benchmarks, metrics, and evaluation protocols for assessing procedural generation and simulation fidelity.

Existing evaluation practice for agentic and simulation models spans interactive benchmarks for LLM agents, survey-level syntheses of autonomous agents, and world-model-driven forecasting and planning protocols in autonomous driving. A landmark work, AgentBench [paper:10.48550/arxiv.2308.03688], addresses the urgent need to quantitatively evaluate LLMs as agents on challenging tasks in interactive environments. The survey on large language model based autonomous agents reviews a field in which the development of autonomous agents has been a long-standing objective in both industry and academia. [paper:10.1007/s11704-024-40231-1]. Driving Into the Future contributes multiview visual forecasting and planning with a world model, in which predicting future events in advance and evaluating the foreseeable risks empowers autonomous vehicles to better plan their actions, enhancing safety and efficiency on the road [paper:10.1109/cvpr52733.2024.01397].

Methodologically, AgentBench establishes a quantitative protocol that evaluates LLMs as agents within interactive environments on challenging tasks [paper:10.48550/arxiv.2308.03688]. In the survey, an autonomous agent is defined as a system that interacts with its environment by perceiving its surroundings and taking actions over time to achieve specific goals. [paper:10.1007/s11704-024-40231-1]. Current autonomous driving systems typically divide the problem into four steps, namely perception, tracking, trajectory prediction, and path planning, among which trajectory prediction plays a pivotal role. [paper:10.1109/cvpr52733.2024.01397].

These works suggest differing evaluation settings, spanning interactive environments for LLM agents in AgentBench, environment interaction over time for autonomous agents in the survey, and forecast-based risk evaluation within driving pipelines. [paper:10.48550/arxiv.2308.03688].

A central limitation recorded in the survey is that previous research historically concentrated on training agents with limited knowledge in isolated environments, a process that differs from human learning and hampers the agents' ability to make decisions akin to those of humans [paper:10.1007/s11704-024-40231-1]. AgentBench likewise reflects an evaluation gap, since the potential of LLMs as agents has been widely acknowledged while quantitative evaluation on challenging tasks in interactive environments remained an urgent need [paper:10.48550/arxiv.2308.03688]. In autonomous driving, current systems typically decompose the problem into perception, tracking, trajectory prediction, and path planning, whereas human drivers predict the motion of other road agents in order to anticipate potentially dangerous situations, underscoring the pivotal role of trajectory prediction [paper:10.1109/cvpr52733.2024.01397].

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

- type=B swap_evidence -> repaired: Preview chunk on Prediction Capability almost verbatim supports the claim (capacity to reason about future evolution of dynamic driving scenes, fundamental to safe/efficient/proactive behavior); re-point citation to it.
- type=B rewrite_claim -> repaired: Drops the code-repository clause not covered by the strongest chunk; remaining assertion is directly supported by the survey's abstract chunk on GPT-4/Sora/AGI and comprehensive review.
- type=B swap_evidence -> repaired: Chunk directly evidences the claim's distinctive assertion: LLMs' strong reasoning capabilities via chain-of-thought prompting with intermediate steps; the planning-as-world-model framing is the paper's own title.
- type=B rewrite_claim -> invalid_action: First half is directly supported by the cited driving-survey chunk; the procedural-content-generation transposition (rolling out world models of levels/storylines) is the author's extrapolation the paper cannot ground, so it is removed.
- type=B rewrite_claim -> repaired: The claimed mapping of methodological families to MLLMs vs. Sora and the game-taxonomy analogy are not stated in the paper; the two-purposes framing is verbatim supported by its foundational-definition chunk.
- type=B rewrite_claim -> repaired: Evidence supports the CoT prompting description and the world-model framing, but not that CoT rollouts are world-model simulations 'directly reusable' for storylines; weaken to a supportable assertion.
- type=B rewrite_claim -> repaired: Cited driving survey supports only the prediction-scoping half; the comparison to the comprehensive survey is not grounded by this paper, so keep only the supported part and leave the taxonomy suggestion in the authorial voice.
- type=B remap_citation -> invalid_action: The sentence explicitly names 'Reasoning with Language Model is Planning with World Model', which is exactly the title of paper:10.48550/arxiv.2305.14992; the cited driving survey cannot support the RAP attribution.
- type=B swap_evidence -> repaired: This chunk directly states the survey closes by outlining key challenges and future research directions, grounding the closing claim without needing a rewrite.
- type=B swap_evidence -> repaired: This chunk ties modeling of multi-agent economic interactions to the importance of predicting collective behavior, directly grounding both parts of the claim.
- type=B rewrite_claim -> repaired: Preview evidence supports the MARL framing and the resource-allocation prediction motivation, but not the public goods games/hypergraph specificity, so weaken to what the chunks state.
- type=B backfill_evidence -> repaired: Claim mirrors the paper's own title/abstract (LLM as world model of the internet for model-based planning); semantic search should retrieve the grounding world-model chunk.
- type=B rewrite_claim -> invalid_action: Tree search outperforming reactive planning is directly supported by the evidence; drop the unsupported implicit-world-model coupling.
- type=B swap_evidence -> repaired: This chunk contains the near-verbatim sentence 'framework for action-controlled and scene-generalizable game video generation', directly grounding the claim.
- type=B swap_evidence -> repaired: This chunk verbatim states MineDojo open-sources its simulation suite, knowledge bases, algorithm implementation, and pretrained models toward generally capable embodied agents.
- type=B None -> unresolved: no usable action list after retry: model reply contained no JSON object
- type=B None -> unresolved: no usable action list after retry: model reply contained no JSON object
- type=B None -> unresolved: no usable action list after retry: model reply contained no JSON object
- type=B None -> unresolved: no usable action list after retry: model reply contained no JSON object
- type=B None -> unresolved: no usable action list after retry: model reply contained no JSON object
- type=B backfill_evidence -> repaired: AgentBench abstract states it evaluates LLMs as agents on challenging tasks in interactive environments; semantic search should retrieve that grounding chunk from the cited paper.
- type=B backfill_evidence -> repaired: The claim paraphrases the survey's framing of prior work training agents with limited knowledge in isolated environments vs real-world conditions; backfill will fetch the abstract passage supporting it.
- type=B backfill_evidence -> repaired: AgentBench's abstract explicitly notes LLMs' agent potential is acknowledged while quantitative evaluation in interactive environments is missing; backfill should retrieve this.
- type=B rewrite_claim -> repaired: Previewed chunks support the four-step decomposition and the human prediction-to-anticipation contrast, but not the world-model assertion, so the claim is weakened to what the evidence grounds.
- type=D rewrite_claim -> repaired: Evidence supports prediction's role in driving safety/efficiency/decision-making, but 'paramount' and 'supplies the conceptual core' for game content generation overstate; weakened to a suggested analogy
- type=D rewrite_claim -> invalid_action: Struggle with multi-step reasoning/planning despite CoT is directly supported, but 'a principal limitation' and the hard 'bounds' on generation go beyond the evidence; hedged
- type=D rewrite_claim -> invalid_action: Evidence supports prediction serving safe/efficient/proactive driving decision-making, but 'confined' and 'must be adapted' are too categorical; softened to 'centered' and 'suggesting'
- type=D keep -> kept: Claim mirrors the paper's own statement ('Recent work has shown that incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning for web agents'), so the assertion is appropriately calibrated
- type=D rewrite_claim -> repaired: Preview evidence grounds multi-agent interactions and LLM-empowered modeling of collective behavior, but does not establish the categorical 'spans disciplines' scope; weakened and anchored to the resource-allocation context
- type=D rewrite_claim -> repaired: Evidence ties the dataset to action control; scene-generalizable generation is a framework-level property driven by pre-trained video diffusion priors, so the attribution to the dataset is weakened.
- type=D keep -> kept: Claim closely mirrors the paper's stated motivation (agents typically learning tabula rasa in isolated environments) and contribution (open-ended embodied agents with internet-scale knowledge); assertion is acceptable as-is.
- type=D rewrite_claim -> repaired: Evidence supports strategic reasoning in game-theoretic settings; the added 'logical reasoning' attribution overstates GTBench's focus.
- type=D rewrite_claim -> repaired: MineDojo is a simulation suite for developing agents, not a generation framework, so 'complementary layers of interactive generation' overstates; framing softened to 'complementary purposes'.
- type=D rewrite_claim -> invalid_action: Drops the subjective 'landmark' label and the 'urgent' intensifier while retaining the core assertion the evidence supports.
- type=D rewrite_claim -> repaired: Evidence supports that agent development is a long-standing objective in academia/industry, but not that the survey 'consolidates' the field; softened to 'reviews' and aligned wording with evidence.
- type=D rewrite_claim -> repaired: Evidence directly supports only the definition itself; removed the unsupported assertion that the survey grounds evaluation in this definition.
- type=D rewrite_claim -> repaired: Dropped the unsupported 'On the simulation side' framing; the four-step division and trajectory prediction's pivotal role match the evidence nearly verbatim.
- type=D rewrite_claim -> repaired: A comparison 'showing' divergent settings is too strong for the cited intro-level evidence; weakened 'shows divergent' to 'suggests differing' while preserving the enumeration.
