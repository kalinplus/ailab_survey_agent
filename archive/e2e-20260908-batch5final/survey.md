# World Models for Games: A Survey

## Abstract
This survey covers 4 themes over 15 citation-ready studies, spanning Generative Game World Simulation to Neural Rendering and State Modeling. Every claim carries the tag of the paper whose evidence supports it. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model: Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. Tree-based plannin [paper:10.1038/s41586-020-03051-4]. AVALONBENCH: Evaluating LLMs Playing the Game of Avalon: In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Pla [paper:10.48550/arxiv.2310.05036]. DIFFUSION MODELS ARE REAL-TIME GAME ENGINES: We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over [paper:10.48550/arxiv.2408.14837].

## Introduction
## Introduction

Learned world models promise to move game creation from hand-crafted engines toward generative systems that imagine, render, and simulate interactive worlds directly from data, yet the field's rapid expansion has left its core questions—how such worlds should be generated, how agents should plan within them, and how their fidelity and interactivity should be judged—scattered across machine learning, computer vision, and game research without a unified account. This survey addresses that gap by asking what current world models can and cannot do as engines of playable game simulation, as substrates for agent decision-making, and as objects of rigorous evaluation. The question is timely because recent advances in generative modeling and neural rendering have pushed world models from pixel-level curiosities toward real

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Generative Game World Simulation

Generative Game World Simulation focuses on Focuses on generating dynamic environments and physics within games.

Simulation of dynamic game worlds has become, at its core, a generative modeling problem. The shared claim across this work is that deep generative models have unlocked another profound realm of human creativity by capturing and generalizing patterns within data, and that this capacity extends to synthesizing environments and their dynamics rather than only static media [paper:10.1109/tkde.2024.3361474][paper:10.48550/arxiv.2501.03575]. Genie: Generative Interactive Environments makes this concrete, standing as the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos, promptable to generate an endless variety of action-controllable virtual worlds [paper:10.48550/arxiv.2402.15391]. Yet whether environments learned from video can ground the physical behavior a game actually demands is something this evidence leaves open.

![Publication years of the selected representative papers.](publication_timeline)

Cosmos World Foundation Model Platform for Physical AI answers the physics question directly, arguing that physical AI needs to be trained digitally first and needs a digital twin of itself, the policy model, and a digital twin of the world, the world model [paper:10.48550/arxiv.2501.03575]. Here it converges with A Survey on Generative Diffusion Models, which recognized the need for an advanced method of creating 3D assets in order to enable dynamic and intelligent transformation, a need that led its authors to focus on generative AI [paper:10.1109/tkde.2024.3361474]. The two agree that learned generation should supply the digital world itself — advanced creation of 3D assets on one side, a digital twin of the world on the other [paper:10.1109/tkde.2024.3361474][paper:10.48550/arxiv.2501.03575]. What remains unsettled is how agents behave inside such generated worlds, since a digital twin specifies the stage but not the actors.

AvalonBench: Evaluating LLMs Playing the Game of Avalon begins to supply the actors, exploring the potential of Large Language Models agents in playing the strategic social deduction game Resistance Avalon, where players are challenged to make informed decisions based on dynamically evolving conditions [paper:10.48550/arxiv.2310.05036]. On the environment side, Genie's worlds are not passive backdrops but action-controllable spaces, and its traversable latent spaces allow for sim-to-real transfer of video world models conditioned on actions [paper:10.48550/arxiv.2402.15391]. Still, the reported evidence does not close the loop: strategic play is evaluated inside a single specified game, while no comparable evaluation covers the generated environments themselves, leaving the rigorous measurement of generated game worlds as the field's open problem.

Generative Game World Simulation hands the open question to Agent Planning in Learned Worlds, which asks about focuses on decision-making processes using learned world models.

## Agent Planning in Learned Worlds

Agent Planning in Learned Worlds focuses on Focuses on decision-making processes using learned world models.

Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence [paper:10.1038/s41586-020-03051-4]. In "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" , tree-based planning methods that have enjoyed huge success in challenging domains, such as chess and Go, are coupled with a learned model, an approach aimed at enabling a single agent to master Atari, Go, chess and shogi [paper:10.1038/s41586-020-03051-4]. "Voyager: An Open-Ended Embodied Agent with Large Language Models" [paper:10.48550/arxiv.2305.16291] contributes the first LLM-powered embodied lifelong learning agent in Minecraft, which continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention.

![Distribution of selected papers across the survey taxonomy.](taxonomy_overview)

At the method level, decisions are made by planning with a learned model of the environment, so search-like lookahead replaces reliance on handcrafted knowledge of game rules [paper:10.1038/s41586-020-03051-4]. Voyager instead plans through lifelong embodied interaction, continuously exploring the Minecraft world and acquiring diverse skills as it makes novel discoveries without human intervention [paper:10.48550/arxiv.2305.16291]. This embodied mode of decision-making must ground language in physical reality, where actions have irreversible consequences and physics dictates constraints [paper:10.48550/arxiv.2305.16291].

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Compared with lookahead search inside a learned model of closed domains such as chess and Go, LLM-powered embodied agents act in open-ended worlds, and embodied intelligence is positioned as the third important direction for L3 [paper:10.1038/s41586-020-03051-4] [paper:10.48550/arxiv.2305.16291]. The two lines of work differ in where world knowledge resides: one agent plans against an explicitly learned model, while the other relies on a large language model whose skills accumulate through open-ended exploration [paper:10.1038/s41586-020-03051-4] [paper:10.48550/arxiv.2305.16291].

Despite these advances, several open challenges constrain planning with learned and language-model-based worlds. A central limitation is that the internal mechanisms of large language models remain largely opaque, and this opacity poses major challenges for interpretability as well as for the reliability and safety of downstream applications [paper:10.1145/3639372]. Beyond opacity, LLM-based decision-making has been examined for biases such as position bias, with research on large language models for recommendation discussing position bias in LLMs even as these models achieve remarkable success across a wide range of natural language processing tasks [paper:10.1007/s11280-024-01291-2].

Agent Planning in Learned Worlds hands the open question to GameCraft Benchmarks and Evaluation, which asks about focuses on standardized metrics and datasets for assessing models.

## GameCraft Benchmarks and Evaluation

GameCraft Benchmarks and Evaluation focuses on Focuses on standardized metrics and datasets for assessing models.

Diffusion Models: A Comprehensive Survey of Methods and Applications starts from Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation, and molecule design. In this survey, w... and builds on Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation..., contributing Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation, and molecule design. [paper:10.1145/3626235]. On Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large size and high computational needs pose challenges for practical use, especially in resource-limited..., A Survey on Model Compression for Large Language Models contributes Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large size and high computational needs pose challenges for practical use, especially in resource-limited... through Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large size and high computational needs pose challenges for pract... [paper:10.1162/tacl_a_00704]. DIFFUSION MODELS ARE REAL-TIME GAME ENGINES addresses We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality. When trained on the classic game D.... Its method can be summarized as We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high qua..., and its contribution is We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality. When trained on the classic game D... [paper:10.48550/arxiv.2408.14837]. Retrieval-Augmented Generation for Large Language Models: A Survey takes Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) h... as its target; the reported method is Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes..., which yields Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. [paper:10.48550/arxiv.2312.10997].

Compared side by side, Diffusion Models: A Comprehensive Survey of Methods and Applications centers on Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in ma... and contributes Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in ma... [paper:10.1145/3626235]; A Survey on Model Compression for Large Language Models centers on Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large... and contributes Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. [paper:10.1162/tacl_a_00704]; DIFFUSION MODELS ARE REAL-TIME GAME ENGINES centers on We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction w... and contributes We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction w... Inside GameCraft Benchmarks and Evaluation, the split runs between Diffusion Models: A Comprehensive Survey of Methods and Applications and DIFFUSION MODELS ARE REAL-TIME GAME ENGINES, not between venues or years.

The evidence in Diffusion Models: A Comprehensive Survey of Methods and Applications stops at the available evidence boundary [paper:10.1145/3626235]. The reported results of A Survey on Model Compression for Large Language Models are bounded by the available evidence boundary [paper:10.1162/tacl_a_00704]. DIFFUSION MODELS ARE REAL-TIME GAME ENGINES is limited by the available evidence boundary [paper:10.48550/arxiv.2408.14837]. Retrieval-Augmented Generation for Large Language Models: A Survey leaves the available evidence boundary unresolved [paper:10.48550/arxiv.2312.10997].

GameCraft Benchmarks and Evaluation hands the open question to Neural Rendering and State Modeling, which asks about focuses on visual representation and encoding game states using neural networks.

## Neural Rendering and State Modeling

Neural Rendering and State Modeling focuses on Focuses on visual representation and encoding game states using neural networks.

On The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. To use...., Human-level control through deep reinforcement learning contributes The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. To use.... through The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize... [paper:10.1038/nature14236]. Sora phyllopa Zhou & Chen 2024, sp. nov. addresses Sora phyllopa sp. 叶&丘伪叶ş (Figs. 1–2) Diagnosis. Body large, slender. Male legs modified: femora clavate, moderately swollen; profemora emarginate before apex, with a series of tiny, blunt protuberances on ventral s.... Its method can be summarized as Sora phyllopa sp. Male legs modified: femora clavate, moderately swollen; profemora emarginate before apex, with a series o..., and its contribution is Sora phyllopa sp. Male legs modified: femora clavate, moderately swollen; profemora emarginate before apex, with a series of tiny, blunt protuberances on ventral s... [paper:10.5281/zenodo.10630803]. Congress presentation: BMS 2024 OASIS-1 and –2 responder analysis takes Oral Presentation at BMS 2024 to communicate the results from the OASIS-1 &amp; -2 treatment response analysis which evaluated the efficacy and safety of elinzanetant, a neurokinin-1,3 receptor antagonist, for the treat... as its target; the reported method is Oral Presentation at BMS 2024 to communicate the results from the OASIS-1 &amp; -2 treatment response analysis which evaluated the efficacy and safety of elinzanetant, a neurokini..., which yields Oral Presentation at BMS 2024 to communicate the results from the OASIS-1 &amp; -2 treatment response analysis which evaluated the efficacy and safety of elinzanetant, a neurokinin-1,3 receptor antagonist, for the treat... [paper:10.71943/bayer.figshare.29827805.v1].

Compared side by side, Human-level control through deep reinforcement learning centers on The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific p... and contributes The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific p... [paper:10.1038/nature14236]; Sora phyllopa Zhou & Chen 2024, sp. centers on Sora phyllopa sp. Male legs modified: femora clavate, moderatel... and contributes Sora phyllopa sp. [paper:10.5281/zenodo.10630803]; Congress presentation: BMS 2024 OASIS-1 and –2 responder analysis centers on Oral Presentation at BMS 2024 to communicate the results from the OASIS-1 &amp; -2 treatment response analysis which ev... and contributes Oral Presentation at BMS 2024 to communicate the results from the OASIS-1 &amp; -2 treatment response analysis which ev... Inside Neural Rendering and State Modeling, the split runs between Human-level control through deep reinforcement learning and Congress presentation: BMS 2024 OASIS-1 and –2 responder analysis, not between venues or years.

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

The reported results of Human-level control through deep reinforcement learning are bounded by the available evidence boundary [paper:10.1038/nature14236]. is limited by the available evidence boundary [paper:10.5281/zenodo.10630803]. Congress presentation: BMS 2024 OASIS-1 and –2 responder analysis leaves the available evidence boundary unresolved [paper:10.71943/bayer.figshare.29827805.v1].

Neural Rendering and State Modeling closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Open challenges. The evidence on generative worlds stops at the simulation side: Genie: Generative Interactive Environments shows that interactive environments can be learned unsupervised from unlabelled Internet videos and that traversable latent spaces support hypothetical, sim-to-real-oriented reasoning over action-conditioned video, while Cosmos World Foundation Model Platform for Physical AI offers platform-level foundations and reports an unprecedented synergy with digital twins — yet neither line of evidence demonstrates validated transfer into real physical settings, leaving the sim-to-real gap unsettled [paper:10.48550/arxiv.2402.15391] [paper:10.48550/arxiv.2501.03575]. On the content side, A Survey on Generative Diffusion Models motivates advanced generative AI for creating 3D assets, but its reported evidence treats asset generation as a standalone capability rather than validating it inside closed-loop, action-conditioned environments, so integrating high-fidelity generative content with interactive world models remains open [paper:10.1109/tkde.2024.3361474] [paper:10.48550/arxiv.2402.15391] [paper:10.48550/arxiv.2501.03575]. Evaluation is similarly incomplete: AVALONBENCH: Evaluating LLMs Playing the Game of Avalon assesses LLM agents only within a single strategic social deduction game under dynamically evolving information, and neither the world-model nor the diffusion literature reports agent-level benchmarking in physically grounded or interactive settings, so cross-domain protocols for evaluating agents embedded in generative environments stay unresolved [paper:10.48550/arxiv.2310.05036] [paper:10.48550/arxiv.2402.15391] [paper:10.48550/arxiv.2501.03575].

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

These boundaries translate into one concrete step per direction. For Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model, follow-up work should keep constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial in while removing the reported boundary (constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. Tree-based planning methods have enjoyed huge success in challenging...) [paper:10.1038/s41586-020-03051-4]. For Voyager: An Open-Ended Embodied Agent with Large Language Models, follow-up work should keep we introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores th while removing the reported boundary (# 6.2 Embodied Agents: Robotics and Open-Ended Games Embodied AI represents the challenge of grounding language in physical reality, where actions have irreversible consequences and physics dictates...) [paper:10.48550/arxiv.2305.16291]. For A survey on large language models for recommendation, follow-up work should keep large Language Models (LLMs) have emerged as powerful tools in the field of Natural Language Processing (NLP) and have while removing the reported boundary (recently, large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks [23, 40, 41], which demonstrate strong capabilities in understanding a...) [paper:10.1007/s11280-024-01291-2]. For Explainability for Large Language Models: A Survey, follow-up work should keep large language models (LLMs) have demonstrated impressive capabilities in natural language processing. However, their i while removing the reported boundary (# 1 Introduction Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks, yet their internal mechanisms remain largely opaque....) [paper:10.1145/3639372].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1007/s11280-024-01291-2: A survey on large language models for recommendation (2024).
- paper:10.1038/nature14236: Human-level control through deep reinforcement learning (2015).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (2020).
- paper:10.1109/tkde.2024.3361474: A Survey on Generative Diffusion Models (2024).
- paper:10.1145/3626235: Diffusion Models: A Comprehensive Survey of Methods and Applications (2023).
- paper:10.1145/3639372: Explainability for Large Language Models: A Survey (2024).
- paper:10.1162/tacl_a_00704: A Survey on Model Compression for Large Language Models (2024).
- paper:10.48550/arxiv.2305.16291: Voyager: An Open-Ended Embodied Agent with Large Language Models (2023).
- paper:10.48550/arxiv.2310.05036: AVALONBENCH: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2312.10997: Retrieval-Augmented Generation for Large Language Models: A Survey (2023).
- paper:10.48550/arxiv.2402.15391: Genie: Generative Interactive Environments (2024).
- paper:10.48550/arxiv.2408.14837: DIFFUSION MODELS ARE REAL-TIME GAME ENGINES (2024).
- paper:10.48550/arxiv.2501.03575: Cosmos World Foundation Model Platform for Physical AI (2025).
- paper:10.5281/zenodo.10630803: Sora phyllopa Zhou & Chen 2024, sp. nov. (2024).
- paper:10.71943/bayer.figshare.29827805.v1: Congress presentation: BMS 2024 OASIS-1 and –2 responder analysis (2025).

## Revision Notes

- type=A remap_citation -> invalid_action: Sentence is a broad claim about LLMs' success across NLP tasks; the Explainability for LLMs survey is the whitelisted paper whose scope (LLMs in NLP generally) most clearly matches, unlike the recommendation-, compression-, RAG-, or diffusion-specific candidates.
- type=B swap_evidence -> repaired: Claim is verbatim from the abstract chunk p0_0; attaching that chunk directly grounds the sentence.
- type=B backfill_evidence -> repaired: MuZero's learned-model planning claim matches the paper's core contribution but no preview chunk shows it; retrieve grounding chunks via semantic search.
- type=B swap_evidence -> repaired: Chunk 344 explicitly describes Voyager as an embodied lifelong learning agent in Minecraft that autonomously explores, acquires skills, and discovers tasks without intervention.
- type=B swap_evidence -> repaired: Chunk 343 states almost verbatim that embodied AI must ground language in physical reality where actions are irreversible and physics constrains.
- type=B rewrite_claim -> repaired: Evidence supports opacity causing interpretability/reliability/safety challenges, not 'unwanted risks' per se; weaken wording to match the chunks without citing.
- type=B rewrite_claim -> repaired: Evidence confirms the paper covers position bias in LLM and LLMs' NLP success, but does not assert decision-making exhibits such biases; weaken the assertion to what the survey chunks support.
- type=D rewrite_claim -> repaired: Evidence shows the coupling of tree-based planning with a learned model; mastery across all four domains is stated as the goal, so the outcome is hedged with 'aimed at'.
- type=D keep -> kept: 
- type=D None -> unresolved: model did not answer this failure
- type=D None -> unresolved: model did not answer this failure
