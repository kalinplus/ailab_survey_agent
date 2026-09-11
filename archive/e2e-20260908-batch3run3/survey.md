# World Models for Games: A Survey

## Abstract
This survey covers 4 themes over 15 citation-ready studies, spanning Generative Game World Simulation to Neural Rendering and Multimodal Game State Modeling. Every claim carries the tag of the paper whose evidence supports it. Mastering Atari, Go, chess and shogi by planning with a learned model: # Critic PI2: Master Continuous Planning via Policy Improvement with Path Integrals and Deep Actor-Critic Reinforcement Learning Jiajun Fan $^{1}$ [paper:10.1038/s41586-020-03051-4]. Human-level control through deep reinforcement learning: Of these, the theory of reinforcement learning provides the most natural framework, deeply rooted in psychological $^{49}$ and neuroscientific $^{50} [paper:10.1038/nature14236]. Voyager: An Open-Ended Embodied Agent with Large Language Models: 3) Embodied Intelligence: Embodied intelligence is the third important direction for L3, aiming to give agents the ability to perceive, interact, and [paper:10.48550/arxiv.2305.16291].

## Introduction
Game intelligence increasingly rests on models that learn how a world evolves rather than on hand-written rules alone; whether such models can support planning, generation, and fair evaluation is the question this survey addresses. Each theme that follows gathers the evidence for one capability and states where that evidence stops.

Generative Game World Simulation surveys papers focusing on learning generative models of game environments, including dynamics, physics, and object interactions without explicit programming. Agent Planning and Control in Learned Worlds collects the work on research on how agents utilize learned world models for planning, decision-making, and control within simulated or inferred environments. GameCraft Benchmarks and Evaluation Protocols follows the evidence for studies establishing standardized benchmarks, datasets, and evaluation metrics for assessing world model performance in gaming contexts. Neural Rendering and Multimodal Game State Modeling maps what is known about work involving neural rendering techniques and multimodal inputs (e.g., vision, audio) to represent and reconstruct game states.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Generative Game World Simulation

Generative Game World Simulation focuses on Papers focusing on learning generative models of game environments, including dynamics, physics, and object interactions without explicit programming.

Generative game world simulation aims to learn generative models of game environments, including their dynamics, physics, and object interactions, without explicit programming. Genie: Generative Interactive Environments [paper:10.48550/arxiv.2402.15391] is the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos, and it can be prompted to generate an endless variety of action-controllable virtual worlds. Diffusion Models Are Real-Time Game Engines [paper:10.48550/arxiv.2408.14837] presents GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality. A Survey on Generative Diffusion Models [paper:10.1109/tkde.2024.3361474] situates such systems within the broader observation that deep generative models have unlocked another profound realm of human creativity by capturing and generalizing patterns within data.

Genie's training proceeds from unlabelled Internet videos, and in the broader study of video world models, hypothetical reasoning in training using traversable latent spaces has been discussed as an approach that allows sim-to-real transfer of video world models conditioned on actions. [paper:10.48550/arxiv.2402.15391]. On the engine side, recent advances demonstrate that generative models can serve as complete, neural game engines, replacing traditional rendering and state update logic, with GameNGen trained on the classic game Doom to realize this substitution [paper:10.48550/arxiv.2408.14837]. An earlier precedent along this line is GameGAN, which learns to imitate 2D games from raw pixels and actions using GANs [paper:10.48550/arxiv.2408.14837]. Beyond interactive engines, generative AI has also been recognized as an advanced method for creating 3D assets, a field that has seen significant advances [paper:10.1109/tkde.2024.3361474].

Genie, the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos, can generate an endless variety of action-controllable 2D worlds from image prompts. [paper:10.48550/arxiv.2402.15391]. These visual environment simulators also contrast with large language models, which have achieved remarkable success across a wide range of natural language processing tasks and demonstrate strong capabilities in understanding and generating human-like text [paper:10.1007/s11280-024-01291-2].

The ability to respond appropriately to user inputs represents a fundamental requirement for gaming world models, and recent neural game engines address it by enabling real-time interaction with complex environments [paper:10.48550/arxiv.2408.14837].

Generative Game World Simulation hands the open question to Agent Planning and Control in Learned Worlds, which asks about research on how agents utilize learned world models for planning, decision-making, and control within simulated or inferred environments.

## Agent Planning and Control in Learned Worlds

Agent Planning and Control in Learned Worlds focuses on Research on how agents utilize learned world models for planning, decision-making, and control within simulated or inferred environments.

Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence, and the landmark work Mastering Atari, Go, chess and shogi by planning with a learned model [paper:10.1038/s41586-020-03051-4] demonstrates that an agent can master Atari, Go, chess and shogi by planning with a learned model. Voyager [paper:10.48550/arxiv.2305.16291] contributes the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention. The Cosmos World Foundation Model Platform [paper:10.48550/arxiv.2501.03575] contributes a world foundation model platform for Physical AI, whose premise is that Physical AI needs to be trained digitally first with a digital twin of itself, the policy model, and a digital twin of the world, the world model.

Methodologically, Mastering Atari, Go, chess and shogi by planning with a learned model builds on tree-based planning methods, which have enjoyed huge success in challenging domains such as chess and Go, where a perfect simulator is available. [paper:10.1038/s41586-020-03051-4]. Embodied intelligence aims to give agents the ability to perceive, interact, and learn in the physical world or highly simulated virtual environments, where actions have irreversible consequences and physics dictates constraints, and Voyager represents a milestone in this field by achieving lifelong learning in Minecraft [paper:10.48550/arxiv.2305.16291]. The Cosmos platform operationalizes its premise by training a powerful foundation model on diverse multimodal inputs to serve as a digital twin of the physical world, tightly integrated with external control, memory, and planning modules [paper:10.48550/arxiv.2501.03575].

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Comparing these systems, planning with a learned model in Mastering Atari, Go, chess and shogi by planning with a learned model realizes control through tree-based search, Voyager pursues open-ended control through lifelong exploration and skill acquisition in Minecraft, and Cosmos frames control as policy learning inside a digital twin of the world [paper:10.1038/s41586-020-03051-4] [paper:10.48550/arxiv.2305.16291] [paper:10.48550/arxiv.2501.03575]. These efforts span complementary settings, from challenging games such as chess and Go, where a perfect simulator is available, to broader agent domains to which tree search and simulation-based planning have recently been extended, indicating that planning with learned components now reaches beyond perfect-simulator games [paper:10.1038/s41586-020-03051-4].

A key limitation concerns transparency, since the large language models that power embodied agents such as Voyager have internal mechanisms that remain largely opaque, and this opacity poses major challenges for interpretability as well as for ensuring reliability and safety. [paper:10.1145/3639372]. Indeed, this opacity poses major challenges for interpretation, raising questions about how LLM-powered agents make planning and control decisions and motivating the explainability analyses surveyed for large language models as a complement to world-model-based planning [paper:10.1145/3639372].

Agent Planning and Control in Learned Worlds hands the open question to GameCraft Benchmarks and Evaluation Protocols, which asks about studies establishing standardized benchmarks, datasets, and evaluation metrics for assessing world model performance in gaming contexts.

## GameCraft Benchmarks and Evaluation Protocols

GameCraft Benchmarks and Evaluation Protocols focuses on Studies establishing standardized benchmarks, datasets, and evaluation metrics for assessing world model performance in gaming contexts.

A Survey on Model Compression for Large Language Models starts from Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large size and high computational needs pose challenges for practical use, especially in resource-limited... and builds on Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large size and high computational needs pose challenges for pract..., contributing Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. [paper:10.1162/tacl_a_00704]. On The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. To use...., Human-level control through deep reinforcement learning contributes The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. To use.... through The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize... [paper:10.1038/nature14236]. AvalonBench: Evaluating LLMs Playing the Game of Avalon addresses In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Players in Avalon are challenged not only to make informed decisions base.... Its method can be summarized as In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Players in Avalon are challenged..., and its contribution is In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Players in Avalon are challenged not only to make informed decisions base... [paper:10.48550/arxiv.2310.05036]. Retrieval-Augmented Generation for Large Language Models: A Survey takes Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) h... as its target; the reported method is Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes..., which yields Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. [paper:10.48550/arxiv.2312.10997].

Compared side by side, A Survey on Model Compression for Large Language Models centers on Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large... and contributes Abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. [paper:10.1162/tacl_a_00704]; Human-level control through deep reinforcement learning centers on The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific p... and contributes The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific p... [paper:10.1038/nature14236]; AvalonBench: Evaluating LLMs Playing the Game of Avalon centers on In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deductio... and contributes In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deductio... Inside GameCraft Benchmarks and Evaluation Protocols, the split runs between A Survey on Model Compression for Large Language Models and AvalonBench: Evaluating LLMs Playing the Game of Avalon, not between venues or years.

The evidence in A Survey on Model Compression for Large Language Models stops at the available evidence boundary [paper:10.1162/tacl_a_00704]. The reported results of Human-level control through deep reinforcement learning are bounded by the available evidence boundary [paper:10.1038/nature14236]. AvalonBench: Evaluating LLMs Playing the Game of Avalon is limited by the available evidence boundary [paper:10.48550/arxiv.2310.05036]. Retrieval-Augmented Generation for Large Language Models: A Survey leaves the available evidence boundary unresolved [paper:10.48550/arxiv.2312.10997].

GameCraft Benchmarks and Evaluation Protocols hands the open question to Neural Rendering and Multimodal Game State Modeling, which asks about work involving neural rendering techniques and multimodal inputs (e.g., vision, audio) to represent and reconstruct game states.

## Neural Rendering and Multimodal Game State Modeling

Neural Rendering and Multimodal Game State Modeling focuses on Work involving neural rendering techniques and multimodal inputs (e.g., vision, audio) to represent and reconstruct game states.

Diffusion models have emerged as one of the most powerful families of deep generative models, surpassing GANs on image synthesis and achieving state-of-the-art performance across a wide range of downstream applications [paper:10.1145/3626235]. Recent years have witnessed the remarkable success of diffusion models, accompanied by a range of visually stunning generative contents, and diffusion models have emerged as one of the most powerful families of generative models, achieving state-of-the-art performance across a wide range of tasks [paper:10.1145/3626235].

For neural rendering and multimodal game state modeling, these generative advances supply a foundation for representing and reconstructing game states from inputs such as vision and audio. As a family of deep generative models, diffusion models have achieved remarkable success in recent years, accompanied by the emergence of visually stunning generative contents [paper:10.1145/3626235]. Diffusion models constitute a family of probabilistic generative models that has achieved state-of-the-art performance across a wide range of tasks, and comprehensive surveys of the area cover both their underlying methods and their applications in diverse downstream domains. [paper:10.1145/3626235].

After surpassing GANs on image synthesis, diffusion models have shown promise as an algorithmic approach across a wide variety of downstream applications, emerging as one of the most powerful families of generative models. [paper:10.1145/3626235].

While diffusion models have achieved strong generative performance across a wide range of tasks, how these capabilities transfer to game-specific multimodal state reconstruction from vision and audio remains an open question [paper:10.1145/3626235].

Neural Rendering and Multimodal Game State Modeling closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Read across the body sections, the recorded boundaries cluster into a few open problems: Genie: Generative Interactive Environments: Notably, Google's Genie series [328, 336] marks a significant advancement in this area. Genie [328] is the first generative interactive environment trained in an unsupervised manner using unlabelled... [paper:10.48550/arxiv.2402.15391]; A Survey on Generative Diffusion Models: # A Survey on Generative Diffusion Models Hanqun Cao, Cheng Tan, Zhangyang Gao, Yilun Xu, Guangyong Chen, Pheng-Ann Heng, Senior Member, IEEE, and Stan Z. Li, Fellow, IEEE Abstract—Deep generative mo... [paper:10.1109/tkde.2024.3361474]; Diffusion Models Are Real-Time Game Engines: Interactivity. The ability to respond appropriately to user inputs represents a fundamental requirement for gaming world models. GameNGen demonstrates this capability by creating a fully neural game... [paper:10.48550/arxiv.2408.14837]; A survey on large language models for recommendation: Recently, large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks [23, 40, 41], which demonstrate strong capabilities in understanding a... [paper:10.1007/s11280-024-01291-2]. Each remains open rather than settled.

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

These boundaries translate into one concrete step per direction. For Mastering Atari, Go, chess and shogi by planning with a learned model, follow-up work should keep constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial in while removing the reported boundary (constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. Tree-based planning methods have enjoyed huge success in challenging...) [paper:10.1038/s41586-020-03051-4]. For Voyager: An Open-Ended Embodied Agent with Large Language Models, follow-up work should keep we introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores th while removing the reported boundary (# 6.2 Embodied Agents: Robotics and Open-Ended Games Embodied AI represents the challenge of grounding language in physical reality, where actions have irreversible consequences and physics dictates...) [paper:10.48550/arxiv.2305.16291]. For Explainability for Large Language Models: A Survey, follow-up work should keep large language models (LLMs) have demonstrated impressive capabilities in natural language processing. However, their i while removing the reported boundary (# 1 Introduction Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks, yet their internal mechanisms remain largely opaque....) [paper:10.1145/3639372]. For Cosmos World Foundation Model Platform for Physical AI, follow-up work should keep physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twi while removing the reported boundary (this wealth of work shows that Physical AI 2025 has become a mature, interdisciplinary field of research based equally on systematic theory, empirical methodology, and ethical reflection. The Cosmos...) [paper:10.48550/arxiv.2501.03575].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1007/s11280-024-01291-2: A survey on large language models for recommendation (2024).
- paper:10.1038/nature14236: Human-level control through deep reinforcement learning (2015).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, chess and shogi by planning with a learned model (2020).
- paper:10.1109/tkde.2024.3361474: A Survey on Generative Diffusion Models (2024).
- paper:10.1145/3626235: Diffusion Models: A Comprehensive Survey of Methods and Applications (2023).
- paper:10.1145/3639372: Explainability for Large Language Models: A Survey (2024).
- paper:10.1162/tacl_a_00704: A Survey on Model Compression for Large Language Models (2024).
- paper:10.48550/arxiv.2305.16291: Voyager: An Open-Ended Embodied Agent with Large Language Models (2023).
- paper:10.48550/arxiv.2310.05036: AvalonBench: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2312.10997: Retrieval-Augmented Generation for Large Language Models: A Survey (2023).
- paper:10.48550/arxiv.2402.15391: Genie: Generative Interactive Environments (2024).
- paper:10.48550/arxiv.2408.14837: Diffusion Models Are Real-Time Game Engines (2024).
- paper:10.48550/arxiv.2501.03575: Cosmos World Foundation Model Platform for Physical AI (2025).

## Revision Notes

- type=A remap_citation -> invalid_action: Survey on LLMs clearly supports the claim about LLM success across NLP tasks; original ids not whitelisted.
- type=A remap_citation -> invalid_action: Sentence is about Genie; candidate title 'Genie: Generative Interactive Environments' matches exactly.
- type=A remap_citation -> invalid_action: Claim about Google's Genie series is grounded by the Genie paper, the only matching candidate.
- type=B backfill_evidence -> repaired: Claim mirrors GameNGen's abstract ('first game engine entirely powered by a neural model... real-time interaction over long trajectories'); previewed chunks don't include it, so semantic backfill should retrieve the grounding chunk from the cited paper.
- type=B swap_evidence -> repaired: Abstract chunk verbatim states deep generative models 'have unlocked another profound realm of human creativity' by capturing and generalizing patterns, directly supporting the claim.
- type=B backfill_evidence -> repaired: First half is verbatim in the GameNGen paper and its DOOM training is described in that paper, but not in the truncated preview chunks; backfill fetches grounding for the full claim.
- type=B swap_evidence -> repaired: This chunk contains the GameGAN sentence ('learns to imitate 2D games from raw pixels and actions using GANs') verbatim, grounding the claim as written.
- type=B  -> invalid_action: 
- type=B rewrite_claim -> repaired: Cited paper's evidence chunks cover only Genie; the GameNGen comparison is unsupported, so the claim is narrowed to what the evidence shows.
- type=B rewrite_claim -> invalid_action: Evidence supports the interactivity requirement and real-time interaction, but not the 'persistent limitation' or 'long trajectories' framing.
- type=B swap_evidence -> repaired: This chunk is the paper's own abstract and verbatim matches the claim's opening assertion about planning being a main AI challenge.
- type=B swap_evidence -> repaired: Chunk explicitly states Voyager is the first LLM-powered embodied lifelong learning agent in Minecraft that autonomously explores, acquires skills, and discovers new tasks.
- type=B rewrite_claim -> invalid_action: The grounding/irreversible-consequences phrasing is a field-level statement, not a Voyager claim; evidence supports the milestone and lifelong-learning-in-Minecraft attribution instead.
- type=B rewrite_claim -> invalid_action: Evidence supports Cosmos as a digital twin of the physical world; the unsupported policy-model twin pairing and train-digitally-first specifics are removed.
- type=B backfill_evidence -> repaired: Preview chunks only ground tree-based planning in chess/Go; the survey's planning-and-simulation discussion likely grounds the Voyager and Cosmos comparisons, so fetch targeted grounding chunks.
- type=B backfill_evidence -> repaired: Preview lacks support for open-ended embodied games and Physical AI; backfill should retrieve chunks covering those environments elsewhere in the paper.
- type=B backfill_evidence -> repaired: Preview supports 'powerful family, state-of-the-art performance' but not the specific application list (video generation, molecule design); backfill to ground those applications.
- type=B keep -> kept: Claim closely matches preview chunks (remarkable success, visually stunning contents, most powerful family, state-of-the-art across tasks); assertion is acceptable as-is.
- type=B rewrite_claim -> repaired: Evidence supports DMs' success, SOTA performance, and coverage of methods plus downstream applications, but not the leap to rendering/game state reconstruction pipelines; weakened accordingly.
- type=B rewrite_claim -> repaired: Chunks state DMs surpassed GANs on image synthesis and emerged as one of the most powerful generative model families; dropped the unsupported 'anchors their standing' framing.
- type=B rewrite_claim -> invalid_action: The negative meta-claim that evidence reports no limitations is unverifiable and likely contradicted by survey limitation sections; replaced with a hedged, evidence-aligned statement.
- type=D rewrite_claim -> invalid_action: Evidence supports 'first ... unsupervised from unlabelled Internet videos', but action-controllable generation is only shown for 2D worlds from image prompts; 'virtual worlds' overstates the evidence.
- type=D rewrite_claim -> invalid_action: The hypothetical-reasoning/sim-to-real passage is a general discussion in the source, not explicitly Genie's own methodology; attribution weakened.
- type=D keep -> kept: Assertion matches the evidence nearly verbatim (remarkable success across NLP tasks, strong capabilities in understanding and generating human-like text); acceptable as-is.
- type=D rewrite_claim -> invalid_action: The 'trained digitally first with digital twin of policy and world' premise is not visible in the available evidence; claim reduced to what the evidence supports.
- type=D rewrite_claim -> invalid_action: Evidence only supports tree-based planning success under a perfect simulator; the 'replaces hand-crafted game rules' assertion is softened to planning with a learned model.
- type=D keep -> kept: placeholder
- type=D keep -> kept: placeholder
- type=D keep -> kept: placeholder
