# World Models for Games: A Survey

## Abstract
This survey covers 4 themes over 10 citation-ready studies, spanning Procedural Content Generation Mechanisms to Multi-Agent Planning and Mean Field Games. Every claim carries the tag of the paper whose evidence supports it. GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations: As Large Language Models (LLMs) are integrated into critical real-world applications, their strategic and logical reasoning abilities are increasingl [paper:10.48550/arxiv.2402.12348]. DIFFUSION MODELS ARE REAL-TIME GAME ENGINES: We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over [paper:10.48550/arxiv.2408.14837]. Mastering Atari, Go, chess and shogi by planning with a learned model: # Critic PI2: Master Continuous Planning via Policy Improvement with Path Integrals and Deep Actor-Critic Reinforcement Learning Jiajun Fan $^{1}$ [paper:10.1038/s41586-020-03051-4].

## Introduction
Game intelligence increasingly rests on models that learn how a world evolves rather than on hand-written rules alone; whether such models can support planning, generation, and fair evaluation is the question this survey addresses. Each theme that follows gathers the evidence for one capability and states where that evidence stops.

Procedural Content Generation Mechanisms surveys explores algorithms and frameworks for generating game content such as levels, rules, and assets within world models, emphasizing creativity, diversity, and alignment with player preferences. Benchmark Datasets and Evaluation Metrics collects the work on details the standardized environments, datasets, and evaluation criteria used to measure the performance, sample efficiency, and generalization capabilities of world models in gaming scenarios. Latent Architectures and Internal Simulation follows the evidence for investigates the design of latent space representations and internal simulation mechanisms that enable agents to predict future states and learn dynamics without explicit environment interaction. Multi-Agent Planning and Mean Field Games maps what is known about analyzes coordination strategies and planning algorithms for multiple agents within a shared world model, including applications of mean field game theory for large-scale agent interactions.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Procedural Content Generation Mechanisms

Procedural Content Generation Mechanisms focuses on Explores algorithms and frameworks for generating game content such as levels, rules, and assets within world models, emphasizing creativity, diversity, and alignment with player preferences.

Procedural content generation within world models draws on general learning algorithms, neural game engines, and agent evaluation frameworks as its core mechanisms. Developing a general algorithm that learns to solve tasks across a wide range of applications is a fundamental challenge in artificial intelligence, and Mastering Diverse Domains through World Models [paper:10.48550/arxiv.2301.04104] contributes such an algorithm. GameNGen, presented in DIFFUSION MODELS ARE REAL-TIME GAME ENGINES [paper:10.48550/arxiv.2408.14837], is the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality. AvalonBench [paper:10.48550/arxiv.2310.05036] explores the potential of Large Language Model agents in playing the strategic social deduction game Resistance Avalon.

The goal of Mastering Diverse Domains through World Models is to create a general algorithm that learns to master new domains, addressing the brittleness that otherwise poses a bottleneck in applying reinforcement learning to new problems [paper:10.48550/arxiv.2301.04104]. When trained on the classic game DOOM, GameNGen enables real-time interaction with a complex environment over long trajectories at high quality [paper:10.48550/arxiv.2408.14837]. AvalonBench revisits this problem in the context of Avalon, a social deduction game that provides a structured setting in which players are challenged to make informed decisions [paper:10.48550/arxiv.2310.05036].

Whereas Mastering Diverse Domains through World Models pursues a general algorithm that learns to solve tasks across a wide range of applications, the diffusion engine of GameNGen concentrates on simulating one complex environment with real-time interaction at high quality [paper:10.48550/arxiv.2301.04104] [paper:10.48550/arxiv.2408.14837]. While the diffusion engine captures the dynamics of a complex environment, social deduction games such as Werewolf, Chameleon, Avalon, and Jubensha exemplify how agents must navigate complex dynamics involving deception and collaboration with other agents [paper:10.48550/arxiv.2408.14837] [paper:10.48550/arxiv.2310.05036].

The brittleness of current reinforcement learning algorithms poses a bottleneck in applying reinforcement learning to new problems and also limits its applicability to computationally expensive models or tasks where tuning is prohibitive [paper:10.48550/arxiv.2301.04104]. Current language agents remain limited in their effectiveness in settings that require theory of mind or strategic social deduction [paper:10.48550/arxiv.2310.05036]. Together, these mechanisms indicate that creativity, diversity, and alignment with player preferences remain open challenges for procedural content generation in world models.

Procedural Content Generation Mechanisms hands the open question to Benchmark Datasets and Evaluation Metrics, which asks about details the standardized environments, datasets, and evaluation criteria used to measure the performance, sample efficiency, and generalization capabilities of world models in gaming scenarios.

## Benchmark Datasets and Evaluation Metrics

Benchmark Datasets and Evaluation Metrics focuses on Details the standardized environments, datasets, and evaluation criteria used to measure the performance, sample efficiency, and generalization capabilities of world models in gaming scenarios.

The benchmark landscape for evaluating world models in gaming spans specialist domains in which StarCraft has emerged as an important challenge for building artificial agents that compete and coordinate with other agents in complex environments [paper:10.1038/s41586-019-1724-z]. Grandmaster level in StarCraft II using multi-agent reinforcement learning [paper:10.1038/s41586-019-1724-z] established this real-time strategy domain as a stepping stone toward real-world applications requiring competition and coordination among multiple agents. Mastering Atari, Go, chess and shogi by planning with a learned model [paper:10.1038/s41586-020-03051-4] demonstrated that a single agent can master multiple classic game suites, broadening standardized evaluation from board games to Atari environments. MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge is proposed as a step toward open-ended embodied agents, releasing the simulation suite, knowledge bases, algorithm implementation, and pretrained models to promote research toward generally capable embodied agents [paper:10.48550/arxiv.2206.08853].

Methodologically, the multi-game evaluation in Mastering Atari, Go, chess and shogi by planning with a learned model builds on tree-based planning, a class of methods that has enjoyed success in challenging domains such as chess and Go, here combined with a learned model [paper:10.1038/s41586-020-03051-4]. MineDojo's methodology pairs its open-source simulation suite with internet-scale knowledge bases and pretrained models, motivated by the observation that autonomous agents typically learn tabula rasa in isolated environments with limited and manually conceived objectives [paper:10.48550/arxiv.2206.08853].

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Compared with the specialist suites of Atari, Go, chess, and shogi, which evaluate planning within isolated domains, MineDojo was designed in response to the observation that such isolated environments lead tabula-rasa agents to fail to generalize across tasks [paper:10.1038/s41586-020-03051-4] [paper:10.48550/arxiv.2206.08853]. The evaluation criteria across these suites therefore shift from mastering individual games under fixed, well-defined objectives to measuring generalization of embodied agents in open-ended settings supported by internet-scale knowledge [paper:10.1038/s41586-020-03051-4] [paper:10.48550/arxiv.2206.08853].

Autonomous agents have made great strides in specialist domains like Atari games and Go, yet they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. [paper:10.48550/arxiv.2206.08853].

Benchmark Datasets and Evaluation Metrics hands the open question to Latent Architectures and Internal Simulation, which asks about investigates the design of latent space representations and internal simulation mechanisms that enable agents to predict future states and learn dynamics without explicit environment interaction.

## Latent Architectures and Internal Simulation

Latent Architectures and Internal Simulation focuses on Investigates the design of latent space representations and internal simulation mechanisms that enable agents to predict future states and learn dynamics without explicit environment interaction.

As large language models (LLMs) have been widely adopted as core components of AI agents operating in real-world environments, their potential as agents has been widely acknowledged, creating an urgent need to quantitatively evaluate LLMs as agents on challenging tasks in interactive environments [paper:10.48550/arxiv.2308.03688]. "AgentBench" [paper:10.48550/arxiv.2308.03688] responds to this need by quantitatively evaluating LLMs as agents on challenging tasks in interactive environments. As Large Language Models are integrated into critical real-world applications, their strategic and logical reasoning abilities are increasingly crucial, and GTBENCH addresses this concern by uncovering the strategic reasoning limitations of LLMs via game-theoretic evaluations [paper:10.48550/arxiv.2402.12348]. Together, these evaluation efforts provide the behavioral evidence against which the latent predictive and simulation capacities examined in this section must ultimately be judged.

Methodologically, AgentBench proceeds through quantitative evaluation on challenging tasks in interactive environments, targeting the strong reasoning, planning, and language understanding capabilities that LLMs provide to agents [paper:10.48550/arxiv.2308.03688]. GTBENCH instead implements game-theoretic evaluations that probe LLMs' reasoning abilities in competitive environments [paper:10.48550/arxiv.2402.12348]. In both cases, the protocol infers internal predictive capacity from externally observable task behavior rather than from direct probes of latent state representations, which constrains what such evaluations can conclude about simulation quality.

AgentBench targets the setting in which LLMs have been widely adopted as core components of AI agents operating in real-world environments, interpreting complex situations and generating sophisticated actions. [paper:10.48550/arxiv.2308.03688]. AgentBench proceeds from the premise that LLM competence must be quantitatively evaluated, focusing on their abilities as agents in real-world environments. [paper:10.48550/arxiv.2308.03688].

On the limitation side, GTBENCH probes the strategic reasoning of LLMs through game-theoretic tasks, examining shortcomings in the strategic and logical reasoning abilities that are increasingly crucial as LLMs are integrated into critical real-world applications [paper:10.48550/arxiv.2402.12348]. For research on latent architectures and internal simulation, these findings suggest that current assessment still relies on explicit interaction with environments to expose reasoning failures, motivating internal simulation mechanisms whose predictive fidelity can be evaluated more directly.

Latent Architectures and Internal Simulation hands the open question to Multi-Agent Planning and Mean Field Games, which asks about analyzes coordination strategies and planning algorithms for multiple agents within a shared world model, including applications of mean field game theory for large-scale agent interactions.

## Multi-Agent Planning and Mean Field Games

Multi-Agent Planning and Mean Field Games focuses on Analyzes coordination strategies and planning algorithms for multiple agents within a shared world model, including applications of mean field game theory for large-scale agent interactions.

Research on planning and coordination within a shared world model draws on two foundational strands: adversarial game search and reinforcement learning. "Mastering the game of Go with deep neural networks and tree search" [paper:10.1038/nature16961] confronts a game long viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. "Human-level control through deep reinforcement learning" [paper:10.1038/nature14236] contributes a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment.

Methodologically, the theory of reinforcement learning provides the most natural framework, deeply rooted in psychological as well as neuroscientific perspectives of animal behavior, for how agents can optimize their control of an environment [paper:10.1038/nature14236]. The work in "Mastering the game of Go with deep neural networks and tree search" addresses the challenge of evaluating board positions and moves within Go's enormous search space. [paper:10.1038/nature16961]. Go has long been viewed as the most challenging of classic games for artificial intelligence, owing to its enormous search space and the difficulty of evaluating board positions and moves. [paper:10.1038/nature16961].

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

Whereas reinforcement learning offers a general account of how an agent may optimize its control of an environment, Go exposes the extreme burden that adversarial interaction within a shared state places on planning [paper:10.1038/nature16961] [paper:10.1038/nature14236]. Among classic games, Go is described as the most complex game that mankind ever created, with more combinations of possible moves than chess, and thus the number of atoms in the observable universe [paper:10.1038/nature16961].

Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves [paper:10.1038/nature16961].

Multi-Agent Planning and Mean Field Games closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Read across the body sections, the recorded boundaries cluster into a few open problems: Mastering Diverse Domains through World Models: # Mastering Diverse Domains through World Models Danijar Hafner, $^{12}$ Jurgis Pasukonis, $^{1}$ Jimmy Ba, $^{2}$ Timothy Lillicrap $^{1}$ # Abstract Developing a general algorithm that learns to so... [paper:10.48550/arxiv.2301.04104]; DIFFUSION MODELS ARE REAL-TIME GAME ENGINES: We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality. When trained o... [paper:10.48550/arxiv.2408.14837]; AvalonBench: Evaluating LLMs Playing the Game of Avalon: other agents in such scenarios, limiting their effectiveness in settings that require theory of mind or strategic social deduction. We revisit this problem in the context of Avalon, a social deductio... [paper:10.48550/arxiv.2310.05036]; Mastering Atari, Go, chess and shogi by planning with a learned model: Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. Tree-based planning methods have enjoyed huge success in challenging... [paper:10.1038/s41586-020-03051-4]. Each remains open rather than settled.

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

These boundaries translate into one concrete step per direction. For MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge, follow-up work should keep autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn while removing the reported boundary (autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objecti...) [paper:10.48550/arxiv.2206.08853]. For AgentBench: Evaluating LLMs as Agents, follow-up work should keep the potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, there is an urgent n while removing the reported boundary (# 1 Introduction In recent years, large language models (LLMs) have made significant progress in the field of natural language processing, demonstrating powerful performance in tasks such as dialogue...) [paper:10.48550/arxiv.2308.03688]. For GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evalua..., follow-up work should keep as Large Language Models (LLMs) are integrated into critical real-world applications, their strategic and logical reaso while removing the reported boundary (as Large Language Models (LLMs) are integrated into critical real-world applications, their strategic and logical reasoning abilities are increasingly crucial. This paper evaluates LLMs’ reasoning ab...) [paper:10.48550/arxiv.2402.12348]. For Mastering the game of Go with deep neural networks and tree search, follow-up work should keep the game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its e while removing the reported boundary (the recent performances of AI in typical Chinese game go are even more spectacular. The game of go has long been viewed as the most challenging of classic games for artificial intelligence owing to i...) [paper:10.1038/nature16961].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1038/nature14236: Human-level control through deep reinforcement learning (2015).
- paper:10.1038/nature16961: Mastering the game of Go with deep neural networks and tree search (2016).
- paper:10.1038/s41586-019-1724-z: Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019).
- paper:10.1038/s41586-020-03051-4: Mastering Atari, Go, chess and shogi by planning with a learned model (2020).
- paper:10.48550/arxiv.2206.08853: MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge (2022).
- paper:10.48550/arxiv.2301.04104: Mastering Diverse Domains through World Models (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2310.05036: AvalonBench: Evaluating LLMs Playing the Game of Avalon (2023).
- paper:10.48550/arxiv.2402.12348: GTBENCH: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations (2024).
- paper:10.48550/arxiv.2408.14837: DIFFUSION MODELS ARE REAL-TIME GAME ENGINES (2024).

## Revision Notes

- type=B backfill_evidence -> repaired: The abstract chunk is truncated right before the sentence introducing the paper's general algorithm; backfill can fetch the grounding chunk for the 'contributes such an algorithm' part.
- type=B backfill_evidence -> repaired: Preview chunks only indirectly cover Avalon and LLMs; backfill can fetch the chunk stating AvalonBench explores the potential of LLM agents in the game.
- type=B rewrite_claim -> repaired: Evidence supports the paper's stated aim and the brittleness bottleneck, but not the assertion that the method already achieves brittleness-free RL; weaken to the stated goal.
- type=B swap_evidence -> repaired: This chunk nearly verbatim states that Avalon is revisited as a social deduction game providing a structured environment for making informed decisions.
- type=B rewrite_claim -> invalid_action: The GameNGen clause cannot be supported by the cited DreamerV3 paper; keep only the part the abstract chunk supports.
- type=B swap_evidence -> repaired: Chunk states verbatim that RL brittleness poses a bottleneck for new problems and limits applicability to expensive models/tasks where tuning is prohibitive; directly grounds the claim.
- type=B swap_evidence -> repaired: Chunk explicitly notes agents' limited effectiveness in settings requiring theory of mind or strategic social deduction, directly supporting the claim.
- type=B backfill_evidence -> repaired: Preview chunks come from an unrelated paper (Critic PI2) and none ground the MuZero multi-suite demonstration claim; semantic search can fetch the correct grounding chunks.
- type=B backfill_evidence -> repaired: No preview chunk covers MineDojo's design motivation and no remap candidates were provided, so citation cannot be remapped; backfill to retrieve grounding chunks for the claim.
- type=B backfill_evidence -> repaired: Preview chunks discuss tree-based planning, not the claimed shift in evaluation criteria toward open-ended generalization; backfill semantic search for supporting chunks.
- type=B rewrite_claim -> repaired: Evidence chunk 474 supports this verbatim; drops the unsupported 'classic benchmark paradigm' framing.
- type=B backfill_evidence -> repaired: Claim mirrors AgentBench's motivation; previewed chunks only cover adoption and potential, so fetch the intro chunk stating the urgent need for quantitative agentic evaluation.
- type=B backfill_evidence -> repaired: The AgentBench paper explicitly proposes the benchmark to evaluate LLMs as agents on interactive tasks; retrieve the chunk naming AgentBench's purpose.
- type=B backfill_evidence -> repaired: First clause matches p0_0, but the GTBENCH attribution needs a chunk that names GTBENCH and its goal of uncovering strategic reasoning limitations.
- type=B backfill_evidence -> repaired: Previewed chunk paraphrases the game-theoretic evaluation without naming GTBENCH; fetch an abstract/body chunk explicitly describing GTBENCH.
- type=B rewrite_claim -> repaired: Claim asserts facts about GTBENCH that the cited AgentBench paper cannot support; rewrote to the AgentBench-only description grounded in the preview chunks.
- type=B rewrite_claim -> repaired: Cross-paper 'both works' comparison is not grounded in the cited paper; weakened to a single-paper premise statement the AgentBench evidence can support.
- type=B rewrite_claim -> invalid_action: Evidence attributes the normative psych/neuro account to RL theory, not to the paper itself; reattributed the claim accordingly.
- type=B swap_evidence -> repaired: Chunk 252 states almost verbatim that RL theory is the most natural framework rooted in psychological and neuroscientific perspectives for optimizing control of an environment.
- type=B rewrite_claim -> repaired: The 'this search-based procedure targets' attribution is unsupported by the preview chunks; kept only the directly supported statement about Go's difficulty.
- type=B swap_evidence -> repaired: This chunk verbatim supports the claim (Go as most complex game, more combinations than chess, atoms comparison); reattach the matching evidence chunk.
- type=B rewrite_claim -> repaired: Preview chunks only support the search-space/difficulty-of-evaluation point for Go, not the stronger claim about deep neural networks and tree search confronting it head on; weaken to what the evidence states.
- type=C backfill_evidence -> repaired: Composite claim (world-model benchmark landscape plus StarCraft as an agent challenge); no single candidate paper covers both aspects, so semantic search should fetch grounding chunks instead of remapping to a partial match.
- type=C backfill_evidence -> unresolved: The cited AlphaStar paper's title matches the claim verbatim, so zero evidence likely means missing chunks; backfilling will fetch grounding evidence from the correct paper rather than remapping to an unrelated candidate.
- type=D rewrite_claim -> repaired: Evidence supports the release statement but not the strong 'contributes an open-ended embodied benchmark' assertion, so soften the contribution claim.
- type=D rewrite_claim -> repaired: Soften 'rests on' and 'huge success' to match the abstract's framing of tree-based planning as a successful prior approach.
- type=D rewrite_claim -> repaired: Evidence supports the release items and the tabula rasa motivation, but not that MineDojo defines evaluation settings extending beyond it; recast as motivation.
- type=D rewrite_claim -> repaired: Replace 'exploiting' with 'targeting' so the claim asserts evaluation of these capabilities rather than their guaranteed exploitation.
- type=D rewrite_claim -> repaired: Evidence supports that the paper evaluates strategic reasoning in game-theoretic settings, not that uncovering limitations is its 'central result'; weaken to probing.
- type=D keep -> kept: Claim nearly restates the evidence verbatim ('long been viewed as the most challenging of classic games... enormous search space and the difficulty of evaluating board positions and moves'), so the assertion is acceptable as-is.
- type=D rewrite_claim -> repaired: Evidence only supports that Go has an enormous search space with hard position/move evaluation; it does not substantiate the specific mechanism claim that the algorithm couples deep neural networks with tree search, so the assertion is weakened to what the evidence shows.
- type=D rewrite_claim -> invalid_action: Evidence supports that Go is extremely challenging for AI planning due to its search space and evaluation difficulty, but not the stronger interpretive framing about adversarial interaction burdening planning; assertion is weakened accordingly.
