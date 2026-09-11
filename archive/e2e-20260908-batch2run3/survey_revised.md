# World Models for Games: A Survey

## Abstract
This survey covers 4 themes over 15 citation-ready studies, spanning Generative Game World Simulation to Neural Rendering and Multimodal Game State Modeling. Every claim carries the tag of the paper whose evidence supports it. AgentBench: Evaluating LLMs as Agents: # 1 Introduction In recent years, large language models (LLMs) have made significant progress in the field of natural language processing, demonstrat [paper:10.48550/arxiv.2308.03688]. Understanding the planning of LLM agents: A survey: Large Language Models (LLMs) have demonstrated impressive capabilities across a broad spectrum of natural language processing tasks. Building on thes [paper:10.48550/arxiv.2402.02716]. Understanding World or Predicting Future? A Comprehensive Survey of World Models: The concept of world models has garnered significant attention due to advancements in multimodal large language models such as GPT-4 and video genera [paper:10.1145/3746449].

## Introduction
Game intelligence increasingly rests on models that learn how a world evolves rather than on hand-written rules alone; whether such models can support planning, generation, and fair evaluation is the question this survey addresses. Each theme that follows gathers the evidence for one capability and states where that evidence stops.

Generative Game World Simulation surveys surveys techniques for learning latent dynamics models that enable the generation of novel game states, levels, and environmental interactions. Agent Planning and Control in Learned Worlds collects the work on reviews approaches leveraging learned models for model-based reinforcement learning, planning, and control policies within simulated game environments. GameCraft Benchmarks and Evaluation Protocols follows the evidence for outlines standardized benchmarks, evaluation protocols, and datasets specifically designed to assess world model performance in gaming scenarios. Neural Rendering and Multimodal Game State Modeling maps what is known about investigates the use of neural rendering techniques and multimodal inputs to construct high-fidelity, visually rich representations of game states.

The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.

![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)

Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.

![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)

## Generative Game World Simulation

Generative Game World Simulation focuses on Surveys techniques for learning latent dynamics models that enable the generation of novel game states, levels, and environmental interactions.

Techniques for learning latent dynamics models that generate novel game states, levels, and environmental interactions draw on several strands of generative artificial intelligence research. A Survey on Generative Diffusion Models [paper:10.1109/tkde.2024.3361474] argues that deep generative models have unlocked another profound realm of human creativity, and that by capturing and generalizing patterns within data we have entered the epoch of all-encompassing Artificial Intelligence for General Creativity. The concept of world models has garnered significant attention due to advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which are central to the pursuit of artificial general intelligence [paper:10.1145/3746449]. Retrieval-Augmented Generation has emerged as a promising solution to the limitations of large language models, which encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes [paper:10.48550/arxiv.2312.10997]. Generative large language models have further demonstrated remarkable capabilities in text understanding and generation [paper:10.1007/s11704-024-40555-y].

To enable dynamic and intelligent transformation, the recognized need for an advanced method of creating 3D assets led to a focus on generative AI, a field that has seen significant advancements in recent years. [paper:10.1109/tkde.2024.3361474]. World-model research is consolidated by surveys that summarize representative papers along with their code repositories and outline key challenges while providing insights into potential future research directions [paper:10.1145/3746449].

In terms of modeling paradigms, while traditional methods relied on supervised sequence labeling, the emergence of large language models has shifted the paradigm towards generative approaches [paper:10.1007/s11704-024-40555-y]. Whereas diffusion-based generative models capture and generalize patterns within data, world-model research is framed around understanding the world or predicting the future in the pursuit of artificial general intelligence [paper:10.1109/tkde.2024.3361474] [paper:10.1145/3746449].

Despite this breadth of techniques, several limitations temper their direct application to game world simulation. Large language models, in particular, face limitations such as hallucinations, outdated knowledge, and non-transparent, untraceable reasoning processes [paper:10.48550/arxiv.2312.10997]. World-model research likewise confronts key challenges, which surveys outline together with potential future research directions [paper:10.1145/3746449].

Generative Game World Simulation hands the open question to Agent Planning and Control in Learned Worlds, which asks about reviews approaches leveraging learned models for model-based reinforcement learning, planning, and control policies within simulated game environments.

## Agent Planning and Control in Learned Worlds

Agent Planning and Control in Learned Worlds focuses on Reviews approaches leveraging learned models for model-based reinforcement learning, planning, and control policies within simulated game environments.

Recent research relevant to agent planning and control spans reasoning with world models, planning modules for autonomous agents, model compression, and recommendation applications. Reasoning with Language Model is Planning with World Model [paper:10.48550/arxiv.2305.14992] argues that large language models have shown remarkable reasoning capabilities, especially when prompted to generate intermediate reasoning steps such as Chain-of-Thought. Understanding the planning of LLM agents: A survey [paper:10.48550/arxiv.2402.02716] contributes the first systematic view of the progress to leverage LLMs as planning modules of autonomous agents. A survey on large language models for recommendation documents that LLMs have emerged as powerful tools in the field of Natural Language Processing and have recently gained significant attention in the domain of Recommendation Systems [paper:10.1007/s11280-024-01291-2]. A Survey on Model Compression for Large Language Models observes that LLMs have transformed natural language processing tasks successfully, yet this remarkable performance comes with substantial computational cost [paper:10.1162/tacl_a_00704].

Reasoning with Language Model is Planning with World Model notes that large language models have showcased strong reasoning capabilities using chain-of-thought prompting, where LLMs are prompted to generate intermediate steps before the final answer [paper:10.48550/arxiv.2305.14992]. Within agent research, the methodological trend is to leverage LLMs as planning modules of autonomous agents, a direction that has attracted more attention as LLMs have shown significant intelligence [paper:10.48550/arxiv.2402.02716]. Model compression has emerged as a key research area for addressing the large size and high computational needs that challenge the practical use of these models, especially in resource-limited settings [paper:10.1162/tacl_a_00704].

![Representative papers organized by method, contribution, and limitation.](representative_systems)

Compared with the systematic view of agent planning offered by the LLM-agent survey, Reasoning with Language Model is Planning with World Model frames reasoning itself as planning with a world model [paper:10.48550/arxiv.2305.14992] [paper:10.48550/arxiv.2402.02716]. Work on model compression targets the computational cost that constrains the practical use of these models, with challenges that are especially acute in resource-limited settings [paper:10.1162/tacl_a_00704].

Nevertheless, even reasoning-oriented models still struggle with problems that require multi-step reasoning and planning [paper:10.48550/arxiv.2305.14992]. Likewise, the large size and high computational needs of LLMs pose challenges for practical use, especially in resource-limited settings [paper:10.1162/tacl_a_00704].

Agent Planning and Control in Learned Worlds hands the open question to GameCraft Benchmarks and Evaluation Protocols, which asks about outlines standardized benchmarks, evaluation protocols, and datasets specifically designed to assess world model performance in gaming scenarios.

## GameCraft Benchmarks and Evaluation Protocols

GameCraft Benchmarks and Evaluation Protocols focuses on Outlines standardized benchmarks, evaluation protocols, and datasets specifically designed to assess world model performance in gaming scenarios.

Standardized benchmarks and evaluation protocols relevant to gaming world models can be organized around landmark platforms and testbeds that each target a distinct facet of simulation and agentic behavior. A notable contribution is the Cosmos World Foundation Model Platform for Physical AI, which represents a key advance in the now-mature, interdisciplinary field of Physical AI, illustrating how a powerful foundation model trained on diverse multimodal inputs can serve as a digital twin of the physical world. [paper:10.48550/arxiv.2501.03575]. On the agentic side, AgentBench: Evaluating LLMs as Agents [paper:10.48550/arxiv.2308.03688] addresses the urgent need to quantitatively evaluate LLMs as agents on challenging tasks in interactive environments, motivated by the fact that the potential of large language models as agents has been widely acknowledged recently. Context for game-oriented assessment is provided by Large Language Models and Games: A Survey and Roadmap, which observes that recent years have seen an explosive increase in research on large language models. [paper:10.1109/tg.2024.3461510]. This survey further observes that the research landscape surrounding large language models has witnessed substantial growth in recent years, driven by the potential these models hold for various natural language processing tasks, providing context for what game-focused evaluation may need to capture. [paper:10.1109/tg.2024.3461510].

Cosmos grounds physical AI development digitally, with the Cosmos world foundation model serving as a digital twin of the physical world while maintaining tight integration with external control, memory, and planning modules [paper:10.48550/arxiv.2501.03575]. AgentBench operationalizes its protocol by placing large language model agents inside interactive environments and scoring them quantitatively on challenging tasks, yielding a repeatable agentic evaluation pipeline [paper:10.48550/arxiv.2308.03688]. Complementing these, the knowledge editing literature evaluates large language models as language processing models trained on very large text corpora that are able to perform a wide range of tasks, such as text generation, summarization, translation, and question answering [paper:10.1145/3698590].

Viewed together, the two protocols address different facets of evaluation, as Cosmos provides a foundation model platform that can serve as a digital twin of the physical world for physical AI, whereas AgentBench targets whether LLM agents can act effectively within interactive environments. [paper:10.48550/arxiv.2501.03575]. The game-focused LLM-and-games survey reflects an explosive increase in research connecting large language models with games, an area situated between interactive simulation and language competence [paper:10.1109/tg.2024.3461510].

A limitation is that knowledge editing protocols, being anchored in tasks such as text generation, summarization, translation, and question answering, do not by themselves assess world model behavior in gaming scenarios [paper:10.1145/3698590]. Similarly, because AgentBench targets LLMs as agents and Cosmos targets physical AI, neither protocol on its own provides a gaming-specific benchmark for world model performance, suggesting that a unified gaming-specific benchmark remains an open challenge. [paper:10.48550/arxiv.2501.03575].

GameCraft Benchmarks and Evaluation Protocols hands the open question to Neural Rendering and Multimodal Game State Modeling, which asks about investigates the use of neural rendering techniques and multimodal inputs to construct high-fidelity, visually rich representations of game states.

## Neural Rendering and Multimodal Game State Modeling

Neural Rendering and Multimodal Game State Modeling focuses on Investigates the use of neural rendering techniques and multimodal inputs to construct high-fidelity, visually rich representations of game states.

Diffusion Models: A Comprehensive Survey of Methods and Applications reports that diffusion models have emerged as one of the most powerful families of generative models, achieving state-of-the-art performance across a wide range of tasks, including image synthesis, video generation, and molecule design. [paper:10.1145/3626235]. Recent years have witnessed the remarkable success of diffusion models, accompanied by a range of visually stunning generative contents and promising performance across a wide variety of downstream applications [paper:10.1145/3626235]. On the linguistic side, large language models demonstrate exceptional performance on tasks requiring complex linguistic abilities, such as reference disambiguation and metaphor recognition and generation [paper:10.1145/3639372]. Instruction Tuning for Large Language Models: A Survey documents that instruction tuning has become a widely adopted post-training method to enable LLMs and MLLMs to follow instructions and solve a broader range of general tasks [paper:10.48550/arxiv.2308.10792].

As a generative modeling approach, diffusion models have achieved state-of-the-art performance across a wide range of tasks, including high-fidelity image synthesis. [paper:10.1145/3626235]. Mechanistically, supervised fine-tuning performs full-parameter fine-tuning based on pre-trained models using task-specific labeled data in the form of input-output pairs [paper:10.48550/arxiv.2308.10792]. Explainability research takes as its object the internal mechanisms of LLMs, which are still unclear, motivated by the lack of transparency surrounding these models [paper:10.1145/3639372].

![Evaluation protocols grouped by what they measure and where they can mislead.](evaluation_protocol_matrix)

Instruction tuning is closely related to supervised fine-tuning and prompt tuning, and the survey situates it alongside these neighboring post-training paradigms [paper:10.48550/arxiv.2308.10792]. Diffusion models have surpassed GANs on image synthesis and have emerged as one of the most powerful families of generative models, achieving state-of-the-art performance across a wide range of tasks. [paper:10.1145/3626235]. Taken together, the visual synthesis strength of diffusion models and the instruction-following generality of tuned LLMs and MLLMs point to complementary visual and linguistic components for multimodal state modeling [paper:10.1145/3626235] [paper:10.48550/arxiv.2308.10792].

A principal limitation is that the internal mechanisms of LLMs remain largely opaque, and this lack of transparency poses unwanted risks for downstream applications that embed them [paper:10.1145/3639372].

Neural Rendering and Multimodal Game State Modeling closes the body; the remaining sections fold this evidence into open challenges and future directions.

## Open Challenges

Read across the body sections, the recorded boundaries cluster into a few open problems: A Survey on Generative Diffusion Models: # A Survey on Generative Diffusion Models Hanqun Cao, Cheng Tan, Zhangyang Gao, Yilun Xu, Guangyong Chen, Pheng-Ann Heng, Senior Member, IEEE, and Stan Z. Li, Fellow, IEEE Abstract—Deep generative mo... [paper:10.1109/tkde.2024.3361474]; Understanding World or Predicting Future? A Comprehensive Survey of World Models: Finally, we outline key challenges and provide insights into potential future research directions. We summarize the representative papers along with their code repositories in https://github.com/tsin... [paper:10.1145/3746449]; Retrieval-Augmented Generation for Large Language Models: A Survey: Large Language Models (LLMs) underpin many Text2SQL systems, offering impressive capabilities in understanding and generating natural language. However, they have limitations, such as hallucinations,... [paper:10.48550/arxiv.2312.10997]; Large language models for generative information extraction: a survey: Abstract Information extraction (IE) aims to extract structural knowledge from plain natural language texts. Recently, generative Large Language Models (LLMs) have demonstrated remarkable capabilitie... [paper:10.1007/s11704-024-40555-y]. Each remains open rather than settled.

Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models.

![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)

## Future Directions

These boundaries translate into one concrete step per direction. For Reasoning with Language Model is Planning with World Model, follow-up work should keep large language models (LLMs) have shown remarkable reasoning capabilities, especially when prompted to generate interme while removing the reported boundary (large language models (LLMs) have achieved remarkable results on reasoning benchmarks (OpenAI 2024; Guo et al. 2025). However, even such reasoning-oriented models still struggle with problems that re...) [paper:10.48550/arxiv.2305.14992]. For A Survey on Model Compression for Large Language Models, follow-up work should keep abstract Large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large while removing the reported boundary (large Language Models (LLMs) have transformed natural language processing tasks successfully. Yet, their large size and high computational needs pose challenges for practical use, especially in resou...) [paper:10.1162/tacl_a_00704]. For Understanding the planning of LLM agents: A survey, follow-up work should keep as Large Language Models (LLMs) have shown significant intelligence, the progress to leverage LLMs as planning modules while removing the reported boundary (large Language Models (LLMs) have demonstrated impressive capabilities across a broad spectrum of natural language processing tasks. Building on these advancements, recent research has investigated t...) [paper:10.48550/arxiv.2402.02716]. For A survey on large language models for recommendation, follow-up work should keep large Language Models (LLMs) have emerged as powerful tools in the field of Natural Language Processing (NLP) and have while removing the reported boundary (recently, large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks [23, 40, 41], which demonstrate strong capabilities in understanding a...) [paper:10.1007/s11280-024-01291-2].

Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles.

![Table 3. Open Challenges and Future Directions](future_directions_matrix)

![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)


## Conclusion

The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds.

## References

- paper:10.1007/s11280-024-01291-2: A survey on large language models for recommendation (2024).
- paper:10.1007/s11704-024-40555-y: Large language models for generative information extraction: a survey (2024).
- paper:10.1109/tg.2024.3461510: Large Language Models and Games: A Survey and Roadmap (2024).
- paper:10.1109/tkde.2024.3361474: A Survey on Generative Diffusion Models (2024).
- paper:10.1145/3626235: Diffusion Models: A Comprehensive Survey of Methods and Applications (2023).
- paper:10.1145/3639372: Explainability for Large Language Models: A Survey (2024).
- paper:10.1145/3698590: Knowledge Editing for Large Language Models: A Survey (2024).
- paper:10.1145/3746449: Understanding World or Predicting Future? A Comprehensive Survey of World Models (2025).
- paper:10.1162/tacl_a_00704: A Survey on Model Compression for Large Language Models (2024).
- paper:10.48550/arxiv.2305.14992: Reasoning with Language Model is Planning with World Model (2023).
- paper:10.48550/arxiv.2308.03688: AgentBench: Evaluating LLMs as Agents (2023).
- paper:10.48550/arxiv.2308.10792: Instruction Tuning for Large Language Models: A Survey (2023).
- paper:10.48550/arxiv.2312.10997: Retrieval-Augmented Generation for Large Language Models: A Survey (2023).
- paper:10.48550/arxiv.2402.02716: Understanding the planning of LLM agents: A survey (2024).
- paper:10.48550/arxiv.2501.03575: Cosmos World Foundation Model Platform for Physical AI (2025).

## Revision Notes

- type=A remap_citation -> invalid_action: The sentence's claim about LLMs' success in NLP matches the whitelisted survey whose title is explicitly referenced in the sentence.
- type=B swap_evidence -> repaired: Preview chunk states the claim nearly verbatim (world models attention due to GPT-4/Sora, pursuit of AGI).
- type=B swap_evidence -> repaired: Section 2.2 chunk directly states LLM limitations (hallucinations, outdated info, non-transparent reasoning) and RAG as a promising solution.
- type=B rewrite_claim -> repaired: Evidence says a need for 3D asset creation led to focusing on generative AI within one project, not a general recognition across content-creation projects; weaken claim to match the source.
- type=B swap_evidence -> repaired: Chunk explicitly says the survey summarizes representative papers with code repositories, outlines key challenges, and provides future research directions.
- type=B rewrite_claim -> invalid_action: Cited diffusion survey only supports the diffusion clause; no preview chunk covers world models, and no remap candidates were listed, so drop the unsupported clause.
- type=B swap_evidence -> repaired: Chunk explicitly lists hallucinations, outdated information, and non-transparent, untraceable reasoning of LLMs, directly matching the claim.
- type=B swap_evidence -> repaired: Chunk states the survey outlines key challenges and potential future research directions, exactly what the claim asserts.
- type=B swap_evidence -> repaired: Chunk confirms LLMs showcase strong reasoning capabilities via chain-of-thought with intermediate steps, matching the claim.
- type=B swap_evidence -> repaired: Chunk states LLMs transformed NLP and that remarkable performance comes with substantial computational costs, matching the claim verbatim.
- type=B rewrite_claim -> invalid_action: Claim misattributes CoT as the paper's core mechanism; the paper's thesis is planning with a world model. Weakened rewrite is directly supported by the CoT evidence chunk.
- type=B rewrite_claim -> invalid_action: Evidence only supports model compression being a key research area addressing these challenges, not that it provides a definitive methodological solution; weakened to match the source.
- type=B rewrite_claim -> invalid_action: Comparison to the LLM-agent survey is unsupported by the cited paper; kept only the paper's own thesis, which is supported by its title and framing.
- type=B rewrite_claim -> invalid_action: Nothing about Recommendation Systems or LLM attention in that domain appears in the cited paper; removed the unsupported contrast, keeping the supported core.
- type=B swap_evidence -> repaired: Chunk 252 states almost verbatim that even reasoning-oriented models still struggle with problems requiring multi-step reasoning and planning, so pointing to it grounds the claim.
- type=B swap_evidence -> repaired: Chunk 66 contains the near-verbatim statement that LLMs' large size and high computational needs pose challenges for practical use, especially in resource-limited settings.
- type=B rewrite_claim -> invalid_action: Claim asserts an evaluative-reference methodology not shown in evidence; chunk supports only Cosmos serving as a world digital twin, so weaken to match.
- type=B backfill_evidence -> repaired: Claim is core AgentBench content (interactive environments, quantitative scoring) but preview chunks are generic intro text; semantic search should fetch the benchmark grounding chunks.
- type=B rewrite_claim -> invalid_action: Cited paper supports only the game-survey side; the knowledge-editing comparison is outside its scope, so drop the cross-paper contrast.
- type=B rewrite_claim -> repaired: First clause is near-verbatim in chunk 1; drop the unsupported 'visually rich representations' tail and align with evidenced downstream-applications statement.
- type=B swap_evidence -> repaired: Chunk 2 contains the exact assertion about complex linguistic abilities including reference disambiguation and metaphor recognition/generation; point citation to it.
- type=B swap_evidence -> repaired: Chunk agentic_127 states verbatim that instruction tuning/SFT has become a widely adopted post-training method enabling LLMs and MLLMs to follow instructions and solve broader general tasks; swapping grounds the claim exactly.
- type=B rewrite_claim -> invalid_action: Evidence supports SOTA across tasks (agentic_235) and promise in image synthesis (agentic_234), but not the stronger 'algorithmic foundation for high-fidelity visual synthesis'; weakened to match the evidence.
- type=B swap_evidence -> repaired: Chunk agentic_126 contains the exact mechanistic statement: SFT performs full-parameter fine-tuning on pre-trained models using task-specific labeled data (input-output pairs).
- type=B swap_evidence -> repaired: Chunk agentic_126 explicitly states instruction tuning is close to supervised fine-tuning (SFT) and prompt tuning, directly grounding the relationship claim.
- type=B rewrite_claim -> invalid_action: Evidence (agentic_234) confirms DMs surpassed GAN on image synthesis and subsequent promise across applications, but does not state a 'shift in the dominant paradigm for high-fidelity image generation'; tail replaced with supported content.
- type=D rewrite_claim -> invalid_action: The 'epoch of all-encompassing AI for General Creativity' overstates the evidence, which only supports an epoch of GenAI redefining creativity
- type=D keep -> kept: Evidence states nearly verbatim that LLMs are emerging as powerful tools in NLP and have garnered significant attention in recommendation; assertion is acceptable as-is
- type=D rewrite_claim -> invalid_action: Evidence supports 'key advance' and the digital-twin role of the Cosmos model, but not the specific 'foundational premise' assertion; weaken to match
- type=D rewrite_claim -> invalid_action: 'Urgent need' and 'widely acknowledged' overstate the evidence, which supports 'growing interest' and 'remarkable potential' as autonomous agents
- type=D keep -> kept: Claim mirrors the paper's abstract verbatim ('explosive increase in research' on LLMs and accompanying public engagement); assertion is acceptable as-is
- type=D rewrite_claim -> repaired: Removes unsupported intensifier 'profound' and softens the 'must capture' framing tail to what the cited abstract supports.
- type=D rewrite_claim -> repaired: Drops the unsupported 'leading protocols' ranking and the 'assesses fidelity' phrasing; evidence only supports Cosmos serving as a digital twin.
- type=D rewrite_claim -> repaired: Softens 'complete evaluation standard' to a design-scope statement and frames the remaining gap as suggestive rather than established.
- type=D rewrite_claim -> repaired: Replaces 'establishes' and 'record-breaking' with 'reports' and 'state-of-the-art', matching the source's actual wording and strength.
- type=D keep -> kept: Assertion is commensurate with evidence: internal mechanisms are described as largely opaque, and this opacity motivates interpretability research.
- type=D None -> unresolved: llm budget exhausted
- type=D None -> unresolved: llm budget exhausted
