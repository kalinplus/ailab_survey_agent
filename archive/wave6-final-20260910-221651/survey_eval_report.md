# Survey Evaluation Report

## Block 1 — Deterministic Profile (L0, no LLM)

**Freshness** — citation year median 2023.0 vs corpus 2023.0 (offset 0.0); recent-two-year share: citations 0.0 vs corpus 0.179; unresolved citation years 0 of 11 cited ids.

**Structure** — 11 sections, prose-length CV 0.477, 21 citations (mean 1.909/section); zero-citation sections: Abstract, Introduction, Open Challenges, Future Directions; figure embeds 8/86 in bank, table embeds 0/45 in bank.

**Redundancy** — section-pair similarity mean 0.742 / max 0.891 over 55 pairs (mode embedding); top pairs: Procedural Generation and Scripting × Conclusion = 0.891; Abstract × Introduction = 0.884; Abstract × Procedural Generation and Scripting = 0.866.

## Block 2 — Faithfulness (NLI)

Headline: overall unsupported rate **0.021** = (unsupported citation pairs + unsupported uncited claims) / all checked claims.

Citation quality (AutoSurvey-style, pair level):

| metric | value |
|---|---|
| Citation recall | 1.0 |
| Citation precision | 1.0 |
| Citation coverage | 0.362 |

Sentences 58 / cited 21 / pairs 21 (supported 21, weak 0, unsupported 0).

Uncited claims: 26 found, 26 verified (searched 26, cache hits 0, truncated 0); supported 9, weak 16, unsupported 1; support rate **0.346**.

Unsupported uncited examples (max 5):

- Game-Oriented and Cross-Domain Applications closes the body; the remaining secti

## Block 3 — Coverage

Gold-free reference quality (primary):

| metric | value |
|---|---|
| Seed-bib recall | 0.348 (8/23 union entries from 7 seed surveys) |
| In-seed-bib rate | 0.119 (8/67 corpus papers) |
| Canonical hit@10 | 0.2 (2/10) |
| Canonical hit rate | 0.3 (6/20) |
| Corpus diversity | 67 papers over 6 aspects, 36 venues, years 2014–2026 (median 2023.0), citations p25/p50/p75 9.5/52.0/176.0 |

Canonical hits: MuZero, AlphaStar, DreamerV3, Genie: Generative interactive environments, GameNGen, GameFactory: Creating new games with generative interactive videos
Canonical missing: World Models, PlaNet, Dreamer, MBPO, NGU, DreamerV2, Agent57: Outperforming the Atari Human Benchmark, Minedojo: Building open-ended embodied agents with internet-scale knowledge, Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos, Generative agents: Interactive simulacra of human behavior, Gaia-1: A generative world model for autonomous driving, UniSim, Diamond: Diffusion for world modeling: Visual details matter in Atari, Cosmos world foundation model platform for physical ai
Seed-bib missed (15/23): agentbench_2023, alphago_2016, alphazero_2017, Cradle-to-cradle business model tool: Innovating circular business models for startups, dqn_2015, DreamerV3: Mastering Diverse Domains through World Models, gamenegen_2024, genie2_2024, genie3_2025, matrix_game_2024 (+5 more)

| aspect | papers | years | venues | citations p25/p50/p75 |
|---|---|---|---|---|
| Procedural Generation and Scripting | 48 | 2014–2025 | 30 | 7.5/33.5/161.5 |
| Latent Representations and World Models | 13 | 2020–2026 | 7 | 8.0/62.0/166.0 |
| Generative Simulation and Interactive Control | 3 | 2024–2025 | 3 | 29.5/30.0/120.0 |
| Multi-Agent and Mean-Field Simulation | 1 | 2019–2019 | 1 | 4276.0/4276.0/4276.0 |
| Benchmarks and Evaluation | 2 | 2023–2024 | 2 | 99.75/125.5/151.25 |
| Game-Oriented and Cross-Domain Applications | 0 | n/a–n/a | 0 | n/a |

Corpus-grounded coverage **1.0** (6/6 categories at threshold 0.5); paper-weighted citation depth 0.164; corpus utilization 0.164.

| category | papers | best sim | covered | citation depth |
|---|---|---|---|---|
| Procedural Generation and Scripting | 48 | 0.826 | yes | 0.104 |
| Latent Representations and World Models | 13 | 0.933 | yes | 0.308 |
| Generative Simulation and Interactive Control | 3 | 0.871 | yes | 0.333 |
| Multi-Agent and Mean-Field Simulation | 1 | 0.883 | yes | 1.0 |
| Benchmarks and Evaluation | 2 | 0.813 | yes | 0.0 |
| Game-Oriented and Cross-Domain Applications | 0 | 0.839 | yes | 0.0 |

Gold survey control (secondary diagnostic — caveat: a single gold survey's reference list is not ground truth, and its size caps recall for a breadth-first corpus): **Understanding World or Predicting Future? A Comprehensive Review of World Models** (338 refs). Reference recall **0.027** (exact 8 + substring 1 + fuzzy 0); system-in-gold rate 0.119 over 67 corpus papers.

Matched:

- Diffusion models are real-time game engines
- Drivedreamer: Towards real-world-driven world models for autonomous driving
- Gamefactory: Creating new games with generative interactive videos
- Genie: Generative interactive environments
- Mastering the game of go with deep neural networks and tree search
- Procthor: Large-scale embodied ai using procedural generation
- Trafficbots: Towards world models for autonomous driving simulation and motion prediction
- World models for autonomous driving: An initial survey
- Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving, 2023

## Block 4 — Academic Value (LLM judge, DeepSurvey rubric)

Overall **3.0** over 3 judged dimensions (whole document, one call per dimension).

- informational_value: **3/5** — The survey outlines core themes and cites representative papers but exhibits uneven content depth, includes irrelevant references (e.g., facial expression recognition), and admits insufficient evidence for critical synthesis in challenges and future directions.
- scholarly_communication_value: **3/5** — The survey features a clear thematic structure and logical flow but exhibits gaps in academic expression due to repetitive content, placeholder image markers, and weak synthesis in the conclusion.
- research_guidance_value: **3/5** — The survey includes critical remarks on limitations and trade-offs but explicitly admits in its Open Challenges and Future Directions sections that it lacks sharper statements or specific actionable proposals for new researchers.

## Meta

- generated_at: 2026-09-10T14:50:20+00:00
- survey: /Users/kalin/github/AILabMacroHard/archive/wave6-final-20260910-221651/output/survey.md
- nli_model: cross-encoder/nli-deberta-v3-base@cpu
- judge_model: intern-s2-preview
- layers: L0 deterministic / L1 citation quality / L1.5 uncited claims / L2' corpus coverage + gold-free reference quality (seed-bib recall, canonical hit@N, corpus diversity) / L3 academic value
- note: Layer 1 is pair-level (every citation per sentence); evaluator shares the NLIVerifier infrastructure with the generation-side claim mapper but runs independently on the final artifact. Gold reference recall is a secondary diagnostic only: one gold survey's reference list is not ground truth and its size caps recall for a breadth-first corpus.
