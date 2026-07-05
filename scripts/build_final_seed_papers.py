"""Build final presentation seed data for the World Models / GameCraft demo."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = [
    {
        "category_id": "cat_world_models",
        "category_name": "Generative Game World Simulation",
        "description": "World models that learn compact dynamics for game-like or interactive environments.",
    },
    {
        "category_id": "cat_planning",
        "category_name": "Planning and Control with Learned Models",
        "description": "Agents that use learned dynamics for planning, control, or policy learning.",
    },
    {
        "category_id": "cat_neural_engines",
        "category_name": "Neural Game Engines and Interactive Environments",
        "description": "Generative systems that synthesize controllable interactive environments.",
    },
    {
        "category_id": "cat_game_agents",
        "category_name": "Game Agents and Multi-Agent Reinforcement Learning",
        "description": "Large-scale agents trained in complex games and multi-agent settings.",
    },
    {
        "category_id": "cat_simulators",
        "category_name": "Open-Ended Simulators and Game Intelligence Benchmarks",
        "description": "Benchmarks and simulators that support open-ended embodied or game intelligence.",
    },
]


PAPERS = [
    {
        "paper_id": "world_models_2018",
        "title": "World Models",
        "authors": ["David Ha", "Jurgen Schmidhuber"],
        "year": 2018,
        "venue": "NeurIPS",
        "category_id": "cat_world_models",
        "category": "Generative Game World Simulation",
        "problem": "How to learn compact spatial and temporal representations of game environments so that agents can train with fewer direct environment interactions.",
        "method": "A variational autoencoder compresses observations, an MDN-RNN predicts latent dynamics, and a compact controller is optimized in latent space.",
        "contribution": "The paper showed that agents can be trained inside learned models of game environments such as CarRacing and VizDoom.",
        "limitations": "The demonstrations use relatively simple environments, and accumulated rollout error can weaken long-horizon planning.",
        "abstract": "World Models introduced a compact latent-dynamics approach for training agents in learned game environments.",
    },
    {
        "paper_id": "dreamerv3_2023",
        "title": "DreamerV3: Mastering Diverse Domains through World Models",
        "authors": ["Danijar Hafner", "Jurgis Pasukonis", "Jimmy Ba", "Timothy Lillicrap"],
        "year": 2023,
        "venue": "arXiv",
        "category_id": "cat_planning",
        "category": "Planning and Control with Learned Models",
        "problem": "How to make world-model reinforcement learning robust across diverse domains with minimal task-specific tuning.",
        "method": "DreamerV3 learns a latent world model and trains actor-critic behavior from imagined trajectories.",
        "contribution": "It broadened the empirical case for world-model agents across visual control and game-like domains.",
        "limitations": "The approach still depends on learned model quality and careful evaluation across domains.",
        "abstract": "DreamerV3 studies scalable latent world models for general reinforcement learning.",
    },
    {
        "paper_id": "muzero_2020",
        "title": "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model",
        "authors": ["Julian Schrittwieser", "Ioannis Antonoglou", "Thomas Hubert", "Karen Simonyan", "David Silver"],
        "year": 2020,
        "venue": "Nature",
        "category_id": "cat_planning",
        "category": "Planning and Control with Learned Models",
        "problem": "How to combine planning with learned dynamics without requiring a hand-coded simulator model.",
        "method": "MuZero learns representation, dynamics, and prediction functions and uses tree search over the learned model.",
        "contribution": "The system achieved strong results in Atari and classic board games while learning the model used for planning.",
        "limitations": "The planning machinery and training pipeline are computationally demanding.",
        "abstract": "MuZero demonstrates planning over learned latent dynamics for games and control tasks.",
    },
    {
        "paper_id": "gamengen_2024",
        "title": "GameNGen: Diffusion Models Are Real-Time Game Engines",
        "authors": ["Dani Valevski", "Yaniv Leviathan", "Moab Arar", "Shlomi Fruchter"],
        "year": 2024,
        "venue": "arXiv",
        "category_id": "cat_neural_engines",
        "category": "Neural Game Engines and Interactive Environments",
        "problem": "How to generate interactive game frames in real time using learned generative dynamics.",
        "method": "The system uses a diffusion model conditioned on previous frames and actions to produce playable game-video dynamics.",
        "contribution": "It framed diffusion models as neural game engines capable of interactive visual simulation.",
        "limitations": "Real-time generative play raises consistency, controllability, and evaluation challenges.",
        "abstract": "GameNGen explores diffusion models as real-time neural game engines.",
    },
    {
        "paper_id": "genie_2024",
        "title": "Genie: Generative Interactive Environments",
        "authors": ["Tim Rocktaschel", "Jack Parker-Holder", "Sasha Salter", "et al."],
        "year": 2024,
        "venue": "arXiv",
        "category_id": "cat_neural_engines",
        "category": "Neural Game Engines and Interactive Environments",
        "problem": "How to learn action-controllable interactive environments from video-scale data.",
        "method": "Genie learns a latent action model and video generator to create controllable environments from visual examples.",
        "contribution": "It connected generative video modeling with interactive environment construction for agents and players.",
        "limitations": "Generated environments require careful evaluation of controllability, temporal coherence, and task semantics.",
        "abstract": "Genie presents a foundation world model for generative interactive environments.",
    },
    {
        "paper_id": "alphastar_2019",
        "title": "Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning",
        "authors": ["Oriol Vinyals", "Igor Babuschkin", "Wojciech Czarnecki", "et al."],
        "year": 2019,
        "venue": "Nature",
        "category_id": "cat_game_agents",
        "category": "Game Agents and Multi-Agent Reinforcement Learning",
        "problem": "How to train agents for a complex real-time strategy game with partial observability and long-horizon decisions.",
        "method": "AlphaStar combines supervised learning, reinforcement learning, and league-based multi-agent training.",
        "contribution": "The work demonstrated grandmaster-level performance in StarCraft II and highlighted the value of competitive training ecosystems.",
        "limitations": "The system relies on extensive engineering, training scale, and domain-specific interfaces.",
        "abstract": "AlphaStar studies large-scale multi-agent reinforcement learning for StarCraft II.",
    },
    {
        "paper_id": "dota2_2019",
        "title": "Dota 2 with Large Scale Deep Reinforcement Learning",
        "authors": ["Christopher Berner", "Greg Brockman", "Brooke Chan", "et al."],
        "year": 2019,
        "venue": "arXiv",
        "category_id": "cat_game_agents",
        "category": "Game Agents and Multi-Agent Reinforcement Learning",
        "problem": "How to train coordinated agents in a high-dimensional, long-horizon multiplayer game.",
        "method": "OpenAI Five uses large-scale self-play reinforcement learning with recurrent policies and extensive distributed training.",
        "contribution": "The project showed that self-play can produce strong coordinated behavior in Dota 2.",
        "limitations": "The training regime is compute-intensive and tied to a specific game abstraction.",
        "abstract": "OpenAI Five demonstrates large-scale reinforcement learning in Dota 2.",
    },
    {
        "paper_id": "generative_agents_2023",
        "title": "Generative Agents: Interactive Simulacra of Human Behavior",
        "authors": ["Joon Sung Park", "Joseph O'Brien", "Carrie Cai", "et al."],
        "year": 2023,
        "venue": "UIST",
        "category_id": "cat_simulators",
        "category": "Open-Ended Simulators and Game Intelligence Benchmarks",
        "problem": "How to create believable interactive agents that remember, plan, and act in a simulated social environment.",
        "method": "The system combines memory streams, reflection, planning, and language-model-driven action selection.",
        "contribution": "It illustrated how generative agents can populate interactive worlds with coherent behavior.",
        "limitations": "Behavioral realism and evaluation remain difficult, and language-model outputs need guardrails.",
        "abstract": "Generative Agents studies language-model-based agents in an interactive sandbox world.",
    },
    {
        "paper_id": "unisim_2024",
        "title": "UniSim: Learning Interactive Real-World Simulators",
        "authors": ["Jiannan Xiang", "Kaiming He", "et al."],
        "year": 2024,
        "venue": "arXiv",
        "category_id": "cat_simulators",
        "category": "Open-Ended Simulators and Game Intelligence Benchmarks",
        "problem": "How to learn interactive simulators that can support agents beyond a single fixed task.",
        "method": "UniSim trains generative world simulators from broad visual interaction data.",
        "contribution": "It positions learned simulators as reusable environments for interactive intelligence.",
        "limitations": "General-purpose simulator quality depends on broad data coverage and robust interaction evaluation.",
        "abstract": "UniSim explores learning interactive simulators from data.",
    },
    {
        "paper_id": "diffusion_world_models_2024",
        "title": "Diffusion World Models",
        "authors": ["Eloi Alonso", "Adam Jelley", "Vincent Micheli", "et al."],
        "year": 2024,
        "venue": "arXiv",
        "category_id": "cat_world_models",
        "category": "Generative Game World Simulation",
        "problem": "How to use diffusion-style generative modeling for world dynamics and agent learning.",
        "method": "The work uses diffusion models to represent or predict environment dynamics for model-based learning.",
        "contribution": "It expands the design space of generative world models beyond recurrent latent dynamics.",
        "limitations": "Diffusion rollouts can be expensive and must be assessed for temporal consistency.",
        "abstract": "Diffusion World Models investigates diffusion-based dynamics for world-model learning.",
    },
    {
        "paper_id": "gaia1_2023",
        "title": "GAIA-1: A Generative World Model for Autonomous Driving",
        "authors": ["Anthony Hu", "Lloyd Russell", "Hudson Yeo", "et al."],
        "year": 2023,
        "venue": "arXiv",
        "category_id": "cat_world_models",
        "category": "Generative Game World Simulation",
        "problem": "How to learn a controllable generative world model for complex embodied environments.",
        "method": "GAIA-1 models video, action, and scene context to generate plausible future driving observations.",
        "contribution": "Although outside games, it shows how controllable world models can transfer to high-stakes embodied simulation.",
        "limitations": "Driving-focused results do not directly solve game intelligence evaluation and require domain-specific validation.",
        "abstract": "GAIA-1 studies controllable generative world models for autonomous driving simulation.",
    },
    {
        "paper_id": "minedojo_2022",
        "title": "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge",
        "authors": ["Linxi Fan", "Guanzhi Wang", "Yunfan Jiang", "et al."],
        "year": 2022,
        "venue": "NeurIPS",
        "category_id": "cat_simulators",
        "category": "Open-Ended Simulators and Game Intelligence Benchmarks",
        "problem": "How to benchmark and train open-ended embodied agents in a rich game environment.",
        "method": "MineDojo builds a Minecraft-based environment with tasks, data, and knowledge resources.",
        "contribution": "It provides a large open-ended game environment for embodied agent research.",
        "limitations": "Minecraft-specific affordances and task design choices shape what can be concluded.",
        "abstract": "MineDojo provides a Minecraft-based platform for open-ended embodied agents.",
    },
]


def build_final_seed_data(task_id: str = "task_world_models_and_gamecraft_final_001") -> dict[str, Any]:
    categories = [dict(item, paper_ids=[]) for item in CATEGORIES]
    by_category = {item["category_id"]: item for item in categories}
    paper_cards = []
    evidence = []
    citation_items = []
    for paper in PAPERS:
        card = dict(paper)
        card["topic_relevance_score"] = 1.0
        card["topic_relevance_negative_hit"] = False
        card["possible_claims"] = {
            "key_results": [{"text": card["contribution"], "evidence_id": f"ev_{card['paper_id']}_contribution"}],
            "method": [{"text": card["method"], "evidence_id": f"ev_{card['paper_id']}_method"}],
            "limitations": [{"text": card["limitations"], "evidence_id": f"ev_{card['paper_id']}_limitations"}],
        }
        paper_cards.append(card)
        by_category[card["category_id"]]["paper_ids"].append(card["paper_id"])
        citation_items.append(
            {
                "paper_id": card["paper_id"],
                "title": card["title"],
                "authors": card["authors"],
                "year": card["year"],
                "venue": card["venue"],
            }
        )
        for kind in ["problem", "method", "contribution", "limitations"]:
            evidence.append(
                {
                    "evidence_id": f"ev_{card['paper_id']}_{kind}",
                    "paper_id": card["paper_id"],
                    "text": card[kind],
                    "source_type": "final_seed_card",
                    "supports": [kind],
                    "confidence": 0.95,
                }
            )
    return {
        "paper_cards": {"task_id": task_id, "paper_cards": paper_cards},
        "taxonomy": {"task_id": task_id, "categories": categories},
        "evidence_store": {"task_id": task_id, "evidence": evidence},
        "citation_ready_set": {
            "task_id": task_id,
            "allowed_paper_ids": [paper["paper_id"] for paper in paper_cards],
            "items": citation_items,
        },
        "citation_index": {
            "task_id": task_id,
            "citations": [{"paper_id": item["paper_id"]} for item in citation_items],
        },
        "figure_bank": {"task_id": task_id, "figures": []},
        "table_bank": {"task_id": task_id, "tables": []},
        "knowledge_bundle": {
            "task_id": task_id,
            "topic": "World Models and GameCraft for Interactive Game Intelligence",
            "status": "success",
            "summary": {
                "paper_count": len(paper_cards),
                "core_paper_count": len(paper_cards),
                "evidence_count": len(evidence),
            },
        },
    }


def write_final_seed_files(root: Path | None = None) -> dict[str, str]:
    root = root or Path(__file__).resolve().parent.parent
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    data = build_final_seed_data()
    paths: dict[str, str] = {}
    for name, payload in data.items():
        for target in [cache / f"{name}.json", cache / f"final_{name}.json"]:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[name] = str((cache / f"{name}.json").relative_to(root)).replace("\\", "/")
    return paths


def main() -> None:
    paths = write_final_seed_files()
    print(json.dumps({"status": "success", "outputs": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
