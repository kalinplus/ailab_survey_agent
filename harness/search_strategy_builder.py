"""Build topic-aware search strategies for B."""

from __future__ import annotations

from typing import Any


DEFAULT_ASPECTS = [
    ("aspect_001", "Foundational Concepts", "Classic papers and definitions that shaped the research area.", 3),
    ("aspect_002", "Methods and Systems", "Representative methods, model architectures, systems, and toolkits.", 5),
    ("aspect_003", "Benchmarks and Evaluation", "Datasets, benchmarks, evaluation protocols, and comparisons.", 3),
    ("aspect_004", "Recent and Emerging Directions", "Recent papers, open problems, and future-facing applications.", 4),
]


def build_search_strategy(
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
) -> dict[str, Any]:
    topic_keywords = _topic_keywords(topic)
    aspects = []
    for index, (aspect_id, name, description, min_papers) in enumerate(DEFAULT_ASPECTS):
        aspects.append(
            {
                "aspect_id": aspect_id,
                "aspect_name": name,
                "description": description,
                "min_papers": min_papers,
                "keywords": _keywords_for_aspect(topic, topic_keywords, index),
            }
        )

    return {
        "task_id": task_id,
        "topic": topic,
        "strategy": "wide_then_deep",
        "topic_understanding": {
            "main_domain": topic,
            "sub_domains": [aspect["aspect_name"] for aspect in aspects],
        },
        "wide_search": {
            "goal": f"Cover the historical development and current frontier of {topic}.",
            "max_papers": max_papers,
            "time_range": {"start_year": 1990, "end_year": end_year},
            "search_aspects": aspects,
            "include_types": [
                "classic_papers",
                "recent_papers",
                "surveys",
                "benchmarks",
                "datasets",
                "toolkits",
            ],
        },
        "deep_search": {
            "goal": f"Select representative core papers for a grounded survey on {topic}.",
            "max_core_papers": max_core_papers,
            "selection_rules": {
                "include_classic_foundational_works": True,
                "include_recent_representative_works": True,
                "balance_search_aspects": True,
                "do_not_rank_only_by_citation": True,
            },
        },
    }


def _topic_keywords(topic: str) -> list[str]:
    words = [word.strip(" ,;:/\\()[]{}").lower() for word in topic.split()]
    return [word for word in words if len(word) > 2]


def _keywords_for_aspect(topic: str, topic_keywords: list[str], index: int) -> list[str]:
    suffixes = [
        ["survey", "foundational paper", "history"],
        ["method", "model", "system"],
        ["benchmark", "dataset", "evaluation"],
        ["recent", "future direction", "application"],
    ][index]
    base = [topic]
    base.extend(f"{topic} {suffix}" for suffix in suffixes)
    base.extend(f"{keyword} {suffixes[0]}" for keyword in topic_keywords[:3])
    return list(dict.fromkeys(base))
