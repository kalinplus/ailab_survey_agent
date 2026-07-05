"""Build topic-aware search strategies for B."""

from __future__ import annotations

import math
import re
from typing import Any, Callable

try:
    import httpx
except ModuleNotFoundError:
    class _MissingHttpx:
        HTTPError = Exception

        class Client:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ModuleNotFoundError("httpx is required for strategy probing")

    httpx = _MissingHttpx()  # type: ignore[assignment]


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
    use_probing: bool = False,
    sciverse_api_key: str = "",
    sciverse_api_base_url: str = "https://api.sciverse.space",
    request_timeout_seconds: float = 30,
    probe_limit: int = 20,
    cluster_count: int = 4,
    memory_context: str = "",
    llm_json_chat: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probe_papers: list[dict[str, Any]] = []
    llm_error: Exception | None = None
    if use_probing and sciverse_api_key:
        try:
            probe_papers = _probe_sciverse(
                topic=topic,
                api_key=sciverse_api_key,
                api_base_url=sciverse_api_base_url,
                timeout_seconds=request_timeout_seconds,
                limit=probe_limit,
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            probe_papers = []

    if llm_json_chat is not None:
        try:
            llm_strategy = _build_llm_strategy(
                task_id=task_id,
                topic=topic,
                max_papers=max_papers,
                max_core_papers=max_core_papers,
                end_year=end_year,
                probe_papers=probe_papers,
                memory_context=memory_context,
                llm_json_chat=llm_json_chat,
            )
            if llm_strategy is not None:
                return llm_strategy
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            llm_error = exc

    if use_probing and probe_papers:
        try:
            clustered = _build_clustered_strategy(
                task_id=task_id,
                topic=topic,
                max_papers=max_papers,
                max_core_papers=max_core_papers,
                end_year=end_year,
                papers=probe_papers,
                cluster_count=cluster_count,
            )
            if clustered is not None:
                return clustered
        except (ValueError, KeyError, TypeError):
            pass

    return _build_default_strategy(
        task_id=task_id,
        topic=topic,
        max_papers=max_papers,
        max_core_papers=max_core_papers,
        end_year=end_year,
    )


def _build_llm_strategy(
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
    probe_papers: list[dict[str, Any]],
    memory_context: str,
    llm_json_chat: Callable[[list[dict[str, str]]], dict[str, Any]],
) -> dict[str, Any] | None:
    prompt = _strategy_prompt(
        task_id=task_id,
        topic=topic,
        max_papers=max_papers,
        max_core_papers=max_core_papers,
        end_year=end_year,
        probe_papers=probe_papers,
        memory_context=memory_context,
    )
    raw = llm_json_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API for EviSurvey's strategy incubator. "
                    "Return only one compact JSON object. Do not include analysis, markdown, or a thinking process. "
                    "The first character of your response must be '{'. "
                    "Design topic-specific academic search aspects."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    strategy = _coerce_strategy(
        raw,
        task_id=task_id,
        topic=topic,
        max_papers=max_papers,
        max_core_papers=max_core_papers,
        end_year=end_year,
    )
    _validate_strategy(strategy, topic)
    return strategy


def _strategy_prompt(
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
    probe_papers: list[dict[str, Any]],
    memory_context: str = "",
) -> str:
    probe_lines = []
    for index, paper in enumerate(probe_papers[:12], start=1):
        title = str(paper.get("title", "")).strip()
        year = paper.get("year") or "unknown"
        if not title:
            continue
        probe_lines.append(f"{index}. ({year}) {title[:160]}")
    probe_text = "\n".join(probe_lines) if probe_lines else "No external probe papers are available."
    memory_text = memory_context.strip() if memory_context.strip() else "No prior project memory is relevant."

    return f"""
Return exactly one compact JSON object and nothing else. Do not output any thinking process.
Topic: "{topic}"

First infer the survey organization mode from the topic:
- chronological: use when the topic asks about development, evolution, history, timeline, progress, or trajectory.
- thematic: use when the topic asks about methods, applications, architectures, benchmarks, challenges, or comparison.
- hybrid: use when the topic asks about development of a broad field where a timeline is needed but each era should still be grouped by technical themes.

For chronological or hybrid topics, aspects should be time-aware stages with clear period hints in description/keywords.
For thematic topics, aspects should be technical blocks or application blocks.
Example: "The development of agent" should be chronological or hybrid, not a flat method taxonomy.

Generate 3-6 topic-specific aspects. Do NOT use generic names:
Foundational Concepts; Methods and Systems; Benchmarks and Evaluation; Recent and Emerging Directions.
For "World Models for GameCraft", prefer technical AI/game-world-model themes over social gaming culture.

Each aspect's keywords must stay inside the topic's own technical field and read like concrete academic
search queries that a paper in THIS field would match. Anchor every keyword on the topic's core concept,
and exclude directions that only overlap lexically: a "world model" survey must not surface pure LLMs,
cognitive-science education, pathology, or any area that merely shares a word like "model"/"world"/"cognitive".
Prefer specific methods/systems (e.g. "dreamer world model reinforcement learning") over broad neighbors
(e.g. "cognitive architecture").

Required compact JSON shape:
{{
  "main_domain": "short academic field name",
  "organization_mode": "chronological | thematic | hybrid",
  "organization_reason": "one short reason",
  "aspects": [
    {{
      "name": "specific aspect or stage name",
      "description": "one sentence, include period hints if time-aware",
      "min_papers": 3,
      "keywords": ["5-10 concrete academic search queries"]
    }}
  ]
}}

Optional probe titles:
{probe_text}

Relevant project memory:
{memory_text}
""".strip()


def _coerce_strategy(
    raw: dict[str, Any],
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
) -> dict[str, Any]:
    strategy = _unwrap_strategy(raw)
    if not isinstance(strategy, dict):
        raise ValueError("LLM strategy must be an object")

    if "wide_search" not in strategy and "aspects" in strategy:
        strategy = _compact_plan_to_strategy(
            strategy,
            task_id=task_id,
            topic=topic,
            max_papers=max_papers,
            max_core_papers=max_core_papers,
            end_year=end_year,
        )

    strategy["task_id"] = task_id
    strategy["topic"] = topic
    strategy["strategy"] = "wide_then_deep"

    topic_understanding = strategy.setdefault("topic_understanding", {})
    if not isinstance(topic_understanding, dict):
        topic_understanding = {}
        strategy["topic_understanding"] = topic_understanding
    topic_understanding["main_domain"] = str(topic_understanding.get("main_domain") or topic)
    organization_mode = str(topic_understanding.get("organization_mode") or _infer_organization_mode(topic)).lower()
    if organization_mode not in {"chronological", "thematic", "hybrid"}:
        organization_mode = _infer_organization_mode(topic)
    topic_understanding["organization_mode"] = organization_mode
    topic_understanding["organization_reason"] = str(
        topic_understanding.get("organization_reason")
        or _organization_reason(topic, organization_mode)
    )

    wide_search = strategy.setdefault("wide_search", {})
    if not isinstance(wide_search, dict):
        wide_search = {}
        strategy["wide_search"] = wide_search
    wide_search["goal"] = str(wide_search.get("goal") or f"Cover the historical development and current frontier of {topic}.")
    wide_search["max_papers"] = max_papers
    wide_search["time_range"] = {"start_year": 1990, "end_year": end_year}
    wide_search["include_types"] = [
        "classic_papers",
        "recent_papers",
        "surveys",
        "benchmarks",
        "datasets",
        "toolkits",
    ]

    aspects = wide_search.get("search_aspects", [])
    if not isinstance(aspects, list):
        raise ValueError("wide_search.search_aspects must be a list")
    normalized_aspects = []
    for index, aspect in enumerate(aspects[:6], start=1):
        if not isinstance(aspect, dict):
            continue
        name = str(aspect.get("aspect_name", "")).strip()
        description = str(aspect.get("description", "")).strip()
        keywords = aspect.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        keywords = _ensure_min_keywords(topic=topic, aspect_name=name, keywords=keywords)
        normalized_aspects.append(
            {
                "aspect_id": f"aspect_{index:03d}",
                "aspect_name": name,
                "description": description,
                "min_papers": max(2, min(8, _safe_int(aspect.get("min_papers")) or 3)),
                "keywords": list(dict.fromkeys(keywords))[:10],
            }
        )
    wide_search["search_aspects"] = normalized_aspects
    topic_understanding["sub_domains"] = [aspect["aspect_name"] for aspect in normalized_aspects]

    deep_search = strategy.setdefault("deep_search", {})
    if not isinstance(deep_search, dict):
        deep_search = {}
        strategy["deep_search"] = deep_search
    deep_search["goal"] = str(deep_search.get("goal") or f"Select representative core papers for a grounded survey on {topic}.")
    deep_search["max_core_papers"] = max_core_papers
    deep_search["selection_rules"] = {
        "include_classic_foundational_works": True,
        "include_recent_representative_works": True,
        "balance_search_aspects": True,
        "do_not_rank_only_by_citation": True,
    }
    return strategy


def _unwrap_strategy(raw: dict[str, Any]) -> dict[str, Any]:
    for key in ["search_strategy", "strategy", "strategy_plan", "search_strategy_plan"]:
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def _compact_plan_to_strategy(
    plan: dict[str, Any],
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
) -> dict[str, Any]:
    aspects = plan.get("aspects", [])
    if not isinstance(aspects, list):
        raise ValueError("compact strategy aspects must be a list")

    normalized_aspects = []
    for index, aspect in enumerate(aspects[:6], start=1):
        if not isinstance(aspect, dict):
            continue
        name = str(aspect.get("name") or aspect.get("aspect_name") or "").strip()
        description = str(aspect.get("description") or "").strip()
        keywords = aspect.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        normalized_aspects.append(
            {
                "aspect_id": f"aspect_{index:03d}",
                "aspect_name": name,
                "description": description,
                "min_papers": max(2, min(8, _safe_int(aspect.get("min_papers")) or 3)),
                "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()],
            }
        )

    organization_mode = str(plan.get("organization_mode") or _infer_organization_mode(topic)).lower()
    if organization_mode not in {"chronological", "thematic", "hybrid"}:
        organization_mode = _infer_organization_mode(topic)

    return {
        "task_id": task_id,
        "topic": topic,
        "strategy": "wide_then_deep",
        "topic_understanding": {
            "main_domain": str(plan.get("main_domain") or topic),
            "sub_domains": [aspect["aspect_name"] for aspect in normalized_aspects],
            "organization_mode": organization_mode,
            "organization_reason": str(plan.get("organization_reason") or _organization_reason(topic, organization_mode)),
        },
        "wide_search": {
            "goal": f"Cover the historical development and current frontier of {topic}.",
            "max_papers": max_papers,
            "time_range": {"start_year": 1990, "end_year": end_year},
            "search_aspects": normalized_aspects,
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


def _validate_strategy(strategy: dict[str, Any], topic: str) -> None:
    generic_names = {name for _, name, _, _ in DEFAULT_ASPECTS}
    aspects = strategy.get("wide_search", {}).get("search_aspects", [])
    organization_mode = strategy.get("topic_understanding", {}).get("organization_mode")
    if organization_mode not in {"chronological", "thematic", "hybrid"}:
        raise ValueError("organization_mode must be chronological, thematic, or hybrid")
    if not isinstance(aspects, list) or not 3 <= len(aspects) <= 6:
        raise ValueError("strategy must contain 3 to 6 aspects")
    names = []
    topic_tokens = set(_topic_keywords(topic))
    for aspect in aspects:
        name = str(aspect.get("aspect_name", "")).strip()
        keywords = aspect.get("keywords", [])
        description = str(aspect.get("description", "")).strip()
        if not name or name in generic_names:
            raise ValueError("aspect name is empty or generic")
        if name.lower() in {"research cluster", "cluster", "topic"}:
            raise ValueError("aspect name is not meaningful")
        if not description:
            raise ValueError("aspect description is missing")
        if not isinstance(keywords, list) or len(keywords) < 5:
            raise ValueError("aspect must have at least 5 keywords")
        aspect_tokens = set(_tokenize(name + " " + " ".join(str(keyword) for keyword in keywords[:5])))
        if topic_tokens and not aspect_tokens:
            raise ValueError("aspect tokens are empty")
        names.append(name.lower())
    if len(set(names)) != len(names):
        raise ValueError("duplicate aspect names")


def _infer_organization_mode(topic: str) -> str:
    text = topic.lower()
    chronological_markers = [
        "development",
        "evolution",
        "history",
        "historical",
        "timeline",
        "trajectory",
        "progress",
        "from ",
        "towards",
        "rise of",
    ]
    thematic_markers = [
        "methods",
        "architectures",
        "applications",
        "benchmarks",
        "evaluation",
        "challenges",
        "comparison",
        "taxonomy",
    ]
    has_chrono = any(marker in text for marker in chronological_markers)
    has_thematic = any(marker in text for marker in thematic_markers)
    if has_chrono and has_thematic:
        return "hybrid"
    if has_chrono:
        return "chronological"
    return "thematic" if has_thematic else "hybrid"


def _organization_reason(topic: str, organization_mode: str) -> str:
    if organization_mode == "chronological":
        return f"The topic '{topic}' asks for development or evolution, so stages should follow time."
    if organization_mode == "hybrid":
        return f"The topic '{topic}' benefits from a timeline plus technical grouping."
    return f"The topic '{topic}' is best covered by technical or application blocks."


def _ensure_min_keywords(topic: str, aspect_name: str, keywords: list[str]) -> list[str]:
    candidates = list(dict.fromkeys(keywords))
    fallback = [
        f"{topic} {aspect_name}",
        f"{topic} {aspect_name} survey",
        f"{topic} {aspect_name} review",
        f"{topic} {aspect_name} benchmark",
        f"{topic} {aspect_name} method",
        f"{aspect_name} academic survey",
    ]
    for keyword in fallback:
        if keyword.strip() and keyword not in candidates:
            candidates.append(keyword)
        if len(candidates) >= 5:
            break
    return candidates


def _build_default_strategy(
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
) -> dict[str, Any]:
    topic_keywords = _topic_keywords(topic)
    fallback_aspects = _fallback_aspect_templates(topic)
    aspects = []
    for index, (aspect_id, name, description, min_papers) in enumerate(fallback_aspects):
        keywords = _keywords_for_aspect(topic, topic_keywords, index)
        keywords.extend(
            [
                f"{topic} {name}",
                f"{topic} {name} survey",
                f"{topic} {name} benchmark",
                f"{name} academic papers",
            ]
        )
        aspects.append(
            {
                "aspect_id": aspect_id,
                "aspect_name": name,
                "description": description,
                "min_papers": min_papers,
                "keywords": list(dict.fromkeys(keywords))[:10],
            }
        )

    organization_mode = _infer_organization_mode(topic)
    return {
        "task_id": task_id,
        "topic": topic,
        "strategy": "wide_then_deep",
        "topic_understanding": {
            "main_domain": topic,
            "sub_domains": [aspect["aspect_name"] for aspect in aspects],
            "organization_mode": organization_mode,
            "organization_reason": _organization_reason(topic, organization_mode),
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


def _fallback_aspect_templates(topic: str) -> list[tuple[str, str, str, int]]:
    text = topic.lower()
    if "world" in text and "game" in text:
        return [
            (
                "aspect_001",
                "Generative Game World Simulation",
                "Models that learn or generate interactive game environments, latent dynamics, video prediction, and controllable world rollouts.",
                3,
            ),
            (
                "aspect_002",
                "Agent Planning and Control in Learned Worlds",
                "Work that uses world models for action prediction, model-based reinforcement learning, planning, and embodied game agents.",
                3,
            ),
            (
                "aspect_003",
                "GameCraft Benchmarks and Evaluation Protocols",
                "Datasets, simulators, benchmarks, human or automated evaluation protocols, and reproducibility practices for game world models.",
                3,
            ),
            (
                "aspect_004",
                "Neural Rendering and Multimodal Game State Modeling",
                "Architectures that connect visual observations, text instructions, physics-like state, memory, and multimodal generation in playable worlds.",
                3,
            ),
        ]
    return DEFAULT_ASPECTS


def _probe_sciverse(
    *,
    topic: str,
    api_key: str,
    api_base_url: str,
    timeout_seconds: float,
    limit: int,
) -> list[dict[str, Any]]:
    if "elsevier.com" in api_base_url:
        return _probe_elsevier_scopus(
            topic=topic,
            api_key=api_key,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            limit=limit,
        )

    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_query_limit = max(5, min(limit, 12))
    for query in _probe_queries(topic):
        try:
            candidates = _probe_sciverse_agentic_search(
                query=query,
                api_key=api_key,
                api_base_url=api_base_url,
                timeout_seconds=timeout_seconds,
                limit=per_query_limit,
            )
        except httpx.HTTPError:
            continue
        for paper in candidates:
            key = _paper_identity(paper)
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(paper)
            if len(papers) >= limit:
                return papers
    return papers


def _probe_queries(topic: str) -> list[str]:
    base = [
        f"{topic} survey review research overview",
        f"{topic} methods architectures benchmark evaluation",
        f"{topic} dataset toolkit system application",
        f"{topic} foundational recent representative papers",
    ]
    text = topic.lower()
    if "world" in text and "game" in text:
        base.extend(
            [
                f"{topic} interactive world model game simulation",
                f"{topic} game world model agent planning benchmark",
                f"{topic} GameNGen Genie DIAMOND Oasis GameCraft WorldMark",
            ]
        )
    return list(dict.fromkeys(base))


def _probe_sciverse_agentic_search(
    *,
    query: str,
    api_key: str,
    api_base_url: str,
    timeout_seconds: float,
    limit: int,
) -> list[dict[str, Any]]:
    endpoint = _join_url(api_base_url, "agentic-search")
    payload = {
        "query": query,
        "limit": max(1, min(limit, 50)),
        "top_k": max(1, min(limit, 50)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    papers = []
    for item in _extract_result_items(data):
        title = _first_text(
            item,
            [
                "title",
                "paper_title",
                "document_title",
                "publication_title",
                "work_title",
            ],
        )
        abstract = _first_text(
            item,
            [
                "abstract",
                "abstract_preview",
                "chunk",
                "text",
                "content",
                "snippet",
                "evidence",
            ],
        )
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if not title:
            title = _first_text(metadata, ["title", "paper_title", "document_title"])
        if not abstract:
            abstract = _first_text(metadata, ["abstract", "abstract_preview", "chunk", "text"])
        if not title and not abstract:
            continue
        text = f"{title}. {abstract}".strip()
        if len(_tokenize(text)) < 4:
            continue
        papers.append(
            {
                "title": title or abstract[:90],
                "abstract": abstract,
                "year": _year_from_date(
                    _first_text(item, ["year", "publication_published_year", "date", "publication_published_date"])
                    or _first_text(metadata, ["year", "publication_published_year", "date", "publication_published_date"])
                ),
                "venue": _first_text(item, ["venue", "publication_venue_name", "journal"])
                or _first_text(metadata, ["venue", "publication_venue_name", "journal"]),
                "cited_by": _safe_int(
                    item.get("citation_count")
                    or item.get("citations")
                    or item.get("cited_by")
                    or metadata.get("citation_count")
                    or metadata.get("citations")
                ),
                "doi": _first_text(item, ["doi"]) or _first_text(metadata, ["doi"]),
                "eid": _first_text(item, ["doc_id", "id", "unique_id"])
                or _first_text(metadata, ["doc_id", "id", "unique_id"]),
            }
        )
    return _filter_probe_papers(query, papers)


def _paper_identity(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = re.sub(r"\W+", " ", str(paper.get("title") or "").lower()).strip()
    year = paper.get("year") or ""
    return f"title:{title[:120]}:{year}" if title else ""


def _filter_probe_papers(query: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_tokens = set(_tokenize(query))
    focused: list[dict[str, Any]] = []
    for paper in papers:
        text = f"{paper.get('title', '')}. {paper.get('abstract', '')}".lower()
        tokens = set(_tokenize(text))
        if not tokens:
            continue
        overlap = len(tokens & query_tokens)
        if overlap < 2:
            continue
        if _looks_like_social_or_educational_game_noise(text, query):
            continue
        focused.append(paper)
    return focused


def _looks_like_social_or_educational_game_noise(text: str, query: str) -> bool:
    if "game" not in query.lower():
        return False
    technical_markers = [
        "world model",
        "reinforcement learning",
        "generative",
        "simulation",
        "interactive",
        "benchmark",
        "agent",
        "diffusion",
        "transformer",
        "latent",
    ]
    if any(marker in text for marker in technical_markers):
        return False
    noise_markers = [
        "world of warcraft",
        "mmog",
        "education",
        "addictive",
        "cyberspace",
        "entertainment",
        "social",
        "minor",
        "book review",
    ]
    return any(marker in text for marker in noise_markers)


def _probe_elsevier_scopus(
    *,
    topic: str,
    api_key: str,
    api_base_url: str,
    timeout_seconds: float,
    limit: int,
) -> list[dict[str, Any]]:
    query = (
        f'TITLE-ABS-KEY("{_escape_scopus_query(topic)}") '
        "AND (DOCTYPE(re) OR TITLE-ABS-KEY(survey) OR TITLE-ABS-KEY(review))"
    )
    params = {
        "query": query,
        "count": max(1, min(limit, 25)),
        "start": 0,
        "sort": "-citedby-count",
        "view": "COMPLETE",
        "field": ",".join(
            [
                "dc:title",
                "dc:description",
                "prism:coverDate",
                "prism:publicationName",
                "citedby-count",
                "prism:doi",
                "eid",
                "subtypeDescription",
            ]
        ),
    }
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(api_base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

    entries = data.get("search-results", {}).get("entry", [])
    papers = []
    for entry in entries:
        title = str(entry.get("dc:title", "")).strip()
        abstract = str(entry.get("dc:description", "")).strip()
        if not title:
            continue
        text = f"{title}. {abstract}".strip()
        if len(_tokenize(text)) < 4:
            continue
        papers.append(
            {
                "title": title,
                "abstract": abstract,
                "year": _year_from_date(entry.get("prism:coverDate")),
                "venue": entry.get("prism:publicationName", ""),
                "cited_by": _safe_int(entry.get("citedby-count")),
                "doi": entry.get("prism:doi", ""),
                "eid": entry.get("eid", ""),
            }
        )
    return papers


def _build_clustered_strategy(
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
    papers: list[dict[str, Any]],
    cluster_count: int,
) -> dict[str, Any] | None:
    if len(papers) < 6:
        return None

    texts = [f"{paper.get('title', '')}. {paper.get('abstract', '')}" for paper in papers]
    vectors, vocab = _vectorize(texts, topic)
    if len(vocab) < 8:
        return None

    k = max(3, min(cluster_count, len(papers), 6))
    assignments = _kmeans(vectors, k)
    clusters = _group_by_cluster(papers, texts, assignments, k)
    clusters = [cluster for cluster in clusters if cluster]
    if len(clusters) < 3:
        return None

    aspects = []
    for index, cluster in enumerate(clusters[:6], start=1):
        keywords = _cluster_keywords(cluster, topic)
        aspect_name = _aspect_name_from_keywords(keywords, fallback=f"Research Cluster {index}")
        aspects.append(
            {
                "aspect_id": f"aspect_{index:03d}",
                "aspect_name": aspect_name,
                "description": _cluster_description(aspect_name, cluster),
                "min_papers": max(3, min(5, len(cluster))),
                "keywords": _aspect_keywords(topic, aspect_name, keywords),
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


STOPWORDS = {
    "about",
    "abstract",
    "after",
    "also",
    "analysis",
    "and",
    "are",
    "based",
    "been",
    "between",
    "book",
    "can",
    "chapter",
    "copyright",
    "data",
    "der",
    "die",
    "doi",
    "development",
    "different",
    "during",
    "each",
    "elsevier",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "jats",
    "journal",
    "learning",
    "license",
    "method",
    "methods",
    "model",
    "models",
    "more",
    "new",
    "not",
    "our",
    "paper",
    "papers",
    "practices",
    "publisher",
    "propose",
    "proposed",
    "provide",
    "recent",
    "research",
    "results",
    "review",
    "rights",
    "section",
    "show",
    "studios",
    "such",
    "survey",
    "system",
    "systems",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "various",
    "with",
    "within",
    "xml",
}


def _escape_scopus_query(value: str) -> str:
    return value.replace('"', " ").strip()


def _join_url(base_url: str, path: str) -> str:
    if base_url.rstrip("/").endswith(path):
        return base_url.rstrip("/")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _extract_result_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ["items", "results", "data", "hits", "documents", "chunks"]:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _extract_result_items(value)
            if nested:
                return nested
    return []


def _first_text(data: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _year_from_date(value: Any) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(match.group(0)) if match else None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _vectorize(texts: list[str], topic: str) -> tuple[list[dict[str, float]], list[str]]:
    topic_tokens = set(_topic_keywords(topic))
    docs = [_tokenize(text) for text in texts]
    doc_freq: dict[str, int] = {}
    for tokens in docs:
        for token in set(tokens):
            if token not in topic_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

    vocab = [
        token
        for token, _ in sorted(doc_freq.items(), key=lambda item: (-item[1], item[0]))
        if doc_freq[token] >= 1
    ][:160]
    vocab_set = set(vocab)
    doc_count = max(1, len(docs))
    vectors = []
    for tokens in docs:
        counts: dict[str, float] = {}
        for token in tokens:
            if token in vocab_set:
                counts[token] = counts.get(token, 0.0) + 1.0
        vector = {}
        for token, count in counts.items():
            idf = math.log((doc_count + 1) / (doc_freq.get(token, 0) + 1)) + 1
            vector[token] = count * idf
        vectors.append(_normalize(vector))
    return vectors, vocab


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {key: value / norm for key, value in vector.items()}


def _kmeans(vectors: list[dict[str, float]], k: int) -> list[int]:
    centroids = _initial_centroids(vectors, k)
    assignments = [0 for _ in vectors]
    for _ in range(8):
        changed = False
        for index, vector in enumerate(vectors):
            scores = [_cosine(vector, centroid) for centroid in centroids]
            best = max(range(k), key=lambda item: scores[item])
            if assignments[index] != best:
                changed = True
                assignments[index] = best
        centroids = [_mean_vector([vectors[i] for i, group in enumerate(assignments) if group == cluster]) for cluster in range(k)]
        if not changed:
            break
    return assignments


def _initial_centroids(vectors: list[dict[str, float]], k: int) -> list[dict[str, float]]:
    if k <= 1:
        return vectors[:1]
    last = len(vectors) - 1
    indices = [round(index * last / (k - 1)) for index in range(k)]
    return [vectors[index] for index in indices]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _mean_vector(vectors: list[dict[str, float]]) -> dict[str, float]:
    if not vectors:
        return {}
    merged: dict[str, float] = {}
    for vector in vectors:
        for key, value in vector.items():
            merged[key] = merged.get(key, 0.0) + value
    return _normalize({key: value / len(vectors) for key, value in merged.items()})


def _group_by_cluster(
    papers: list[dict[str, Any]],
    texts: list[str],
    assignments: list[int],
    k: int,
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for paper, text, cluster in zip(papers, texts, assignments):
        item = dict(paper)
        item["_tokens"] = _tokenize(text)
        clusters[cluster].append(item)
    return sorted(clusters, key=lambda cluster: (-len(cluster), -sum(paper.get("cited_by", 0) for paper in cluster)))


def _cluster_keywords(cluster: list[dict[str, Any]], topic: str) -> list[str]:
    topic_tokens = set(_topic_keywords(topic))
    scores: dict[str, float] = {}
    for paper in cluster:
        weight = 1.0 + math.log1p(paper.get("cited_by", 0)) / 10
        for token in paper.get("_tokens", []):
            if token in topic_tokens or token in STOPWORDS:
                continue
            scores[token] = scores.get(token, 0.0) + weight
    return [token for token, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:8]]


def _aspect_name_from_keywords(keywords: list[str], *, fallback: str) -> str:
    if not keywords:
        return fallback
    words = [keyword.replace("_", " ").replace("-", " ") for keyword in keywords[:3]]
    return " ".join(word.title() for word in words)


def _cluster_description(aspect_name: str, cluster: list[dict[str, Any]]) -> str:
    years = [paper.get("year") for paper in cluster if paper.get("year")]
    year_hint = f" ({min(years)}-{max(years)})" if years else ""
    titles = [paper.get("title", "") for paper in cluster[:2] if paper.get("title")]
    title_hint = "; ".join(titles)
    if title_hint:
        return f"Probe-derived cluster{year_hint} around {aspect_name}, represented by papers such as {title_hint}."
    return f"Probe-derived cluster{year_hint} around {aspect_name}."


def _aspect_keywords(topic: str, aspect_name: str, keywords: list[str]) -> list[str]:
    base = [
        topic,
        f"{topic} {aspect_name}",
        f"{topic} {aspect_name} survey",
        f"{topic} {aspect_name} review",
    ]
    base.extend(f"{topic} {keyword}" for keyword in keywords[:5])
    base.extend(keywords[:5])
    return list(dict.fromkeys(item for item in base if item.strip()))
