"""Joint 1: retrieval-strategy agent — open-book aspect decomposition.

probe (zero LLM) -> deterministic clustering -> one naming call -> full
sample per aspect -> report card -> bounded edit loop (<= max_rounds) ->
deterministic gate + best-so-far rollback -> validated strategy.

Spec: docs/RealAgent/关节1-策略Agent-方案与测试.md (decisions §2, plan §3).
The toolbox is the permission boundary; the gate is deterministic code,
never the LLM (spec §3.6).
"""

from __future__ import annotations

import copy
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..search_strategy_builder import (
    DEFAULT_ASPECTS,
    _build_clustered_strategy,
    _build_default_strategy,
    _cluster_keywords,
    _compact_plan_to_strategy,
    _ensure_min_keywords,
    _group_by_cluster,
    _kmeans,
    _validate_strategy,
    _vectorize,
)
from .loop import AgentConfig, BoundedAgentLoop, LoopFinished, LoopParseError, trajectory_writer
from .relevance import (
    EmbeddingScorer,
    aspect_text,
    build_report_card,
    global_score,
    hit_year,
    paper_identity,
    paper_text,
    render_card,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = 3
DEFAULT_MAX_LLM_CALLS = 7
DEFAULT_MAX_SCIVERSE_CALLS = 20
FIXPOINT_EPSILON = 0.02

STRATEGY_SYSTEM = """You are EviSurvey's retrieval strategist for an academic survey pipeline.
Each turn you see a report card: one row per search aspect with n_hits (raw results),
n_unique (papers no other aspect already claimed), overlap (Jaccard vs other aspects),
rel (relevance of returned papers to the aspect), sample titles, and year span.

Reply with exactly ONE JSON object choosing one action:
- {"action": "probe_query", "query": "...", "filters": "none" | "default", "page_size": 8}
  Test a query without touching the strategy. filters="none" bypasses the year filter
  (many zero-result aspects are killed by it); filters="default" applies the same filter
  the pipeline uses. Does NOT consume a round.
- {"action": "discover_by_description", "text": "natural-language description of the sub-field"}
  Semantic search that tolerates missing jargon; returns real titles and field keywords
  you can mine into concrete queries. Does NOT consume a round.
- {"action": "delegate_search", "goal": "...", "queries": ["...", "..."]}
  Hand ONE deep-exploration goal to a bounded search subagent (max 2 delegations per
  run, does NOT consume a round). It runs its own SciVerse searches and returns ONLY a
  compact summary: n_queries, up to 8 paper titles with years, at most 10 field
  keywords, and its notes — raw hits never reach this conversation. Its SciVerse calls
  share your budget. Use it for depth after R2/R3 fail, or to scout a brand-new
  direction before committing keywords to it.
- {"action": "commit_edits", "edits": [{"...see ops below...}]}
  Apply edits and trigger a full re-sample + a new report card. CONSUMES ONE ROUND.
  Ops:
    {"op": "rewrite", "aspect_id": "...", "name"?, "description"?, "keywords"?}
    {"op": "merge", "aspect_ids": ["...", "..."], "name", "description", "keywords"?}
    {"op": "add", "name", "description", "keywords": [...]}
    {"op": "delete", "aspect_id": "..."}
- {"action": "accept"}
  Finish. Only passes when EVERY aspect has n_unique >= 3.

Playbook, in order, for a starving aspect (n_unique < 3):
R0: probe_query with filters="none". If that returns hits, the year filter is the killer;
    rewrite keywords toward older work and commit.
R1: rewrite the keywords with DIFFERENT field jargon (synonyms the community actually
    uses), then probe again. After 2 failed rewrites you may deliberately try a
    neighboring-field term (controlled divergence).
R2: discover_by_description with a plain-language description of what the aspect WANTS;
    mine the returned titles/keywords into queries.
R3: the framework state lists "donated keywords" from the nearest already-retrieved
    papers — reuse them.
Deep exploration: when R2/R3 leave you guessing, delegate_search with a crisp goal and
    2-3 candidate queries instead of spending your own turns on repeated probes.
R4: if everything fails, delete the aspect in a commit (an empty sub-area is itself a
    finding) — keep 3 to 6 aspects in total.

Other signals: two aspects with high mutual overlap are the same aspect -> merge them.
High n_hits but low rel -> keywords too broad -> narrow them. You have few rounds and
few SciVerse calls (see budget line) — do not commit trivial edits."""

MAX_DELEGATIONS = 2
SUBAGENT_MAX_TURNS = 4
SUBAGENT_MAX_LLM_CALLS = 3
SUBAGENT_MAX_WALLCLOCK = 90.0

SUBAGENT_SYSTEM = """You are EviSurvey's delegated search subagent. The strategy agent
hands you ONE exploration goal; you run the searches and report a compact summary.
Reply with exactly ONE JSON object choosing one action:
- {"action": "search_papers", "query": "...", "page_size": 8}
  Meta-search with no year filter; returns mined title/year/keywords per hit.
- {"action": "discover", "text": "plain-language description of the sub-field"}
  Semantic search that tolerates missing jargon; returns titles and field keywords.
- {"action": "report_findings", "notes": "one short line: does the area exist, which keywords worked"}
  Finish. The papers and keywords you actually retrieved are attached automatically —
  never invent papers or keywords.

You have very few turns and you share the strategy joint's SciVerse budget: stop
exploring as soon as you hold enough titles/keywords and report."""


def _sampling_filters(end_year: int) -> list[dict]:
    # Mirrors P3's _year_filters(2018, end) so the card measures what P3 would see.
    return [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2018},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": end_year},
    ]


class _BudgetExceeded(RuntimeError):
    pass


class _CountingSciverse:
    """Counting proxy enforcing the joint's SciVerse call budget."""

    def __init__(self, client: Any, limit: int) -> None:
        self._client = client
        self.limit = limit
        self.calls = 0

    @property
    def left(self) -> int:
        return self.limit - self.calls

    def _check(self) -> None:
        if self.calls >= self.limit:
            raise _BudgetExceeded(
                f"SciVerse budget exhausted ({self.limit} calls used); "
                "decide from the data you already have: commit_edits or accept")

    def meta_search(self, query, **kwargs):
        self._check()
        self.calls += 1
        return self._client.meta_search(query, **kwargs)

    def agentic_search(self, query, **kwargs):
        self._check()
        self.calls += 1
        return self._client.agentic_search(query, **kwargs)


def _make_sciverse(api_key: str, base_url: str) -> Any:
    # Lazy import keeps the harness->tools edge at call time (unit tests never hit it);
    # reuses the verified throttle/retry/backoff instead of raw httpx.
    from tools.clients.sciverse_client import SciVerseClient

    return SciVerseClient(base_url=base_url or None, api_key=api_key or None)


class _Snapshot:
    """One round's frozen (strategy, card) pair for best-so-far rollback."""

    def __init__(self, strategy: dict, card: dict, per_hits: list) -> None:
        self.strategy = strategy
        self.card = card
        self.per_hits = per_hits
        self.score = global_score(card)


# --- t0: probe + deterministic clustering (zero LLM) -------------------------


def _probe_view(hit: dict) -> dict:
    return {
        "title": str(hit.get("title", "") or ""),
        "abstract": str(hit.get("abstract", "") or ""),
        "year": hit.get("publication_published_year") or hit.get("year"),
        "venue": hit.get("publication_venue_name_unified") or hit.get("venue") or "",
        "cited_by": int(hit.get("citation_count") or 0),
        "doi": hit.get("doi", "") or "",
        "keywords": [str(k) for k in (hit.get("keywords") or [])][:8],
    }


def _probe(sciverse: Any, topic: str, limit: int) -> list[dict]:
    papers: list[dict] = []
    seen: set[str] = set()
    for query in (f"{topic} survey review", f"{topic} methods benchmark dataset"):
        try:
            res = sciverse.meta_search(query=query, page_size=min(12, max(5, limit)))
            hits = res.get("results", [])
        except _BudgetExceeded:
            raise
        except Exception as exc:  # external API boundary
            logger.warning(f"[strategy-agent] probe failed q={query!r}: {exc}")
            continue
        for hit in hits:
            key = paper_identity(hit)
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(_probe_view(hit))
            if len(papers) >= limit:
                return papers
    return papers


def _cluster_probe_papers(papers: list[dict], topic: str, cluster_count: int) -> list[list[dict]]:
    """Deterministic clusters via the builder's token kmeans; [] when not viable."""
    if len(papers) < 6:
        return []
    texts = [f"{p.get('title', '')}. {p.get('abstract', '')}" for p in papers]
    vectors, vocab = _vectorize(texts, topic)
    if len(vocab) < 8:
        return []
    k = max(3, min(cluster_count, len(papers), 6))
    assignments = _kmeans(vectors, k)
    clusters = [c for c in _group_by_cluster(papers, texts, assignments, k) if c]
    return clusters if len(clusters) >= 3 else []


# --- LLM call 1: naming -------------------------------------------------------


def _cluster_brief(clusters: list[list[dict]], topic: str) -> str:
    lines = []
    for index, cluster in enumerate(clusters[:6], start=1):
        titles = [p["title"][:110] for p in cluster[:3] if p.get("title")]
        keywords = _cluster_keywords(cluster, topic)[:6]
        years = [p.get("year") for p in cluster if p.get("year")]
        span = f"{min(years)}-{max(years)}" if years else "n/a"
        lines.append(
            f"cluster {index} ({len(cluster)} papers, years {span}): "
            f"keywords={', '.join(keywords) or '-'}; e.g. {'; '.join(titles) or '-'}")
    return "\n".join(lines)


def _naming_prompt(topic: str, clusters: list[list[dict]], probe_papers: list[dict],
                   memory_context: str) -> str:
    if clusters:
        context = (
            "Deterministic clusters computed from a real corpus probe "
            f"({len(probe_papers)} papers):\n" + _cluster_brief(clusters, topic))
    elif probe_papers:
        titles = "\n".join(
            f"- ({p.get('year') or '?'}) {p['title'][:120]}" for p in probe_papers[:12])
        context = f"Probe papers (no stable clusters):\n{titles}"
    else:
        context = "No probe papers are available; rely on your field knowledge and memory."
    memory = memory_context.strip() or "No prior project memory is relevant."
    return f"""Topic: "{topic}"

{context}

Relevant project memory:
{memory}

Design 3-6 topic-specific search aspects that name and sharpen these clusters.
Hard rules:
- Do NOT use generic names (Foundational Concepts; Methods and Systems; Benchmarks and
  Evaluation; Recent and Emerging Directions).
- Keywords must stay inside the topic's own technical field and read like concrete
  academic search queries; exclude directions that only overlap lexically.
- Prefer specific methods/systems over broad neighbors.

Return exactly one compact JSON object:
{{
  "main_domain": "short academic field name",
  "organization_mode": "chronological | thematic | hybrid",
  "organization_reason": "one short reason",
  "aspects": [
    {{"name": "specific aspect name", "description": "one sentence",
      "min_papers": 3, "keywords": ["5-10 concrete academic search queries"]}}
  ]
}}"""


def _anchor_generic_names(strategy: dict, topic: str) -> None:
    """Rename generic template aspects to topic-anchored names so they pass validation."""
    generic = {name for _, name, _, _ in DEFAULT_ASPECTS}
    for aspect in strategy["wide_search"]["search_aspects"]:
        if aspect["aspect_name"] in generic:
            aspect["aspect_name"] = f"{aspect['aspect_name']} for {topic}"
    strategy["topic_understanding"]["sub_domains"] = [
        a["aspect_name"] for a in strategy["wide_search"]["search_aspects"]]


def _name_aspects(llm_json_chat, *, topic, clusters, probe_papers, memory_context,
                  task_id, max_papers, max_core_papers, end_year,
                  cluster_count) -> tuple[dict, str]:
    prompt = _naming_prompt(topic, clusters, probe_papers, memory_context)
    system = (
        "You are a strict JSON API for EviSurvey's strategy incubator. "
        "Return only one compact JSON object. Do not include analysis or a thinking "
        "process. The first character of your response must be '{'."
    )
    try:
        raw = llm_json_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ])
        strategy = _compact_plan_to_strategy(
            raw, task_id=task_id, topic=topic, max_papers=max_papers,
            max_core_papers=max_core_papers, end_year=end_year)
        # top up short keyword lists before validation — same courtesy _apply_edits
        # gives edits; without it a 4-keyword aspect wastes the whole naming call
        for aspect in strategy["wide_search"]["search_aspects"]:
            aspect["keywords"] = list(dict.fromkeys(_ensure_min_keywords(
                topic=topic, aspect_name=aspect["aspect_name"],
                keywords=aspect["keywords"])))[:10]
        _validate_strategy(strategy, topic)
        return strategy, "llm_naming"
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        # LLM boundary: degrade clustered (probe-derived) -> topic-anchored template.
        # Every snapshot must pass _validate_strategy or the best-so-far invariant
        # breaks, so generic template names get a topic anchor.
        logger.warning(f"[strategy-agent] naming failed ({exc}) -> fallback chain")
        if probe_papers:
            try:
                clustered = _build_clustered_strategy(
                    task_id=task_id, topic=topic, max_papers=max_papers,
                    max_core_papers=max_core_papers, end_year=end_year,
                    papers=probe_papers, cluster_count=cluster_count)
                if clustered is not None:
                    _validate_strategy(clustered, topic)
                    return clustered, "cluster_fallback"
            except (ValueError, KeyError, TypeError) as cluster_exc:
                logger.warning(f"[strategy-agent] cluster fallback failed: {cluster_exc}")
        template = _build_default_strategy(
            task_id=task_id, topic=topic, max_papers=max_papers,
            max_core_papers=max_core_papers, end_year=end_year)
        _anchor_generic_names(template, topic)
        _validate_strategy(template, topic)
        return template, "template_fallback"


# --- sampling + edits ---------------------------------------------------------


def _sample(sciverse: _CountingSciverse, aspects: list[dict], end_year: int,
            page_size: int = 8) -> list[list[dict]]:
    """One meta-search per aspect with the P3 year filter (signal, not the real retrieval)."""
    per_hits: list[list[dict]] = []
    for aspect in aspects:
        query = " ".join(str(k) for k in aspect.get("keywords", []))
        try:
            res = sciverse.meta_search(query=query, filters=_sampling_filters(end_year),
                                       page_size=page_size)
            per_hits.append(res.get("results", []))
        except _BudgetExceeded:
            raise
        except Exception as exc:  # external API boundary: one dead aspect ≠ dead run
            logger.warning(f"[strategy-agent] sample failed {aspect.get('aspect_id')}: {exc}")
            per_hits.append([])
    return per_hits


def _apply_edits(strategy: dict, edits: list[dict], topic: str) -> dict:
    """Apply add/delete/rewrite/merge edits to a copy; ValueError when the result is invalid."""
    by_id = {a["aspect_id"]: a for a in strategy["wide_search"]["search_aspects"]}
    aspects: list[dict] = []
    for aspect in strategy["wide_search"]["search_aspects"]:
        aspects.append({
            "aspect_id": aspect["aspect_id"],
            "aspect_name": aspect["aspect_name"],
            "description": aspect.get("description", ""),
            "min_papers": aspect.get("min_papers", 3),
            "keywords": list(aspect.get("keywords", [])),
        })
    for edit in edits:
        op = str(edit.get("op", ""))
        if op == "delete":
            aspect_id = str(edit.get("aspect_id", ""))
            if aspect_id not in by_id:
                raise ValueError(f"delete: unknown aspect_id {aspect_id!r}")
            aspects = [a for a in aspects if a["aspect_id"] != aspect_id]
        elif op == "add":
            name = str(edit.get("name", "")).strip()
            if not name:
                raise ValueError("add: name is required")
            keywords = [str(k).strip() for k in edit.get("keywords", []) if str(k).strip()]
            if not keywords:
                raise ValueError(f"add {name!r}: keywords are required")
            aspects.append({
                "aspect_id": f"aspect_add_{len(aspects) + 1:03d}",
                "aspect_name": name,
                "description": str(edit.get("description", "")).strip() or f"Papers about {name}.",
                "min_papers": max(2, min(8, _to_int(edit.get("min_papers"), 3))),
                "keywords": keywords,
            })
        elif op == "rewrite":
            aspect_id = str(edit.get("aspect_id", ""))
            if aspect_id not in by_id:
                raise ValueError(f"rewrite: unknown aspect_id {aspect_id!r}")
            target = next(a for a in aspects if a["aspect_id"] == aspect_id)
            if str(edit.get("name", "")).strip():
                target["aspect_name"] = str(edit["name"]).strip()
            if str(edit.get("description", "")).strip():
                target["description"] = str(edit["description"]).strip()
            if edit.get("keywords"):
                target["keywords"] = [str(k).strip() for k in edit["keywords"] if str(k).strip()]
        elif op == "merge":
            ids = [str(i) for i in edit.get("aspect_ids", [])]
            if len(ids) < 2:
                raise ValueError("merge: needs at least 2 aspect_ids")
            for aspect_id in ids:
                if aspect_id not in by_id:
                    raise ValueError(f"merge: unknown aspect_id {aspect_id!r}")
            merged = [by_id[i] for i in ids]
            default_keywords = list(dict.fromkeys(k for a in merged for k in a.get("keywords", [])))
            keywords = [str(k).strip() for k in edit.get("keywords", default_keywords)
                        if str(k).strip()] or default_keywords
            aspects = ([a for a in aspects if a["aspect_id"] not in set(ids)]
                       + [{
                           "aspect_id": f"aspect_add_{len(aspects) + 1:03d}",
                           "aspect_name": str(edit.get("name", "")).strip()
                               or " / ".join(a["aspect_name"] for a in merged),
                           "description": str(edit.get("description", "")).strip()
                               or " ".join(a.get("description", "") for a in merged),
                           "min_papers": max(2, min(8, _to_int(edit.get("min_papers"), 3))),
                           "keywords": keywords,
                       }])
        else:
            raise ValueError(f"unknown edit op {op!r} (use add/delete/rewrite/merge)")

    normalized = []
    for index, aspect in enumerate(aspects, start=1):
        keywords = _ensure_min_keywords(
            topic=topic, aspect_name=aspect["aspect_name"], keywords=aspect["keywords"])
        normalized.append({
            "aspect_id": f"aspect_{index:03d}",
            "aspect_name": aspect["aspect_name"],
            "description": aspect["description"],
            "min_papers": aspect["min_papers"],
            "keywords": list(dict.fromkeys(keywords))[:10],
        })
    new_strategy = copy.deepcopy(strategy)
    new_strategy["wide_search"]["search_aspects"] = normalized
    new_strategy["topic_understanding"]["sub_domains"] = [a["aspect_name"] for a in normalized]
    _validate_strategy(new_strategy, topic)
    return new_strategy


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _donated_keywords(card: dict, aspects: list[dict], pool_hits: list[dict],
                      scorer: EmbeddingScorer) -> dict[str, list[str]]:
    """R3: starving aspects receive keywords from their nearest retrieved papers."""
    pool = [h for h in pool_hits if h.get("keywords")]
    if not pool:
        return {}
    donations: dict[str, list[str]] = {}
    for row, aspect in zip(card["aspects"], aspects):
        if row["n_unique"] >= 3:
            continue
        scores = scorer.score_pairs([(aspect_text(aspect), paper_text(h)) for h in pool])
        nearest = sorted(zip(scores, pool), key=lambda pair: -pair[0])[:3]
        collected: list[str] = []
        for _, hit in nearest:
            for keyword in hit.get("keywords", []):
                text = str(keyword).strip()
                if text and text.lower() not in {c.lower() for c in collected}:
                    collected.append(text)
        if collected:
            donations[row["aspect_id"]] = collected[:8]
    return donations


def _mine_discovery_hits(hits: list[dict]) -> dict:
    """R2: extract titles + frequent field tokens from agentic-search hits."""
    titles = [str(h.get("title", "")).strip() for h in hits if str(h.get("title", "")).strip()]
    counter: Counter = Counter()
    for hit in hits:
        for keyword in hit.get("keywords") or []:
            text = str(keyword).strip()
            if len(text) > 2:
                counter[text] += 1
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", str(hit.get("title", ""))):
            counter[token.lower()] += 1
    return {
        "n_hits": len(hits),
        "titles": titles[:6],
        "field_keywords": [k for k, _ in counter.most_common(8)],
        "hint": "mine these titles/keywords into concrete queries for the dead aspect",
    }


# --- the joint ----------------------------------------------------------------


def run_strategy_agent(
    *,
    task_id: str,
    topic: str,
    max_papers: int,
    max_core_papers: int,
    end_year: int,
    llm_json_chat: Callable[[list[dict[str, str]]], dict],
    sciverse: Any | None = None,
    sciverse_api_key: str = "",
    sciverse_api_base_url: str = "https://api.sciverse.space",
    memory_context: str = "",
    probe_limit: int = 20,
    cluster_count: int = 4,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    max_sciverse_calls: int = DEFAULT_MAX_SCIVERSE_CALLS,
    max_wallclock: float = 240.0,
    fixpoint_epsilon: float = FIXPOINT_EPSILON,
    trajectory_dir: str | Path | None = None,
    scorer: EmbeddingScorer | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    client = sciverse if sciverse is not None else _make_sciverse(sciverse_api_key, sciverse_api_base_url)
    budget = _CountingSciverse(client, max_sciverse_calls)
    scorer = scorer or EmbeddingScorer()
    trajectory_dir = Path(trajectory_dir) if trajectory_dir is not None else Path("logs/trajectory")
    log = trajectory_writer(trajectory_dir / f"{task_id}_strategy.jsonl")
    log({"agent": "strategy", "event": "start", "topic": topic, "task_id": task_id,
         "max_rounds": max_rounds, "max_llm_calls": max_llm_calls,
         "max_sciverse_calls": max_sciverse_calls})

    # t0: probe + deterministic clustering (zero LLM)
    probe_papers = _probe(budget, topic, probe_limit)
    clusters = _cluster_probe_papers(probe_papers, topic, cluster_count)
    log({"agent": "strategy", "event": "probe", "n_papers": len(probe_papers),
         "n_clusters": len(clusters), "sciverse_calls": budget.calls})

    # LLM call 1: name the clusters into initial aspects
    strategy, naming_mode = _name_aspects(
        llm_json_chat, topic=topic, clusters=clusters, probe_papers=probe_papers,
        memory_context=memory_context, task_id=task_id, max_papers=max_papers,
        max_core_papers=max_core_papers, end_year=end_year, cluster_count=cluster_count)
    aspects = strategy["wide_search"]["search_aspects"]
    log({"agent": "strategy", "event": "naming", "mode": naming_mode,
         "aspects": [a["aspect_name"] for a in aspects]})

    # round 1: full sample + report card (zero LLM)
    per_hits = _sample(budget, aspects, end_year)
    card = build_report_card(aspects, per_hits, scorer)
    snapshots: list[_Snapshot] = [_Snapshot(strategy, card, per_hits)]
    log({"agent": "strategy", "event": "report_card", "round": 1,
         "score": card["global_score"], "card": card})

    frame: dict[str, Any] = {
        "strategy": strategy, "card": card, "per_hits": per_hits, "rounds_used": 1,
        "donations": _donated_keywords(card, aspects, [h for hits in per_hits for h in hits], scorer),
    }
    lessons: list[str] = []

    # -- toolbox == permission boundary (spec §3.4) -------------------------

    def probe_query(decision: dict) -> dict:
        query = str(decision.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        filters_mode = str(decision.get("filters", "none"))
        filters = _sampling_filters(end_year) if filters_mode == "default" else None
        page_size = max(1, min(8, _to_int(decision.get("page_size"), 8)))
        try:
            res = budget.meta_search(query=query, filters=filters, page_size=page_size)
        except _BudgetExceeded as exc:
            return {"error": str(exc)}
        except Exception as exc:  # external API boundary
            return {"error": f"probe failed: {exc}"}
        hits = res.get("results", [])
        years = [y for y in (hit_year(h) for h in hits) if y]
        return {
            "query": query,
            "filters": "default" if filters else "none",
            "n_hits": len(hits),
            "sample_titles": [str(h.get("title", "")) for h in hits[:3]],
            "year_span": [min(years), max(years)] if years else None,
        }

    def discover_by_description(decision: dict) -> dict:
        text = str(decision.get("text", "")).strip()
        if not text:
            return {"error": "text is required"}
        try:
            res = budget.agentic_search(query=text, top_k=8)
        except _BudgetExceeded as exc:
            return {"error": str(exc)}
        except Exception as exc:  # external API boundary
            return {"error": f"agentic_search failed: {exc}"}
        return _mine_discovery_hits(res.get("hits", []))

    def commit_edits(decision: dict) -> dict:
        edits = decision.get("edits")
        if not isinstance(edits, list) or not edits:
            return {"error": "edits must be a non-empty list"}
        if frame["rounds_used"] >= max_rounds:
            return {"error": f"no rounds left (max_rounds={max_rounds}); call accept to finish"}
        try:
            new_strategy = _apply_edits(frame["strategy"], edits, topic)
        except (ValueError, KeyError, TypeError) as exc:
            return {"error": f"edit rejected: {exc}"}
        new_aspects = new_strategy["wide_search"]["search_aspects"]
        if budget.left < len(new_aspects):
            return {"error": (
                f"not enough SciVerse calls left for a full re-sample "
                f"(need {len(new_aspects)}, have {budget.left}); call accept instead)")}
        for edit in edits:
            if str(edit.get("op")) == "delete":
                deleted_name = next(
                    (a["aspect_name"] for a in frame["strategy"]["wide_search"]["search_aspects"]
                     if a["aspect_id"] == str(edit.get("aspect_id"))), "?")
                lessons.append(
                    f"For {topic}: aspect '{deleted_name}' starved and was deleted — "
                    f"this sub-area appears absent from the corpus; skip it on retry.")
        new_per_hits = _sample(budget, new_aspects, end_year)
        new_card = build_report_card(new_aspects, new_per_hits, scorer)
        frame["strategy"], frame["card"], frame["per_hits"] = new_strategy, new_card, new_per_hits
        frame["rounds_used"] += 1
        frame["donations"] = _donated_keywords(
            new_card, new_aspects, [h for hits in new_per_hits for h in hits], scorer)
        prev_score = snapshots[-1].score
        snapshots.append(_Snapshot(new_strategy, new_card, new_per_hits))
        log({"agent": "strategy", "event": "report_card", "round": frame["rounds_used"],
             "score": new_card["global_score"], "prev_score": round(prev_score, 4),
             "card": new_card})
        improvement = new_card["global_score"] - prev_score
        if frame["rounds_used"] >= 2 and improvement < fixpoint_epsilon:
            raise LoopFinished({
                "status": "fixpoint",
                "reason": f"score improvement {improvement:.4f} < epsilon {fixpoint_epsilon}",
            })
        return {"applied": len(edits), "round": frame["rounds_used"],
                "note": "strategy re-sampled; the new report card is in the framework state"}

    def accept(decision: dict) -> dict:
        failing = [f"{r['aspect_id']}({r['aspect_name']}) n_unique={r['n_unique']}"
                   for r in frame["card"]["aspects"] if r["n_unique"] < 3]
        if failing:
            return {"error": "gate rejected — aspects below n_unique>=3: "
                    + "; ".join(failing)
                    + ". Fix their keywords (playbook R0-R3) or delete truly empty ones, "
                      "then commit_edits and accept."}
        raise LoopFinished({"status": "accept"})

    # -- delegated exploration: bounded search subagent (spec 搜索SubAgent工具) --
    # The subagent shares budget (same _CountingSciverse), llm_json_chat and the
    # trajectory writer; only its compact summary ever returns to the main loop.

    sub: dict[str, Any] = {
        "delegations": 0, "llm_calls": 0, "queries": [], "papers": [],
        "keywords": Counter(), "errors": [],
    }

    def _sub_record_hits(hits: list[dict]) -> None:
        seen = {p["title"] for p in sub["papers"]}
        for hit in hits:
            title = str(hit.get("title", "") or "").strip()
            if title and title not in seen:
                seen.add(title)
                sub["papers"].append({"title": title[:140], "year": hit_year(hit)})
            for keyword in hit.get("keywords") or []:
                text = str(keyword).strip()
                if len(text) > 2:
                    sub["keywords"][text] += 1
            for token in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", title):
                sub["keywords"][token.lower()] += 1

    def _sub_summary(notes: str) -> dict:
        summary = {
            "n_queries": len(sub["queries"]),
            "papers": sub["papers"][:8],
            "field_keywords": [k for k, _ in sub["keywords"].most_common(10)],
            "notes": notes,
        }
        if sub["errors"]:
            summary["error"] = "; ".join(dict.fromkeys(sub["errors"]))
        return summary

    def sub_search_papers(decision: dict) -> dict:
        query = str(decision.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        page_size = max(1, min(8, _to_int(decision.get("page_size"), 8)))
        try:
            res = budget.meta_search(query=query, filters=None, page_size=page_size)
        except _BudgetExceeded as exc:
            sub["errors"].append(str(exc))
            return {"error": str(exc)}
        except Exception as exc:  # external API boundary
            sub["errors"].append(f"search failed: {exc}")
            return {"error": f"search failed: {exc}"}
        hits = res.get("results", [])
        sub["queries"].append(query)
        _sub_record_hits(hits)
        return {"query": query, "n_hits": len(hits),
                "papers": [{"title": str(h.get("title", ""))[:140], "year": hit_year(h),
                            "keywords": [str(k) for k in (h.get("keywords") or [])][:6]}
                           for h in hits[:page_size]]}

    def sub_discover(decision: dict) -> dict:
        text = str(decision.get("text", "")).strip()
        if not text:
            return {"error": "text is required"}
        try:
            res = budget.agentic_search(query=text, top_k=8)
        except _BudgetExceeded as exc:
            sub["errors"].append(str(exc))
            return {"error": str(exc)}
        except Exception as exc:  # external API boundary
            sub["errors"].append(f"agentic_search failed: {exc}")
            return {"error": f"agentic_search failed: {exc}"}
        hits = res.get("hits", [])
        sub["queries"].append(text)
        _sub_record_hits(hits)
        return _mine_discovery_hits(hits)

    def sub_report_findings(decision: dict) -> dict:
        raise LoopFinished(_sub_summary(str(decision.get("notes", "")).strip()))

    def delegate_search(decision: dict) -> dict:
        goal = str(decision.get("goal", "")).strip()
        if not goal:
            return {"error": "goal is required"}
        if sub["delegations"] >= MAX_DELEGATIONS:
            return {"error": (f"delegation budget exhausted ({MAX_DELEGATIONS} per run); "
                              "explore yourself with probe_query / discover_by_description")}
        sub["delegations"] += 1
        suggested = [str(q).strip() for q in decision.get("queries", []) if str(q).strip()][:5]
        sub_task = (f"Exploration goal from the strategy agent: {goal}\n"
                    + (f"Suggested starting queries: {suggested}\n" if suggested else "")
                    + "Search for what this goal needs, then report_findings "
                      "with one short notes line.")
        sub_config = AgentConfig(
            name="strategy_subagent",
            system_prompt=SUBAGENT_SYSTEM,
            tools={"search_papers": sub_search_papers,
                   "discover": sub_discover,
                   "report_findings": sub_report_findings},
            max_turns=SUBAGENT_MAX_TURNS,
            max_llm_calls=SUBAGENT_MAX_LLM_CALLS,
            max_wallclock=SUBAGENT_MAX_WALLCLOCK,
            on_action=log,
            on_finish=lambda summary: log(
                {"agent": "strategy_subagent", "event": "loop_finish", **summary}),
        )
        sub_loop = BoundedAgentLoop(sub_config, llm_json_chat)
        try:
            result = sub_loop.run(sub_task)
        except Exception as exc:  # delegation boundary: a broken subagent must not abort the main loop
            logger.warning(f"[strategy-agent] subagent aborted: {exc}")
            sub["errors"].append(f"subagent aborted: {exc}")
            sub["llm_calls"] += sub_loop.llm_calls
            return _sub_summary("")
        sub["llm_calls"] += sub_loop.llm_calls
        if result.status == "finished" and result.payload:
            return result.payload
        return _sub_summary(f"subagent stopped on {result.status} before reporting")

    def state_provider() -> str:
        state = render_card(frame["card"],
                            rounds_left=max_rounds - frame["rounds_used"],
                            sciverse_left=budget.left)
        if frame["donations"]:
            lines = [f"- {aspect_id}: {', '.join(keywords)}"
                     for aspect_id, keywords in frame["donations"].items()]
            state += "\ndonated keywords from nearest retrieved papers:\n" + "\n".join(lines)
        return state

    # -- bounded edit loop ---------------------------------------------------

    config = AgentConfig(
        name="strategy",
        system_prompt=STRATEGY_SYSTEM,
        tools={"probe_query": probe_query,
               "discover_by_description": discover_by_description,
               "delegate_search": delegate_search,
               "commit_edits": commit_edits,
               "accept": accept},
        max_turns=2 * max_llm_calls,
        max_llm_calls=max_llm_calls - 1,  # naming already spent one
        max_wallclock=max(1.0, max_wallclock - (time.monotonic() - started)),
        on_action=log,
        on_finish=lambda summary: log({"agent": "strategy", "event": "loop_finish", **summary}),
        state_provider=state_provider,
    )
    task = (f"Survey topic: {topic}\n"
            "Improve the search aspects until every report-card row has n_unique >= 3 "
            "(or the sub-area is proven absent and deleted), then accept.")
    loop = BoundedAgentLoop(config, llm_json_chat)
    loop_llm_calls = 0
    try:
        result = loop.run(task)
        finish_status = result.status
        if result.payload and result.payload.get("status"):
            finish_status = f"{result.status}:{result.payload['status']}"
        loop_llm_calls = loop.llm_calls
    except (LoopParseError, ValueError, RuntimeError) as exc:
        # Joint boundary: an aborted loop must not lose the completed rounds —
        # fall through to best-so-far (loop lower bound = round-1 level, spec §3.6).
        logger.warning(f"[strategy-agent] loop aborted -> best-so-far: {exc}")
        finish_status = "aborted"
        loop_llm_calls = loop.llm_calls

    # best-so-far rollback: never worse than round 1 (spec §3.6 condition 4)
    best = snapshots[0]
    for snapshot in snapshots[1:]:
        if snapshot.score >= best.score:  # >=: later round wins ties (more refined)
            best = snapshot
    final = copy.deepcopy(best.strategy)
    _validate_strategy(final, topic)  # invariant: every snapshot was validated at commit
    meta = {
        "rounds": frame["rounds_used"],
        "scores": [round(s.score, 4) for s in snapshots],
        "final_score": round(best.score, 4),
        "status": finish_status,
        "naming_mode": naming_mode,
        "llm_calls": 1 + loop_llm_calls,
        "subagent": {"delegations": sub["delegations"], "llm_calls": sub["llm_calls"]},
        "sciverse_calls": budget.calls,
        "elapsed_sec": round(time.monotonic() - started, 1),
        "rel_mode": best.card["rel_mode"],
        "lessons": lessons,
    }
    final["strategy_agent"] = meta
    log({"agent": "strategy", "event": "finish", **meta})
    logger.info(f"[strategy-agent] done: status={finish_status} rounds={meta['rounds']} "
                f"scores={meta['scores']} sciverse={budget.calls} llm={meta['llm_calls']}")
    return final
