"""Build CitationReadySet before C writes the survey."""

from __future__ import annotations

import math
from typing import Any

# Whitelist selection = recency x influence. Pure recency squeezed the field's
# canonical papers (World Models 2018, Dreamer series, MuZero) out of the window
# whenever the pool exceeded the cap (S0 eval: system-in-gold 0.583 -> 0.118). The
# influence signal is what retrieval knows about a paper: SciVerse citation_count
# plus how many seed surveys cite it (survey_ref_count, see P2 bib candidates).
RECENCY_WEIGHT = 0.5
INFLUENCE_WEIGHT = 0.5


def _influence(item: dict[str, Any]) -> float:
    raw = int(item.get("citation_count") or 0) + int(item.get("survey_ref_count") or 0)
    return math.log1p(max(raw, 0))


def _selection_order(items: list[dict[str, Any]]) -> None:
    """Rank whitelisted candidates by an even recency/influence blend, in place.

    Influence breaks blend ties (the pool's oldest paper caps at 0.5 and would
    otherwise lose to an uncited newest paper on the year tie-break). An all-zero
    influence signal degrades to the legacy pure-recency order.
    """
    max_influence = max((_influence(item) for item in items), default=0.0)
    years = [item.get("year") or 0 for item in items]
    min_year, max_year = min(years), max(years)

    def key(item: dict[str, Any]) -> tuple:
        year = item.get("year") or 0
        year_score = (year - min_year) / (max_year - min_year) if max_year > min_year else 0.0
        influence = _influence(item)
        influence_score = influence / max_influence if max_influence > 0 else 0.0
        return (RECENCY_WEIGHT * year_score + INFLUENCE_WEIGHT * influence_score,
                influence, year, item["paper_id"])

    items.sort(key=key, reverse=True)


def build_citation_ready_set(
    *,
    task_id: str,
    paper_cards: Any,
    citation_index: Any,
    evidence_store: Any,
    max_core_papers: int,
) -> dict[str, Any]:
    cards = _as_list(paper_cards, "paper_cards")
    citation_ids = set(_citation_ids(citation_index))
    evidence_by_paper = _evidence_by_paper(evidence_store)

    ready_items = []
    for card in cards:
        paper_id = card.get("paper_id")
        if not paper_id or paper_id not in citation_ids:
            continue
        ready_items.append(
            {
                "paper_id": paper_id,
                "title": card.get("title", ""),
                "authors": card.get("authors", []),
                "year": card.get("year"),
                "venue": card.get("venue", ""),
                "category": card.get("category", ""),
                "citation_count": int(card.get("citation_count") or 0),
                "survey_ref_count": int(card.get("survey_ref_count") or 0),
                "allowed_citation": f"[{paper_id}]",
                "evidence_ids": evidence_by_paper.get(paper_id, []),
            }
        )

    _selection_order(ready_items)
    ready_items = ready_items[:max_core_papers]
    return {
        "task_id": task_id,
        "citation_format": "[paper_id]",
        "allowed_paper_ids": [item["paper_id"] for item in ready_items],
        "items": ready_items,
        "rules": {
            "only_these_citations_allowed": True,
            "references_must_be_generated_from_paper_cards": True,
            "claims_should_use_supporting_evidence_ids": True,
        },
    }


def _as_list(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get(key, [])
    return data if isinstance(data, list) else []


def _citation_ids(citation_index: Any) -> list[str]:
    if isinstance(citation_index, dict) and isinstance(citation_index.get("citations"), list):
        return [item["paper_id"] for item in citation_index["citations"] if item.get("paper_id")]
    if isinstance(citation_index, dict):
        return list(citation_index.keys())
    if isinstance(citation_index, list):
        return [item["paper_id"] for item in citation_index if isinstance(item, dict) and item.get("paper_id")]
    return []


def _evidence_by_paper(evidence_store: Any) -> dict[str, list[str]]:
    evidence = evidence_store.get("evidence", evidence_store) if isinstance(evidence_store, dict) else evidence_store
    grouped: dict[str, list[str]] = {}
    if not isinstance(evidence, list):
        return grouped
    for item in evidence:
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paper_id")
        evidence_id = item.get("evidence_id")
        if paper_id and evidence_id:
            grouped.setdefault(paper_id, []).append(evidence_id)
    return grouped
