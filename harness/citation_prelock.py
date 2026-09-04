"""Build CitationReadySet before C writes the survey."""

from __future__ import annotations

from typing import Any


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
                "allowed_citation": f"[{paper_id}]",
                "evidence_ids": evidence_by_paper.get(paper_id, []),
            }
        )

    # Newest first: with a cap below the pool size, whitelisting the oldest
    # papers tanks the survey's freshness profile (S0 eval L0 regression).
    ready_items.sort(key=lambda item: (item.get("year") or 0, item["paper_id"]), reverse=True)
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
