"""A-side validation for B's KnowledgeBundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_io import read_json
from .sandbox_guard import SandboxGuard


class KnowledgeBundleValidationError(ValueError):
    pass


class KnowledgeBundleValidator:
    REQUIRED_ARTIFACTS = [
        "retrieved_papers",
        "parsed_papers",
        "figure_bank",
        "table_bank",
        "paper_cards",
        "evidence_store",
        "taxonomy",
        "citation_index",
    ]

    def __init__(self, root_dir: Path) -> None:
        self.guard = SandboxGuard(root_dir)

    def validate(self, bundle_path: str | Path) -> dict[str, Any]:
        path = self.guard.ensure_existing_file(bundle_path)
        bundle = read_json(path)
        errors: list[str] = []

        if bundle.get("status") not in {"success", "partial_success"}:
            errors.append("knowledge_bundle.status must be success or partial_success")
        artifacts = bundle.get("artifacts")
        if not isinstance(artifacts, dict):
            errors.append("knowledge_bundle.artifacts must be an object")
            artifacts = {}

        loaded: dict[str, Any] = {}
        for name in self.REQUIRED_ARTIFACTS:
            artifact_path = artifacts.get(name)
            if not artifact_path:
                errors.append(f"missing artifact path: {name}")
                continue
            try:
                loaded[name] = read_json(self.guard.ensure_existing_file(artifact_path))
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))

        if loaded:
            errors.extend(self._validate_id_consistency(loaded))

        if errors:
            raise KnowledgeBundleValidationError("; ".join(errors))

        return {"bundle": bundle, "artifacts": loaded}

    def _validate_id_consistency(self, loaded: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        paper_ids = self._extract_paper_ids(loaded.get("paper_cards"))
        citation_ids = set(self._extract_citation_ids(loaded.get("citation_index")))
        evidence_paper_ids = set(self._extract_evidence_paper_ids(loaded.get("evidence_store")))

        if not paper_ids:
            errors.append("paper_cards contains no paper_id")
        missing_citations = paper_ids - citation_ids
        if missing_citations:
            errors.append(f"citation_index missing paper_ids: {sorted(missing_citations)[:10]}")
        unknown_evidence = evidence_paper_ids - paper_ids
        if unknown_evidence:
            errors.append(f"evidence_store references unknown paper_ids: {sorted(unknown_evidence)[:10]}")
        return errors

    def _extract_paper_ids(self, paper_cards: Any) -> set[str]:
        cards = paper_cards.get("paper_cards", paper_cards) if isinstance(paper_cards, dict) else paper_cards
        if not isinstance(cards, list):
            return set()
        return {card["paper_id"] for card in cards if isinstance(card, dict) and card.get("paper_id")}

    def _extract_citation_ids(self, citation_index: Any) -> list[str]:
        if isinstance(citation_index, dict):
            if isinstance(citation_index.get("citations"), list):
                return [
                    item["paper_id"]
                    for item in citation_index["citations"]
                    if isinstance(item, dict) and item.get("paper_id")
                ]
            return [key for key in citation_index if isinstance(key, str)]
        if isinstance(citation_index, list):
            return [item["paper_id"] for item in citation_index if isinstance(item, dict) and item.get("paper_id")]
        return []

    def _extract_evidence_paper_ids(self, evidence_store: Any) -> list[str]:
        evidence = evidence_store.get("evidence", evidence_store) if isinstance(evidence_store, dict) else evidence_store
        if not isinstance(evidence, list):
            return []
        return [item["paper_id"] for item in evidence if isinstance(item, dict) and item.get("paper_id")]
