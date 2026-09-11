"""B tool entry: verify survey citations (structural) + map claims to evidence (NLI). Thin orchestrator."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from config import load_config
from llm_client import InternS2Client
from harness.json_io import read_json, write_json
from harness.logger import setup_logging

from tools.models.artifacts import CitationIndex, EvidenceStore, Figure, FigureBank, ParsedPapers, TableBank
from tools.nlp.nli_verifier import FakeNLIModel, NLIVerifier
from tools.verify import claim_mapper, structural
from tools.verify.text_units import sentence_units

logger = logging.getLogger(__name__)


def run(request_path: str) -> dict:
    setup_logging()
    cfg = load_config()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else cfg.root_dir / path

    req = read_json(resolve(request_path))
    task_id = req["task_id"]
    inputs = req["inputs"]
    outputs = req["outputs"]

    survey_md = resolve(inputs["survey_markdown_path"]).read_text(encoding="utf-8")

    ready_set = _final_or_requested_json(cfg.root_dir, "citation_ready_set", resolve(inputs["citation_ready_set_path"]))
    citation_index = CitationIndex(
        task_id=task_id,
        citations=[{"paper_id": pid} for pid in ready_set.get("allowed_paper_ids", [])],
    )

    figure_bank = FigureBank(**_final_or_requested_json(cfg.root_dir, "figure_bank", resolve(inputs["figure_bank_path"])))
    gen_bank = read_json(resolve(inputs["generated_artifact_bank_path"]))
    gen_figures = [
        Figure(figure_id=a["artifact_id"], paper_id="generated", caption=a.get("artifact_path", ""), page=0)
        for a in gen_bank.get("artifacts", [])
        if a.get("artifact_id")
    ]
    figure_bank = FigureBank(task_id=task_id, figures=list(figure_bank.figures) + gen_figures)

    table_bank = TableBank(**_final_or_requested_json(cfg.root_dir, "table_bank", resolve(inputs["table_bank_path"])))
    evidence_store = EvidenceStore(**_final_or_requested_json(cfg.root_dir, "evidence_store", resolve(inputs["evidence_store_path"])))
    logger.info(
        "[verify] start task_id=%s survey_md=%s chars ready_citations=%s evidence=%s",
        task_id,
        len(survey_md),
        len(citation_index.citations),
        len(evidence_store.evidence),
    )

    citation_result = structural.run(task_id, survey_md, citation_index, figure_bank, table_bank)
    logger.info(
        "[verify] structural: total=%s invalid=%s score=%s",
        citation_result.total_citations,
        citation_result.invalid_citations,
        citation_result.citation_validity_score,
    )

    nli_model = _nli_model()
    llm_fallback = (
        InternS2Client(cfg)
        if os.getenv("EVISURVEY_CLAIM_LLM_FALLBACK", "false").lower() in {"1", "true", "yes"}
        else None
    )
    # T5: the writer's structured claim source is the truth; missing file means
    # a template/freeform run falls back to reverse-parsing via the shared kernel.
    structured_claims_path = cfg.root_dir / "cache" / "structured_claims.json"
    structured_claims = None
    if structured_claims_path.exists():
        saved_claims = read_json(structured_claims_path)
        if saved_claims.get("task_id") == task_id:
            structured_claims = saved_claims.get("claims") or None
    claim_map = claim_mapper.run(
        task_id,
        survey_md,
        evidence_store,
        ParsedPapers(task_id=task_id, papers=[]),
        nli_model,
        llm_fallback,
        structured_claims=structured_claims,
    )

    write_json(resolve(outputs["citation_result_path"]), citation_result.model_dump())
    write_json(resolve(outputs["claim_map_path"]), claim_map.model_dump())

    source_violations = sum(bool(e.source_role_violation) for e in claim_map.entries)
    quote_like = sum(bool(e.quote_diagnostic.get("quote_like")) for e in claim_map.entries)
    plan_path = cfg.root_dir / "cache" / "section_claim_plan.json"
    empty_sections = 0
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan.get("task_id") == task_id:
            for section in plan.get("sections", []):
                title = re.escape(str(section.get("section_title", "")))
                match = re.search(r"(?ms)^## " + title + r"\s*\n(.*?)(?=^## |\Z)", survey_md)
                units = sentence_units(match.group(1), set(ready_set.get("allowed_paper_ids", []))) if match else []
                if not any(cites for _, cites in units):
                    empty_sections += 1
    unsupported = sum(1 for e in claim_map.entries if e.status == "unsupported")
    status = "success" if (citation_result.invalid_citations == 0 and unsupported == 0 and not empty_sections and bool(claim_map.entries)) else "partial_success"
    logger.info("[verify] claim_map: %s entries, %s unsupported -> status=%s", len(claim_map.entries), unsupported, status)

    metrics = {
        "total_citations": citation_result.total_citations,
        "valid_citations": citation_result.valid_citations,
        "invalid_citations": citation_result.invalid_citations,
        "citation_validity_score": citation_result.citation_validity_score,
        "total_claims": len(claim_map.entries),
        "supported_claims": sum(1 for e in claim_map.entries if e.status == "supported"),
        "weak_claims": sum(1 for e in claim_map.entries if e.status == "weak"),
        "unsupported_claims": unsupported,
        "source_role_violations": source_violations,
        "quote_like_claims": quote_like,
        "empty_evidence_sections": empty_sections,
        "no_verifiable_claims": int(not claim_map.entries),
    }
    return {
        "status": status,
        "outputs": [outputs["citation_result_path"], outputs["claim_map_path"]],
        "metrics": metrics,
        "message": (
            f"verify_citations: {status} "
            f"(citation_score={citation_result.citation_validity_score}, {unsupported} unsupported claims)"
        ),
    }


_NLI_MODEL = None


def _nli_model():
    # One model per process: constructing NLIVerifier() per verify/repair round
    # reloads the CrossEncoder each time and the copies accumulate — two full
    # E2E runs were jetsam-killed at the post-repair verify boundary on a 16GB
    # machine before this cache existed.
    global _NLI_MODEL
    if _NLI_MODEL is None:
        if os.getenv("EVISURVEY_REAL_NLI", "false").lower() in {"1", "true", "yes"}:
            _NLI_MODEL = NLIVerifier()
        else:
            _NLI_MODEL = FakeNLIModel(mapping=_demo_entailment_keywords())
    return _NLI_MODEL


def _final_or_requested_json(root: Path, name: str, requested_path: Path) -> dict:
    if os.getenv("FINAL_SEED_PAPERS") == "1":
        final_path = root / "cache" / f"final_{name}.json"
        if final_path.exists():
            return read_json(final_path)
    return read_json(requested_path)


def _demo_entailment_keywords():
    return {
        "world": "entailment",
        "model": "entailment",
        "game": "entailment",
        "agent": "entailment",
        "simulation": "entailment",
        "learning": "entailment",
        "interactive": "entailment",
    }
