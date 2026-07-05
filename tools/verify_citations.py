"""B tool entry: verify survey citations (structural) + map claims to evidence (NLI). Thin orchestrator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from config import load_config
from llm_client import InternS2Client
from harness.json_io import read_json, write_json
from harness.logger import setup_logging

from tools.models.artifacts import CitationIndex, EvidenceStore, Figure, FigureBank, ParsedPapers, TableBank
from tools.nlp.nli_verifier import FakeNLIModel, NLIVerifier
from tools.verify import claim_mapper, structural

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

    ready_set = read_json(resolve(inputs["citation_ready_set_path"]))
    citation_index = CitationIndex(
        task_id=task_id,
        citations=[{"paper_id": pid} for pid in ready_set.get("allowed_paper_ids", [])],
    )

    figure_bank = FigureBank(**read_json(resolve(inputs["figure_bank_path"])))
    gen_bank = read_json(resolve(inputs["generated_artifact_bank_path"]))
    gen_figures = [
        Figure(figure_id=a["artifact_id"], paper_id="generated", caption=a.get("artifact_path", ""), page=0)
        for a in gen_bank.get("artifacts", [])
        if a.get("artifact_id")
    ]
    figure_bank = FigureBank(task_id=task_id, figures=list(figure_bank.figures) + gen_figures)

    table_bank = TableBank(**read_json(resolve(inputs["table_bank_path"])))
    evidence_store = EvidenceStore(**read_json(resolve(inputs["evidence_store_path"])))
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
    claim_map = claim_mapper.run(
        task_id,
        survey_md,
        evidence_store,
        ParsedPapers(task_id=task_id, papers=[]),
        nli_model,
        llm_fallback,
    )

    write_json(resolve(outputs["citation_result_path"]), citation_result.model_dump())
    write_json(resolve(outputs["claim_map_path"]), claim_map.model_dump())

    unsupported = sum(1 for e in claim_map.entries if e.status == "unsupported")
    status = "success" if (citation_result.invalid_citations == 0 and unsupported == 0) else "partial_success"
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


def _nli_model():
    if os.getenv("EVISURVEY_REAL_NLI", "false").lower() in {"1", "true", "yes"}:
        return NLIVerifier()
    return FakeNLIModel(mapping=_demo_entailment_keywords())


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
