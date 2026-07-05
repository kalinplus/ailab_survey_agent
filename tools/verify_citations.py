"""B tool entry: verify survey citations (structural) + map claims to evidence (NLI). Thin orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from config import load_config
from llm_client import InternS2Client
from harness.json_io import read_json, write_json
from harness.logger import setup_logging

from tools.models.artifacts import FigureBank, Figure, TableBank, EvidenceStore, CitationIndex, ParsedPapers
from tools.nlp.nli_verifier import NLIVerifier
from tools.verify import structural, claim_mapper

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

    # citation_ready_set -> CitationIndex adapter (structural reads .citations)
    ready_set = read_json(resolve(inputs["citation_ready_set_path"]))
    citation_index = CitationIndex(
        task_id=task_id,
        citations=[{"paper_id": pid} for pid in ready_set.get("allowed_paper_ids", [])],
    )

    # figure validation: B's paper figures + C's generated artifacts (both referenced as ![...](id))
    figure_bank = FigureBank(**read_json(resolve(inputs["figure_bank_path"])))
    gen_bank = read_json(resolve(inputs["generated_artifact_bank_path"]))
    gen_figures = [
        Figure(figure_id=a["artifact_id"], paper_id="generated",
               caption=a.get("artifact_path", ""), page=0)
        for a in gen_bank.get("artifacts", []) if a.get("artifact_id")
    ]
    figure_bank = FigureBank(task_id=task_id, figures=list(figure_bank.figures) + gen_figures)

    table_bank = TableBank(**read_json(resolve(inputs["table_bank_path"])))  # unused by structural (deferred)
    evidence_store = EvidenceStore(**read_json(resolve(inputs["evidence_store_path"])))
    logger.info(f"[verify] start task_id={task_id} survey_md={len(survey_md)} chars "
                f"ready_citations={len(citation_index.citations)} evidence={len(evidence_store.evidence)}")

    # §4.2 structural + §4.3 claim mapping
    citation_result = structural.run(task_id, survey_md, citation_index, figure_bank, table_bank)
    logger.info(f"[verify] structural: total={citation_result.total_citations} "
                f"invalid={citation_result.invalid_citations} score={citation_result.citation_validity_score}")
    claim_map = claim_mapper.run(
        task_id, survey_md, evidence_store,
        ParsedPapers(task_id=task_id, papers=[]),  # unused by claim_mapper
        NLIVerifier(), InternS2Client(cfg))

    # write outputs
    write_json(resolve(outputs["citation_result_path"]), citation_result.model_dump())
    write_json(resolve(outputs["claim_map_path"]), claim_map.model_dump())

    # derive status: clean only if no invalid citations and no unsupported claims
    unsupported = sum(1 for e in claim_map.entries if e.status == "unsupported")
    status = "success" if (citation_result.invalid_citations == 0 and unsupported == 0) else "partial_success"
    logger.info(f"[verify] claim_map: {len(claim_map.entries)} entries, {unsupported} unsupported -> status={status}")

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
