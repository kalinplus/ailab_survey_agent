import logging
from tools.models.bundle import KnowledgeBundle

logger = logging.getLogger(__name__)


def run(task_id, topic, artifacts_paths, retrieved, parsed, paper_cards, evidence_store,
        figure_bank, table_bank, taxonomy, citation_index, structure_errors, coverage_warnings,
        quality_requirements):
    paper_count = len(retrieved.papers)
    core_count = sum(1 for c in paper_cards.paper_cards if c.card_type == "deep")
    warnings = list(structure_errors) + list(coverage_warnings)
    parse_rate = (len(parsed.papers) / paper_count) if paper_count else 0.0
    card_rate = (len(paper_cards.paper_cards) / paper_count) if paper_count else 0.0
    ev_rate = (sum(len(c.evidence_ids) for c in paper_cards.paper_cards) / max(core_count, 1))
    covered = sorted({a["aspect_id"] for c in paper_cards.paper_cards for a in c.matched_aspects})
    per_aspect = {}  # filled by caller if needed

    if paper_count >= quality_requirements.min_total_papers and core_count >= quality_requirements.min_core_papers:
        status = "success" if not warnings else "partial_success"
    else:
        status = "partial_success" if paper_count else "failed"

    logger.info(f"[P6] bundle status={status} papers={paper_count} core={core_count} "
                f"parsed={len(parsed.papers)} evidence={len(evidence_store.evidence)} "
                f"figures={len(figure_bank.figures)} citations={len(citation_index.citations)}")
    if status == "failed":
        logger.error(f"[P6] bundle FAILED (paper_count={paper_count})")

    return KnowledgeBundle(
        task_id=task_id, topic=topic, status=status, artifacts=artifacts_paths,
        summary={
            "paper_count": paper_count,
            "core_paper_count": core_count,
            "parsed_paper_count": len(parsed.papers),
            "figure_count": len(figure_bank.figures),
            "table_count": len(table_bank.tables),
            "evidence_count": len(evidence_store.evidence),
            "taxonomy_category_count": len(taxonomy.categories),
            "citation_count": len(citation_index.citations),
        },
        coverage_report={"covered_aspects": covered, "undercovered_aspects": [], "papers_per_aspect": per_aspect},
        quality_report={
            "search_success": paper_count > 0,
            "parse_success_rate": round(parse_rate, 3),
            "paper_card_success_rate": round(card_rate, 3),
            "evidence_coverage_rate": round(ev_rate, 3),
            "has_multimodal_evidence": len(figure_bank.figures) > 0,
            "warnings": warnings,
        },
    )
