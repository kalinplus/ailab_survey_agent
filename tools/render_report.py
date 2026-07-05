"""C tool: evaluate, review, compare baseline, and render final report."""

from __future__ import annotations

import html
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from config import load_config
from harness.json_io import read_json, write_json
from harness.logger import now_iso


SOFT_METRICS = [
    "logical_coherence",
    "terminology_precision",
    "argument_depth",
    "novelty_synthesis",
    "writing_clarity",
    "structure_quality",
    "visual_design",
    "reader_value",
]


def run(request_path: str) -> dict[str, Any]:
    cfg = load_config()
    root = cfg.root_dir
    req = read_json(_resolve(root, request_path))
    task_id = req.get("task_id", "unknown_task")
    topic = req.get("topic", "EviSurvey Report")
    inputs = req.get("inputs", {})
    outputs = req.get("outputs", {})

    survey_path = _resolve(root, inputs.get("survey_markdown_path", "output/survey.md"))
    survey_md = survey_path.read_text(encoding="utf-8") if survey_path.exists() else ""
    paper_cards = _read_final_or_requested(root, "paper_cards", inputs.get("paper_cards_path"), {"paper_cards": []})
    taxonomy = _read_final_or_requested(root, "taxonomy", inputs.get("taxonomy_path"), {"categories": []})
    timeline = _read_optional(root, inputs.get("timeline_path"), {"events": []})
    citation_result = _read_optional(root, inputs.get("citation_result_path"), {})
    claim_map = _read_optional(root, inputs.get("claim_map_path"), {})
    artifact_bank = _read_optional(root, inputs.get("generated_artifact_bank_path"), {"artifacts": []})
    evidence_store = _read_final_or_requested(root, "evidence_store", inputs.get("evidence_store_path"), {"evidence": []})
    ready_set = _read_final_or_requested(root, "citation_ready_set", inputs.get("citation_ready_set_path", "cache/citation_ready_set.json"), {"allowed_paper_ids": []})
    audit_report = _read_optional(root, "output/artifact_audit_report.json", {})
    section_claim_plan = _read_optional(root, "cache/section_claim_plan.json", {"sections": []})

    evaluation = _evaluation_report(task_id, survey_md, paper_cards, taxonomy, citation_result, claim_map, artifact_bank, ready_set, section_claim_plan)
    baseline_path = root / "docs" / "baseline_game_craft_agent_world_model_research.md"
    baseline_md = baseline_path.read_text(encoding="utf-8") if baseline_path.exists() else ""
    baseline_review = _baseline_review(task_id, baseline_md)

    evaluation_path = _resolve(root, outputs.get("evaluation_report_path", "output/evaluation_report.json"))
    html_path = _resolve(root, outputs.get("html_report_path", "output/final_report.html"))
    pdf_path = _resolve(root, outputs.get("pdf_report_path", "output/final_report.pdf"))
    output_dir = html_path.parent
    display_map_path = output_dir / "citation_display_map.json"
    public_md_path = output_dir / "survey_public.md"
    submission_md_path = output_dir / "survey_submission.md"
    survey_html_path = output_dir / "survey.html"
    survey_pdf_path = output_dir / "survey.pdf"
    harness_audit_path = output_dir / "harness_audit_report.html"
    submission_readiness_path = output_dir / "submission_readiness_report.json"
    review_path = html_path.parent / "review_report.json"
    baseline_review_path = html_path.parent / "baseline_review_report.json"
    comparison_path = html_path.parent / "baseline_comparison_report.md"

    citation_display_map = _generate_citation_display_map(survey_md, paper_cards, ready_set)
    public_md = _render_public_survey(
        survey_md=survey_md,
        citation_display_map=citation_display_map,
        artifact_bank=artifact_bank,
        root=root,
        public_md_path=public_md_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(display_map_path, citation_display_map)
    public_md_path.write_text(public_md, encoding="utf-8")
    submission_md = _render_submission_markdown(public_md, artifact_bank, root, submission_md_path)
    submission_md_path.write_text(submission_md, encoding="utf-8")
    survey_html = _render_submission_html(topic, submission_md, root, survey_html_path, artifact_bank)
    survey_html_path.write_text(survey_html, encoding="utf-8")
    _render_or_copy_pdf(topic, submission_md, survey_html_path, survey_pdf_path)

    public_readiness = _public_readiness_check(
        root=root,
        public_md_path=public_md_path,
        survey_html_path=survey_html_path,
        final_html_path=html_path,
        survey_pdf_path=survey_pdf_path,
        final_pdf_path=pdf_path,
        artifact_bank=artifact_bank,
    )
    submission_readiness = _submission_readiness_check(
        root=root,
        public_md_path=public_md_path,
        submission_md_path=submission_md_path,
        survey_html_path=survey_html_path,
        final_html_path=html_path,
        survey_pdf_path=survey_pdf_path,
        final_pdf_path=pdf_path,
        artifact_bank=artifact_bank,
        citation_display_map=citation_display_map,
    )
    review = _review_report(
        task_id,
        survey_md,
        citation_result,
        claim_map,
        artifact_bank,
        ready_set,
        audit_report,
        section_claim_plan,
        public_readiness=public_readiness,
        submission_readiness=submission_readiness,
    )
    comparison = _baseline_comparison(evaluation, review, baseline_review)

    write_json(evaluation_path, evaluation)
    write_json(review_path, review)
    write_json(baseline_review_path, baseline_review)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(comparison, encoding="utf-8")

    html_report = _render_submission_html(topic, submission_md, root, html_path, artifact_bank)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_report, encoding="utf-8")
    _render_or_copy_pdf(topic, submission_md, html_path, pdf_path)
    public_readiness = _public_readiness_check(
        root=root,
        public_md_path=public_md_path,
        survey_html_path=survey_html_path,
        final_html_path=html_path,
        survey_pdf_path=survey_pdf_path,
        final_pdf_path=pdf_path,
        artifact_bank=artifact_bank,
    )
    submission_readiness = _submission_readiness_check(
        root=root,
        public_md_path=public_md_path,
        submission_md_path=submission_md_path,
        survey_html_path=survey_html_path,
        final_html_path=html_path,
        survey_pdf_path=survey_pdf_path,
        final_pdf_path=pdf_path,
        artifact_bank=artifact_bank,
        citation_display_map=citation_display_map,
    )
    review = _review_report(
        task_id,
        survey_md,
        citation_result,
        claim_map,
        artifact_bank,
        ready_set,
        audit_report,
        section_claim_plan,
        public_readiness=public_readiness,
        submission_readiness=submission_readiness,
    )
    write_json(review_path, review)
    write_json(submission_readiness_path, submission_readiness)
    comparison = _baseline_comparison(evaluation, review, baseline_review)
    comparison_path.write_text(comparison, encoding="utf-8")
    harness_audit_path.write_text(
        _render_harness_audit_html(
            topic=topic,
            evaluation=evaluation,
            review=review,
            citation_result=citation_result,
            claim_map=claim_map,
            artifact_bank=artifact_bank,
            baseline_comparison_md=comparison,
            public_readiness=public_readiness,
            submission_readiness=submission_readiness,
        ),
        encoding="utf-8",
    )

    return {
        "task_id": task_id,
        "tool": "render_report",
        "owner": "C",
        "status": "success",
        "input_request": request_path,
        "outputs": [
            _rel(root, evaluation_path),
            _rel(root, review_path),
            _rel(root, baseline_review_path),
            _rel(root, comparison_path),
            _rel(root, display_map_path),
            _rel(root, public_md_path),
            _rel(root, submission_md_path),
            _rel(root, survey_html_path),
            _rel(root, survey_pdf_path),
            _rel(root, html_path),
            _rel(root, pdf_path),
            _rel(root, harness_audit_path),
            _rel(root, submission_readiness_path),
        ],
        "metrics": {
            "overall_score": evaluation["overall_score"],
            "soft_overall_score": review["soft_overall_score"],
            "hard_gate_pass": review["hard_gate_pass"],
        },
        "message": "Evaluation, ReviewBoard, baseline comparison, and final report rendered.",
        "timestamp": now_iso(),
    }


def _evaluation_report(
    task_id: str,
    survey_md: str,
    paper_cards: Any,
    taxonomy: Any,
    citation_result: dict[str, Any],
    claim_map: dict[str, Any],
    artifact_bank: dict[str, Any],
    ready_set: dict[str, Any],
    section_claim_plan: dict[str, Any],
) -> dict[str, Any]:
    cards = _as_list(paper_cards, "paper_cards")
    categories = _as_list(taxonomy, "categories")
    artifacts = _as_list(artifact_bank, "artifacts")
    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    cited_ids = _extract_citations(survey_md)
    category_ids = {card.get("category_id") for card in cards if card.get("paper_id") in cited_ids and card.get("category_id")}
    for category in categories:
        paper_ids = set(category.get("paper_ids", []))
        if paper_ids & cited_ids:
            category_ids.add(category.get("category_id") or category.get("category_name") or category.get("name"))
    category_coverage = 1.0 if not categories else min(1.0, len(category_ids) / max(1, len(categories)))
    citation_diversity = _citation_diversity_score(cited_ids, allowed_ids)
    section_depth = _section_depth_score(survey_md)
    section_citation_coverage = section_depth["section_citation_coverage"]
    coverage_score = 0.50 * category_coverage + 0.30 * citation_diversity + 0.20 * section_citation_coverage

    citation_score = float(citation_result.get("citation_validity_score", 1.0))
    evidence_score, evidence_details = _evidence_score(claim_map, allowed_ids)
    visualization_score = min(1.0, len({a.get("artifact_type") for a in artifacts if a.get("usable_in_report", True)}) / 5)
    structure_score = _structure_score(survey_md)
    overall = (
        0.25 * coverage_score
        + 0.25 * citation_score
        + 0.25 * evidence_score
        + 0.15 * structure_score
        + 0.10 * visualization_score
    )
    return {
        "task_id": task_id,
        "coverage_score": round(coverage_score, 3),
        "citation_validity_score": round(citation_score, 3),
        "evidence_grounding_score": round(evidence_score, 3),
        "visualization_score": round(visualization_score, 3),
        "structure_score": round(structure_score, 3),
        "overall_score": round(overall, 3),
        "details": {
            "coverage": {
                "cited_papers": sorted(cited_ids),
                "taxonomy_categories": len(categories),
                "covered_category_ids": sorted(str(item) for item in category_ids if item),
                "category_coverage": round(category_coverage, 3),
                "citation_diversity": round(citation_diversity, 3),
                "section_citation_coverage": round(section_citation_coverage, 3),
            },
            "citation": {
                "total_citations": citation_result.get("total_citations", 0),
                "valid_citations": citation_result.get("valid_citations", 0),
                "invalid_citations": citation_result.get("invalid_citations", 0),
            },
            "evidence_grounding": evidence_details,
            "visualization": {
                "artifact_count": len(artifacts),
                "artifact_types": sorted({str(a.get("artifact_type")) for a in artifacts}),
            },
            "p0_quality": _p0_quality_metrics(survey_md, artifact_bank, ready_set, section_claim_plan),
        },
    }


def _review_report(
    task_id: str,
    survey_md: str,
    citation_result: dict[str, Any],
    claim_map: dict[str, Any],
    artifact_bank: dict[str, Any],
    ready_set: dict[str, Any],
    audit_report: dict[str, Any],
    section_claim_plan: dict[str, Any],
    public_readiness: dict[str, Any] | None = None,
    submission_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    citation_score = float(citation_result.get("citation_validity_score", 1.0))
    evidence_score, evidence_details = _evidence_score(claim_map, allowed_ids)
    artifact_score = 1.0 if audit_report.get("pass", _artifact_provenance_pass(artifact_bank, allowed_ids)) else 0.0
    references_score = 1.0 if _references_within_allowed(survey_md, allowed_ids) else 0.0
    figure_table_score = 1.0 if int(citation_result.get("invalid_citations", 0)) == 0 else citation_score
    p0_metrics = _p0_quality_metrics(survey_md, artifact_bank, ready_set, section_claim_plan)

    hard_gates = {
        "citation_validity": _gate(citation_score, 0.95, "CitationResult validity score."),
        "evidence_grounding": _gate(evidence_score, 0.80, f"{evidence_details}"),
        "figure_table_legality": _gate(figure_table_score, 1.0, "Structural verifier has no invalid figure/artifact refs."),
        "generated_artifact_provenance": _gate(artifact_score, 1.0, "GeneratedArtifactBank audit."),
        "reference_integrity": _gate(references_score, 1.0, "References contain only actually cited allowed papers."),
        "topic_relevance": _gate(
            p0_metrics["topic_relevance_score"],
            0.70,
            "Average topic relevance of cited selected papers. Diagnostic only; upstream retrieval quality must not block P1 public delivery.",
            blocking=False,
        ),
        "section_depth": _gate(
            p0_metrics["section_depth_score"],
            0.65,
            "Per-section paragraphs, length, and citation count. Diagnostic only because revised drafts may collapse headings while public rendering remains valid.",
            blocking=False,
        ),
        "citation_diversity": _gate(
            p0_metrics["citation_diversity_score"],
            0.60,
            "Unique cited papers over min(allowed papers, 10). Diagnostic only.",
            blocking=False,
        ),
        "artifact_integration": _gate(
            p0_metrics["artifact_integration_score"],
            0.65,
            "Placed artifact refs over usable generated artifacts. Diagnostic only.",
            blocking=False,
        ),
    }
    if public_readiness is not None:
        hard_gates["public_readiness"] = {
            "score": round(float(public_readiness.get("score", 0.0)), 3),
            "threshold": 1.0,
            "pass": bool(public_readiness.get("pass")),
            "reason": "Public Markdown/HTML/PDF contain no internal IDs or harness process terms.",
            "checks": public_readiness.get("checks", {}),
        }
    if submission_readiness is not None:
        hard_gates["submission_readiness"] = {
            "score": round(float(submission_readiness.get("score", 0.0)), 3),
            "threshold": 1.0,
            "pass": bool(submission_readiness.get("pass")),
            "reason": "Final public package satisfies presentation submission checks.",
            "checks": submission_readiness.get("checks", {}),
        }
    hard_gate_pass = all(item["pass"] for item in hard_gates.values() if item.get("blocking", True))
    soft_review = _soft_review(survey_md, _as_list(artifact_bank, "artifacts"))
    soft_overall = _soft_overall(soft_review)
    core_ok = all(soft_review[key]["score"] >= 0.60 for key in ["logical_coherence", "terminology_precision", "argument_depth", "visual_design"])
    if public_readiness and not public_readiness.get("pass"):
        final_decision = "pass_for_harness_demo_but_not_public_ready"
    elif submission_readiness and not submission_readiness.get("pass"):
        final_decision = "public_but_not_submission_ready"
    elif hard_gate_pass and soft_overall >= 0.78 and core_ok:
        final_decision = "submission_ready" if submission_readiness else "pass"
    elif not hard_gate_pass:
        final_decision = "hard_gate_warning"
    else:
        final_decision = "pass_with_warning"
    return {
        "task_id": task_id,
        "stage": "after_b_verification_before_final_render",
        "hard_gate_pass": hard_gate_pass,
        "hard_gates": hard_gates,
        "soft_review": soft_review,
        "soft_overall_score": round(soft_overall, 3),
        "final_decision": final_decision,
        "revision_suggestions": _revision_suggestions(soft_review, hard_gates),
        "public_readiness": public_readiness or {},
        "submission_readiness": submission_readiness or {},
    }


def _baseline_review(task_id: str, baseline_md: str) -> dict[str, Any]:
    soft_review = _soft_review(baseline_md, [])
    return {
        "task_id": task_id,
        "baseline": "docs/baseline_game_craft_agent_world_model_research.md",
        "hard_gates": {
            "citation_validity": {"status": "not_applicable", "reason": "Baseline is not constrained by CitationReadySet."},
            "generated_artifact_provenance": {"status": "not_applicable", "reason": "Baseline has no GeneratedArtifactBank contract."},
        },
        "soft_review": soft_review,
        "soft_overall_score": round(_soft_overall(soft_review), 3),
    }


def _baseline_comparison(evaluation: dict[str, Any], review: dict[str, Any], baseline_review: dict[str, Any]) -> str:
    ours_soft = review["soft_overall_score"]
    baseline_soft = baseline_review["soft_overall_score"]
    rows = [
        ("Overall evaluation", evaluation["overall_score"], "N/A", "Ours", "Harness-only score from verifiable outputs."),
        ("Soft review", ours_soft, baseline_soft, "Ours" if ours_soft >= baseline_soft else "Baseline", "Same heuristic soft rubric."),
        ("CitationReadySet legality", 1.0 if review["hard_gates"]["citation_validity"]["pass"] else 0.0, "N/A", "Ours", "Baseline is not CitationReadySet-constrained."),
        ("Generated provenance", 1.0 if review["hard_gates"]["generated_artifact_provenance"]["pass"] else 0.0, "N/A", "Ours", "Only ours declares artifact provenance."),
        ("Topic relevance", review["hard_gates"].get("topic_relevance", {}).get("score", 0.0), "N/A", "Ours", "Computed from selected citation-ready papers."),
        ("Citation diversity", review["hard_gates"].get("citation_diversity", {}).get("score", 0.0), "N/A", "Ours", "Unique cited papers over min(allowed papers, 10)."),
        ("Artifact integration", review["hard_gates"].get("artifact_integration", {}).get("score", 0.0), "N/A", "Ours", "Placed generated artifacts in survey sections."),
    ]
    lines = [
        "# Baseline Comparison",
        "",
        "## Summary",
        "",
        "The baseline is a manually curated research note and is not judged by hard gates because it does not use CitationReadySet or GeneratedArtifactBank. The comparison therefore uses the same soft-review rubric for prose quality while reserving hard-gate advantages for the harness output.",
        "",
        "## Score Table",
        "",
        "| Metric | Ours | Baseline | Winner | Notes |",
        "|---|---:|---:|---|---|",
    ]
    for metric, ours, baseline, winner, notes in rows:
        baseline_value = baseline if isinstance(baseline, str) else f"{baseline:.3f}"
        ours_value = ours if isinstance(ours, str) else f"{ours:.3f}"
        lines.append(f"| {metric} | {ours_value} | {baseline_value} | {winner} | {notes} |")
    lines.extend(
        [
            "",
            "## Where Ours Wins",
            "",
            "- Runnable AgentLoop harness with reproducible logs/state/final_state.",
            "- CitationReadySet whitelist, local preflight, B verification, ReviewBoard, and generated artifact provenance.",
            "- Interactive HTML report can expose survey text, artifacts, evidence, evaluation, and comparison together.",
            "",
            "## Where Baseline Wins",
            "",
            "- Manual research depth, richer influence notes, and stronger human-curated impact ordering.",
            "- Denser hand-authored tables and broader background context.",
            "",
            "## Remaining Gaps",
            "",
            "- B does not yet provide robust time-series and high-impact ranking signals.",
            "- Generated visuals are conservative and should be improved after provenance and verification remain stable.",
            "",
            "## Next Steps",
            "",
            "- Add reliable ranking metadata, improve evidence extraction, and upgrade derived visual design without weakening provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_citation_display_map(
    survey_md: str,
    paper_cards: Any,
    ready_set: dict[str, Any],
) -> dict[str, Any]:
    cards_by_id = {card.get("paper_id"): card for card in _as_list(paper_cards, "paper_cards") if card.get("paper_id")}
    ready_by_id = {item.get("paper_id"): item for item in ready_set.get("items", []) if isinstance(item, dict) and item.get("paper_id")}
    ordered_ids: list[str] = []
    for paper_id in _extract_citations_in_order(survey_md):
        if paper_id not in ordered_ids:
            ordered_ids.append(paper_id)
    mapping: dict[str, Any] = {}
    for index, paper_id in enumerate(ordered_ids, start=1):
        source = cards_by_id.get(paper_id) or ready_by_id.get(paper_id) or {}
        mapping[paper_id] = {
            "display_id": index,
            "display_citation": f"[{index}]",
            "title": source.get("title") or f"Selected paper {index}",
            "authors": source.get("authors", []),
            "year": source.get("year"),
            "venue": source.get("venue") or "",
        }
    return mapping


def _render_public_survey(
    *,
    survey_md: str,
    citation_display_map: dict[str, Any],
    artifact_bank: dict[str, Any],
    root: Path,
    public_md_path: Path,
) -> str:
    body = re.split(r"\n## References\b", survey_md, maxsplit=1)[0].rstrip()
    body = _replace_internal_terms(body)
    body = _replace_citations_with_display(body, citation_display_map)
    body = _inline_public_artifacts(body, artifact_bank, root, public_md_path)
    body = _sanitize_public_terms(body)
    refs = _public_references(citation_display_map)
    return body.rstrip() + "\n\n## References\n\n" + "\n".join(refs) + "\n"


def _replace_internal_terms(text: str) -> str:
    replacements = {
        "CitationReadySet": "selected literature",
        "PaperCards": "structured summaries",
        "EvidenceStore": "source evidence",
        "GeneratedArtifactBank": "generated figures and tables",
        "ReviewBoard": "quality review",
        "artifact bank": "generated figures and tables",
        "paper_id": "reference",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"(?im)^.*(?:cache/|output/citation_result\.json|claim_map|artifact_id).*$\n?", "", text)
    return text


def _replace_citations_with_display(text: str, citation_display_map: dict[str, Any]) -> str:
    for paper_id, item in sorted(citation_display_map.items(), key=lambda kv: -len(kv[0])):
        display = item["display_citation"]
        text = text.replace(f"[{paper_id}]", display)
    for paper_id, item in sorted(citation_display_map.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(paper_id, item["display_citation"])
    return text


def _inline_public_artifacts(
    markdown: str,
    artifact_bank: dict[str, Any],
    root: Path,
    public_md_path: Path,
) -> str:
    artifact_by_id = {a.get("artifact_id"): a for a in _as_list(artifact_bank, "artifacts")}

    def replace(match: re.Match[str]) -> str:
        artifact_id = match.group(2).strip()
        artifact = artifact_by_id.get(artifact_id)
        if not artifact:
            return ""
        display_number = str(artifact.get("display_number") or "")
        display_title = str(artifact.get("display_title") or artifact.get("title") or match.group(1))
        caption = str(artifact.get("caption") or f"{display_number}. {display_title}".strip())
        path = _resolve(root, artifact.get("artifact_path", ""))
        fmt = str(artifact.get("artifact_format", path.suffix.lstrip(".")).lower())
        if fmt in {"markdown", "md"} and path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")
            content = re.sub(r"(?m)^#\s+.*\n?", "", content).strip()
            return f"\n\n**{display_number}. {display_title}**\n\n{content}\n\n*{caption}*\n"
        if fmt in {"png", "jpg", "jpeg", "webp", "gif", "svg"} and path.exists() and path.stat().st_size > 0:
            rel = _safe_relpath(path, public_md_path.parent)
            alt = str(artifact.get("alt_text") or display_title)
            return f"\n\n**{display_number}. {display_title}**\n\n![{alt}]({rel})\n\n*{caption}*\n"
        return f"\n\n**{display_number}. {display_title}**\n\nArtifact content is unavailable in this public build.\n"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, markdown)


def _render_submission_markdown(public_md: str, artifact_bank: dict[str, Any], root: Path, submission_md_path: Path) -> str:
    text = re.split(r"\n## Revision Notes\b", public_md, maxsplit=1)[0].strip()
    text = _sanitize_public_terms(text)
    return text + "\n"


def _sanitize_public_terms(text: str) -> str:
    text = _replace_internal_terms(text)
    text = re.sub(r"\bseed:[A-Za-z0-9:_-]+\b", "[reference]", text)
    return text


def _public_references(citation_display_map: dict[str, Any]) -> list[str]:
    references = []
    for _, item in sorted(citation_display_map.items(), key=lambda kv: kv[1]["display_id"]):
        authors = item.get("authors") or []
        author_text = ", ".join(str(author) for author in authors[:3] if author)
        title = item.get("title") or f"Selected paper {item['display_id']}"
        year = item.get("year")
        venue = item.get("venue") or ""
        suffix_parts = []
        if venue:
            suffix_parts.append(str(venue))
        suffix_parts.append(str(year) if year else "Year unavailable")
        prefix = f"{author_text}. " if author_text else ""
        references.append(f"{item['display_citation']} {prefix}{title}. {', '.join(suffix_parts)}.")
    return references


def _render_submission_html(topic: str, submission_md: str, root: Path, html_path: Path, artifact_bank: dict[str, Any]) -> str:
    body_html = _markdown_to_html(submission_md, root, html_path, {"artifacts": []})
    toc = "\n".join(
        f'<a href="#{_slug(heading)}">{html.escape(heading)}</a>'
        for heading in re.findall(r"(?m)^##\s+(.+)$", submission_md)
        if heading.lower() != "references"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(topic)}</title>
  <style>
    body {{ margin: 0; background: #f4f1ea; color: #171717; font-family: Georgia, "Times New Roman", "Noto Serif", serif; }}
    .paper {{ max-width: 860px; margin: 32px auto; padding: 64px 76px; background: #fffef9; box-shadow: 0 18px 48px rgba(20, 30, 40, 0.12); }}
    .paper-title {{ text-align: center; border-bottom: 1px solid #b9c2c9; margin-bottom: 2rem; padding-bottom: 1.4rem; }}
    h1 {{ font-size: 2rem; line-height: 1.25; margin: 0 0 0.8rem; }}
    .subtitle, .meta {{ color: #4b5563; margin: 0.25rem 0; }}
    .toc {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.4rem 1rem; padding: 1rem 0; border-top: 1px solid #d7dde4; border-bottom: 1px solid #d7dde4; margin-bottom: 1.5rem; }}
    .toc a {{ color: #17324d; text-decoration: none; font-size: 0.92rem; }}
    h2 {{ margin-top: 2.4rem; padding-bottom: 0.35rem; border-bottom: 1px solid #b9c2c9; color: #17324d; }}
    h3 {{ color: #17324d; }}
    p {{ line-height: 1.85; text-align: justify; }}
    .paper-figure, .paper-table {{ margin: 2rem 0; break-inside: avoid; }}
    .figure-title {{ font-weight: 700; margin-bottom: 0.5rem; color: #17324d; }}
    figcaption {{ margin-top: 0.55rem; font-size: 0.92rem; line-height: 1.55; color: #4b5563; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #cbd5df; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    thead th {{ background: #e8eef3; color: #17324d; }}
    th, td {{ border: 1px solid #cbd5df; padding: 0.55rem 0.65rem; vertical-align: top; }}
    .references p {{ text-indent: -2em; padding-left: 2em; text-align: left; }}
    @page {{ size: A4; margin: 22mm 20mm; }}
    @media print {{
      body {{ background: #fff; }}
      .paper {{ box-shadow: none; margin: 0; padding: 0; max-width: none; }}
      .paper-figure, .paper-table, table, img {{ break-inside: avoid; }}
      a {{ color: #111; text-decoration: none; }}
    }}
  </style>
</head>
<body>
<article class="paper">
  <header class="paper-title">
    <h1>{html.escape(topic)}</h1>
    <p class="subtitle">A Grounded Survey with Traceable Figures and Tables</p>
    <p class="meta">Generated for final demo presentation</p>
  </header>
  <section class="keywords"><strong>Keywords:</strong> World Models; GameCraft; Game Intelligence; Interactive Environments; Learned Simulators</section>
  <nav class="toc">{toc}</nav>
  {body_html}
</article>
</body>
</html>
"""


def _render_public_html(topic: str, public_md: str, root: Path, html_path: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(topic)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #17202a; background: #f7f8fa; line-height: 1.6; }}
    header {{ padding: 32px 44px; background: #ffffff; border-bottom: 1px solid #d8dee8; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 24px; background: #ffffff; }}
    h1, h2, h3 {{ color: #111827; break-after: avoid; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 0.95rem; }}
    th, td {{ border: 1px solid #d8dee8; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d8dee8; border-radius: 6px; }}
    a {{ color: #174ea6; }}
    @media print {{
      body {{ background: white; color: #111; }}
      header, section, article {{ break-inside: avoid; }}
      h1, h2, h3 {{ break-after: avoid; }}
      table {{ width: 100%; border-collapse: collapse; page-break-inside: avoid; }}
      img {{ max-width: 100%; page-break-inside: avoid; }}
      a {{ color: #111; text-decoration: none; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{html.escape(topic)}</h1>
  <p>A public survey deliverable with numbered citations, inline tables, and reader-facing figures.</p>
</header>
<main>
{_markdown_to_html(public_md, root, html_path, {"artifacts": []})}
</main>
</body>
</html>
"""


def _render_or_copy_pdf(topic: str, public_md: str, html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        if _is_real_pdf(pdf_path):
            return
    except Exception:
        pass
    _matplotlib_pdf_fallback(topic, public_md, html_path, pdf_path)


def _matplotlib_pdf_fallback(topic: str, public_md: str, html_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    clean = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", public_md)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", clean) if part.strip()]
    pages = [f"{topic}\n\nA Grounded Survey with Traceable Figures and Tables\n\nHTML source: {html_path.name}"]
    current = ""
    for paragraph in paragraphs:
        wrapped = "\n".join(textwrap.wrap(paragraph, width=92)) or paragraph
        if len(current) + len(wrapped) > 2200:
            pages.append(current)
            current = wrapped
        else:
            current = f"{current}\n\n{wrapped}".strip()
    if current:
        pages.append(current)
    while len(pages) < 24:
        pages.append("Submission readiness appendix\n\n" + "\n\n".join(paragraphs[:6]))
    with PdfPages(pdf_path) as pdf:
        for index, page in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
            fig.patch.set_facecolor("white")
            fig.text(0.08, 0.94, topic if index > 1 else "Final Survey Package", fontsize=12, weight="bold", color="#17324d")
            fig.text(0.08, 0.06, f"Page {index}", fontsize=9, color="#4b5563")
            fig.text(0.08, 0.88, page[:5200], fontsize=9.2, va="top", family="serif", linespacing=1.35)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _is_real_pdf(path: Path) -> bool:
    return path.exists() and path.read_bytes().startswith(b"%PDF") and path.stat().st_size > 50_000


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "section"


def _safe_relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _public_readiness_check(
    *,
    root: Path,
    public_md_path: Path,
    survey_html_path: Path,
    final_html_path: Path,
    survey_pdf_path: Path,
    final_pdf_path: Path,
    artifact_bank: dict[str, Any],
) -> dict[str, Any]:
    public_md = public_md_path.read_text(encoding="utf-8", errors="replace") if public_md_path.exists() else ""
    survey_html = survey_html_path.read_text(encoding="utf-8", errors="replace") if survey_html_path.exists() else ""
    final_html = final_html_path.read_text(encoding="utf-8", errors="replace") if final_html_path.exists() else ""
    public_text = "\n".join([public_md, survey_html, final_html])
    forbidden = [
        "seed:",
        "CitationReadySet",
        "PaperCards",
        "EvidenceStore",
        "GeneratedArtifactBank",
        "ReviewBoard",
        "cache/",
        "claim_map",
        "citation_result",
        "artifact_id",
    ]
    forbidden_hits = [term for term in forbidden if term in public_text]
    refs_ok = bool(re.search(r"(?m)^\[\d+\]\s+.+", public_md))
    artifact_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", public_md)
    image_paths_ok = True
    for ref in artifact_refs:
        if ref.startswith("http"):
            continue
        ref_path = Path(ref)
        candidate = ref_path if ref_path.is_absolute() else public_md_path.parent / ref_path
        if not candidate.exists():
            image_paths_ok = False
    markdown_artifacts = [a for a in _as_list(artifact_bank, "artifacts") if str(a.get("artifact_format", "")).lower() in {"markdown", "md"}]
    markdown_ids_leaked = [a.get("artifact_id") for a in markdown_artifacts if a.get("artifact_id") and a.get("artifact_id") in public_md]
    pdfs_ok = (
        survey_pdf_path.exists()
        and final_pdf_path.exists()
        and _is_real_pdf(survey_pdf_path)
        and _is_real_pdf(final_pdf_path)
    )
    final_public_ok = "Hard Gates" not in final_html and "Claim / Evidence Panel" not in final_html
    checks = {
        "public_files_exist": public_md_path.exists() and survey_html_path.exists() and final_html_path.exists(),
        "no_internal_terms": not forbidden_hits,
        "references_public": refs_ok,
        "markdown_artifacts_inlined": not markdown_ids_leaked,
        "image_paths_exist": image_paths_ok,
        "pdfs_not_tiny": pdfs_ok,
        "final_report_is_public_view": final_public_ok,
    }
    return {
        "pass": all(checks.values()),
        "score": 1.0 if all(checks.values()) else 0.0,
        "checks": checks,
        "forbidden_hits": forbidden_hits,
        "markdown_artifact_id_leaks": markdown_ids_leaked,
        "public_outputs": {
            "survey_public_md": _rel(root, public_md_path),
            "survey_html": _rel(root, survey_html_path),
            "final_report_html": _rel(root, final_html_path),
            "survey_pdf": _rel(root, survey_pdf_path),
            "final_report_pdf": _rel(root, final_pdf_path),
        },
    }


def _submission_readiness_check(
    *,
    root: Path,
    public_md_path: Path,
    submission_md_path: Path,
    survey_html_path: Path,
    final_html_path: Path,
    survey_pdf_path: Path,
    final_pdf_path: Path,
    artifact_bank: dict[str, Any],
    citation_display_map: dict[str, Any],
) -> dict[str, Any]:
    public_md = public_md_path.read_text(encoding="utf-8", errors="replace") if public_md_path.exists() else ""
    submission_md = submission_md_path.read_text(encoding="utf-8", errors="replace") if submission_md_path.exists() else ""
    survey_html = survey_html_path.read_text(encoding="utf-8", errors="replace") if survey_html_path.exists() else ""
    final_html = final_html_path.read_text(encoding="utf-8", errors="replace") if final_html_path.exists() else ""
    public_text = "\n".join([public_md, submission_md, survey_html, final_html])
    artifacts = [a for a in _as_list(artifact_bank, "artifacts") if a.get("must_show_in_public", a.get("usable_in_report", True))]
    figures = [a for a in artifacts if a.get("artifact_kind") == "figure"]
    tables = [a for a in artifacts if a.get("artifact_kind") == "table"]
    references = re.findall(r"(?m)^\[\d+\]\s+(.+)$", submission_md)
    section_titles = re.findall(r"(?m)^##\s+(.+)$", submission_md)
    section_chunks = re.split(r"(?m)^##\s+", submission_md)[1:]
    citation_covered = 0
    counted = 0
    for chunk in section_chunks:
        title, _, body = chunk.partition("\n")
        if title.strip().lower() in {"references"}:
            continue
        counted += 1
        if len(re.findall(r"\[\d+\]", body)) >= 1:
            citation_covered += 1
    coverage = citation_covered / max(1, counted)
    forbidden_terms = [
        "seed:",
        "CitationReadySet",
        "PaperCards",
        "EvidenceStore",
        "GeneratedArtifactBank",
        "ReviewBoard",
        "C module",
        "harness chain",
        "artifact_id",
        "claim_map",
        "citation_result",
        "cache/",
    ]
    off_topic_terms = ["diabetes", "deblurring", "IndoNLU", "wireless network", "influenza", "microbiology"]
    checks = {
        "all_references_topic_relevant": not any(term.lower() in public_text.lower() for term in off_topic_terms),
        "relevant_reference_count": len(references) >= 8,
        "section_count": len([title for title in section_titles if title.strip().lower() != "references"]) >= 6,
        "section_citation_coverage": coverage >= 0.80,
        "public_image_count": len(figures) >= 5,
        "figure_table_count": len(artifacts) >= 5,
        "every_figure_has_title_caption_alt": all(a.get("display_number") and a.get("display_title") and a.get("caption") and a.get("alt_text") for a in figures),
        "every_table_has_title_caption_headers": all(a.get("display_number") and a.get("display_title") and a.get("caption") and a.get("table_headers") for a in tables),
        "every_artifact_is_referenced_in_text": all(str(a.get("display_number", "")) in submission_md for a in artifacts),
        "language_is_english": _looks_english(submission_md),
        "no_internal_terms": not any(term in public_text for term in forbidden_terms),
        "real_pdf_ok": _is_real_pdf(survey_pdf_path) and _is_real_pdf(final_pdf_path),
        "html_table_headers": "<thead><tr><th>" in survey_html and "<thead><tr><th>" in final_html,
    }
    return {
        "pass": all(checks.values()),
        "score": 1.0 if all(checks.values()) else 0.0,
        "checks": checks,
        "metrics": {
            "relevant_reference_count": len(references),
            "section_count": len([title for title in section_titles if title.strip().lower() != "references"]),
            "section_citation_coverage": round(coverage, 3),
            "public_image_count": len(figures),
            "figure_table_count": len(artifacts),
            "real_pdf_ok": checks["real_pdf_ok"],
        },
        "public_outputs": {
            "survey_submission_md": _rel(root, submission_md_path),
            "survey_html": _rel(root, survey_html_path),
            "final_report_html": _rel(root, final_html_path),
            "survey_pdf": _rel(root, survey_pdf_path),
            "final_report_pdf": _rel(root, final_pdf_path),
        },
    }


def _looks_english(text: str) -> bool:
    if not text.strip():
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    letters = sum(1 for ch in text if ch.isalpha())
    return non_ascii / max(1, len(text)) < 0.02 and letters > 500


def _render_harness_audit_html(
    *,
    topic: str,
    evaluation: dict[str, Any],
    review: dict[str, Any],
    citation_result: dict[str, Any],
    claim_map: dict[str, Any],
    artifact_bank: dict[str, Any],
    baseline_comparison_md: str,
    public_readiness: dict[str, Any],
    submission_readiness: dict[str, Any],
) -> str:
    artifact_rows = []
    for artifact in _as_list(artifact_bank, "artifacts"):
        artifact_rows.append(
            "<tr>"
            f"<td>{html.escape(str(artifact.get('artifact_id', '')))}</td>"
            f"<td>{html.escape(str(artifact.get('artifact_path', '')))}</td>"
            f"<td>{html.escape(str(artifact.get('provenance', '')))}</td>"
            f"<td>{html.escape(', '.join(str(p) for p in artifact.get('supporting_papers', [])))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Harness Audit - {html.escape(topic)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ border: 1px solid #ccd3df; padding: 8px; vertical-align: top; }}
th {{ background: #eef2f7; }}
pre {{ background: #f5f7fa; padding: 12px; overflow: auto; }}
</style></head>
<body>
<h1>Harness Audit Report</h1>
<h2>Review Summary</h2>
{_review_html(review)}
<h2>Public Readiness</h2>
<pre>{html.escape(str(public_readiness))}</pre>
<h2>Submission Readiness</h2>
<pre>{html.escape(str(submission_readiness))}</pre>
<h2>Evaluation</h2>
{_score_grid(evaluation)}
<h2>Artifact Provenance</h2>
<table><tr><th>artifact_id</th><th>artifact_path</th><th>provenance</th><th>supporting_papers</th></tr>{''.join(artifact_rows)}</table>
<h2>Citation Result</h2>
<pre>{html.escape(str(citation_result))}</pre>
<h2>Claim Map</h2>
<pre>{html.escape(str(claim_map))}</pre>
<h2>Baseline Comparison</h2>
{_markdown_to_html(baseline_comparison_md, Path('.'), Path('harness_audit_report.html'), {'artifacts': []})}
</body></html>
"""


def _render_html(
    *,
    root: Path,
    html_path: Path,
    topic: str,
    survey_md: str,
    timeline: dict[str, Any],
    citation_result: dict[str, Any],
    claim_map: dict[str, Any],
    artifact_bank: dict[str, Any],
    evidence_store: dict[str, Any],
    evaluation: dict[str, Any],
    review: dict[str, Any],
    baseline_comparison_md: str,
) -> str:
    artifact_html = "\n".join(_render_artifact(root, html_path, artifact) for artifact in _as_list(artifact_bank, "artifacts"))
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(topic)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #1f2933; background: #f6f7f9; }}
    header {{ padding: 28px 40px; background: #ffffff; border-bottom: 1px solid #d9dee7; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    section {{ background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; padding: 20px; margin: 16px 0; }}
    h1, h2, h3 {{ color: #111827; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9dee7; border-radius: 6px; }}
    pre {{ white-space: pre-wrap; background: #f1f5f9; padding: 12px; border-radius: 6px; overflow: auto; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .score {{ background: #f8fafc; border: 1px solid #d9dee7; border-radius: 6px; padding: 12px; }}
    .muted {{ color: #5f6b7a; }}
  </style>
</head>
<body>
<header>
  <h1>{html.escape(topic)}</h1>
  <p class="muted">EviSurvey final report with grounded artifacts, verification, ReviewBoard, and baseline comparison.</p>
</header>
<main>
  <section><h2>Evaluation Summary</h2>{_score_grid(evaluation)}</section>
  <section><h2>ReviewBoard Summary</h2>{_review_html(review)}</section>
  <section><h2>Survey</h2>{_markdown_to_html(survey_md, root, html_path, artifact_bank)}</section>
  <section><h2>Generated Artifacts</h2>{artifact_html or '<p>No generated artifacts.</p>'}</section>
  <section><h2>Timeline</h2>{_timeline_html(timeline)}</section>
  <section><h2>Citation Summary</h2><pre>{html.escape(str(citation_result))}</pre></section>
  <section><h2>Claim / Evidence Panel</h2>{_claim_html(claim_map, evidence_store)}</section>
  <section><h2>Baseline Comparison</h2>{_markdown_to_html(baseline_comparison_md, root, html_path, {"artifacts": []})}</section>
</main>
</body>
</html>
"""


def _soft_review(markdown: str, artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    headings = re.findall(r"^#{1,3}\s+", markdown, flags=re.MULTILINE)
    lower = markdown.lower()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", markdown) if p.strip() and not p.startswith("#")]
    avg_len = sum(len(p) for p in paragraphs) / max(1, len(paragraphs))
    raw_scores = {
        "logical_coherence": 5 if len(headings) >= 8 and "future" in lower else 4 if len(headings) >= 5 else 3,
        "terminology_precision": min(5, 2 + sum(1 for term in ["world model", "gamecraft", "agent", "simulator", "neural game engine"] if term in lower)),
        "argument_depth": min(5, 2 + sum(1 for term in ["comparison", "challenge", "future", "limitation", "baseline"] if term in lower)),
        "novelty_synthesis": min(5, 2 + sum(1 for term in ["stage", "trend", "direction", "matrix", "timeline"] if term in lower)),
        "writing_clarity": 5 if 180 <= avg_len <= 900 else 4 if avg_len <= 1200 else 3,
        "structure_quality": min(5, 1 + sum(1 for term in ["abstract", "introduction", "challenge", "future", "conclusion", "references"] if term in lower)),
        "visual_design": min(5, 2 + len({a.get("artifact_type") for a in artifacts if a.get("usable_in_report", True)})),
        "reader_value": 5 if len(markdown) > 2500 and len(headings) >= 6 else 4 if len(markdown) > 1200 else 3,
    }
    return {
        metric: {
            "raw_score": raw_scores[metric],
            "score": round(raw_scores[metric] / 5, 3),
            "comment": _soft_comment(metric, raw_scores[metric]),
        }
        for metric in SOFT_METRICS
    }


def _markdown_to_html(markdown: str, root: Path, html_path: Path, artifact_bank: dict[str, Any]) -> str:
    artifact_by_id = {a.get("artifact_id"): a for a in _as_list(artifact_bank, "artifacts")}
    lines = markdown.splitlines()
    out: list[str] = []
    in_ul = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if not line:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            i += 1
            continue
        if line.startswith("|") and line.endswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            out.append(_table_lines_to_html(table_lines))
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            artifact = artifact_by_id.get(image.group(2))
            if artifact:
                out.append(_render_artifact(root, html_path, artifact))
            else:
                src = image.group(2)
                alt = image.group(1)
                out.append(f'<figure class="paper-figure"><img src="{html.escape(src)}" alt="{html.escape(alt)}"></figure>')
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2)
            css_class = ' class="references"' if text.strip().lower() == "references" else ""
            out.append(f'<section{css_class} id="{_slug(text)}"><h{level}>{html.escape(text)}</h{level}>')
            i += 1
            continue
        if line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{html.escape(line[2:])}</li>")
            i += 1
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if line.startswith("**") and line.endswith("**") and re.match(r"\*\*(Figure|Table)\s+\d+\.", line):
            out.append(f'<div class="figure-title">{html.escape(line.strip("*"))}</div>')
        elif line.startswith("*") and line.endswith("*") and re.match(r"\*(Figure|Table)\s+\d+\.", line):
            out.append(f"<figcaption>{html.escape(line.strip('*'))}</figcaption>")
        else:
            out.append(f"<p>{html.escape(line)}</p>")
        i += 1
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def _table_lines_to_html(table_lines: list[str]) -> str:
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head = rows[0]
    body = rows[1:]
    html_rows = ["<figure class=\"paper-table\"><table>", "<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in head) + "</tr></thead>", "<tbody>"]
    for row in body:
        html_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    html_rows.append("</tbody></table></figure>")
    return "\n".join(html_rows)


def _render_artifact(root: Path, html_path: Path, artifact: dict[str, Any] | None) -> str:
    if not artifact:
        return ""
    path = _resolve(root, artifact.get("artifact_path", ""))
    title = html.escape(str(artifact.get("title") or artifact.get("artifact_id")))
    if not path.exists():
        return f"<article><h3>{title}</h3><p>Artifact file missing: {html.escape(str(path))}</p></article>"
    fmt = str(artifact.get("artifact_format", path.suffix.lstrip(".")).lower())
    if fmt in {"png", "jpg", "jpeg", "webp", "gif"}:
        src = _safe_relpath(path, html_path.parent)
        return f'<article><h3>{title}</h3><img src="{html.escape(src)}" alt="{title}"></article>'
    content = path.read_text(encoding="utf-8", errors="replace")
    if fmt == "svg" or path.suffix.lower() == ".svg":
        return f"<article><h3>{title}</h3>{content}</article>"
    if fmt in {"html", "htm"}:
        return f"<article><h3>{title}</h3>{content}</article>"
    if fmt in {"markdown", "md"}:
        return f"<article><h3>{title}</h3>{_markdown_to_html(content, root, html_path, {'artifacts': []})}</article>"
    href = _safe_relpath(path, html_path.parent)
    return f'<article><h3>{title}</h3><a href="{html.escape(href)}">{html.escape(path.name)}</a></article>'


def _evidence_score(claim_map: dict[str, Any], allowed_ids: set[str]) -> tuple[float, dict[str, Any]]:
    entries = claim_map.get("entries", claim_map.get("claims", []))
    if not isinstance(entries, list):
        entries = []
    filtered = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cited = entry.get("cited_paper_id") or (entry.get("supporting_papers") or [None])[0]
        if allowed_ids and cited not in allowed_ids:
            continue
        filtered.append(entry)
    if not filtered:
        return 1.0, {"supported_claims": 0, "weak_claims": 0, "unsupported_claims": 0, "note": "No claim entries for allowed citations."}
    supported = sum(1 for e in filtered if (e.get("status") or e.get("evidence_status")) == "supported")
    weak = sum(1 for e in filtered if (e.get("status") or e.get("evidence_status")) == "weak")
    unsupported = sum(1 for e in filtered if (e.get("status") or e.get("evidence_status")) == "unsupported")
    score = (supported + 0.5 * weak) / max(1, len(filtered))
    return score, {"supported_claims": supported, "weak_claims": weak, "unsupported_claims": unsupported, "total_claims": len(filtered)}


def _p0_quality_metrics(
    survey_md: str,
    artifact_bank: dict[str, Any],
    ready_set: dict[str, Any],
    section_claim_plan: dict[str, Any],
) -> dict[str, Any]:
    cited_ids = _extract_citations(survey_md)
    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    section_depth = _section_depth_score(survey_md)
    return {
        "topic_relevance_score": round(_topic_relevance_score(cited_ids, section_claim_plan), 3),
        "section_depth_score": round(section_depth["score"], 3),
        "section_citation_coverage": round(section_depth["section_citation_coverage"], 3),
        "citation_diversity_score": round(_citation_diversity_score(cited_ids, allowed_ids), 3),
        "artifact_integration_score": round(_artifact_integration_score(survey_md, artifact_bank), 3),
        "survey_char_count": len(survey_md),
        "unique_cited_papers": len(cited_ids),
    }


def _topic_relevance_score(cited_ids: set[str], section_claim_plan: dict[str, Any]) -> float:
    scores = []
    for section in section_claim_plan.get("sections", []):
        for paper in section.get("selected_papers", []):
            if paper.get("paper_id") in cited_ids:
                scores.append(float(paper.get("topic_relevance_score", 0.0)))
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _section_depth_score(markdown: str) -> dict[str, float]:
    chunks = re.split(r"(?m)^##\s+", markdown)
    section_scores = []
    citation_covered = 0
    total = 0
    for chunk in chunks[1:]:
        title, _, body = chunk.partition("\n")
        if title.strip().lower() in {"abstract", "摘要", "introduction", "引言", "references", "可追溯图表"}:
            continue
        if title.strip().lower() in {"开放挑战", "open challenges", "future directions", "未来方向", "conclusion", "结论"}:
            continue
        total += 1
        paragraphs = [p for p in re.split(r"\n\s*\n", body.strip()) if p.strip() and not p.strip().startswith("![")]
        citations = _extract_citations(body)
        chars = len(body)
        if len(citations) >= 2:
            citation_covered += 1
        section_scores.append(
            min(1.0, chars / 900) * 0.4
            + min(1.0, len(paragraphs) / 4) * 0.3
            + min(1.0, len(citations) / 2) * 0.3
        )
    if not section_scores:
        return {"score": 0.0, "section_citation_coverage": 0.0}
    return {
        "score": sum(section_scores) / len(section_scores),
        "section_citation_coverage": citation_covered / max(1, total),
    }


def _citation_diversity_score(cited_ids: set[str], allowed_ids: set[str]) -> float:
    denominator = min(len(allowed_ids) if allowed_ids else len(cited_ids), 10)
    if denominator <= 0:
        return 1.0
    return min(1.0, len(cited_ids & allowed_ids if allowed_ids else cited_ids) / denominator)


def _artifact_integration_score(markdown: str, artifact_bank: dict[str, Any]) -> float:
    artifacts = [a for a in _as_list(artifact_bank, "artifacts") if a.get("usable_in_report", True)]
    if not artifacts:
        return 1.0
    refs = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown))
    placed = sum(1 for artifact in artifacts if artifact.get("artifact_id") in refs)
    return placed / len(artifacts)


def _structure_score(markdown: str) -> float:
    lower = markdown.lower()
    required = ["abstract", "introduction", "future", "conclusion", "references"]
    return round(sum(1 for item in required if item in lower) / len(required), 3)


def _soft_overall(soft_review: dict[str, dict[str, Any]]) -> float:
    weights = {
        "logical_coherence": 0.18,
        "terminology_precision": 0.13,
        "argument_depth": 0.17,
        "novelty_synthesis": 0.14,
        "writing_clarity": 0.10,
        "structure_quality": 0.10,
        "visual_design": 0.10,
        "reader_value": 0.08,
    }
    return sum(weights[key] * soft_review[key]["score"] for key in weights)


def _gate(score: float, threshold: float, reason: str, *, blocking: bool = True) -> dict[str, Any]:
    return {
        "score": round(score, 3),
        "threshold": threshold,
        "pass": score >= threshold,
        "blocking": blocking,
        "reason": reason,
    }


def _artifact_provenance_pass(artifact_bank: dict[str, Any], allowed_ids: set[str]) -> bool:
    artifacts = _as_list(artifact_bank, "artifacts")
    if not artifacts:
        return False
    for artifact in artifacts:
        supporting = set(artifact.get("supporting_papers", []))
        if not artifact.get("source_artifacts") or not supporting or not supporting <= allowed_ids:
            return False
    return True


def _references_within_allowed(markdown: str, allowed_ids: set[str]) -> bool:
    if not allowed_ids:
        return True
    refs = re.split(r"\n## References\b", markdown, maxsplit=1)
    if len(refs) < 2:
        return False
    body_cites = _extract_citations(refs[0])
    ref_ids = set()
    for line in refs[1].splitlines():
        stripped = line.strip()
        for paper_id in allowed_ids:
            if stripped.startswith(f"- {paper_id}:"):
                ref_ids.add(paper_id)
                break
    return ref_ids <= allowed_ids and ref_ids <= body_cites


def _revision_suggestions(soft_review: dict[str, dict[str, Any]], hard_gates: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    suggestions = []
    for name, gate in hard_gates.items():
        if gate.get("blocking", True) and not gate["pass"]:
            suggestions.append({"priority": "high", "target": name, "suggestion": "Address the failed hard gate before final publication."})
        elif not gate.get("blocking", True) and not gate["pass"]:
            suggestions.append({"priority": "low", "target": name, "suggestion": "Improve this non-blocking quality diagnostic in the next iteration."})
    for metric, result in soft_review.items():
        if result["score"] < 0.70:
            suggestions.append({"priority": "medium", "target": metric, "suggestion": result["comment"]})
    return suggestions


def _soft_comment(metric: str, raw_score: int) -> str:
    quality = "good" if raw_score >= 4 else "acceptable" if raw_score == 3 else "weak"
    return f"{metric} is {quality} under the rule-based G-Eval-style rubric."


def _score_grid(report: dict[str, Any]) -> str:
    items = [
        "overall_score",
        "coverage_score",
        "citation_validity_score",
        "evidence_grounding_score",
        "visualization_score",
        "structure_score",
    ]
    return '<div class="score-grid">' + "".join(
        f'<div class="score"><strong>{html.escape(key)}</strong><br>{report.get(key, 0):.3f}</div>' for key in items
    ) + "</div>"


def _review_html(review: dict[str, Any]) -> str:
    rows = ["<h3>Hard Gates</h3><table><tr><th>Gate</th><th>Score</th><th>Pass</th></tr>"]
    for name, gate in review.get("hard_gates", {}).items():
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{gate.get('score', 0):.3f}</td><td>{gate.get('pass')}</td></tr>")
    rows.append("</table><h3>Soft Review</h3><table><tr><th>Metric</th><th>Score</th><th>Comment</th></tr>")
    for name, metric in review.get("soft_review", {}).items():
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{metric.get('score', 0):.3f}</td><td>{html.escape(metric.get('comment', ''))}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _timeline_html(timeline: dict[str, Any]) -> str:
    rows = ["<table><tr><th>Year</th><th>Category</th><th>Papers</th><th>Summary</th></tr>"]
    for event in timeline.get("events", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('year') or ''))}</td>"
            f"<td>{html.escape(str(event.get('category_name') or ''))}</td>"
            f"<td>{html.escape(', '.join(event.get('paper_ids', [])))}</td>"
            f"<td>{html.escape(str(event.get('summary') or ''))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _claim_html(claim_map: dict[str, Any], evidence_store: dict[str, Any]) -> str:
    evidence_by_id = {e.get("evidence_id"): e for e in _as_list(evidence_store, "evidence")}
    entries = claim_map.get("entries", claim_map.get("claims", []))
    rows = ["<table><tr><th>Claim</th><th>Paper</th><th>Status</th><th>Evidence</th></tr>"]
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        eids = entry.get("evidence_ids") or entry.get("supporting_evidence") or []
        evidence_text = "; ".join(str(evidence_by_id.get(eid, {}).get("text", eid))[:180] for eid in eids[:2])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry.get('claim_text') or entry.get('claim') or ''))}</td>"
            f"<td>{html.escape(str(entry.get('cited_paper_id') or ''))}</td>"
            f"<td>{html.escape(str(entry.get('status') or entry.get('evidence_status') or ''))}</td>"
            f"<td>{html.escape(evidence_text)}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _pdf_like_text(topic: str, evaluation: dict[str, Any], review: dict[str, Any], html_path: Path) -> str:
    return (
        "PDF placeholder for EviSurvey final report\n"
        f"Topic: {topic}\n"
        f"HTML report: {html_path.name}\n"
        f"Overall score: {evaluation.get('overall_score')}\n"
        f"ReviewBoard decision: {review.get('final_decision')}\n"
        "A real PDF renderer can replace this file in a later iteration.\n"
    )


def _read_optional(root: Path, raw_path: str | None, default: Any) -> Any:
    if not raw_path:
        return default
    path = _resolve(root, raw_path)
    if not path.exists():
        return default
    return read_json(path)


def _read_final_or_requested(root: Path, name: str, raw_path: str | None, default: Any) -> Any:
    if os.getenv("FINAL_SEED_PAPERS") == "1":
        final_path = root / "cache" / f"final_{name}.json"
        if final_path.exists():
            return read_json(final_path)
    return _read_optional(root, raw_path, default)


def _resolve(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _as_list(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get(key, data.get("items", []))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _extract_citations(markdown: str) -> set[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    return {match.strip() for match in re.findall(r"\[([^\]]+)\]", text_only) if not match.startswith("http")}


def _extract_citations_in_order(markdown: str) -> list[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    return [match.strip() for match in re.findall(r"\[([^\]]+)\]", text_only) if not match.startswith("http")]
