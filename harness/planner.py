"""A-side request planner."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from config import AppConfig
from .search_strategy_builder import build_search_strategy


class Planner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def make_task_id(self, topic: str) -> str:
        slug = re.sub(r"\W+", "_", topic.lower(), flags=re.UNICODE).strip("_")
        slug = slug[:32] or "survey"
        return f"task_{slug}_001"

    def build_task_request(
        self,
        *,
        task_id: str,
        topic: str,
        language: str,
        mode: str,
        max_papers: int,
        max_core_papers: int,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "topic": topic,
            "task_type": "academic_survey",
            "language": language,
            "output_formats": ["html", "pdf"],
            "requirements": {
                "must_include": [
                    "search_strategy",
                    "paper_cards",
                    "multimodal_evidence",
                    "citation_pre_lock",
                    "citation_verification",
                    "claim_to_evidence_map",
                    "timeline",
                    "interactive_report",
                ],
                "avoid": [
                    "hallucinated_citations",
                    "unsupported_claims",
                    "figures_without_source",
                ],
            },
            "run_config": {
                "use_online_search": mode == "full",
                "use_seed_fallback": True,
                "use_mineru": mode == "full",
                "max_papers": max_papers,
                "max_core_papers": max_core_papers,
                "mode": mode,
            },
        }

    def build_search_strategy(
        self,
        *,
        task_id: str,
        topic: str,
        max_papers: int,
        max_core_papers: int,
    ) -> dict[str, Any]:
        return build_search_strategy(
            task_id=task_id,
            topic=topic,
            max_papers=max_papers,
            max_core_papers=max_core_papers,
            end_year=datetime.now().year,
        )

    def build_knowledge_build_request(self, task_request: dict[str, Any]) -> dict[str, Any]:
        run_config = task_request["run_config"]
        return {
            "task_id": task_request["task_id"],
            "topic": task_request["topic"],
            "request_type": "knowledge_build",
            "inputs": {
                "task_request_path": "cache/task_request.json",
                "search_strategy_path": "cache/search_strategy.json",
            },
            "outputs": {
                "knowledge_bundle_path": "cache/knowledge_bundle.json",
                "retrieved_papers_path": "cache/retrieved_papers.json",
                "parsed_papers_path": "cache/parsed_papers.json",
                "figure_bank_path": "cache/figure_bank.json",
                "table_bank_path": "cache/table_bank.json",
                "paper_cards_path": "cache/paper_cards.json",
                "evidence_store_path": "cache/evidence_store.json",
                "taxonomy_path": "cache/taxonomy.json",
                "citation_index_path": "cache/citation_index.json",
            },
            "pipeline_config": {
                "use_online_search": run_config["use_online_search"],
                "use_seed_fallback": run_config["use_seed_fallback"],
                "use_mineru": run_config["use_mineru"],
                "use_mock_mineru_if_failed": False,
                "max_papers": run_config["max_papers"],
                "max_core_papers": run_config["max_core_papers"],
                "download_pdfs": run_config["mode"] == "full",
                "extract_figures": True,
                "extract_tables": True,
                "extract_captions": True,
                "build_taxonomy": True,
                "build_citation_index": True,
            },
            "quality_requirements": {
                "min_total_papers": 10,
                "min_core_papers": 5,
                "min_figures": 1,
                "min_evidence_per_core_paper": 1,
                "allow_abstract_only_fallback": True,
            },
        }

    def build_survey_generation_request(self, task_id: str, topic: str, language: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "topic": topic,
            "request_type": "survey_generation",
            "language": language,
            "inputs": {
                "knowledge_bundle_path": "cache/knowledge_bundle.json",
                "paper_cards_path": "cache/paper_cards.json",
                "evidence_store_path": "cache/evidence_store.json",
                "figure_bank_path": "cache/figure_bank.json",
                "table_bank_path": "cache/table_bank.json",
                "taxonomy_path": "cache/taxonomy.json",
                "citation_ready_set_path": "cache/citation_ready_set.json",
            },
            "outputs": {
                "survey_markdown_path": "output/survey.md",
                "timeline_path": "cache/timeline.json",
                "generated_artifact_bank_path": "cache/generated_artifact_bank.json",
            },
            "writing_rules": {
                "citation_format": "[paper_id]",
                "only_use_citation_ready_set": True,
                "paper_figures_must_come_from_figure_bank": True,
                "paper_tables_must_come_from_table_bank": True,
                "generated_artifacts_allowed": True,
                "generated_artifacts_must_be_declared": True,
                "do_not_introduce_new_references": True,
            },
            "generated_artifact_policy": {
                "allowed_types": [
                    "timeline",
                    "taxonomy_graph",
                    "comparison_chart",
                    "comparison_table",
                    "future_matrix",
                    "evaluation_chart",
                    "summary_table",
                ],
                "output_dir": "output/generated_assets",
                "provenance_required": True,
                "must_not_be_claimed_as_paper_extracted_artifact": True,
            },
        }

    def build_verification_request(self, task_id: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "inputs": {
                "survey_markdown_path": "output/survey.md",
                "paper_cards_path": "cache/paper_cards.json",
                "citation_ready_set_path": "cache/citation_ready_set.json",
                "evidence_store_path": "cache/evidence_store.json",
                "figure_bank_path": "cache/figure_bank.json",
                "table_bank_path": "cache/table_bank.json",
                "generated_artifact_bank_path": "cache/generated_artifact_bank.json",
            },
            "outputs": {
                "citation_result_path": "output/citation_result.json",
                "claim_map_path": "cache/claim_map.json",
            },
            "verification_rules": {
                "citation_format": "[paper_id]",
                "citation_must_exist_in_ready_set": True,
                "paper_figures_must_exist_in_figure_bank": True,
                "paper_tables_must_exist_in_table_bank": True,
                "generated_artifacts_must_exist_in_generated_artifact_bank": True,
                "claim_should_map_to_evidence": True,
                "unsupported_claim_policy": "mark_unsupported",
            },
        }

    def build_revision_request(self, task_id: str, reason: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "request_type": "survey_revision",
            "reason": reason,
            "inputs": {
                "survey_markdown_path": "output/survey.md",
                "citation_result_path": "output/citation_result.json",
                "claim_map_path": "cache/claim_map.json",
                "citation_ready_set_path": "cache/citation_ready_set.json",
                "evidence_store_path": "cache/evidence_store.json",
                "figure_bank_path": "cache/figure_bank.json",
                "table_bank_path": "cache/table_bank.json",
                "generated_artifact_bank_path": "cache/generated_artifact_bank.json",
            },
            "outputs": {"revised_survey_markdown_path": "output/survey_revised.md"},
            "required_changes": [
                "Remove citations not in CitationReadySet.",
                "Remove figures not in FigureBank.",
                "Remove tables not in TableBank.",
                "Rewrite unsupported claims.",
                "Do not introduce new references.",
            ],
        }

    def build_evaluation_render_request(self, task_id: str, topic: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "topic": topic,
            "inputs": {
                "survey_markdown_path": "output/survey.md",
                "knowledge_bundle_path": "cache/knowledge_bundle.json",
                "paper_cards_path": "cache/paper_cards.json",
                "taxonomy_path": "cache/taxonomy.json",
                "timeline_path": "cache/timeline.json",
                "citation_result_path": "output/citation_result.json",
                "claim_map_path": "cache/claim_map.json",
                "figure_bank_path": "cache/figure_bank.json",
                "table_bank_path": "cache/table_bank.json",
                "generated_artifact_bank_path": "cache/generated_artifact_bank.json",
                "evidence_store_path": "cache/evidence_store.json",
            },
            "outputs": {
                "evaluation_report_path": "output/evaluation_report.json",
                "html_report_path": "output/final_report.html",
                "pdf_report_path": "output/final_report.pdf",
            },
            "render_requirements": {
                "include_timeline": True,
                "include_taxonomy_graph": True,
                "include_paper_figures": True,
                "include_paper_tables": True,
                "include_comparison_table": True,
                "include_future_matrix": True,
                "include_evidence_panel": True,
                "include_citation_summary": True,
                "include_evaluation_summary": True,
            },
        }
