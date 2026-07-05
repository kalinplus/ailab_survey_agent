"""A-side request planner."""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from typing import Any

from llm_client import InternS2Client
from config import AppConfig
from .memory_manager import MemoryManager
from .search_strategy_builder import build_search_strategy


class Planner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm_client = InternS2Client(config)
        self.memory_manager = MemoryManager(
            config.root_dir,
            enabled=config.strategy_memory_enabled,
            max_chars=config.strategy_memory_max_chars,
        )

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
                "use_mineru": False,
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
        mode: str = "demo",
    ) -> dict[str, Any]:
        memory_context = self.memory_manager.load_for_strategy(topic)
        use_probing = mode == "full" and self.config.strategy_probing_enabled
        strategy = build_search_strategy(
            task_id=task_id,
            topic=topic,
            max_papers=max_papers,
            max_core_papers=max_core_papers,
            end_year=datetime.now().year,
            use_probing=use_probing,
            sciverse_api_key=self.config.sciverse_api_token,
            sciverse_api_base_url=self.config.sciverse_api_base_url,
            request_timeout_seconds=self.config.request_timeout_seconds,
            probe_limit=self.config.strategy_probe_limit,
            cluster_count=self.config.strategy_cluster_count,
            memory_context=memory_context,
            llm_json_chat=self._strategy_json_chat if self.llm_client.is_configured() else None,
        )
        strategy["strategy_generation"] = {
            "mode": mode,
            "memory_enabled": self.config.strategy_memory_enabled,
            "memory_used": bool(memory_context),
            "memory_chars": len(memory_context),
            "probing_requested": use_probing,
            "probing_effective": use_probing and bool(self.config.sciverse_api_token),
            "probe_limit": self.config.strategy_probe_limit if use_probing else 0,
        }
        self.memory_manager.record_strategy_run(
            topic=topic,
            strategy=strategy,
            used_memory=bool(memory_context),
            mode=mode,
            used_probing=use_probing and bool(self.config.sciverse_api_token),
        )
        return strategy

    def build_knowledge_build_request(
        self,
        task_request: dict[str, Any],
        *,
        active_skills: list[dict[str, Any]] | None = None,
        skill_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_config = task_request["run_config"]
        request = {
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
        return self._attach_skills(request, active_skills, skill_policy)

    def build_survey_generation_request(
        self,
        task_id: str,
        topic: str,
        language: str,
        *,
        active_skills: list[dict[str, Any]] | None = None,
        skill_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = {
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
        return self._attach_skills(request, active_skills, skill_policy)

    def build_verification_request(
        self,
        task_id: str,
        *,
        active_skills: list[dict[str, Any]] | None = None,
        skill_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = {
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
        return self._attach_skills(request, active_skills, skill_policy)

    def build_revision_request(
        self,
        task_id: str,
        reason: str,
        *,
        active_skills: list[dict[str, Any]] | None = None,
        skill_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = {
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
        return self._attach_skills(request, active_skills, skill_policy)

    def build_evaluation_render_request(
        self,
        task_id: str,
        topic: str,
        *,
        active_skills: list[dict[str, Any]] | None = None,
        skill_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = {
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
        return self._attach_skills(request, active_skills, skill_policy)

    def _attach_skills(
        self,
        request: dict[str, Any],
        active_skills: list[dict[str, Any]] | None,
        skill_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if active_skills is not None:
            request["active_skills"] = active_skills
        if skill_policy is not None:
            request["skill_policy"] = skill_policy
        return request

    def _strategy_json_chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            return self.llm_client.json_chat(messages, temperature=0.1, max_tokens=4000)
        except Exception:
            content = self.llm_client.chat(messages, temperature=0.1, max_tokens=4000)
            return self._extract_json_object(content)

    def _extract_json_object(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for start in [index for index, char in enumerate(content) if char == "{"]:
            try:
                parsed, _ = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and self._looks_like_strategy(parsed):
                return parsed

        for candidate in self._balanced_brace_candidates(content):
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, dict) and self._looks_like_strategy(parsed):
                return parsed

        raise json.JSONDecodeError("No valid JSON object found in model response", content, 0)

    def _balanced_brace_candidates(self, content: str) -> list[str]:
        candidates = []
        starts = [index for index, char in enumerate(content) if char == "{"]
        for start in starts:
            depth = 0
            in_string = False
            escape = False
            quote = ""
            for index in range(start, len(content)):
                char = content[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == quote:
                        in_string = False
                    continue
                if char in {"'", '"'}:
                    in_string = True
                    quote = char
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(content[start : index + 1])
                        break
        return candidates

    def _looks_like_strategy(self, data: dict[str, Any]) -> bool:
        strategy = data.get("search_strategy", data)
        return isinstance(strategy, dict) and (
            "wide_search" in strategy
            or "topic_understanding" in strategy
            or "search_aspects" in strategy
            or "aspects" in strategy
        )
