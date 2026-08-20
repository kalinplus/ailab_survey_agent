"""A-owned controller loop for the single-request EviSurvey workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import AppConfig
from . import goal_gate
from .agent_context import build_agent_context
from .citation_prelock import build_citation_ready_set
from .json_io import read_json, write_json
from .knowledge_bundle_validator import KnowledgeBundleValidator
from .logger import WorkflowLogger
from .planner import Planner
from .skill_manager import SkillManager
from .state_manager import StateManager
from .tool_registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        *,
        config: AppConfig,
        planner: Planner,
        tool_registry: ToolRegistry,
        state_manager: StateManager,
        logger: WorkflowLogger,
    ) -> None:
        self.config = config
        self.planner = planner
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.logger = logger
        self.validator = KnowledgeBundleValidator(config.root_dir)
        self.skill_manager = SkillManager(config.root_dir)

    def run(
        self,
        *,
        topic: str,
        language: str,
        mode: str,
        max_papers: int,
        max_core_papers: int,
        prepare_only: bool = False,
        use_mineru: bool = False,
    ) -> dict[str, Any]:
        task_id = self.planner.make_task_id(topic)
        state = self.state_manager.init(task_id)
        self.logger.log("run_started", task_id=task_id, topic=topic, mode=mode)

        task_request = self.planner.build_task_request(
            task_id=task_id,
            topic=topic,
            language=language,
            mode=mode,
            max_papers=max_papers,
            max_core_papers=max_core_papers,
            use_mineru=use_mineru,
        )
        self._write("cache/task_request.json", task_request)
        state = self.state_manager.advance(
            state,
            current_stage="search_strategy",
            finished_stage="task_request",
            ready_artifacts={"task_request": "cache/task_request.json"},
        )

        search_strategy = self.planner.build_search_strategy(
            task_id=task_id,
            topic=topic,
            max_papers=max_papers,
            max_core_papers=max_core_papers,
            mode=mode,
        )
        self._write("cache/search_strategy.json", search_strategy)
        self._write(
            "cache/agent_context.json",
            build_agent_context(
                task_request=task_request,
                search_strategy=search_strategy,
                tool_registry=self.tool_registry,
                skill_manager=self.skill_manager,
            ),
        )
        self._write("cache/tool_manifest.json", self.tool_registry.list_specs())
        self._write("cache/skill_manifest.json", self.skill_manager.list_manifest())

        knowledge_request = self.planner.build_knowledge_build_request(
            task_request,
            active_skills=self._active_skills("knowledge_build"),
            skill_policy=self.skill_manager.policy(),
        )
        self._write("requests/knowledge_build_request.json", knowledge_request)
        state = self.state_manager.advance(
            state,
            current_stage="waiting_for_b_knowledge_bundle",
            finished_stage="knowledge_build_request",
            ready_artifacts={
                "search_strategy": "cache/search_strategy.json",
                "knowledge_build_request": "requests/knowledge_build_request.json",
                "agent_context": "cache/agent_context.json",
                "tool_manifest": "cache/tool_manifest.json",
                "skill_manifest": "cache/skill_manifest.json",
            },
            pending_artifacts={"knowledge_bundle": "cache/knowledge_bundle.json"},
        )

        if prepare_only:
            final_state = self._prepared_state(task_id, topic)
            self._write("output/final_state.json", final_state)
            self.state_manager.advance(state, current_stage="prepared", status="prepared")
            self.logger.log("run_prepared", task_id=task_id)
            return final_state

        self._run_tool("knowledge_pipeline_worker", "requests/knowledge_build_request.json")
        validation = self.validator.validate("cache/knowledge_bundle.json")
        state = self.state_manager.advance(
            state,
            current_stage="citation_prelock",
            finished_stage="knowledge_bundle",
            ready_artifacts={"knowledge_bundle": "cache/knowledge_bundle.json"},
        )

        citation_ready_set = build_citation_ready_set(
            task_id=task_id,
            paper_cards=validation["artifacts"]["paper_cards"],
            citation_index=validation["artifacts"]["citation_index"],
            evidence_store=validation["artifacts"]["evidence_store"],
            max_core_papers=max_core_papers,
        )
        self._write("cache/citation_ready_set.json", citation_ready_set)

        survey_request = self.planner.build_survey_generation_request(
            task_id,
            topic,
            language,
            active_skills=self._active_skills("survey_generation"),
            skill_policy=self.skill_manager.policy(),
        )
        self._write("requests/survey_generation_request.json", survey_request)
        state = self.state_manager.advance(
            state,
            current_stage="waiting_for_c_survey",
            finished_stage="survey_generation_request",
            ready_artifacts={
                "citation_ready_set": "cache/citation_ready_set.json",
                "survey_generation_request": "requests/survey_generation_request.json",
            },
            pending_artifacts={
                "survey_markdown": "output/survey.md",
                "timeline": "cache/timeline.json",
                "generated_artifact_bank": "cache/generated_artifact_bank.json",
            },
        )

        self._run_tool("write_survey", "requests/survey_generation_request.json")

        verification_request = self.planner.build_verification_request(
            task_id,
            active_skills=self._active_skills("verification"),
            skill_policy=self.skill_manager.policy(),
        )
        self._write("requests/verification_request.json", verification_request)
        gate_result = self._verify_repair_loop(task_id)

        evaluation_request = self.planner.build_evaluation_render_request(
            task_id,
            topic,
            active_skills=self._active_skills("evaluation_render"),
            skill_policy=self.skill_manager.policy(),
        )
        self._write("requests/evaluation_render_request.json", evaluation_request)
        self._run_tool("render_report", "requests/evaluation_render_request.json")

        final_state = self._completed_state(task_id, topic, gate_result)
        self._write("output/final_state.json", final_state)
        self.state_manager.advance(
            state,
            current_stage="completed",
            finished_stage="final_state",
            ready_artifacts={"final_state": "output/final_state.json"},
            status="completed",
        )
        self.logger.log("run_completed", task_id=task_id)
        return final_state

    def _verify_repair_loop(self, task_id: str) -> goal_gate.GateResult:
        """verify -> (revise -> verify)* under the deterministic Goal Gate.

        Replaces the old half-loop whose second verify result was discarded:
        every round's result is now consumed by the gate. Keeps the best-so-far
        survey ((unsupported, invalid, -chars) lexicographic) and restores it
        if a later repair round made things worse.
        """
        best_metrics: dict[str, Any] | None = None
        best_text: str | None = None
        prev_unsupported: int | None = None
        gate_result = goal_gate.GateResult(False, "not_run", 0, 0, 0.0)
        for repair_round in range(goal_gate.MAX_REPAIR_ROUNDS + 1):
            verify_result = self._run_tool("verify_citations", "requests/verification_request.json")
            metrics = verify_result.get("metrics", {})
            gate_result = goal_gate.evaluate(
                metrics, repair_round=repair_round, prev_unsupported=prev_unsupported)
            survey_path = self.config.output_dir / "survey.md"
            survey_text = survey_path.read_text(encoding="utf-8") if survey_path.exists() else ""
            if best_metrics is None or _repair_rank(metrics, survey_text) < _repair_rank(best_metrics, best_text or ""):
                best_metrics, best_text = metrics, survey_text
            self.logger.log("goal_gate", task_id=task_id, round=repair_round, **gate_result.to_dict())
            if gate_result.stop_reason != goal_gate.STOP_REPAIRING:
                break
            revision_request = self.planner.build_revision_request(
                task_id,
                "Evidence verification failed.",
                active_skills=self._active_skills("revision"),
                skill_policy=self.skill_manager.policy(),
            )
            self._write("requests/revision_request.json", revision_request)
            self._run_tool("revise_survey", "requests/revision_request.json")
            revised = self.config.output_dir / "survey_revised.md"
            if revised.exists():
                (self.config.output_dir / "survey.md").write_text(
                    revised.read_text(encoding="utf-8"), encoding="utf-8")
            prev_unsupported = gate_result.unsupported

        # rollback guard: never leave a worse round's text as the delivered survey
        survey_path = self.config.output_dir / "survey.md"
        if best_text is not None and survey_path.exists() \
                and survey_path.read_text(encoding="utf-8") != best_text:
            self.logger.log("repair_rollback", task_id=task_id)
            survey_path.write_text(best_text, encoding="utf-8")
        return gate_result

    def _run_tool(self, tool_name: str, request_path: str) -> dict[str, Any]:
        result = self.tool_registry.run(tool_name, request_path)
        if result.get("status") not in {"success", "partial_success"}:
            raise RuntimeError(f"Tool {tool_name} failed: {result.get('message', result)}")
        return result

    def _prepared_state(self, task_id: str, topic: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "topic": topic,
            "status": "prepared",
            "final_outputs": {
                "knowledge_build_request": "requests/knowledge_build_request.json",
                "task_request": "cache/task_request.json",
                "search_strategy": "cache/search_strategy.json",
                "skill_manifest": "cache/skill_manifest.json",
                "state": "logs/state.json",
                "workflow_log": "logs/run.jsonl",
            },
            "scores": {},
        }

    def _completed_state(
        self, task_id: str, topic: str, gate_result: goal_gate.GateResult | None = None,
    ) -> dict[str, Any]:
        evaluation_report = self._read_optional("output/evaluation_report.json", {})
        state = {
            "task_id": task_id,
            "topic": topic,
            "status": "completed",
            "goal_gate": gate_result.to_dict() if gate_result else None,
            "final_outputs": {
                "survey_markdown": "output/survey.md",
                "html_report": "output/final_report.html",
                "pdf_report": "output/final_report.pdf",
                "evaluation_report": "output/evaluation_report.json",
                "citation_result": "output/citation_result.json",
                "claim_map": "cache/claim_map.json",
                "knowledge_bundle": "cache/knowledge_bundle.json",
                "citation_ready_set": "cache/citation_ready_set.json",
                "generated_artifact_bank": "cache/generated_artifact_bank.json",
            },
            "scores": {
                "overall_score": evaluation_report.get("overall_score", 0.0),
                "citation_validity_score": evaluation_report.get("citation_validity_score", 0.0),
                "evidence_grounding_score": evaluation_report.get("evidence_grounding_score", 0.0),
                "visualization_score": evaluation_report.get("visualization_score", 0.0),
            },
        }
        return state

    def _path(self, relative_path: str) -> Path:
        return self.config.root_dir / relative_path

    def _active_skills(self, stage: str) -> list[dict[str, Any]]:
        return self.skill_manager.active_for_stage(stage)

    def _write(self, relative_path: str, data: Any) -> None:
        write_json(self._path(relative_path), data)
        self.logger.log("artifact_written", path=relative_path)

    def _read(self, relative_path: str) -> Any:
        return read_json(self._path(relative_path))

    def _read_optional(self, relative_path: str, default: Any) -> Any:
        path = self._path(relative_path)
        if not path.exists():
            return default
        return read_json(path)


def _repair_rank(metrics: dict[str, Any], survey_text: str) -> tuple[int, int, int]:
    """Best-so-far ordering for the repair loop: lower is better.

    Fewer unsupported claims first, then fewer invalid citations, then the
    longer text wins (deletion-only repair shrinks the survey for free).
    """
    return (
        int(metrics.get("unsupported_claims", 0)),
        int(metrics.get("invalid_citations", 0)),
        -len(survey_text),
    )
