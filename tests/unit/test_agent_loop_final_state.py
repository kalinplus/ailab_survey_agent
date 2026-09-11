"""Terminal-state honesty tests for harness/agent_loop.py and main.py.

Spec: 状态机与退出码诚实化 — a failed Goal Gate must surface as
status "quality_failed" (not "completed") in final_state.json while
rendering still runs, and main.py must map it to a non-zero exit code.
prepared/completed keep exit code 0 (no regression).
"""

import json
from pathlib import Path

from config import AppConfig, ensure_project_dirs
from harness.agent_loop import AgentLoop
from harness.logger import WorkflowLogger
from harness.planner import Planner
from harness.state_manager import StateManager
from main import exit_code_for_status


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root_dir=tmp_path,
        intern_api_base_url="http://localhost/v1",
        intern_api_key="",
        intern_model_name="intern-s2-preview",
        sciverse_api_token="",
        sciverse_api_base_url="https://api.sciverse.space",
        strategy_probing_enabled=False,
        strategy_probe_limit=0,
        strategy_cluster_count=4,
        strategy_memory_enabled=True,
        strategy_memory_max_chars=4000,
        request_timeout_seconds=1.0,
        max_llm_retries=1,
        tool_timeout_seconds=30.0,
        tool_result_max_chars=2000,
        default_language="zh",
        default_mode="demo",
    )


class FakeToolRegistry:
    """Scripted stand-in for ToolRegistry.

    Writes the on-disk artifacts the real B/C tools would produce (bundle,
    survey.md, citation_result.json) so the validator, repair loop, and Goal
    Gate operate on real files; record tool call order for assertions.
    """

    def __init__(self, root_dir: Path, verify_metrics: dict):
        self.root_dir = root_dir
        self.verify_metrics = verify_metrics
        self.calls: list[str] = []
        self.max_result_chars = 2000

    def list_specs(self):
        return []

    def tool_prompt(self):
        return "Available tools: (fake)"

    def run(self, tool_name: str, request_path: str) -> dict:
        self.calls.append(tool_name)
        if tool_name == "knowledge_pipeline_worker":
            self._write_bundle()
            return {"tool": tool_name, "status": "success", "outputs": ["cache/knowledge_bundle.json"]}
        if tool_name == "write_survey":
            self._write("output/survey.md", "# Survey\n\nA claim [p1].\n")
            return {"tool": tool_name, "status": "success", "outputs": ["output/survey.md"]}
        if tool_name == "verify_citations":
            result = {"tool": tool_name, "status": "success", "metrics": dict(self.verify_metrics)}
            self._write("output/citation_result.json", result)
            return result
        if tool_name == "revise_survey":
            # same text: keeps coverage retention at 1.0, forces a fixpoint stop
            self._write("output/survey_revised.md", "# Survey\n\nA claim [p1].\n")
            return {"tool": tool_name, "status": "success", "outputs": ["output/survey_revised.md"]}
        if tool_name == "render_report":
            self._write("output/evaluation_report.json", {"overall_score": 0.8})
            return {"tool": tool_name, "status": "success", "outputs": ["output/final_report.html"]}
        raise AssertionError(f"unexpected tool called: {tool_name}")

    def _write(self, relative_path: str, payload) -> None:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_bundle(self) -> None:
        # consistent ids: paper_cards p1 in citation_index, evidence only for p1
        artifacts = {
            "retrieved_papers": {"papers": [{"paper_id": "p1", "title": "World Models"}]},
            "parsed_papers": {"parsed": [{"paper_id": "p1"}]},
            "figure_bank": {"figures": []},
            "table_bank": {"tables": []},
            "paper_cards": {"paper_cards": [{"paper_id": "p1", "title": "World Models", "year": 2018}]},
            "evidence_store": {"evidence": [{"paper_id": "p1", "evidence_id": "e1", "text": "a claim"}]},
            "taxonomy": {"categories": []},
            "citation_index": {"citations": [{"paper_id": "p1", "title": "World Models"}]},
        }
        paths = {}
        for name, payload in artifacts.items():
            self._write(f"cache/{name}.json", payload)
            paths[name] = f"cache/{name}.json"
        self._write("cache/knowledge_bundle.json", {"status": "success", "artifacts": paths})


def _run(tmp_path: Path, *, verify_metrics: dict | None = None, prepare_only: bool = False):
    config = _config(tmp_path)
    ensure_project_dirs(config)
    registry = FakeToolRegistry(tmp_path, verify_metrics or {})
    loop = AgentLoop(
        config=config,
        planner=Planner(config),
        tool_registry=registry,
        state_manager=StateManager(config.logs_dir / "state.json"),
        logger=WorkflowLogger(config.logs_dir / "run.jsonl"),
    )
    final_state = loop.run(
        topic="world models test",
        language="zh",
        mode="demo",
        max_papers=5,
        max_core_papers=3,
        prepare_only=prepare_only,
    )
    return final_state, registry


def test_gate_fail_ends_quality_failed(tmp_path):
    # constant unsupported_claims -> round 0 repairs, round 1 hits fixpoint stop
    final_state, registry = _run(
        tmp_path,
        verify_metrics={"unsupported_claims": 3, "invalid_citations": 0, "citation_validity_score": 0.5},
    )
    assert final_state["status"] == "quality_failed"
    assert final_state["goal_gate"]["passed"] is False
    assert final_state["goal_gate"]["unsupported"] == 3
    assert final_state["goal_gate"]["stop_reason"] == "fixpoint"
    # scores kept for diagnosis; rendering still ran on the gate-failed run
    assert final_state["scores"]["overall_score"] == 0.8
    assert "render_report" in registry.calls
    # on-disk final_state.json and persisted state.json agree
    on_disk = json.loads((tmp_path / "output" / "final_state.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "quality_failed"
    persisted = json.loads((tmp_path / "logs" / "state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "quality_failed"


def test_gate_pass_ends_completed(tmp_path):
    final_state, registry = _run(
        tmp_path,
        verify_metrics={"unsupported_claims": 0, "invalid_citations": 0, "citation_validity_score": 1.0},
    )
    assert final_state["status"] == "completed"
    assert final_state["goal_gate"]["passed"] is True
    assert final_state["goal_gate"]["stop_reason"] == "passed"
    assert "revise_survey" not in registry.calls
    assert "render_report" in registry.calls


def test_prepare_only_ends_prepared_without_tools(tmp_path):
    final_state, registry = _run(tmp_path, prepare_only=True)
    assert final_state["status"] == "prepared"
    assert registry.calls == []
    assert (tmp_path / "output" / "final_state.json").exists()


def test_exit_code_for_status():
    assert exit_code_for_status("completed") == 0
    assert exit_code_for_status("prepared") == 0
    assert exit_code_for_status("quality_failed") == 1
    assert exit_code_for_status("running") == 1
