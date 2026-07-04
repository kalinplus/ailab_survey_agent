"""Tool registry that keeps B/C boundaries explicit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from typing import Any, Callable

from .logger import WorkflowLogger, now_iso
from .sandbox_guard import SandboxGuard

ToolFn = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    owner: str
    description: str
    request_path: str
    expected_outputs: list[str]
    allowed_read_paths: list[str]
    allowed_write_paths: list[str]
    timeout_seconds: float
    max_result_chars: int


class ToolRegistry:
    def __init__(
        self,
        root_dir: Path,
        logger: WorkflowLogger,
        *,
        default_timeout_seconds: float = 300,
        max_result_chars: int = 6000,
    ) -> None:
        self.root_dir = root_dir
        self.logger = logger
        self.guard = SandboxGuard(root_dir)
        self.default_timeout_seconds = default_timeout_seconds
        self.max_result_chars = max_result_chars
        self._tools: dict[str, ToolFn] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._register_builtin_specs()
        self._register_optional("knowledge_pipeline_worker", "tools.knowledge_pipeline_worker", "run")
        self._register_optional("write_survey", "tools.write_survey", "run")
        self._register_optional("verify_citations", "tools.verify_citations", "run")
        self._register_optional("revise_survey", "tools.revise_survey", "run")
        self._register_optional("render_report", "tools.render_report", "run")

    def register(self, name: str, fn: ToolFn, spec: ToolSpec | None = None) -> None:
        self._tools[name] = fn
        if spec:
            self._specs[name] = spec

    def run(self, tool_name: str, request_path: str) -> dict[str, Any]:
        if tool_name not in self._tools:
            raise KeyError(
                f"Tool '{tool_name}' is not registered. "
                "Ask the B/C owner to provide tools/<module>.py with run(request_path)."
            )

        self.guard.ensure_existing_file(request_path)
        spec = self._specs.get(tool_name)
        timeout_seconds = spec.timeout_seconds if spec else self.default_timeout_seconds
        self.logger.log("tool_started", tool=tool_name, input_request=request_path)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._tools[tool_name], request_path)
            result = future.result(timeout=timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            result = {
                "tool": tool_name,
                "status": "failed",
                "input_request": request_path,
                "outputs": [],
                "metrics": {"timeout_seconds": timeout_seconds},
                "message": f"Tool timed out after {timeout_seconds} seconds.",
                "timestamp": now_iso(),
            }
            self.logger.log("tool_timeout", tool=tool_name, input_request=request_path, timeout_seconds=timeout_seconds)
            return result
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            self.logger.log("tool_failed", tool=tool_name, input_request=request_path, error=str(exc))
            raise
        finally:
            if "result" in locals():
                executor.shutdown(wait=False, cancel_futures=True)

        result.setdefault("tool", tool_name)
        result.setdefault("input_request", request_path)
        result.setdefault("timestamp", now_iso())
        context_result = self.truncate_for_context(result)
        self.logger.log(
            "tool_finished",
            tool=tool_name,
            input_request=request_path,
            status=result.get("status", "unknown"),
            outputs=result.get("outputs", []),
            metrics=result.get("metrics", {}),
            context_result=context_result,
        )
        return result

    def list_specs(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self._specs.values()]

    def tool_prompt(self) -> str:
        lines = [
            "Available tools:",
            "Return JSON only. To call a tool, use:",
            '{"action":"call_tool","tool_name":"...","request_path":"...","reason":"..."}',
            'To finish, use: {"action":"final","answer":"..."}',
        ]
        for spec in self._specs.values():
            lines.extend(
                [
                    f"- {spec.name} ({spec.owner}): {spec.description}",
                    f"  request_path: {spec.request_path}",
                    f"  expected_outputs: {', '.join(spec.expected_outputs)}",
                    f"  read_paths: {', '.join(spec.allowed_read_paths)}",
                    f"  write_paths: {', '.join(spec.allowed_write_paths)}",
                    f"  timeout_seconds: {spec.timeout_seconds}",
                ]
            )
        return "\n".join(lines)

    def truncate_for_context(self, result: dict[str, Any]) -> dict[str, Any]:
        text = repr(result)
        limit = self.max_result_chars
        if len(text) <= limit:
            return result
        return {
            "tool": result.get("tool"),
            "status": result.get("status"),
            "input_request": result.get("input_request"),
            "outputs": result.get("outputs", []),
            "metrics": result.get("metrics", {}),
            "message": str(result.get("message", ""))[:1000],
            "truncated": True,
            "original_result_chars": len(text),
            "max_result_chars": limit,
        }

    def _register_optional(self, tool_name: str, module_name: str, function_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, function_name)
        except (ModuleNotFoundError, AttributeError):
            return
        self.register(tool_name, fn)

    def _register_builtin_specs(self) -> None:
        specs = [
            ToolSpec(
                name="knowledge_pipeline_worker",
                owner="B",
                description=(
                    "Build the complete KnowledgeBundle from task_request and search_strategy. "
                    "It performs paper retrieval, parsing, PaperCards, EvidenceStore, Taxonomy, and CitationIndex."
                ),
                request_path="requests/knowledge_build_request.json",
                expected_outputs=[
                    "cache/knowledge_bundle.json",
                    "cache/retrieved_papers.json",
                    "cache/parsed_papers.json",
                    "cache/figure_bank.json",
                    "cache/table_bank.json",
                    "cache/paper_cards.json",
                    "cache/evidence_store.json",
                    "cache/taxonomy.json",
                    "cache/citation_index.json",
                ],
                allowed_read_paths=["cache/task_request.json", "cache/search_strategy.json", "cache/assets/"],
                allowed_write_paths=["cache/", "logs/"],
                timeout_seconds=self.default_timeout_seconds,
                max_result_chars=self.max_result_chars,
            ),
            ToolSpec(
                name="write_survey",
                owner="C",
                description=(
                    "Generate grounded survey markdown, timeline, and declared C-generated artifacts. "
                    "Paper-extracted figures/tables must come from B banks; derived charts/tables may be generated by C."
                ),
                request_path="requests/survey_generation_request.json",
                expected_outputs=["output/survey.md", "cache/timeline.json", "cache/generated_artifact_bank.json"],
                allowed_read_paths=["cache/knowledge_bundle.json", "cache/paper_cards.json", "cache/evidence_store.json"],
                allowed_write_paths=[
                    "output/survey.md",
                    "cache/timeline.json",
                    "cache/generated_artifact_bank.json",
                    "output/generated_assets/",
                ],
                timeout_seconds=self.default_timeout_seconds,
                max_result_chars=self.max_result_chars,
            ),
            ToolSpec(
                name="verify_citations",
                owner="B",
                description="Verify survey citations, paper-extracted artifacts, C-generated artifacts, and claim grounding.",
                request_path="requests/verification_request.json",
                expected_outputs=["output/citation_result.json", "cache/claim_map.json"],
                allowed_read_paths=[
                    "output/survey.md",
                    "cache/citation_ready_set.json",
                    "cache/evidence_store.json",
                    "cache/generated_artifact_bank.json",
                ],
                allowed_write_paths=["output/citation_result.json", "cache/claim_map.json"],
                timeout_seconds=self.default_timeout_seconds,
                max_result_chars=self.max_result_chars,
            ),
            ToolSpec(
                name="revise_survey",
                owner="C",
                description="Revise survey.md after verification failure without introducing new references.",
                request_path="requests/revision_request.json",
                expected_outputs=["output/survey_revised.md"],
                allowed_read_paths=["output/survey.md", "output/citation_result.json", "cache/claim_map.json"],
                allowed_write_paths=["output/survey_revised.md"],
                timeout_seconds=self.default_timeout_seconds,
                max_result_chars=self.max_result_chars,
            ),
            ToolSpec(
                name="render_report",
                owner="C",
                description="Evaluate and render the final interactive HTML/PDF report.",
                request_path="requests/evaluation_render_request.json",
                expected_outputs=["output/evaluation_report.json", "output/final_report.html", "output/final_report.pdf"],
                allowed_read_paths=["output/survey.md", "cache/", "output/citation_result.json", "output/generated_assets/"],
                allowed_write_paths=["output/"],
                timeout_seconds=self.default_timeout_seconds,
                max_result_chars=self.max_result_chars,
            ),
        ]
        self._specs.update({spec.name: spec for spec in specs})
