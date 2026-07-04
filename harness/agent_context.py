"""Prompt/context construction for a tool-using agent loop."""

from __future__ import annotations

from typing import Any

from .skill_manager import SkillManager
from .tool_registry import ToolRegistry


def build_system_prompt(tool_registry: ToolRegistry) -> str:
    return "\n\n".join(
        [
            "You are EviSurvey's controller agent. Your job is to produce a grounded academic survey workflow.",
            "You must call tools only through the provided tool manifest.",
            (
                "Never invent paper metadata, citations, paper-extracted figures, or tables. "
                "All citations must come from CitationReadySet. C-generated derived artifacts such as charts, "
                "summary tables, and matrices are allowed only when declared in generated_artifact_bank with provenance."
            ),
            "Respect each tool's allowed read/write paths. Do not ask tools to access paths outside their manifest.",
            "If a tool fails, decide whether to retry once, request a revision, or stop with a clear failure reason.",
            "If a tool result is truncated, rely on its output files rather than the visible text summary.",
            (
                "Skills are loaded with progressive disclosure. Use the skill manifest to select only stage-relevant "
                "skills; do not load every SKILL.md into context at once."
            ),
            tool_registry.tool_prompt(),
        ]
    )


def build_agent_context(
    *,
    task_request: dict[str, Any],
    search_strategy: dict[str, Any],
    tool_registry: ToolRegistry,
    skill_manager: SkillManager,
) -> dict[str, Any]:
    return {
        "system_prompt": build_system_prompt(tool_registry),
        "task_request": task_request,
        "search_strategy": search_strategy,
        "tool_manifest": tool_registry.list_specs(),
        "skill_manifest": skill_manager.list_manifest(),
        "skill_policy": skill_manager.policy(),
        "loop_policy": {
            "max_turns": 8,
            "retry_failed_tool_once": True,
            "tool_result_max_chars": tool_registry.max_result_chars,
            "stop_when_final_state_written": True,
        },
    }
