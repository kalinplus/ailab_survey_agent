"""Optional LLM-driven tool loop for demos that need explicit agent behavior."""

from __future__ import annotations

from typing import Any

from llm_client import InternS2Client
from .agent_context import build_system_prompt
from .logger import WorkflowLogger
from .tool_registry import ToolRegistry


class AgentToolLoop:
    """Minimal JSON-action loop: model chooses tool calls, registry executes them."""

    def __init__(
        self,
        *,
        llm_client: InternS2Client,
        tool_registry: ToolRegistry,
        logger: WorkflowLogger,
        max_turns: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.logger = logger
        self.max_turns = max_turns

    def run(self, user_request: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": build_system_prompt(self.tool_registry)},
            {"role": "user", "content": user_request},
        ]
        tool_history: list[dict[str, Any]] = []
        failed_once: set[tuple[str, str]] = set()

        for turn in range(1, self.max_turns + 1):
            decision = self.llm_client.json_chat(messages)
            self.logger.log("agent_decision", turn=turn, decision=decision)

            if decision.get("action") == "final":
                return {
                    "status": "success",
                    "answer": decision.get("answer", ""),
                    "tool_history": tool_history,
                }

            if decision.get("action") != "call_tool":
                return {
                    "status": "failed",
                    "message": f"Unsupported model action: {decision}",
                    "tool_history": tool_history,
                }

            tool_name = str(decision.get("tool_name", ""))
            request_path = str(decision.get("request_path", ""))
            result = self.tool_registry.run(tool_name, request_path)
            context_result = self.tool_registry.truncate_for_context(result)
            tool_history.append(context_result)

            retry_key = (tool_name, request_path)
            if result.get("status") == "failed":
                if retry_key in failed_once:
                    return {
                        "status": "failed",
                        "message": f"Tool failed twice: {tool_name}",
                        "tool_history": tool_history,
                    }
                failed_once.add(retry_key)

            messages.append({"role": "assistant", "content": repr(decision)})
            messages.append({"role": "user", "content": f"Tool result:\n{context_result!r}"})

        return {
            "status": "failed",
            "message": f"Agent tool loop exceeded max_turns={self.max_turns}.",
            "tool_history": tool_history,
        }
