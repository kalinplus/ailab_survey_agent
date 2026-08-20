"""Shared bounded agent-loop chassis (joint 1 strategy / joint 2 repair).

Typed-action loop: the LLM answers with one JSON object
``{"action": "<tool-name>", ...args}``; the chassis dispatches to the tool,
feeds the observation back, auto-injects framework state each turn, enforces
budgets, and emits one trajectory record per turn.

The toolbox IS the permission boundary — an action not in ``tools`` is
rejected without execution and the model sees the error.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# A tool receives the full decision dict and returns an observation (str or dict).
ToolFn = Callable[[dict[str, Any]], Any]
# LLM boundary: messages -> parsed JSON dict (mirrors InternS2Client.json_chat).
JsonChat = Callable[[list[dict[str, str]]], dict[str, Any]]


class LoopFinished(Exception):
    """Raised by a tool to end the loop with a payload (e.g. gate-passed accept)."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        super().__init__("loop finished")
        self.payload = payload or {}


class LoopParseError(RuntimeError):
    """Model reply was not a usable action JSON even after one repair retry."""


@dataclass
class AgentConfig:
    name: str                       # identity, lands in logs / prompt / trajectory
    system_prompt: str              # per-joint role setting
    tools: dict[str, ToolFn]        # capability == permission: absent = uncallable
    max_turns: int = 8
    max_llm_calls: int = 8
    max_wallclock: float = 300.0
    on_action: Callable[[dict[str, Any]], None] | None = None    # trajectory hook
    on_finish: Callable[[dict[str, Any]], None] | None = None    # memory / experiment log
    state_provider: Callable[[], str] | None = None              # framework state, injected every turn
    result_max_chars: int = 1500


@dataclass
class AgentResult:
    status: str                     # finished | budget_turns | budget_llm_calls | budget_wallclock
    payload: dict[str, Any] | None  # LoopFinished payload (or last decision on budget exit)
    turns: int
    llm_calls: int
    elapsed: float
    reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


class BoundedAgentLoop:
    def __init__(self, config: AgentConfig, llm_json_chat: JsonChat) -> None:
        self.config = config
        self.llm_json_chat = llm_json_chat
        self.llm_calls = 0

    def run(self, task: str) -> AgentResult:
        started = time.monotonic()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": task + self._state_block()},
        ]
        history: list[dict[str, Any]] = []
        turn = 0
        while True:
            budget = self._budget_status(turn, started)
            if budget:
                return self._finish(budget, None, turn, started, history, reason=budget)
            turn += 1
            decision = self._decide(messages)
            action = str(decision.get("action", ""))
            args = {key: value for key, value in decision.items() if key != "action"}
            turn_started = time.monotonic()
            if action in self.config.tools:
                try:
                    result = self.config.tools[action](decision)
                except LoopFinished as stop:
                    record = self._record(turn, action, args, "accepted", turn_started)
                    history.append(record)
                    return self._finish("finished", stop.payload, turn, started, history)
            else:
                result = {"error": f"unknown action {action!r}; allowed actions: {sorted(self.config.tools)}"}
            digest = _digest(result, self.config.result_max_chars)
            record = self._record(turn, action, args, digest, turn_started)
            history.append(record)
            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
            messages.append({
                "role": "user",
                "content": f"Observation:\n{digest}" + self._state_block(),
            })

    # -- internals ---------------------------------------------------------

    def _decide(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """One LLM call; on an unusable reply, one repair retry, then bubble."""
        for attempt in (1, 2):
            if attempt == 2 and self.llm_calls >= self.config.max_llm_calls:
                raise LoopParseError("repair retry skipped: llm call budget exhausted")
            self.llm_calls += 1
            try:
                decision = self.llm_json_chat(messages)
            except LoopParseError:
                raise
            except Exception as exc:  # LLM boundary: json parse / API failure
                decision = None
                parse_error = str(exc)
            else:
                if isinstance(decision, dict) and decision.get("action"):
                    return decision
                parse_error = f"reply is not an action object: {decision!r}"
            if attempt == 1:
                messages.append({"role": "user", "content": (
                    f"Your previous reply was not a usable action JSON ({parse_error}). "
                    "Reply again with exactly ONE JSON object: "
                    '{"action": "<one of the allowed actions>", ...args}'
                )})
        raise LoopParseError(f"model did not return an action JSON after repair: {parse_error}")

    def _budget_status(self, turn: int, started: float) -> str | None:
        if turn >= self.config.max_turns:
            return "budget_turns"
        if self.llm_calls >= self.config.max_llm_calls:
            return "budget_llm_calls"
        if time.monotonic() - started >= self.config.max_wallclock:
            return "budget_wallclock"
        return None

    def _state_block(self) -> str:
        if self.config.state_provider is None:
            return ""
        state = self.config.state_provider()
        return f"\n\n[framework state]\n{state}" if state else ""

    def _record(self, turn: int, action: str, args: dict[str, Any], digest: Any, turn_started: float) -> dict[str, Any]:
        record = {
            "agent": self.config.name,
            "turn": turn,
            "action": action,
            "args": args,
            "result_digest": str(digest),
            "llm_idx": self.llm_calls,
            "elapsed": round(time.monotonic() - turn_started, 3),
        }
        if self.config.on_action:
            self.config.on_action(record)
        return record

    def _finish(
        self,
        status: str,
        payload: dict[str, Any] | None,
        turn: int,
        started: float,
        history: list[dict[str, Any]],
        reason: str = "",
    ) -> AgentResult:
        elapsed = round(time.monotonic() - started, 3)
        if self.config.on_finish:
            self.config.on_finish({
                "agent": self.config.name,
                "status": status,
                "turns": turn,
                "llm_calls": self.llm_calls,
                "elapsed": elapsed,
            })
        return AgentResult(
            status=status, payload=payload, turns=turn, llm_calls=self.llm_calls,
            elapsed=elapsed, reason=reason, history=history,
        )


def _digest(result: Any, max_chars: int) -> str:
    text = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
    return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"


def trajectory_writer(path: str | Path) -> Callable[[dict[str, Any]], None]:
    """Build an on_action hook appending one jsonl line per record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return write
