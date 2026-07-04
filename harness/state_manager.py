"""Persistent state for resume/debug visibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_io import read_json, write_json
from .logger import now_iso


class StateManager:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def init(self, task_id: str) -> dict[str, Any]:
        state = {
            "task_id": task_id,
            "status": "running",
            "current_stage": "init",
            "finished_stages": [],
            "ready_artifacts": {},
            "pending_artifacts": {},
            "last_update_time": now_iso(),
        }
        self.save(state)
        return state

    def load(self) -> dict[str, Any]:
        return read_json(self.state_path)

    def save(self, state: dict[str, Any]) -> None:
        state["last_update_time"] = now_iso()
        write_json(self.state_path, state)

    def advance(
        self,
        state: dict[str, Any],
        *,
        current_stage: str,
        finished_stage: str | None = None,
        ready_artifacts: dict[str, str] | None = None,
        pending_artifacts: dict[str, str] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        if finished_stage and finished_stage not in state["finished_stages"]:
            state["finished_stages"].append(finished_stage)
        state["current_stage"] = current_stage
        if status:
            state["status"] = status
        if ready_artifacts:
            state["ready_artifacts"].update(ready_artifacts)
            for key in ready_artifacts:
                state["pending_artifacts"].pop(key, None)
        if pending_artifacts:
            state["pending_artifacts"].update(pending_artifacts)
        self.save(state)
        return state
