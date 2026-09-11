"""CLI entry point for A's EviSurvey orchestrator."""

from __future__ import annotations

import argparse

from config import ensure_project_dirs, load_config
from harness.agent_loop import AgentLoop
from harness.logger import WorkflowLogger
from harness.planner import Planner
from harness.state_manager import StateManager
from harness.tool_registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EviSurvey academic survey harness.")
    parser.add_argument("--topic", required=True, help="Academic survey topic.")
    parser.add_argument("--language", default=None, choices=["zh", "en"], help="Survey language.")
    parser.add_argument("--mode", default=None, choices=["demo", "full"], help="Run mode.")
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--max-core-papers", type=int, default=15)
    parser.add_argument(
        "--use-mineru",
        action="store_true",
        help="Parse candidate PDFs with MinerU (feeds the evidence store; off by default).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate A-owned requests and state files; do not invoke B/C tools.",
    )
    return parser


def exit_code_for_status(status: str) -> int:
    """Map final run status to process exit code.

    quality_failed (Goal Gate failed) must not exit 0 even though the
    artifacts were rendered; only prepared/completed mean the run met its goal.
    """
    return 0 if status in {"prepared", "completed"} else 1


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    ensure_project_dirs(config)

    logger = WorkflowLogger(config.logs_dir / "run.jsonl")
    state_manager = StateManager(config.logs_dir / "state.json")
    registry = ToolRegistry(
        config.root_dir,
        logger,
        default_timeout_seconds=config.tool_timeout_seconds,
        max_result_chars=config.tool_result_max_chars,
    )
    planner = Planner(config)

    loop = AgentLoop(
        config=config,
        planner=planner,
        tool_registry=registry,
        state_manager=state_manager,
        logger=logger,
    )
    final_state = loop.run(
        topic=args.topic,
        language=args.language or config.default_language,
        mode=args.mode or config.default_mode,
        max_papers=args.max_papers,
        max_core_papers=args.max_core_papers,
        prepare_only=args.prepare_only,
        use_mineru=args.use_mineru,
    )

    print(f"status: {final_state['status']}")
    print(f"task_id: {final_state['task_id']}")
    print(f"final_state: {config.output_dir / 'final_state.json'}")
    return exit_code_for_status(final_state["status"])


if __name__ == "__main__":
    raise SystemExit(main())
