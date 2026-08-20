"""Lightweight markdown memory for A-side planning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any


class MemoryManager:
    """Manage small markdown memories with progressive loading."""

    FILES = {
        "project": "project.md",
        "preferences": "preferences.md",
        "strategy_lessons": "strategy_lessons.md",
        "failed_cases": "failed_cases.md",
        "run_summaries": "run_summaries.md",
    }

    def __init__(self, root_dir: Path, *, enabled: bool = True, max_chars: int = 4000) -> None:
        self.root_dir = root_dir
        self.memory_dir = root_dir / "memory"
        self.enabled = enabled
        self.max_chars = max_chars
        if self.enabled:
            self.ensure_defaults()

    def ensure_defaults(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        defaults = {
            "project.md": (
                "# Project Memory\n\n"
                "EviSurvey is an evidence-grounded academic survey harness. "
                "A owns task planning, search strategy, tool orchestration, state, skills, memory, and lightweight sandbox policy.\n"
            ),
            "preferences.md": (
                "# User Preferences\n\n"
                "- Prefer stable JSON artifacts over free-form text between A/B/C.\n"
                "- Keep A changes non-invasive: do not change B/C request schemas unless necessary.\n"
            ),
            "strategy_lessons.md": (
                "# Strategy Lessons\n\n"
                "- For World Models for GameCraft, prefer internal world models, neural game engines, interactive foundation world models, "
                "GameCraft-style controllable generation, benchmarks, memory, and agent integration. Avoid social gaming and World of Warcraft noise.\n"
            ),
            "failed_cases.md": "# Failed Cases\n\n",
            "run_summaries.md": "# Run Summaries\n\n",
        }
        for filename, content in defaults.items():
            path = self.memory_dir / filename
            if not path.exists():
                path.write_text(content, encoding="utf-8")

    def load_for_strategy(self, topic: str) -> str:
        if not self.enabled:
            return ""
        topic_tokens = set(_tokens(topic))
        sections = []
        for key in ["strategy_lessons", "failed_cases", "project", "preferences"]:
            path = self.memory_dir / self.FILES[key]
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            relevant = _relevant_lines(text, topic_tokens)
            if relevant:
                sections.append(f"## {path.name}\n" + "\n".join(relevant))
        memory = "\n\n".join(sections).strip()
        if len(memory) <= self.max_chars:
            return memory
        return memory[: self.max_chars].rsplit("\n", 1)[0]

    def record_strategy_selection(
        self,
        *,
        topic: str,
        selected_mode: str,
        report: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self.ensure_defaults()
        timestamp = datetime.now().isoformat(timespec="seconds")
        candidates = report.get("candidates", {})
        false_score = candidates.get("probing_false", {}).get("score")
        true_score = candidates.get("probing_true", {}).get("score")
        reason = report.get("reason", "")
        entry = (
            f"\n## {timestamp} - {topic}\n\n"
            f"- selected: {selected_mode}\n"
            f"- probing_false_score: {false_score}\n"
            f"- probing_true_score: {true_score}\n"
            f"- reason: {reason}\n"
        )
        with (self.memory_dir / "run_summaries.md").open("a", encoding="utf-8") as handle:
            handle.write(entry)

        lesson = _lesson_from_report(topic, selected_mode, report)
        if lesson:
            with (self.memory_dir / "strategy_lessons.md").open("a", encoding="utf-8") as handle:
                handle.write(f"\n- {timestamp}: {lesson}\n")

    def append_lessons(self, topic: str, lessons: list[str]) -> None:
        """Persist strategy-agent lessons (R4 deletions, dead-query findings)."""
        if not self.enabled or not lessons:
            return
        self.ensure_defaults()
        timestamp = datetime.now().isoformat(timespec="seconds")
        with (self.memory_dir / "strategy_lessons.md").open("a", encoding="utf-8") as handle:
            for lesson in lessons:
                handle.write(f"\n- {timestamp}: {lesson}\n")

    def record_strategy_run(
        self,
        *,
        topic: str,
        strategy: dict[str, Any],
        used_memory: bool,
        mode: str,
        used_probing: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self.ensure_defaults()
        timestamp = datetime.now().isoformat(timespec="seconds")
        topic_info = strategy.get("topic_understanding", {})
        aspects = strategy.get("wide_search", {}).get("search_aspects", [])
        names = [str(aspect.get("aspect_name", "")).strip() for aspect in aspects if isinstance(aspect, dict)]
        entry = (
            f"\n## {timestamp} - {topic}\n\n"
            f"- event: search_strategy_generated\n"
            f"- mode: {mode}\n"
            f"- used_memory: {str(used_memory).lower()}\n"
            f"- used_probing: {str(used_probing).lower()}\n"
            f"- organization_mode: {topic_info.get('organization_mode', '')}\n"
            f"- aspects: {', '.join(names[:6])}\n"
        )
        with (self.memory_dir / "run_summaries.md").open("a", encoding="utf-8") as handle:
            handle.write(entry)

        if names:
            lesson = (
                f"For {topic}, the latest {mode} strategy "
                f"{'used' if used_memory else 'did not use'} memory, "
                f"{'used' if used_probing else 'did not use'} probing, "
                f"and produced aspects: {', '.join(names[:6])}."
            )
            with (self.memory_dir / "strategy_lessons.md").open("a", encoding="utf-8") as handle:
                handle.write(f"\n- {timestamp}: {lesson}\n")


def _lesson_from_report(topic: str, selected_mode: str, report: dict[str, Any]) -> str:
    selected = report.get("selected_strategy", {})
    aspects = selected.get("wide_search", {}).get("search_aspects", [])
    names = [str(aspect.get("aspect_name", "")).strip() for aspect in aspects if isinstance(aspect, dict)]
    if not names:
        return ""
    return f"For {topic}, {selected_mode} produced stronger A-side strategy aspects: {', '.join(names[:6])}."


def _relevant_lines(text: str, topic_tokens: set[str]) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line_tokens = set(_tokens(line))
        if not topic_tokens or line.startswith("-") and (line_tokens & topic_tokens):
            lines.append(line)
        elif any(marker in line.lower() for marker in ["evisurvey", "a owns", "non-invasive", "json artifacts"]):
            lines.append(line)
    return lines[:18]


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()) if token not in {"the", "and", "for", "with"}]
