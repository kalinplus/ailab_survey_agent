"""Stage-scoped skill discovery for A-owned orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    path: str
    applies_to: list[str]
    owner: str
    load_mode: str = "stage_scoped"


class SkillManager:
    """Expose skill metadata without loading full skill bodies by default."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self._skills = self._default_skills()

    def list_manifest(self) -> list[dict[str, Any]]:
        return [asdict(skill) for skill in self._skills]

    def active_for_stage(self, stage: str) -> list[dict[str, Any]]:
        active = []
        for skill in self._skills:
            if stage not in skill.applies_to:
                continue
            item = asdict(skill)
            item["purpose"] = skill.description
            active.append(item)
        return active

    def policy(self) -> dict[str, Any]:
        return {
            "progressive_disclosure": True,
            "load_index_first": True,
            "only_load_stage_relevant_skills": True,
            "request_scoped_activation": True,
            "unload_after_stage": True,
            "skill_body_loading": (
                "A passes skill paths in requests. B/C load the referenced SKILL.md only "
                "when their worker uses an LLM or needs the SOP."
            ),
        }

    def _default_skills(self) -> list[SkillSpec]:
        return [
            SkillSpec(
                name="knowledge_build",
                description="Build KnowledgeBundle artifacts from task and search strategy with fallback rules.",
                path="skills/knowledge_build/SKILL.md",
                applies_to=["knowledge_build"],
                owner="B",
            ),
            SkillSpec(
                name="paper_card",
                description="Normalize papers into PaperCards and bind every core claim to evidence.",
                path="skills/paper_card/SKILL.md",
                applies_to=["knowledge_build"],
                owner="B",
            ),
            SkillSpec(
                name="survey_writing",
                description="Write grounded survey sections using CitationReadySet and EvidenceStore only.",
                path="skills/survey_writing/SKILL.md",
                applies_to=["survey_generation"],
                owner="C",
            ),
            SkillSpec(
                name="artifact_generation",
                description="Generate timeline, taxonomy graph, tables, and matrices with declared provenance.",
                path="skills/artifact_generation/SKILL.md",
                applies_to=["survey_generation", "evaluation_render"],
                owner="C",
            ),
            SkillSpec(
                name="citation_verification",
                description="Verify citations, paper artifacts, generated artifacts, and claim grounding.",
                path="skills/citation_verification/SKILL.md",
                applies_to=["verification"],
                owner="B",
            ),
            SkillSpec(
                name="survey_revision",
                description="Revise invalid citations, illegal artifacts, and unsupported claims without new references.",
                path="skills/survey_revision/SKILL.md",
                applies_to=["revision"],
                owner="C",
            ),
            SkillSpec(
                name="evaluation",
                description="Score coverage, citation validity, evidence grounding, structure, and visualization.",
                path="skills/evaluation/SKILL.md",
                applies_to=["evaluation_render"],
                owner="C",
            ),
            SkillSpec(
                name="report_rendering",
                description="Render final HTML/PDF with citation summary, evidence panel, and artifact provenance.",
                path="skills/report_rendering/SKILL.md",
                applies_to=["evaluation_render"],
                owner="C",
            ),
        ]
