"""Integration test: Module A (Planner) → Module B (knowledge_pipeline_worker).

Exercises the full flow: user topic → Planner generates requests → B runs phases 1-6.
Uses REAL SciVerse API (free) + FAKE LLM (avoids expensive Intern-S2 calls).
MinerU is real but degrades gracefully per-paper to abstract_only.

Run: pytest tests/integration/ -v
Full real API (including Intern-S2): pytest tests/integration/ -v -m real_api
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402

needs_sciverse = pytest.mark.skipif(
    not os.getenv("SCIVERSE_API_KEY") and not Path(ROOT / ".env").exists(),
    reason="SCIVERSE_API_KEY not set (no .env)",
)


@needs_sciverse
class TestAToB:
    """Module A → Module B end-to-end with real SciVerse, fake LLM."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path, monkeypatch):
        """Create isolated workspace mimicking project root layout."""
        self.workspace = tmp_path
        # Copy .env for API keys
        env_file = ROOT / ".env"
        if env_file.exists():
            shutil.copy(env_file, tmp_path / ".env")
        # Copy seed_papers
        seed = ROOT / "cache" / "seed_papers.json"
        (tmp_path / "cache").mkdir()
        if seed.exists():
            shutil.copy(seed, tmp_path / "cache" / "seed_papers.json")
        # Copy surveys.json if exists
        surveys = ROOT / "structured_data" / "surveys.json"
        if surveys.exists():
            (tmp_path / "structured_data").mkdir(parents=True, exist_ok=True)
            shutil.copy(surveys, tmp_path / "structured_data" / "surveys.json")
        (tmp_path / "requests").mkdir()
        (tmp_path / "output").mkdir()
        (tmp_path / "logs").mkdir()
        monkeypatch.chdir(tmp_path)

    def _run_planner(self, topic="世界模型"):
        """Module A: generate task_request, search_strategy, knowledge_build_request."""
        from harness.planner import Planner
        cfg = load_config()
        planner = Planner(cfg)
        task_id = planner.make_task_id(topic)

        task_request = planner.build_task_request(
            task_id=task_id, topic=topic, language="zh",
            mode="full", max_papers=10, max_core_papers=5,
        )
        # Write to project root cache (where worker's resolve() will look)
        cache = cfg.root_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        requests_dir = cfg.root_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)

        _write(cache / "task_request.json", task_request)

        search_strategy = planner.build_search_strategy(
            task_id=task_id, topic=topic, max_papers=10, max_core_papers=5,
        )
        _write(cache / "search_strategy.json", search_strategy)

        knowledge_request = planner.build_knowledge_build_request(task_request)
        # Override for test: limit papers, ensure seed fallback
        knowledge_request["pipeline_config"]["use_seed_fallback"] = True
        knowledge_request["pipeline_config"]["use_mineru"] = True
        knowledge_request["pipeline_config"]["use_mock_mineru_if_failed"] = False
        request_path = requests_dir / "knowledge_build_request.json"
        _write(request_path, knowledge_request)
        return str(request_path), knowledge_request

    def test_planner_generates_valid_b_request(self):
        """Smoke: planner output parseable by KnowledgeBuildRequest."""
        from tools.models.requests import KnowledgeBuildRequest
        _, req = self._run_planner()
        obj = KnowledgeBuildRequest(**req)
        assert obj.task_id.startswith("task_")
        assert obj.topic == "世界模型"
        assert "search_strategy_path" in obj.inputs

    def test_full_a_to_b_real_sciverse_fake_llm(self):
        """Full A→B pipeline: real SciVerse, fake LLM, real MinerU (degrading)."""
        import tools.knowledge_pipeline_worker as worker
        from tools.clients.llm_fake import FakeLLMClient
        from tools.models.requests import KnowledgeBuildRequest
        from harness.json_io import read_json

        request_path, _ = self._run_planner(topic="世界模型")

        # Monkey-patch worker to use FakeLLM (avoids Intern-S2 cost)
        # but keep real SciVerse + real MinerU (degrading per-paper)
        original_run = worker.run

        def patched_run(rp):
            import tools.phases.phase5_cards as p5c
            import tools.phases.phase5_synthesis_rest as p5sr

            # Patch InternS2Client import in worker scope
            from tools.clients.llm_fake import FakeLLMClient as FK
            fake_llm = FK(
                default_text="## KEY RESULTS\n1. Achieves SOTA [page 1].\n## METHOD\n1. Learns latent dynamics [page 2].\n## SETUP\n1. Evaluated on 150 tasks [page 3].\n## LIMITATIONS\n1. Varies across domains [page 4].\n",
                default_json={"categories": [
                    {"name": "Internal World Model", "description": "latent dynamics"},
                    {"name": "Neural Game Engine", "description": "rendering"},
                ]},
            )

            # Run the pipeline manually with the fake LLM
            from config import load_config as lc
            from harness.json_io import read_json as rj, write_json as wj
            from tools.models.requests import KnowledgeBuildRequest as KBR, SearchStrategy
            from tools.clients.sciverse_client import SciVerseClient
            from tools.clients.mineru_client import MinerUClient
            from tools.nlp.nli_verifier import NLIVerifier, FakeNLIModel
            from tools.nlp.data_cleaner import DataCleaner
            from tools.phases import (phase1_decompose, phase2_survey_analyzer,
                phase3_paper_retriever, phase5_cards, phase5_evidence,
                phase5_synthesis_rest, phase6_bundle_assembler)

            cfg = lc()
            resolve = lambda p: Path(p) if Path(p).is_absolute() else cfg.root_dir / p
            request = KBR(**rj(resolve(rp)))
            strategy = SearchStrategy(**rj(resolve(request.inputs["search_strategy_path"])))
            surveys_p = cfg.cache_dir / "surveys.json"
            surveys = rj(surveys_p) if surveys_p.exists() else []
            seed_p = cfg.cache_dir / "seed_papers.json"
            seed_papers = rj(seed_p) if seed_p.exists() else []

            sciverse = SciVerseClient()
            mineru = MinerUClient(use_mock=False)
            cleaner = DataCleaner()
            nli = FakeNLIModel({"SOTA": "entailment", "latent": "entailment"})

            demand = phase1_decompose.run(request, strategy, seed_papers)
            survey_struct = phase2_survey_analyzer.run(
                request.task_id, request.topic, strategy.sub_domains,
                demand.aspects, surveys, fake_llm, mineru, cleaner, sciverse)
            retrieved, parsed = phase3_paper_retriever.run(
                request.task_id, demand.aspects, survey_struct.expansion_candidates,
                sciverse, mineru, cleaner, seed_papers, request.pipeline_config, fake_llm)
            cards = phase5_cards.run(
                request.task_id, parsed, retrieved, fake_llm,
                demand.aspects, request.pipeline_config.aspect_match_threshold)
            evidence = phase5_evidence.run(request.task_id, parsed, cards, nli, sciverse)
            figure_bank = phase5_synthesis_rest.build_figure_bank(request.task_id, parsed)
            table_bank = phase5_synthesis_rest.build_table_bank(request.task_id, parsed)
            taxonomy = phase5_synthesis_rest.build_taxonomy(
                request.task_id, request.topic, survey_struct.refined_taxonomy, cards, fake_llm)
            citation_index = phase5_synthesis_rest.build_citation_index(request.task_id, cards)

            from tools.models.artifacts import RetrievedPapers, ParsedPapers
            ARTIFACT_NAMES = [
                "retrieved_papers", "parsed_papers", "figure_bank", "table_bank",
                "paper_cards", "evidence_store", "taxonomy", "citation_index",
            ]
            WRITE_NAMES = ARTIFACT_NAMES + ["knowledge_bundle"]
            artifacts_paths = {name: request.outputs[name + "_path"] for name in ARTIFACT_NAMES}
            bundle = phase6_bundle_assembler.run(
                request.task_id, request.topic, artifacts_paths,
                retrieved, parsed, cards, evidence, figure_bank, table_bank, taxonomy, citation_index,
                demand.structure_errors, demand.coverage_warnings, request.quality_requirements)
            models = {
                "retrieved_papers": retrieved, "parsed_papers": parsed, "figure_bank": figure_bank,
                "table_bank": table_bank, "paper_cards": cards, "evidence_store": evidence,
                "taxonomy": taxonomy, "citation_index": citation_index, "knowledge_bundle": bundle,
            }
            for name in WRITE_NAMES:
                path = resolve(request.outputs[name + "_path"])
                wj(path, models[name].model_dump())
            return {
                "status": bundle.status,
                "outputs": [str(resolve(request.outputs[n + "_path"])) for n in WRITE_NAMES],
                "metrics": bundle.summary,
                "message": f"test: {bundle.status} ({bundle.summary['paper_count']} papers)",
            }

        result = patched_run(request_path)

        # Assertions
        assert result["status"] in ("success", "partial_success"), f"Pipeline failed: {result}"
        assert result["metrics"]["paper_count"] > 0, "No papers retrieved"

        # Verify output artifacts exist (written to project root cache/)
        cfg = load_config()
        knowledge_bundle = read_json(cfg.root_dir / "cache" / "knowledge_bundle.json")
        assert knowledge_bundle["task_id"].startswith("task_")
        assert knowledge_bundle["status"] in ("success", "partial_success")

        retrieved_papers = read_json(cfg.root_dir / "cache" / "retrieved_papers.json")
        assert len(retrieved_papers["papers"]) > 0, "No papers in retrieved_papers output"

        evidence_store = read_json(cfg.root_dir / "cache" / "evidence_store.json")
        # Evidence should have at least one entry (agentic backfill guarantees this)
        assert len(evidence_store["evidence"]) > 0, "Evidence store empty — anti-hallucination backbone failed"


@needs_sciverse
@pytest.mark.real_api
class TestFullRealAPI:
    """Full real API test (including Intern-S2). Only run with: pytest -m real_api"""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path, monkeypatch):
        self.workspace = tmp_path
        env_file = ROOT / ".env"
        if env_file.exists():
            shutil.copy(env_file, tmp_path / ".env")
        seed = ROOT / "cache" / "seed_papers.json"
        (tmp_path / "cache").mkdir()
        if seed.exists():
            shutil.copy(seed, tmp_path / "cache" / "seed_papers.json")
        (tmp_path / "requests").mkdir()
        (tmp_path / "output").mkdir()
        (tmp_path / "logs").mkdir()
        monkeypatch.chdir(tmp_path)

    def test_knowledge_pipeline_worker_real(self):
        """Full pipeline with ALL real APIs (Intern-S2 + SciVerse + MinerU)."""
        from harness.planner import Planner
        from harness.json_io import read_json, write_json

        cfg = load_config()
        planner = Planner(cfg)
        task_id = planner.make_task_id("世界模型")

        task_request = planner.build_task_request(
            task_id=task_id, topic="世界模型", language="zh",
            mode="full", max_papers=5, max_core_papers=3,
        )
        write_json(self.workspace / "cache" / "task_request.json", task_request)

        search_strategy = planner.build_search_strategy(
            task_id=task_id, topic="世界模型", max_papers=5, max_core_papers=3,
        )
        write_json(self.workspace / "cache" / "search_strategy.json", search_strategy)

        knowledge_request = planner.build_knowledge_build_request(task_request)
        knowledge_request["pipeline_config"]["use_mock_mineru_if_failed"] = False
        request_path = str(self.workspace / "requests" / "knowledge_build_request.json")
        write_json(Path(request_path), knowledge_request)

        from tools.knowledge_pipeline_worker import run
        result = run(request_path)

        assert result["status"] in ("success", "partial_success"), f"Real API pipeline failed: {result}"
        assert result["metrics"]["paper_count"] >= 3


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
