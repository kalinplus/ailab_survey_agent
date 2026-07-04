"""B tool entry: build the full KnowledgeBundle (Phases 1-6). Thin orchestrator."""

from __future__ import annotations

from pathlib import Path

from config import load_config
from llm_client import InternS2Client
from harness.json_io import read_json, write_json

from tools.models.requests import KnowledgeBuildRequest, SearchStrategy
from tools.clients.sciverse_client import SciVerseClient
from tools.clients.mineru_client import MinerUClient
from tools.nlp.nli_verifier import NLIVerifier
from tools.nlp.data_cleaner import DataCleaner
from tools.phases import (phase1_decompose, phase2_survey_analyzer, phase3_paper_retriever,
    phase5_cards, phase5_evidence, phase5_synthesis_rest, phase6_bundle_assembler)

# The 8 artifact names the harness validator requires as bare keys in bundle.artifacts
# (harness/knowledge_bundle_validator.py REQUIRED_ARTIFACTS). request.outputs carries them
# with a "_path" suffix + a 9th knowledge_bundle_path; we map bare name -> path here.
ARTIFACT_NAMES = [
    "retrieved_papers", "parsed_papers", "figure_bank", "table_bank",
    "paper_cards", "evidence_store", "taxonomy", "citation_index",
]
WRITE_NAMES = ARTIFACT_NAMES + ["knowledge_bundle"]


def run(request_path: str) -> dict:
    cfg = load_config()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else cfg.root_dir / path

    request = KnowledgeBuildRequest(**read_json(resolve(request_path)))

    # inputs (interface doc §6.2): task_request_path, search_strategy_path
    strategy = SearchStrategy(**read_json(resolve(request.inputs["search_strategy_path"])))
    # surveys + seed_papers live in cache/ (plan File Structure); fall back to [] when absent.
    surveys = read_json(resolve(str(cfg.cache_dir / "surveys.json"))) if (cfg.cache_dir / "surveys.json").exists() else []
    seed_path = cfg.cache_dir / "seed_papers.json"
    seed_papers = read_json(seed_path) if seed_path.exists() else []

    # clients
    llm = InternS2Client(cfg)
    sciverse = SciVerseClient()
    mineru = MinerUClient(use_mock=request.pipeline_config.use_mock_mineru_if_failed)
    cleaner = DataCleaner()

    # Phase 1-6 (signatures verified against source; Phase 4 RAG removed — superseded by agentic-search)
    demand = phase1_decompose.run(request, strategy, seed_papers)
    survey_struct = phase2_survey_analyzer.run(
        request.task_id, request.topic, strategy.sub_domains,
        demand.aspects, surveys, llm, mineru, cleaner, sciverse)
    retrieved, parsed = phase3_paper_retriever.run(
        request.task_id, demand.aspects, survey_struct.expansion_candidates,
        sciverse, mineru, cleaner, seed_papers, request.pipeline_config)
    cards = phase5_cards.run(
        request.task_id, parsed, retrieved, llm,
        demand.aspects, request.pipeline_config.aspect_match_threshold)
    evidence = phase5_evidence.run(request.task_id, parsed, cards, NLIVerifier(), sciverse)
    figure_bank = phase5_synthesis_rest.build_figure_bank(request.task_id, parsed)
    table_bank = phase5_synthesis_rest.build_table_bank(request.task_id, parsed)
    taxonomy = phase5_synthesis_rest.build_taxonomy(
        request.task_id, request.topic, survey_struct.refined_taxonomy, cards, llm)
    citation_index = phase5_synthesis_rest.build_citation_index(request.task_id, cards)

    # bundle.artifacts MUST use the 8 BARE names (validator reads artifacts["retrieved_papers"] etc.)
    artifacts_paths = {name: request.outputs[name + "_path"] for name in ARTIFACT_NAMES}
    bundle = phase6_bundle_assembler.run(
        request.task_id, request.topic, artifacts_paths,
        retrieved, parsed, cards, evidence, figure_bank, table_bank, taxonomy, citation_index,
        demand.structure_errors, demand.coverage_warnings, request.quality_requirements)

    # write all 9 outputs
    models = {
        "retrieved_papers": retrieved, "parsed_papers": parsed, "figure_bank": figure_bank,
        "table_bank": table_bank, "paper_cards": cards, "evidence_store": evidence,
        "taxonomy": taxonomy, "citation_index": citation_index, "knowledge_bundle": bundle,
    }
    outputs = []
    for name in WRITE_NAMES:
        path = resolve(request.outputs[name + "_path"])
        write_json(path, models[name].model_dump())
        outputs.append(str(path))

    return {
        "status": bundle.status,
        "outputs": outputs,
        "metrics": bundle.summary,
        "message": (
            f"knowledge_pipeline_worker: {bundle.status} "
            f"({bundle.summary['paper_count']} papers, {bundle.summary['citation_count']} citations)"
        ),
    }
