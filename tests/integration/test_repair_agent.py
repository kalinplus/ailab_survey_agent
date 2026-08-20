"""Integration test: joint-2 repair agent with REAL NLI (+ REAL SciVerse backfill).

Spec §4.2: the real DeBERTa cross-encoder judges repair outcomes (incremental
re-verification), and a real agentic-search backfill grounds a zero-evidence
claim. LLM decisions stay scripted (fake) — Intern-S2 runs in the real_api
smoke / the A/B experiment instead.

Real NLI model downloads ~700MB on first use (set HF_ENDPOINT for the mirror):
RUN_NLI_REAL=1 pytest -m nli_real tests/integration/test_repair_agent.py -v
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from harness.agents.repair_agent import run_repair  # noqa: E402

needs_nli_real = pytest.mark.skipif(
    os.getenv("RUN_NLI_REAL") != "1", reason="set RUN_NLI_REAL=1 to load the real NLI model")
needs_sciverse = pytest.mark.skipif(
    not os.getenv("SCIVERSE_API_KEY") and not Path(ROOT / ".env").exists(),
    reason="SCIVERSE_API_KEY not set (no .env)")


class ScriptedLLM:
    def __init__(self, planner_fn):
        self.planner_fn = planner_fn

    def __call__(self, messages):
        return {"actions": self.planner_fn(messages[-1]["content"])}


def _fixture():
    """One B-type overclaim (rewriteable) + one C-type zero-evidence claim."""
    evidence_store = {"evidence": [
        {"evidence_id": "wm_ev1", "paper_id": "p_wm",
         "text": "DreamerV3 demonstrates that world models mastering diverse domains "
                 "through latent dynamics improve data efficiency in reinforcement learning."},
    ]}
    claim_map = {"entries": [
        # overclaim: evidence says "improve", claim says "prove optimal" -> unsupported/weak
        {"claim_text": "World models prove that latent dynamics guarantee optimal control",
         "cited_paper_id": "p_wm", "status": "unsupported", "confidence": 0.3,
         "evidence_ids": ["wm_ev1"]},
        # zero evidence for this paper
        {"claim_text": "Neural game engines generate playable game environments in real time",
         "cited_paper_id": "p_gen", "status": "unsupported", "confidence": 0.0,
         "evidence_ids": []},
    ]}
    ready_set = {"allowed_paper_ids": ["p_wm", "p_gen"], "items": [
        {"paper_id": "p_wm", "title": "World Model Reinforcement Learning", "year": 2024},
        {"paper_id": "p_gen", "title": "Neural Game Engine Generation", "year": 2024},
    ]}
    survey_md = (
        "World models prove that latent dynamics guarantee optimal control [p_wm]. "
        "Neural game engines generate playable game environments in real time [p_gen].\n"
        "## References\n\n- p_wm: World Model Reinforcement Learning (2024).\n")
    return survey_md, claim_map, evidence_store, ready_set


@pytest.mark.nli_real
@needs_nli_real
class TestRepairRealNLI:
    def test_rewrite_weakening_flips_real_nli_verdict(self, tmp_path):
        """Rewrite to the evidence-supported weak form -> real cross-encoder flips it."""
        survey_md, claim_map, evidence_store, ready_set = _fixture()

        def plan(content):
            if "Failure type B" in content:
                return [{"id": "claim_0", "action": "rewrite_claim",
                         "params": {"new_text":
                                     "World models with latent dynamics have been shown to "
                                     "improve data efficiency in reinforcement learning"},
                         "reason": "weaken to what the evidence supports"}]
            if "Failure type C" in content:
                return [{"id": "claim_1", "action": "keep", "params": {}, "reason": "later"}]
            return []

        from tools.nlp.nli_verifier import NLIVerifier

        result = run_repair(
            task_id="task_int_repair_001", round_no=1, survey_md=survey_md,
            claim_map=claim_map, evidence_store=evidence_store, ready_set=ready_set,
            figure_items=[], allowed_artifacts=set(),
            llm_json_chat=ScriptedLLM(plan), nli=NLIVerifier(), sciverse=None,
            trajectory_dir=tmp_path / "traj")
        log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
        assert log[("B", "claim_0")]["outcome"] == "repaired"
        # the weakened sentence replaced the overclaim, citation kept
        assert "improve data efficiency in reinforcement learning [p_wm]" in result["revised_md"]
        assert "guarantee optimal control" not in result["revised_md"]

    def test_backfill_real_sciverse_grounded(self, tmp_path):
        """C-type zero-evidence claim grounded by a REAL agentic-search chunk."""
        survey_md, claim_map, evidence_store, ready_set = _fixture()

        def plan(content):
            if "Failure type C" in content:
                return [{"id": "claim_1", "action": "backfill_evidence",
                         "params": {}, "reason": "paper has zero evidence"}]
            if "Failure type B" in content:
                return [{"id": "claim_0", "action": "keep", "params": {}, "reason": "later"}]
            return []

        from config import load_config
        from tools.clients.sciverse_client import SciVerseClient
        from tools.nlp.nli_verifier import NLIVerifier

        cfg = load_config()
        sciverse = SciVerseClient(base_url=cfg.sciverse_api_base_url,
                                  api_key=cfg.sciverse_api_token)
        result = run_repair(
            task_id="task_int_repair_002", round_no=1, survey_md=survey_md,
            claim_map=claim_map, evidence_store=evidence_store, ready_set=ready_set,
            figure_items=[], allowed_artifacts=set(),
            llm_json_chat=ScriptedLLM(plan), nli=NLIVerifier(), sciverse=sciverse,
            trajectory_dir=tmp_path / "traj")
        log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
        added = [e for e in result["evidence_store"]["evidence"]
                 if e.get("source_type") == "agentic_chunk"]
        # a real search ran and its chunks entered the store bound to the cited paper
        assert added and all(e["paper_id"] == "p_gen" for e in added)
        assert log[("C", "claim_1")]["outcome"] in {"repaired", "still_unsupported"}
        # whether the real chunks entail the claim is the model's honest verdict —
        # both outcomes are acceptable, but a repaired verdict must mean real support
        if log[("C", "claim_1")]["outcome"] == "repaired":
            assert result["metrics"].get("repaired", 0) >= 1
