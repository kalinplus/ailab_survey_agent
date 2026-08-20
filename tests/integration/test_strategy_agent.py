"""Integration test: strategy agent joint with REAL SciVerse + fake LLM.

Spec §4.2: real probe + real sampling against the live API, LLM decisions
scripted. Verifies schema legality, budget compliance, convergence, and that
a deliberately dead aspect really triggers the rescue paths on the real API.

Run: pytest tests/integration/test_strategy_agent.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from harness.agents.relevance import EmbeddingScorer  # noqa: E402
from harness.agents.strategy_agent import run_strategy_agent  # noqa: E402
from harness.search_strategy_builder import _validate_strategy  # noqa: E402

needs_sciverse = pytest.mark.skipif(
    not os.getenv("SCIVERSE_API_KEY") and not Path(ROOT / ".env").exists(),
    reason="SCIVERSE_API_KEY not set (no .env)",
)


def _keyword_scorer():
    # keyword column: keeps this test off the embedding download (bge runs in
    # the real_api smoke and the A/B script instead)
    scorer = EmbeddingScorer()
    scorer._failed = True
    return scorer


class ScriptedLLM:
    """Fake decision LLM: reply 0 names aspects, then scripted actions, then accept."""

    def __init__(self, naming_plan, actions):
        self.replies = [naming_plan] + list(actions) + [{"action": "accept"}] * 8
        self.n_messages = 0

    def __call__(self, messages):
        self.n_messages = len(messages)
        reply = self.replies.pop(0) if self.replies else {"action": "accept"}
        return reply


def _plan_with_dead_aspect():
    return {
        "main_domain": "World Models",
        "organization_mode": "thematic",
        "organization_reason": "technical blocks",
        "aspects": [
            {"name": "Latent World Model Learning", "min_papers": 3,
             "description": "World models learned in latent space for control.",
             "keywords": ["dreamer world model", "latent dynamics model",
                          "world model reinforcement learning", "recurrent state space model",
                          "model-based reinforcement learning"]},
            {"name": "Neural Game Engines", "min_papers": 3,
             "description": "Generative models that render playable game worlds.",
             "keywords": ["neural game engine", "video game diffusion",
                          "interactive video generation", "game world simulation",
                          "diffusion game rendering"]},
            {"name": "Zzxqv Flotmint Studies", "min_papers": 3,  # deliberately dead
             "description": "A fabricated sub-field that cannot exist in any corpus.",
             "keywords": ["zzxqv flotmint query", "zzxqv flotmint method",
                          "zzxqv flotmint model", "zzxqv flotmint survey",
                          "zzxqv flotmint dataset"]},
        ],
    }


@needs_sciverse
class TestStrategyAgentRealSciverse:
    def _run(self, tmp_path, actions, **kwargs):
        from tools.clients.sciverse_client import SciVerseClient

        config = load_config()
        sv = SciVerseClient(base_url=config.sciverse_api_base_url,
                            api_key=config.sciverse_api_token)
        llm = ScriptedLLM(_plan_with_dead_aspect(), actions)
        strategy = run_strategy_agent(
            task_id="task_int_strategy_001", topic="world models for games",
            max_papers=20, max_core_papers=8, end_year=2026,
            llm_json_chat=llm, sciverse=sv,
            trajectory_dir=tmp_path / "traj", scorer=_keyword_scorer(),
            **kwargs,
        )
        return strategy, llm

    def _trajectory(self, tmp_path):
        path = tmp_path / "traj" / "task_int_strategy_001_strategy.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_real_probe_sample_and_convergence(self, tmp_path):
        strategy, _ = self._run(tmp_path, actions=[
            {"action": "probe_query", "query": "zzxqv flotmint query", "filters": "none"},
            {"action": "commit_edits", "edits": [
                {"op": "rewrite", "aspect_id": "aspect_003",
                 "name": "Game Agent Benchmarks",
                 "description": "Benchmarks and evaluation for game-playing agents.",
                 "keywords": ["game agent benchmark", "atari benchmark world model",
                              "minecraft agent evaluation", "game playing llm benchmark",
                              "agent evaluation protocol"]}]},
        ])
        # schema legality: downstream P1-P6 contract unchanged
        _validate_strategy(strategy, topic="world models for games")
        aspects = strategy["wide_search"]["search_aspects"]
        assert 3 <= len(aspects) <= 6
        meta = strategy["strategy_agent"]
        # budget compliance (spec §4.4 hard assertions)
        assert meta["llm_calls"] <= 7
        assert meta["sciverse_calls"] <= 20
        assert meta["rounds"] <= 3
        # rollback guarantee
        assert meta["final_score"] >= meta["scores"][0]

        events = self._trajectory(tmp_path)
        kinds = {e.get("event") for e in events}
        assert {"start", "probe", "naming", "report_card", "finish"} <= kinds
        # real probe actually returned papers (real SciVerse)
        probe_event = next(e for e in events if e.get("event") == "probe")
        assert probe_event["n_papers"] >= 6
        # card rows carry the full schema against real hits
        first_card = next(e for e in events if e.get("event") == "report_card")
        for row in first_card["card"]["aspects"]:
            assert isinstance(row["n_hits"], int)
            assert isinstance(row["top5_titles"], list)

    def test_rescue_paths_trigger_on_real_api(self, tmp_path):
        strategy, _ = self._run(tmp_path, actions=[
            # R0: filter bypass on the dead query
            {"action": "probe_query", "query": "zzxqv flotmint query", "filters": "none"},
            # R2: semantic discovery for the dead aspect
            {"action": "discover_by_description",
             "text": "benchmarks and evaluation protocols for game-playing agents"},
        ])
        meta = strategy["strategy_agent"]
        assert meta["llm_calls"] <= 7 and meta["sciverse_calls"] <= 20
        turns = [e for e in self._trajectory(tmp_path) if "action" in e]
        by_action = {t["action"]: t for t in turns}
        # both rescue tools really executed against the real API
        assert "probe_query" in by_action
        assert "discover_by_description" in by_action
        probe_turn = by_action["probe_query"]
        assert probe_turn["args"].get("filters") == "none"  # R0 recorded in trajectory
