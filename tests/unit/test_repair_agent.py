"""Unit tests for the joint-2 repair agent (harness/agents/repair_agent.py).

Spec §4.1 test_repair_agent: grouping, batch decision dispatch, param
validation (anti-hallucinated ids), action execution + incremental NLI
re-verification, backfill evidence persistence, delete-with-reason,
budget breakers, repair_log. Fakes mirror the real contracts.
"""

import json

import pytest

from harness.agents.repair_agent import (
    execute_decisions,
    group_failures,
    run_repair,
    _SciverseBudget,
)


# --- fakes --------------------------------------------------------------------


class ScriptedNLI:
    """NLI double: scriptable status per (claim substring)."""

    def __init__(self, script=None, default="unsupported"):
        self.script = script or {}      # claim substring -> status
        self.default = default
        self.calls = []

    def best_match(self, claim, evidences):
        self.calls.append((claim, evidences))
        for needle, status in self.script.items():
            if needle in claim:
                return _Result(status)
        return _Result(self.default)


class _Result:
    def __init__(self, status):
        self.status = status


class ScriptedLLM:
    """Returns {"actions": [...]} built per call by inspecting the failures listed."""

    def __init__(self, planner_fn):
        self.planner_fn = planner_fn   # (group, records, call_idx) -> actions list
        self.calls = 0
        self.messages_seen = []

    def __call__(self, messages):
        self.calls += 1
        self.messages_seen.append(messages[-1]["content"])
        return {"actions": self.planner_fn(self.calls, messages)}


class FakeSciverse:
    def __init__(self, hits_by_needle=None):
        self.hits_by_needle = hits_by_needle or {}
        self.calls = []

    def agentic_search(self, query, top_k=3):
        self.calls.append({"query": query, "top_k": top_k})
        for needle, hits in self.hits_by_needle.items():
            if needle in query:
                return {"hits": hits}
        return {"hits": []}


# --- fixtures -----------------------------------------------------------------


def _ready_set():
    return {
        "allowed_paper_ids": ["p_good", "p_other", "p_dead"],
        "items": [
            {"paper_id": "p_good", "title": "World Model Reinforcement Learning Survey", "year": 2024},
            {"paper_id": "p_other", "title": "Neural Game Engine Diffusion", "year": 2023},
            {"paper_id": "p_dead", "title": "Game Environment Generation", "year": 2022},
        ],
    }


def _evidence_store():
    return {"evidence": [
        {"evidence_id": "p_good_ev1", "paper_id": "p_good",
         "text": "World models learn latent dynamics for control in games."},
        {"evidence_id": "p_good_ev2", "paper_id": "p_good",
         "text": "DreamerV3 shows world models master diverse domains."},
    ]}


def _claim_map():
    return {"entries": [
        {"claim_text": "World models learn latent dynamics for control in games",
         "cited_paper_id": "p_good", "status": "unsupported",
         "confidence": 0.2, "evidence_ids": ["p_good_ev1"]},          # B (has evidence)
        {"claim_text": "Neural engines generate playable worlds end to end",
         "cited_paper_id": "p_dead", "status": "unsupported",
         "confidence": 0.0, "evidence_ids": []},                      # C (whitelisted, zero evidence)
        {"claim_text": "World models may improve sample efficiency",
         "cited_paper_id": "p_good", "status": "weak",
         "confidence": 0.5, "evidence_ids": ["p_good_ev2"]},          # D
        {"claim_text": "Fine already", "cited_paper_id": "p_good",
         "status": "supported", "confidence": 0.9, "evidence_ids": []},  # not a failure
    ]}


def _survey_md():
    return (
        "World models learn latent dynamics for control in games [p_good]. "
        "Neural engines generate playable worlds end to end [p_dead]. "
        "World models may improve sample efficiency [p_good]. "
        "Fine already [p_good]. "
        "A ghost sentence cites a non-whitelisted id [p_ghost].\n\n"
        "![bad figure](fig_missing_one)\n\n"
        "## References\n\n- p_good: World Model Reinforcement Learning Survey (2024).\n"
    )


def _run(tmp_path, llm, nli, sciverse=None, **kwargs):
    defaults = dict(
        task_id="task_repair_test", round_no=1, survey_md=_survey_md(),
        claim_map=_claim_map(), evidence_store=_evidence_store(),
        ready_set=_ready_set(),
        figure_items=[{"figure_id": "fig_correct", "caption": "world model game rollout figure"}],
        allowed_artifacts={"gen_timeline_1"},
        llm_json_chat=llm, nli=nli, sciverse=sciverse,
        trajectory_dir=tmp_path / "traj",
    )
    defaults.update(kwargs)
    return run_repair(**defaults)


def _group_with_evidence():
    claim_map = dict(_claim_map())
    by_paper = {"p_good": _evidence_store()["evidence"]}
    claim_map["_evidence_by_paper"] = by_paper
    return claim_map


# --- grouping -----------------------------------------------------------------


def test_grouping_routes_five_types():
    groups = group_failures(_survey_md(), _group_with_evidence(), _ready_set(),
                            [{"figure_id": "fig_correct", "caption": "world model game rollout figure"}],
                            {"gen_timeline_1"})
    # p_ghost is NOT in the whitelist -> A; p_dead is whitelisted but evidence-less -> C
    assert [r["citation_id"] for r in groups["A"]] == ["p_ghost"]
    assert len(groups["B"]) == 1 and groups["B"][0]["cited_paper_id"] == "p_good"
    assert len(groups["C"]) == 1 and groups["C"][0]["cited_paper_id"] == "p_dead"
    assert len(groups["D"]) == 1
    assert [r["ref"] for r in groups["E"]] == ["fig_missing_one"]
    # candidates attached by code: C gets paper candidates, E gets figure candidates
    assert any(c["paper_id"] == "p_other" or c["paper_id"] == "p_good"
               for c in groups["C"][0]["candidates"])
    assert groups["E"][0]["candidates"][0]["figure_id"] == "fig_correct"


def test_grouping_weak_gets_evidence_preview():
    groups = group_failures(_survey_md(), _group_with_evidence(), _ready_set(), [], set())
    # preview carries the paper's evidence chunks (store order) for the LLM to cite
    assert groups["D"][0]["evidence_preview"][0]["evidence_id"] == "p_good_ev1"
    assert len(groups["D"][0]["evidence_preview"]) == 2


# --- decisions ----------------------------------------------------------------


def test_batch_decision_dispatch_and_anti_hallucination(tmp_path):
    def plan(call_idx, messages):
        # one decision per failure the prompt lists, keyed by id
        if "Failure type A" in messages[-1]["content"]:
            return [{"id": "p_ghost", "action": "remap_citation",
                     "params": {"new_id": "MADE_UP_ID"}, "reason": "guess"}]
        if "Failure type B" in messages[-1]["content"]:
            return [{"id": "claim_0", "action": "rewrite_claim",
                     "params": {"new_text": "World models relate to latent dynamics in games."},
                     "reason": "weaken"}]
        if "Failure type C" in messages[-1]["content"]:
            return [{"id": "claim_1", "action": "backfill_evidence", "params": {},
                     "reason": "no evidence at all"}]
        if "Failure type D" in messages[-1]["content"]:
            return [{"id": "claim_2", "action": "keep", "params": {}, "reason": "acceptable"}]
        if "Failure type E" in messages[-1]["content"]:
            return [{"id": "fig_missing_one", "action": "remap_figure",
                     "params": {"new_figure_id": "fig_correct"}, "reason": "same content"}]
        return []

    llm = ScriptedLLM(plan)
    nli = ScriptedNLI(script={"latent dynamics in games": "supported",
                              "playable worlds": "supported"})
    sciverse = FakeSciverse(hits_by_needle={
        "playable worlds": [{"chunk": "Neural game engines render playable worlds from prompts.",
                             "page_no": 3}]})
    result = _run(tmp_path, llm, nli, sciverse)

    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    # A: hallucinated id rejected without execution
    assert log[("A", "p_ghost")]["outcome"] == "invalid_action"
    assert "[MADE_UP_ID]" not in result["revised_md"]
    # B: rewrite applied, citation kept, re-verified as supported
    assert log[("B", "claim_0")]["outcome"] == "repaired"
    assert "World models relate to latent dynamics in games. [p_good]" in result["revised_md"]
    # C: backfill chunk entered the evidence store and the claim flipped
    assert log[("C", "claim_1")]["outcome"] == "repaired"
    added = [e for e in result["evidence_store"]["evidence"]
             if e.get("source_type") == "agentic_chunk"]
    assert added and added[0]["paper_id"] == "p_dead"
    assert any("render playable worlds" in e["text"] for e in added)
    assert sciverse.calls[0]["top_k"] == 3
    # D: kept without changes
    assert log[("D", "claim_2")]["outcome"] == "kept"
    # E: figure remapped to the code-chosen candidate
    assert log[("E", "fig_missing_one")]["outcome"] == "repaired"
    assert "![bad figure](fig_correct)" in result["revised_md"]
    assert result["metrics"]["llm_calls"] == 5


def test_delete_requires_reason_and_removes_sentence(tmp_path):
    def plan(call_idx, messages):
        if "Failure type C" in messages[-1]["content"]:
            return [{"id": "claim_1", "action": "delete_claim", "params": {},
                     "reason": ""}]   # no reason -> invalid
        if "Failure type B" in messages[-1]["content"]:
            return [{"id": "claim_0", "action": "delete_claim",
                     "params": {"reason": "evidence contradicts"}, "reason": "x"}]
        return []

    llm = ScriptedLLM(plan)
    result = _run(tmp_path, llm, ScriptedNLI())
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("C", "claim_1")]["outcome"] == "invalid_action"   # reason required
    assert log[("B", "claim_0")]["outcome"] == "deleted"          # with reason -> executed
    assert "World models learn latent dynamics" not in result["revised_md"]


def test_llm_budget_marks_unresolved(tmp_path):
    llm = ScriptedLLM(lambda i, m: [])
    result = _run(tmp_path, llm, ScriptedNLI(), max_llm_calls=0)
    assert result["metrics"]["llm_calls"] == 0
    assert all(e["outcome"] == "unresolved" for e in result["repair_log"])
    assert result["revised_md"] == _survey_md()   # nothing executed


def test_unusable_reply_marks_batch_unresolved_after_retry(tmp_path):
    class BrokenLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, messages):
            self.calls += 1
            raise ValueError("api down")

    llm = BrokenLLM()
    result = _run(tmp_path, llm, ScriptedNLI())
    assert llm.calls >= 2                    # one repair retry happened
    assert all(e["outcome"] == "unresolved" for e in result["repair_log"])


def test_sciverse_budget_refusal(tmp_path):
    def plan(call_idx, messages):
        if "Failure type C" in messages[-1]["content"]:
            return [{"id": "claim_1", "action": "backfill_evidence", "params": {}, "reason": "r"}]
        return []

    sciverse = FakeSciverse(hits_by_needle={"playable": [{"chunk": "x"}]})
    result = _run(tmp_path, ScriptedLLM(plan), ScriptedNLI(), sciverse,
                  max_sciverse_calls=0)
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("C", "claim_1")]["outcome"] == "unresolved"
    assert sciverse.calls == []


def test_rewrite_rejected_when_it_injects_citations(tmp_path):
    def plan(call_idx, messages):
        if "Failure type B" in messages[-1]["content"]:
            return [{"id": "claim_0", "action": "rewrite_claim",
                     "params": {"new_text": "See [p_other] for latent dynamics."},
                     "reason": "sneaky"}]
        return []

    result = _run(tmp_path, ScriptedLLM(plan), ScriptedNLI())
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("B", "claim_0")]["outcome"] == "invalid_action"
    assert "See [p_other] for latent dynamics" not in result["revised_md"]


def test_trajectory_and_repair_log_written(tmp_path):
    result = _run(tmp_path, ScriptedLLM(lambda i, m: []), ScriptedNLI())
    lines = (tmp_path / "traj" / "task_repair_test_repair.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    kinds = {e["event"] for e in events}
    assert {"start", "finish"} <= kinds
    assert events[0]["failures"]["B"] == 1
    assert isinstance(result["repair_log"], list) and result["repair_log"]


# --- revise_survey seam --------------------------------------------------------


def test_revise_delegation_and_fallback(tmp_path, monkeypatch):
    """EVISURVEY_REPAIR_AGENT=1 delegates; agent crash falls back to deletion."""
    import tools.revise_survey as revise

    root = tmp_path
    (root / "output").mkdir(parents=True)
    (root / "cache").mkdir(parents=True)
    (root / "requests").mkdir()
    (root / "logs").mkdir()
    survey = root / "output" / "survey.md"
    survey.write_text(_survey_md(), encoding="utf-8")
    (root / "cache" / "claim_map.json").write_text(json.dumps(_claim_map()), encoding="utf-8")
    (root / "cache" / "citation_ready_set.json").write_text(json.dumps(_ready_set()), encoding="utf-8")
    (root / "cache" / "evidence_store.json").write_text(json.dumps(_evidence_store()), encoding="utf-8")
    (root / "cache" / "generated_artifact_bank.json").write_text('{"artifacts": []}', encoding="utf-8")
    (root / "cache" / "figure_bank.json").write_text(
        json.dumps({"figures": [{"figure_id": "fig_correct", "caption": "world model game rollout"}]}),
        encoding="utf-8")
    (root / "output" / "citation_result.json").write_text('{"entries": []}', encoding="utf-8")
    request = {"task_id": "t1", "inputs": {
        "survey_markdown_path": "output/survey.md",
        "citation_result_path": "output/citation_result.json",
        "claim_map_path": "cache/claim_map.json",
        "citation_ready_set_path": "cache/citation_ready_set.json",
        "evidence_store_path": "cache/evidence_store.json",
        "figure_bank_path": "cache/figure_bank.json",
        "generated_artifact_bank_path": "cache/generated_artifact_bank.json",
    }, "outputs": {"revised_survey_markdown_path": "output/survey_revised.md"}}
    request_path = root / "requests" / "revision_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    monkeypatch.chdir(root)
    monkeypatch.setenv("EVISURVEY_REPAIR_AGENT", "1")

    # revise resolves everything against load_config().root_dir — point it at tmp
    import types

    fake_cfg = types.SimpleNamespace(root_dir=root, intern_api_key="fake-key",
                                     intern_api_base_url="https://intern.example/api/v1",
                                     intern_model_name="intern-s2-preview",
                                     sciverse_api_base_url="https://api.sciverse.space",
                                     sciverse_api_token="")
    monkeypatch.setattr(revise, "load_config", lambda: fake_cfg)

    import harness.agents.repair_agent as repair_mod

    original = repair_mod.run_repair
    repair_mod.run_repair = lambda **kw: (_ for _ in ()).throw(RuntimeError("agent boom"))
    try:
        result = revise.run(str(request_path))
        assert result["status"] == "success"     # fell back to deletion, run not lost
        assert (root / "output" / "survey_revised.md").exists()
    finally:
        repair_mod.run_repair = original

    # second call with a working agent end-to-end through the seam
    monkeypatch.setattr(revise, "_make_llm_json_chat",
                        lambda cfg: ScriptedLLM(lambda i, m: []))
    monkeypatch.setattr(revise, "_make_sciverse", lambda cfg: FakeSciverse())
    monkeypatch.setattr(revise, "_nli_model", lambda: ScriptedNLI(), raising=False)
    result = revise.run(str(request_path))
    assert result["status"] == "success"
    assert (root / "output" / "repair_log.json").exists()
    log_entries = json.loads((root / "output" / "repair_log.json").read_text())
    assert log_entries  # unresolved round still logged honestly


def test_claims_citing_invalid_ids_skip_bc_groups():
    """Claims citing a non-whitelist id are A-type collateral: excluded from
    B/C/D so one remap fixes every occurrence instead of per-claim re-deciding."""
    claim_map = dict(_claim_map())
    claim_map["entries"] = claim_map["entries"] + [
        {"claim_text": "Ghost claim repeats the bad id", "cited_paper_id": "p_ghost",
         "status": "unsupported", "confidence": 0.0, "evidence_ids": []},
    ]
    groups = group_failures(_survey_md(), _claim_map_with(claim_map), _ready_set(), [], set())
    assert [r["citation_id"] for r in groups["A"]] == ["p_ghost"]
    cited = [r["cited_paper_id"] for r in groups["B"] + groups["C"] + groups["D"]]
    assert "p_ghost" not in cited          # collateral claim skipped
    assert groups["C"][0]["cited_paper_id"] == "p_dead"   # real zero-evidence stays


def _claim_map_with(claim_map):
    by_paper = {"p_good": _evidence_store()["evidence"]}
    claim_map["_evidence_by_paper"] = by_paper
    return claim_map
