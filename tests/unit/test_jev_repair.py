"""Unit tests for the JEV decision path in the repair agent.

Covers tools/clients/jev_client.py (request builder, confidence gating,
speculative param questions) and its seam in repair_agent.decide_batches
(settled decisions, rewrite generation via Intern-S2, low-confidence /
no-candidate / API-failure fallback to the Intern-S2 batch prompt).

Fakes mirror the real contracts: FakeJevClient stands in for
TypeSafeClient.system_one (state + questions -> answers with choice /
confidence / probabilities), JevRepairDecider runs unmocked on top of it.
"""

import pytest

from harness.agents.repair_agent import run_repair
from tools.clients.jev_client import (
    ACTION_DESCRIPTIONS,
    JevRepairDecider,
    build_request,
    make_jev_decider,
)

from tests.unit.test_repair_agent import (
    FakeSciverse,
    ScriptedLLM,
    ScriptedNLI,
    _claim_map,
    _evidence_store,
    _ready_set,
    _survey_md,
)


# --- fakes (SDK boundary) ------------------------------------------------------


class _ChoiceAnswer:
    def __init__(self, choice, confidence=0.9, probabilities=None):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities or {choice: 1.0}


class _Response:
    usage = None

    def __init__(self, answers):
        self.answers = answers


class FakeJevClient:
    """Scripted system_one: failure id -> {"action", "param", "conf"};
    "param": None answers the speculative question none_of_the_above."""

    def __init__(self, script, confidence=0.9, fail_groups=()):
        self.script = script
        self.confidence = confidence
        self.fail_groups = fail_groups
        self.calls = []  # (state, questions) per call

    def system_one(self, state, questions):
        group = state["failure_group"]
        if group in self.fail_groups:
            raise RuntimeError("jev api down")
        self.calls.append((state, questions))
        answers = {}
        for i, record in enumerate(state["failures"]):
            fid = record.get("citation_id") or record.get("claim_id") or record.get("ref")
            row = self.script.get(fid, {})
            if not row:
                continue  # jev did not answer this record
            answers[f"act_{i}"] = _ChoiceAnswer(
                row["action"], row.get("conf", self.confidence))
            if f"param_{i}" in questions and row.get("param", "") != "":
                answers[f"param_{i}"] = _ChoiceAnswer(
                    row.get("param") or "none_of_the_above")
        return _Response(answers)


def _decider(script, **kwargs):
    return JevRepairDecider(FakeJevClient(script, **kwargs))


def _run(tmp_path, llm, nli, sciverse=None, jev="unset", **kwargs):
    defaults = dict(
        task_id="task_jev_test", round_no=1, survey_md=_survey_md(),
        claim_map=_claim_map(), evidence_store=_evidence_store(),
        ready_set=_ready_set(),
        figure_items=[{"figure_id": "fig_correct", "caption": "world model game rollout figure"}],
        allowed_artifacts={"gen_timeline_1"},
        llm_json_chat=llm, nli=nli, sciverse=sciverse,
        trajectory_dir=tmp_path / "traj",
        jev=jev,
    )
    defaults.update(kwargs)
    return run_repair(**defaults)


# --- request builder ------------------------------------------------------------


def test_build_request_adapts_criteria_and_param_questions():
    records = [
        {  # has paper candidates -> remap offered + param question
            "citation_id": "p_ghost", "sentence": "World models in games",
            "candidates": [{"paper_id": "p_good", "title": "World Model RL"}],
        },
        {  # no candidates -> selection action dropped from this record's options
            "citation_id": "p_dead", "sentence": "Nothing matches here",
            "candidates": [],
        },
    ]
    state, questions, param_slots = build_request(
        "A", ["remap_citation", "delete_claim"], records)
    assert state["failures"] == records
    assert set(questions) == {"act_0", "act_1", "param_0"}
    assert set(questions["act_0"].criteria) == {"remap_citation", "delete_claim"}
    assert set(questions["act_1"].criteria) == {"delete_claim"}  # remap withheld
    assert param_slots == {0: {"action": "remap_citation", "param": "new_id",
                               "question": "param_0"}}
    param_criteria = questions["param_0"].criteria
    assert set(param_criteria) == {"p_good", "none_of_the_above"}


def test_build_request_swap_evidence_param_from_preview():
    records = [{"claim_id": "claim_0", "claim_text": "x", "cited_paper_id": "p_good",
                "evidence_preview": [{"evidence_id": "p_good_ev1", "text": "t"}]}]
    _state, questions, param_slots = build_request(
        "B", ["rewrite_claim", "swap_evidence", "backfill_evidence", "delete_claim"], records)
    assert param_slots == {0: {"action": "swap_evidence", "param": "evidence_id",
                               "question": "param_0"}}
    assert set(questions["param_0"].criteria) == {"p_good_ev1", "none_of_the_above"}


def test_decide_group_synthesizes_delete_reason_and_confidence_gate():
    records = [{"citation_id": "p_ghost", "sentence": "s", "candidates": []}]
    jev = JevRepairDecider(FakeJevClient({"p_ghost": {"action": "delete_claim", "conf": 0.42}}),
                           min_confidence=0.5)
    out = jev.decide_group("A", ["remap_citation", "delete_claim"], records)
    assert out[0]["action"] is None
    assert "confidence 0.42" in out[0]["reason"]

    jev = JevRepairDecider(FakeJevClient({"p_ghost": {"action": "delete_claim", "conf": 0.8}}))
    out = jev.decide_group("A", ["remap_citation", "delete_claim"], records)
    assert out[0]["action"] == "delete_claim"
    assert out[0]["reason"].startswith("jev p(delete_claim)=")


# --- end-to-end through run_repair ----------------------------------------------


def test_jev_full_path_settles_rewrites_and_remaps(tmp_path):
    script = {
        "p_ghost": {"action": "delete_claim"},          # no candidates -> delete is the only fit
        "claim_0": {"action": "rewrite_claim"},         # needs Intern-S2 for new_text
        "claim_1": {"action": "backfill_evidence"},
        "claim_2": {"action": "keep"},
        "fig_missing_one": {"action": "remap_figure", "param": "fig_correct"},
    }

    def plan(call_idx, messages):
        content = messages[-1]["content"]
        assert "repair writer" in messages[0]["content"]  # rewrite-only prompt
        if "claim_0" in content:
            return [{"id": "claim_0", "action": "rewrite_claim",
                     "params": {"new_text": "World models relate to latent dynamics in games."},
                     "reason": "weaken"}]
        return []

    sciverse = FakeSciverse(hits_by_needle={
        "playable worlds": [{"chunk": "Neural game engines render playable worlds from prompts.",
                             "title": "Game Environment Generation", "doc_id": "doc_dead",
                             "page_no": 3}]})
    result = _run(tmp_path, ScriptedLLM(plan),
                  ScriptedNLI(script={"latent dynamics in games": "supported",
                                      "playable worlds": "supported"}),
                  sciverse, jev=_decider(script))

    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("A", "p_ghost")]["outcome"] == "deleted"
    assert "p_ghost" not in result["revised_md"]
    assert log[("B", "claim_0")]["outcome"] == "repaired"
    assert "World models relate to latent dynamics in games [p_good]." in result["revised_md"]
    assert log[("C", "claim_1")]["outcome"] == "repaired"
    assert log[("D", "claim_2")]["outcome"] == "kept"
    assert log[("E", "fig_missing_one")]["outcome"] == "repaired"
    assert "![bad figure](fig_correct)" in result["revised_md"]
    m = result["metrics"]
    assert m["jev_calls"] == 5          # one per non-empty group
    assert m["llm_calls"] == 1          # only the B rewrite generation
    assert m["jev_rewrites"] == 1


def test_jev_low_confidence_record_falls_back_to_intern(tmp_path):
    script = {"p_ghost": {"action": "delete_claim", "conf": 0.3},   # below gate
              "claim_0": {"action": "rewrite_claim", "conf": 0.3},
              "claim_1": {"action": "backfill_evidence", "conf": 0.3},
              "claim_2": {"action": "keep", "conf": 0.3},
              "fig_missing_one": {"action": "remap_figure", "conf": 0.3,
                                  "param": "fig_correct"}}

    def plan(call_idx, messages):
        content = messages[-1]["content"]
        if "Failure type E" in content and "repair writer" not in messages[0]["content"]:
            return [{"id": "fig_missing_one", "action": "remap_figure",
                     "params": {"new_figure_id": "fig_correct"}, "reason": "same content"}]
        return []

    result = _run(tmp_path, ScriptedLLM(plan), ScriptedNLI(), jev=_decider(script))
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("E", "fig_missing_one")]["outcome"] == "repaired"   # decided by Intern-S2
    assert result["metrics"]["jev_low_confidence"] == 5
    assert result["metrics"]["llm_calls"] == 5       # every group fell back to the batch prompt


def test_jev_no_fitting_candidate_routes_to_intern(tmp_path):
    script = {"claim_0": {"action": "swap_evidence", "param": None},  # param: none_of_the_above
              "claim_1": {"action": "backfill_evidence"},
              "claim_2": {"action": "keep"}}

    def plan(call_idx, messages):
        content = messages[-1]["content"]
        if "Failure type B" in content and "repair writer" not in messages[0]["content"]:
            return [{"id": "claim_0", "action": "swap_evidence",
                     "params": {"evidence_id": "p_good_ev1"}, "reason": "preview row"}]
        return []

    result = _run(tmp_path, ScriptedLLM(plan),
                  ScriptedNLI(script={"latent dynamics": "supported"}), jev=_decider(script))
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("B", "claim_0")]["outcome"] == "repaired"
    assert result["metrics"]["jev_no_candidate"] == 1
    assert result["metrics"]["llm_calls"] == 3  # A + B batch fallback (D settled by jev... A has delete from jev)


def test_jev_api_failure_falls_back_whole_group(tmp_path):
    script = {"claim_0": {"action": "rewrite_claim"},
              "claim_1": {"action": "backfill_evidence"},
              "claim_2": {"action": "keep"}}

    def plan(call_idx, messages):
        content = messages[-1]["content"]
        if "Failure type B" in content and "repair writer" not in messages[0]["content"]:
            return [{"id": "claim_0", "action": "backfill_evidence", "params": {}, "reason": "r"}]
        return []

    jev = JevRepairDecider(FakeJevClient(script, fail_groups=("B",)))
    result = _run(tmp_path, ScriptedLLM(plan), ScriptedNLI(), jev=jev)
    assert result["metrics"]["jev_llm_fallback"] == 1
    # B fell back via API failure; A and E had no scripted answer and fell back
    # through the no-answer path -> three Intern-S2 batches in total
    assert result["metrics"]["llm_calls"] == 3


def test_jev_rewrite_budget_exhausted_marks_unresolved(tmp_path):
    script = {"claim_0": {"action": "rewrite_claim"}}
    claim_map = {"entries": [_claim_map()["entries"][0]]}
    result = _run(tmp_path, ScriptedLLM(lambda i, m: []), ScriptedNLI(),
                  jev=_decider(script), max_llm_calls=0, claim_map=claim_map)
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("B", "claim_0")]["outcome"] == "unresolved"
    assert "rewrite generation" in log[("B", "claim_0")]["reason"]


def test_jev_unset_sentinel_builds_decider_from_env(tmp_path, monkeypatch):
    import tools.clients.jev_client as jev_mod

    # EVISURVEY_JEV=off (autouse fixture) -> no decider, legacy metrics shape
    result = _run(tmp_path, ScriptedLLM(lambda i, m: []), ScriptedNLI(), jev="unset")
    assert "jev_calls" not in result["metrics"]

    monkeypatch.setattr(jev_mod, "make_jev_decider",
                        lambda: _decider({"claim_2": {"action": "keep"}}))
    result = _run(tmp_path, ScriptedLLM(lambda i, m: []), ScriptedNLI(), jev="unset")
    assert result["metrics"]["jev_calls"] >= 1   # env-built decider was used
    log = {(e["failure_type"], e["id"]): e for e in result["repair_log"]}
    assert log[("D", "claim_2")]["outcome"] == "kept"


# --- env switch ------------------------------------------------------------------


def test_make_jev_decider_env_matrix(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("EVISURVEY_JEV", "off")
    assert make_jev_decider() is None
    monkeypatch.setenv("EVISURVEY_JEV", "auto")
    assert make_jev_decider() is None           # no key -> auto stays off
    monkeypatch.setenv("EVISURVEY_JEV", "on")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        make_jev_decider()
    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_test_only")
    decider = make_jev_decider()
    assert isinstance(decider, JevRepairDecider)


def test_action_descriptions_cover_all_schema_actions():
    from harness.agents.repair_agent import ACTION_SCHEMA
    assert set(ACTION_SCHEMA) <= set(ACTION_DESCRIPTIONS)
