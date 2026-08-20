"""Unit tests for the Goal Gate + verify->repair loop wiring (joint 2 slice 1).

Spec §4.1 test_goal_gate: pass/fail semantics, budget/fixpoint stops,
improved(), and the regression that the SECOND verify result is consumed
(old agent_loop discarded it).
"""

from pathlib import Path

from harness import goal_gate
from harness.agent_loop import AgentLoop, _repair_rank


# --- pure gate semantics ------------------------------------------------------


def _metrics(unsupported=0, invalid=0, score=1.0):
    return {"unsupported_claims": unsupported, "invalid_citations": invalid,
            "citation_validity_score": score}


def test_all_pass():
    gate = goal_gate.evaluate(_metrics(), repair_round=0)
    assert gate.passed and gate.stop_reason == goal_gate.STOP_PASSED


def test_structural_fail_not_passed():
    gate = goal_gate.evaluate(_metrics(invalid=2), repair_round=0)
    assert not gate.passed
    assert gate.stop_reason == goal_gate.STOP_REPAIRING  # budget left -> repair
    assert gate.invalid_citations == 2


def test_unsupported_with_budget_left_continues():
    gate = goal_gate.evaluate(_metrics(unsupported=5), repair_round=0)
    assert not gate.passed and gate.stop_reason == goal_gate.STOP_REPAIRING


def test_budget_exhausted_stop():
    gate = goal_gate.evaluate(_metrics(unsupported=5), repair_round=goal_gate.MAX_REPAIR_ROUNDS)
    assert not gate.passed and gate.stop_reason == goal_gate.STOP_BUDGET


def test_fixpoint_stop_when_no_improvement():
    gate = goal_gate.evaluate(_metrics(unsupported=5), repair_round=1, prev_unsupported=5)
    assert not gate.passed and gate.stop_reason == goal_gate.STOP_FIXPOINT
    # worse is also no improvement
    gate = goal_gate.evaluate(_metrics(unsupported=6), repair_round=1, prev_unsupported=5)
    assert gate.stop_reason == goal_gate.STOP_FIXPOINT


def test_improved_strictly_decreasing():
    assert goal_gate.improved(3, 5)
    assert not goal_gate.improved(5, 5)
    assert not goal_gate.improved(6, 5)


def test_repair_rank_prefers_supported_then_longer_text():
    better = _repair_rank(_metrics(unsupported=0), "x" * 100)
    worse = _repair_rank(_metrics(unsupported=0), "x" * 50)
    assert better < worse  # tie on failures -> longer text wins
    assert _repair_rank(_metrics(unsupported=1), "x" * 999) > _repair_rank(_metrics(unsupported=0), "x")


# --- coverage gate (joint 3, spec §4.1 evaluate layer) -------------------------


def test_coverage_all_three_gates_pass():
    gate = goal_gate.evaluate(
        _metrics(), repair_round=1, text_retention=0.9, figure_ref_retention=0.8)
    assert gate.passed and gate.stop_reason == goal_gate.STOP_PASSED


def test_coverage_text_retention_fail_blocks_pass():
    # joint-2 B-config pathology: deletion clears unsupported but hollows the
    # survey -> must NOT count as passed
    gate = goal_gate.evaluate(
        _metrics(), repair_round=1, text_retention=0.5, figure_ref_retention=1.0)
    assert not gate.passed and gate.stop_reason == goal_gate.STOP_COVERAGE
    assert gate.coverage == {
        "text_retention": 0.5, "figure_ref_retention": 1.0,
        "min_retention": goal_gate.MIN_RETENTION_DEFAULT, "ok": False,
    }


def test_coverage_figure_ref_retention_fail():
    gate = goal_gate.evaluate(
        _metrics(), repair_round=1, text_retention=1.0, figure_ref_retention=0.2)
    assert not gate.passed and gate.stop_reason == goal_gate.STOP_COVERAGE


def test_coverage_fail_priority_over_budget_and_fixpoint():
    gate = goal_gate.evaluate(
        _metrics(unsupported=5), repair_round=goal_gate.MAX_REPAIR_ROUNDS,
        text_retention=0.5)
    assert gate.stop_reason == goal_gate.STOP_COVERAGE
    gate = goal_gate.evaluate(
        _metrics(unsupported=5), repair_round=1, prev_unsupported=5,
        text_retention=0.5)
    assert gate.stop_reason == goal_gate.STOP_COVERAGE


def test_coverage_ok_with_unsupported_repairs_on():
    gate = goal_gate.evaluate(
        _metrics(unsupported=5), repair_round=0, text_retention=0.8)
    assert not gate.passed and gate.stop_reason == goal_gate.STOP_REPAIRING


def test_coverage_default_params_keep_old_behavior():
    gate = goal_gate.evaluate(_metrics(), repair_round=0)
    assert gate.passed and gate.coverage["ok"] is True


def test_figure_ref_count():
    md = "text ![fig a](figs/a.png) middle\n![b](b.png) tail ![c](c.png)"
    assert goal_gate.figure_ref_count(md) == 3
    assert goal_gate.figure_ref_count("no images [p1] here") == 0


# --- agent_loop wiring --------------------------------------------------------


class _FakePlanner:
    def __init__(self):
        self.revision_requests = 0

    def build_revision_request(self, task_id, reason, **kw):
        self.revision_requests += 1
        return {"task_id": task_id, "reason": reason, "inputs": {}, "outputs": {}}


class _FakeLogger:
    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append((event, fields))


class _ScriptedRegistry:
    """verify returns scripted metrics; revise appends a marker to survey_revised.md."""

    def __init__(self, verify_metrics_sequence, revise_texts):
        self.verify_metrics = list(verify_metrics_sequence)
        self.revise_texts = list(revise_texts)
        self.verify_calls = 0
        self.revise_calls = 0

    def run(self, tool_name, request_path):
        if tool_name == "verify_citations":
            self.verify_calls += 1
            return {"status": "success", "metrics": self.verify_metrics.pop(0)}
        if tool_name == "revise_survey":
            self.revise_calls += 1
            text = self.revise_texts.pop(0) if self.revise_texts else "revised"
            Path(self.output_dir, "survey_revised.md").write_text(text, encoding="utf-8")
            return {"status": "success"}
        raise AssertionError(f"unexpected tool {tool_name}")


def _make_loop(tmp_path, registry):
    from config import AppConfig

    fields = {
        "root_dir": tmp_path,
        "intern_api_base_url": "http://localhost",
        "intern_api_key": "",
        "intern_model_name": "intern-s2-preview",
        "sciverse_api_token": "",
        "sciverse_api_base_url": "https://api.sciverse.space",
        "strategy_probing_enabled": False,
        "strategy_probe_limit": 20,
        "strategy_cluster_count": 4,
        "strategy_memory_enabled": False,
        "strategy_memory_max_chars": 4000,
        "request_timeout_seconds": 30.0,
        "max_llm_retries": 1,
        "tool_timeout_seconds": 60.0,
        "tool_result_max_chars": 6000,
        "default_language": "zh",
        "default_mode": "demo",
    }
    logger = _FakeLogger()
    loop = AgentLoop(
        config=AppConfig(**fields),
        planner=_FakePlanner(),
        tool_registry=registry,
        state_manager=object(),  # not touched by _verify_repair_loop
        logger=logger,
    )
    return loop, logger


def test_loop_second_verify_result_is_consumed(tmp_path):
    # regression for the old half-loop: round1 fails, revise, round2 passes
    # -> the loop must exit on the SECOND verify's result (texts sized so the
    # joint-3 coverage gate stays above threshold — not this test's concern)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text(
        "draft with one bad claim [p1] plus enough surrounding context", encoding="utf-8")
    fixed_text = "fixed draft [p1] keeping all the surrounding context intact"
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=3), _metrics(unsupported=0)],
        revise_texts=[fixed_text],
    )
    registry.output_dir = output_dir
    loop, logger = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert gate.passed and gate.stop_reason == goal_gate.STOP_PASSED
    assert registry.verify_calls == 2          # second verify really ran
    assert registry.revise_calls == 1
    assert (output_dir / "survey.md").read_text(encoding="utf-8") == fixed_text
    gate_events = [f for name, f in logger.events if name == "goal_gate"]
    assert len(gate_events) == 2               # one per verify round
    assert gate_events[-1]["passed"] is True


def test_loop_fixpoint_stops_before_third_verify(tmp_path):
    # round1: 5 unsupported -> revise; round2: still 5 -> fixpoint stop,
    # no round3 even though budget (2 rounds) is not exhausted
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("draft [p1]", encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=5), _metrics(unsupported=5)],
        revise_texts=["still bad [p1]"],
    )
    registry.output_dir = output_dir
    loop, _ = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert not gate.passed and gate.stop_reason == goal_gate.STOP_FIXPOINT
    assert registry.verify_calls == 2 and registry.revise_calls == 1


def test_loop_budget_exhausted_after_two_repairs(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("draft [p1]", encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=5), _metrics(unsupported=3),
                                 _metrics(unsupported=3)],
        revise_texts=["r1 [p1]", "r2 [p1]"],
    )
    registry.output_dir = output_dir
    loop, logger = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert not gate.passed and gate.stop_reason == goal_gate.STOP_BUDGET
    assert registry.verify_calls == 3 and registry.revise_calls == 2


def test_loop_rolls_back_worse_round(tmp_path):
    # round1: 3 unsupported, 100 chars; round2 (after a light-trim revise):
    # 2 unsupported, 80 chars. Rank prefers fewer unsupported, so round2
    # text is kept. Then a round3 with SAME failures but shorter text loses ->
    # rollback to the round2 text. (Both trims stay above the joint-3
    # coverage threshold so this test stays about rollback, not coverage.)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("x" * 100, encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[
            _metrics(unsupported=3), _metrics(unsupported=2), _metrics(unsupported=2),
        ],
        revise_texts=["y" * 80, "z" * 75],
    )
    registry.output_dir = output_dir
    loop, logger = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert gate.stop_reason == goal_gate.STOP_BUDGET
    # round2's 80-char text beats round3's 10-char text at equal failures
    assert (output_dir / "survey.md").read_text(encoding="utf-8") == "y" * 80
    assert any(name == "repair_rollback" for name, _ in logger.events)


def test_loop_first_pass_skips_revise_entirely(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("clean draft [p1]", encoding="utf-8")
    registry = _ScriptedRegistry(verify_metrics_sequence=[_metrics()], revise_texts=[])
    registry.output_dir = output_dir
    loop, _ = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert gate.passed
    assert registry.verify_calls == 1 and registry.revise_calls == 0


# --- coverage gate wiring (joint 3, spec §4.1 loop layer) ----------------------


def test_loop_coverage_fail_stops_and_rolls_back(tmp_path):
    # deletion-happy revise shrinks the survey to 30% of the round-0 baseline
    # with no unsupported improvement -> coverage_fail stop + rollback to the
    # longer round-0 text (spec §4.1 ⑨)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("x" * 100, encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=5), _metrics(unsupported=5)],
        revise_texts=["y" * 30],
    )
    registry.output_dir = output_dir
    loop, logger = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert not gate.passed and gate.stop_reason == goal_gate.STOP_COVERAGE
    assert gate.coverage["text_retention"] == 0.3
    assert registry.verify_calls == 2 and registry.revise_calls == 1
    assert (output_dir / "survey.md").read_text(encoding="utf-8") == "x" * 100
    assert any(name == "repair_rollback" for name, _ in logger.events)
    goal_gate_events = [f for name, f in logger.events if name == "goal_gate"]
    assert goal_gate_events[-1]["stop_reason"] == goal_gate.STOP_COVERAGE


def test_loop_coverage_fail_annotates_when_hollowed_text_is_rank_best(tmp_path):
    # hollowed round-1 text has FEWER unsupported -> _repair_rank keeps it as
    # best (spec §5: rank stays unchanged); the gate still annotates honestly
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("x" * 100, encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=3), _metrics()],
        revise_texts=["y" * 30],
    )
    registry.output_dir = output_dir
    loop, _ = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert not gate.passed and gate.stop_reason == goal_gate.STOP_COVERAGE
    assert (output_dir / "survey.md").read_text(encoding="utf-8") == "y" * 30


def test_loop_coverage_threshold_from_env(tmp_path, monkeypatch):
    # 80% retention passes the default 0.7 threshold but fails an env-raised one
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("x" * 100, encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=5), _metrics(unsupported=5)],
        revise_texts=["y" * 80],
    )
    registry.output_dir = output_dir
    loop, _ = _make_loop(tmp_path, registry)
    monkeypatch.setenv("EVISURVEY_COVERAGE_MIN", "0.9")

    gate = loop._verify_repair_loop("task_x")

    assert gate.stop_reason == goal_gate.STOP_COVERAGE
    assert gate.coverage["min_retention"] == 0.9


def test_loop_figure_ref_drop_triggers_coverage_fail(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    baseline = "a ![f1](figs/1.png)\n" + "x" * 90 + "\nb ![f2](figs/2.png)"
    shrunk = "a ![f1](figs/1.png)\n" + "x" * 80  # same length band, one ref gone
    (output_dir / "survey.md").write_text(baseline, encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=5), _metrics(unsupported=5)],
        revise_texts=[shrunk],
    )
    registry.output_dir = output_dir
    loop, _ = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert gate.stop_reason == goal_gate.STOP_COVERAGE
    assert gate.coverage["figure_ref_retention"] == 0.5
    assert gate.coverage["text_retention"] > goal_gate.MIN_RETENTION_DEFAULT


def test_loop_zero_baseline_figure_refs_guarded(tmp_path):
    # no figure refs in the baseline -> figure ratio is 1.0, never a false fail
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "survey.md").write_text("x" * 100, encoding="utf-8")
    registry = _ScriptedRegistry(
        verify_metrics_sequence=[_metrics(unsupported=5), _metrics(unsupported=5)],
        revise_texts=["y" * 80],
    )
    registry.output_dir = output_dir
    loop, _ = _make_loop(tmp_path, registry)

    gate = loop._verify_repair_loop("task_x")

    assert gate.coverage["figure_ref_retention"] == 1.0
    assert gate.stop_reason == goal_gate.STOP_FIXPOINT  # old semantics intact
