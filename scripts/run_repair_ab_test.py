#!/usr/bin/env python
"""Three-config comparison for the verify->repair loop (joint 2, spec §4.4).

Configs (same survey + evidence store per topic, pristine snapshot restored
before each config):
  A no-repair    — verify only (exposes the raw failure load)
  B delete-only  — the delivered deletion repair
  C action-repair— joint-2 failure-typed repair (EVISURVEY_REPAIR_AGENT=1)

Each config runs the same outer loop as agent_loop: verify -> Goal Gate ->
(revise -> verify)* <= 2 repair rounds. Real NLI throughout
(EVISURVEY_REAL_NLI=1 is set by this script — spec §3.4: experiments on the
fake keyword NLI are meaningless). Hard assertions fail the script.

Usage:
  python scripts/run_repair_ab_test.py                    # 3 topics x A/B/C
  python scripts/run_repair_ab_test.py --topics hot       # subset
Output: output/repair_ab_report.md + output/repair_ab_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("EVISURVEY_REAL_NLI", "1")  # experiment precondition (spec §3.4)
os.environ.pop("FINAL_SEED_PAPERS", None)          # live bundle, not the seed replay

from config import load_config  # noqa: E402
from harness import goal_gate  # noqa: E402
from harness.citation_prelock import build_citation_ready_set  # noqa: E402
from harness.json_io import read_json, write_json  # noqa: E402
from harness.planner import Planner  # noqa: E402

# spec §4.4: hot (regression) + mid + cold (failure-rich)
TOPICS = [
    ("hot", "world models for games"),
    ("mid", "vector databases for retrieval"),
    ("cold", "neurosymbolic program synthesis"),
]

SNAPSHOT_FILES = [
    "output/survey.md",
    "cache/evidence_store.json",
    "output/repair_log.json",
]


def _log(message: str) -> None:
    print(f"[repair-ab] {message}", flush=True)


# --- pipeline stage (once per topic) ------------------------------------------


def build_pipeline(cfg, topic: str, max_papers: int, max_core: int, use_mineru: bool) -> None:
    """A->B->write_survey: produce a real survey + evidence store for the topic."""
    from tools.knowledge_pipeline_worker import run as run_knowledge
    from tools.write_survey import run as run_write

    planner = Planner(cfg)
    task_id = planner.make_task_id(topic)
    task_request = planner.build_task_request(
        task_id=task_id, topic=topic, language="zh", mode="full",
        max_papers=max_papers, max_core_papers=max_core, use_mineru=use_mineru)
    write_json(ROOT / "cache/task_request.json", task_request)
    strategy = planner.build_search_strategy(
        task_id=task_id, topic=topic, max_papers=max_papers,
        max_core_papers=max_core, mode="full")
    write_json(ROOT / "cache/search_strategy.json", strategy)
    knowledge_request = planner.build_knowledge_build_request(task_request)
    write_json(ROOT / "requests/knowledge_build_request.json", knowledge_request)
    result = run_knowledge("requests/knowledge_build_request.json")
    if result.get("status") not in {"success", "partial_success"}:
        raise RuntimeError(f"knowledge pipeline failed: {result.get('message')}")

    from harness.knowledge_bundle_validator import KnowledgeBundleValidator

    validation = KnowledgeBundleValidator(cfg.root_dir).validate("cache/knowledge_bundle.json")
    citation_ready_set = build_citation_ready_set(
        task_id=task_id,
        paper_cards=validation["artifacts"]["paper_cards"],
        citation_index=validation["artifacts"]["citation_index"],
        evidence_store=validation["artifacts"]["evidence_store"],
        max_core_papers=max_core,
    )
    write_json(ROOT / "cache/citation_ready_set.json", citation_ready_set)
    survey_request = planner.build_survey_generation_request(task_id, topic, "zh")
    write_json(ROOT / "requests/survey_generation_request.json", survey_request)
    result = run_write("requests/survey_generation_request.json")
    if result.get("status") not in {"success", "partial_success"}:
        raise RuntimeError(f"write_survey failed: {result.get('message')}")
    _log(f"pipeline done for {topic!r}: survey "
         f"{len((ROOT / 'output/survey.md').read_text(encoding='utf-8'))} chars")


# --- verify / repair loop (per config) -----------------------------------------


def _snapshot() -> dict[str, bytes | None]:
    snap = {}
    for rel in SNAPSHOT_FILES:
        path = ROOT / rel
        snap[rel] = path.read_bytes() if path.exists() else None
    return snap


def _restore(snap: dict[str, bytes | None]) -> None:
    for rel, data in snap.items():
        path = ROOT / rel
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def _body_chars(survey_text: str) -> int:
    """Char count excluding the appended Revision Notes section (fair retention)."""
    return len(survey_text.split("## Revision Notes")[0])


def run_config(cfg, topic: str, config: str) -> dict:
    """One (topic, config) cell: verify -> gate -> (revise -> verify)*."""
    from tools.revise_survey import run as run_revise
    from tools.verify_citations import run as run_verify

    planner = Planner(cfg)
    task_id = planner.make_task_id(topic)
    verification_request = planner.build_verification_request(task_id)
    write_json(ROOT / "requests/verification_request.json", verification_request)
    revision_request = planner.build_revision_request(task_id, "Evidence verification failed.")
    write_json(ROOT / "requests/revision_request.json", revision_request)

    os.environ["EVISURVEY_REPAIR_AGENT"] = {"A": "0", "B": "0", "C": "1"}[config]

    rounds = 0
    prev_unsupported = None
    metrics: dict = {}
    verify_elapsed = []
    best = None  # (rank_tuple, metrics, survey_text) — best-so-far rollback, as in agent_loop
    import time

    for repair_round in range(goal_gate.MAX_REPAIR_ROUNDS + 1):
        if config == "A" and repair_round > 0:
            break  # no-repair baseline: stop after the first verify
        started = time.monotonic()
        result = run_verify("requests/verification_request.json")
        verify_elapsed.append(round(time.monotonic() - started, 1))
        if result.get("status") not in {"success", "partial_success"}:
            raise RuntimeError(f"verify failed: {result.get('message')}")
        metrics = result["metrics"]
        survey_text = (ROOT / "output/survey.md").read_text(encoding="utf-8")
        rank = (metrics.get("unsupported_claims", 0),
                metrics.get("invalid_citations", 0), -len(survey_text))
        if best is None or rank < best[0]:
            best = (rank, metrics, survey_text)
        gate = goal_gate.evaluate(metrics, repair_round=repair_round,
                                  prev_unsupported=prev_unsupported)
        rounds = repair_round + 1
        _log(f"  {config} round {repair_round}: unsupported={gate.unsupported} "
             f"invalid={gate.invalid_citations} score={gate.citation_validity_score} "
             f"-> {gate.stop_reason}")
        if config == "A" or gate.stop_reason != goal_gate.STOP_REPAIRING:
            break
        result = run_revise("requests/revision_request.json")
        if result.get("status") not in {"success", "partial_success"}:
            raise RuntimeError(f"revise failed: {result.get('message')}")
        revised = ROOT / "output/survey_revised.md"
        if revised.exists():
            (ROOT / "output/survey.md").write_text(
                revised.read_text(encoding="utf-8"), encoding="utf-8")
        prev_unsupported = gate.unsupported

    if best is not None and best[2] is not None:
        # rollback: report (and leave on disk) the best round's survey
        metrics = best[1]
        current = (ROOT / "output/survey.md").read_text(encoding="utf-8")
        if current != best[2]:
            _log(f"  {config}: rolling back to best-so-far round")
            (ROOT / "output/survey.md").write_text(best[2], encoding="utf-8")

    repair_log = []
    log_path = ROOT / "output/repair_log.json"
    if config == "C" and log_path.exists():
        repair_log = read_json(log_path)

    action_counts: dict[str, int] = {}
    for entry in repair_log:
        action_counts[entry.get("action") or "none"] = \
            action_counts.get(entry.get("action") or "none", 0) + 1
    return {
        "config": config, "topic": topic, "rounds": rounds,
        "unsupported": metrics.get("unsupported_claims", 0),
        "invalid_citations": metrics.get("invalid_citations", 0),
        "weak": metrics.get("weak_claims", 0),
        "total_claims": metrics.get("total_claims", 0),
        "citation_validity_score": metrics.get("citation_validity_score", 0.0),
        "survey_chars": len((ROOT / "output/survey.md").read_text(encoding="utf-8")),
        "body_chars": _body_chars((ROOT / "output/survey.md").read_text(encoding="utf-8")),
        "verify_elapsed_sec": verify_elapsed,
        "repair_actions": action_counts,
        "n_repaired": sum(1 for e in repair_log if e.get("outcome") == "repaired"),
        "n_deleted": sum(1 for e in repair_log if e.get("outcome") == "deleted"),
    }


# --- hard assertions (spec §4.4) ------------------------------------------------


def hard_assertions(runs: list[dict], initial_by_topic: dict[str, dict]) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    failures: list[str] = []
    for run in [r for r in runs if r["config"] == "C"]:
        topic = run["topic"]
        initial = initial_by_topic[topic]["unsupported"]
        label = f"C x {topic}"
        ok = run["rounds"] <= goal_gate.MAX_REPAIR_ROUNDS + 1
        lines.append(f"- {label} rounds <= {goal_gate.MAX_REPAIR_ROUNDS + 1}: "
                     f"{'PASS' if ok else 'FAIL'} ({run['rounds']})")
        ok2 = run["unsupported"] <= initial
        lines.append(f"- {label} unsupported(final) <= unsupported(initial {initial}): "
                     f"{'PASS' if ok2 else 'FAIL'} ({run['unsupported']})")
        if not ok:
            failures.append(f"{label}: rounds exceeded")
        if not ok2:
            failures.append(f"{label}: unsupported regressed")
    log_path = ROOT / "output/repair_log.json"
    try:
        entries = read_json(log_path) if log_path.exists() else []
        deletes_without_reason = [e for e in entries
                                  if e.get("outcome") == "deleted" and not e.get("reason")]
        ok = not deletes_without_reason
        lines.append(f"- every delete carries a reason: {'PASS' if ok else 'FAIL'} "
                     f"({len(deletes_without_reason)} missing)")
        if not ok:
            failures.append(f"{len(deletes_without_reason)} deletes without reason")
    except (json.JSONDecodeError, ValueError) as exc:
        lines.append(f"- repair_log parseable: FAIL ({exc})")
        failures.append("repair_log unparseable")
    return lines, failures


def write_report(runs: list[dict], initial_by_topic: dict[str, dict],
                 assertion_lines: list[str], out_md: Path) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# 修复关节 A/B/C 对照报告",
        "",
        f"- 生成时间：{now}",
        "- 真 NLI（cross-encoder/nli-deberta-v3-base）全程开启（spec §3.4 前提）",
        "- A=不修复｜B=删除式修复（交付版）｜C=动作修复（关节 2），每配置前恢复同一份 survey+证据快照",
        "",
        "## 汇总表",
        "",
        "| topic | tier | config | unsupported 残留 | invalid | weak | total claims | 正文字数(去 Notes) | 轮数 | verify 耗时(s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    tier_by_topic = {topic: tier for tier, topic in TOPICS}
    for run in runs:
        lines.append(
            f"| {run['topic']} | {tier_by_topic.get(run['topic'], '?')} | {run['config']} | "
            f"{run['unsupported']} | {run['invalid_citations']} | {run['weak']} | "
            f"{run['total_claims']} | {run['body_chars']} | {run['rounds']} | "
            f"{run['verify_elapsed_sec']} |")
    lines += ["", "## C 配置动作分布（repair_log）", ""]
    for run in [r for r in runs if r["config"] == "C"]:
        lines.append(f"- {run['topic']}: {run['repair_actions']} "
                     f"(repaired={run['n_repaired']}, deleted={run['n_deleted']})")
    lines += ["", "## 硬断言（自动化，全部必须 PASS）", ""]
    lines += assertion_lines
    lines += [
        "",
        "## 方向性结论（人工判读）",
        "",
        "- 核心预期：unsupported 残留 C < B ≤ A；B 的字数显著缩水而 C 基本保留；",
        "- C 的动作分布中 delete 应为少数（设计意图：删除是最后手段）；",
        "- 成本：C 每主题 ≈ 2 轮 × (≤8 次 LLM + 真 NLI 复验) + ≤10 次 agentic 回填。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    _log(f"report written: {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", default="all", help="all | hot | mid | cold")
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--max-papers", type=int, default=8)
    parser.add_argument("--max-core-papers", type=int, default=4)
    parser.add_argument("--no-mineru", action="store_true",
                        help="skip MinerU parsing (faster, emptier evidence store)")
    args = parser.parse_args()

    tiers = ["hot", "mid", "cold"] if args.topics == "all" else [args.topics]
    topics = [(tier, topic) for tier, topic in TOPICS if tier in tiers]
    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]

    cfg = load_config()
    if not cfg.intern_api_key or not cfg.sciverse_api_token:
        print("[repair-ab] INTERN_API_KEY / SCIVERSE_API_KEY required", file=sys.stderr)
        return 2
    for rel in ["cache", "output", "requests", "logs/trajectory"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    use_mineru = not args.no_mineru
    runs: list[dict] = []
    initial_by_topic: dict[str, dict] = {}
    for tier, topic in topics:
        _log(f"=== {tier} topic: {topic} (pipeline, use_mineru={use_mineru}) ===")
        build_pipeline(cfg, topic, args.max_papers, args.max_core_papers, use_mineru)
        pristine = _snapshot()
        for config in configs:
            _restore(pristine)
            _log(f"--- config {config} on {topic} ---")
            run = run_config(cfg, topic, config)
            runs.append(run)
            if config == "A":
                initial_by_topic[topic] = run
        _restore(pristine)  # leave the workspace as the pipeline produced it

    assertion_lines, failures = hard_assertions(runs, initial_by_topic)
    out_md = ROOT / "output" / "repair_ab_report.md"
    write_report(runs, initial_by_topic, assertion_lines, out_md)
    (ROOT / "output" / "repair_ab_report.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print("[repair-ab] HARD ASSERTION FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    _log("all hard assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
