#!/usr/bin/env python
"""A/B/C comparison for the retrieval-strategy joint (spec §4.4).

Configs:
  A template    — demo/template path (baseline, zero LLM)
  B single-shot — the delivered full-mode one-shot LLM strategy (no loop)
  C agent-loop  — the joint-1 strategy agent (probe -> name -> edit loop)

Each generated strategy is evaluated with the SAME sampler (one meta-search
per aspect, P3 year filter) and report card. Hard assertions (budget caps,
rollback guarantee, trajectory parseability) fail the script when violated.

Usage:
  python scripts/run_strategy_ab_test.py                     # full matrix
  python scripts/run_strategy_ab_test.py --topics hot --limit 1   # quick smoke
Output: output/strategy_ab_report.md + output/strategy_ab_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from llm_client import InternS2Client, _extract_json_object  # noqa: E402
from harness.agents.relevance import EmbeddingScorer, build_report_card, global_score  # noqa: E402
from harness.agents.strategy_agent import (  # noqa: E402
    _CountingSciverse,
    _sample,
    run_strategy_agent,
)
from harness import search_strategy_builder  # noqa: E402
from harness.search_strategy_builder import build_search_strategy  # noqa: E402
from tools.clients.sciverse_client import SciVerseClient  # noqa: E402

# De-bias the baselines (2026-08-20 decision): the demo-topic priors (game/world
# templates, GameCraft prompt hints, extra probe queries) would tilt A/B on the
# hackathon topic. C's agent path is already topic-neutral (live probe -> cluster
# -> naming with a generic prompt, memory isolated). Delivery path untouched.
search_strategy_builder.TOPIC_NEUTRAL = True

# Fixed topic set (spec §4.4): hot x2 (template should be enough — checks
# no-regression), mid x1, cold x2 (expected template collapse).
TOPICS = [
    ("hot", "world models for games"),
    ("hot", "embodied intelligence agents"),
    ("mid", "vector databases for retrieval"),
    ("cold", "neurosymbolic program synthesis"),
    ("cold", "morphological computation in soft robotics"),
]

MAX_LLM = 7
MAX_SCIVERSE = 20
MAX_ROUNDS = 3


class CountingJsonChat:
    """llm_json_chat wrapper mirroring Planner._strategy_json_chat + call counting."""

    def __init__(self, client: InternS2Client) -> None:
        self.client = client
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        try:
            return self.client.json_chat(messages, temperature=0.1, max_tokens=4000)
        except Exception:
            content = self.client.chat(messages, temperature=0.1, max_tokens=4000)
            extracted = _extract_json_object(content)
            if extracted is None:
                raise ValueError("model reply contained no JSON object")
            return extracted


def _slug(topic: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in topic.lower())[:40]


def _metrics_from_card(card: dict) -> dict:
    rows = card["aspects"]
    n = len(rows) or 1
    return {
        "n_aspects": len(rows),
        "survival_rate": round(sum(r["n_unique"] >= 3 for r in rows) / n, 3),
        "zero_rate": round(sum(r["n_unique"] == 0 for r in rows) / n, 3),
        "mean_rel": round(sum(r["rel"] for r in rows) / n, 3),
        "mean_overlap": round(sum(r["overlap"] for r in rows) / n, 3),
        "global_score": card["global_score"],
    }


def _evaluate_strategy(strategy: dict, sv, scorer, end_year: int) -> tuple[dict, int]:
    """Sample every aspect once and build the card — the shared evaluation probe."""
    budget = _CountingSciverse(sv, 50)
    aspects = strategy["wide_search"]["search_aspects"]
    per_hits = _sample(budget, aspects, end_year)
    card = build_report_card(aspects, per_hits, scorer)
    return card, budget.calls


def _best_card_from_trajectory(path: Path) -> tuple[dict, list[dict]]:
    """Replay report_card events applying the same best-so-far rule as the joint."""
    events = [json.loads(line) for line in path.read_text().splitlines()]
    cards = [e for e in events if e.get("event") == "report_card"]
    best = cards[0]
    for card in cards[1:]:
        if card["score"] >= best["score"]:
            best = card
    return best["card"], events


def run_one(config: str, tier: str, topic: str, *, sv, llm, scorer, traj_dir: Path,
            end_year: int) -> dict:
    task_id = f"task_ab_{config}_{_slug(topic)}"
    started = time.monotonic()
    if config == "A":
        strategy = build_search_strategy(
            task_id=task_id, topic=topic, max_papers=20, max_core_papers=8,
            end_year=end_year, use_probing=False, sciverse_api_key="")
        llm_calls, sv_gen, rounds = 0, 0, 1
        card, eval_calls = _evaluate_strategy(strategy, sv, scorer, end_year)
    elif config == "B":
        counting = CountingJsonChat(llm)
        strategy = build_search_strategy(
            task_id=task_id, topic=topic, max_papers=20, max_core_papers=8,
            end_year=end_year, use_probing=False, sciverse_api_key="",
            llm_json_chat=counting)
        llm_calls, sv_gen, rounds = counting.calls, 0, 1
        card, eval_calls = _evaluate_strategy(strategy, sv, scorer, end_year)
    elif config == "C":
        counting = CountingJsonChat(llm)
        traj_file = traj_dir / f"{task_id}_strategy.jsonl"
        traj_file.unlink(missing_ok=True)  # trajectory appends; drop stale events from reruns
        strategy = run_strategy_agent(
            task_id=task_id, topic=topic, max_papers=20, max_core_papers=8,
            end_year=end_year, llm_json_chat=counting, sciverse=sv,
            scorer=scorer, trajectory_dir=traj_dir, memory_context="")
        meta = strategy["strategy_agent"]
        llm_calls, sv_gen, rounds = meta["llm_calls"], meta["sciverse_calls"], meta["rounds"]
        card, _ = _best_card_from_trajectory(traj_file)
        eval_calls = 0  # the agent's own cards already used generation calls
    else:
        raise ValueError(f"unknown config {config!r}")
    elapsed = round(time.monotonic() - started, 1)
    return {
        "config": config, "tier": tier, "topic": topic, "task_id": task_id,
        "llm_calls": llm_calls, "sciverse_gen_calls": sv_gen,
        "eval_sample_calls": eval_calls, "rounds": rounds, "elapsed_sec": elapsed,
        "rel_mode": card["rel_mode"],
        **_metrics_from_card(card),
        "_strategy": strategy,
    }


def hard_assertions(results: list[dict], traj_dir: Path) -> tuple[list[str], list[str]]:
    """Spec §4.4 hard assertions — must hold on every C run; failure = script failure."""
    lines: list[str] = []
    failures: list[str] = []
    c_runs = [r for r in results if r["config"] == "C"]
    if c_runs:
        for label, key, cap in (("LLM", "llm_calls", MAX_LLM),
                                ("SciVerse", "sciverse_gen_calls", MAX_SCIVERSE),
                                ("rounds", "rounds", MAX_ROUNDS)):
            worst = max(r[key] for r in c_runs)
            ok = worst <= cap
            lines.append(f"- C runs {label} <= {cap}: {'PASS' if ok else 'FAIL'} (max={worst})")
            if not ok:
                failures.append(f"C runs exceed {label} cap: {worst} > {cap}")
    for path in sorted(traj_dir.glob("*.jsonl")):
        try:
            events = [json.loads(line) for line in path.read_text().splitlines()]
            lines.append(f"- trajectory parseable {path.name}: PASS")
        except json.JSONDecodeError as exc:
            lines.append(f"- trajectory parseable {path.name}: FAIL ({exc})")
            failures.append(f"unparseable trajectory {path}")
            continue
        # rollback guarantee: finish.final_score >= first report_card score
        first = next(e["score"] for e in events if e.get("event") == "report_card")
        final = next(e["final_score"] for e in events if e.get("event") == "finish")
        passed = final >= first
        lines.append(f"- rollback guarantee {path.stem}: "
                     f"{'PASS' if passed else 'FAIL'} (final={final} >= round1={first})")
        if not passed:
            failures.append(f"rollback violated for {path.stem}")
    return lines, failures


def write_report(results: list[dict], assertion_lines: list[str], out_md: Path,
                 rel_mode: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# 策略关节 A/B/C 对照报告",
        "",
        f"- 生成时间：{now}",
        f"- 相关度模式：{rel_mode}（embedding=bge 余弦；keyword=降级关键词列）",
        "- A=模板（零 LLM）｜B=单发 LLM｜C=策略 Agent 循环",
        "- A/B 的 `eval_sample_calls` 是评测采样（每方向 1 次 meta-search），非生成成本",
        "",
        "## 汇总表",
        "",
        "| topic | tier | config | 方向成活率 | 零方向率 | 平均相关度 | 方向重叠度 | global_score | LLM | SciVerse(生成) | 轮数 | 墙钟(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['topic']} | {r['tier']} | {r['config']} | "
            f"{r['survival_rate']} | {r['zero_rate']} | {r['mean_rel']} | "
            f"{r['mean_overlap']} | {r['global_score']} | {r['llm_calls']} | "
            f"{r['sciverse_gen_calls']} | {r['rounds']} | {r['elapsed_sec']} |")
    lines += ["", "## 硬断言（自动化，全部必须 PASS）", ""]
    lines += assertion_lines
    lines += [
        "",
        "## 方向性结论（人工判读）",
        "",
        "- 冷主题：C 的方向成活率应显著高于 A（模板塌方），且不低于 B；",
        "- 热主题：C 不劣于 B（回滚保证下界 = 单发水平）；",
        "- 代价：C 每主题约 +3~6 次 LLM 调用（Intern-S2 ~2s/次）与 ≤20 次 SciVerse 调用。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ab-test] report written: {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="A,B,C", help="comma list of A/B/C")
    parser.add_argument("--topics", default="all", help="all | hot | mid | cold")
    parser.add_argument("--limit", type=int, default=0, help="cap topics per tier (smoke)")
    args = parser.parse_args()

    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    tiers = ["hot", "mid", "cold"] if args.topics == "all" else [args.topics]
    topics = [(tier, topic) for tier, topic in TOPICS if tier in tiers]
    if args.limit:
        counts: dict[str, int] = {}
        capped = []
        for tier, topic in topics:
            counts[tier] = counts.get(tier, 0) + 1
            if counts[tier] <= args.limit:
                capped.append((tier, topic))
        topics = capped

    config = load_config()
    if not config.sciverse_api_token:
        print("[ab-test] SCIVERSE_API_KEY missing; cannot run", file=sys.stderr)
        return 2
    llm = None
    if {"B", "C"} & set(configs):
        llm = InternS2Client(config)
        if not llm.is_configured():
            print("[ab-test] INTERN_API_KEY missing; B/C need it", file=sys.stderr)
            return 2

    sv = SciVerseClient(base_url=config.sciverse_api_base_url, api_key=config.sciverse_api_token)
    scorer = EmbeddingScorer()  # bge; degrades to keyword column if download fails
    traj_dir = ROOT / "output" / "strategy_ab" / "traj"
    end_year = datetime.now().year

    print(f"[ab-test] configs={configs} topics={[t for _, t in topics]} rel={scorer.mode}")
    results = []
    for tier, topic in topics:
        for cfg in configs:
            print(f"[ab-test] running {cfg} x {topic} ...", flush=True)
            result = run_one(cfg, tier, topic, sv=sv, llm=llm, scorer=scorer,
                             traj_dir=traj_dir, end_year=end_year)
            results.append(result)
            print(f"[ab-test]   survival={result['survival_rate']} zero={result['zero_rate']} "
                  f"rel={result['mean_rel']} llm={result['llm_calls']} "
                  f"sv={result['sciverse_gen_calls']} rounds={result['rounds']} "
                  f"{result['elapsed_sec']}s", flush=True)

    assertion_lines, failures = hard_assertions(results, traj_dir)
    out_md = ROOT / "output" / "strategy_ab_report.md"
    write_report(results, assertion_lines, out_md, scorer.mode)
    payload = [{k: v for k, v in r.items() if k != "_strategy"} for r in results]
    (ROOT / "output" / "strategy_ab_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        print("[ab-test] HARD ASSERTION FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("[ab-test] all hard assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
