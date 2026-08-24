#!/usr/bin/env python3
"""Run the post-hoc survey evaluation over existing run artifacts.

Layers: (1) NLI citation quality on (sentence, citation) pairs,
(2) gold-reference coverage vs a real published survey,
(3) Intern-S2 LLM-as-judge rubric scoring, plus the repair-joint A/B table.
Writes output/survey_eval_report.json + output/survey_eval_report.md.

Usage:
  EVISURVEY_REAL_NLI=1 python scripts/run_survey_eval.py
  python scripts/run_survey_eval.py --skip-judge            # no LLM calls
  python scripts/run_survey_eval.py --skip-fuzzy            # exact title match only
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from harness.json_io import read_json, write_json  # noqa: E402
from tools.evaluate_survey import (  # noqa: E402
    ab_comparison,
    citation_quality,
    llm_judge,
    reference_coverage,
    render_md,
)
from tools.models.artifacts import EvidenceStore  # noqa: E402
from tools.nlp.nli_verifier import FakeNLIModel, NLIVerifier  # noqa: E402


def _log(message: str) -> None:
    print(f"[eval] {message}", flush=True)


def _nli_model():
    if os.getenv("EVISURVEY_REAL_NLI", "").lower() in {"1", "true", "yes"}:
        model = NLIVerifier()
        return model, f"cross-encoder/nli-deberta-v3-base@{model.model.device}"
    return FakeNLIModel(), "FakeNLIModel(keyword)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default="output/survey.md")
    ap.add_argument("--evidence-store", default="cache/evidence_store.json")
    ap.add_argument("--papers", default="cache/final_paper_cards.json",
                    help="system paper set for Layer 2 (curated final corpus)")
    ap.add_argument("--gold-refs", default="cache/gold_refs.json")
    ap.add_argument("--ab-report", default="output/repair_ab_report.json")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--skip-fuzzy", action="store_true")
    ap.add_argument("--out", default="output/survey_eval_report")
    args = ap.parse_args()

    survey_md = (ROOT / args.survey).read_text(encoding="utf-8")
    evidence_store = EvidenceStore(**read_json(ROOT / args.evidence_store))
    _log(f"survey={len(survey_md)} chars, evidence={len(evidence_store.evidence)}")

    report: dict = {"task_id": "survey_eval"}

    # Layer 1 — NLI citation quality
    nli, nli_name = _nli_model()
    _log(f"Layer 1 citation quality via {nli_name}")
    cq = citation_quality(survey_md, evidence_store, nli)
    cq["nli_model"] = nli_name
    report["citation_quality"] = cq
    _log(f"Layer 1: recall={cq['citation_recall']} precision={cq['citation_precision']} "
         f"coverage={cq['citation_coverage']} pairs={cq['n_pairs']}")

    # Layer 2 — gold reference coverage
    gold_path = ROOT / args.gold_refs
    if gold_path.exists():
        gold = read_json(gold_path)
        papers_data = read_json(ROOT / args.papers)
        pool = papers_data.get("papers") or papers_data.get("paper_cards") or []
        titles = [p["title"] for p in pool if p.get("title")]
        scorer = None
        if not args.skip_fuzzy:
            from harness.agents.relevance import EmbeddingScorer
            scorer = EmbeddingScorer()
            _log(f"Layer 2 fuzzy scorer mode={scorer.mode}")
        rc = reference_coverage(titles, gold, scorer=scorer)
        rc["fuzzy"] = not args.skip_fuzzy
        report["reference_coverage"] = rc
        _log(f"Layer 2: reference_recall={rc['reference_recall']} "
             f"(exact={rc['exact_matches']} fuzzy={rc['fuzzy_matches']} / {rc['n_gold_refs']})")
    else:
        _log(f"Layer 2 skipped: {gold_path} not found")

    # Layer 3 — LLM-as-judge
    if args.skip_judge:
        _log("Layer 3 skipped (--skip-judge)")
    else:
        cfg = load_config()
        from llm_client import InternS2Client
        judge = InternS2Client(cfg)
        _log(f"Layer 3 judging via {cfg.intern_model_name}")
        lj = llm_judge(survey_md, judge)
        report["llm_judge"] = lj
        _log(f"Layer 3: overall={lj['overall']} over {lj['n_sections']} sections")

    # A/B comparison from the repair joint report
    ab_path = ROOT / args.ab_report
    if ab_path.exists():
        report["ab_comparison"] = ab_comparison(read_json(ab_path))
        _log(f"A/B table: {len(report['ab_comparison'])} rows")

    report["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "survey": args.survey,
        "nli_model": nli_name,
        "judge_model": None if args.skip_judge else load_config().intern_model_name,
        "note": "Layer 1 is pair-level (every citation per sentence); evaluator shares "
                "the NLIVerifier infrastructure with the generation-side claim mapper "
                "but runs independently on the final artifact.",
    }

    out_json = ROOT / f"{args.out}.json"
    out_md = ROOT / f"{args.out}.md"
    write_json(out_json, report)
    out_md.write_text(render_md(report), encoding="utf-8")
    _log(f"wrote {out_json.relative_to(ROOT)} + {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
