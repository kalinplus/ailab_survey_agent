#!/usr/bin/env python3
"""Run the post-hoc survey evaluation over existing run artifacts.

Four layers (specs/评测体系v2-离线四层改造.md):
  L0   deterministic profile (freshness / structure / redundancy, no LLM)
  L1   NLI citation quality on (sentence, citation) pairs
  L1.5 uncited-claim verification via SciVerse agentic-search + NLI
       (--skip-uncited to stay offline; resumable through the claim cache)
  L2'  corpus-grounded coverage over the P5 taxonomy (+ optional gold control)
  L3   DeepSurvey three-dimension academic-value judge
plus the repair-joint A/B table.
Writes output/survey_eval_report.json + output/survey_eval_report.md.

Usage:
  EVISURVEY_REAL_NLI=1 python scripts/run_survey_eval.py --evidence-store cache/eval_evidence_store.json
  python scripts/run_survey_eval.py --skip-judge --skip-uncited   # fully offline
  python scripts/run_survey_eval.py --skip-fuzzy                  # exact title match only
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
    academic_value,
    citation_quality,
    corpus_coverage,
    deterministic_profile,
    overall_unsupported_rate,
    reference_coverage,
    render_md,
    uncited_claims,
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


def _embed_scorer():
    from harness.agents.relevance import EmbeddingScorer
    scorer = EmbeddingScorer()
    _log(f"embedding scorer mode={scorer.mode}")
    return scorer


def _make_sciverse():
    """External-API boundary: constructed only when the uncited layer runs."""
    from tools.clients.sciverse_client import SciVerseClient

    cfg = load_config()
    return SciVerseClient(base_url=cfg.sciverse_api_base_url, api_key=cfg.sciverse_api_token)


def _read_list(path: Path, key: str) -> list:
    if not path.exists():
        _log(f"{key}: {path} not found -> empty")
        return []
    return read_json(path).get(key, [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default="output/survey.md")
    ap.add_argument("--evidence-store", default="cache/evidence_store.json")
    ap.add_argument("--papers", default="cache/final_paper_cards.json",
                    help="system paper set (corpus years / coverage denominator)")
    ap.add_argument("--taxonomy", default="cache/final_taxonomy.json",
                    help="P5 taxonomy categories for corpus-grounded coverage")
    ap.add_argument("--figure-bank", default="cache/final_figure_bank.json")
    ap.add_argument("--table-bank", default="cache/final_table_bank.json")
    ap.add_argument("--gold-refs", default="cache/gold_refs.json")
    ap.add_argument("--ab-report", default="output/repair_ab_report.json")
    ap.add_argument("--uncited-cache", default="output/uncited_claim_cache.json")
    ap.add_argument("--skip-judge", action="store_true", help="skip L3 academic value")
    ap.add_argument("--skip-fuzzy", action="store_true")
    ap.add_argument("--skip-uncited", action="store_true",
                    help="skip L1.5; no SciVerse client is constructed")
    ap.add_argument("--max-uncited-claims", type=int, default=60,
                    help="L1.5 budget: claims verified per run, excess counted only")
    ap.add_argument("--cov-threshold", type=float, default=None,
                    help="L2' coverage threshold (default $EVISURVEY_EVAL_COV_MIN, 0.5)")
    ap.add_argument("--out", default="output/survey_eval_report")
    args = ap.parse_args()

    survey_md = (ROOT / args.survey).read_text(encoding="utf-8")
    evidence_store = EvidenceStore(**read_json(ROOT / args.evidence_store))
    papers_data = read_json(ROOT / args.papers)
    pool = papers_data.get("papers") or papers_data.get("paper_cards") or []
    _log(f"survey={len(survey_md)} chars, evidence={len(evidence_store.evidence)}, "
         f"papers={len(pool)}")

    report: dict = {"task_id": "survey_eval"}

    # Layer 1 — NLI citation quality
    nli, nli_name = _nli_model()
    _log(f"Layer 1 citation quality via {nli_name}")
    cq = citation_quality(survey_md, evidence_store, nli)
    cq["nli_model"] = nli_name
    report["citation_quality"] = cq
    _log(f"Layer 1: recall={cq['citation_recall']} precision={cq['citation_precision']} "
         f"coverage={cq['citation_coverage']} pairs={cq['n_pairs']}")

    # Layer 0 — deterministic profile (freshness / structure / redundancy)
    scorer = _embed_scorer()
    dp = deterministic_profile(
        survey_md, pool,
        figure_bank=_read_list(ROOT / args.figure_bank, "figures"),
        table_bank=_read_list(ROOT / args.table_bank, "tables"),
        scorer=scorer)
    report["deterministic_profile"] = dp
    _log(f"Layer 0: citation_median={dp['freshness']['citation_year_median']} "
         f"corpus_median={dp['freshness']['corpus_year_median']} "
         f"sections={dp['structure']['n_sections']} "
         f"zero_citation={len(dp['structure']['zero_citation_sections'])} "
         f"redundancy_mean={dp['redundancy']['mean_similarity']}")

    # Layer 2' — corpus-grounded coverage over the P5 taxonomy
    taxonomy_categories = read_json(ROOT / args.taxonomy)["categories"]
    cc = corpus_coverage(survey_md, taxonomy_categories, pool, scorer=scorer,
                         cov_threshold=args.cov_threshold)
    report["corpus_coverage"] = cc
    _log(f"Layer 2': coverage={cc['category_coverage_rate']} "
         f"({cc['n_covered']}/{cc['n_categories']} cats, threshold={cc['cov_threshold']}) "
         f"weighted_depth={cc['weighted_citation_depth']} "
         f"utilization={cc['corpus_utilization']}")

    # Layer 2 (gold control) — reference recall against a real survey
    gold_path = ROOT / args.gold_refs
    if gold_path.exists():
        gold = read_json(gold_path)
        titles = [p["title"] for p in pool if p.get("title")]
        rc = reference_coverage(titles, gold, scorer=None if args.skip_fuzzy else scorer)
        rc["fuzzy"] = not args.skip_fuzzy
        report["reference_coverage"] = rc
        _log(f"Layer 2: reference_recall={rc['reference_recall']} "
             f"(exact={rc['exact_matches']} fuzzy={rc['fuzzy_matches']} / {rc['n_gold_refs']})")
    else:
        _log(f"Layer 2 skipped: {gold_path} not found")

    # Layer 1.5 — uncited-claim verification (external retrieval, resumable)
    if args.skip_uncited:
        _log(f"Layer 1.5 skipped (--skip-uncited): no SciVerse client constructed, "
             f"{cq['n_sentences'] - cq['n_cited_sentences']} uncited sentences unaudited")
    else:
        uncited = uncited_claims(
            survey_md, evidence_store, nli, _make_sciverse(),
            cache_path=ROOT / args.uncited_cache, max_claims=args.max_uncited_claims)
        report["uncited_claims"] = uncited
        _log(f"Layer 1.5: total={uncited['n_uncited_total']} evaluated={uncited['n_evaluated']} "
             f"searched={uncited['n_searched']} cache_hits={uncited['n_cache_hits']} "
             f"truncated={uncited['n_truncated']} "
             f"support_rate={uncited['uncited_support_rate']}")

    report["overall_unsupported_rate"] = overall_unsupported_rate(cq, report.get("uncited_claims"))
    _log(f"headline: overall_unsupported_rate={report['overall_unsupported_rate']}")

    # Layer 3 — academic value judge
    if args.skip_judge:
        _log("Layer 3 skipped (--skip-judge)")
    else:
        cfg = load_config()
        from llm_client import InternS2Client
        judge = InternS2Client(cfg)
        _log(f"Layer 3 judging via {cfg.intern_model_name}")
        av = academic_value(survey_md, judge)
        report["academic_value"] = av
        _log(f"Layer 3: overall={av['overall']} over {av['n_calls']} dimensions")

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
        "layers": "L0 deterministic / L1 citation quality / L1.5 uncited claims / "
                  "L2' corpus coverage (+gold control) / L3 academic value",
        "note": "Layer 1 is pair-level (every citation per sentence); evaluator shares "
                "the NLIVerifier infrastructure with the generation-side claim mapper "
                "but runs independently on the final artifact.",
    }

    out_json = ROOT / f"{args.out}.json"
    out_md = ROOT / f"{args.out}.md"
    write_json(out_json, report)
    out_md.write_text(render_md(report), encoding="utf-8")
    _log(f"wrote {out_json} + {out_md}")


if __name__ == "__main__":
    main()
