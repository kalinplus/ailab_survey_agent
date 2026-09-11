#!/usr/bin/env python3
"""Canonical survival-chain audit (specs/canonical存活链诊断.md).

For every hand-curated landmark in cache/canonical_papers.json, locate the first
pipeline stage it died at:

  search_miss -> prefilter_drop -> rank_cut -> corpus_cut -> no_card ->
  whitelist_cut -> cited

Inputs: cache/p3_stage_trace.json (written by phase3_paper_retriever, one run),
cache/paper_cards.json, cache/citation_ready_set.json, output/survey.md.
Alias/title matching copies the normalization of evaluate_survey.canonical_hits
(exact equality on normalized labels, colon-split title parts included) so both
tools judge the same fixture identically. Missing input files fail fast.

Usage:
  python scripts/audit_canonical_survival.py
  python scripts/audit_canonical_survival.py --trace cache/p3_stage_trace.json --out cache/canonical_survival.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Chain order used for the summary and the output file.
STAGES = [
    "search_miss",
    "prefilter_drop",
    "rank_cut",
    "corpus_cut",
    "no_card",
    "whitelist_cut",
    "cited",
]

# Copied verbatim from tools/evaluate_survey.py (_normalize_title + _TITLE_NOISE):
# alias matching must stay judgment-identical to canonical_hits, reference only.
_TITLE_NOISE = re.compile(r"[^a-z0-9一-鿿]+")
_CITATION = re.compile(r"\[([^\]]+)\]")


def _normalize_title(title: str) -> str:
    return _TITLE_NOISE.sub("", title.lower())


def _title_labels(title: str) -> set[str]:
    """Normalized labels for a corpus-side title: the full title plus each
    colon-separated part ("DreamerV3: Mastering ..." carries the alias head)."""
    title = str(title or "")
    labels = {_normalize_title(part) for part in (title, *title.split(":"))}
    labels.discard("")
    return labels


def _canonical_labels(entry: dict) -> list[str]:
    labels = [_normalize_title(entry[key]) for key in ("title", "alias") if entry.get(key)]
    return [label for label in labels if label]


def _find_paper(labels: list[str], papers: list[dict]) -> dict | None:
    for paper in papers:
        if any(label in _title_labels(paper.get("title") or "") for label in labels):
            return paper
    return None


def _whitelist_item(labels: list[str], card: dict | None, whitelist: dict) -> dict | None:
    item = _find_paper(labels, whitelist.get("items", []))
    if item is not None:
        return item
    # Titles can drift between card and whitelist entry; fall back to the id the
    # whitelist actually allows.
    if card and card.get("paper_id") in set(whitelist.get("allowed_paper_ids", [])):
        return {"paper_id": card["paper_id"], "title": card.get("title", "")}
    return None


def classify(
    entry: dict,
    stages: dict,
    cards: list[dict],
    whitelist: dict,
    cited_ids: set[str],
) -> dict:
    label = entry.get("alias") or entry.get("title") or "?"
    labels = _canonical_labels(entry)
    if not labels:
        return {"label": label, "stage": "search_miss", "detail": "canonical entry has no title/alias"}

    raw_hit = _find_paper(labels, stages.get("raw_hits", []))
    if raw_hit is None:
        return {"label": label, "stage": "search_miss",
                "detail": "no raw meta-search hit matched title/alias"}
    prefiltered = _find_paper(labels, stages.get("after_prefilter", []))
    if prefiltered is None:
        return {"label": label, "stage": "prefilter_drop",
                "detail": f"raw hit {raw_hit['paper_id']} dropped by the relevance prefilter"}
    ranked = _find_paper(labels, stages.get("after_rank_cut", []))
    if ranked is None:
        return {"label": label, "stage": "rank_cut",
                "detail": f"{prefiltered['paper_id']} survived the prefilter but was cut "
                          f"by the rank-ordered corpus slice (cap_mode=top-slice)"}
    final = _find_paper(labels, stages.get("after_corpus_cap", []))
    if final is None:
        return {"label": label, "stage": "corpus_cut",
                "detail": f"{ranked['paper_id']} was in the ranked corpus but was trimmed "
                          f"by the aspect-balanced corpus cap"}
    card = _find_paper(labels, cards)
    if card is None:
        return {"label": label, "stage": "no_card",
                "detail": f"{final!r} reached the final corpus but no paper card was built"}
    item = _whitelist_item(labels, card, whitelist)
    if item is None:
        return {"label": label, "stage": "whitelist_cut",
                "detail": f"card {card.get('paper_id')!r} never entered citation_ready_set"}
    ids = {final.get("paper_id"), card.get("paper_id"), item.get("paper_id")} - {None}
    cited = ids & cited_ids
    if cited:
        return {"label": label, "stage": "cited",
                "detail": f"cited in survey.md as {sorted(cited)}"}
    return {"label": label, "stage": "whitelist_cut",
            "detail": f"in citation_ready_set ({item.get('paper_id')}) but never cited "
                      f"in survey.md"}


def run_audit(
    canonical_path: str,
    trace_path: str,
    cards_path: str,
    whitelist_path: str,
    survey_path: str,
    out_path: str,
) -> list[dict]:
    inputs = [
        ("canonical papers", canonical_path),
        ("p3 stage trace", trace_path),
        ("paper cards", cards_path),
        ("citation whitelist", whitelist_path),
        ("survey markdown", survey_path),
    ]
    missing = [f"{name} ({path})" for name, path in inputs if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"missing canonical-survival input(s): {', '.join(missing)}")

    def _load(path: str):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    canonical = _load(canonical_path)
    trace = _load(trace_path)
    cards = _load(cards_path).get("paper_cards", [])
    whitelist = _load(whitelist_path)
    with open(survey_path, encoding="utf-8") as f:
        cited_ids = set(_CITATION.findall(f.read()))

    stages = trace.get("stages")
    if not stages:
        raise ValueError(f"{trace_path} has no 'stages' — rerun P3 to regenerate the trace")

    results = [
        classify(entry, stages, cards, whitelist, cited_ids)
        for entry in canonical.get("papers", [])
    ]
    summary = {stage: 0 for stage in STAGES}
    for row in results:
        summary[row["stage"]] += 1

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"task_id": trace.get("task_id"), "cap_mode": trace.get("cap_mode"),
             "summary": summary, "results": results},
            f, ensure_ascii=False, indent=2,
        )
        f.write("\n")

    print(f"canonical survival ({len(results)} papers, cap_mode={trace.get('cap_mode')})")
    for stage in STAGES:
        print(f"  {stage:<15} {summary[stage]}")
    print(f"-> {out_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", default="cache/canonical_papers.json")
    ap.add_argument("--trace", default="cache/p3_stage_trace.json")
    ap.add_argument("--cards", default="cache/paper_cards.json")
    ap.add_argument("--whitelist", default="cache/citation_ready_set.json")
    ap.add_argument("--survey", default="output/survey.md")
    ap.add_argument("--out", default="cache/canonical_survival.json")
    args = ap.parse_args()
    run_audit(
        canonical_path=str(ROOT / args.canonical),
        trace_path=str(ROOT / args.trace),
        cards_path=str(ROOT / args.cards),
        whitelist_path=str(ROOT / args.whitelist),
        survey_path=str(ROOT / args.survey),
        out_path=str(ROOT / args.out),
    )


if __name__ == "__main__":
    main()
