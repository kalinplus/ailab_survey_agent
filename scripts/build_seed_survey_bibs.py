#!/usr/bin/env python3
"""Build cache/seed_survey_bibs.json — the union bibliography of the seed surveys.

Gold-free replacement denominator for the eval's reference recall: instead of one
gold survey's reference list, the union of what the seed surveys themselves cite.

Source is the seed survey artifact Module B already consumes (P2 reads it through
tools/knowledge_pipeline_worker.py), so nothing is re-fetched from the network:
each survey's reference list is meta_data.top_referenced_papers (id hints). The
hints are resolved to real titles from artifacts already on disk, in priority
order — exact paper_id match first (paper cards), then P3's survey_ref_hints
back-link on retrieval cards — and deduped by normalized title. Hints with no
resolvable title keep their id so the eval can still match them by id.

Usage:
  python scripts/build_seed_survey_bibs.py                 # default paths
  python scripts/build_seed_survey_bibs.py --out cache/seed_survey_bibs.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.json_io import read_json, write_json  # noqa: E402
from tools.evaluate_survey import _normalize_title, _norm_pid  # noqa: E402


def _log(message: str) -> None:
    print(f"[seed-bibs] {message}", flush=True)


def _papers(path: Path) -> list[dict]:
    """Card list from a card artifact ({"papers": ...} / {"paper_cards": ...} / list)."""
    if not path.exists():
        return []
    data = read_json(path)
    if isinstance(data, list):
        return data
    return data.get("papers") or data.get("paper_cards") or []


def _resolve(hint: str, sources: list[tuple[str, list[dict]]]) -> tuple[dict | None, str | None]:
    """Real card for a seed-survey id hint: exact paper id beats P3's hint back-link."""
    for match_by_id in (True, False):
        for name, cards in sources:
            for card in cards:
                if match_by_id:
                    if _norm_pid(card.get("paper_id") or "") == _norm_pid(hint):
                        return card, name
                elif hint in (card.get("survey_ref_hints") or []):
                    return card, name
    return None, None


def build(surveys_path: Path, out_path: Path, card_paths: list[Path]) -> dict:
    surveys = read_json(surveys_path)
    sources = [(path.name, _papers(path)) for path in card_paths]
    _log(f"surveys={len(surveys)} from {surveys_path}, "
         f"card sources={[(n, len(c)) for n, c in sources]}")

    union: dict[str, dict] = {}
    n_refs = 0
    for survey in surveys:
        refs = (survey.get("meta_data") or {}).get("top_referenced_papers") or []
        origin = survey.get("title") or survey.get("paper_id")
        for hint in dict.fromkeys(refs):
            n_refs += 1
            card, resolved_from = _resolve(str(hint), sources)
            entry = union.setdefault(str(hint), {
                "id": str(hint), "title": None, "year": card.get("year") if card else None,
                "venue": card.get("venue") if card else None, "resolved_from": None,
                "survey_ref_count": 0, "surveys": [],
            })
            entry["survey_ref_count"] += 1
            entry["surveys"].append(origin)
            title = (card or {}).get("title")
            # first resolvable title wins; later ones are dropped as duplicates
            if title and not entry["title"]:
                entry["title"], entry["resolved_from"] = title, resolved_from
            elif title and _normalize_title(title) != _normalize_title(entry["title"]):
                _log(f"conflicting titles for {hint}: {entry['title']!r} vs {title!r} "
                     f"(keeping {entry['title']!r} from {entry['resolved_from']})")

    entries = sorted(union.values(), key=lambda e: e["id"])
    resolved = sum(1 for e in entries if e["title"])
    artifact = {
        "source": str(surveys_path),
        "generated_by": "scripts/build_seed_survey_bibs.py",
        "n_surveys": len(surveys),
        "n_survey_refs": n_refs,
        "n_title_resolved": resolved,
        "entries": entries,
    }
    write_json(out_path, artifact)
    _log(f"wrote {out_path}: {len(entries)} union refs from {n_refs} survey refs, "
         f"{resolved} resolved to a title, {len(entries) - resolved} id-only")
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surveys", default="cache/surveys.json",
                    help="seed survey artifact P2 consumes (meta_data.top_referenced_papers)")
    ap.add_argument("--out", default="cache/seed_survey_bibs.json")
    ap.add_argument("--cards", nargs="+",
                    default=["cache/final_paper_cards.json", "cache/paper_cards.json",
                             "cache/retrieved_papers.json"],
                    help="title sources, tried in order (exact paper_id, then survey_ref_hints)")
    args = ap.parse_args()

    surveys_path = ROOT / args.surveys
    if not surveys_path.exists():
        raise SystemExit(f"seed survey artifact not found: {surveys_path}")
    build(surveys_path, ROOT / args.out, [ROOT / p for p in args.cards])


if __name__ == "__main__":
    main()
