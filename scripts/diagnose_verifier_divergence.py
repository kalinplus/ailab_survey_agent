"""Diagnose the internal-claimer vs evaluator-L1 verdict divergence (wave 5, item 3).

Wave-4 cold run: internal claim_map 0/28 supported vs independent L1 11/28,
same evidence store, same NLI model. This script recomputes BOTH sides under
one process/model, joins them per (claim, citation) pair, and prints the
agreement matrix plus divergence samples — before/after the best_match wrapper
fix it is the empirical record of the root cause.

Usage:
  EVISURVEY_REAL_NLI=1 EVISURVEY_NLI_DEVICE=cpu HF_HUB_OFFLINE=1 \
  python scripts/diagnose_verifier_divergence.py [--survey output/survey.md] \
      [--evidence cache/evidence_store.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from tools.models.artifacts import EvidenceStore, ParsedPapers  # noqa: E402
from tools.nlp.nli_verifier import FakeNLIModel, NLIVerifier  # noqa: E402
from tools.verify import claim_mapper  # noqa: E402
from tools.verify.text_units import normalize_text  # noqa: E402


def _nli():
    if os.getenv("EVISURVEY_REAL_NLI", "").lower() in {"1", "true", "yes"}:
        return NLIVerifier()
    return FakeNLIModel(mapping={"world": "entailment", "model": "entailment",
                                 "game": "entailment", "agent": "entailment"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default="output/survey.md")
    ap.add_argument("--evidence", default="cache/evidence_store.json")
    args = ap.parse_args()

    load_config()
    md = Path(args.survey).read_text(encoding="utf-8")
    store = EvidenceStore(**json.loads(Path(args.evidence).read_text(encoding="utf-8")))
    nli = _nli()
    print(f"[diag] survey={len(md)} chars, evidence={len(store.evidence)}, nli={type(nli).__name__}")

    structured = None
    sc_path = Path("cache/structured_claims.json")
    if sc_path.exists():
        structured = json.loads(sc_path.read_text(encoding="utf-8")).get("claims") or None

    cm = claim_mapper.run("diag", md, store, ParsedPapers(task_id="diag", papers=[]), nli,
                          structured_claims=structured)
    internal = {(normalize_text(e.claim_text)[:80], e.cited_paper_id): e.status
                for e in cm.entries}

    from tools.evaluate_survey import citation_quality  # local import: heavier module
    l1 = citation_quality(md, store, nli)

    joined = []
    for p in l1["pairs"]:
        key = (normalize_text(p["claim_text"])[:80], p["cited_paper_id"])
        joined.append((key, internal.get(key, "<no-internal-entry>"), p["status"]))

    from collections import Counter
    matrix = Counter((i, l) for _, i, l in joined)
    print(f"[diag] L1 pairs={l1['n_pairs']} supported={l1['supported_pairs']} "
          f"weak={l1['weak_pairs']} unsupported={l1['unsupported_pairs']}")
    print(f"[diag] internal statuses: {dict(Counter(i for _, i, _ in joined))}")
    print("[diag] agreement matrix (internal -> L1):")
    for (i, l), n in sorted(matrix.items()):
        marker = "  <-- DIVERGENT" if i != l else ""
        print(f"    {i:>12} -> {l:<12} {n}{marker}")

    print("[diag] divergent samples (claim prefix | internal | L1):")
    shown = 0
    for key, i, l in joined:
        if i != l and shown < 8:
            print(f"    [{i} vs {l}] {key[1]} :: {key[0][:70]}")
            shown += 1


if __name__ == "__main__":
    main()
