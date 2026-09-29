"""A/B: repair-agent action decisions via Intern-S2 (EVISURVEY_JEV=off) vs
TypeSafe JEV (auto), on an identically damaged real survey.

Base: output/survey.md + cache/structured_claims.json + cache/final_* from a
FINAL_SEED + writer-LLM run (real content, real NLI verify). Damage is
injected surgically, mirroring the failure types documented in the wave
archives (A wrong id / B unsupported overclaim / D over-strong assertion /
E broken figure ref). Both arms repair the exact same damaged state; the
driver then re-verifies each arm's revised markdown with the real NLI model.

Usage: python scripts/ab_jev_repair.py   (needs INTERN_API_KEY + TYPESAFE_API_KEY)
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ab_jev")

OUT_DIR = ROOT / "output" / "ab_jev"
ROUND_NO = 1

# (group, find, replacement) — find must occur exactly once in the base.
REPLACEMENTS = [
    ("A", "league-based multi-agent training [alphastar_2019]", "league-based multi-agent training [alphastar_2018]"),  # wrong-but-plausible id
    ("E", "](publication_timeline)", "](fig_nonexistent_timeline)"),   # broken figure ref
]

# Post-wave8 failure shape: structured claims with VALID bindings whose text
# drifts from the evidence — the mapper re-checks NLI against the bound
# source_quote, so overclaims land as B (unsupported) / D (weak), exactly like
# organic writer drift. (sentence, paper_id, evidence_id, scope)
OVERCLAIMS = [
    ("DreamerV3 achieves superhuman scores on every Atari 57 game and proves world models scale to any domain.",
     "dreamerv3_2023", "ev_dreamerv3_2023_contribution", "contribution"),
    ("UniSim simulators already drive production autonomous vehicles and robot fleets end to end.",
     "unisim_2024", "ev_unisim_2024_contribution", "contribution"),
    ("Genie's latent action model has influenced most interactive world generation systems since its release.",
     "genie_2024", "ev_genie_2024_method", "method"),
    ("MineDojo's benchmark tasks have been adopted as the standard evaluation suite across all embodied agent research.",
     "minedojo_2022", "ev_minedojo_2022_contribution", "contribution"),
]


def build_damaged_survey(base_md: str, evidence_by_id: dict, structured_claims: list):
    """Returns (damaged markdown, structured claims + overclaims)."""
    for _group, needle, replacement in REPLACEMENTS:
        assert base_md.count(needle) == 1, f"damage anchor not unique: {needle}"
        base_md = base_md.replace(needle, replacement)
    sentences, new_claims = [], []
    for sentence, paper_id, evidence_id, scope in OVERCLAIMS:
        row = dict(evidence_by_id[evidence_id]["supports_claims"][0])
        # writer-shape binding: the supports row fields PLUS the singular
        # evidence_id pointing at the bound evidence row (claim_mapper looks
        # the row up by that key)
        row["evidence_id"] = evidence_id
        sentences.append(f"{sentence.rstrip('.')} [{paper_id}].")
        new_claims.append({"text": sentence, "cites": [paper_id], "scope": scope,
                           "source_bindings": [row]})
    lines = base_md.splitlines(keepends=True)
    anchor = next(i for i, ln in enumerate(lines) if ln.startswith("## "))
    pos = len("".join(lines[: anchor + 1]))
    damaged = base_md[:pos] + " ".join(sentences) + "\n\n" + base_md[pos:]
    return damaged, structured_claims + new_claims


def run_arm(arm: str, damaged_md: str, claim_map: dict, evidence_store: dict,
            ready_set: dict, figure_items: list, allowed_artifacts: set,
            structured_claims: list) -> dict:
    env = dict(os.environ)
    env["EVISURVEY_JEV"] = "off" if arm == "intern" else "auto"
    arm_dir = OUT_DIR / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm, "round_no": ROUND_NO, "survey_md": damaged_md,
        "claim_map": claim_map, "evidence_store": evidence_store,
        "ready_set": ready_set, "figure_items": figure_items,
        "allowed_artifacts": sorted(allowed_artifacts),
        "structured_claims": structured_claims, "out_dir": str(arm_dir),
    }
    job = arm_dir / "job.json"
    job.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "ab_jev_arm.py"), str(job)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        log.error(f"[{arm}] arm failed:\n{proc.stderr[-2000:]}")
        raise SystemExit(1)
    result = json.loads((arm_dir / "result.json").read_text())
    result["wall_seconds"] = round(elapsed, 1)
    return result


def main() -> None:
    from tools.verify_citations import _nli_model  # real NLI via env
    from tools.verify.claim_mapper import run as claim_map_run
    from tools.models.artifacts import EvidenceStore
    from tools.revise_survey import _read_optional
    from harness.agents.repair_agent import group_failures

    os.environ.setdefault("EVISURVEY_REAL_NLI", "1")
    base_md = (ROOT / "output" / "survey.md").read_text(encoding="utf-8")
    structured = _read_optional(ROOT, "cache/structured_claims.json", {})
    structured_claims = structured.get("claims") or []
    evidence_store = _read_optional(ROOT, "cache/final_evidence_store.json", {"evidence": []})
    ready_set = _read_optional(ROOT, "cache/final_citation_ready_set.json", {})
    figure_bank = _read_optional(ROOT, "cache/final_figure_bank.json", {"figures": []})
    gen_bank = _read_optional(ROOT, "cache/generated_artifact_bank.json", {"artifacts": []})
    figure_items = [f for f in figure_bank.get("figures", []) if isinstance(f, dict)]
    allowed_artifacts = {a.get("artifact_id") for a in gen_bank.get("artifacts", []) if a.get("artifact_id")}

    damaged_md, damaged_claims = build_damaged_survey(
        base_md, {e["evidence_id"]: e for e in evidence_store.get("evidence", [])},
        structured_claims)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "survey_damaged.md").write_text(damaged_md, encoding="utf-8")

    nli = _nli_model()
    store = EvidenceStore(**evidence_store)
    claim_map = claim_map_run("ab_jev", damaged_md, store, None, nli, None,
                              structured_claims=damaged_claims)
    cm_dict = claim_map.model_dump()
    (OUT_DIR / "claim_map_damaged.json").write_text(
        json.dumps(cm_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    by_paper: dict[str, list] = {}
    for e in evidence_store.get("evidence", []):
        by_paper.setdefault(e.get("paper_id", ""), []).append(e)
    groups = group_failures(damaged_md, dict(cm_dict, _evidence_by_paper=by_paper),
                            ready_set, figure_items, allowed_artifacts)
    failures = {g: len(v) for g, v in groups.items() if v}
    log.info(f"damaged base failures: {failures} (total {sum(failures.values())})")
    (OUT_DIR / "base_failures.json").write_text(json.dumps(failures), encoding="utf-8")
    if sum(failures.values()) == 0:
        raise SystemExit("no failures on damaged base — nothing to compare")

    results = {}
    for arm in ("intern", "jev"):
        log.info(f"=== arm {arm} ===")
        results[arm] = run_arm(arm, damaged_md, cm_dict, evidence_store, ready_set,
                               figure_items, allowed_artifacts, damaged_claims)

    (OUT_DIR / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
