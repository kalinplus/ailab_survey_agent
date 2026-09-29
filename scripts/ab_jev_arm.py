"""One A/B arm: repair the shared damaged survey, then re-verify.

Runs in a subprocess with EVISURVEY_JEV set per arm. Mirrors the production
revise seam: run_repair with heavy-LLM json_chat, real NLI, sciverse=None
(no key — backfill is honestly unavailable in both arms), then the
structured-claims rewrite sync and a fresh claim_map on the revised markdown.
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from config import load_config

from harness.agents.repair_agent import group_failures, run_repair
from tools.models.artifacts import EvidenceStore
from tools.revise_survey import _make_llm_json_chat
from tools.verify.claim_mapper import run as claim_map_run
from tools.verify.text_units import normalize_text
from tools.verify_citations import _nli_model


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    arm_dir = Path(job["out_dir"])
    cfg = load_config()

    repaired = run_repair(
        task_id=f"ab_jev_{job['arm']}", round_no=job["round_no"],
        survey_md=job["survey_md"], claim_map=job["claim_map"],
        evidence_store=job["evidence_store"], ready_set=job["ready_set"],
        figure_items=job["figure_items"], allowed_artifacts=set(job["allowed_artifacts"]),
        llm_json_chat=_make_llm_json_chat(cfg), nli=_nli_model(), sciverse=None,
        trajectory_dir=arm_dir / "trajectory", jev="unset",
    )
    revised_md = repaired["revised_md"]
    (arm_dir / "survey_revised.md").write_text(revised_md, encoding="utf-8")
    (arm_dir / "repair_log.json").write_text(
        json.dumps(repaired["repair_log"], ensure_ascii=False, indent=2), encoding="utf-8")

    # production rewrite sync: structured claims follow rewritten sentences
    structured = [dict(c) for c in job["structured_claims"]]
    for action in repaired["repair_log"]:
        if not action.get("rewritten_text"):
            continue
        for claim in structured:
            if (normalize_text(claim.get("text", "")) == normalize_text(action["claim_text"])
                    and action.get("cited_paper_id") in claim.get("cites", [])):
                claim["text"] = action["rewritten_text"]
                claim["scope"] = action.get("rewritten_scope") or claim.get("scope", "")

    # post verify: fresh claim map + failure groups on the revised markdown
    nli = _nli_model()
    claim_map = claim_map_run("ab_jev", revised_md, EvidenceStore(**job["evidence_store"]),
                              None, nli, None, structured_claims=structured)
    cm = claim_map.model_dump()
    (arm_dir / "claim_map_post.json").write_text(
        json.dumps(cm, ensure_ascii=False, indent=2), encoding="utf-8")
    statuses = Counter(e["status"] for e in cm["entries"])
    by_paper: dict[str, list] = {}
    for e in job["evidence_store"].get("evidence", []):
        by_paper.setdefault(e.get("paper_id", ""), []).append(e)
    groups = group_failures(revised_md, dict(cm, _evidence_by_paper=by_paper),
                            job["ready_set"], job["figure_items"], set(job["allowed_artifacts"]))
    post_failures = {g: len(v) for g, v in groups.items() if v}
    actions = Counter((e["failure_type"], e["action"]) for e in repaired["repair_log"])

    result = {
        "arm": job["arm"],
        "jev_env": os.getenv("EVISURVEY_JEV"),
        "metrics": repaired["metrics"],
        "repair_actions": {f"{g}:{a}": n for (g, a), n in sorted(actions.items())},
        "outcomes": dict(Counter(e["outcome"] for e in repaired["repair_log"])),
        "post_statuses": dict(statuses),
        "post_failures": post_failures,
        "post_failure_total": sum(post_failures.values()),
    }
    (arm_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
