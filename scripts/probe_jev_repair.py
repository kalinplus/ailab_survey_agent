"""Live probe: JEV repair-action decisions on realistic failure records.

Builds failure-group records from the real wave8 fixtures
(cache/final_citation_ready_set.json + final_evidence_store.json), calls the
real TypeSafe JEV API through JevRepairDecider, and prints one decision per
record with confidence and the full action distribution.

Usage: python scripts/probe_jev_repair.py   (needs TYPESAFE_API_KEY in .env)
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from tools.clients.jev_client import JevRepairDecider, make_jev_decider  # noqa: E402

# Hand-built failure records mirroring group_failures() output shapes; content
# is taken from the real fixtures so the judgments are realistic.
RECORDS = {
    # B: unsupported but the cited paper has evidence chunks
    "B": [
        {
            "claim_id": "claim_b1",
            "claim_text": "DreamerV3 首次在世界模型方法中引入了隐式规划机制，使其在所有 Atari 游戏上超越人类水平。",
            "cited_paper_id": "world_models_2018",
            "source_bindings": [],
            "source_role_violation": "",
            "scope": "",
            "evidence_preview": [
                {"evidence_id": "ev_world_models_2018_problem", "text": "How to learn compact spatial and temporal representations of game environments so that agents can plan and act efficiently"},
                {"evidence_id": "ev_world_models_2018_method", "text": "We train an agent whose policy is a mixture of a V model (vision) learning a compressed spatial representation, an M model (memory) predicting future states, and a C model (controller)"},
            ],
        },
        {
            "claim_id": "claim_b2",
            "claim_text": "World Models 提出 V-M-C 三组件架构，其中控制器 C 是唯一可训练的组件。",
            "cited_paper_id": "world_models_2018",
            "source_bindings": [],
            "source_role_violation": "",
            "scope": "",
            "evidence_preview": [
                {"evidence_id": "ev_world_models_2018_method", "text": "We train an agent whose policy is a mixture of a V model (vision) learning a compressed spatial representation, an M model (memory) predicting future states, and a C model (controller)"},
            ],
        },
    ],
    # D: weak (direction ok, assertion too strong)
    "D": [
        {
            "claim_id": "claim_d1",
            "claim_text": "Genie 类生成式环境未来大概率会完全取代所有基于规则的仿真器。",
            "cited_paper_id": "world_models_2018",
            "source_bindings": [],
            "source_role_violation": "",
            "scope": "",
            "evidence_preview": [
                {"evidence_id": "ev_world_models_2018_problem", "text": "How to learn compact spatial and temporal representations of game environments so that agents can plan and act efficiently"},
            ],
        },
    ],
    # A: wrong citation id with real candidate titles
    "A": [
        {
            "citation_id": "ha_world_models_2018",
            "sentence": "The V-M-C decomposition lets an agent learn inside its own dream [ha_world_models_2018].",
            "candidates": [
                {"paper_id": "world_models_2018", "title": "World Models"},
                {"paper_id": "dreamer_v3", "title": "Mastering Diverse Domains through World Models"},
            ],
        },
    ],
}

GROUP_ACTIONS = {
    "A": ["remap_citation", "delete_claim"],
    "B": ["rewrite_claim", "swap_evidence", "backfill_evidence", "delete_claim"],
    "D": ["rewrite_claim", "keep"],
}


def main():
    decider = make_jev_decider()
    if decider is None:
        raise SystemExit("JEV disabled: set TYPESAFE_API_KEY / EVISURVEY_JEV in .env")

    total_started = time.monotonic()
    for group, records in RECORDS.items():
        actions = GROUP_ACTIONS[group]
        print(f"\n=== group {group} ({', '.join(actions)}) ===")
        answers = decider.decide_group(group, actions, records)
        for i, record in enumerate(records):
            a = answers.get(i, {})
            fid = record.get("citation_id") or record.get("claim_id")
            if a.get("action") is None:
                print(f"  {fid}: PUNTED -> {a.get('reason')}")
                continue
            params = a.get("params") or {}
            print(f"  {fid}: {a['action']}"
                  + (f" {params}" if params else "")
                  + f"  conf={a['confidence']:.2f}")
            print(f"      p: {a['probabilities']}")
    print(f"\ntotal elapsed: {time.monotonic() - total_started:.1f}s")


if __name__ == "__main__":
    main()
