"""Seed-claim contract for the FINAL_SEED replay: every card claim must have a
supports_claims row that passes the writer's valid_support gate. Regression:
build_final_seed_papers emitted evidence without bindings, the wave8 claim gate
rejected every seed claim, and the replay degraded to a claim-less survey."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_final_seed_papers import build_final_seed_data
from tools.verify.source_contract import valid_support


def test_every_seed_claim_has_a_valid_support_binding():
    data = build_final_seed_data()
    evidence_by_id = {e["evidence_id"]: e for e in data["evidence_store"]["evidence"]}
    bound_claims = 0
    for card in data["paper_cards"]["paper_cards"]:
        for bucket, claims in card["possible_claims"].items():
            for claim in claims:
                evidence = evidence_by_id[claim["evidence_id"]]
                rows = [r for r in evidence["supports_claims"] if r["claim_text"] == claim["text"]]
                assert rows, f"{card['paper_id']} {bucket}: no binding for claim"
                for row in rows:
                    assert valid_support(row, evidence), (
                        f"{card['paper_id']} {bucket}: binding fails the writer gate: {row}")
                bound_claims += 1
    assert bound_claims >= 30  # 12 papers x 3 buckets


def test_seed_evidence_roles_are_scope_consistent():
    from tools.verify.source_contract import SCOPE_ROLES

    data = build_final_seed_data()
    for evidence in data["evidence_store"]["evidence"]:
        for row in evidence["supports_claims"]:
            scope = {"own_method": "method", "own_contribution": "contribution",
                     "own_limitation": "limitation"}[row["source_role"]]
            assert row["source_role"] in SCOPE_ROLES[scope]
