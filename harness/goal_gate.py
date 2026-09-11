"""Deterministic Goal Gate for the verify->repair loop (joint 2, spec §3.3).

Pure code over data-dependent signals — the LLM never decides "passed".
v2 gates: structural validity (zero invalid citations, figure/table refs
included as entries) and unsupported claims (cleared, or repair budget
exhausted). Joint 3 adds the coverage gate: text/figure-ref retention
against the round-0 baseline, so deletion-style repair cannot hollow out
the survey and still pass.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

MAX_REPAIR_ROUNDS = 2
MIN_RETENTION_DEFAULT = 0.7

STOP_PASSED = "passed"
STOP_BUDGET = "repair_budget_exhausted"
STOP_FIXPOINT = "fixpoint"
STOP_COVERAGE = "coverage_fail"
STOP_REPAIRING = "repairing"  # not a stop: the loop should repair another round

_FIG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass
class GateResult:
    passed: bool
    stop_reason: str
    unsupported: int
    invalid_citations: int
    citation_validity_score: float
    coverage: dict[str, Any] | None = None
    source_role_violations: int = 0
    evidence_gaps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def figure_ref_count(md: str) -> int:
    """Count ``![alt](target)`` refs, same syntax as tools/verify/structural.py."""
    return len(_FIG_RE.findall(md))


def evaluate(
    metrics: dict[str, Any],
    *,
    repair_round: int,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    prev_unsupported: int | None = None,
    text_retention: float = 1.0,
    figure_ref_retention: float = 1.0,
    min_retention: float = MIN_RETENTION_DEFAULT,
) -> GateResult:
    """Judge one verify round. ``metrics`` = verify_citations tool result metrics."""
    unsupported = int(metrics.get("unsupported_claims", 0))
    invalid = int(metrics.get("invalid_citations", 0))
    score = float(metrics.get("citation_validity_score", 0.0))
    coverage = {
        "text_retention": float(text_retention),
        "figure_ref_retention": float(figure_ref_retention),
        "min_retention": float(min_retention),
        "ok": text_retention >= min_retention and figure_ref_retention >= min_retention,
    }
    roles = int(metrics.get("source_role_violations", 0))
    gaps = int(metrics.get("empty_evidence_sections", 0)) + int(metrics.get("no_verifiable_claims", 0))
    if not coverage["ok"]:
        return GateResult(False, STOP_COVERAGE, unsupported, invalid, score, coverage, roles, gaps)
    if roles or gaps:
        reason = STOP_BUDGET if repair_round >= max_repair_rounds else STOP_REPAIRING
        return GateResult(False, reason, unsupported, invalid, score, coverage, roles, gaps)
    if invalid == 0 and unsupported == 0 and coverage["ok"]:
        return GateResult(True, STOP_PASSED, unsupported, invalid, score, coverage)
    if repair_round >= max_repair_rounds:
        return GateResult(False, STOP_BUDGET, unsupported, invalid, score, coverage)
    if prev_unsupported is not None and not improved(unsupported, prev_unsupported):
        return GateResult(False, STOP_FIXPOINT, unsupported, invalid, score, coverage)
    return GateResult(False, STOP_REPAIRING, unsupported, invalid, score, coverage)


def improved(current_unsupported: int, prev_unsupported: int) -> bool:
    """Strictly fewer unsupported claims counts as progress (spec §3.3)."""
    return current_unsupported < prev_unsupported
