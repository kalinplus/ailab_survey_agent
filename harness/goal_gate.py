"""Deterministic Goal Gate for the verify->repair loop (joint 2, spec §3.3).

Pure code over data-dependent signals — the LLM never decides "passed".
v2 gates: structural validity (zero invalid citations, figure/table refs
included as entries) and unsupported claims (cleared, or repair budget
exhausted). The coverage gate is deferred to v3 (stop_reason keeps room).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MAX_REPAIR_ROUNDS = 2

STOP_PASSED = "passed"
STOP_BUDGET = "repair_budget_exhausted"
STOP_FIXPOINT = "fixpoint"
STOP_REPAIRING = "repairing"  # not a stop: the loop should repair another round


@dataclass
class GateResult:
    passed: bool
    stop_reason: str
    unsupported: int
    invalid_citations: int
    citation_validity_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    metrics: dict[str, Any],
    *,
    repair_round: int,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    prev_unsupported: int | None = None,
) -> GateResult:
    """Judge one verify round. ``metrics`` = verify_citations tool result metrics."""
    unsupported = int(metrics.get("unsupported_claims", 0))
    invalid = int(metrics.get("invalid_citations", 0))
    score = float(metrics.get("citation_validity_score", 0.0))
    if invalid == 0 and unsupported == 0:
        return GateResult(True, STOP_PASSED, unsupported, invalid, score)
    if repair_round >= max_repair_rounds:
        return GateResult(False, STOP_BUDGET, unsupported, invalid, score)
    if prev_unsupported is not None and not improved(unsupported, prev_unsupported):
        return GateResult(False, STOP_FIXPOINT, unsupported, invalid, score)
    return GateResult(False, STOP_REPAIRING, unsupported, invalid, score)


def improved(current_unsupported: int, prev_unsupported: int) -> bool:
    """Strictly fewer unsupported claims counts as progress (spec §3.3)."""
    return current_unsupported < prev_unsupported
