"""TypeSafe JEV client — repair-action decisions via System One Choice questions.

Docs: https://docs.typesafe.ai. JEV answers typed questions (choice/score/
noul) about a state with probability distributions — no text generation, no
response parsing. Here it replaces the Intern-S2 json_chat batch decision in
the repair agent (joint 2): action selection is a closed-set classification,
exactly the shape System One models are built for. Param questions (which
candidate paper / evidence / figure) are asked speculatively in the same
call. rewrite_claim's new_text generation stays with Intern-S2; delete
reasons are synthesized from the answer distribution.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_MIN_CONFIDENCE = 0.5

# Action rubrics for Choice criteria (what each action means, not how to run it).
ACTION_DESCRIPTIONS = {
    "remap_citation": "A listed candidate paper's title clearly matches the sentence; the citation id is simply wrong",
    "rewrite_claim": "Weaken or generalize the assertion so the listed evidence can support it",
    "swap_evidence": "A different listed evidence row supports the claim as-is",
    "backfill_evidence": "Fetch more grounding text for the claim before deciding",
    "remap_figure": "A listed candidate figure shows the same content as the broken reference",
    "delete_claim": "Nothing simpler works; remove the claim sentence (last resort)",
    "delete_line": "Nothing simpler works; remove the whole line (last resort)",
    "keep": "The weak assertion is acceptable as-is",
}

# action -> (param key, candidate field on the record, question id prefix)
_PARAM_QUESTION = {
    "remap_citation": ("new_id", "candidates", "paper_id"),
    "swap_evidence": ("evidence_id", "evidence_preview", "evidence_id"),
    "remap_figure": ("new_figure_id", "candidates", "figure_id"),
}
_NONE_OPTION = "none_of_the_above"


def make_jev_decider():
    """EVISURVEY_JEV=auto (default) enables JEV when TYPESAFE_API_KEY is set;
    off disables; on requires the key (raises — caller falls back honestly)."""
    mode = (os.getenv("EVISURVEY_JEV", "auto") or "auto").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return None
    api_key = (os.getenv("TYPESAFE_API_KEY") or "").strip()
    if not api_key:
        if mode in {"1", "true", "yes", "on"}:
            raise ValueError("EVISURVEY_JEV=on but TYPESAFE_API_KEY is empty")
        return None
    from typesafe_sdk import TypeSafeClient

    min_confidence = float(os.getenv("EVISURVEY_JEV_MIN_CONFIDENCE", "") or DEFAULT_MIN_CONFIDENCE)
    return JevRepairDecider(
        TypeSafeClient(api_key=api_key), min_confidence=min_confidence)


class JevRepairDecider:
    """One system_one call per failure group: action Choice per record plus
    speculative param Choice questions, all against the same state."""

    def __init__(self, client, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.client = client
        self.min_confidence = min_confidence

    def decide_group(self, group: str, actions: list[str], records: list[dict]) -> dict[int, dict]:
        """Ask JEV for one action per record. Returns {record_index: decision}.

        Decision shape: {"action": str} — action None means JEV could not
        decide confidently (low confidence, or a param question answered
        none_of_the_above); the caller routes those records to the Intern-S2
        fallback. Raises on API/payload errors (caller falls back per group).
        """
        state, questions, param_slots = build_request(group, actions, records)
        started = time.monotonic()
        response = self.client.system_one(state=state, questions=questions)
        elapsed = time.monotonic() - started
        answers = response.answers

        out: dict[int, dict] = {}
        for i, record in enumerate(records):
            answer = answers.get(f"act_{i}")
            if answer is None:
                # Real JEV answers every question; a missing answer is a payload
                # anomaly -> let the caller route this record to Intern-S2.
                out[i] = {"action": None, "reason": "jev did not answer this record"}
                continue
            top = answer.choice
            confidence = float(answer.confidence)
            decision = {
                "action": top,
                "confidence": confidence,
                "probabilities": {k: round(float(v), 3) for k, v in answer.probabilities.items()},
                "reason": f"jev p({top})={answer.probabilities.get(top, 0):.2f}",
            }
            if confidence < self.min_confidence:
                decision.update(action=None, reason=f"jev confidence {confidence:.2f} < {self.min_confidence}")
                out[i] = decision
                continue
            slot = param_slots.get(i)
            if slot is not None and top == slot["action"]:
                chosen = answers[slot["question"]].choice
                if chosen == _NONE_OPTION:
                    decision.update(action=None,
                                    reason=f"jev: no candidate fits ({slot['action']})")
                else:
                    decision["params"] = {slot["param"]: chosen}
            out[i] = decision
        usage = getattr(response, "usage", None)
        logger.info(
            "[repair][jev] group %s: %d records, %.1fs, actions=%s%s",
            group, len(records), elapsed,
            {r: d.get("action") for r, d in sorted(out.items())},
            f", tokens={usage.input_tokens}/{usage.output_tokens}" if usage else "")
        return out


def build_request(group: str, actions: list[str], records: list[dict]):
    """Pure request builder — shared with unit tests.

    Returns (state, questions, param_slots) where param_slots maps record
    index -> {"action", "param", "question"} for the speculative param Choice
    (absent when the record lists no candidates to select from).
    """
    from typesafe_sdk import Choice

    state = {
        "failure_group": group,
        "failures": records,
    }
    questions: dict[str, Choice] = {}
    param_slots: dict[int, dict] = {}
    for i in range(len(records)):
        # per-record options: selection actions stay only when the record
        # actually lists candidates for them
        record_actions = list(actions)
        for action, (_param, field, _key) in _PARAM_QUESTION.items():
            if action in record_actions and not records[i].get(field):
                record_actions.remove(action)
        questions[f"act_{i}"] = Choice(
            instructions=f"Which repair action fits `failures[{i}]`",
            criteria={a: ACTION_DESCRIPTIONS[a] for a in record_actions},
        )
        for action, (param, field, id_key) in _PARAM_QUESTION.items():
            options = records[i].get(field) or []
            if action not in actions or not options or action not in record_actions:
                continue
            question = f"param_{i}"
            questions[question] = Choice(
                instructions=(
                    f"If the right action for `failures[{i}]` is {action}, which of "
                    f"its listed candidates should be used"),
                criteria={
                    str(option.get(id_key)): str(option.get("title") or option.get("text") or "")[:160]
                    for option in options if option.get(id_key)
                } | {_NONE_OPTION: "No listed candidate fits"},
            )
            param_slots[i] = {"action": action, "param": param, "question": question}
    return state, questions, param_slots
