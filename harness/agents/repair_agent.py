"""Joint 2: repair agent — failure-typed corrective actions.

Spec: docs/RealAgent/关节2-修复Agent-方案与测试.md. Replaces deletion-only
repair: failures are grouped mechanically (A wrong citation id / B unsupported
with evidence / C unsupported zero evidence / D weak overclaim / E bad figure
ref), an LLM decides actions in typed batches (<=5 items per call), the
framework executes them with strict param validation (LLM-proposed ids must
be inside code-chosen candidate sets), and every mutated claim is re-judged
by NLI before it counts as repaired. Deletion is the last resort and always
carries a reason.

Unlike joint 1 this is NOT a multi-round BoundedAgentLoop: the decision
structure is one-shot batch classification (finite failure list, no
intra-batch dependencies). We reuse loop.py's trajectory writer and its
repair-retry-parse pattern; round control lives in the outer Goal Gate loop.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from .loop import trajectory_writer
from tools.models.common import evidence_id as make_evidence_id
from tools.verify.text_units import evidence_units, sentence_units
from tools.verify.source_contract import SCOPE_ROLES, valid_support, quote_diagnostic
from tools.verify.provenance import chunk_id, hit_matches_paper

logger = logging.getLogger(__name__)

DEFAULT_MAX_LLM_CALLS = 8
DEFAULT_MAX_SCIVERSE_CALLS = 10
BATCH_SIZE = 5
DELETE_RATIO_WARN = 0.5

# group -> (allowed actions, human semantics for the prompt)
GROUP_ACTIONS = {
    "A": ("remap_citation, delete_claim", "wrong citation id (not in whitelist)"),
    "B": ("rewrite_claim, swap_evidence, backfill_evidence, delete_claim",
          "claim unsupported but the cited paper HAS evidence chunks"),
    "C": ("backfill_evidence, remap_citation, delete_claim",
          "claim unsupported and the cited paper has ZERO evidence"),
    "D": ("rewrite_claim, keep", "weak claim (evidence direction ok, assertion too strong)"),
    "E": ("remap_figure, delete_line", "figure reference not in the figure bank"),
    "F": ("rewrite_claim, delete_claim", "source role or evidence binding violation; NLI alone cannot clear attribution"),
    "Q": ("rewrite_claim, keep", "advisory quote-like copying; paraphrase while preserving scope and technical terms; overlap is not semantic support"),
}

ACTION_SCHEMA = {
    "remap_citation": ["new_id"],
    "delete_claim": ["reason"],
    "rewrite_claim": ["new_text"],
    "swap_evidence": ["evidence_id"],
    "backfill_evidence": [],
    "keep": [],
    "remap_figure": ["new_figure_id"],
    "delete_line": ["reason"],
}


# ---------------------------------------------------------------------------
# failure grouping (zero LLM)
# ---------------------------------------------------------------------------


def _extract_citations(md: str) -> set[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md)
    return {m.strip() for m in re.findall(r"\[([^\]]+)\]", text_only)
            if not m.strip().startswith("http")}


def _extract_figure_refs(md: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)


def _sentences(md: str) -> list[str]:
    return re.split(r"(?<=[.。])\s+", md)


def _sentence_containing(md: str, needle: str) -> str:
    for sentence in _sentences(md):
        if needle in sentence:
            return sentence
    return ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))


def _paper_candidates(text: str, items: list[dict], limit: int = 5) -> list[dict]:
    """Code-chosen candidate papers whose titles overlap the claim text."""
    text_tokens = _tokens(text)
    scored = []
    for item in items:
        title = str(item.get("title", ""))
        overlap = len(_tokens(title) & text_tokens)
        if overlap:
            scored.append((-overlap, item.get("paper_id", ""), title[:110], item))
    scored.sort()
    return [{"paper_id": s[3].get("paper_id"), "title": s[2]} for s in scored[:limit]]


def _figure_candidates(alt_text: str, ref: str, figure_items: list[dict], limit: int = 3) -> list[dict]:
    """Code-chosen candidate figures by caption token overlap with the alt text."""
    ref_tokens = _tokens(alt_text) | _tokens(ref)
    scored = []
    for figure in figure_items:
        caption = str(figure.get("caption", ""))
        overlap = len(_tokens(caption) & ref_tokens)
        if overlap:
            scored.append((-overlap, figure.get("figure_id", ""), caption[:110]))
    scored.sort()
    return [{"figure_id": s[1], "caption": s[2]} for s in scored[:limit]]


def _preview(evidences: list[dict], limit: int = 3) -> list[dict]:
    return [{"evidence_id": e.get("evidence_id"), "text": str(e.get("text", ""))[:300]}
            for e in evidences[:limit]]


def group_failures(
    survey_md: str,
    claim_map: dict[str, Any],
    ready_set: dict[str, Any],
    figure_items: list[dict],
    allowed_artifacts: set[str],
) -> dict[str, list[dict]]:
    """Mechanically classify verification failures into groups A-E.

    A/E are re-derived from the markdown vs the whitelists (equivalent to the
    structural check's valid=false entries); B/C/D come from claim_map.
    """
    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    items = [i for i in ready_set.get("items", []) if isinstance(i, dict)]

    failures_a = []
    for citation_id in sorted(_extract_citations(survey_md) - allowed_ids):
        sentence = _sentence_containing(survey_md, f"[{citation_id}]")
        failures_a.append({
            "citation_id": citation_id,
            "sentence": sentence[:400],
            "candidates": _paper_candidates(sentence or citation_id, items),
        })

    failures_b: list[dict] = []
    failures_c: list[dict] = []
    failures_d: list[dict] = []
    failures_f, failures_q = [], []
    for index, entry in enumerate(claim_map.get("entries", [])):
        status = entry.get("status")
        if status not in {"unsupported", "weak"} and not entry.get("source_role_violation") and not entry.get("quote_diagnostic", {}).get("quote_like"):
            continue
        # claims citing a non-whitelist id are collateral damage of an A-type
        # failure: one remap fixes every occurrence, so they must NOT be
        # re-decided per claim here — they re-classify after the next full verify
        if entry.get("cited_paper_id") not in allowed_ids:
            continue
        record = {
            "claim_id": f"claim_{index}",
            "claim_text": str(entry.get("claim_text", "")),
            "scope": entry.get("scope", ""),
            "source_bindings": entry.get("source_bindings", []),
            "source_role_violation": entry.get("source_role_violation", ""),
            "quote_diagnostic": entry.get("quote_diagnostic", {}),
            "cited_paper_id": entry.get("cited_paper_id", ""),
        }
        evidences = claim_map.get("_evidence_by_paper", {}).get(record["cited_paper_id"], [])
        bound_ids = {b.get("evidence_id") for b in record["source_bindings"]}
        if bound_ids:
            evidences = [e for e in evidences if e.get("evidence_id") in bound_ids]
        if entry.get("source_role_violation"):
            record["evidence_preview"] = _preview(evidences)
            failures_f.append(record)
        elif entry.get("quote_diagnostic", {}).get("quote_like") and status == "supported":
            record["evidence_preview"] = _preview(evidences)
            failures_q.append(record)
        elif status == "weak":
            record["evidence_preview"] = _preview(evidences)
            failures_d.append(record)
        elif evidences:
            record["evidence_preview"] = _preview(evidences)
            failures_b.append(record)
        else:
            record["candidates"] = _paper_candidates(record["claim_text"], items)
            failures_c.append(record)

    known_figures = {f.get("figure_id") for f in figure_items} | set(allowed_artifacts)
    failures_e = []
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", survey_md):
        alt_text, ref = match.group(1), match.group(2)
        if ref not in known_figures:
            failures_e.append({"ref": ref, "alt_text": alt_text,
                               "candidates": _figure_candidates(alt_text, ref, figure_items)})

    return {"A": failures_a, "B": failures_b, "C": failures_c,
            "D": failures_d, "E": failures_e, "F": failures_f, "Q": failures_q}


# ---------------------------------------------------------------------------
# batch decision (LLM)
# ---------------------------------------------------------------------------


def _decision_messages(group: str, records: list[dict]) -> list[dict[str, str]]:
    actions, semantics = GROUP_ACTIONS[group]
    system = (
        "You are EviSurvey's survey repair agent. Reply with exactly ONE JSON "
        'object: {"actions": [...]} — no prose. Each action is '
        '{"id": "<failure id>", "action": "<one of the allowed actions>", '
        '"params": {...}, "reason": "<short reason>"}.'
    )
    lines = [
        f"Failure type {group}: {semantics}.",
        f"Allowed actions: {actions}.",
        "Action params:",
        '- remap_citation: {"new_id": "<paper_id>"} — ONLY if the failure lists '
        "candidates (or pick an allowed paper whose title clearly matches the sentence); "
        "when no candidate fits, choose a different action instead of inventing an id.",
        '- rewrite_claim: {"new_text": "<full replacement sentence WITHOUT any [citation]>"} '
        "-- weaker assertion the evidence can support; do NOT add citations. For attribution repair, optional scope must match the supplied source roles (background stays background).",
        '- swap_evidence: {"evidence_id": "<id from evidence_preview>"}',
        "- backfill_evidence: {} — semantic search will fetch grounding chunks for the claim.",
        '- remap_figure: {"new_figure_id": "<figure_id from candidates>"}',
        '- delete_claim / delete_line: {"reason": "<why nothing simpler works>"} — LAST RESORT only.',
        '- keep: {} — for weak claims whose assertion is acceptable as-is.',
        "",
        "Failures:",
    ]
    for record in records:
        lines.append(json.dumps(record, ensure_ascii=False))
    lines.append(
        f'Return one action per failure id: {[r.get("citation_id") or r.get("claim_id") or r.get("ref") for r in records]}')
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n".join(lines)}]


def _json_chat_with_retry(llm_json_chat, messages) -> dict[str, Any]:
    """One repair retry on an unusable reply, then raise (caller marks unresolved)."""
    last_error = ""
    for attempt in (1, 2):
        try:
            reply = llm_json_chat(messages)
        except Exception as exc:  # LLM boundary
            last_error = str(exc)
            reply = None
        if isinstance(reply, dict) and isinstance(reply.get("actions"), list):
            return reply
        last_error = last_error or f"not an actions object: {reply!r}"
        if attempt == 1:
            messages = messages + [{"role": "user", "content": (
                f"Your reply was unusable ({last_error}). Reply again with exactly "
                '{"actions": [{"id", "action", "params", "reason"}, ...]}'
            )}]
    raise ValueError(f"no usable action list after retry: {last_error}")


def decide_batches(
    groups: dict[str, list[dict]],
    llm_json_chat: Callable,
    *,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    on_event: Callable[[dict], None] | None = None,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """LLM batch decisions per failure type. Returns ({group: [decisions]}, stats).

    Gate-relevant groups go first (A, B, C, E); D (quality-only) is dropped
    first when the LLM budget runs out. Unusable batches are marked unresolved.
    """
    decisions: dict[str, list[dict]] = {}
    stats = {"llm_calls": 0, "unresolved_batches": 0}
    for group in ["A", "F", "B", "C", "E", "D", "Q"]:
        records = groups.get(group, [])
        decisions[group] = []
        for start in range(0, len(records), BATCH_SIZE):
            chunk = records[start:start + BATCH_SIZE]
            if stats["llm_calls"] >= max_llm_calls:
                for record in chunk:
                    decisions[group].append({"record": record, "action": None,
                                             "outcome": "unresolved",
                                             "reason": "llm budget exhausted"})
                stats["unresolved_batches"] += 1
                continue
            stats["llm_calls"] += 1
            try:
                reply = _json_chat_with_retry(llm_json_chat, _decision_messages(group, chunk))
            except (ValueError, RuntimeError) as exc:
                logger.warning(f"[repair] group {group} batch unusable -> unresolved: {exc}")
                for record in chunk:
                    decisions[group].append({"record": record, "action": None,
                                             "outcome": "unresolved", "reason": str(exc)[:200]})
                stats["unresolved_batches"] += 1
                if on_event:
                    on_event({"agent": "repair", "event": "batch_unresolved",
                              "group": group, "error": str(exc)[:200]})
                continue
            by_id = {str(a.get("id")): a for a in reply["actions"] if isinstance(a, dict)}
            for record in chunk:
                failure_id = record.get("citation_id") or record.get("claim_id") or record.get("ref")
                decision = by_id.get(str(failure_id))
                if decision is None:
                    decisions[group].append({"record": record, "action": None,
                                             "outcome": "unresolved",
                                             "reason": "model did not answer this failure"})
                else:
                    decisions[group].append({"record": record, "decision": decision,
                                             "action": str(decision.get("action", "")),
                                             "params": decision.get("params") or {},
                                             "reason": str(decision.get("reason", ""))})
            if on_event:
                on_event({"agent": "repair", "event": "batch_decided", "group": group,
                          "n": len(chunk),
                          "actions": [d.get("action") for d in decisions[group][-len(chunk):]]})
    return decisions, stats


# ---------------------------------------------------------------------------
# action execution (mechanical, validated)
# ---------------------------------------------------------------------------


def _headings(md: str) -> list[str]:
    return re.findall(r"(?m)^## .+$", md)


def _references_tail(md: str) -> str:
    """Everything from the first ## References heading on: repair must not touch it."""
    match = re.search(r"(?m)^## References\b", md)
    return md[match.start():] if match else ""


def _invariant_violation(before: str, after: str, allowed_ids: set[str]) -> str:
    """Structural post-condition for one repair action; '' means the edit is legal.

    A repair pass may never change the set of '## ' section headings, never
    write content after '## References', and never introduce a citation id
    outside the whitelist (it may only shrink the set of offending ids).
    """
    if _headings(before) != _headings(after):
        return "section heading set changed"
    if _references_tail(before) != _references_tail(after):
        return "content after ## References changed"
    injected = _extract_citations(after) - _extract_citations(before) - allowed_ids
    if injected:
        return f"introduced non-whitelisted citation ids: {sorted(injected)[:3]}"
    return ""


class _SciverseBudget:
    def __init__(self, client, limit: int):
        self.client = client
        self.limit = limit
        self.calls = 0

    def agentic_search(self, query, top_k=3):
        if self.client is None or self.calls >= self.limit:
            raise RuntimeError("sciverse budget exhausted")
        self.calls += 1
        return self.client.agentic_search(query=query, top_k=top_k)

    def read_full_text(self, doc_id, max_pages=8):
        if self.client is None or self.calls >= self.limit:
            raise RuntimeError("sciverse budget exhausted")
        self.calls += 1
        return self.client.read_full_text(doc_id, max_pages=max_pages)


def _claim_status(claim_text: str, evidence_texts: list[str], nli) -> str:
    """Incremental re-verification: supported | weak | unsupported."""
    if not evidence_texts:
        return "unsupported"
    label = nli.best_match(claim_text, evidence_units(evidence_texts)).label
    return "supported" if label == "entailment" else "weak" if label == "neutral" else "unsupported"


def _strip_citation(sentence: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", sentence).strip()


def execute_decisions(
    decisions: dict[str, list[dict]],
    *,
    survey_md: str,
    evidence_store: dict[str, Any],
    allowed_ids: set[str],
    nli,
    sciverse_budget: _SciverseBudget,
    paper_titles: dict[str, str] | None = None,
    paper_doc_ids: dict[str, str] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply validated actions, writing each outcome back into its decision item.

    Returns (mutated markdown, mutated evidence_store); the caller builds
    repair_log from the decision items. paper_titles (paper_id -> title, from
    the ready set) feeds the backfill provenance gate's title fallback.
    """
    md = survey_md
    evidence_store = copy.deepcopy(evidence_store)
    paper_titles = paper_titles or {}
    paper_doc_ids = paper_doc_ids or {}
    evidence_by_paper: dict[str, list[dict]] = {}
    for evidence in evidence_store.get("evidence", []):
        evidence_by_paper.setdefault(evidence.get("paper_id", ""), []).append(evidence)

    for group, items in decisions.items():
        for item in items:
            action = item.get("action")
            if action is None:  # unresolved during decision phase
                continue
            before = md
            if action not in GROUP_ACTIONS[group][0].split(", "):
                item["outcome"] = "invalid_action"
                continue
            outcome, md = _execute_one(
                action, item.get("params", {}), item["record"], md, evidence_store,
                evidence_by_paper, allowed_ids, nli, sciverse_budget, paper_titles,
                paper_doc_ids)
            violation = _invariant_violation(before, md, allowed_ids)
            if violation:
                # fail per action, not per run: revert only this mutation
                target = (item["record"].get("citation_id")
                          or item["record"].get("claim_id") or item["record"].get("ref"))
                logger.warning(f"[repair] reverted {action} on {target}: {violation}")
                outcome, md = "invalid_action", before
            item["outcome"] = outcome
            if on_event:
                on_event({"agent": "repair", "event": "action", "failure_type": group,
                          "id": item["record"].get("citation_id")
                               or item["record"].get("claim_id") or item["record"].get("ref"),
                          "action": action, "reason": item.get("reason", ""),
                          "outcome": outcome})
    _warn_deletion_ratio([item for items in decisions.values() for item in items])
    return md, evidence_store


_ID_SHAPED_RE = re.compile(r"paper:|_|/")


def _execute_one(action, params, record, md, evidence_store,
                 evidence_by_paper, allowed_ids, nli, sciverse_budget,
                 paper_titles, paper_doc_ids) -> tuple[str, str]:
    """Execute one validated action; returns (outcome, mutated markdown)."""
    try:
        if action == "remap_citation":
            new_id = str(params.get("new_id", ""))
            if new_id not in allowed_ids:
                return "invalid_action", md
            old_id = record.get("citation_id") or record.get("cited_paper_id", "")
            # Academic numeric markers ([1], [31-36]) are not citation ids:
            # remapping them onto a confabulated whitelist paper mangles the
            # document (S0 round-2/3: blind global replaces spliced DOIs and
            # rendered sections into surviving citations). Only id-shaped
            # sources may remap; anything else goes to delete_claim.
            if not _ID_SHAPED_RE.search(str(old_id)):
                return "invalid_action", md
            sentences = [s for s in _sentences(md) if f"[{old_id}]" in s]
            if not sentences:
                return "invalid_action", md
            for sentence in sentences:
                remapped = sentence.replace(f"[{old_id}]", f"[{new_id}]", 1)
                md = md.replace(sentence, remapped, 1)
            repaired = all(
                _claim_status(_strip_citation(s), [e["text"] for e in evidence_by_paper.get(new_id, [])], nli)
                != "unsupported" for s in sentences)
            return ("repaired" if repaired else "still_unsupported"), md

        if action == "rewrite_claim":
            new_text = str(params.get("new_text", "")).strip()
            old_sentence = _match_claim_sentence(md, record)
            if not old_sentence or "[" in new_text or not (
                    0.5 * len(old_sentence) <= len(new_text) <= 2 * len(old_sentence) + 20):
                return "invalid_action", md
            cited_id = record.get("cited_paper_id", "")
            # keep the sentence-ending punctuation so adjacent sentences don't fuse
            tail = old_sentence[-1] if old_sentence and old_sentence[-1] in ".。!?" else ""
            actual_cites = [pid for _, cites in sentence_units(old_sentence, allowed_ids) for pid in cites]
            tags = " ".join(f"[{pid}]" for pid in dict.fromkeys(actual_cites or [cited_id]) if pid)
            replacement = f"{new_text.rstrip('.。')} {tags}{tail or '.'}"
            md = md.replace(old_sentence, replacement, 1)
            evidences = evidence_by_paper.get(cited_id, [])
            bindings = record.get("source_bindings", [])
            if bindings:
                evidences = [e for e in evidences if e.get("evidence_id") in {b.get("evidence_id") for b in bindings}]
            premises = [b["source_quote"] for b in bindings] if bindings else [e["text"] for e in evidences]
            status = _claim_status(new_text, premises, nli)
            for other_id in set(actual_cites) - {cited_id}:
                if _claim_status(new_text, [e["text"] for e in evidence_by_paper.get(other_id, [])], nli) != "supported":
                    status = "unsupported"
            # Preserve claim scope/source provenance through the outer reverify.
            new_scope = params.get("scope", record.get("scope", ""))
            role_ok = (not record.get("source_role_violation") or
                       record.get("source_role_violation") == "source_role_scope_mismatch")
            if new_scope != record.get("scope", "") and not bindings:
                return "invalid_action", md.replace(replacement, old_sentence, 1)
            for binding in bindings:
                ev = next((e for e in evidences if e.get("evidence_id") == binding.get("evidence_id")), None)
                role_ok = role_ok and ev is not None and valid_support(binding, ev)
                role_ok = role_ok and binding.get("source_role") in SCOPE_ROLES.get(new_scope, set())
            record["rewritten_text"] = new_text.rstrip(".。")
            record["rewritten_scope"] = new_scope
            copied = quote_diagnostic(new_text, [e["text"] for e in evidences])["quote_like"]
            return ("repaired" if status == "supported" and role_ok and not copied else "still_unsupported"), md

        if action == "swap_evidence":
            evidence_id = str(params.get("evidence_id", ""))
            evs = {e.get("evidence_id"): e for e in evidence_by_paper.get(record.get("cited_paper_id", ""), [])}
            if evidence_id not in evs:
                return "invalid_action", md
            status = _claim_status(record["claim_text"], [evs[evidence_id]["text"]], nli)
            return ("repaired" if status != "unsupported" else "still_unsupported"), md

        if action == "backfill_evidence":
            cited_id = record.get("cited_paper_id", "")
            # own-paper fulltext first: identity by construction (the doc_id
            # channel), no gate needed. Falls through to the gated agentic path
            # when there is no doc_id or the fulltext does not support the claim.
            doc_id = paper_doc_ids.get(cited_id, "")
            if doc_id:
                try:
                    own_md = sciverse_budget.read_full_text(doc_id)
                except RuntimeError:
                    own_md = ""
                own_text = own_md.strip()[:30000]  # one row; verifiers window it downstream
                if own_text and _claim_status(record["claim_text"], evidence_units([own_text]), nli) != "unsupported":
                    evidence_store.setdefault("evidence", []).append({
                        "evidence_id": make_evidence_id(
                            cited_id, None, len(evidence_store["evidence"]), "repair"),
                        "paper_id": cited_id, "source_type": "content_chunk",
                        "source_page": None, "source_paragraph_index": 0,
                        "text": own_text, "supports_claims": [],
                        "source_doc_id": doc_id,
                        "source_title": paper_titles.get(cited_id, ""),
                        "source_chunk_id": f"{doc_id}:fulltext",
                    })
                    evidence_by_paper.setdefault(cited_id, []).append(
                        evidence_store["evidence"][-1])
                    return "repaired", md
            try:
                hits = sciverse_budget.agentic_search(record["claim_text"], top_k=3).get("hits", [])
            except RuntimeError:
                return "unresolved", md
            cited_id = record.get("cited_paper_id", "")
            accepted = []
            for hit in hits:
                chunk = str(hit.get("chunk", "")).strip()
                if not chunk:
                    continue
                # Provenance gate: the chunk must belong to the cited paper
                # (doc_id for DOI ids, title otherwise); off-topic hits are
                # rejected instead of renamed onto the cited paper.
                if not hit_matches_paper(cited_id, paper_titles.get(cited_id, ""), hit):
                    logger.warning(
                        f"[repair] provenance gate rejected hit for {cited_id}: "
                        f"doc_id={hit.get('doc_id')!r} title={str(hit.get('title'))[:80]!r}")
                    continue
                accepted.append((hit, chunk))
            if not accepted:
                return "still_unsupported", md
            for hit, chunk in accepted:
                page_no = hit.get("page_no")
                evidence_store.setdefault("evidence", []).append({
                    "evidence_id": make_evidence_id(
                        cited_id, page_no, len(evidence_store["evidence"]), "repair"),
                    "paper_id": cited_id, "source_type": "agentic_chunk",
                    "source_page": int(page_no) if page_no is not None else None,
                    "source_paragraph_index": int(hit.get("offset") or 0),
                    "text": chunk, "supports_claims": [],
                    "source_doc_id": str(hit.get("doc_id") or ""),
                    "source_title": str(hit.get("title") or ""),
                    "source_chunk_id": chunk_id(hit),
                })
                evidence_by_paper.setdefault(cited_id, []).append(
                    evidence_store["evidence"][-1])
            status = _claim_status(record["claim_text"], [chunk for _, chunk in accepted], nli)
            return ("repaired" if status != "unsupported" else "still_unsupported"), md

        if action == "delete_claim":
            if not str(params.get("reason", "")).strip():
                return "invalid_action", md
            target = _match_claim_sentence(md, record) or \
                _sentence_containing(md, f"[{record.get('citation_id', '__none__')}]")
            if not target:
                return "invalid_action", md
            return "deleted", md.replace(target, "", 1)

        if action == "keep":
            return "kept", md

        if action == "remap_figure":
            new_figure_id = str(params.get("new_figure_id", ""))
            if new_figure_id not in {c["figure_id"] for c in record.get("candidates", [])}:
                return "invalid_action", md
            return "repaired", md.replace(f"]({record['ref']})", f"]({new_figure_id})")

        if action == "delete_line":
            if not str(params.get("reason", "")).strip():
                return "invalid_action", md
            lines = [line for line in md.splitlines() if record["ref"] in line]
            if not lines:
                return "invalid_action", md
            return "deleted", md.replace(lines[0] + "\n", "", 1)

        return "invalid_action", md
    except Exception as exc:  # action executor boundary: LLM params are untrusted input
        logger.warning(f"[repair] action {action} failed on {record}: {exc}")
        return "invalid_action", md


def _match_claim_sentence(md: str, record: dict) -> str:
    """The sentence carrying this claim: same matching rule as the old remover."""
    claim_text = str(record.get("claim_text", "")).strip()
    cited_id = record.get("cited_paper_id", "")
    if not claim_text or not cited_id:
        return ""
    normalized = " ".join(claim_text.split())[:60]
    for sentence in _sentences(md):
        if f"[{cited_id}]" in sentence and normalized in " ".join(sentence.split()):
            return sentence
    return ""


def _warn_deletion_ratio(log: list[dict[str, Any]]) -> None:
    resolved = [e for e in log if e["outcome"] not in {"unresolved", "invalid_action", ""}]
    deleted = sum(1 for e in resolved if e["outcome"] == "deleted")
    if resolved and deleted / len(resolved) > DELETE_RATIO_WARN:
        logger.warning(
            f"[repair] deletion ratio {deleted}/{len(resolved)} > {DELETE_RATIO_WARN:.0%} "
            "-- repair degenerated toward deletion; inspect repair_log")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_repair(
    *,
    task_id: str,
    round_no: int,
    survey_md: str,
    claim_map: dict[str, Any],
    evidence_store: dict[str, Any],
    ready_set: dict[str, Any],
    figure_items: list[dict],
    allowed_artifacts: set[str],
    llm_json_chat: Callable,
    nli,
    sciverse=None,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    max_sciverse_calls: int = DEFAULT_MAX_SCIVERSE_CALLS,
    trajectory_dir: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    trajectory_dir = Path(trajectory_dir) if trajectory_dir is not None else Path("logs/trajectory")
    log_line = trajectory_writer(trajectory_dir / f"{task_id}_repair.jsonl")

    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    evidence_by_paper: dict[str, list[dict]] = {}
    for evidence in evidence_store.get("evidence", []):
        evidence_by_paper.setdefault(evidence.get("paper_id", ""), []).append(evidence)
    claim_map = dict(claim_map)
    claim_map["_evidence_by_paper"] = evidence_by_paper  # grouping helper input

    groups = group_failures(survey_md, claim_map, ready_set, figure_items, allowed_artifacts)
    total = sum(len(v) for v in groups.values())
    log_line({"agent": "repair", "event": "start", "task_id": task_id, "round": round_no,
              "failures": {g: len(v) for g, v in groups.items()}, "total": total})
    logger.info(f"[repair] round {round_no}: {total} failures "
                f"{ {g: len(v) for g, v in groups.items()} }")

    decisions, stats = decide_batches(groups, llm_json_chat, max_llm_calls=max_llm_calls,
                                      on_event=log_line)
    sciverse_budget = _SciverseBudget(sciverse, max_sciverse_calls)
    ready_items = [item for item in ready_set.get("items", []) if isinstance(item, dict)]
    paper_titles = {str(item.get("paper_id")): str(item.get("title") or "") for item in ready_items}
    paper_doc_ids = {str(item.get("paper_id")): str(item.get("doc_id") or "") for item in ready_items}
    md, mutated_evidence = execute_decisions(
        decisions, survey_md=survey_md, evidence_store=evidence_store,
        allowed_ids=allowed_ids, nli=nli,
        sciverse_budget=sciverse_budget, paper_titles=paper_titles,
        paper_doc_ids=paper_doc_ids, on_event=log_line)

    repair_log = []
    for group, items in decisions.items():
        for item in items:
            repair_log.append({
                "round": round_no, "failure_type": group,
                "id": item["record"].get("citation_id") or item["record"].get("claim_id")
                      or item["record"].get("ref"),
                "action": item.get("action"),
                "reason": item.get("reason", ""),
                "outcome": item.get("outcome", ""),
                "claim_text": item["record"].get("claim_text", ""),
                "rewritten_text": item["record"].get("rewritten_text", ""),
                "rewritten_scope": item["record"].get("rewritten_scope", ""),
                "cited_paper_id": item["record"].get("cited_paper_id", ""),
                "source_bindings": item["record"].get("source_bindings", []),
                "scope": item["record"].get("scope", ""),
            })

    outcomes: dict[str, int] = {}
    for entry in repair_log:
        key = entry["outcome"] or "unresolved"
        outcomes[key] = outcomes.get(key, 0) + 1
    metrics = {"failures": total, "llm_calls": stats["llm_calls"],
               "sciverse_calls": sciverse_budget.calls,
               "elapsed_sec": round(time.monotonic() - started, 1), **outcomes}
    log_line({"agent": "repair", "event": "finish", "task_id": task_id, "round": round_no,
              "metrics": metrics})
    logger.info(f"[repair] round {round_no} done: {metrics}")
    return {"revised_md": md, "evidence_store": mutated_evidence,
            "repair_log": repair_log, "metrics": metrics}
