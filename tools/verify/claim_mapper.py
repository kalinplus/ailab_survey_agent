"""Pair-level NLI grounding with claim source contracts and independent copy diagnostics.

Every current sentence/citation pair is reverified. Structured bindings restrict
premises to actual same-paper source assertions; roles constrain attribution.
Legacy cards can be read, but a cached direct label never bypasses NLI.
"""

import logging

from tools.verify.source_contract import SCOPE_ROLES, valid_support, role_violation, quote_diagnostic

from tools.models.bundle import ClaimMap, ClaimEntry
from tools.verify.text_units import (
    CITATION,
    evidence_units,
    is_citation,
    normalize_text,
    sentence_units,
)

logger = logging.getLogger(__name__)


def extract_claims_with_citations(md: str, known_ids: set[str] | None = None):
    """Cited claim sentences from survey markdown via the shared kernel:
    structural lines (headings/images/tables/quotes) are dropped, brackets
    count as citations only when validated against known_ids, and
    citation-only fragments are merged back into the preceding sentence.

    Returns list of (claim_text_without_brackets, first_citation_id) —
    sentence granularity; the pair-level expansion lives in run().
    """
    return [(text, cites[0]) for text, cites in sentence_units(md, known_ids) if cites]


def run(task_id, survey_md, evidence_store, parsed_papers, nli, llm=None, structured_claims=None):
    """Build a pair-level ClaimMap from the FINAL survey markdown using NLI-first grounding.

    structured_claims: the writer's structured claim source (the "claims"
    entries of cache/structured_claims.json: {"text": str, "cites": [paper_id,
    ...], ...}). Freshness gate: only entries still present in the final
    markdown (normalized text match) survive — repair rounds rewrite or delete
    sentences and the persisted list goes stale. The shared kernel then
    re-extracts every cited sentence the survivors do not already cover, so
    abstract/intro/closing freeform citations are verified too.

    parsed_papers is accepted for interface compatibility but unused.
    """
    by_paper: dict[str, list] = {}
    for e in evidence_store.evidence:
        by_paper.setdefault(e.paper_id, []).append(e)
    known_ids = set(by_paper)

    kernel_units = [(text, cites) for text, cites in sentence_units(survey_md, known_ids) if cites]
    kernel_pairs = {(normalize_text(text), tuple(cites)) for text, cites in kernel_units}

    if structured_claims:
        fresh = [
            (str(claim.get("text", "")), [str(pid) for pid in claim.get("cites") or []])
            for claim in structured_claims
        ]
        fresh = [(text, cites) for text, cites in fresh
                 if text and cites and (normalize_text(text), tuple(cites)) in kernel_pairs]
        fresh_pairs = {(normalize_text(text), tuple(cites)) for text, cites in fresh}
        kernel_extra = [(text, cites) for text, cites in kernel_units
                        if (normalize_text(text), tuple(cites)) not in fresh_pairs]
        units = fresh + kernel_extra
        logger.info(
            "[claim_map] structured_claims: %d listed, %d still present in the final "
            "markdown, %d kernel-extracted units added",
            len(structured_claims), len(fresh), len(kernel_extra))
    else:
        units = kernel_units

    # Pair expansion: one entry per (claim, cited paper) — a sentence citing
    # [A, B] is judged against A's and B's evidence separately.
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for claim_text, cites in units:
        for paper_id in cites:
            key = (normalize_text(claim_text), paper_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((claim_text, paper_id))

    strict_sources = any("source_bindings" in c for c in structured_claims or [])
    metadata = {(normalize_text(c.get("text", "")), pid): c
                for c in structured_claims or [] for pid in c.get("cites", [])}
    entries = []
    for claim_text, paper_id in pairs:
        evidences = by_paper.get(paper_id, [])

        row = metadata.get((normalize_text(claim_text), paper_id), {})
        bindings = [b for b in row.get("source_bindings", []) if b.get("paper_id") == paper_id]
        scope = row.get("scope", "")
        violation = ""
        if strict_sources or row.get("source_bindings") is not None or scope:
            if not bindings:
                violation = "missing_source_binding"
            for binding in bindings:
                ev = next((e for e in evidences if e.evidence_id == binding.get("evidence_id")), None)
                if (ev is None or not valid_support(binding, ev.model_dump())
                        or not any(all(s.get(key) == binding.get(key) for key in
                                       ("claim_text", "source_role", "source_quote", "evidence_ids"))
                                   for s in ev.supports_claims)):
                    violation = "invalid_source_binding"
                elif binding.get("source_role") not in SCOPE_ROLES.get(scope, set()):
                    violation = "source_role_scope_mismatch"
            selected_ids = {b.get("evidence_id") for b in bindings}
            evidences = [e for e in evidences if e.evidence_id in selected_ids]
        # Even legacy/freeform text must not launder a known background claim
        # into the paper's own method merely because a whole abstract overlaps.
        for ev in evidences:
            for support in ev.supports_claims:
                if normalize_text(support.get("claim_text", "")) == normalize_text(claim_text):
                    violation = violation or role_violation(support.get("source_role", "unknown"),
                                                            support.get("source_quote", ""))
        if evidences:
            premises = [b["source_quote"] for b in bindings] if bindings and not violation else [e.text for e in evidences]
            res = nli.best_match(claim_text, evidence_units(premises))
            eids = [e.evidence_id for e in evidences]
            nli_status = ("supported" if res.label == "entailment"
                          else "weak" if res.label == "neutral" else "unsupported")
            conf = res.confidence
        else:
            nli_status, conf, eids = "unsupported", 0.0, []
        status = "unsupported" if violation else nli_status
        diagnostic = quote_diagnostic(claim_text, [e.text for e in evidences])

        entries.append(ClaimEntry(
            claim_text=claim_text,
            cited_paper_id=paper_id,
            status=status,
            evidence_ids=eids,
            confidence=conf, nli_status=nli_status, source_role_violation=violation,
            source_bindings=bindings, scope=scope, quote_diagnostic=diagnostic,
        ))

    _warn_uncovered_citations(survey_md, known_ids, {paper_id for _, paper_id in pairs})
    return ClaimMap(task_id=task_id, entries=entries)


def _warn_uncovered_citations(survey_md: str, known_ids: set[str], covered: set[str]) -> None:
    """Invariant: every legal citation id in the final markdown must be covered
    by at least one claim unit. Structural lines (headings) are invisible to
    the kernel, so violations are possible — surfaced, not silently dropped."""
    legal = {c for c in CITATION.findall(survey_md) if is_citation(c, known_ids)}
    uncovered = sorted(legal - covered)
    if uncovered:
        logger.warning(
            "[claim_map] %d legal citation ids in the markdown are covered by no "
            "claim unit: %s", len(uncovered), uncovered[:5])
