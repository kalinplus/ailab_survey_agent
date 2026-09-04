"""NLI-first claim mapper with LLM fallback.

Given survey markdown, extracts claim sentences (each ending with a
[paper_id] citation), then grounds each claim in 3 stages:

1. Exact direct EvidenceStore match (supports_claims with support_type=="direct")
2. NLI entailment against the cited paper's evidence
3. LLM fallback for the 0.4-0.6 confidence band
"""

import re

from tools.models.bundle import ClaimMap, ClaimEntry


def extract_claims_with_citations(md: str):
    """Split markdown into sentences; keep those containing a [citation].

    Image embeds (`![caption](artifact_id)`) are dropped line-wise first: the
    square brackets inside `![...]` would otherwise be extracted as a citation
    whose id is the caption text (S0: claims citing "Publication years of ...").
    Returns list of (claim_text_without_brackets, first_citation_id).
    """
    prose = "\n".join(
        line for line in md.splitlines() if not line.lstrip().startswith("![")
    )
    out = []
    for sentence in re.split(r"(?<=[.。])\s+", prose):
        cites = re.findall(r"\[([^\]]+)\]", sentence)
        if cites:
            text = re.sub(r"\[[^\]]+\]", "", sentence).strip().rstrip(".。").strip()
            if text:
                out.append((text, cites[0]))
    return out


def llm_fallback(llm, claim, evidence):
    """Ask the LLM to judge support; return status string."""
    prompt = (
        f"Does this evidence support the claim? "
        f"Answer one word: supported, weak, or unsupported.\n"
        f"Evidence: {evidence}\nClaim: {claim}"
    )
    out = llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
    low = out.lower()
    if "unsupported" in low:
        return "unsupported"
    if "weak" in low:
        return "weak"
    if "support" in low:
        return "supported"
    return "unsupported"


def run(task_id, survey_md, evidence_store, parsed_papers, nli, llm=None):
    """Build a ClaimMap from survey markdown using NLI-first grounding.

    parsed_papers is accepted for interface compatibility but unused.
    """
    by_paper: dict[str, list] = {}
    for e in evidence_store.evidence:
        by_paper.setdefault(e.paper_id, []).append(e)

    entries = []
    for claim_text, paper_id in extract_claims_with_citations(survey_md):
        evidences = by_paper.get(paper_id, [])

        # Stage 1: exact direct match
        direct = any(
            any(sc.get("support_type") == "direct" for sc in e.supports_claims)
            for e in evidences
        )
        if direct:
            status, conf, eids = "supported", 1.0, [e.evidence_id for e in evidences]

        # Stage 2: NLI
        elif evidences:
            res = nli.best_match(claim_text, [e.text for e in evidences])
            eids = [e.evidence_id for e in evidences]
            if res.confidence >= 0.6:
                status = (
                    "supported" if res.label == "entailment"
                    else ("weak" if res.label == "neutral" else "unsupported")
                )
                conf = res.confidence
            elif res.confidence >= 0.4 and llm is not None:
                status = llm_fallback(llm, claim_text, evidences[0].text)
                conf = 0.5
            else:
                status = "unsupported"
                conf = res.confidence

        # Stage 3: no evidence
        else:
            status, conf, eids = "unsupported", 0.0, []

        entries.append(ClaimEntry(
            claim_text=claim_text,
            cited_paper_id=paper_id,
            status=status,
            evidence_ids=eids,
            confidence=conf,
        ))

    return ClaimMap(task_id=task_id, entries=entries)
