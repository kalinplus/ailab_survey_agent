from tools.models.artifacts import Evidence, EvidenceStore
from tools.models.common import evidence_id


def run(task_id, parsed_papers, paper_cards, nli):
    card_by_pid = {c.paper_id: c for c in paper_cards.paper_cards}
    evidences = []
    for p in parsed_papers.papers:
        card = card_by_pid.get(p.paper_id)
        claim_texts = [cl.text for bucket in card.possible_claims.values() for cl in bucket] if card else []
        # abstract as one evidence
        if p.abstract:
            evidences.append(_make(p.paper_id, 0, 0, p.abstract, "abstract", claim_texts, nli))
        for para in p.paragraphs:
            evidences.append(_make(p.paper_id, para.page, para.index, para.text, "paragraph", claim_texts, nli))
        for fig in p.figures:
            if fig.get("caption"):
                evidences.append(_make(p.paper_id, fig.get("page", 0), fig["num"], fig["caption"], "caption", claim_texts, nli))
    return EvidenceStore(task_id=task_id, evidence=evidences)


def _make(paper_id, page, idx, text, source_type, claim_texts, nli):
    supports = []
    if claim_texts:
        res = nli.best_match(text, claim_texts)  # NLIResult
        if res.support_type != "contradictory" or res.confidence > 0:
            supports.append({"claim_text": claim_texts[0], "support_type": res.support_type, "confidence": res.confidence})
    return Evidence(evidence_id=evidence_id(paper_id, page, idx), paper_id=paper_id,
        source_type=source_type, source_page=page, source_paragraph_index=idx,
        text=text, supports_claims=supports)
