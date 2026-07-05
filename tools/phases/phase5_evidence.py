import logging
from tools.models.artifacts import Evidence, EvidenceStore
from tools.models.common import evidence_id, paper_id_from_seed

logger = logging.getLogger(__name__)


def run(task_id, parsed_papers, paper_cards, nli, sciverse=None):
    logger.info(f"[P5.2] evidence: {len(parsed_papers.papers)} parsed, {len(paper_cards.paper_cards)} cards")
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
    logger.info(f"[P5.2] parsed evidence: {len(evidences)}")

    # agentic-search backfill: ground claims with no parsed-paper evidence against real SciVerse
    # chunks (chunk + page_no + doc_id). MinerU-independent anti-hallucination backbone.
    if sciverse is not None:
        evidences = _agentic_backfill(paper_cards, evidences, sciverse, nli)

    logger.info(f"[P5.2] evidence total: {len(evidences)}")
    return EvidenceStore(task_id=task_id, evidence=evidences)


def _agentic_backfill(paper_cards, evidences, sciverse, nli):
    supported = {s["claim_text"] for e in evidences for s in e.supports_claims}
    n_searches = 0
    n_added = 0
    for card in paper_cards.paper_cards:
        for bucket in card.possible_claims.values():
            for claim in bucket:
                if claim.text in supported:
                    continue  # already grounded by a parsed-paper fragment
                try:
                    hits = sciverse.agentic_search(claim.text, top_k=3).get("hits", [])
                    n_searches += 1
                except Exception as e:
                    logger.warning(f"[P5.2] agentic_search failed: {e}")
                    hits = []  # external API boundary: skip this claim, parsed evidence still stands
                for hit in hits:
                    chunk = hit.get("chunk") or ""
                    if not chunk:
                        continue
                    res = nli.best_match(chunk, [claim.text])
                    if res.support_type == "contradictory" and res.confidence <= 0:
                        continue
                    pid = paper_id_from_seed(hit.get("title", ""), int(hit.get("publication_published_year") or 0))
                    evidences.append(Evidence(
                        evidence_id=f"{pid[:16]}_p{hit.get('page_no', 0)}_{hit.get('offset', 0)}",
                        paper_id=pid, source_type="agentic_chunk",
                        source_page=int(hit.get("page_no") or 0),
                        source_paragraph_index=int(hit.get("offset") or 0),
                        text=chunk,
                        supports_claims=[{"claim_text": claim.text, "support_type": res.support_type, "confidence": res.confidence}],
                    ))
                    n_added += 1
    logger.info(f"[P5.2] agentic backfill: {n_searches} searches, +{n_added} evidences")
    return evidences


def _make(paper_id, page, idx, text, source_type, claim_texts, nli):
    supports = []
    if claim_texts:
        res = nli.best_match(text, claim_texts)  # NLIResult
        if res.support_type != "contradictory" or res.confidence > 0:
            supports.append({"claim_text": claim_texts[0], "support_type": res.support_type, "confidence": res.confidence})
    return Evidence(evidence_id=evidence_id(paper_id, page, idx), paper_id=paper_id,
        source_type=source_type, source_page=page, source_paragraph_index=idx,
        text=text, supports_claims=supports)
