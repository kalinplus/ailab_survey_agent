import logging
from tools.clients.mineru_client import _split_markdown
from tools.models.artifacts import Evidence, EvidenceStore, Claim
from tools.models.common import evidence_id
from tools.verify.provenance import chunk_id, hit_matches_paper
from tools.verify.source_contract import SUPPORTED_TYPES
from tools.verify.text_units import normalize_text

logger = logging.getLogger(__name__)


def run(task_id, parsed_papers, paper_cards, nli, sciverse=None):
    logger.info(f"[P5.2] evidence: {len(parsed_papers.papers)} parsed, {len(paper_cards.paper_cards)} cards")
    card_by_pid = {c.paper_id: c for c in paper_cards.paper_cards}
    evidences = []
    for p in parsed_papers.papers:
        card = card_by_pid.get(p.paper_id)
        claim_texts = [cl for bucket in card.possible_claims.values() for cl in bucket] if card else []
        # abstract as one evidence
        if p.abstract:
            evidences.append(_make(p.paper_id, 0, 0, p.abstract, "abstract", claim_texts, nli))
        for para in p.paragraphs:
            # Bibliography blocks restate OTHER papers' results; they must never
            # count as this paper's own evidence (role absent -> old behavior).
            if para.role == "reference":
                continue
            evidences.append(_make(p.paper_id, para.page, para.index, para.text, "paragraph", claim_texts, nli))
        for fig in p.figures:
            if fig.get("caption"):
                evidences.append(_make(p.paper_id, fig.get("page", 0), fig["num"], fig["caption"], "caption", claim_texts, nli))
    # Shallow cards retain the actual supplied abstract source; no invented
    # paragraph/page link and no dependency on a later search to recover it.
    present = {e.evidence_id for e in evidences}
    for card in paper_cards.paper_cards:
        claims = [cl for bucket in card.possible_claims.values() for cl in bucket]
        for block in card.source_blocks:
            if block["paper_id"] != card.paper_id or block["evidence_id"] in present:
                continue
            ev = _make(card.paper_id, block["source_page"], block["source_paragraph_index"],
                       block["text"], block["source_type"], claims, nli)
            if ev.evidence_id != block["evidence_id"]:
                raise ValueError("card source identity disagrees with supplied block")
            evidences.append(ev)
            present.add(ev.evidence_id)
    for ev in evidences:
        card = card_by_pid.get(ev.paper_id)
        if card:
            ev.source_doc_id = card.doc_id
            ev.source_title = card.title
            ev.source_chunk_id = ev.evidence_id
    logger.info(f"[P5.2] parsed evidence: {len(evidences)}")

    if sciverse is not None:
        # Own-paper fulltext grounding first: /content fetches the cited paper's
        # OWN text by doc_id — identity by construction, no gate needed. This is
        # the MinerU-independent backbone agentic-search could never provide
        # (it retrieves topically RELATED papers, never the paper itself).
        content_parsed = {p.paper_id for p in parsed_papers.papers
                          if p.parse_status == "content_fulltext"}
        evidences = _content_backfill(paper_cards, evidences, sciverse, nli,
                                      already_parsed=content_parsed)
        # agentic-search backfill: gated provenance (doc_id/title must verify);
        # most claims are already grounded by content chunks, so this rarely fires.
        evidences = _agentic_backfill(paper_cards, evidences, sciverse, nli)

    logger.info(f"[P5.2] evidence total: {len(evidences)}")
    return EvidenceStore(task_id=task_id, evidence=evidences)


def _content_backfill(paper_cards, evidences, sciverse, nli, already_parsed=frozenset()):
    supported = {(e.paper_id, s["claim_text"]) for e in evidences for s in e.supports_claims
                 if s.get("support_type") in SUPPORTED_TYPES}
    n_fetched = 0
    n_added = 0
    for card in paper_cards.paper_cards:
        if card.paper_id in already_parsed:
            continue  # its /content fulltext already entered as parsed paragraphs
        claim_texts = [cl for bucket in card.possible_claims.values() for cl in bucket
                       if (card.paper_id, cl.text) not in supported]
        if not claim_texts or not card.doc_id:
            continue  # nothing to ground, or no identity link to fetch by
        try:
            md = sciverse.read_full_text(card.doc_id)
        except Exception as e:
            # external API boundary: skip this paper, parsed evidence still stands
            logger.warning(f"[P5.2] content fetch failed for {card.paper_id} (doc_id={card.doc_id[:16]}…): {e}")
            continue
        n_fetched += 1
        if not md.strip():
            continue
        # same markdown kernel as MinerU's fallback channel: page=None (unknown),
        # paper-global block ids — content Markdown carries no page numbers
        _sections, flat, _title = _split_markdown(md)
        for para in flat:
            if para.get("role") == "reference":
                continue
            ev = _make(card.paper_id, para["page"], para["index"], para["text"],
                       "content_chunk", claim_texts, nli)
            ev.source_doc_id = card.doc_id
            ev.source_title = card.title
            ev.source_chunk_id = f"{card.doc_id}:b{para['index']}"
            evidences.append(ev)
            n_added += 1
            for s in ev.supports_claims:
                # a grounded claim drops out of later fetches only within this run
                supported.add((card.paper_id, s["claim_text"]))
    logger.info(f"[P5.2] content backfill: {n_fetched} fulltexts fetched, +{n_added} content chunks")
    return evidences


def _agentic_backfill(paper_cards, evidences, sciverse, nli):
    supported = {(e.paper_id, s["claim_text"]) for e in evidences for s in e.supports_claims
                 if s.get("support_type") in SUPPORTED_TYPES}
    n_skipped_declared = 0
    n_searches = 0
    n_added = 0
    n_rejected = 0
    for card in paper_cards.paper_cards:
        for bucket in card.possible_claims.values():
            for claim in bucket:
                if (card.paper_id, claim.text) in supported:
                    continue  # already grounded by a parsed-paper fragment
                if claim.evidence_ids:
                    # Declared claims are supportable only by their own bound
                    # blocks (valid_support requires evidence_id membership), so
                    # agentic chunks can never ground them - measured dead
                    # weight: 10 min, 162 searches, 0 support rows (wave6 run).
                    n_skipped_declared += 1
                    continue
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
                    # Provenance gate: a hit belongs under this paper id only when
                    # its doc_id (DOI paper ids) or title (non-DOI ids) identifies
                    # the same paper. Off-topic hits are rejected, never renamed.
                    if not hit_matches_paper(card.paper_id, card.title, hit):
                        n_rejected += 1
                        logger.warning(
                            f"[P5.2] provenance gate rejected hit for {card.paper_id}: "
                            f"doc_id={hit.get('doc_id')!r} title={str(hit.get('title'))[:80]!r}")
                        continue
                    # NLI direction: claim is the claim, the fetched chunk is evidence.
                    res = nli.best_match(claim.text, [chunk])
                    if res.support_type not in SUPPORTED_TYPES:
                        continue
                    page_no = hit.get("page_no")
                    evidences.append(Evidence(
                        evidence_id=evidence_id(card.paper_id, page_no, n_added, "agentic_chunk"),
                        paper_id=card.paper_id, source_type="agentic_chunk",
                        source_page=int(page_no) if page_no is not None else None,
                        source_paragraph_index=int(hit.get("offset") or 0),
                        text=chunk,
                        supports_claims=_support_rows(claim, res, card.paper_id,
                            evidence_id(card.paper_id, page_no, n_added, "agentic_chunk"), chunk),
                        source_doc_id=str(hit.get("doc_id") or ""),
                        source_title=str(hit.get("title") or ""),
                        source_chunk_id=chunk_id(hit),
                    ))
                    n_added += 1
    summary = (f"[P5.2] agentic backfill: {n_searches} searches, +{n_added} evidences "
               f"({n_skipped_declared} declared claims skipped - ungroundable by agentic chunks)")
    if n_rejected:
        logger.warning(f"{summary}, {n_rejected} hits rejected by provenance gate")
    else:
        logger.info(summary)
    return evidences


def _support_rows(claim, result, paper_id, eid, text):
    if result.support_type not in SUPPORTED_TYPES:
        return []
    if claim.evidence_ids and eid not in claim.evidence_ids:
        return []
    if claim.source_quote and normalize_text(claim.source_quote) not in normalize_text(text):
        return []
    return [{"paper_id": paper_id, "claim_text": claim.text,
             "dimension": claim.dimension, "source_role": claim.source_role,
             "source_quote": claim.source_quote, "evidence_ids": [eid],
             "support_type": result.support_type, "confidence": result.confidence}]


def _make(paper_id, page, idx, text, source_type, claims, nli):
    eid = evidence_id(paper_id, page, idx, source_type)
    supports = []
    for claim in claims:
        if isinstance(claim, str):
            claim = Claim(text=claim, dimension="unknown")
        if claim.evidence_ids and eid not in claim.evidence_ids:
            continue
        res = nli.best_match(claim.text, [claim.source_quote or text])
        supports.extend(_support_rows(claim, res, paper_id, eid, text))
    return Evidence(evidence_id=eid, paper_id=paper_id,
        source_type=source_type, source_page=page, source_paragraph_index=idx,
        text=text, supports_claims=supports)
