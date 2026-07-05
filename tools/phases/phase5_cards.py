import logging
import re
from tools.models.artifacts import PaperCard, PaperCards, Claim
from tools.models.common import evidence_id

logger = logging.getLogger(__name__)

CARD_PROMPT = """Extract a structured paper card from this paper.
Title: {title}
Abstract: {abstract}
Body excerpts: {body}

Return EXACTLY these sections with numbered lists. ONLY points you are CERTAIN of.
## KEY RESULTS
## METHOD
## SETUP
## LIMITATIONS
Each line: "<claim text> [page P]" if a page number is citable."""

BUCKETS = {"KEY RESULTS": "key_results", "METHOD": "method", "SETUP": "setup", "LIMITATIONS": "limitations"}


def parse_card_response(text, paper_id):
    claims = {b: [] for b in BUCKETS.values()}
    current = None
    for line in text.splitlines():
        h = re.match(r"^##\s*(.+)$", line.strip())
        if h and h.group(1).strip() in BUCKETS:
            current = BUCKETS[h.group(1).strip()]
            continue
        m = re.match(r"^\s*\d+\.\s+(.+?)(?:\s*\[page (\d+)\])?$", line)
        if m and current:
            claim_text = m.group(1).strip()
            evid = []
            if m.group(2):
                evid.append(evidence_id(paper_id, int(m.group(2)), 0))
            claims[current].append(Claim(text=claim_text, dimension=current, evidence_ids=evid))
    return claims


def build_card(parsed, retrieved_map, llm, aspects, threshold=0.6):
    meta = retrieved_map.get(parsed.paper_id)
    body = " ".join(p.text for p in parsed.paragraphs[:20])
    raw = llm.chat([{"role": "user", "content": CARD_PROMPT.format(
        title=parsed.title, abstract=parsed.abstract, body=body)}])
    claims = parse_card_response(raw, parsed.paper_id)
    return PaperCard(
        paper_id=parsed.paper_id, title=parsed.title,
        authors=getattr(meta, "authors", []), year=getattr(meta, "year", None),
        venue=getattr(meta, "venue", None), matched_aspects=[], card_type="deep",
        problem="", method="", contribution="", limitations="",
        evidence_ids=[], figure_ids=[], table_ids=[], possible_claims=claims,
        bibtex_key=None)


def build_shallow_card(retrieved, llm, aspects, threshold=0.6):
    """Card from abstract only (no parsed body). Used when MinerU is off."""
    raw = llm.chat([{"role": "user", "content": CARD_PROMPT.format(
        title=retrieved.title, abstract=retrieved.abstract, body="")}])
    claims = parse_card_response(raw, retrieved.paper_id)
    return PaperCard(
        paper_id=retrieved.paper_id, title=retrieved.title,
        authors=retrieved.authors, year=retrieved.year,
        venue=retrieved.venue, matched_aspects=[], card_type="shallow",
        problem="", method="", contribution="", limitations="",
        evidence_ids=[], figure_ids=[], table_ids=[], possible_claims=claims,
        bibtex_key=None)


def run(task_id, parsed_papers, retrieved_papers, llm, aspects, threshold=0.6):
    logger.info(f"[P5.1] cards: {len(parsed_papers.papers)} parsed, {len(retrieved_papers.papers)} retrieved")
    rmap = {p.paper_id: p for p in retrieved_papers.papers}
    parsed_ids = {p.paper_id for p in parsed_papers.papers}
    cards = [build_card(p, rmap, llm, aspects, threshold) for p in parsed_papers.papers]
    # Shallow cards for abstract-only papers (not parsed by MinerU)
    shallow = 0
    for rp in retrieved_papers.papers:
        if rp.paper_id not in parsed_ids and rp.abstract:
            cards.append(build_shallow_card(rp, llm, aspects, threshold))
            shallow += 1
    logger.info(f"[P5.1] built {len(cards)} cards (deep={len(parsed_papers.papers)}, shallow={shallow})")
    return PaperCards(task_id=task_id, paper_cards=cards)
