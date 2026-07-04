import re
from tools.models.artifacts import PaperCard, PaperCards, Claim
from tools.models.common import evidence_id

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


def run(task_id, parsed_papers, retrieved_papers, llm, aspects, threshold=0.6):
    rmap = {p.paper_id: p for p in retrieved_papers.papers}
    cards = [build_card(p, rmap, llm, aspects, threshold) for p in parsed_papers.papers]
    return PaperCards(task_id=task_id, paper_cards=cards)
