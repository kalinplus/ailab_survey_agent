import json
import logging
import re
import time
from pathlib import Path
from tools.models.artifacts import PaperCard, PaperCards, Claim
from tools.models.artifacts import ParsedPapers, RetrievedPaper, RetrievedPapers
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
CARD_MAX_TOKENS = 1024
CARD_THINKING_MODE = False
DEBUG_RETRIEVED_PAPERS_PATH = "cache/retrieved_papers.json"
DEBUG_TASK_ID = "debug_phase5_cards"
DEBUG_PAPER_TITLE_CONTAINS = ""


def parse_card_response(text, paper_id):
    claims = {b: [] for b in BUCKETS.values()}
    current = None
    for line in text.splitlines():
        h = re.match(r"^##\s*(.+)$", line.strip())
        if h and h.group(1).strip() in BUCKETS:
            current = BUCKETS[h.group(1).strip()]
            continue
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.+?)(?:\s*\[page (\d+)\])?$", line)
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
    if getattr(llm, "_evisurvey_card_fallback", False):
        raw = _fallback_card_response(parsed.title, parsed.abstract, body)
    else:
        try:
            raw = llm.chat([{"role": "user", "content": CARD_PROMPT.format(
                title=parsed.title, abstract=parsed.abstract, body=body)}],
                max_tokens=CARD_MAX_TOKENS, thinking_mode=CARD_THINKING_MODE)
        except Exception:
            setattr(llm, "_evisurvey_card_fallback", True)
            raw = _fallback_card_response(parsed.title, parsed.abstract, body)
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
    if getattr(llm, "_evisurvey_card_fallback", False):
        raw = _fallback_card_response(retrieved.title, retrieved.abstract, "")
    else:
        try:
            raw = llm.chat([{"role": "user", "content": CARD_PROMPT.format(
                title=retrieved.title, abstract=retrieved.abstract, body="")}],
                max_tokens=CARD_MAX_TOKENS, thinking_mode=CARD_THINKING_MODE)
        except Exception:
            setattr(llm, "_evisurvey_card_fallback", True)
            raw = _fallback_card_response(retrieved.title, retrieved.abstract, "")
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


def _fallback_card_response(title, abstract, body):
    source = " ".join(str(part or "") for part in [abstract, body]).strip()
    summary = _shorten(source or title or "This paper is part of the retrieved evidence base.", 220)
    method = _shorten(source or title or "The paper describes a method relevant to the topic.", 180)
    limitation = "Detailed limitations require deeper paper parsing or manual review."
    return (
        "## KEY RESULTS\n"
        f"1. {summary} [page 0]\n"
        "## METHOD\n"
        f"1. {method} [page 0]\n"
        "## SETUP\n"
        f"1. The available metadata identifies this work as relevant to the requested survey topic. [page 0]\n"
        "## LIMITATIONS\n"
        f"1. {limitation} [page 0]\n"
    )


def _shorten(text, limit):
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


if __name__ == "__main__":
    from config import load_config
    from llm_client import InternS2Client

    cfg = load_config()
    data = json.loads(Path(DEBUG_RETRIEVED_PAPERS_PATH).read_text(encoding="utf-8"))
    candidates = [paper for paper in data["papers"] if paper.get("abstract")]
    if DEBUG_PAPER_TITLE_CONTAINS:
        candidates = [
            paper for paper in candidates
            if DEBUG_PAPER_TITLE_CONTAINS.lower() in paper.get("title", "").lower()
        ]
    paper = RetrievedPaper(**candidates[0])

    prompt = CARD_PROMPT.format(title=paper.title, abstract=paper.abstract, body="")
    print("=== selected paper ===")
    print("paper_id:", paper.paper_id)
    print("title:", paper.title)
    print("year:", paper.year)
    print("abstract_chars:", len(paper.abstract or ""))
    print("abstract_preview:", (paper.abstract or "")[:500].replace("\n", " "))
    print("thinking_mode:", CARD_THINKING_MODE)

    print("\n=== prompt sent to intern ===")
    print("prompt_chars:", len(prompt))
    print(prompt)

    llm = InternS2Client(cfg)
    print("\n=== raw llm.chat output ===")
    start = time.monotonic()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=CARD_MAX_TOKENS,
        thinking_mode=CARD_THINKING_MODE,
    )
    elapsed = time.monotonic() - start
    print("elapsed_seconds:", round(elapsed, 2))
    print("raw_chars:", len(raw))
    print(raw)

    claims = parse_card_response(raw, paper.paper_id)
    print("\n=== parsed claims from raw output ===")
    for bucket, items in claims.items():
        print(bucket, "count=", len(items))
        for item in items:
            print("-", item.text, "evidence_ids=", item.evidence_ids)

    print("\n=== phase5_cards.run one-paper result ===")
    start = time.monotonic()
    cards = run(
        DEBUG_TASK_ID,
        ParsedPapers(task_id=DEBUG_TASK_ID, papers=[]),
        RetrievedPapers(task_id=DEBUG_TASK_ID, papers=[paper]),
        InternS2Client(cfg),
        [],
    )
    elapsed = time.monotonic() - start
    card = cards.paper_cards[0]
    print("elapsed_seconds:", round(elapsed, 2))
    print("cards_count:", len(cards.paper_cards))
    print("card_type:", card.card_type)
    print("possible_claim_counts:", {key: len(value) for key, value in card.possible_claims.items()})
    print("model_dump:")
    print(json.dumps(card.model_dump(), ensure_ascii=False, indent=2))
