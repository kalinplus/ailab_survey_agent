import json
import logging
import re
import time
from pathlib import Path
from tools.models.artifacts import PaperCard, PaperCards, Claim
from tools.models.artifacts import ParsedPapers, RetrievedPaper, RetrievedPapers
from tools.models.common import evidence_id
from tools.verify.source_contract import role_violation
from tools.verify.text_units import normalize_text

logger = logging.getLogger(__name__)

CARD_PROMPT = """Extract claim cards from the numbered source blocks of one paper.
Title (metadata only): {title}

SOURCE BLOCKS (format: [evidence_id] text):
{blocks}

Return ONLY a JSON array; each element:
{{"text": "<self-contained paraphrase, max 40 words>", "dimension": "key_results|method|setup|limitations", "source_role": "own_method|own_contribution|own_result|own_setup|own_limitation|background|related_work", "evidence_ids": ["<one block id the quote comes from>"], "source_quote": "<exact substring copied from that block, max 55 words>"}}

Rules:
- source_role classifies the CLAIM's attribution, not the paragraph heading: statements about earlier or other methods are background/related_work; only the paper's own statements get own_* (e.g. perfect simulators in MuZero's background are never MuZero's own method).
- When the paper itself states limitations, failure modes, or negative results of its own method, extract 1-2 of them as own_limitation claims (dimension "limitations"); do not skip them - they ground the survey's open-challenges discussion. Still only from verbatim quotes; never invent limitations.
- source_quote MUST be copied verbatim from the named block; evidence_ids MUST be one of the block ids above.
- Keep technical names, numbers, negation and scope accurate. Do not copy whole source sentences into text. Never fill missing slots.
- Extract 3 to 6 claims covering the paper's own method and results when present. Omit a claim you cannot ground, but never return an empty array when the paper's own method or results are visible in the blocks."""

CARD_SYSTEM = ("You are a strict JSON extraction engine. Do not write any reasoning "
               "or explanation. Output the JSON array and nothing else.")

BUCKETS = {"KEY RESULTS": "key_results", "METHOD": "method", "SETUP": "setup", "LIMITATIONS": "limitations"}
CARD_MAX_TOKENS = 4096
CARD_THINKING_MODE = False
DEBUG_RETRIEVED_PAPERS_PATH = "cache/retrieved_papers.json"
DEBUG_TASK_ID = "debug_phase5_cards"
DEBUG_PAPER_TITLE_CONTAINS = ""


def parse_card_response(text, paper_id, source_blocks=None):
    claims = {b: [] for b in BUCKETS.values()}
    sources = {b["evidence_id"]: b for b in source_blocks or []}
    rows = _json_array_rows(text)
    if rows is not None:
        for row in rows[:6]:
            if not isinstance(row, dict) or row.get("dimension") not in claims:
                continue
            role = row.get("source_role", "unknown")
            quote = row.get("source_quote", "")
            ids = row.get("evidence_ids", [])
            if (not isinstance(ids, list) or not ids or not isinstance(quote, str)
                    or not quote or role_violation(role, quote)
                    or not isinstance(row.get("text"), str) or not row["text"].strip()):
                logger.warning("[P5.1] rejected unbound/unknown-role claim for %s", paper_id)
                continue
            if any(not isinstance(eid, str) or eid not in sources
                   or sources[eid]["paper_id"] != paper_id
                   or normalize_text(quote) not in normalize_text(sources[eid]["text"])
                   for eid in ids):
                logger.warning("[P5.1] rejected foreign or fabricated source binding for %s", paper_id)
                continue
            claims[row["dimension"]].append(Claim(
                text=row["text"].strip(), dimension=row["dimension"], source_role=role,
                source_quote=quote, evidence_ids=ids))
        return claims
    # Legacy cards are readable, but their page-only links are not real block
    # identities and their bucket names do not establish attribution.
    current = None
    for line in text.splitlines():
        h = re.match(r"^##\s*(.+)$", line.strip())
        if h:
            current = BUCKETS.get(h.group(1).strip())
            continue
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.+?)(?:\s*\[page \d+\])?$", line)
        if m and current:
            claims[current].append(Claim(text=m.group(1).strip(), dimension=current))
    return claims


def _json_array_rows(text):
    """Tolerant JSON-array extraction: strip code fences and any reasoning
    preamble around the array. Returns None when no complete array parses —
    reasoning models can burn the whole completion budget before the array
    appears, and that truncated reply must fail, not parse as empty."""
    body = re.sub(r"^```(?:json)?\s*|\s^```$", "", text.strip())
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        rows = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    return rows if isinstance(rows, list) else None


_EMAIL_RE = re.compile(r"\b\w+@\w+\.\w+\b")
_AFFILIATION_RE = re.compile(r"Department of|Universit|Institute of")


def _is_author_block(text: str) -> bool:
    """Leading title/author/affiliation blocks of a /content fulltext. They were
    pasted verbatim into card fields and then into survey prose (live wave-4:
    'starts from # Genie: ... Jake Bruce<sup>...'). Only block-shaped text
    matches: superscripts/emails/leading headings always; affiliation markers
    only on lines without sentence flow, so body prose mentioning a university
    stays."""
    t = text.strip()
    if "<sup>" in t or _EMAIL_RE.search(t) or t.startswith("#"):
        return True
    if len(t) >= 400 or len(t.split()) >= 60:
        return False
    return bool(_AFFILIATION_RE.search(t)) and t.count(". ") == 0 and t.count(".") <= 1


def _extract_claims(paper_id, title, blocks, llm):
    lines = [f"[{b['evidence_id']}] {b['text']}" for b in blocks]
    started = time.monotonic()
    try:
        raw = llm.chat([
            {"role": "system", "content": CARD_SYSTEM},
            {"role": "user", "content": CARD_PROMPT.format(
                title=title, blocks="\n".join(lines))}],
            max_tokens=CARD_MAX_TOKENS, thinking_mode=CARD_THINKING_MODE)
        claims = parse_card_response(raw, paper_id, blocks)
    except Exception as exc:  # external model/response boundary, isolated per paper
        logger.warning("[P5.1] card failed paper=%s blocks=%d elapsed=%.1fs: %s",
                       paper_id, len(blocks), time.monotonic() - started, exc)
        return {b: [] for b in BUCKETS.values()}, "api_or_response_failed"
    safe_count = sum(c.source_role != "unknown" for bucket in claims.values() for c in bucket)
    status = "extracted" if safe_count else "no_safe_claims"
    logger.log(logging.INFO if safe_count else logging.WARNING,
               "[P5.1] card paper=%s blocks=%d safe_claims=%d status=%s elapsed=%.1fs",
               paper_id, len(blocks), safe_count, status, time.monotonic() - started)
    return claims, status


def build_card(parsed, retrieved_map, llm, aspects, threshold=0.6):
    meta = retrieved_map.get(parsed.paper_id)
    blocks = []
    if parsed.abstract:
        blocks.append(dict(evidence_id=evidence_id(parsed.paper_id, 0, 0, "abstract"),
                           paper_id=parsed.paper_id, text=parsed.abstract, source_type="abstract",
                           source_page=0, source_paragraph_index=0))
    eligible = [p for p in parsed.paragraphs if p.role != "reference" and not _is_author_block(p.text)]
    priority = [p for p in eligible if re.search(
        r"\b(?:we (?:propose|introduce|show|find|train|evaluate)|our (?:method|model|results)|limitations?)\b", p.text, re.I)]
    # Mix explicit own-work assertions with evenly spaced body blocks. IDs
    # remain the parser's actual page/index, regardless of input ordering.
    spread = [eligible[round(i * (len(eligible) - 1) / min(11, len(eligible) - 1))]
              for i in range(min(12, len(eligible)))] if len(eligible) > 1 else eligible
    chosen = []
    for para in priority[:12] + spread:
        if (para.page, para.index) not in {(p.page, p.index) for p in chosen}:
            chosen.append(para)
    for para in chosen[:20]:
        blocks.append(dict(evidence_id=evidence_id(parsed.paper_id, para.page, para.index),
                           paper_id=parsed.paper_id, text=para.text, source_type="paragraph",
                           source_page=para.page, source_paragraph_index=para.index))

    claims, status = _extract_claims(parsed.paper_id, parsed.title, blocks, llm)
    return PaperCard(
        paper_id=parsed.paper_id, title=parsed.title,
        doc_id=getattr(meta, "doc_id", "") or "",
        authors=getattr(meta, "authors", []), year=getattr(meta, "year", None),
        venue=getattr(meta, "venue", None), citation_count=getattr(meta, "citation_count", 0),
        survey_ref_count=getattr(meta, "survey_ref_count", 0), card_type="deep",
        extraction_status=status, source_blocks=blocks, possible_claims=claims)


def build_shallow_card(retrieved, llm, aspects, threshold=0.6):
    blocks = [dict(evidence_id=evidence_id(retrieved.paper_id, 0, 0, "abstract"),
                   paper_id=retrieved.paper_id, text=retrieved.abstract, source_type="abstract",
                   source_page=0, source_paragraph_index=0)] if retrieved.abstract else []
    claims, status = _extract_claims(retrieved.paper_id, retrieved.title, blocks, llm)
    return PaperCard(
        paper_id=retrieved.paper_id, title=retrieved.title, doc_id=retrieved.doc_id,
        authors=retrieved.authors, year=retrieved.year, venue=retrieved.venue,
        citation_count=retrieved.citation_count, survey_ref_count=retrieved.survey_ref_count,
        card_type="shallow", extraction_status=status, source_blocks=blocks, possible_claims=claims)


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


def _fallback_card_response(abstract, body):
    # Raw source text is retained in source_blocks, never assigned a claim role.
    return "[]"


def _shorten(text, limit):
    # Cut on a word boundary when one exists: a mid-word stub ("w...",
    # "high qua...") survives into card fields and then into survey prose.
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    cut = value[: limit - 3]
    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip()
    return cut + "..."


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

    block = dict(evidence_id=evidence_id(paper.paper_id, 0, 0, "abstract"),
                 paper_id=paper.paper_id, text=paper.abstract or "",
                 source_type="abstract", source_page=0, source_paragraph_index=0)
    prompt = CARD_PROMPT.format(title=paper.title, blocks=f"[{block['evidence_id']}] {block['text']}")
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
        [
            {"role": "system", "content": CARD_SYSTEM},
            {"role": "user", "content": prompt},
        ],
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
