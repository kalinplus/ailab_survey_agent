import hashlib


def paper_id_from_seed(title: str, year: int) -> str:
    h = hashlib.sha256(f"{title}|{year}".encode()).hexdigest()[:12]
    return f"seed:{h}"


# Per-source-type namespaces: a caption's (page, num) and a paragraph's
# (page, index) used to collide into the same id (e.g. "pid_p5_3"); ids are
# opaque downstream, so the prefix makes every source type its own namespace.
_EVIDENCE_ID_PREFIXES = {
    "abstract": "abs",
    "paragraph": "para",
    "caption": "cap",
    "agentic_chunk": "agentic",
    "content_chunk": "content",
    "repair": "repair",
}


def evidence_id(paper_id: str, page: int | None, idx: int, source_type: str = "paragraph") -> str:
    prefix = _EVIDENCE_ID_PREFIXES.get(source_type, "src")
    # page=None (markdown-only parse channel) gets its own token; idx keeps
    # ids unique within a namespace.
    page_token = "x" if page is None else str(page)
    return f"{paper_id}_{prefix}_p{page_token}_{idx}"


def figure_id(paper_id: str, num: int) -> str:
    return f"{paper_id}_fig{num}"


def table_id(paper_id: str, num: int) -> str:
    return f"{paper_id}_tbl{num}"


def category_id(n: int) -> str:
    return f"cat_{n:03d}"
