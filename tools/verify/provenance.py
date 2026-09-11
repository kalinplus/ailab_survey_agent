"""Provenance gate for externally fetched chunks (agentic-search / repair backfill).

A SciVerse hit may be attached under a paper's id only when the hit actually
belongs to that paper. The paper id carries a DOI when it looks like
``paper:10.48550/arxiv.2408.14837``; the hit then must carry the same DOI in
``doc_id`` (prefix/case-insensitive). Non-DOI ids (e.g. ``seed:*``) fall back
to normalized title equality. Anything else is rejected: a relevant chunk from
a different paper must never be renamed onto the citing paper (the
AvalonBench/Avalanche cross-paper pollution).
"""

import re

from tools.verify.text_units import normalize_text  # re-exported for callers

_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/",
    "doi.org/", "dx.doi.org/", "doi:", "info:doi/",
)
_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def extract_doi(paper_id: str) -> str:
    """DOI carried by a paper id ('paper:10.x/y' -> '10.x/y'), '' when absent."""
    pid = str(paper_id)
    if pid.startswith("paper:"):
        pid = pid[len("paper:"):]
    pid = pid.strip().casefold()
    if pid.startswith(_DOI_PREFIXES):
        for prefix in _DOI_PREFIXES:
            if pid.startswith(prefix):
                pid = pid[len(prefix):]
                break
    return pid if pid.startswith("10.") else ""


def normalize_doi(text: str) -> str:
    """Comparable DOI form: strip url/doi prefixes, map arXiv aliases onto the
    arXiv DOI (10.48550/arXiv.<id>) so 'arxiv:2408.14837', '2408.14837' and
    '10.48550/arxiv.2408.14837' compare equal."""
    s = str(text).strip().casefold()
    for prefix in _DOI_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("arxiv:"):
        s = "10.48550/arxiv." + s[len("arxiv:"):]
    elif _ARXIV_ID_RE.match(s):
        s = "10.48550/arxiv." + s
    return s.rstrip(".")


def chunk_id(hit: dict) -> str:
    """Opaque chunk identity retained as Evidence.source_chunk_id."""
    hit_id = str(hit.get("id") or "").strip()
    if hit_id:
        return hit_id
    return f"{hit.get('doc_id')}:{hit.get('page_no') or 0}:{hit.get('offset') or 0}"


def hit_matches_paper(paper_id: str, paper_title: str, hit: dict) -> bool:
    """True only when the hit verifiably belongs to the cited paper.

    Real agentic-search doc_ids are opaque internal hashes (verified live
    2026-09-09), so the DOI branch alone never fires and would reject even a
    paper's own chunks. Identity is therefore DOI-normalized equality on
    doc_id OR normalized title equality — either is sufficient, empty strings
    never match, and substring similarity is never accepted.
    """
    doi = extract_doi(paper_id)
    if doi and normalize_doi(doi) == normalize_doi(str(hit.get("doc_id") or "")):
        return True
    title = normalize_text(paper_title)
    hit_title = normalize_text(str(hit.get("title") or ""))
    return bool(title) and title == hit_title
