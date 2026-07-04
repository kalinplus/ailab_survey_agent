from tools.models.artifacts import RetrievedPaper, RetrievedPapers, ParsedPaper, ParsedPapers
from tools.models.common import paper_id_from_seed


def _is_pdf_url(url: str) -> bool:
    u = (url or "").lower()
    return "/pdf" in u or u.endswith(".pdf")


def _derive_url(hit: dict) -> str | None:
    # SciVerse results mix direct PDFs (arxiv access_oa_url, locations type=download) with
    # doi.org landing pages. Prefer a direct PDF so MinerU can actually parse it; else fall
    # back to the first candidate / doi landing page (parse may then degrade to abstract_only).
    candidates = list(hit.get("access_oa_url") or [])
    for loc in hit.get("locations") or []:
        if isinstance(loc, dict) and loc.get("url"):
            candidates.append(loc["url"])
    for url in candidates:
        if _is_pdf_url(url):
            return url
    if candidates:
        return candidates[0]
    doi = hit.get("doi")
    if doi:
        return f"https://doi.org/{doi}"
    return None


def _to_retrieved(hit: dict, source="sciverse") -> RetrievedPaper:
    # Dual-schema: SciVerse native fields, falling back to seed-corpus field names.
    year_raw = hit.get("publication_published_year")
    year = int(year_raw) if year_raw is not None else hit.get("year")
    authors = [a["name"] for a in (hit.get("author") or []) if isinstance(a, dict) and a.get("name")]
    if not authors:
        authors = hit.get("authors", []) or []
    return RetrievedPaper(
        paper_id=hit.get("unique_id") or paper_id_from_seed(hit.get("title", ""), year or 0),
        title=hit.get("title", ""),
        authors=authors,
        year=year,
        venue=hit.get("publication_venue_name_unified") or hit.get("venue"),
        url=hit.get("url") or _derive_url(hit),
        abstract=hit.get("abstract", "") or "",
        keywords=hit.get("keywords", []) or [],
        citation_count=int(hit.get("citation_count") or 0),
        source=source,
        parse_status="pending",
    )


def _has_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _generate_search_queries(keywords: list[str], llm) -> list[str]:
    """Use LLM to translate/expand non-English keywords into English academic queries."""
    raw = " ".join(keywords)
    if not _has_non_ascii(raw):
        return [raw]
    prompt = (
        f"Translate the following topic keywords into 2-3 concise English academic "
        f"search queries for finding research papers. Return ONLY the queries, one per line.\n"
        f"Keywords: {raw}"
    )
    try:
        result = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
        lines = [ln.strip().strip("-•*").strip() for ln in result.strip().splitlines() if ln.strip()]
        return lines[:3] if lines else [raw]
    except Exception:
        return [raw]


def dedup(papers: list[RetrievedPaper]) -> list[RetrievedPaper]:
    seen, out = set(), []
    for p in papers:
        key = p.paper_id
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def run(
    task_id,
    aspects,
    expansion_candidates,
    sciverse,
    mineru,
    cleaner,
    seed_papers,
    pipeline_config,
    llm=None,
):
    retrieved = []
    # 1. expansion refs via meta-paper-relations (skip if sciverse offline -> caught upstream)
    # 2. meta-search per aspect (verified SciVerse filter/response contract)
    year_filters = [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2018},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2026},
    ]
    for a in aspects:
        keywords = a.get("keywords", [])
        # If keywords contain non-ASCII (Chinese topic) and LLM available, generate English queries
        if llm and any(_has_non_ascii(kw) for kw in keywords):
            queries = _generate_search_queries(keywords, llm)
        else:
            queries = [" ".join(keywords)]
        for query in queries:
            try:
                res = sciverse.meta_search(
                    query=query,
                    filters=year_filters,
                    impact_boost="MILD",
                    page_size=25,
                )
                for hit in res.get("results", []):
                    retrieved.append(_to_retrieved(hit))
            except Exception:
                pass
    # 3. seed fallback if empty
    if not retrieved and pipeline_config.use_seed_fallback:
        retrieved = [_to_retrieved(s, source="seed") for s in seed_papers]
    retrieved = dedup(retrieved)[:pipeline_config_extra(pipeline_config, "max_papers", 40)]
    # 4. parse via real MinerU; only feed actual PDF URLs; degrade per-paper on failure or empty result
    parsed = []
    for p in retrieved:
        if not p.url or not pipeline_config.use_mineru or not _is_pdf_url(p.url):
            p.parse_status = "abstract_only"
            continue
        try:
            raw = mineru.parse_url(p.url, light=True)
            raw = cleaner.clean_parsed(raw)
            paper = _to_parsed(p, raw)
            if not paper.paragraphs and not paper.sections:
                p.parse_status = "abstract_only"
            else:
                p.parse_status = "light"
                parsed.append(paper)
        except Exception:
            p.parse_status = "abstract_only"
    return (RetrievedPapers(task_id=task_id, papers=retrieved),
            ParsedPapers(task_id=task_id, papers=parsed))


def pipeline_config_extra(cfg, key, default):
    return getattr(cfg, key, default) if hasattr(cfg, key) else default


def _to_parsed(p, raw):
    return ParsedPaper(
        paper_id=p.paper_id,
        title=raw.get("title", p.title),
        abstract=raw.get("abstract", ""),
        sections=raw.get("sections", []),
        paragraphs=raw.get("paragraphs", []),
        figures=raw.get("figures", []),
        tables=raw.get("tables", []),
        parse_status="light",
    )
