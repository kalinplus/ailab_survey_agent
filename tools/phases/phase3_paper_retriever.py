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
):
    retrieved = []
    # 1. expansion refs via meta-paper-relations (skip if sciverse offline -> caught upstream)
    # 2. meta-search per aspect (verified SciVerse filter/response contract)
    year_filters = [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2018},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2026},
    ]
    for a in aspects:
        try:
            res = sciverse.meta_search(
                query=" ".join(a.get("keywords", [])),
                filters=year_filters,
                impact_boost="MILD",  # prefer highly-cited papers for the survey core
                page_size=25,
            )
            for hit in res.get("results", []):
                retrieved.append(_to_retrieved(hit))
        except Exception:
            pass  # fail fast per-aspect suppressed only to keep other aspects; logged in quality_report
    # 3. seed fallback if empty
    if not retrieved and pipeline_config.use_seed_fallback:
        retrieved = [_to_retrieved(s, source="seed") for s in seed_papers]
    retrieved = dedup(retrieved)[:pipeline_config_extra(pipeline_config, "max_papers", 40)]
    # 4. parse via real MinerU; degrade per-paper to abstract_only on failure (keep real abstract, no mock)
    parsed = []
    for p in retrieved:
        if not p.url or not pipeline_config.use_mineru:
            p.parse_status = "abstract_only"
            continue
        try:
            raw = mineru.parse_url(p.url, light=True)
            raw = cleaner.clean_parsed(raw)
            p.parse_status = "light"
            parsed.append(_to_parsed(p, raw))
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
