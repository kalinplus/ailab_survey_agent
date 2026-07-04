from tools.models.artifacts import RetrievedPaper, RetrievedPapers, ParsedPaper, ParsedPapers
from tools.models.common import paper_id_from_seed


def _to_retrieved(hit: dict, source="sciverse") -> RetrievedPaper:
    return RetrievedPaper(
        paper_id=hit.get("unique_id") or paper_id_from_seed(hit.get("title", ""), hit.get("year", 0)),
        title=hit.get("title", ""),
        authors=hit.get("authors", []),
        year=hit.get("year"),
        venue=hit.get("venue"),
        url=hit.get("url"),
        abstract=hit.get("abstract", ""),
        keywords=hit.get("keywords", []),
        citation_count=hit.get("citation_count", 0),
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
    # 2. meta-search per aspect
    for a in aspects:
        try:
            res = sciverse.meta_search(
                query=" ".join(a.get("keywords", [])),
                filters=[{"field": "publication_published_year", "value": {"gte": 2018, "lte": 2026}}],
                sort=[{"field": "citation_count", "order": "SORT_ORDER_DESC"}],
                freshness_boost="MILD",
                impact_boost="MILD",
                page_size=25,
            )
            for hit in res.get("hits", []):
                retrieved.append(_to_retrieved(hit))
        except Exception:
            pass  # fail fast per-aspect suppressed only to keep other aspects; logged in quality_report
    # 3. seed fallback if empty
    if not retrieved and pipeline_config.use_seed_fallback:
        retrieved = [_to_retrieved(s, source="seed") for s in seed_papers]
    retrieved = dedup(retrieved)[:pipeline_config_extra(pipeline_config, "max_papers", 40)]
    # 4. parse
    parsed = []
    for p in retrieved:
        if not p.url or not pipeline_config.use_mineru:
            p.parse_status = "abstract_only"
            continue
        raw = mineru.parse_url(p.url, light=True)
        raw = cleaner.clean_parsed(raw)
        p.parse_status = "light"
        parsed.append(_to_parsed(p, raw))
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
