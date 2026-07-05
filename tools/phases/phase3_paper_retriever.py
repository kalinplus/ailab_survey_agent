import logging
import math
import re
from tools.models.artifacts import RetrievedPaper, RetrievedPapers, ParsedPaper, ParsedPapers
from tools.models.common import paper_id_from_seed

logger = logging.getLogger(__name__)

INFLUENCE_YEAR_BANDS = [
    (2024, 2026, 0),
    (2021, 2023, 5),
    (2018, 2020, 20),
]
SURVEY_REF_WEIGHT = 0.07
RELEVANCE_MIN_SCORE = 0.45
GENERIC_RELEVANCE_TERMS = {
    "a", "an", "and", "ai", "agent", "agents", "analysis", "approach", "based",
    "data", "deep", "generation", "generative", "learning", "machine", "method",
    "model", "models", "neural", "paper", "research", "study", "system", "systems",
    "the", "training", "using", "video", "world",
}
SHORT_RELEVANCE_TERMS = {"rl", "vae"}


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
    seen, out = {}, []
    for p in papers:
        key = p.paper_id
        if key in seen:
            existing = seen[key]
            existing.survey_ref_count = max(existing.survey_ref_count, p.survey_ref_count)
            for hint in p.survey_ref_hints:
                if hint not in existing.survey_ref_hints:
                    existing.survey_ref_hints.append(hint)
            continue
        seen[key] = p
        out.append(p)
    return out


def _year_filters(start_year: int, end_year: int, min_citations: int | None = None) -> list[dict]:
    filters = [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": start_year},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": end_year},
    ]
    if min_citations is not None and min_citations > 0:
        filters.append({"field": "citation_count", "operator": "FILTER_OP_GTE", "value": min_citations})
    return filters


def _influence_filter_sets() -> list[list[dict]]:
    return [_year_filters(start, end, min_citations) for start, end, min_citations in INFLUENCE_YEAR_BANDS]


def _normalize_relevance_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _aspect_relevance_terms(aspects: list[dict]) -> list[str]:
    seen, terms = set(), []
    for aspect in aspects:
        for raw_keyword in aspect.get("keywords", []):
            for raw_part in re.split(r"[,;/|()]+", str(raw_keyword)):
                term = _normalize_relevance_text(raw_part)
                if not term or term in seen:
                    continue
                tokens = term.split()
                if len(tokens) == 1:
                    token = tokens[0]
                    if token in GENERIC_RELEVANCE_TERMS:
                        continue
                    if len(token) < 3 and token not in SHORT_RELEVANCE_TERMS:
                        continue
                seen.add(term)
                terms.append(term)
    return terms


def _paper_relevance_score(paper: RetrievedPaper, aspects: list[dict]) -> float:
    terms = _aspect_relevance_terms(aspects)
    if not terms:
        return 1.0
    title = _normalize_relevance_text(paper.title)
    body = _normalize_relevance_text(
        " ".join([paper.abstract, " ".join(str(k) for k in paper.keywords)])
    )
    title_tokens = set(title.split())
    body_tokens = set(body.split())
    best_score = 0.0
    match_count = 0
    for term in terms:
        tokens = term.split()
        if len(tokens) == 1:
            term_score = 0.0
            if term in title_tokens or any(token.startswith(term) for token in title_tokens):
                term_score = 0.85
            elif term in body_tokens or any(token.startswith(term) for token in body_tokens):
                term_score = 0.55
        else:
            term_score = 0.0
            if term in title:
                term_score = 1.0
            elif term in body:
                term_score = 0.7
            else:
                useful_tokens = [
                    token for token in tokens
                    if token not in GENERIC_RELEVANCE_TERMS
                    and (len(token) >= 3 or token in SHORT_RELEVANCE_TERMS)
                ]
                useful_tokens = useful_tokens or tokens
                required = 1 if len(useful_tokens) == 1 else 2
                title_matches = sum(
                    1 for token in useful_tokens
                    if token in title_tokens or any(t.startswith(token) for t in title_tokens)
                )
                body_matches = sum(
                    1 for token in useful_tokens
                    if token in body_tokens or any(t.startswith(token) for t in body_tokens)
                )
                if title_matches >= required:
                    term_score = 0.85
                elif body_matches >= required:
                    term_score = 0.6
                elif title_matches + body_matches >= required:
                    term_score = 0.55
        if term_score > 0:
            match_count += 1
            best_score = max(best_score, term_score)
    coverage_score = min(match_count / min(len(terms), 4), 1.0)
    return min(1.0, 0.75 * best_score + 0.25 * coverage_score)


def _filter_relevant_candidates(papers: list[RetrievedPaper], aspects: list[dict]) -> list[RetrievedPaper]:
    if not papers or not _aspect_relevance_terms(aspects):
        return papers
    filtered = [
        paper for paper in papers
        if _paper_relevance_score(paper, aspects) >= RELEVANCE_MIN_SCORE
    ]
    if not filtered:
        logger.info("[P3] relevance prefilter kept 0/%d papers -> fail-closed", len(papers))
        return []
    logger.info("[P3] relevance prefilter kept %d/%d papers", len(filtered), len(papers))
    return filtered


def _rank_by_influence(
    papers: list[RetrievedPaper],
    first_seen_rank: dict[str, int],
    aspects: list[dict] | None = None,
) -> list[RetrievedPaper]:
    if not papers:
        return []
    max_rank = max(first_seen_rank.values()) if first_seen_rank else 0
    max_citations = max(math.log1p(max(p.citation_count, 0)) for p in papers)
    max_survey_refs = max(p.survey_ref_count for p in papers)

    def score(paper: RetrievedPaper) -> float:
        rank = first_seen_rank.get(paper.paper_id, max_rank)
        source_rank_score = 1.0 if max_rank <= 0 else 1.0 - (rank / max_rank)
        citation_score = 0.0
        if max_citations > 0:
            citation_score = math.log1p(max(paper.citation_count, 0)) / max_citations
        recency_score = 0.0
        if paper.year is not None:
            recency_score = min(max((paper.year - 2018) / (2026 - 2018), 0.0), 1.0)
        survey_score = 0.0
        if max_survey_refs > 0:
            survey_score = paper.survey_ref_count / max_survey_refs
        metadata_quality_score = (
            int(bool(paper.abstract)) + int(bool(paper.venue)) + int(bool(paper.url))
        ) / 3
        relevance_score = _paper_relevance_score(paper, aspects or [])
        return (
            0.55 * relevance_score
            + 0.18 * source_rank_score
            + 0.07 * citation_score
            + 0.10 * recency_score
            + 0.07 * survey_score
            + 0.03 * metadata_quality_score
        )

    return sorted(papers, key=lambda paper: score(paper), reverse=True)


def _candidate_query(candidate: dict) -> str:
    return str(candidate.get("paper_id_hint", "")).replace("_", " ").strip()


def _candidate_terms(candidate: dict) -> list[str]:
    raw_terms = []
    for part in str(candidate.get("paper_id_hint", "")).split("_"):
        term = _normalize_relevance_text(part)
        if not term or (len(term) == 4 and term.isdigit()):
            continue
        raw_terms.append(term)
        prefix = re.sub(r"\d+$", "", term)
        if prefix and prefix != term and len(prefix) >= 3:
            raw_terms.append(prefix)
    return list(dict.fromkeys(raw_terms))


def _candidate_matches_hit(candidate: dict, hit: dict) -> bool:
    terms = _candidate_terms(candidate)
    if not terms:
        return False
    text = _normalize_relevance_text(
        " ".join([
            hit.get("title", "") or "",
            hit.get("abstract", "") or "",
            " ".join(str(k) for k in (hit.get("keywords") or [])),
            hit.get("publication_venue_name_unified") or hit.get("venue") or "",
        ])
    )
    tokens = set(text.split())
    return any(term in text if " " in term else term in tokens for term in terms)


def _candidate_year(candidate: dict) -> int | None:
    parts = str(candidate.get("paper_id_hint", "")).split("_")
    if parts and len(parts[-1]) == 4 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def _search_expansion_candidates(expansion_candidates, sciverse, retrieved, first_seen_rank):
    for candidate in expansion_candidates:
        query = _candidate_query(candidate)
        if not query:
            continue
        year = _candidate_year(candidate)
        filters = _year_filters(year, year) if year else _year_filters(2018, 2026)
        try:
            res = sciverse.meta_search(query=query, filters=filters, impact_boost="MILD", page_size=3)
            hits = res.get("results", [])
            logger.info(f"[P3] expansion_search q={query!r} filters={filters} -> {len(hits)} hits")
            for hit in hits:
                if not _candidate_matches_hit(candidate, hit):
                    continue
                paper = _to_retrieved(hit, source="survey_expansion")
                paper.survey_ref_count = int(candidate.get("survey_ref_count") or 1)
                paper.survey_ref_hints = [candidate["paper_id_hint"]]
                first_seen_rank.setdefault(paper.paper_id, len(retrieved))
                retrieved.append(paper)
        except Exception as e:
            logger.warning(f"[P3] expansion_search failed q={query!r}: {e}")


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
    logger.info(f"[P3] retrieve: {len(aspects)} aspects, use_mineru={pipeline_config.use_mineru}, "
                f"use_seed_fallback={pipeline_config.use_seed_fallback}, "
                f"use_influence_score={pipeline_config.use_influence_score}")
    retrieved = []
    first_seen_rank = {}
    relevance_aspects = list(aspects)
    # 1. expansion candidates from curated surveys: exact, small-page searches.
    _search_expansion_candidates(expansion_candidates, sciverse, retrieved, first_seen_rank)

    # 2. meta-search per aspect (verified SciVerse filter/response contract)
    filter_sets = (
        _influence_filter_sets()
        if pipeline_config.use_influence_score
        else [_year_filters(2018, 2026)]
    )
    for a in aspects:
        keywords = a.get("keywords", [])
        # If keywords contain non-ASCII (Chinese topic) and LLM available, generate English queries
        if llm and any(_has_non_ascii(kw) for kw in keywords):
            queries = _generate_search_queries(keywords, llm)
            logger.info(f"[P3] aspect {a.get('aspect_id', '?')} translate -> {queries}")
        else:
            queries = [" ".join(keywords)]
        relevance_aspects.append({"keywords": queries})
        for query in queries:
            for filters in filter_sets:
                try:
                    kwargs = {
                        "query": query,
                        "filters": filters,
                        "impact_boost": "MILD",
                        "page_size": 15 if pipeline_config.use_influence_score else 25,
                    }
                    if pipeline_config.use_influence_score:
                        kwargs["freshness_boost"] = "MILD"
                    res = sciverse.meta_search(**kwargs)
                    hits = res.get("results", [])
                    logger.info(f"[P3] meta_search q={query!r} filters={filters} -> {len(hits)} hits")
                    for hit in hits:
                        paper = _to_retrieved(hit)
                        first_seen_rank.setdefault(paper.paper_id, len(retrieved))
                        retrieved.append(paper)
                except Exception as e:
                    logger.warning(f"[P3] meta_search failed q={query!r} filters={filters}: {e}")
    # 3. seed fallback if empty
    if not retrieved and pipeline_config.use_seed_fallback:
        logger.warning(f"[P3] empty retrieval -> seed fallback ({len(seed_papers)} papers)")
        retrieved = [_to_retrieved(s, source="seed") for s in seed_papers]
        first_seen_rank = {p.paper_id: i for i, p in enumerate(retrieved)}
    retrieved = dedup(retrieved)
    if pipeline_config.use_influence_score:
        retrieved = _filter_relevant_candidates(retrieved, relevance_aspects)
        retrieved = _rank_by_influence(retrieved, first_seen_rank, relevance_aspects)
    retrieved = retrieved[:pipeline_config_extra(pipeline_config, "max_papers", 40)]
    logger.info(f"[P3] after dedup/cap: {len(retrieved)} papers")
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
        except Exception as e:
            logger.warning(f"[P3] mineru parse failed {p.paper_id}: {e}")
            p.parse_status = "abstract_only"
    light_count = sum(1 for p in retrieved if p.parse_status == "light")
    logger.info(f"[P3] parse: {light_count} light / {len(retrieved) - light_count} abstract_only, "
                f"parsed_papers={len(parsed)}")
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
