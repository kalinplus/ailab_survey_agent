import json
import logging
import math
import os
import re
import unicodedata
from tools.models.artifacts import RetrievedPaper, RetrievedPapers, ParsedPaper, ParsedPapers
from tools.models.common import paper_id_from_seed
# shared markdown kernel (page=None + paper-global block ids + reference roles):
# the same splitter behind MinerU's fallback channel parses SciVerse /content fulltext
from tools.clients.mineru_client import _markdown_parsed

logger = logging.getLogger(__name__)

INFLUENCE_YEAR_BANDS = [
    (2024, 2026, 0),
    (2021, 2023, 5),
    (2018, 2020, 20),
]
SURVEY_REF_WEIGHT = 0.07
RELEVANCE_MIN_SCORE = 0.45
DEFAULT_PAGE_SIZE_INFLUENCE = 15
DEFAULT_PAGE_SIZE_BROAD = 25
# Retrieval provenance buckets for papers not bound to a search aspect.
EXPANSION_BUCKET = "expansion"
SEED_BUCKET = "seed"
GENERIC_RELEVANCE_TERMS = {
    "a", "an", "and", "ai", "agent", "agents", "analysis", "approach", "based",
    "data", "deep", "generation", "generative", "learning", "machine", "method",
    "model", "models", "neural", "paper", "research", "study", "system", "systems",
    "the", "training", "using", "video", "world",
}
SHORT_RELEVANCE_TERMS = {"rl", "vae"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    return default if not raw else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    return default if not raw else float(raw)


def relevance_min_score() -> float:
    """Lower threshold keeps more candidates in the pooled relevance prefilter."""
    return _env_float("EVISURVEY_RELEVANCE_MIN_SCORE", RELEVANCE_MIN_SCORE)


def meta_page_size(use_influence_score: bool) -> int:
    """Per-query meta-search page_size; 0 = legacy 15 (influence) / 25 (broad) split."""
    override = _env_int("EVISURVEY_META_PAGE_SIZE", 0)
    if override > 0:
        return override
    return DEFAULT_PAGE_SIZE_INFLUENCE if use_influence_score else DEFAULT_PAGE_SIZE_BROAD


def max_queries_per_aspect() -> int:
    """Target query count per aspect; extra queries come from the aspect's own keywords."""
    return max(1, _env_int("EVISURVEY_MAX_QUERIES_PER_ASPECT", 1))


def aspect_min_papers() -> int:
    """Coverage floor per aspect after the relevance prefilter; default 2 keeps
    every aspect represented (a 1-paper section means a hollow survey skeleton)."""
    return _env_int("EVISURVEY_ASPECT_MIN_PAPERS", 2)


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


def _oa_pdf_url(hit: dict) -> str | None:
    # MinerU needs a direct PDF; the OA direct link (e.g. arxiv.org/pdf/xxx) beats
    # whatever landing page the hit's url field carries.
    for url in hit.get("access_oa_url") or []:
        if isinstance(url, str) and url.startswith("http") and _is_pdf_url(url):
            return url
    return None


def _to_retrieved(hit: dict, source="sciverse") -> RetrievedPaper:
    # Dual-schema: SciVerse native fields, falling back to seed-corpus field names.
    year_raw = hit.get("publication_published_year")
    year = int(year_raw) if year_raw is not None else hit.get("year")
    authors = [a["name"] for a in (hit.get("author") or []) if isinstance(a, dict) and a.get("name")]
    if not authors:
        authors = hit.get("authors", []) or []
    topic = hit.get("primary_topic") or {}
    return RetrievedPaper(
        paper_id=hit.get("unique_id") or paper_id_from_seed(hit.get("title", ""), year or 0),
        title=hit.get("title", ""),
        doc_id=str(hit.get("doc_id") or ""),
        topic_domain=str((topic.get("domain") or {}).get("display_name") or ""),
        topic_field=str((topic.get("field") or {}).get("display_name") or ""),
        authors=authors,
        year=year,
        venue=hit.get("publication_venue_name_unified") or hit.get("venue"),
        url=_oa_pdf_url(hit) or hit.get("url") or _derive_url(hit),
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


def _title_key(title: str) -> str:
    """Version-insensitive title identity: lowercase, punctuation-free, no "v2" tail.

    Same-work duplicates (preprint vs. published version) often carry different DOIs,
    so paper_id alone cannot collapse them.
    """
    tokens = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
    while tokens and re.fullmatch(r"v\d+", tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


def dedup(papers: list[RetrievedPaper]) -> list[RetrievedPaper]:
    seen, titles, out = {}, {}, []
    for p in papers:
        existing = seen.get(p.paper_id)
        title_key = _title_key(p.title)
        if existing is None and title_key:
            existing = titles.get(title_key)
        if existing is not None:
            existing.survey_ref_count = max(existing.survey_ref_count, p.survey_ref_count)
            for hint in p.survey_ref_hints:
                if hint not in existing.survey_ref_hints:
                    existing.survey_ref_hints.append(hint)
            continue
        seen[p.paper_id] = p
        if title_key:
            titles[title_key] = p
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


# Off-topic families observed surviving the keyword gate in real runs (diagnosis
# §2: a whole crypto-mining section, drug-response and remote-sensing papers).
# Anchored phrases only — bare "mining" would catch Minecraft-domain work.
_OFFTOPIC_PATTERNS = [
    r"cryptocurrenc|bitcoin|ethereum|blockchain|mining (reward|pool|rig)|proof of work|hash rate",
    r"\bdrug\b|drug response|clinical (trial|patient|outcome)|patient cohort|disease|cancer|biomarker|gene expression|pharmac",
    r"remote sensing|land cover|satellite (imag|image)|hyperspectral|earth observation|lidar",
    r"\binsect|entomolog|pollinat|pest (species|control|management)",
    r"wind (farm|turbine) layout|power grid (scheduling|dispatch)|traffic (flow|signal) prediction",
]
_OFFTOPIC_RES = [re.compile(p) for p in _OFFTOPIC_PATTERNS]

# meta-search column projection: everything _to_retrieved consumes, plus
# primary_topic for the wave-5 topic gate (fields REPLACES the default set).
_META_FIELDS = [
    "unique_id", "doc_id", "title", "abstract", "author", "doi",
    "publication_published_year", "publication_venue_name_unified",
    "keywords", "citation_count", "access_oa_url", "locations",
    "primary_topic",
]


def _offtopic_text(title: str, abstract: str = "") -> bool:
    """Hard negative gate on observable text; independent of aspect scoring so
    'game/benchmark/model' token overlap can no longer smuggle these through."""
    text = _normalize_relevance_text(f"{title} {abstract[:300]}")
    return any(rx.search(text) for rx in _OFFTOPIC_RES)


# Wave 5 (1): structured domain/field blacklist. Phrasing patterns cannot catch
# titles like "Sora phyllopa Zhou & Chen 2024, sp. nov." (a cephalopod species
# description named after the video model) or a pharma congress abstract —
# SciVerse's OpenAlex-shaped primary_topic labels judge them wordlessly.
# Missing labels pass: the gate must never outguess an absent signal.
_TOPIC_REJECT_DOMAINS = {"life sciences", "health sciences"}
_TOPIC_REJECT_FIELDS = {
    "medicine", "agricultural and biological sciences", "health professions",
    "pharmacology, toxicology and pharmaceutics", "neuroscience", "nursing",
    "veterinary", "dentistry",
}


def _topic_gate_ok(paper: RetrievedPaper) -> bool:
    domain = (paper.topic_domain or "").strip().lower()
    field = (paper.topic_field or "").strip().lower()
    if domain in _TOPIC_REJECT_DOMAINS or field in _TOPIC_REJECT_FIELDS:
        return False
    return True


def _drop_offtopic(papers: list[RetrievedPaper], where: str) -> list[RetrievedPaper]:
    kept = [p for p in papers
            if not _offtopic_text(p.title, p.abstract) and _topic_gate_ok(p)]
    if len(kept) != len(papers):
        dropped = [f"{p.title[:50]} [{p.topic_domain or p.topic_field or 'pattern'}]"
                   for p in papers if p not in kept]
        logger.warning(f"[P3] offtopic gate dropped {len(papers) - len(kept)} at {where}: {dropped[:6]}")
    return kept


def topic_sim_min() -> float:
    return _env_float("EVISURVEY_TOPIC_SIM_MIN", 0.60)


# Words dropped from the topic string before it can veto a paper: pure syntax
# and survey-generic vocabulary that would match almost anything.
_TOPIC_TERM_STOPWORDS = GENERIC_RELEVANCE_TERMS | {
    "or", "for", "of", "in", "on", "to", "with", "via", "from", "at", "by",
    "is", "are", "towards", "toward", "survey", "surveys", "review", "reviews",
    "applications", "application", "overview", "introduction", "advances",
}

_EMBED_MODEL = None


def _load_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(
            os.getenv("EVISURVEY_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    return _EMBED_MODEL


def _topic_core_terms(topic: str) -> list[str]:
    return [t for t in _normalize_relevance_text(topic).split()
            if len(t) >= 3 and t not in _TOPIC_TERM_STOPWORDS][:6]


def _drop_topic_mismatch(papers: list[RetrievedPaper], topic: str, where: str) -> list[RetrievedPaper]:
    """Positive topical-fit gate for corpus admission (wave 7).

    The negative pattern/domain gates cannot catch a paper whose title merely
    echoes aspect keywords (the FER survey: Social Sciences/Psychology, zero
    game/world-model content, 0.557 cosine to the topic). Dual condition so a
    borderline on-topic paper is never killed: similarity below the floor AND
    no topic-core term anywhere in title+abstract. Embedding-model failures
    fail open (gate skips) - the gate must never outguess a missing signal.
    """
    core_terms = _topic_core_terms(topic)
    if not core_terms:
        return papers
    try:
        embedder = _load_embedder()
        topic_vec = embedder.encode([topic], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
    except Exception as exc:  # model unavailable -> fail open, loudly
        logger.warning(f"[P3] topic-embedding gate unavailable, skipping ({exc})")
        return papers
    kept, dropped = [], []
    texts = [f"{p.title} {p.abstract[:1500]}".strip() for p in papers]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    floor = topic_sim_min()
    for paper, vec, text in zip(papers, vectors, texts):
        sim = float((topic_vec * vec).sum())
        term_hit = any(term in _normalize_relevance_text(text) for term in core_terms)
        if sim < floor and not term_hit:
            dropped.append(f"{paper.title[:50]} [sim={sim:.3f}]")
        else:
            kept.append(paper)
    if dropped:
        logger.warning(f"[P3] topic-fit gate dropped {len(dropped)} at {where} "
                       f"(floor={floor}): {dropped[:6]}")
    return kept


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


def _expand_aspect_queries(keywords: list[str], queries: list[str], max_queries: int) -> list[str]:
    """Extend the legacy query list with per-keyword queries up to max_queries.

    The legacy list (one joined-keywords query, or the LLM-translated ones) is never
    truncated, so the default width keeps the delivered search behavior.
    """
    if len(queries) >= max_queries:
        return queries
    out = list(queries)
    for raw_keyword in keywords:
        query = str(raw_keyword).strip()
        if not query or query in out:
            continue
        out.append(query)
        if len(out) >= max_queries:
            break
    return out


def _landmark_query(aspect: dict) -> str:
    """One landmark-flavored variant per aspect: survey framing, no year window.

    Aspect keyword queries only reach fresh work; the field's canonical staples
    (World Models, Dreamer, MuZero) predate the 2018 window and surface through
    citation-boosted survey phrasing instead.
    """
    base = str(aspect.get("aspect_name") or "").strip()
    if not base:
        keywords = [str(k).strip() for k in aspect.get("keywords", []) if str(k).strip()]
        base = keywords[0] if keywords else ""
    return f"{base} survey".strip()


def _aspect_label(aspect: dict, index: int) -> str:
    return str(aspect.get("aspect_id") or f"aspect_{index + 1:03d}")


def _aspect_buckets(papers: list[RetrievedPaper], aspect_of: dict[str, list[str]], label_order: list[str]) -> dict[str, list[RetrievedPaper]]:
    buckets: dict[str, list[RetrievedPaper]] = {}
    for paper in papers:
        for label in aspect_of.get(paper.paper_id) or [EXPANSION_BUCKET]:
            buckets.setdefault(label, []).append(paper)
    ordered = {label: buckets.pop(label) for label in label_order if label in buckets}
    ordered.update(buckets)
    return ordered


def _trim_corpus_balanced(papers: list[RetrievedPaper], buckets: dict[str, list[RetrievedPaper]], cap: int) -> list[RetrievedPaper]:
    """Aspect-balanced corpus cap: round-robin over aspect buckets so a single aspect
    cannot crowd the others out of the corpus the way a global top-slice does."""
    if cap <= 0 or len(papers) <= cap:
        return papers
    kept: list[RetrievedPaper] = []
    kept_ids: set[str] = set()
    taken = [0] * len(buckets)
    while len(kept) < cap:
        added = False
        for i, bucket in enumerate(buckets.values()):
            if taken[i] >= len(bucket):
                continue
            paper = bucket[taken[i]]
            taken[i] += 1
            if paper.paper_id in kept_ids:
                continue
            kept_ids.add(paper.paper_id)
            kept.append(paper)
            added = True
            if len(kept) >= cap:
                break
        if not added:
            break
    return kept


def _retain_expansion_hits(ranked: list[RetrievedPaper], kept: list[RetrievedPaper]) -> list[RetrievedPaper]:
    """Hard retention: verified survey_expansion hits survive the corpus cap.

    Bib-sourced landmarks are the field's pseudo-gold, so the cap trims around
    them instead of deleting them; the relevance prefilter above still applies
    unchanged (only papers already inside the ranked list are re-admitted).
    """
    kept_ids = {p.paper_id for p in kept}
    pinned = [p for p in ranked if p.paper_id not in kept_ids and p.source == "survey_expansion"]
    if pinned:
        logger.info(f"[P3] corpus cap hard-kept {len(pinned)} survey_expansion papers past the cap")
    return kept + pinned


def _rescue_aspect_minima(
    kept: list[RetrievedPaper],
    dropped: list[RetrievedPaper],
    aspect_of: dict[str, list[str]],
    aspect_terms: dict[str, list[str]],
    min_per_aspect: int,
) -> list[RetrievedPaper]:
    """Re-admit each aspect's own best dropped papers until the coverage floor holds.

    Provenance is the relevance signal here: these hits came from a search issued for
    that aspect, so they are admissible even when the pooled lexical prefilter scored
    them 0. Off by default (EVISURVEY_ASPECT_MIN_PAPERS=0).
    """
    if min_per_aspect <= 0 or not dropped:
        return kept
    kept_ids = {id(p) for p in kept}
    out = list(kept)
    for label, terms in aspect_terms.items():
        owned = [p for p in dropped if label in (aspect_of.get(p.paper_id) or [])]
        if not owned:
            continue
        survivors = sum(1 for p in out if label in (aspect_of.get(p.paper_id) or []))
        if survivors >= min_per_aspect:
            continue
        scored = sorted(
            ((_paper_relevance_score(p, [{"keywords": terms}]), p) for p in owned),
            key=lambda item: item[0],
            reverse=True,
        )
        for _, paper in scored:
            if survivors >= min_per_aspect:
                break
            if id(paper) in kept_ids:
                continue
            kept_ids.add(id(paper))
            out.append(paper)
            survivors += 1
    return out


def _filter_relevant_candidates(papers: list[RetrievedPaper], aspects: list[dict]) -> list[RetrievedPaper]:
    if not papers or not _aspect_relevance_terms(aspects):
        return papers
    min_score = relevance_min_score()
    filtered = [
        paper for paper in papers
        if _paper_relevance_score(paper, aspects) >= min_score
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
    # A full title is the most selective query: on SciVerse the original paper is
    # the top hit, while alias+year queries return year-matched unrelated work.
    title = str(candidate.get("title", "") or "").strip()
    if title:
        return title
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


def _normalize_title(title: str) -> str:
    """Canonical-title equality: NFKC, lowercase, non-alphanumerics folded to spaces.

    Local copy of the evaluate_survey convention (no cross-module import);
    CJK letters stay significant so non-English titles cannot all collapse to "".
    """
    folded = unicodedata.normalize("NFKC", str(title or "")).lower()
    return " ".join(re.sub(r"[^a-z0-9一-鿿]+", " ", folded).split())


def _candidate_matches_hit(candidate: dict, hit: dict) -> bool:
    title = str(candidate.get("title", "") or "").strip()
    if title:
        # Titled bib candidates accept the hit only when it is the cited paper
        # itself; token matching would pass derivative/survey papers instead.
        return _normalize_title(hit.get("title", "") or "") == _normalize_title(title)
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


def _search_expansion_candidates(expansion_candidates, sciverse, retrieved, first_seen_rank,
                                 source="survey_expansion"):
    for candidate in expansion_candidates:
        title = str(candidate.get("title", "") or "")
        if _offtopic_text(title, str(candidate.get("query", "") or "")):
            logger.info(f"[P3] expansion candidate rejected by offtopic gate: {title[:60]}")
            continue
        query = _candidate_query(candidate)
        if not query:
            continue
        year = _candidate_year(candidate)
        # arXiv vs journal publication years disagree on SciVerse (MuZero has
        # 2019 and 2020 entries), so widen the hint year by one instead of an
        # exact match; without a year the title query stands alone.
        filters = _year_filters(year - 1, year + 1) if year else []
        try:
            res = sciverse.meta_search(query=query, filters=filters, impact_boost="MILD",
                                       page_size=3, fields=_META_FIELDS)
            hits = res.get("results", [])
            logger.info(f"[P3] expansion_search q={query!r} filters={filters} -> {len(hits)} hits")
            for hit in hits:
                if not _candidate_matches_hit(candidate, hit):
                    continue
                paper = _to_retrieved(hit, source=source)
                if not _topic_gate_ok(paper):
                    logger.info(f"[P3] injected candidate rejected by topic gate: "
                                f"{paper.title[:50]} [{paper.topic_domain or paper.topic_field}]")
                    continue
                paper.survey_ref_count = int(candidate.get("survey_ref_count") or 1)
                # reference-seed candidates carry no paper_id_hint (they come
                # from relations titles, not survey bibs) — the hint is optional
                paper.survey_ref_hints = [str(candidate.get("paper_id_hint"))] \
                    if candidate.get("paper_id_hint") else []
                first_seen_rank.setdefault(paper.paper_id, len(retrieved))
                retrieved.append(paper)
        except Exception as e:
            logger.warning(f"[P3] expansion_search failed q={query!r}: {e}")


def _stage_snapshot(papers: list[RetrievedPaper]) -> list[dict]:
    return [{"paper_id": p.paper_id, "title": p.title} for p in papers]


def _search_reference_seeds(sciverse, retrieved, first_seen_rank):
    """Classics from the core papers' OWN reference lists (meta-paper-relations,
    POST unique_id): a second verifiable bib source alongside the seed-survey
    expansion path. References cited by >=2 core papers, not already in the
    corpus, passing the offtopic gate, capped by EVISURVEY_REFERENCE_SEEDS."""
    if sciverse is None or not retrieved:
        return
    # rank DISTINCT papers — the corpus still holds aspect-window duplicates here
    top, seen_ids = [], set()
    for p in sorted(retrieved, key=lambda q: q.citation_count, reverse=True):
        if p.paper_id.startswith("paper:") and p.paper_id not in seen_ids:
            top.append(p)
            seen_ids.add(p.paper_id)
        if len(top) >= 5:
            break
    freq: dict[str, tuple[str, int]] = {}  # norm title -> (raw title, count)
    for p in top:
        try:
            res = sciverse.meta_paper_relations(p.paper_id, relation="REFERENCES", page=1, page_size=200)
        except Exception as e:
            logger.warning(f"[P3] reference-seed relations failed for {p.paper_id}: {e}")
            continue
        items = (res.get("data") or res).get("items") or []
        for item in items:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            norm = _normalize_title(title)
            raw, count = freq.get(norm, (title, 0))
            freq[norm] = (raw, count + 1)
    cap = _env_int("EVISURVEY_REFERENCE_SEEDS", 8)
    existing = {_normalize_title(p.title) for p in retrieved}
    candidates = [
        {"title": raw, "survey_ref_count": count}
        for norm, (raw, count) in freq.items()
        if count >= 2 and norm not in existing and not _offtopic_text(raw)
    ]
    candidates.sort(key=lambda c: c["survey_ref_count"], reverse=True)
    candidates = candidates[:cap]
    if not candidates:
        logger.info("[P3] reference seeds: none eligible "
                    f"({len(freq)} distinct references from {len(top)} core papers)")
        return
    logger.info(f"[P3] reference seeds: injecting {len(candidates)} classics "
                f"(top freq={candidates[0]['survey_ref_count']})")
    _search_expansion_candidates(candidates, sciverse, retrieved, first_seen_rank,
                                 source="reference_seed")


def stage_trace_path() -> str:
    """Survival-chain trace target; EVISURVEY_P3_STAGE_TRACE redirects it (tests)."""
    override = (os.getenv("EVISURVEY_P3_STAGE_TRACE") or "").strip()
    return override or os.path.join("cache", "p3_stage_trace.json")


def _write_stage_trace(task_id, cap_mode: str, stages: dict[str, list[dict]]) -> None:
    """Diagnostic only (specs/canonical存活链诊断.md): record the candidate set at
    each survival-chain checkpoint so audit_canonical_survival can locate where a
    canonical landmark died. Retrieval/ranking behavior is untouched."""
    path = stage_trace_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"task_id": task_id, "cap_mode": cap_mode, "stages": stages},
            f, ensure_ascii=False, indent=2,
        )
    logger.info(
        "[P3] stage trace -> %s (raw_hits=%d after_prefilter=%d after_rank_cut=%d "
        "after_corpus_cap=%d)",
        path, len(stages["raw_hits"]), len(stages["after_prefilter"]),
        len(stages["after_rank_cut"]), len(stages["after_corpus_cap"]),
    )


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
    topic: str = "",
):
    logger.info(f"[P3] retrieve: {len(aspects)} aspects, use_mineru={pipeline_config.use_mineru}, "
                f"use_seed_fallback={pipeline_config.use_seed_fallback}, "
                f"use_influence_score={pipeline_config.use_influence_score}")
    # S1 breadth/depth knobs (specs/检索广度与来源深度.md). Defaults reproduce the
    # delivered behavior; an explicit EVISURVEY_MAX_CORPUS both overrides the cap and
    # switches trimming from a global top-slice to aspect-balanced.
    knob_queries = max_queries_per_aspect()
    knob_page_size = meta_page_size(pipeline_config.use_influence_score)
    knob_aspect_min = aspect_min_papers()
    raw_corpus_cap = (os.getenv("EVISURVEY_MAX_CORPUS") or "").strip()
    corpus_cap = int(raw_corpus_cap) if raw_corpus_cap else pipeline_config_extra(pipeline_config, "max_papers", 40)
    core_limit = pipeline_config_extra(pipeline_config, "max_core_papers", 15)
    logger.info(f"[P3] breadth knobs: queries/aspect>={knob_queries} page_size={knob_page_size} "
                f"relevance_min={relevance_min_score()} aspect_min_papers={knob_aspect_min} "
                f"max_corpus={corpus_cap}({'aspect-balanced' if raw_corpus_cap else 'top-slice'}) "
                f"fulltext_core_papers={core_limit if pipeline_config.use_mineru else 0}")
    retrieved = []
    first_seen_rank = {}
    relevance_aspects = list(aspects)
    # paper_id -> provenance labels (aspects in search order, then expansion/seed);
    # drives the per-aspect coverage floor and the aspect-balanced corpus cap.
    aspect_of: dict[str, list[str]] = {}
    label_order = [_aspect_label(a, i) for i, a in enumerate(aspects)] + [EXPANSION_BUCKET, SEED_BUCKET]

    # 1. expansion candidates from curated surveys: exact, small-page searches.
    expansion_start = len(retrieved)
    _search_expansion_candidates(expansion_candidates, sciverse, retrieved, first_seen_rank)
    for p in retrieved[expansion_start:]:
        labels = aspect_of.setdefault(p.paper_id, [])
        if EXPANSION_BUCKET not in labels:
            labels.append(EXPANSION_BUCKET)

    # 2. meta-search per aspect (verified SciVerse filter/response contract)
    filter_sets = (
        _influence_filter_sets()
        if pipeline_config.use_influence_score
        else [_year_filters(2018, 2026)]
    )
    aspect_terms: dict[str, list[str]] = {}
    for a_index, a in enumerate(aspects):
        label = _aspect_label(a, a_index)
        keywords = a.get("keywords", [])
        # If keywords contain non-ASCII (Chinese topic) and LLM available, generate English queries
        if llm and any(_has_non_ascii(kw) for kw in keywords):
            queries = _generate_search_queries(keywords, llm)
            logger.info(f"[P3] aspect {label} translate -> {queries}")
        else:
            queries = [" ".join(keywords)]
        queries = _expand_aspect_queries(keywords, queries, knob_queries)
        landmark = _landmark_query(a)
        relevance_aspects.append({"keywords": queries})
        aspect_terms[label] = list(keywords) + list(queries)
        # (query, filters, freshness-biased): the landmark variant runs with no year
        # window and no freshness boost — influence only. Its query text stays out of
        # the pooled relevance terms so it cannot activate a prefilter that the
        # aspect's own keywords would leave inert.
        searches = [(query, filters, True) for query in queries for filters in filter_sets]
        if landmark:
            searches.append((landmark, [], False))
        for query, filters, freshness_biased in searches:
            try:
                kwargs = {
                    "query": query,
                    "filters": filters,
                    "impact_boost": "MILD",
                    "page_size": knob_page_size,
                    # fields is a PROJECTION, not an addition: asking for
                    # primary_topic alone strips unique_id/title/abstract and the
                    # whole corpus collapses into one deduped seed-hash id (live
                    # 2026-09-10). The list must cover every column retrieval
                    # consumes, primary_topic included (default_returned=0).
                    "fields": _META_FIELDS,
                }
                if pipeline_config.use_influence_score and freshness_biased:
                    kwargs["freshness_boost"] = "MILD"
                res = sciverse.meta_search(**kwargs)
                hits = res.get("results", [])
                variant = "landmark " if not freshness_biased else ""
                logger.info(f"[P3] meta_search {variant}q={query!r} filters={filters} -> {len(hits)} hits")
                for hit in hits:
                    paper = _to_retrieved(hit)
                    first_seen_rank.setdefault(paper.paper_id, len(retrieved))
                    labels = aspect_of.setdefault(paper.paper_id, [])
                    if label not in labels:
                        labels.append(label)
                    retrieved.append(paper)
            except Exception as e:
                logger.warning(f"[P3] meta_search failed q={query!r} filters={filters}: {e}")
    # 3. seed fallback if empty
    if not retrieved and pipeline_config.use_seed_fallback:
        logger.warning(f"[P3] empty retrieval -> seed fallback ({len(seed_papers)} papers)")
        retrieved = [_to_retrieved(s, source="seed") for s in seed_papers]
        first_seen_rank = {p.paper_id: i for i, p in enumerate(retrieved)}
        for p in retrieved:
            labels = aspect_of.setdefault(p.paper_id, [])
            if SEED_BUCKET not in labels:
                labels.append(SEED_BUCKET)
    # 3.5 classics from the core papers' own reference lists (verifiable bib)
    seed_start = len(retrieved)
    _search_reference_seeds(sciverse, retrieved, first_seen_rank)
    for p in retrieved[seed_start:]:
        labels = aspect_of.setdefault(p.paper_id, [])
        if EXPANSION_BUCKET not in labels:
            labels.append(EXPANSION_BUCKET)
    stage_raw_hits = _stage_snapshot(retrieved)
    retrieved = dedup(retrieved)
    # off-topic hard gate AFTER dedup so neither the relevance prefilter's
    # rescue path nor the corpus cap can resurrect what it rejects
    retrieved = _drop_offtopic(retrieved, "post-dedup")
    gate_topic = topic.strip() or "; ".join(
        str(a.get("aspect_name") or "") for a in aspects)
    retrieved = _drop_topic_mismatch(retrieved, gate_topic, "post-dedup")
    if pipeline_config.use_influence_score:
        survived = _filter_relevant_candidates(retrieved, relevance_aspects)
        survived_ids = {id(p) for p in survived}
        dropped = [p for p in retrieved if id(p) not in survived_ids]
        survived = _rescue_aspect_minima(survived, dropped, aspect_of, aspect_terms, knob_aspect_min)
        retrieved = _rank_by_influence(survived, first_seen_rank, relevance_aspects)
    stage_after_prefilter = _stage_snapshot(retrieved)
    if raw_corpus_cap:
        # Aspect-balanced cap: the trim is attributed to corpus_cut, so the rank
        # checkpoint records the uncut ranked list.
        stage_after_rank_cut = _stage_snapshot(retrieved)
        buckets = _aspect_buckets(retrieved, aspect_of, label_order)
        retrieved = _retain_expansion_hits(
            retrieved, _trim_corpus_balanced(retrieved, buckets, corpus_cap))
    else:
        # Default top-slice: rank position decides survival, so this cut lands on
        # the rank checkpoint and the corpus-cap checkpoint records the same set
        # (hard-kept expansion hits are part of the cut, never cut by it).
        retrieved = _retain_expansion_hits(retrieved, retrieved[:corpus_cap])
        stage_after_rank_cut = _stage_snapshot(retrieved)
    stage_after_corpus_cap = _stage_snapshot(retrieved)
    logger.info(f"[P3] after dedup/cap: {len(retrieved)} papers")
    _write_stage_trace(
        task_id,
        "aspect-balanced" if raw_corpus_cap else "top-slice",
        {
            "raw_hits": stage_raw_hits,
            "after_prefilter": stage_after_prefilter,
            "after_rank_cut": stage_after_rank_cut,
            "after_corpus_cap": stage_after_corpus_cap,
        },
    )
    # 4. own-content first, then MinerU for the core window; every paper gets an
    # accounted outcome so parse success rates have honest denominators (a paper
    # with a SciVerse doc_id pays ONE cheap /content call — when it returns the
    # fulltext the MinerU submit/poll/download cycle is skipped entirely).
    # Fulltext via MinerU stays bounded to the core window: non-core papers that
    # miss content stay abstract_only instead of paying a MinerU parse each.
    parsed = []
    parse_records = []  # {paper_id, outcome, skip_reason, failures: [{channel, stage, kind}]}
    for index, p in enumerate(retrieved):
        failures = []
        md = ""
        if sciverse is not None and getattr(p, "doc_id", ""):
            try:
                md = sciverse.read_full_text(p.doc_id)
            except Exception as e:
                logger.warning(f"[P3] content fetch failed {p.paper_id}: {e}")
                failures.append({"channel": "content", "stage": "fetch", "kind": type(e).__name__})
        if len(md.strip()) > 2000:
            try:
                raw = _markdown_parsed(md, parse_status="content_fulltext")
                raw["figures"] = _content_figures(md, sciverse, p.doc_id)
                raw = cleaner.clean_parsed(raw)
                paper = _to_parsed(p, raw)
                if paper.paragraphs or paper.sections:
                    p.parse_status = "content_fulltext"
                    parsed.append(paper)
                    parse_records.append({"paper_id": p.paper_id, "outcome": "content_fulltext",
                                          "skip_reason": "", "failures": failures})
                    continue
            except Exception as e:
                logger.warning(f"[P3] content parse failed {p.paper_id}: {e}")
                failures.append({"channel": "content", "stage": "unpack", "kind": type(e).__name__})
        if (not p.url or not pipeline_config.use_mineru or index >= core_limit
                or not _is_pdf_url(p.url)):
            p.parse_status = "abstract_only"
            reason = ("no_pdf_url" if not p.url or not _is_pdf_url(p.url)
                      else ("out_of_window" if index >= core_limit else "mineru_disabled"))
            parse_records.append({"paper_id": p.paper_id, "outcome": "abstract_only",
                                  "skip_reason": reason, "failures": failures})
            continue
        try:
            # v4 extract_fulltext is the main channel; the agent parse endpoint is the
            # fallback. Clients without extract_fulltext (legacy fakes) fall through too.
            try:
                raw = mineru.extract_fulltext(p.url)
            except Exception as e:
                stage = getattr(e, "stage", "poll")
                kind = getattr(e, "kind", type(e).__name__)
                logger.warning(f"[P3] mineru v4 fulltext failed {p.paper_id} "
                               f"(stage={stage} kind={kind}); falling back to agent parse")
                failures.append({"channel": "v4", "stage": stage, "kind": kind})
                raw = mineru.parse_url(p.url, light=True)
            raw = cleaner.clean_parsed(raw)
            paper = _to_parsed(p, raw)
            if not paper.paragraphs and not paper.sections:
                p.parse_status = "abstract_only"
                parse_records.append({"paper_id": p.paper_id, "outcome": "abstract_only",
                                      "skip_reason": "empty_parse", "failures": failures})
            else:
                p.parse_status = raw.get("parse_status") or "light"
                parsed.append(paper)
                parse_records.append({"paper_id": p.paper_id, "outcome": p.parse_status,
                                      "skip_reason": "", "failures": failures})
        except Exception as e:
            logger.warning(f"[P3] mineru parse failed {p.paper_id}: {e}")
            failures.append({"channel": "agent", "stage": "parse", "kind": type(e).__name__})
            p.parse_status = "abstract_only"
            parse_records.append({"paper_id": p.paper_id, "outcome": "abstract_only",
                                  "skip_reason": "mineru_failed", "failures": failures})
    _log_parse_accounting(retrieved, parse_records)
    _write_parse_trace(task_id, parse_records)
    return (RetrievedPapers(task_id=task_id, papers=retrieved),
            ParsedPapers(task_id=task_id, papers=parsed))


_CONTENT_IMAGE_RE = re.compile(r"!\[([^\]]*[^\s\]][^\]]*)\]\(([^)\s]+)\)")


def _content_figures(md: str, sciverse, doc_id: str) -> list[dict]:
    """Captioned figures embedded in a /content fulltext, downloaded via
    /resource into cache/sciverse_assets/<doc_id>/. Bounded by
    EVISURVEY_CONTENT_FIGURES_MAX (default 8); a failed download keeps the
    captioned record with img_path=None rather than failing the parse."""
    cap = _env_int("EVISURVEY_CONTENT_FIGURES_MAX", 8)
    figures: list[dict] = []
    seen: set[str] = set()
    for caption, path in _CONTENT_IMAGE_RE.findall(md):
        if len(figures) >= cap:
            break
        if path in seen:
            continue
        seen.add(path)
        record = {"num": len(figures) + 1, "page": None,
                  "caption": caption.strip(), "img_path": None}
        try:
            data = sciverse.get_resource(path)
            if data:
                out_dir = os.path.join("cache", "sciverse_assets", re.sub(r"[^a-z0-9]", "", doc_id.lower())[:24])
                os.makedirs(out_dir, exist_ok=True)
                suffix = os.path.splitext(path)[1][:5] or ".jpg"
                rel = os.path.join(out_dir, f"fig{record['num']}{suffix}")
                with open(rel, "wb") as f:
                    f.write(data)
                record["img_path"] = rel
        except Exception as e:
            logger.warning(f"[P3] content figure download failed ({path[:60]}): {e}")
        figures.append(record)
    if figures:
        n_assets = sum(1 for f in figures if f["img_path"])
        logger.info(f"[P3] content figures: {len(figures)} captioned, {n_assets} assets on disk")
    return figures


def _log_parse_accounting(retrieved, records):
    """One honest-denominator line (diagnosis §3.1: '2/70' was never a success
    rate) — outcomes, skip reasons and failure (channel, stage, kind) counts."""
    outcomes = {}
    skips = {}
    fails = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        if r["skip_reason"]:
            skips[r["skip_reason"]] = skips.get(r["skip_reason"], 0) + 1
        for f in r["failures"]:
            k = f"{f['channel']}/{f['stage']}/{f['kind']}"
            fails[k] = fails.get(k, 0) + 1
    total = len(records)
    parsed_count = total - outcomes.get("abstract_only", 0)
    parts = [f"content_fulltext={outcomes.get('content_fulltext', 0)}",
             f"mineru={sum(v for k, v in outcomes.items() if k in ('fulltext', 'light'))}",
             f"abstract_only={outcomes.get('abstract_only', 0)}"]
    if skips:
        parts.append("(" + " ".join(f"{k}={v}" for k, v in sorted(skips.items())) + ")")
    if fails:
        parts.append("failures: " + " ".join(f"{k}={v}" for k, v in sorted(fails.items())))
    logger.info(f"[P3] parse accounting: {parsed_count}/{total} parsed — " + ", ".join(parts))


def _write_parse_trace(task_id, records):
    override = (os.getenv("EVISURVEY_P3_PARSE_TRACE") or "").strip()
    path = override or os.path.join("cache", "parse_trace.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"task_id": task_id, "records": records}, f, ensure_ascii=False, indent=1)
        logger.info(f"[P3] parse trace -> {path} ({len(records)} papers)")
    except OSError as e:
        logger.warning(f"[P3] parse trace write failed: {e}")


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
        parse_status=raw.get("parse_status") or "light",
    )
