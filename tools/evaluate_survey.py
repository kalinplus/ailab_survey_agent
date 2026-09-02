"""Post-hoc survey evaluator — four-layer offline audit over existing artifacts.

Layers (specs/评测体系v2-离线四层改造.md):
  L0   deterministic profile: freshness / structure / redundancy (no LLM, no network)
  L1   citation quality: AutoSurvey/ALCE-style NLI metrics on (sentence, citation) pairs
  L1.5 uncited-claim verification: SciVerse agentic-search + NLI (skippable, resumable)
  L2'  corpus-grounded coverage over the P5 taxonomy (+ optional gold-reference recall)
  L3   academic value: DeepSurvey-Bench three-dimension judge

Standalone audit tool over existing run artifacts (output/survey.md,
cache/evidence_store.json); deliberately NOT part of the agent pipeline.
Entry point: scripts/run_survey_eval.py.
"""

import json
import logging
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.。])\s+")
_CITATION = re.compile(r"\[([^\]]+)\]")
_STRUCTURAL = ("#", "|", "!", ">")
_FIGURE_EMBED = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _body_text(md: str) -> str:
    """Drop headings / tables / images / quotes so only prose is measured."""
    lines = [ln for ln in md.splitlines() if not ln.lstrip().startswith(_STRUCTURAL)]
    return "\n".join(lines)


def _sentences(md: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(_body_text(md)) if s.strip()]


def _is_citation(cited_id: str, known_ids: set[str] | None) -> bool:
    """Bracket content counts as a citation only if it is a paper id
    ('paper:...' convention) or resolvable in the evidence store — figure/
    table references like '[Future Matrix]' are not citations."""
    return cited_id.startswith("paper:") or (known_ids is not None and cited_id in known_ids)


def _iter_sentence_claims(md: str, known_ids: set[str] | None = None):
    """Yield (sentence_index, claim_text, citation_ids) per cited sentence."""
    for i, sentence in enumerate(_sentences(md)):
        cites = [c for c in _CITATION.findall(sentence) if _is_citation(c, known_ids)]
        text = _CITATION.sub("", sentence).strip().rstrip(".。").strip()
        if cites and text:
            yield i, text, cites


def extract_claim_pairs(md: str, known_ids: set[str] | None = None) -> list[tuple[str, str]]:
    """Every (claim sentence, citation) pair — unlike claim_mapper, which
    binds only the first citation of a sentence."""
    return [(text, c) for _, text, cites in _iter_sentence_claims(md, known_ids) for c in cites]


_LABEL_RANK = {"entailment": 2, "neutral": 1, "contradiction": 0}
_LABEL_STATUS = {"entailment": "supported", "neutral": "weak", "contradiction": "unsupported"}
_SENT_BOUNDARY = re.compile(r"(?<=[.!?。])\s+")
_MIN_UNIT_CHARS = 20


def _evidence_units(texts: list[str]) -> list[str]:
    """Sentence windows: cross-encoder NLI fails on long-passage premises
    (a verbatim quote inside a 1700-char abstract scores neutral), so each
    multi-sentence evidence text is split into sentence-level units."""
    units: list[str] = []
    for t in texts:
        parts = [p.strip() for p in _SENT_BOUNDARY.split(t)]
        if len(parts) == 1:
            units.append(t.strip())
            continue
        units.extend(p for p in parts if len(p) >= _MIN_UNIT_CHARS)
    return units or [t.strip() for t in texts]


def _best_judgment(nli, claim: str, evidences: list):
    """Judge claim against every evidence unit directly (no template
    wrapper) and keep the best verdict: entailment > neutral > contradiction,
    margin as tiebreaker."""
    best_key, best = None, None
    for unit in _evidence_units(evidences):
        r = nli.judge(unit, claim)
        key = (_LABEL_RANK[r.label], r.confidence)
        if best_key is None or key > best_key:
            best_key, best = key, r
    return best


def citation_quality(survey_md: str, evidence_store, nli) -> dict:
    """AutoSurvey-style citation metrics over (sentence, citation) pairs.

    Verdict per pair = NLI label (entail or not, ALCE/AutoSurvey convention):
    entailment→supported, neutral→weak, contradiction→unsupported. The
    cross-encoder margin is reported but does not gate the verdict — its
    distribution is compressed (identical text scores ~0.56).
    citation_recall: cited sentences with >=1 supported citation / cited sentences.
    citation_precision: supported citation instances / all citation instances.
    citation_coverage: cited sentences / all body sentences.
    """
    by_paper: dict[str, list] = {}
    for e in evidence_store.evidence:
        by_paper.setdefault(e.paper_id, []).append(e)
    known_ids = set(by_paper)

    sentences = _sentences(survey_md)
    n_cited_sentences = 0
    n_recalled_sentences = 0
    pairs: list[dict] = []

    for _, text, cites in _iter_sentence_claims(survey_md, known_ids):
        n_cited_sentences += 1
        sentence_supported = False
        for cited_id in cites:
            evidences = by_paper.get(cited_id, [])
            if evidences:
                res = _best_judgment(nli, text, [e.text for e in evidences])
                status = _LABEL_STATUS[res.label]
                conf = res.confidence
            else:
                status, conf = "unsupported", 0.0
            sentence_supported = sentence_supported or status == "supported"
            pairs.append({
                "claim_text": text,
                "cited_paper_id": cited_id,
                "status": status,
                "confidence": round(conf, 3),
            })
        if sentence_supported:
            n_recalled_sentences += 1

    n_pairs = len(pairs)
    supported_pairs = sum(1 for p in pairs if p["status"] == "supported")
    weak_pairs = sum(1 for p in pairs if p["status"] == "weak")
    unsupported_pairs = sum(1 for p in pairs if p["status"] == "unsupported")

    return {
        "n_sentences": len(sentences),
        "n_cited_sentences": n_cited_sentences,
        "n_pairs": n_pairs,
        "supported_pairs": supported_pairs,
        "weak_pairs": weak_pairs,
        "unsupported_pairs": unsupported_pairs,
        "citation_recall": round(n_recalled_sentences / n_cited_sentences, 3) if n_cited_sentences else 0.0,
        "citation_precision": round(supported_pairs / n_pairs, 3) if n_pairs else 0.0,
        "citation_coverage": round(n_cited_sentences / len(sentences), 3) if sentences else 0.0,
        "pairs": pairs,
    }


def ab_comparison(report_rows: list[dict]) -> list[dict]:
    """Convert repair A/B report rows into a recall-equivalent comparison table.

    Caveat (documented in the report): these per-config numbers come from the
    repair joint's claim_map, whose claim granularity is first-citation-per-
    sentence — coarser than citation_quality's pair level.
    """
    out = []
    for row in report_rows:
        total = row.get("total_claims", 0)
        supported = total - row.get("unsupported", 0) - row.get("weak", 0)
        out.append({
            "config": row.get("config"),
            "topic": row.get("topic"),
            "total_claims": total,
            "unsupported": row.get("unsupported", 0),
            "weak": row.get("weak", 0),
            "invalid_citations": row.get("invalid_citations", 0),
            "citation_validity_score": row.get("citation_validity_score"),
            "citation_recall_equiv": round(supported / total, 3) if total else 0.0,
        })
    return out


_TITLE_NOISE = re.compile(r"[^a-z0-9一-鿿]+")


def _normalize_title(title: str) -> str:
    return _TITLE_NOISE.sub("", title.lower())


def reference_coverage(
    system_titles: list[str],
    gold_refs: dict,
    scorer=None,
    fuzzy_threshold: float = 0.85,
) -> dict:
    """SurGE-style reference recall: how much of a real survey's reference
    list the system's paper set covers. Normalized-title exact match first,
    then embedding fuzzy match (via relevance.EmbeddingScorer) for leftovers."""
    gold_titles = [r["title"] for r in gold_refs.get("references", [])]
    gold_norm = {_normalize_title(t): t for t in gold_titles}
    malformed_norm = {_normalize_title(r["title"]) for r in gold_refs.get("references", [])
                      if r.get("malformed")}
    system_norm: list[str] = []
    seen: set[str] = set()
    for t in system_titles:
        n = _normalize_title(t)
        if n and n not in seen:
            seen.add(n)
            system_norm.append(n)

    matched: dict[str, str] = {}
    system_matched: set[str] = set()
    for n in system_norm:
        if n in gold_norm:
            matched[n] = gold_norm[n]
            system_matched.add(n)
    exact_matches = len(matched)

    # substring pass: ONLY for malformed bib entries kept as full text — a
    # clean gold title that merely contains a short system title ("World
    # models" ⊂ "Daydreamer: World models ...") is NOT a match
    substring_matches = 0
    for g_key, g_orig in gold_norm.items():
        if g_key not in malformed_norm:
            continue
        if g_key in matched:
            continue
        for n in system_norm:
            if n in g_key:
                matched[g_key] = g_orig
                system_matched.add(n)
                substring_matches += 1
                break

    fuzzy_matches = 0
    if scorer is not None:
        leftovers_g = [(n, orig) for n, orig in gold_norm.items() if n not in matched]
        leftovers_s = [n for n in system_norm if n not in system_matched]
        if leftovers_g and leftovers_s:
            pairs = [(g_orig, s) for _, g_orig in leftovers_g for s in leftovers_s]
            scores = scorer.score_pairs(pairs)
            width = len(leftovers_s)
            for gi, (g_key, g_orig) in enumerate(leftovers_g):
                row = scores[gi * width:(gi + 1) * width]
                best_j = max(range(width), key=lambda j: row[j])
                if row[best_j] >= fuzzy_threshold:
                    matched[g_key] = g_orig
                    system_matched.add(leftovers_s[best_j])
                    fuzzy_matches += 1

    total = len(gold_norm)
    matched_count = exact_matches + substring_matches + fuzzy_matches
    return {
        "gold_source_title": gold_refs.get("source", {}).get("title"),
        "n_gold_refs": total,
        "n_system_titles": len(system_norm),
        "exact_matches": exact_matches,
        "substring_matches": substring_matches,
        "fuzzy_matches": fuzzy_matches,
        "reference_recall": round(matched_count / total, 3) if total else 0.0,
        "missed_count": total - matched_count,
        "matched_titles": sorted(matched.values()),
        "system_in_gold_rate": round(len(system_matched) / len(system_norm), 3) if system_norm else 0.0,
    }


# ── section helpers shared by L0 / L2' / L1.5 ──────────────────────

# Trailing appendix sections written by the pipeline itself, not survey prose.
_META_SECTION_TITLES = {"references", "bibliography", "revision notes"}


def _is_body_section(title: str) -> bool:
    return title.strip().lower() not in _META_SECTION_TITLES


def _split_sections(md: str) -> list[tuple[str, str]]:
    """(heading, body) pairs split on '## ' headings; leading body without a
    heading becomes one '(untitled)' section. H1 lines are the document title,
    not a section, so they are dropped."""
    sections: list[tuple[str, str]] = []
    title, buf = None, []
    for line in md.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in buf):
                sections.append((title or "(untitled)", "\n".join(buf).strip()))
            title, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((title or "(untitled)", "\n".join(buf).strip()))
    return sections


def _body_sections(md: str) -> list[tuple[str, str]]:
    """(title, prose) for body sections only — meta sections are pipeline
    appendices, not survey prose."""
    return [(title, _body_text(body).strip()) for title, body in _split_sections(md)
            if _is_body_section(title)]


def _cited_ids(md: str, known_ids: set[str]) -> set[str]:
    """Paper ids cited anywhere in the body (meta sections excluded)."""
    body = "\n".join(body for title, body in _split_sections(md) if _is_body_section(title))
    return {c for c in _CITATION.findall(body) if _is_citation(c, known_ids)}


def _norm_pid(pid: str) -> str:
    """Case/prefix-insensitive paper identity: taxonomy and survey citations
    use both `world_models_2018` and `paper:<doi>` conventions."""
    low = pid.strip().lower()
    return low[6:] if low.startswith("paper:") else low


# ── L0 deterministic profile ───────────────────────────────────────

_REFERENCE_ENTRY = re.compile(r"^\s*[-*]?\s*((?:paper:)?[^:]+):\s*(.+?)\s*\((\d{4})\)\s*\.?\s*$")


def _reference_years(md: str) -> dict[str, int]:
    """paper id -> year from the survey's own References section
    (format `- paper:<doi>: <title> (YYYY).`)."""
    years: dict[str, int] = {}
    for title, body in _split_sections(md):
        if title.strip().lower() not in {"references", "bibliography"}:
            continue
        for line in body.splitlines():
            m = _REFERENCE_ENTRY.match(line)
            if m:
                years[m.group(1).strip()] = int(m.group(3))
    return years


def _median(values: list) -> float | None:
    return float(statistics.median(values)) if values else None


def _freshness(survey_md: str, papers: list[dict], ref_years: dict[str, int],
               current_year: int) -> dict:
    """Citation freshness vs the corpus it was drawn from. References entries
    whose year cannot be parsed are reported and excluded from the shares."""
    cited = _cited_ids(survey_md, set(ref_years))
    cited_years = [ref_years[c] for c in sorted(cited) if c in ref_years]
    corpus_years = [int(p["year"]) for p in papers if p.get("year")]
    floor = current_year - 1
    median_cited, median_corpus = _median(cited_years), _median(corpus_years)
    return {
        "recent_floor": floor,
        "n_cited_ids": len(cited),
        "years_unresolved": sorted(cited - set(ref_years)),
        "citation_year_median": median_cited,
        "corpus_year_median": median_corpus,
        "year_offset": median_cited - median_corpus
        if median_cited is not None and median_corpus is not None else None,
        "citation_recent_share": round(sum(1 for y in cited_years if y >= floor) / len(cited_years), 3)
        if cited_years else None,
        "corpus_recent_share": round(sum(1 for y in corpus_years if y >= floor) / len(corpus_years), 3)
        if corpus_years else None,
    }


def _structure(survey_md: str, ref_years: dict[str, int],
               figure_bank: list, table_bank: list) -> dict:
    """Per-section shape: prose-length balance, citation density, zero-citation
    sections, and figure/table embeds in the body vs the mined banks."""
    rows: list[dict] = []
    n_figure_refs = 0
    n_table_refs = 0
    for title, body in _split_sections(survey_md):
        if not _is_body_section(title):
            continue
        prose = _body_text(body).strip()
        sentences = [s for s in _SENT_SPLIT.split(prose) if s.strip()]
        cites = [c for c in _CITATION.findall(body)
                 if _is_citation(c, ref_years)]
        n_figure_refs += len(_FIGURE_EMBED.findall(body))
        n_table_refs += _table_blocks(body)
        rows.append({
            "section": title,
            "prose_chars": len(prose),
            "n_sentences": len(sentences),
            "n_citations": len(cites),
            "citations_per_1000_chars": round(1000 * len(cites) / len(prose), 3) if prose else 0.0,
        })
    lengths = [r["prose_chars"] for r in rows]
    mean_len = statistics.mean(lengths) if lengths else 0.0
    return {
        "n_sections": len(rows),
        "length_cv": round(statistics.stdev(lengths) / mean_len, 3)
        if len(lengths) > 1 and mean_len else 0.0,
        "n_citations": sum(r["n_citations"] for r in rows),
        "mean_citations_per_section": round(sum(r["n_citations"] for r in rows) / len(rows), 3)
        if rows else 0.0,
        "zero_citation_sections": [r["section"] for r in rows if r["n_citations"] == 0],
        "figure_refs_in_body": n_figure_refs,
        "table_refs_in_body": n_table_refs,
        "figure_bank_size": len(figure_bank),
        "table_bank_size": len(table_bank),
        "figure_usage": round(n_figure_refs / len(figure_bank), 3) if figure_bank else None,
        "table_usage": round(n_table_refs / len(table_bank), 3) if table_bank else None,
        "sections": rows,
    }


def _table_blocks(body: str) -> int:
    """Markdown tables = contiguous runs of pipe lines."""
    runs, in_run = 0, False
    for line in body.splitlines():
        if line.lstrip().startswith("|"):
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs


def _redundancy(survey_md: str, scorer) -> dict:
    """Cross-section text similarity: how much the sections repeat each other."""
    sections = [(t, p) for t, p in _body_sections(survey_md) if p]
    mode = scorer.mode if scorer is not None else None
    pairs, names = [], []
    for i in range(len(sections)):
        for j in range(i + 1, len(sections)):
            pairs.append((sections[i][1], sections[j][1]))
            names.append((sections[i][0], sections[j][0]))
    if not pairs:
        return {"mode": mode, "n_section_pairs": 0, "mean_similarity": None,
                "max_similarity": None, "top_pairs": []}
    scores = list(scorer.score_pairs(pairs))
    ranked = sorted(
        ({"a": a, "b": b, "similarity": round(s, 3)} for (a, b), s in zip(names, scores)),
        key=lambda r: r["similarity"], reverse=True)
    return {
        "mode": mode,
        "n_section_pairs": len(pairs),
        "mean_similarity": round(sum(scores) / len(scores), 3),
        "max_similarity": round(max(scores), 3),
        "top_pairs": ranked[:3],
    }


def deterministic_profile(survey_md: str, papers: list[dict], figure_bank: list,
                          table_bank: list, scorer, current_year: int | None = None) -> dict:
    """L0 — zero-LLM, zero-network portrait of the survey artifact.

    papers: paper cards (--papers) for the corpus year distribution;
    figure_bank / table_bank: mined asset lists the body embeds are compared to;
    scorer: EmbeddingScorer (or duck-typed stand-in) for the redundancy column.
    current_year pins "now" so the freshness window is reproducible in tests.
    """
    if current_year is None:
        current_year = datetime.now().year
    ref_years = _reference_years(survey_md)
    return {
        "current_year": current_year,
        "freshness": _freshness(survey_md, papers, ref_years, current_year),
        "structure": _structure(survey_md, ref_years, figure_bank, table_bank),
        "redundancy": _redundancy(survey_md, scorer),
    }


# ── L2' corpus-grounded coverage ───────────────────────────────────


def _category_fields(category) -> dict:
    """Taxonomy categories arrive as pydantic models (in memory) or as the JSON
    dicts persisted by P5 (final_taxonomy.json has no `topic`, so the artifact
    does not round-trip through the Taxonomy model)."""
    if isinstance(category, dict):
        return category
    return category.model_dump()


def corpus_coverage(survey_md: str, categories: list, papers: list[dict], scorer,
                    cov_threshold: float | None = None) -> dict:
    """L2' — how much of the P5 taxonomy (its `categories` list) the survey text
    actually covers and how deeply it cites each category's papers. Needs no
    gold survey, so it works in new domains (reference_coverage stays the
    optional gold control).

    Per category: covered = best (name+description vs section text) similarity
    >= cov_threshold; citation_depth = share of the category's papers cited.
    """
    if cov_threshold is None:
        cov_threshold = float(os.environ.get("EVISURVEY_EVAL_COV_MIN", "0.5"))
    cats = [_category_fields(c) for c in categories]
    sections = [(t, p) for t, p in _body_sections(survey_md) if p]
    corpus_ids = {_norm_pid(pid) for cat in cats for pid in cat["paper_ids"]}
    corpus_ids |= {_norm_pid(p["paper_id"]) for p in papers if p.get("paper_id")}
    cited = {_norm_pid(c) for c in _cited_ids(survey_md, corpus_ids)}
    cited_in_corpus = cited & corpus_ids
    # Depth 0 can mean "nothing cited" or "the two artifacts use different id
    # spaces" (output/survey.md cites paper:<doi>, cache/final_* use seed ids).
    # Report which one it is instead of letting the zeros look like a verdict.
    id_space_mismatch = bool(cited) and bool(corpus_ids) and not cited_in_corpus
    if id_space_mismatch:
        logger.warning(
            "[L2'] cited ids and corpus ids share no element "
            f"(cited sample {sorted(cited)[0]!r}, corpus sample {sorted(corpus_ids)[0]!r}) -> "
            "citation_depth / corpus_utilization are likely an artifact id-space mismatch, "
            "not zero citing")

    cat_texts = [f"{c['category_name']}. {c['description']}".strip() for c in cats]
    pairs = [(cat_text, section_text) for cat_text in cat_texts for _, section_text in sections]
    scores = list(scorer.score_pairs(pairs)) if pairs else []
    width = max(len(sections), 1)

    rows: list[dict] = []
    for index, cat in enumerate(cats):
        chunk = scores[index * width:(index + 1) * width]
        if chunk:
            best_index = max(range(len(chunk)), key=lambda k: chunk[k])
            best_score, best_section = chunk[best_index], sections[best_index][0]
        else:
            best_score, best_section = 0.0, None
        ids = {_norm_pid(pid) for pid in cat["paper_ids"]}
        depth = len(ids & cited) / len(ids) if ids else 0.0
        rows.append({
            "category_id": cat["category_id"],
            "category_name": cat["category_name"],
            "n_papers": len(cat["paper_ids"]),
            "best_similarity": round(best_score, 3),
            "best_section": best_section,
            "covered": bool(chunk) and best_score >= cov_threshold,
            "citation_depth": round(depth, 3),
        })

    total_papers = sum(r["n_papers"] for r in rows)
    return {
        "cov_threshold": cov_threshold,
        "n_categories": len(rows),
        "n_covered": sum(1 for r in rows if r["covered"]),
        "category_coverage_rate": round(sum(1 for r in rows if r["covered"]) / len(rows), 3)
        if rows else 0.0,
        "weighted_citation_depth": round(
            sum(r["citation_depth"] * r["n_papers"] for r in rows) / total_papers, 3)
        if total_papers else 0.0,
        "mean_citation_depth": round(sum(r["citation_depth"] for r in rows) / len(rows), 3)
        if rows else 0.0,
        "corpus_utilization": round(len(cited_in_corpus) / len(papers), 3) if papers else 0.0,
        "n_cited_ids": len(cited),
        "n_cited_ids_in_corpus": len(cited_in_corpus),
        "id_space_mismatch": id_space_mismatch,
        "missed_categories": [r["category_name"] for r in rows if not r["covered"]],
        "categories": rows,
    }


# ── L1.5 uncited-claim verification ────────────────────────────────


def _iter_uncited_claims(md: str, known_ids: set[str]):
    """Complement of _iter_sentence_claims: body sentences carrying no valid
    citation (meta sections excluded)."""
    for title, body in _split_sections(md):
        if not _is_body_section(title):
            continue
        for sentence in _sentences(body):
            cites = [c for c in _CITATION.findall(sentence) if _is_citation(c, known_ids)]
            text = _CITATION.sub("", sentence).strip().rstrip(".。").strip()
            if not cites and text:
                yield text


def uncited_claims(survey_md: str, evidence_store, nli, sciverse, cache_path=None,
                   max_claims: int = 60, top_k: int = 3) -> dict:
    """L1.5 — verify body sentences that carry no citation by retrieving
    external evidence per claim (ReportBench-style) and judging the hit
    chunks with the same sentence-window NLI as the cited layer.

    Expensive layer: bounded by max_claims (excess counted, not fetched) and
    resumable — verdicts are cached by claim text, cache hits issue no search.
    A claim with no retrievable evidence counts as unsupported (same rule as a
    cited paper with no evidence in citation_quality).
    """
    known_ids = {e.paper_id for e in evidence_store.evidence}
    cache: dict = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    rows: list[dict] = []
    n_evaluated = 0
    n_truncated = 0
    for text in _iter_uncited_claims(survey_md, known_ids):
        cached = cache.get(text)
        if cached is not None:
            rows.append({"claim_text": text, "status": cached["status"],
                         "confidence": cached["confidence"], "from_cache": True,
                         "n_hits": cached.get("n_hits", 0)})
            continue
        if n_evaluated >= max_claims:
            n_truncated += 1
            continue
        n_evaluated += 1
        hits = sciverse.agentic_search(text, top_k=top_k).get("hits", [])
        chunks = [h.get("chunk") or "" for h in hits]
        chunks = [c for c in chunks if c.strip()]
        if chunks:
            res = _best_judgment(nli, text, chunks)
            status, confidence = _LABEL_STATUS[res.label], round(res.confidence, 3)
        else:
            status, confidence = "unsupported", 0.0
        rows.append({"claim_text": text, "status": status, "confidence": confidence,
                     "from_cache": False, "n_hits": len(chunks)})
        cache[text] = {"status": status, "confidence": confidence, "n_hits": len(chunks)}

    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    supported = sum(1 for r in rows if r["status"] == "supported")
    weak = sum(1 for r in rows if r["status"] == "weak")
    unsupported = sum(1 for r in rows if r["status"] == "unsupported")
    return {
        "max_claims": max_claims,
        "n_uncited_total": len(rows) + n_truncated,
        "n_evaluated": len(rows),
        "n_searched": n_evaluated,
        "n_cache_hits": len(rows) - n_evaluated,
        "n_truncated": n_truncated,
        "supported": supported,
        "weak": weak,
        "unsupported": unsupported,
        "uncited_support_rate": round(supported / len(rows), 3) if rows else None,
        "claims": rows,
    }


def overall_unsupported_rate(citation_quality_result: dict,
                             uncited_result: dict | None) -> float | None:
    """Headline faithfulness number: unsupported share over ALL checked claims,
    cited pairs and externally verified uncited claims alike."""
    total = citation_quality_result["n_pairs"]
    unsupported = citation_quality_result["unsupported_pairs"]
    if uncited_result is not None:
        total += uncited_result["n_evaluated"]
        unsupported += uncited_result["unsupported"]
    return round(unsupported / total, 3) if total else None


# ── L3 academic value (DeepSurvey-Bench three-dimension judge) ─────

JUDGE_DIMENSIONS = ("informational_value", "scholarly_communication_value",
                    "research_guidance_value")

_DIM_RUBRIC = {
    "informational_value": (
        "informational value — how much of the topic's core research objectives "
        "and key content the survey conveys",
        "1 = core objectives or key content are missing or wrong; "
        "3 = main objectives are covered but key content is thin or uneven; "
        "5 = core objectives and key content are conveyed thoroughly and precisely",
    ),
    "scholarly_communication_value": (
        "scholarly communication value — organization, logical flow and academic "
        "expression (clarity and fluency of the writing)",
        "1 = disorganized and hard to follow; "
        "3 = readable but with structural or clarity gaps; "
        "5 = well organized, logically coherent and precisely expressed",
    ),
    "research_guidance_value": (
        "research guidance value — critical analysis plus open challenges and "
        "future research directions a new researcher could act on",
        "1 = descriptive only, no critical analysis; "
        "3 = some critical remarks but no actionable gaps; "
        "5 = sharp critical analysis with concrete open problems and future directions",
    ),
}


def academic_value(survey_md: str, llm) -> dict:
    """L3 — DeepSurvey-Bench academic-value judge: three dimensions, whole
    document, one json_chat call per dimension with 1/3/5 anchor descriptions.
    Replaces the old seven-dimension per-section rubric (its logic / clarity /
    coverage / relevance facets live in the anchors now)."""
    dims: dict[str, dict] = {}
    for dim in JUDGE_DIMENSIONS:
        definition, anchors = _DIM_RUBRIC[dim]
        prompt = (
            "You are judging the academic value of an academic survey.\n"
            f"Dimension key: {dim}\nDimension: {definition}.\n"
            f"Anchor descriptions:\n{anchors}\n\n"
            "Score the whole survey below on this single dimension, 1-5.\n"
            'Return JSON: {"score": <int 1-5>, "rationale": "<one line>"}\n\n'
            f"# Survey\n\n{survey_md}"
        )
        out = llm.json_chat([{"role": "user", "content": prompt}], temperature=0.0)
        score = out.get("score") if isinstance(out, dict) else None
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = None
        rationale = out.get("rationale") if isinstance(out, dict) else None
        if (not isinstance(score, (int, float)) or isinstance(score, bool)
                or not 1 <= score <= 5
                or not isinstance(rationale, str) or not rationale.strip()):
            raise ValueError(f"academic_value: unusable judgment for {dim}: {out!r}")
        dims[dim] = {"score": score, "rationale": rationale.strip()}
    return {
        "n_calls": len(dims),
        "dimensions": dims,
        "overall": round(sum(d["score"] for d in dims.values()) / len(dims), 3),
    }


# ── report rendering ───────────────────────────────────────────────


def _fmt(value) -> str:
    return "n/a" if value is None else str(value)


def render_md(report: dict) -> str:
    """Human-readable summary: Deterministic Profile → Faithfulness →
    Coverage → Academic Value → A/B table."""
    lines: list[str] = ["# Survey Evaluation Report", ""]

    dp = report.get("deterministic_profile")
    if dp:
        fr, st, rd = dp["freshness"], dp["structure"], dp["redundancy"]
        top_pairs = "; ".join(f"{r['a']} × {r['b']} = {r['similarity']}" for r in rd["top_pairs"])
        lines += [
            "## Block 1 — Deterministic Profile (L0, no LLM)", "",
            f"**Freshness** — citation year median {_fmt(fr['citation_year_median'])} vs corpus "
            f"{_fmt(fr['corpus_year_median'])} (offset {_fmt(fr['year_offset'])}); "
            f"recent-two-year share: citations {_fmt(fr['citation_recent_share'])} vs corpus "
            f"{_fmt(fr['corpus_recent_share'])}; unresolved citation years "
            f"{len(fr['years_unresolved'])} of {fr['n_cited_ids']} cited ids.",
            "",
            f"**Structure** — {st['n_sections']} sections, prose-length CV {st['length_cv']}, "
            f"{st['n_citations']} citations (mean {st['mean_citations_per_section']}/section); "
            f"zero-citation sections: "
            f"{', '.join(st['zero_citation_sections']) or 'none'}; "
            f"figure embeds {st['figure_refs_in_body']}/{st['figure_bank_size']} in bank, "
            f"table embeds {st['table_refs_in_body']}/{st['table_bank_size']} in bank.",
            "",
            f"**Redundancy** — section-pair similarity mean {_fmt(rd['mean_similarity'])} / max "
            f"{_fmt(rd['max_similarity'])} over {rd['n_section_pairs']} pairs (mode "
            f"{rd['mode']}); top pairs: {top_pairs or 'none'}.",
            "",
        ]

    cq = report.get("citation_quality")
    uc = report.get("uncited_claims")
    our = report.get("overall_unsupported_rate")
    if cq or uc or our is not None:
        lines += ["## Block 2 — Faithfulness (NLI)", ""]
        if our is not None:
            lines += [
                f"Headline: overall unsupported rate **{our}** = (unsupported citation pairs + "
                "unsupported uncited claims) / all checked claims.",
                "",
            ]
        if cq:
            lines += [
                "Citation quality (AutoSurvey-style, pair level):",
                "",
                "| metric | value |",
                "|---|---|",
                f"| Citation recall | {cq['citation_recall']} |",
                f"| Citation precision | {cq['citation_precision']} |",
                f"| Citation coverage | {cq['citation_coverage']} |",
                "",
                f"Sentences {cq['n_sentences']} / cited {cq['n_cited_sentences']} / "
                f"pairs {cq['n_pairs']} (supported {cq['supported_pairs']}, "
                f"weak {cq['weak_pairs']}, unsupported {cq['unsupported_pairs']}).",
                "",
            ]
            bad = [p for p in cq.get("pairs", []) if p["status"] == "unsupported"][:5]
            if bad:
                lines += ["Unsupported examples (max 5):", ""]
                lines += [f"- `{p['cited_paper_id']}` — {p['claim_text'][:80]}" for p in bad]
                lines.append("")
        if uc:
            lines += [
                f"Uncited claims: {uc['n_uncited_total']} found, {uc['n_evaluated']} verified "
                f"(searched {uc['n_searched']}, cache hits {uc['n_cache_hits']}, "
                f"truncated {uc['n_truncated']}); supported {uc['supported']}, "
                f"weak {uc['weak']}, unsupported {uc['unsupported']}; support rate "
                f"**{uc['uncited_support_rate']}**.",
                "",
            ]
            bad = [c for c in uc.get("claims", []) if c["status"] == "unsupported"][:5]
            if bad:
                lines += ["Unsupported uncited examples (max 5):", ""]
                lines += [f"- {c['claim_text'][:80]}" for c in bad]
                lines.append("")

    cc = report.get("corpus_coverage")
    rc = report.get("reference_coverage")
    if cc or rc:
        lines += ["## Block 3 — Coverage", ""]
        if cc:
            lines += [
                f"Corpus-grounded coverage **{cc['category_coverage_rate']}** "
                f"({cc['n_covered']}/{cc['n_categories']} categories at threshold "
                f"{cc['cov_threshold']}); paper-weighted citation depth "
                f"{cc['weighted_citation_depth']}; corpus utilization "
                f"{cc['corpus_utilization']}.",
                "",
                "| category | papers | best sim | covered | citation depth |",
                "|---|---|---|---|---|",
            ]
            lines += [
                f"| {r['category_name']} | {r['n_papers']} | {r['best_similarity']} | "
                f"{'yes' if r['covered'] else 'no'} | {r['citation_depth']} |"
                for r in cc["categories"]
            ]
            if cc["missed_categories"]:
                lines += ["", "Missed categories: " + ", ".join(cc["missed_categories"])]
            if cc.get("id_space_mismatch"):
                lines += [
                    "",
                    f"Note: {cc['n_cited_ids']} cited ids but none of them appears in the corpus "
                    f"({cc['n_cited_ids_in_corpus']} in common) — citation depth and corpus "
                    "utilization likely reflect an artifact id-space mismatch, not zero citing.",
                ]
            lines.append("")
        if rc:
            lines += [
                f"Gold survey control: **{rc['gold_source_title']}** ({rc['n_gold_refs']} refs). "
                f"Reference recall **{rc['reference_recall']}** "
                f"(exact {rc['exact_matches']} + substring {rc.get('substring_matches', 0)} "
                f"+ fuzzy {rc['fuzzy_matches']}); "
                f"system-in-gold rate {rc['system_in_gold_rate']} "
                f"over {rc['n_system_titles']} corpus papers.",
                "",
            ]
            if rc.get("matched_titles"):
                lines += ["Matched:", ""] + [f"- {t}" for t in rc["matched_titles"]] + [""]

    av = report.get("academic_value")
    if av:
        lines += [
            "## Block 4 — Academic Value (LLM judge, DeepSurvey rubric)", "",
            f"Overall **{av['overall']}** over {av['n_calls']} judged dimensions (whole document, "
            "one call per dimension).",
            "",
        ]
        lines += [f"- {dim}: **{v['score']}/5** — {v['rationale']}"
                  for dim, v in av["dimensions"].items()]
        lines.append("")

    ab = report.get("ab_comparison")
    if ab:
        lines += [
            "## A/B — repair joint before vs after", "",
            "| config | topic | claims | unsupported | weak | invalid | validity | recall_equiv |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['config']} | {r.get('topic') or ''} | {r['total_claims']} | "
            f"{r['unsupported']} | {r['weak']} | {r['invalid_citations']} | "
            f"{r['citation_validity_score']} | {r['citation_recall_equiv']} |"
            for r in ab
        ]
        lines += [
            "",
            "Recall-equivalent = (total − unsupported − weak) / total from the repair A/B "
            "claim_map (first-citation-per-sentence granularity, coarser than Layer 1 pairs). "
            "Retention story: delete-only repair reaches validity 1.0 by shrinking claims "
            "(61→20); action repair keeps claims (50) at the same validity.",
            "",
        ]

    meta = report.get("meta", {})
    if meta:
        lines += ["## Meta", ""] + [f"- {k}: {v}" for k, v in meta.items()] + [""]
    return "\n".join(lines)
