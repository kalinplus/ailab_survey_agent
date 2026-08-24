"""Post-hoc survey evaluator — AutoSurvey/ALCE-style citation metrics,
gold-reference coverage, and LLM-as-judge scoring.

Standalone audit tool over existing run artifacts (output/survey.md,
cache/evidence_store.json); deliberately NOT part of the agent pipeline.
Entry point: scripts/run_survey_eval.py.
"""

import re

_SENT_SPLIT = re.compile(r"(?<=[.。])\s+")
_CITATION = re.compile(r"\[([^\]]+)\]")
_STRUCTURAL = ("#", "|", "!", ">")


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


JUDGE_DIMENSIONS = ("fluency", "logic", "redundancy", "clarity", "accuracy", "coverage", "relevance")

_DIM_HELP = {
    "fluency": "fluent and coherent language",
    "logic": "clear logical flow and structure",
    "redundancy": "freedom from redundant/repeated content (5 = concise)",
    "clarity": "clear, precise description",
    "accuracy": "freedom from factual errors",
    "coverage": "covers the important aspects of the topic",
    "relevance": "content stays on-topic",
}


def _split_sections(md: str) -> list[tuple[str, str]]:
    """(heading, body) pairs split on '## ' headings; leading body without a
    heading becomes one '(untitled)' section."""
    sections: list[tuple[str, str]] = []
    title, buf = None, []
    for line in md.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in buf):
                sections.append((title or "(untitled)", "\n".join(buf).strip()))
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((title or "(untitled)", "\n".join(buf).strip()))
    return sections


def llm_judge(survey_md: str, llm) -> dict:
    """LLM-as-judge rubric scoring — SurGE CQS five criteria plus
    coverage/relevance, 1-5 per dimension, one json_chat call per section."""
    per_section: list[dict] = []
    dim_scores: dict[str, list[float]] = {}
    for title, body in _split_sections(survey_md):
        rubric = "\n".join(f"- {d}: {h}" for d, h in _DIM_HELP.items())
        prompt = (
            "You are reviewing one section of an academic survey. Score each "
            f"dimension 1-5 (5 = best).\n{rubric}\n\n"
            'Return JSON: {"scores": {"fluency": <int>, ...}, "rationale": "<one line>"}\n\n'
            f"# Section: {title}\n\n{body}"
        )
        out = llm.json_chat([{"role": "user", "content": prompt}], temperature=0.0)
        scores = out.get("scores") if isinstance(out, dict) else None
        clean = {d: float(scores[d]) for d in JUDGE_DIMENSIONS
                 if isinstance(scores, dict) and isinstance(scores.get(d), (int, float))}
        if not clean:
            raise ValueError(f"llm_judge: no usable scores for section {title!r}: {out!r}")
        for dim, v in clean.items():
            dim_scores.setdefault(dim, []).append(v)
        per_section.append({"section": title, "scores": clean})

    dimensions = {d: round(sum(v) / len(v), 3) for d, v in dim_scores.items()}
    all_vals = [v for vs in dim_scores.values() for v in vs]
    return {
        "n_sections": len(per_section),
        "dimensions": dimensions,
        "overall": round(sum(all_vals) / len(all_vals), 3) if all_vals else 0.0,
        "per_section": per_section,
    }


def render_md(report: dict) -> str:
    """Human-readable summary of the full eval report (three layers + A/B)."""
    lines: list[str] = ["# Survey Evaluation Report", ""]

    cq = report.get("citation_quality")
    if cq:
        lines += [
            "## Layer 1 — Citation Quality (NLI, AutoSurvey-style)", "",
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

    rc = report.get("reference_coverage")
    if rc:
        lines += [
            "## Layer 2 — Reference Coverage (gold survey)", "",
            f"Gold: **{rc['gold_source_title']}** ({rc['n_gold_refs']} refs). "
            f"Reference recall **{rc['reference_recall']}** "
            f"(exact {rc['exact_matches']} + substring {rc.get('substring_matches', 0)} "
            f"+ fuzzy {rc['fuzzy_matches']}); "
            f"system-in-gold rate {rc['system_in_gold_rate']} "
            f"over {rc['n_system_titles']} corpus papers.",
            "",
        ]
        if rc.get("matched_titles"):
            lines += ["Matched:", ""] + [f"- {t}" for t in rc["matched_titles"]] + [""]

    lj = report.get("llm_judge")
    if lj:
        dims = ", ".join(f"{d} {v}" for d, v in lj.get("dimensions", {}).items())
        lines += [
            "## Layer 3 — LLM-as-Judge (rubric 1-5)", "",
            f"Overall **{lj['overall']}** over {lj['n_sections']} sections. {dims}",
            "",
        ]

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

    lines += [
        "## AHA adversarial judge (in design)", "",
        "Next step: replace the single judge with the AHA adversarial audit loop "
        "(initial judge → challenger → defender → auditor) to correct systematic judge bias.",
        "",
    ]

    meta = report.get("meta", {})
    if meta:
        lines += ["## Meta", ""] + [f"- {k}: {v}" for k, v in meta.items()] + [""]
    return "\n".join(lines)
