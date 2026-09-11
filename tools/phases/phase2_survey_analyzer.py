import logging
import re
from pydantic import BaseModel
from tools.models.common import paper_id_from_seed

logger = logging.getLogger(__name__)


class SurveyStructure(BaseModel):
    task_id: str
    analyzed_surveys: list[dict]
    preliminary_taxonomy: list[dict]
    refined_taxonomy: list[dict]
    expansion_candidates: list[dict]
    survey_update_log: list[dict] = []


PRELIM_PROMPT = """Generate a paper taxonomy for the topic: {topic}.
Sub-domains: {sub_domains}. Aspects: {aspects}.
Constraints: exactly 3 to 6 categories; each category name is at most 6 words; do not include synonymous or duplicate categories.
Return ONLY JSON: {{"categories":[{{"name":str,"description":str}}]}}"""

REFINE_PROMPT = """Refine this taxonomy using the survey structures below.
Preliminary: {prelim}
Survey taxonomies: {survey_skels}
Merge, dedupe, fix gaps. Constraints: exactly 3 to 6 categories; each category name is at most 6 words; do not include synonymous or duplicate categories.
Return ONLY JSON: {{"categories":[{{"name":str,"description":str,"incorporated_from":[str]}}]}}"""


def _hint_year(hint: str) -> int | None:
    parts = hint.split("_")
    if parts and len(parts[-1]) == 4 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def _int_year(year) -> int | None:
    if isinstance(year, int):
        return year
    if isinstance(year, float):
        return int(year)
    text = str(year or "").strip()
    return int(text) if text.isdigit() else None


def _hint_from_title(title: str, year) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    parsed_year = _int_year(year)
    return f"{slug}_{parsed_year}" if slug and parsed_year else slug


def _reference_entry(ref) -> dict | None:
    """Normalize one seed-survey reference into a searchable bib entry.

    References arrive either as id hints ("dreamerv3_2023") or as structured
    entries ({paper_id/paper_id_hint, title, authors, year}); the hint is the
    grouping/search key, title/authors/year are the citation metadata.
    """
    if isinstance(ref, dict):
        title = " ".join(str(ref.get("title") or "").split())
        hint = str(ref.get("paper_id_hint") or ref.get("paper_id") or "").strip()
        if not title and not hint:
            return None
        return {
            "paper_id_hint": hint or _hint_from_title(title, ref.get("year")),
            "title": title,
            "authors": [str(a).strip() for a in (ref.get("authors") or []) if str(a).strip()],
            "year": _int_year(ref.get("year")),
        }
    hint = str(ref).strip()
    if not hint:
        return None
    return {"paper_id_hint": hint, "title": "", "authors": [], "year": _hint_year(hint)}


def analyze_surveys(surveys: list[dict], mineru=None, cleaner=None) -> list[dict]:
    out = []
    seen = set()
    for s in surveys:
        paper_id = s.get("paper_id") or paper_id_from_seed(s["title"], s.get("year", 0))
        if paper_id in seen:
            continue
        seen.add(paper_id)
        meta = s.get("meta_data", {})
        skel = meta.get("taxonomy_skeleton", [])
        entries: dict[str, dict] = {}
        for raw_ref in meta.get("top_referenced_papers", []):
            entry = _reference_entry(raw_ref)
            if entry:
                entries.setdefault(entry["paper_id_hint"], entry)
        out.append({
            "paper_id": paper_id,
            "taxonomy_skeleton": skel,
            "key_sections": meta.get("key_sections", []),
            "referenced_paper_ids": list(entries),
            "reference_entries": list(entries.values()),
            "key_claims": [],
        })
    return out


def build_expansion_candidates(analyzed_surveys: list[dict]) -> list[dict]:
    """Turn the seed surveys' own reference lists into bib-sourced search targets.

    These references are the field's pseudo-gold: a paper cited by more seed
    surveys is more canonical, so survey_ref_count orders the list.
    """
    candidates = {}
    for survey in analyzed_surveys:
        survey_id = survey["paper_id"]
        entries = survey.get("reference_entries")
        if entries is None:
            # Legacy callers pass bare hints; one citation per survey counts once.
            per_survey: dict[str, dict] = {}
            for hint in survey["referenced_paper_ids"]:
                entry = _reference_entry(hint)
                if entry:
                    per_survey.setdefault(entry["paper_id_hint"], entry)
            entries = list(per_survey.values())
        for entry in entries:
            hint = entry["paper_id_hint"]
            item = candidates.setdefault(hint, {
                "paper_id_hint": hint,
                "title": entry["title"],
                "authors": list(entry["authors"]),
                "year": entry["year"],
                "source": "bib",
                "source_survey": survey_id,
                "source_surveys": [],
                "survey_ref_count": 0,
                "priority": "high",
            })
            item["source_surveys"].append(survey_id)
            item["survey_ref_count"] += 1
    return sorted(
        candidates.values(),
        key=lambda item: (-item["survey_ref_count"], item["paper_id_hint"]),
    )


def run(
    task_id: str,
    topic: str,
    sub_domains: list[str],
    aspects: list[dict],
    surveys: list[dict],
    llm,
    mineru=None,
    cleaner=None,
    sciverse=None,
) -> SurveyStructure:
    logger.info(f"[P2] survey_analyzer: {len(surveys)} surveys, {len(aspects)} aspects")
    analyzed = analyze_surveys(surveys, mineru, cleaner)
    # Step 1: independent preliminary taxonomy
    aspect_names = [a["aspect_name"] for a in aspects]
    if getattr(llm, "_evisurvey_taxonomy_fallback", False):
        prelim_raw = {"categories": _fallback_categories(sub_domains, aspect_names)}
    else:
        try:
            prelim_raw = llm.json_chat([{"role": "user", "content": PRELIM_PROMPT.format(
                topic=topic, sub_domains=sub_domains, aspects=aspect_names)}], max_tokens=2000)
        except Exception:
            setattr(llm, "_evisurvey_taxonomy_fallback", True)
            setattr(llm, "_evisurvey_card_fallback", True)
            prelim_raw = {"categories": _fallback_categories(sub_domains, aspect_names)}
    prelim = prelim_raw.get("categories", [])
    logger.info(f"[P2] prelim taxonomy -> {len(prelim)} categories")
    # Step 2: refine with survey skeletons
    survey_skels = [a["taxonomy_skeleton"] for a in analyzed]
    if getattr(llm, "_evisurvey_taxonomy_fallback", False):
        refined_raw = {"categories": prelim or _fallback_categories(sub_domains, aspect_names)}
    else:
        try:
            refined_raw = llm.json_chat([{"role": "user", "content": REFINE_PROMPT.format(
                prelim=prelim, survey_skels=survey_skels)}], max_tokens=2000)
        except Exception:
            setattr(llm, "_evisurvey_taxonomy_fallback", True)
            setattr(llm, "_evisurvey_card_fallback", True)
            refined_raw = {"categories": prelim or _fallback_categories(sub_domains, aspect_names)}
    refined = _gate_taxonomy(refined_raw.get("categories", []), survey_skels)
    expansion = build_expansion_candidates(analyzed)
    logger.info(f"[P2] refined taxonomy -> {len(refined)} categories, {len(expansion)} expansion_candidates")
    return SurveyStructure(
        task_id=task_id,
        analyzed_surveys=analyzed,
        preliminary_taxonomy=prelim,
        refined_taxonomy=refined,
        expansion_candidates=expansion,
    )


TAXONOMY_MIN_CATEGORIES = 3
TAXONOMY_MAX_CATEGORIES = 6


def _normalize_category_name(name) -> str:
    return " ".join(str(name or "").split())


def _dedupe_categories(categories: list) -> list[dict]:
    """Normalize category names (collapse whitespace) and merge case-insensitive
    exact-name duplicates, keeping the first occurrence in order of appearance.
    Bare-string entries are coerced to category dicts."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for cat in categories:
        raw_name = cat.get("name") if isinstance(cat, dict) else cat
        name = _normalize_category_name(raw_name)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        entry = dict(cat) if isinstance(cat, dict) else {}
        entry["name"] = name
        deduped.append(entry)
    return deduped


def _skeleton_categories(survey_skels: list) -> list[dict]:
    """Flatten seed-survey taxonomy skeletons into category dicts; skeletons are
    plain strings in practice but may carry {"name": ...} entries."""
    cats: list[dict] = []
    for skel in survey_skels:
        for entry in skel or []:
            name = _normalize_category_name(entry.get("name") if isinstance(entry, dict) else entry)
            if name:
                # A name-derived description keeps the placeholder text out of
                # downstream aspect queries and section goals; skeletons carry
                # names only, never authored descriptions.
                cats.append({"name": name, "description": f"Reported work on {name}."})
    return _dedupe_categories(cats)


def _gate_taxonomy(categories: list, survey_skels: list) -> list[dict]:
    """Deterministic taxonomy count gate: normalize/dedupe names, then >6 -> keep the
    first 6 in order of appearance; <3 -> backfill from seed skeletons; still <3 ->
    keep the result and warn (never block)."""
    before = len(categories)
    gated = _dedupe_categories(categories)
    actions: list[str] = []
    if len(gated) < before:
        actions.append("merged")
    if len(gated) > TAXONOMY_MAX_CATEGORIES:
        gated = gated[:TAXONOMY_MAX_CATEGORIES]
        actions.append("capped")
    if len(gated) < TAXONOMY_MIN_CATEGORIES:
        existing = {c["name"].lower() for c in gated}
        for skel_cat in _skeleton_categories(survey_skels):
            if len(gated) >= TAXONOMY_MIN_CATEGORIES:
                break
            key = skel_cat["name"].lower()
            if key not in existing:
                gated.append(skel_cat)
                existing.add(key)
        actions.append("skeleton-filled")
    suffix = f" ({', '.join(actions)})" if actions else ""
    if len(gated) < TAXONOMY_MIN_CATEGORIES:
        logger.warning(f"[P2] taxonomy gate: {before} -> {len(gated)}{suffix}; still below "
                       f"{TAXONOMY_MIN_CATEGORIES}, no usable seed taxonomy skeleton")
    else:
        logger.info(f"[P2] taxonomy gate: {before} -> {len(gated)}{suffix}")
    logger.info(f"[P2] taxonomy categories: {[c['name'] for c in gated]}")
    return gated


def _fallback_categories(sub_domains: list[str], aspect_names: list[str]) -> list[dict]:
    names = [name for name in [*sub_domains, *aspect_names] if name]
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        deduped = ["Foundational Methods", "Modeling and Simulation", "Evaluation and Benchmarks"]
    return [{"name": name, "description": f"Rule-based taxonomy fallback for {name}."} for name in deduped[:6]]
