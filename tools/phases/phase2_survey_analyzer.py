import logging
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
Return ONLY JSON: {{"categories":[{{"name":str,"description":str}}]}}"""

REFINE_PROMPT = """Refine this taxonomy using the survey structures below.
Preliminary: {prelim}
Survey taxonomies: {survey_skels}
Merge, dedupe, fix gaps. Return ONLY JSON: {{"categories":[{{"name":str,"description":str,"incorporated_from":[str]}}]}}"""


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
        refs = meta.get("top_referenced_papers", [])
        out.append({
            "paper_id": paper_id,
            "taxonomy_skeleton": skel,
            "key_sections": meta.get("key_sections", []),
            "referenced_paper_ids": refs,
            "key_claims": [],
        })
    return out


def build_expansion_candidates(analyzed_surveys: list[dict]) -> list[dict]:
    candidates = {}
    for survey in analyzed_surveys:
        survey_id = survey["paper_id"]
        for paper_id_hint in dict.fromkeys(survey["referenced_paper_ids"]):
            item = candidates.setdefault(paper_id_hint, {
                "paper_id_hint": paper_id_hint,
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
    refined = refined_raw.get("categories", [])
    expansion = build_expansion_candidates(analyzed)
    logger.info(f"[P2] refined taxonomy -> {len(refined)} categories, {len(expansion)} expansion_candidates")
    return SurveyStructure(
        task_id=task_id,
        analyzed_surveys=analyzed,
        preliminary_taxonomy=prelim,
        refined_taxonomy=refined,
        expansion_candidates=expansion,
    )


def _fallback_categories(sub_domains: list[str], aspect_names: list[str]) -> list[dict]:
    names = [name for name in [*sub_domains, *aspect_names] if name]
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        deduped = ["Foundational Methods", "Modeling and Simulation", "Evaluation and Benchmarks"]
    return [{"name": name, "description": f"Rule-based taxonomy fallback for {name}."} for name in deduped[:6]]
