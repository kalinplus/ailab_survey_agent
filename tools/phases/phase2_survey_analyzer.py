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
    for s in surveys:
        meta = s.get("meta_data", {})
        skel = meta.get("taxonomy_skeleton", [])
        refs = meta.get("top_referenced_papers", [])
        out.append({
            "paper_id": s.get("paper_id") or paper_id_from_seed(s["title"], s.get("year", 0)),
            "taxonomy_skeleton": skel,
            "key_sections": meta.get("key_sections", []),
            "referenced_paper_ids": refs,
            "key_claims": [],
        })
    return out


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
    prelim_raw = llm.json_chat([{"role": "user", "content": PRELIM_PROMPT.format(
        topic=topic, sub_domains=sub_domains, aspects=[a["aspect_name"] for a in aspects])}])
    prelim = prelim_raw.get("categories", [])
    logger.info(f"[P2] prelim taxonomy -> {len(prelim)} categories")
    # Step 2: refine with survey skeletons
    survey_skels = [a["taxonomy_skeleton"] for a in analyzed]
    refined_raw = llm.json_chat([{"role": "user", "content": REFINE_PROMPT.format(
        prelim=prelim, survey_skels=survey_skels)}])
    refined = refined_raw.get("categories", [])
    # expansion candidates from referenced papers
    expansion = [{"paper_id_hint": pid, "source_survey": a["paper_id"], "priority": "high"}
                 for a in analyzed for pid in a["referenced_paper_ids"]]
    logger.info(f"[P2] refined taxonomy -> {len(refined)} categories, {len(expansion)} expansion_candidates")
    return SurveyStructure(
        task_id=task_id,
        analyzed_surveys=analyzed,
        preliminary_taxonomy=prelim,
        refined_taxonomy=refined,
        expansion_candidates=expansion,
    )
