from dataclasses import dataclass
from tools.models.requests import SearchStrategy, PipelineConfig


@dataclass
class DecomposedDemand:
    aspects: list[dict]
    constraints: dict
    pipeline_config: PipelineConfig
    structure_errors: list[str]
    coverage_warnings: list[str]


def validate_structure(strategy: SearchStrategy) -> list[str]:
    errors = []
    ws = strategy.wide_search
    aspects = ws.get("search_aspects", [])
    if len(aspects) < 3:
        errors.append("aspects fewer than 3, coverage may be insufficient")
    for a in aspects:
        if not a.get("keywords"):
            errors.append(f"{a.get('aspect_id','?')} missing keywords")
    tr = ws.get("time_range")
    if tr and (tr.get("end_year", 0) - tr.get("start_year", 0) < 3):
        errors.append("time span < 3 years, may miss classic works")
    return errors


def validate_coverage(strategy: SearchStrategy, seed_papers: list[dict]) -> list[str]:
    warnings = []
    aspects = strategy.wide_search.get("search_aspects", [])
    for domain in strategy.sub_domains:
        matched = any(
            domain.lower() in a.get("aspect_name","").lower()
            or any(kw.lower() in domain.lower() for kw in a.get("keywords", []))
            for a in aspects)
        if not matched:
            warnings.append(f"A's sub_domain '{domain}' has no matching aspect")
    for paper in seed_papers:
        ptext = f"{paper.get('title','')} {' '.join(paper.get('keywords',[]))}".lower()
        matched = any(any(kw.lower() in ptext for kw in a.get("keywords", [])) for a in aspects)
        if not matched:
            warnings.append(f"seed paper matches no aspect: {paper.get('title')}")
    return warnings


def run(request, strategy: SearchStrategy, seed_papers: list[dict]) -> DecomposedDemand:
    return DecomposedDemand(
        aspects=strategy.wide_search.get("search_aspects", []),
        constraints=request.pipeline_config.model_dump(),
        pipeline_config=request.pipeline_config,
        structure_errors=validate_structure(strategy),
        coverage_warnings=validate_coverage(strategy, seed_papers),
    )
