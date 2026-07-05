import logging
from dataclasses import dataclass
from tools.models.requests import SearchStrategy, PipelineConfig

logger = logging.getLogger(__name__)


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


SEED_COVERAGE_BATCH = 25


def _llm_seed_unmatched(aspects: list[dict], seed_papers: list[dict], llm) -> set[int] | None:
    """Ask LLM which aspect(s) each seed belongs to. Returns indices judged 'no aspect';
    None signals an LLM failure (caller records a single coverage-check warning)."""
    aspect_block = "\n".join(
        f"- {a.get('aspect_id', '?')}: {a.get('aspect_name', '')} — {a.get('description', '')}"
        for a in aspects)
    unmatched: set[int] = set()
    for start in range(0, len(seed_papers), SEED_COVERAGE_BATCH):
        batch = seed_papers[start:start + SEED_COVERAGE_BATCH]
        paper_block = "\n".join(
            f"[{i}] {p.get('title', '')} | keywords: {', '.join(p.get('keywords', []) or [])}"
            for i, p in enumerate(batch))
        prompt = (
            "Judge which search aspect(s) each paper belongs to by topical scope, not keyword matching. "
            "A paper matches an aspect if its topic falls within that aspect's scope. "
            'Return ONLY JSON: {"results": [{"index": 0, "aspects": ["aspect_id", ...]}]}. '
            "An empty aspects list means the paper matches no aspect.\n\n"
            f"Aspects:\n{aspect_block}\n\nPapers:\n{paper_block}"
        )
        try:
            raw = llm.json_chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1500)
        except Exception as e:
            logger.warning(f"[P1] coverage LLM failed (batch@{start}): {e}; skipping seed coverage check")
            return None
        judged: dict[int, list] = {}
        for r in (raw or {}).get("results", []) or []:
            if isinstance(r, dict):
                judged[int(r.get("index"))] = r.get("aspects") or []
        for i in range(len(batch)):
            if not judged.get(i):
                unmatched.add(start + i)
    return unmatched


def validate_coverage(strategy: SearchStrategy, seed_papers: list[dict], llm=None) -> list[str]:
    warnings = []
    aspects = strategy.wide_search.get("search_aspects", [])
    for domain in strategy.sub_domains:
        matched = any(
            domain.lower() in a.get("aspect_name","").lower()
            or any(kw.lower() in domain.lower() for kw in a.get("keywords", []))
            for a in aspects)
        if not matched:
            warnings.append(f"A's sub_domain '{domain}' has no matching aspect")
    if seed_papers and llm is not None:
        unmatched = _llm_seed_unmatched(aspects, seed_papers, llm)
        if unmatched is None:
            warnings.append("coverage check LLM failed; seed-aspect coverage not verified")
        else:
            for idx in sorted(unmatched):
                warnings.append(f"seed paper matches no aspect: {seed_papers[idx].get('title')}")
    return warnings


def run(request, strategy: SearchStrategy, seed_papers: list[dict], llm=None) -> DecomposedDemand:
    aspects = strategy.wide_search.get("search_aspects", [])
    logger.info(f"[P1] decompose: {len(aspects)} aspects, "
                f"{len(strategy.sub_domains)} sub_domains, {len(seed_papers)} seed_papers")
    demand = DecomposedDemand(
        aspects=aspects,
        constraints=_model_to_dict(request.pipeline_config),
        pipeline_config=request.pipeline_config,
        structure_errors=validate_structure(strategy),
        coverage_warnings=validate_coverage(strategy, seed_papers, llm),
    )
    if demand.structure_errors:
        logger.warning(f"[P1] {len(demand.structure_errors)} structure_errors: {demand.structure_errors}")
    if demand.coverage_warnings:
        logger.warning(f"[P1] {len(demand.coverage_warnings)} coverage_warnings (first: {demand.coverage_warnings[0]})")
    return demand


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
