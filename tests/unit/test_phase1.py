from tools.phases.phase1_decompose import run, validate_structure, validate_coverage
from tools.models.requests import SearchStrategy, KnowledgeBuildRequest
from tools.clients.llm_fake import FakeLLMClient


def _strategy():
    return SearchStrategy(topic="wm", sub_domains=["Game Benchmark"],
        wide_search={"search_aspects":[
            {"aspect_id":"a1","aspect_name":"Game Benchmark","keywords":["game","benchmark"]},
            {"aspect_id":"a2","aspect_name":"Internal WM","keywords":["world model","latent"]},
            {"aspect_id":"a3","aspect_name":"Game Engine","keywords":["neural","engine"]}],
            "time_range":{"start_year":2018,"end_year":2026}})


def test_structure_ok():
    assert validate_structure(_strategy()) == []


def test_structure_flags_few_aspects():
    s = _strategy()
    s.wide_search["search_aspects"] = s.wide_search["search_aspects"][:1]
    errors = validate_structure(s)
    assert any("fewer than 3" in e for e in errors)


def test_coverage_warns_unmatched_seed():
    s = _strategy()
    seeds = [{"title":"Quantum Computing Paper","keywords":["quantum"]}]
    fake = FakeLLMClient(responses=[("Quantum", {"results": [{"index": 0, "aspects": []}]})])
    w = validate_coverage(s, seeds, fake)
    assert any("quantum" in x.lower() for x in w) or any("no aspect" in x for x in w)


def test_coverage_llm_overrides_substring_match():
    """LLM judges a seed (that substring would match via 'game') as no-aspect -> warn."""
    s = _strategy()
    seeds = [{"title": "A Game Design Paper", "keywords": ["game"]}]
    fake = FakeLLMClient(responses=[("Game Design Paper", {"results": [{"index": 0, "aspects": []}]})])
    warnings = validate_coverage(s, seeds, fake)
    assert any("Game Design Paper" in w and "no aspect" in w for w in warnings)
    assert fake.calls >= 1


def test_coverage_llm_matched_seed_no_warning():
    """LLM judges seed as matching an aspect -> no seed 'no aspect' warning."""
    s = _strategy()
    seeds = [{"title": "Obscure Paper", "keywords": ["nothingmatching"]}]
    fake = FakeLLMClient(responses=[("Obscure", {"results": [{"index": 0, "aspects": ["a1"]}]})])
    warnings = validate_coverage(s, seeds, fake)
    assert not any("no aspect" in w for w in warnings)


def test_coverage_llm_failure_records_warning_no_crash():
    """LLM raises -> a coverage-check failure warning is recorded, no crash, no per-seed warnings."""
    s = _strategy()
    seeds = [{"title": "Any Paper", "keywords": []}]

    def boom(*a, **k):
        raise RuntimeError("llm down")
    fake = FakeLLMClient()
    fake.json_chat = boom
    warnings = validate_coverage(s, seeds, fake)
    assert any("coverage" in w.lower() and "fail" in w.lower() for w in warnings)
    assert not any("no aspect" in w for w in warnings)


def test_run_returns_demand():
    req = KnowledgeBuildRequest(task_id="t", topic="wm", inputs={}, outputs={})
    d = run(req, _strategy(), [])
    assert d.aspects
    assert d.pipeline_config.use_seed_fallback is True


def test_structure_flags_short_time_span():
    """Test that validate_structure flags time span < 3 years."""
    s = _strategy()
    s.wide_search["time_range"] = {"start_year": 2024, "end_year": 2025}
    errors = validate_structure(s)
    assert any("time span < 3 years" in e for e in errors)


def test_structure_flags_aspect_missing_keywords():
    """Test that validate_structure flags an aspect missing keywords."""
    s = _strategy()
    s.wide_search["search_aspects"] = [
        {"aspect_id": "a1", "aspect_name": "Game Benchmark", "keywords": []},
        {"aspect_id": "a2", "aspect_name": "Internal WM", "keywords": ["world model"]},
        {"aspect_id": "a3", "aspect_name": "Game Engine", "keywords": ["neural"]},
    ]
    errors = validate_structure(s)
    assert any("a1 missing keywords" in e for e in errors)


def test_coverage_warns_unmatched_sub_domain():
    """Test that validate_coverage warns when a sub_domain has no matching aspect."""
    s = SearchStrategy(
        topic="wm",
        sub_domains=["Quantum"],
        wide_search={"search_aspects": [
            {"aspect_id": "a1", "aspect_name": "Game Benchmark", "keywords": ["game", "benchmark"]},
            {"aspect_id": "a2", "aspect_name": "Internal WM", "keywords": ["world model", "latent"]},
            {"aspect_id": "a3", "aspect_name": "Game Engine", "keywords": ["neural", "engine"]},
        ]}
    )
    warnings = validate_coverage(s, [])
    assert any("sub_domain 'Quantum' has no matching aspect" in w for w in warnings)


def test_run_propagates_errors_and_warnings():
    """Test that run propagates structure_errors and coverage_warnings, and aspects equals strategy's search_aspects."""
    s = _strategy()
    # Trigger structure error by reducing aspects
    s.wide_search["search_aspects"] = s.wide_search["search_aspects"][:1]
    # Trigger coverage warning by using unmatched seed paper
    seeds = [{"title": "Quantum Paper", "keywords": ["quantum"]}]

    req = KnowledgeBuildRequest(task_id="t", topic="wm", inputs={}, outputs={})
    fake = FakeLLMClient(responses=[("Quantum", {"results": [{"index": 0, "aspects": []}]})])
    d = run(req, s, seeds, fake)

    # Check aspects equals strategy's search_aspects
    assert d.aspects == s.wide_search.get("search_aspects", [])

    # Check structure_errors propagated
    assert len(d.structure_errors) > 0
    assert any("fewer than 3" in e for e in d.structure_errors)

    # Check coverage_warnings propagated
    assert len(d.coverage_warnings) > 0
    assert any("quantum" in w.lower() or "no aspect" in w for w in d.coverage_warnings)
