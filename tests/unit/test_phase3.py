from tools.phases.phase3_paper_retriever import (
    run,
    dedup,
    _influence_filter_sets,
    _filter_relevant_candidates,
    _rank_by_influence,
    _to_retrieved,
    pipeline_config_extra,
)
from tools.models.artifacts import RetrievedPaper


class FakeSV:
    def meta_search(self, query, **kw):
        return {
            "results": [
                {"unique_id": "paper:1", "title": "A", "year": 2023, "url": "http://a/paper.pdf"},
                {"unique_id": "paper:1", "title": "A", "year": 2023},
            ]  # dup
        }


class FakeMU:
    def parse_url(self, url, light=True):
        return {
            "title": "A",
            "abstract": "abs",
            "sections": [],
            "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
            "figures": [],
            "tables": [],
        }


class FakeCleaner:
    def clean_parsed(self, d):
        return d


# --- dedup ---


def test_dedup_by_paper_id():
    ps = [
        _to_retrieved({"unique_id": "paper:1", "title": "A"}),
        _to_retrieved({"unique_id": "paper:1", "title": "A"}),
    ]
    assert len(dedup(ps)) == 1


def test_dedup_no_dups():
    ps = [
        _to_retrieved({"unique_id": "p1", "title": "A"}),
        _to_retrieved({"unique_id": "p2", "title": "B"}),
    ]
    assert len(dedup(ps)) == 2


def test_dedup_merges_survey_ref_signal():
    ps = [
        RetrievedPaper(paper_id="p1", title="A"),
        RetrievedPaper(paper_id="p1", title="A", survey_ref_count=2, survey_ref_hints=["genie_2024"]),
    ]
    result = dedup(ps)
    assert len(result) == 1
    assert result[0].survey_ref_count == 2
    assert result[0].survey_ref_hints == ["genie_2024"]


def test_dedup_seed_id_stable():
    """Seed fallback uses title+year hash; same title+year should produce same paper_id."""
    ps = [
        _to_retrieved({"title": "DreamerV3", "year": 2023}, source="seed"),
        _to_retrieved({"title": "DreamerV3", "year": 2023}, source="seed"),
    ]
    result = dedup(ps)
    assert len(result) == 1
    assert result[0].paper_id.startswith("seed:")


# --- seed fallback ---


def test_seed_fallback_when_no_hits():
    from tools.models.requests import PipelineConfig

    class EmptySV:
        def meta_search(self, query, **kw):
            return {"results": []}

    seeds = [{"title": "DreamerV3", "year": 2023, "keywords": ["wm"]}]
    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], EmptySV(), FakeMU(), FakeCleaner(),
        seeds, PipelineConfig(use_seed_fallback=True),
    )
    assert len(rp.papers) == 1
    assert rp.papers[0].source == "seed"


def test_no_seed_fallback_when_disabled():
    from tools.models.requests import PipelineConfig

    class EmptySV:
        def meta_search(self, query, **kw):
            return {"results": []}

    seeds = [{"title": "DreamerV3", "year": 2023, "keywords": ["wm"]}]
    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], EmptySV(), FakeMU(), FakeCleaner(),
        seeds, PipelineConfig(use_seed_fallback=False),
    )
    assert len(rp.papers) == 0


# --- parse status ---


def test_parse_status_light_when_url_and_mineru():
    from tools.models.requests import PipelineConfig

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], FakeSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=True),
    )
    assert rp.papers[0].parse_status == "light"
    assert len(pp.papers) == 1  # only the one with a URL gets parsed


def test_parse_status_abstract_only_when_no_url():
    from tools.models.requests import PipelineConfig

    class NoUrlSV:
        def meta_search(self, query, **kw):
            return {"results": [{"unique_id": "p1", "title": "NoUrl", "year": 2024}]}

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], NoUrlSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=True),
    )
    assert rp.papers[0].parse_status == "abstract_only"
    assert len(pp.papers) == 0


def test_parse_status_abstract_only_when_mineru_disabled():
    from tools.models.requests import PipelineConfig

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], FakeSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=False),
    )
    assert rp.papers[0].parse_status == "abstract_only"
    assert len(pp.papers) == 0


# --- max_papers truncation ---


def test_max_papers_default_40():
    from tools.models.requests import PipelineConfig

    cfg = PipelineConfig()
    assert pipeline_config_extra(cfg, "max_papers", 40) == 40


def test_pipeline_config_keeps_request_limits():
    from tools.models.requests import PipelineConfig

    cfg = PipelineConfig(max_papers=5, max_core_papers=3)
    assert cfg.max_papers == 5
    assert cfg.max_core_papers == 3
    assert cfg.use_influence_score is True


def test_max_papers_truncation():
    from tools.models.requests import PipelineConfig

    class ManySV:
        def meta_search(self, query, **kw):
            return {
                "results": [
                    {"unique_id": f"paper:{i}", "title": f"P{i}", "year": 2023, "url": f"http://p{i}"}
                    for i in range(100)
                ]
            }

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], ManySV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=False, max_papers=1),
    )
    assert len(rp.papers) == 1


# --- influence score ---


def test_influence_filter_sets_use_year_bands_with_citation_thresholds():
    filters = _influence_filter_sets()
    assert filters[0] == [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2024},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2026},
    ]
    assert filters[1][-1] == {"field": "citation_count", "operator": "FILTER_OP_GTE", "value": 5}
    assert filters[2][-1] == {"field": "citation_count", "operator": "FILTER_OP_GTE", "value": 20}


def test_influence_search_uses_banded_filters_and_freshness_boost():
    from tools.models.requests import PipelineConfig

    calls = []

    class RecordingSV:
        def meta_search(self, query, **kw):
            calls.append(kw)
            return {"results": []}

    run(
        "t", [{"keywords": ["wm"]}], [], RecordingSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_influence_score=True),
    )
    assert len(calls) == 3
    assert calls[0]["page_size"] == 15
    assert calls[0]["impact_boost"] == "MILD"
    assert calls[0]["freshness_boost"] == "MILD"
    assert calls[1]["filters"][-1] == {"field": "citation_count", "operator": "FILTER_OP_GTE", "value": 5}
    assert calls[2]["filters"][-1] == {"field": "citation_count", "operator": "FILTER_OP_GTE", "value": 20}


def test_influence_disabled_keeps_single_broad_search_without_freshness_boost():
    from tools.models.requests import PipelineConfig

    calls = []

    class RecordingSV:
        def meta_search(self, query, **kw):
            calls.append(kw)
            return {"results": []}

    run(
        "t", [{"keywords": ["wm"]}], [], RecordingSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_influence_score=False),
    )
    assert len(calls) == 1
    assert calls[0]["page_size"] == 25
    assert "freshness_boost" not in calls[0]
    assert calls[0]["filters"] == [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2018},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2026},
    ]


def test_rank_by_influence_balances_source_rank_citations_and_recency():
    low_quality_first = RetrievedPaper(
        paper_id="paper:old-low", title="Old Low", year=2018, citation_count=0)
    stronger_second = RetrievedPaper(
        paper_id="paper:new-cited", title="New Cited", year=2025, citation_count=80,
        abstract="abs", venue="ICML", url="https://x/p.pdf")

    ranked = _rank_by_influence(
        [low_quality_first, stronger_second],
        {"paper:old-low": 0, "paper:new-cited": 1},
    )
    assert [p.paper_id for p in ranked] == ["paper:new-cited", "paper:old-low"]


def test_rank_by_influence_uses_survey_ref_as_light_boost():
    first = RetrievedPaper(
        paper_id="paper:first", title="First", year=2024, citation_count=10,
        abstract="abs", venue="arXiv", url="https://x/first.pdf")
    survey_backed = RetrievedPaper(
        paper_id="paper:survey", title="Survey Backed", year=2024, citation_count=10,
        abstract="abs", venue="arXiv", url="https://x/survey.pdf", survey_ref_count=1)

    ranked = _rank_by_influence(
        [first, survey_backed],
        {"paper:first": 0, "paper:survey": 1, "paper:tail": 10},
    )
    assert [p.paper_id for p in ranked] == ["paper:survey", "paper:first"]


def test_relevance_prefilter_removes_highly_cited_off_topic_paper():
    cancer = RetrievedPaper(
        paper_id="paper:cancer", title="Cancer statistics 2024", year=2024,
        citation_count=10_000, abstract="Annual cancer incidence and mortality statistics.",
        venue="CA Cancer J Clin", url="https://x/cancer", source="survey_expansion")
    world_model = RetrievedPaper(
        paper_id="paper:wm", title="Learning World Models for Game Agents", year=2024,
        citation_count=2, abstract="Latent dynamics and world models for interactive environments.",
        venue="arXiv", url="https://x/wm.pdf")

    filtered = _filter_relevant_candidates(
        [cancer, world_model],
        [{"keywords": ["world model", "dreamer", "latent dynamics"]}],
    )
    ranked = _rank_by_influence(
        filtered,
        {"paper:cancer": 0, "paper:wm": 1},
        [{"keywords": ["world model", "dreamer", "latent dynamics"]}],
    )
    assert [p.paper_id for p in ranked] == ["paper:wm"]


def test_relevance_prefilter_fails_closed_when_nothing_matches():
    off_topic = RetrievedPaper(
        paper_id="paper:only", title="Cancer statistics 2024", year=2024,
        citation_count=10_000, abstract="Annual cancer incidence and mortality statistics.",
        venue="CA Cancer J Clin", url="https://x/cancer")

    filtered = _filter_relevant_candidates(
        [off_topic],
        [{"keywords": ["world model", "dreamer"]}],
    )
    assert filtered == []


def test_relevance_prefilter_uses_multiword_token_overlap():
    off_topic = RetrievedPaper(
        paper_id="paper:cancer", title="Cancer statistics 2024", year=2024,
        citation_count=10_000, abstract="Annual cancer incidence and mortality statistics.",
        venue="CA Cancer J Clin", url="https://x/cancer")
    relevant = RetrievedPaper(
        paper_id="paper:game", title="Controllable Game World Simulation", year=2024,
        citation_count=1, abstract="A user-in-the-loop framework for interactive environments.",
        venue="arXiv", url="https://x/game.pdf")

    filtered = _filter_relevant_candidates(
        [off_topic, relevant],
        [{"keywords": ["game world controllable simulation user-in-the-loop"]}],
    )
    assert [p.paper_id for p in filtered] == ["paper:game"]


def test_generated_queries_are_used_for_relevance_prefilter():
    from tools.models.requests import PipelineConfig

    class QueryLLM:
        def chat(self, messages, temperature=0.1):
            return "game world controllable simulation user-in-the-loop"

    class MixedSV:
        def meta_search(self, query, **kw):
            return {"results": [
                {
                    "unique_id": "paper:cancer",
                    "title": "Cancer statistics 2024",
                    "publication_published_year": 2024,
                    "abstract": "Annual cancer incidence and mortality statistics.",
                    "citation_count": 10_000,
                },
                {
                    "unique_id": "paper:game",
                    "title": "Controllable Game World Simulation",
                    "publication_published_year": 2024,
                    "abstract": "A user-in-the-loop framework for interactive environments.",
                    "citation_count": 1,
                },
            ]}

    rp, pp = run(
        "t", [{"aspect_id": "a1", "keywords": ["游戏世界模型"]}], [], MixedSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=5), QueryLLM(),
    )
    assert [p.paper_id for p in rp.papers] == ["paper:game"]
    assert pp.papers == []


# --- expansion candidates ---


def test_expansion_candidates_are_searched_and_marked():
    from tools.models.requests import PipelineConfig

    calls = []

    class ExpansionSV:
        def meta_search(self, query, **kw):
            calls.append({"query": query, **kw})
            return {"results": [{
                "unique_id": "paper:genie",
                "title": "Genie: Generative Interactive Environments",
                "publication_published_year": 2024,
                "url": "https://arxiv.org/pdf/2402.15391",
            }]}

    rp, pp = run(
        "t", [], [{"paper_id_hint": "genie_2024", "survey_ref_count": 2}],
        ExpansionSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False),
    )
    assert calls[0]["query"] == "genie 2024"
    assert calls[0]["filters"] == [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2024},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2024},
    ]
    assert calls[0]["page_size"] == 3
    assert rp.papers[0].source == "survey_expansion"
    assert rp.papers[0].survey_ref_count == 2
    assert rp.papers[0].survey_ref_hints == ["genie_2024"]
    assert len(pp.papers) == 0


def test_expansion_candidates_skip_hits_that_do_not_match_hint():
    from tools.models.requests import PipelineConfig

    class OffTopicExpansionSV:
        def meta_search(self, query, **kw):
            return {"results": [{
                "unique_id": "paper:cancer",
                "title": "Cancer statistics, 2024",
                "publication_published_year": 2024,
                "abstract": "Annual cancer incidence and mortality statistics.",
                "citation_count": 10_000,
            }]}

    rp, pp = run(
        "t", [], [{"paper_id_hint": "genie_2024", "survey_ref_count": 2}],
        OffTopicExpansionSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False),
    )
    assert rp.papers == []
    assert pp.papers == []


# --- per-aspect resilience ---


def test_per_aspect_resilience():
    from tools.models.requests import PipelineConfig

    call_count = {"n": 0}

    class FlakySV:
        def meta_search(self, query, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("aspect 1 search failed")
            return {"results": [{"unique_id": "p2", "title": "Ok World Model", "year": 2024, "url": "http://s"}]}

    aspects = [{"keywords": ["fail"]}, {"keywords": ["world model"]}]
    rp, pp = run(
        "t", aspects, [], FlakySV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=False),
    )
    assert len(rp.papers) == 1
    assert rp.papers[0].title == "Ok World Model"


# --- return tuple structure ---


def test_run_returns_tuple_of_models():
    from tools.models.requests import PipelineConfig

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], FakeSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(),
    )
    assert rp.task_id == "t"
    assert pp.task_id == "t"
    assert hasattr(rp, "papers")
    assert hasattr(pp, "papers")


# --- native SciVerse field mapping ---


def test_to_retrieved_maps_native_sciverse_fields():
    hit = {
        "unique_id": "paper:42",
        "title": "DreamerV3",
        "author": [{"orcid": "0000-0001", "name": "Danijar Hafner"}, {"name": "Jurgis Pasukonis"}],
        "publication_published_year": 2023.0,
        "publication_venue_name_unified": "arXiv",
        "abstract": "Mastering diverse domains through world models.",
        "keywords": ["world model", "rl"],
        "citation_count": 100.0,
        "doi": "10.48550/arXiv.2301.04104",
        "access_oa_url": [],
        "locations": [{"type": "pdf", "is_oa": True, "url": "https://arxiv.org/pdf/2301.04104"}],
    }
    p = _to_retrieved(hit)
    assert p.paper_id == "paper:42"
    assert p.authors == ["Danijar Hafner", "Jurgis Pasukonis"]
    assert p.year == 2023 and isinstance(p.year, int)
    assert p.venue == "arXiv"
    assert p.url == "https://arxiv.org/pdf/2301.04104"  # from locations
    assert p.citation_count == 100 and isinstance(p.citation_count, int)
    assert p.source == "sciverse"


def test_to_retrieved_url_falls_back_to_doi():
    p = _to_retrieved({"unique_id": "p1", "title": "T", "doi": "10.1/x", "author": []})
    assert p.url == "https://doi.org/10.1/x"
    assert p.authors == []


# --- MinerU graceful degrade (no mock in production) ---


def test_parse_degrades_to_abstract_only_when_mineru_raises():
    from tools.models.requests import PipelineConfig

    class BrokenMU:
        def parse_url(self, url, light=True):
            raise RuntimeError("mineru down")

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], FakeSV(), BrokenMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=True),
    )
    assert rp.papers[0].parse_status == "abstract_only"
    assert len(pp.papers) == 0  # degraded; not crashed
