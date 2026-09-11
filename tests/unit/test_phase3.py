import json
import re

import pytest

from tools.phases.phase3_paper_retriever import (
    run,
    dedup,
    _candidate_matches_hit,
    _influence_filter_sets,
    _filter_relevant_candidates,
    _paper_relevance_score,
    _rank_by_influence,
    _to_retrieved,
    pipeline_config_extra,
    stage_trace_path,
)
from tools.models.artifacts import RetrievedPaper


@pytest.fixture(autouse=True)
def _redirect_stage_trace(tmp_path, monkeypatch):
    """run() writes cache/p3_stage_trace.json by default; keep unit-test runs
    out of the repo's cache/ (specs/canonical存活链诊断.md)."""
    monkeypatch.setenv("EVISURVEY_P3_STAGE_TRACE", str(tmp_path / "p3_stage_trace.json"))
    monkeypatch.setenv("EVISURVEY_P3_PARSE_TRACE", str(tmp_path / "parse_trace.json"))


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


def test_dedup_collapses_same_title_with_different_ids():
    """Spec T1 acceptance 2: preprint and published versions have different DOIs but
    the same title, so title identity must collapse them to one corpus entry."""
    ps = [
        _to_retrieved({
            "unique_id": "paper:10.48550/arxiv.2301.04104",
            "title": "DreamerV3: Mastering Diverse Domains through World Models",
            "publication_published_year": 2023,
        }),
        _to_retrieved({
            "unique_id": "paper:10.1038/s41586-024-08406-2",
            "title": "DreamerV3: Mastering Diverse Domains through World Models!",
            "publication_published_year": 2025,
            "citation_count": 30,
        }),
    ]
    result = dedup(ps)
    assert len(result) == 1
    assert result[0].year == 2023  # first seen wins


def test_dedup_merges_survey_ref_signal_across_title_duplicates():
    ps = [
        RetrievedPaper(paper_id="paper:a", title="World Models", citation_count=5),
        RetrievedPaper(paper_id="paper:b", title="world  models",
                       survey_ref_count=3, survey_ref_hints=["ha_2018"]),
    ]
    result = dedup(ps)
    assert len(result) == 1
    assert result[0].paper_id == "paper:a"
    assert result[0].survey_ref_count == 3
    assert result[0].survey_ref_hints == ["ha_2018"]


def test_title_key_strips_punctuation_case_and_version_tail():
    from tools.phases.phase3_paper_retriever import _title_key

    assert _title_key("DreamerV3: Mastering Diverse Domains") == _title_key("dreamerv3 mastering diverse domains")
    assert _title_key("World Models (v2)") == _title_key("world models")
    assert _title_key("") == ""


def test_dedup_keeps_similar_but_distinct_titles():
    """Exact normalized equality only — near-miss titles are different papers."""
    ps = [
        _to_retrieved({"unique_id": "p1", "title": "World Models"}),
        _to_retrieved({"unique_id": "p2", "title": "World Model"}),
    ]
    assert len(dedup(ps)) == 2


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
    assert len(calls) == 4  # 1 query x 3 influence bands + 1 landmark variant
    assert calls[0]["page_size"] == 15
    assert calls[0]["impact_boost"] == "MILD"
    assert calls[0]["freshness_boost"] == "MILD"
    assert calls[1]["filters"][-1] == {"field": "citation_count", "operator": "FILTER_OP_GTE", "value": 5}
    assert calls[2]["filters"][-1] == {"field": "citation_count", "operator": "FILTER_OP_GTE", "value": 20}


def test_landmark_query_runs_without_year_window_or_freshness_boost():
    from tools.models.requests import PipelineConfig

    calls = []

    class RecordingSV:
        def meta_search(self, query, **kw):
            calls.append({"query": query, **kw})
            return {"results": []}

    run(
        "t", [{"aspect_id": "aspect_001", "aspect_name": "World Models", "keywords": ["wm"]}],
        [], RecordingSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False),
    )
    landmark = calls[-1]
    assert landmark["query"] == "World Models survey"
    assert landmark["filters"] == []  # no year window: pre-2018 staples stay reachable
    assert landmark["impact_boost"] == "MILD"
    assert "freshness_boost" not in landmark


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
    assert len(calls) == 2  # 1 broad query + 1 landmark variant
    assert calls[0]["page_size"] == 25
    assert "freshness_boost" not in calls[0]
    assert calls[0]["filters"] == [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2018},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2026},
    ]
    assert calls[1]["filters"] == []


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


def test_generated_queries_are_used_for_relevance_prefilter(monkeypatch):
    from tools.models.requests import PipelineConfig

    # This test isolates the prefilter; the default aspect floor would rescue
    # the dropped off-topic paper back into a starving aspect.
    monkeypatch.setenv("EVISURVEY_ASPECT_MIN_PAPERS", "0")

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
    # hint-tail year widened to a ±1 window (arXiv vs journal year mismatch)
    assert calls[0]["filters"] == [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2023},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2025},
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


# --- T9: MinerU v4 fulltext main channel + access_oa_url PDF preference -----------


class ChannelMU:
    """Dual-channel fake: records the URL each channel receives; the v4 channel
    can be made to fail to exercise the agent-parse fallback."""

    def __init__(self, fulltext_error=None):
        self.fulltext_error = fulltext_error
        self.fulltext_calls = []
        self.parse_calls = []

    def extract_fulltext(self, url):
        self.fulltext_calls.append(url)
        if self.fulltext_error:
            raise RuntimeError(self.fulltext_error)
        return {
            "title": "A",
            "abstract": "abs",
            "sections": [{"name": "Method", "paragraphs": [{"page": 1, "index": 0, "text": "t"}]}],
            "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
            "figures": [{"num": 1, "page": 2, "caption": "Figure 1."}],
            "tables": [],
            "parse_status": "fulltext",
        }

    def parse_url(self, url, light=True):
        self.parse_calls.append(url)
        return {
            "title": "A",
            "abstract": "abs",
            "sections": [],
            "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
            "figures": [],
            "tables": [],
        }


class HitSV:
    """Single hit with configurable url / access_oa_url fields."""

    def __init__(self, hit):
        self.hit = hit

    def meta_search(self, query, **kw):
        return {"results": [self.hit]}


def test_to_retrieved_prefers_oa_direct_pdf():
    p = _to_retrieved({
        "unique_id": "p1", "title": "T",
        "url": "https://doi.org/10.1/x",  # landing page, not a PDF
        "access_oa_url": ["https://arxiv.org/pdf/2301.04104"],
    })
    assert p.url == "https://arxiv.org/pdf/2301.04104"


def test_mineru_feeds_access_oa_pdf_over_landing_url():
    from tools.models.requests import PipelineConfig

    sv = HitSV({
        "unique_id": "paper:1", "title": "A", "year": 2023,
        "url": "https://doi.org/10.1234/landing",
        "access_oa_url": ["https://arxiv.org/pdf/2301.04104"],
    })
    mu = ChannelMU()
    rp, pp = run("t", [{"keywords": ["wm"]}], [], sv, mu, FakeCleaner(), [],
                 PipelineConfig(use_mineru=True))

    # the landing page would have failed the PDF gate; the OA direct link parses
    assert mu.fulltext_calls == ["https://arxiv.org/pdf/2301.04104"]
    assert rp.papers[0].parse_status == "fulltext"
    assert pp.papers[0].parse_status == "fulltext"
    assert pp.papers[0].figures == [{"num": 1, "page": 2, "caption": "Figure 1."}]


def test_mineru_feeds_first_oa_direct_pdf_among_multiple_entries():
    from tools.models.requests import PipelineConfig

    sv = HitSV({
        "unique_id": "paper:1", "title": "A", "year": 2023,
        "url": "https://doi.org/10.1234/landing",
        "access_oa_url": [
            "https://publisher.example/toc/articleLanding",  # http but not a PDF
            "https://arxiv.org/pdf/1803.10122",
            "https://arxiv.org/pdf/2301.04104",
        ],
    })
    mu = ChannelMU()
    run("t", [{"keywords": ["wm"]}], [], sv, mu, FakeCleaner(), [], PipelineConfig(use_mineru=True))
    assert mu.fulltext_calls == ["https://arxiv.org/pdf/1803.10122"]


def test_mineru_feeds_paper_url_when_no_oa_direct_pdf():
    from tools.models.requests import PipelineConfig

    sv = HitSV({
        "unique_id": "paper:1", "title": "A", "year": 2023,
        "url": "https://arxiv.org/pdf/2301.04104",
        "access_oa_url": ["https://publisher.example/toc/articleLanding"],  # not a PDF
    })
    mu = ChannelMU()
    run("t", [{"keywords": ["wm"]}], [], sv, mu, FakeCleaner(), [], PipelineConfig(use_mineru=True))
    assert mu.fulltext_calls == ["https://arxiv.org/pdf/2301.04104"]


def test_parse_falls_back_to_agent_channel_when_v4_fails():
    from tools.models.requests import PipelineConfig

    mu = ChannelMU(fulltext_error="v4 down")
    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], FakeSV(), mu, FakeCleaner(),
        [], PipelineConfig(use_mineru=True),
    )
    assert mu.parse_calls == ["http://a/paper.pdf"]  # same URL on the fallback channel
    assert rp.papers[0].parse_status == "light"
    assert len(pp.papers) == 1


def test_parse_degrades_to_abstract_only_when_both_channels_fail():
    from tools.models.requests import PipelineConfig

    class BothBrokenMU:
        def extract_fulltext(self, url):
            raise RuntimeError("v4 down")

        def parse_url(self, url, light=True):
            raise RuntimeError("agent down")

    rp, pp = run(
        "t", [{"keywords": ["wm"]}], [], FakeSV(), BothBrokenMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=True),
    )
    assert rp.papers[0].parse_status == "abstract_only"
    assert pp.papers == []


# --- S1 breadth knobs (specs/检索广度与来源深度.md) -------------------------------


class BreadthSV:
    """Fake meta-search mirroring the real contract shape ({"results": [...]} + page_size).

    Hits are tagged with the aspect token (asp<k>) and the query slug found in the query,
    so every retrieved paper can be attributed to the aspect/query that produced it.
    """

    def __init__(self, pool=40):
        self.pool = pool
        self.calls = []
        self.hits_by_query = {}

    def meta_search(self, query, **kw):
        self.calls.append({"query": query, **kw})
        if query not in self.hits_by_query:
            aspect = re.search(r"asp(\d+)", query)
            prefix = f"asp{aspect.group(1)}" if aspect else "q"
            slug = re.sub(r"\W+", "", query)
            self.hits_by_query[query] = [
                {
                    "unique_id": f"paper:{prefix}:{slug}:{i}",
                    "title": f"{query} study {i}",
                    "publication_published_year": 2023,
                    "abstract": f"{query} abstract {i}",
                    "citation_count": i,
                    "url": f"https://arxiv.org/pdf/{prefix}{slug}{i}.pdf",
                }
                for i in range(self.pool)
            ]
        return {"results": self.hits_by_query[query][: kw.get("page_size", self.pool)]}


def _breadth_aspects(n=5):
    return [
        {
            "aspect_id": f"aspect_{k + 1:03d}",
            "keywords": [f"asp{k} world model", f"asp{k} latent dynamics"],
        }
        for k in range(n)
    ]


def _breadth_cfg():
    from tools.models.requests import PipelineConfig

    # corpus cap far above any pool, so breadth — not max_papers — sets the corpus size
    return PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=1000)


def test_breadth_knobs_widen_corpus_monotonically(monkeypatch):
    for name in ["EVISURVEY_MAX_QUERIES_PER_ASPECT", "EVISURVEY_META_PAGE_SIZE", "EVISURVEY_MAX_CORPUS"]:
        monkeypatch.delenv(name, raising=False)

    rp, _ = run("t", _breadth_aspects(), [], BreadthSV(), FakeMU(), FakeCleaner(), [], _breadth_cfg())
    baseline = len(rp.papers)

    monkeypatch.setenv("EVISURVEY_META_PAGE_SIZE", "40")
    rp, _ = run("t", _breadth_aspects(), [], BreadthSV(), FakeMU(), FakeCleaner(), [], _breadth_cfg())
    wider_page = len(rp.papers)

    monkeypatch.setenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", "3")
    rp, _ = run("t", _breadth_aspects(), [], BreadthSV(), FakeMU(), FakeCleaner(), [], _breadth_cfg())
    widest = len(rp.papers)

    assert 0 < baseline < wider_page < widest
    assert widest == 5 * (3 + 1) * 40  # 5 aspects x (3 queries + 1 landmark) x page_size 40


def test_breadth_knobs_reach_meta_search(monkeypatch):
    monkeypatch.delenv("EVISURVEY_META_PAGE_SIZE", raising=False)
    monkeypatch.delenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", raising=False)

    sv = BreadthSV()
    run("t", _breadth_aspects(1), [], sv, FakeMU(), FakeCleaner(), [], _breadth_cfg())
    assert len(sv.calls) == 4  # 1 query x 3 influence bands + 1 landmark variant
    assert sv.calls[0]["page_size"] == 15

    monkeypatch.setenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", "3")
    monkeypatch.setenv("EVISURVEY_META_PAGE_SIZE", "40")
    wide_sv = BreadthSV()
    run("t", _breadth_aspects(1), [], wide_sv, FakeMU(), FakeCleaner(), [], _breadth_cfg())
    assert len(wide_sv.calls) == 10  # 3 queries x 3 influence bands + 1 landmark variant
    assert {call["page_size"] for call in wide_sv.calls} == {40}


def test_max_corpus_caps_with_aspect_balance(monkeypatch):
    monkeypatch.setenv("EVISURVEY_META_PAGE_SIZE", "40")
    monkeypatch.setenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", "3")
    monkeypatch.setenv("EVISURVEY_MAX_CORPUS", "12")

    rp, _ = run("t", _breadth_aspects(), [], BreadthSV(), FakeMU(), FakeCleaner(), [], _breadth_cfg())

    assert len(rp.papers) == 12
    aspects_seen = {re.match(r"paper:(asp\d+):", p.paper_id).group(1) for p in rp.papers}
    assert aspects_seen == {f"asp{k}" for k in range(5)}  # the cap erased no aspect


def test_max_corpus_keeps_corpus_when_cap_not_binding(monkeypatch):
    monkeypatch.setenv("EVISURVEY_META_PAGE_SIZE", "40")
    monkeypatch.setenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", "3")
    monkeypatch.setenv("EVISURVEY_MAX_CORPUS", "1000")

    rp, _ = run("t", _breadth_aspects(), [], BreadthSV(pool=4), FakeMU(), FakeCleaner(), [], _breadth_cfg())

    assert len(rp.papers) == 5 * (3 + 1) * 4  # nothing trimmed, no reordering side effect


def test_relevance_threshold_knob_is_monotone(monkeypatch):
    aspects = [{"keywords": ["world model", "latent dynamics", "gamecraft benchmark"]}]
    papers = [
        RetrievedPaper(paper_id="p:strong", title="World model latent dynamics for gamecraft benchmark",
                       year=2024, abstract="world model latent dynamics for gamecraft benchmark"),
        RetrievedPaper(paper_id="p:partial", title="Latent dynamics for control",
                       year=2024, abstract="latent dynamics planning"),
        RetrievedPaper(paper_id="p:weak", title="Latent-space regularization study",
                       year=2024, abstract="regularizers over latent space"),
        RetrievedPaper(paper_id="p:offtopic", title="Cancer statistics 2024",
                       year=2024, abstract="Annual cancer incidence and mortality statistics."),
    ]
    scores = {p.paper_id: _paper_relevance_score(p, aspects) for p in papers}

    monkeypatch.setenv("EVISURVEY_RELEVANCE_MIN_SCORE", "0.6")
    strict = {p.paper_id for p in _filter_relevant_candidates(papers, aspects)}
    monkeypatch.setenv("EVISURVEY_RELEVANCE_MIN_SCORE", "0.1")
    loose = {p.paper_id for p in _filter_relevant_candidates(papers, aspects)}

    assert strict == {pid for pid, s in scores.items() if s >= 0.6}
    assert loose == {pid for pid, s in scores.items() if s >= 0.1}
    assert strict < loose


# --- S1 relevance prefilter must not erase whole aspects -------------------------


def _coverage_fixture():
    """5 aspects; aspect 0's hits match no pooled relevance term."""
    aspects, papers_by_query = [], {}
    for k in range(5):
        query = f"asp{k} world model benchmark"
        aspects.append({"aspect_id": f"aspect_{k + 1:03d}", "keywords": [query]})
        if k == 0:
            papers_by_query[query] = [
                {
                    "unique_id": f"paper:asp0:pottery{i}",
                    "title": f"Unrelated pottery glaze study {i}",
                    "publication_published_year": 2023,
                    "abstract": "pottery glaze chemistry",
                    "citation_count": i,
                }
                for i in range(4)
            ]
        else:
            papers_by_query[query] = [
                {
                    "unique_id": f"paper:asp{k}:hit{i}",
                    "title": f"{query} study {i}",
                    "publication_published_year": 2023,
                    "abstract": f"{query} abstract {i}",
                    "citation_count": i,
                }
                for i in range(4)
            ]
    return aspects, papers_by_query


class QuerySV:
    def __init__(self, papers_by_query):
        self.papers_by_query = papers_by_query

    def meta_search(self, query, **kw):
        return {"results": self.papers_by_query.get(query, [])}


def _corpus_counts(papers):
    counts = {}
    for p in papers:
        key = p.paper_id.split(":")[1]
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_aspect_coverage_floor_keeps_every_aspect_represented(monkeypatch):
    from tools.models.requests import PipelineConfig

    aspects, papers_by_query = _coverage_fixture()
    monkeypatch.setenv("EVISURVEY_ASPECT_MIN_PAPERS", "2")
    rp, _ = run(
        "t", aspects, [], QuerySV(papers_by_query), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=1000),
    )

    counts = _corpus_counts(rp.papers)
    for k in range(5):
        assert counts.get(f"asp{k}", 0) >= 2


def test_aspect_coverage_floor_on_by_default(monkeypatch):
    """Default floor 2: a pooled-prefilter wipe of a whole aspect gets rescued."""
    from tools.models.requests import PipelineConfig

    monkeypatch.delenv("EVISURVEY_ASPECT_MIN_PAPERS", raising=False)
    aspects, papers_by_query = _coverage_fixture()
    rp, _ = run(
        "t", aspects, [], QuerySV(papers_by_query), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=1000),
    )

    counts = _corpus_counts(rp.papers)
    assert counts.get("asp0", 0) >= 2
    assert all(counts.get(f"asp{k}", 0) >= 4 for k in range(1, 5))


# --- S1 depth: MinerU fulltext bounded to the core window ------------------------


class CountingMU:
    def __init__(self):
        self.calls = 0

    def parse_url(self, url, light=True):
        self.calls += 1
        return {
            "title": "A",
            "abstract": "abs",
            "sections": [],
            "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
            "figures": [],
            "tables": [],
        }


class ManyPdfSV:
    def meta_search(self, query, **kw):
        return {"results": [
            {
                "unique_id": f"paper:{i}",
                "title": f"World model survey {i}",
                "year": 2023,
                "abstract": "world model survey",
                "url": f"https://arxiv.org/pdf/{i}",
            }
            for i in range(8)
        ]}


def test_mineru_fulltext_bounded_to_core_window():
    from tools.models.requests import PipelineConfig

    mu = CountingMU()
    cfg = PipelineConfig(use_mineru=True, max_core_papers=3)
    rp, pp = run("t", [{"keywords": ["world model"]}], [], ManyPdfSV(), mu, FakeCleaner(), [], cfg)

    assert mu.calls == 3  # only the core window reaches MinerU
    assert {p.paper_id for p in pp.papers} == {"paper:0", "paper:1", "paper:2"}
    assert sum(1 for p in rp.papers if p.parse_status == "light") == 3
    assert all(p.parse_status == "abstract_only" for p in rp.papers[3:])


def test_mineru_fulltext_covers_all_when_below_core_window():
    from tools.models.requests import PipelineConfig

    mu = CountingMU()
    cfg = PipelineConfig(use_mineru=True)  # default max_core_papers=15 > 8 papers
    rp, pp = run("t", [{"keywords": ["world model"]}], [], ManyPdfSV(), mu, FakeCleaner(), [], cfg)

    assert mu.calls == 8
    assert len(pp.papers) == 8


# --- canonical survival stage trace (specs/canonical存活链诊断.md) -----------------


class TraceSV:
    """Three hits per search: one on-topic strong, one off-topic (prefilter
    kills it), one on-topic weak (the corpus cap cut kills it)."""

    def meta_search(self, query, **kw):
        return {"results": [
            {"unique_id": "paper:keep", "title": "World model survey keep",
             "year": 2024, "abstract": "world model survey", "citation_count": 50,
             "url": "http://keep"},
            {"unique_id": "paper:cancer", "title": "Cancer statistics 2024",
             "year": 2024, "abstract": "Annual cancer incidence and mortality statistics.",
             "citation_count": 10, "url": "http://c"},
            {"unique_id": "paper:capped", "title": "World model survey tail",
             "year": 2024, "abstract": "world model survey", "citation_count": 1,
             "url": "http://t"},
        ]}


def _read_trace():
    with open(stage_trace_path(), encoding="utf-8") as f:
        return json.load(f)


def _ids(stage):
    return [p["paper_id"] for p in _read_trace()["stages"][stage]]


def test_stage_trace_records_each_checkpoint(tmp_path):
    from tools.models.requests import PipelineConfig

    assert stage_trace_path() == str(tmp_path / "p3_stage_trace.json")
    rp, _ = run(
        "t", [{"keywords": ["world model"]}], [], TraceSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=False, max_papers=1),
    )

    trace = _read_trace()
    assert trace["task_id"] == "t"
    assert trace["cap_mode"] == "top-slice"
    assert set(trace["stages"]) == {
        "raw_hits", "after_prefilter", "after_rank_cut", "after_corpus_cap"
    }
    # every entry carries both identity fields
    assert all({"paper_id", "title"} <= set(p) for p in trace["stages"]["raw_hits"])

    # raw hits keep duplicates from the 4 searches (1 query x 3 bands + landmark)
    assert len(_ids("raw_hits")) == 12
    # prefilter kills exactly the off-topic paper
    assert set(_ids("after_prefilter")) == {"paper:keep", "paper:capped"}
    # top-slice cap: the weak on-topic paper dies on the rank-ordered cut
    assert _ids("after_rank_cut") == ["paper:keep"]
    assert _ids("after_corpus_cap") == ["paper:keep"]
    # the final checkpoint mirrors the returned corpus
    assert _ids("after_corpus_cap") == [p.paper_id for p in rp.papers]


def test_stage_trace_aspect_balanced_cap_attributes_cut_to_corpus_stage(monkeypatch):
    from tools.models.requests import PipelineConfig

    monkeypatch.setenv("EVISURVEY_MAX_CORPUS", "1")
    rp, _ = run(
        "t", [{"keywords": ["world model"]}], [], TraceSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=False, max_papers=40),
    )

    trace = _read_trace()
    assert trace["cap_mode"] == "aspect-balanced"
    # the rank checkpoint records the uncut ranked list; the trim lands on the
    # corpus-cap checkpoint instead
    assert set(_ids("after_rank_cut")) == {"paper:keep", "paper:capped"}
    assert _ids("after_corpus_cap") == ["paper:keep"]
    assert _ids("after_corpus_cap") == [p.paper_id for p in rp.papers]


# --- T6: canonical-title retrieval + hard retention (specs/canonical标题检索与硬保留.md) ---

MUZERO_TITLE = "MuZero: Mastering Atari, Go, Chess and Shogi by Search"


def test_expansion_candidate_with_title_queries_full_title_with_year_window():
    from tools.models.requests import PipelineConfig

    calls = []

    class TitleExpansionSV:
        def meta_search(self, query, **kw):
            calls.append({"query": query, **kw})
            return {"results": [{
                # arXiv entry (2019) vs hint year 2020; punctuation/case differ
                "unique_id": "paper:muzero",
                "title": "muzero: mastering atari, go, chess and shogi by search",
                "publication_published_year": 2019,
                "abstract": "Mastering Atari, Go, chess and shogi with learned models.",
            }]}

    rp, pp = run(
        "t", [],
        [{"paper_id_hint": "muzero_2020", "title": MUZERO_TITLE, "survey_ref_count": 3}],
        TitleExpansionSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False),
    )
    assert calls[0]["query"] == MUZERO_TITLE
    assert calls[0]["filters"] == [
        {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2019},
        {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2021},
    ]
    assert [p.paper_id for p in rp.papers] == ["paper:muzero"]
    assert rp.papers[0].source == "survey_expansion"
    assert rp.papers[0].survey_ref_count == 3
    assert rp.papers[0].survey_ref_hints == ["muzero_2020"]
    assert pp.papers == []


def test_expansion_candidate_without_year_runs_unfiltered():
    from tools.models.requests import PipelineConfig

    calls = []

    class HintSV:
        def meta_search(self, query, **kw):
            calls.append({"query": query, **kw})
            return {"results": [{
                "unique_id": "paper:wm", "title": "World Models",
                "publication_published_year": 2018,
                "abstract": "world models",
            }]}

    rp, _ = run(
        "t", [], [{"paper_id_hint": "world_models", "survey_ref_count": 1}],
        HintSV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False),
    )
    assert calls[0]["query"] == "world models"
    assert calls[0]["filters"] == []
    assert [p.paper_id for p in rp.papers] == ["paper:wm"]


def test_normalize_title_nfkc_and_punctuation_folding():
    from tools.phases.phase3_paper_retriever import _normalize_title

    assert _normalize_title("ＭｕＺｅｒｏ： Ｍastering—atari") == _normalize_title("MuZero: Mastering, Atari!")
    assert _normalize_title("") == ""


def test_expansion_title_match_rejects_derivative_and_different_titles():
    candidate = {"paper_id_hint": "muzero_2020", "title": MUZERO_TITLE}
    # same title modulo case/punctuation is the cited paper
    assert _candidate_matches_hit(
        candidate, {"title": "MUZERO — mastering ATARI, go, chess and shogi by search"})
    # derivative work that quotes the landmark in its abstract must not pass
    assert not _candidate_matches_hit(candidate, {
        "title": "Mastering Atari Games with Deep Reinforcement Learning",
        "abstract": f"{MUZERO_TITLE} revisited",
    })
    # an empty hit title never equals a non-empty candidate title
    assert not _candidate_matches_hit(candidate, {"title": ""})


def test_expansion_no_title_candidate_keeps_term_matching():
    candidate = {"paper_id_hint": "genie_2024"}
    assert _candidate_matches_hit(candidate, {"title": "Genie: Generative Interactive Environments"})
    assert not _candidate_matches_hit(candidate, {
        "title": "Cancer statistics, 2024",
        "abstract": "Annual cancer incidence and mortality statistics.",
    })


def _expansion_cap_fixture():
    """One verified landmark (expansion search) + 5 on-topic aspect hits (aspect
    search); the landmark ranks below the aspect hits, so only hard retention
    can carry it past the corpus cap."""
    per_query = {
        MUZERO_TITLE: [{
            "unique_id": "paper:muzero",
            "title": MUZERO_TITLE,
            "publication_published_year": 2020,
            "abstract": "Learning a world model of the environment to master atari, go, chess and shogi.",
            "citation_count": 0,
        }],
        "world model": [
            {
                "unique_id": f"paper:wm{i}",
                "title": f"World model study {i}",
                "publication_published_year": 2024,
                "abstract": "world model study",
                "citation_count": 50 + i,
                "url": f"https://arxiv.org/pdf/wm{i}",
            }
            for i in range(5)
        ],
    }
    candidate = {"paper_id_hint": "muzero_2020", "title": MUZERO_TITLE, "survey_ref_count": 1}
    return per_query, candidate


def test_corpus_cap_top_slice_hard_keeps_expansion_hits():
    from tools.models.requests import PipelineConfig

    per_query, candidate = _expansion_cap_fixture()
    rp, _ = run(
        "t", [{"keywords": ["world model"]}], [candidate],
        QuerySV(per_query), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=3),
    )

    ids = [p.paper_id for p in rp.papers]
    assert len(ids) == 4  # cap 3 + 1 hard-kept landmark
    assert "paper:muzero" in ids
    # retention is part of the top-slice cut: the landmark was never cut
    assert "paper:muzero" in _ids("after_rank_cut")
    assert _ids("after_corpus_cap") == ids


def test_corpus_cap_balanced_hard_keeps_expansion_hits(monkeypatch):
    from tools.models.requests import PipelineConfig

    monkeypatch.setenv("EVISURVEY_MAX_CORPUS", "2")
    per_query, candidate = _expansion_cap_fixture()
    # "world"/"model" are generic relevance terms, so the aspect phrases use
    # "atari search" — tokens the landmark's own title carries.
    aspects = [
        {"aspect_id": f"aspect_{k + 1:03d}", "keywords": [f"atari search part{k}"]}
        for k in range(3)
    ]
    for k in range(3):
        per_query[f"atari search part{k}"] = [
            {
                "unique_id": f"paper:p{k}:{i}",
                "title": f"atari search part{k} study {i}",
                "publication_published_year": 2024,
                "abstract": f"atari search part{k} abstract",
                "citation_count": 50 + i,
            }
            for i in range(4)
        ]
    rp, _ = run(
        "t", aspects, [candidate],
        QuerySV(per_query), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=40),
    )

    ids = [p.paper_id for p in rp.papers]
    assert len(ids) == 3  # balanced cap 2 never reached the expansion bucket -> retention re-added it
    assert "paper:muzero" in ids
    assert "paper:muzero" in _ids("after_rank_cut")  # uncut ranked list
    assert _ids("after_corpus_cap") == ids


def test_expansion_hard_retention_does_not_bypass_relevance_prefilter():
    from tools.models.requests import PipelineConfig

    per_query = {
        "Cancer statistics, 2024": [{
            "unique_id": "paper:cancer",
            "title": "Cancer statistics 2024",  # title-verified: accepted as the hit
            "publication_published_year": 2024,
            "abstract": "Annual cancer incidence and mortality statistics.",
            "citation_count": 10_000,
        }],
    }
    rp, _ = run(
        "t", [{"keywords": ["world model"]}],
        [{"paper_id_hint": "cancer_2024", "title": "Cancer statistics, 2024", "survey_ref_count": 5}],
        QuerySV(per_query), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=1000),
    )
    assert rp.papers == []
    assert _ids("after_prefilter") == []


def test_to_retrieved_keeps_doc_id():
    """The SciVerse opaque doc_id is the /content key — it must survive retrieval
    (wave 1.5 identity chain: hit -> RetrievedPaper -> PaperCard -> evidence)."""
    rp = _to_retrieved({"unique_id": "paper:10.1/x", "title": "A", "doc_id": "hash-1"})
    assert rp.doc_id == "hash-1"
    # seed-corpus hits carry no doc_id -> empty string, never fabricated
    assert _to_retrieved({"unique_id": "seed:1", "title": "B"}).doc_id == ""


# --- wave 2: content-first precheck + honest parse accounting ---


class ContentSV:
    """meta-search hits carry doc_id; /content fulltext only for known docs."""

    def __init__(self, fulltext_docs):
        self.fulltext_docs = fulltext_docs  # doc_id -> markdown
        self.content_calls = []

    def meta_search(self, query, **kw):
        return {"results": [
            {"unique_id": "paper:10.1/a", "title": "A", "doc_id": "doc-a",
             "url": "http://a/paper.pdf"},
            {"unique_id": "paper:10.1/b", "title": "B", "doc_id": "doc-b",
             "url": "http://b/paper.pdf"},
            {"unique_id": "paper:10.1/c", "title": "C"},  # no doc_id, no pdf url
        ]}

    def read_full_text(self, doc_id, max_pages=8):
        self.content_calls.append(doc_id)
        return self.fulltext_docs.get(doc_id, "")


class RecordingMU:
    def __init__(self):
        self.v4_calls = []
        self.agent_calls = []

    def extract_fulltext(self, url):
        self.v4_calls.append(url)
        return {"title": "B", "abstract": "abs", "sections": [],
                "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
                "figures": [], "tables": [], "parse_status": "fulltext"}

    def parse_url(self, url, light=True):
        self.agent_calls.append(url)
        return {"title": "B", "abstract": "abs", "sections": [],
                "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
                "figures": [], "tables": []}


LONG_MD = "# B paper\n\n" + ("We study world models in games. " * 200)


def test_content_fulltext_skips_mineru_and_is_accounted():
    from tools.models.requests import PipelineConfig

    sv = ContentSV({"doc-a": LONG_MD})  # only A has SciVerse fulltext
    mu = RecordingMU()
    rp, pp = run("t2", [{"keywords": ["wm"]}], [], sv, mu, FakeCleaner(),
                 [], PipelineConfig(use_mineru=True))
    by_pid = {p.paper_id: p for p in rp.papers}
    assert by_pid["paper:10.1/a"].parse_status == "content_fulltext"
    assert by_pid["paper:10.1/b"].parse_status == "fulltext"      # mineru path
    assert by_pid["paper:10.1/c"].parse_status == "abstract_only"
    assert sv.content_calls == ["doc-a", "doc-b"]                  # every doc_id probed once
    assert mu.v4_calls == ["http://b/paper.pdf"]                   # A skipped MinerU
    assert [p.paper_id for p in pp.papers] == ["paper:10.1/a", "paper:10.1/b"]
    # page=None + reference roles flow through the shared markdown kernel
    a = next(p for p in pp.papers if p.paper_id == "paper:10.1/a")
    assert all(para.page is None for para in a.paragraphs)


def test_parse_accounting_trace_written(caplog, tmp_path):
    import json as _json
    from tools.models.requests import PipelineConfig

    sv = ContentSV({"doc-a": LONG_MD})
    mu = RecordingMU()

    class FailingMU(RecordingMU):
        def extract_fulltext(self, url):
            self.v4_calls.append(url)
            raise RuntimeError("boom")  # v4 down -> agent fallback used

        def parse_url(self, url, light=True):
            self.agent_calls.append(url)
            return {"title": "B", "abstract": "abs", "sections": [],
                    "paragraphs": [{"page": 1, "index": 0, "text": "t"}],
                    "figures": [], "tables": []}

    mu = FailingMU()
    with caplog.at_level("INFO", logger="tools.phases.phase3_paper_retriever"):
        run("t3", [{"keywords": ["wm"]}], [], sv, mu, FakeCleaner(),
            [], PipelineConfig(use_mineru=True))
    line = next(r.getMessage() for r in caplog.records if "parse accounting" in r.getMessage())
    assert "content_fulltext=1" in line and "abstract_only=1" in line
    assert "no_pdf_url=1" in line and "failures: v4" in line and "RuntimeError=1" in line
    trace = _json.loads((tmp_path / "parse_trace.json").read_text())
    assert trace["task_id"] == "t3" and len(trace["records"]) == 3
    b = next(r for r in trace["records"] if r["paper_id"] == "paper:10.1/b")
    assert b["outcome"] == "light" and b["failures"][0]["channel"] == "v4"
    c = next(r for r in trace["records"] if r["paper_id"] == "paper:10.1/c")
    assert c["skip_reason"] == "no_pdf_url"


# --- wave 3: offtopic gate / reference seeds / content figures -------------------


OFFTOPIC_HITS = [
    {"unique_id": "paper:10.1/m1", "title": "A Mean Field Games Model for Cryptocurrency Mining",
     "abstract": "We model bitcoin mining rewards as a mean field game.", "citation_count": 10},
    {"unique_id": "paper:10.1/m2", "title": "Benchmarking community drug response prediction models",
     "abstract": "Datasets and metrics for drug response prediction.", "citation_count": 20},
    {"unique_id": "paper:10.1/m3", "title": "Multimodal remote sensing benchmark datasets for land cover",
     "abstract": "Satellite imagery benchmarks for land cover classification.", "citation_count": 5},
]
CORE_HITS = [
    {"unique_id": "paper:10.1/g1", "title": "DreamerV3: Mastering Diverse Domains",
     "abstract": "World models for reinforcement learning in games.", "citation_count": 500},
    {"unique_id": "paper:10.1/g2", "title": "GameNGen: Diffusion Models are Game Engines",
     "abstract": "A neural game engine generates playable game worlds.", "citation_count": 90},
]


class MixedSV:
    """hits = the aspect-search corpus; seed_hits = records resolvable ONLY by
    exact-title lookup (how expansion/reference-seed injection finds classics)."""

    def __init__(self, hits, fulltext_docs=None, seed_hits=None):
        self.hits = hits
        self.fulltext_docs = fulltext_docs or {}
        self.seed_hits = {h["title"].strip().lower(): h for h in (seed_hits or [])}
        self.relations_calls = []

    def meta_search(self, query, **kw):
        exact = self.seed_hits.get(query.strip().lower())
        if exact is not None:
            return {"results": [exact]}
        return {"results": self.hits}

    def read_full_text(self, doc_id, max_pages=8):
        return self.fulltext_docs.get(doc_id, "")

    def get_resource(self, file_name):
        return b"\x89PNG\r\n\x1a\nasset-bytes"

    def meta_paper_relations(self, unique_id, relation="REFERENCES", page=1, page_size=200):
        self.relations_calls.append(unique_id)
        return {"items": [], "total_count": 0}


def test_offtopic_gate_drops_known_false_positives(caplog):
    from tools.models.requests import PipelineConfig

    sv = MixedSV(OFFTOPIC_HITS + CORE_HITS)
    with caplog.at_level("WARNING", logger="tools.phases.phase3_paper_retriever"):
        rp, _ = run("tw3a", [{"keywords": ["world model games"]}], [], sv,
                    RecordingMU(), FakeCleaner(), [], PipelineConfig(use_mineru=True))
    titles = [p.title for p in rp.papers]
    assert any("DreamerV3" in t for t in titles) and any("GameNGen" in t for t in titles)
    assert not any("Cryptocurrency" in t or "drug" in t.lower() or "remote sensing" in t.lower()
                   for t in titles)
    assert any("offtopic gate dropped" in r.getMessage() for r in caplog.records)


def test_reference_seeds_inject_high_frequency_classics():
    from tools.models.requests import PipelineConfig

    hits = CORE_HITS + [
        {"unique_id": "paper:10.1/bad", "title": "A Drug Response Benchmark Classic",
         "abstract": "cancer drug response", "citation_count": 900},
    ]

    class SeedSV(MixedSV):
        def meta_paper_relations(self, unique_id, relation="REFERENCES", page=1, page_size=200):
            self.relations_calls.append(unique_id)
            refs = [
                {"id": "1", "id_type": "semantic_scholar", "title": "World Models"},
                {"id": "2", "id_type": "semantic_scholar", "title": "A Drug Response Benchmark Classic"},
            ]
            if len(self.relations_calls) == 1:  # only one core paper cites it
                refs.append({"id": "3", "id_type": "semantic_scholar", "title": "Low Frequency Paper"})
            return {"items": refs, "total_count": len(refs)}

    sv = SeedSV(hits, seed_hits=[
        {"unique_id": "paper:10.1/wm", "title": "World Models",
         "abstract": "World models for learning dynamics in games and control.",
         "citation_count": 3000},
    ])
    rp, _ = run("tw3b", [{"keywords": ["world model games"]}], [], sv,
                RecordingMU(), FakeCleaner(), [], PipelineConfig(use_mineru=True))
    seeds = [p for p in rp.papers if p.source == "reference_seed"]
    assert [p.title for p in seeds] == ["World Models"]
    assert seeds[0].survey_ref_count == 3  # referenced by every distinct top paper
    assert sv.relations_calls  # relations actually consulted
    # offtopic classic candidate never injected; low-frequency one skipped
    assert not any("Drug" in p.title for p in rp.papers)
    assert not any("Low Frequency" in p.title for p in rp.papers)


def test_reference_seeds_respect_cap_and_skip_existing(monkeypatch):
    from tools.models.requests import PipelineConfig

    hits = CORE_HITS + [
        {"unique_id": "paper:10.1/wm", "title": "World Models", "citation_count": 900},
    ]

    class SeedSV(MixedSV):
        def meta_paper_relations(self, unique_id, relation="REFERENCES", page=1, page_size=200):
            return {"items": [
                {"title": "World Models"},          # already in corpus
                {"title": "PlaNet: Learning to Plan"},
                {"title": "MuZero: Mastering Games"},
            ], "total_count": 3}

    monkeypatch.setenv("EVISURVEY_REFERENCE_SEEDS", "1")
    sv = SeedSV(hits, seed_hits=[
        {"unique_id": "paper:10.1/planet", "title": "PlaNET: Learning to Plan",
         "abstract": "latent space model planning for world models in games.",
         "citation_count": 2000},
        {"unique_id": "paper:10.1/mz", "title": "MuZero: Mastering Games",
         "abstract": "model-based planning masters game environments.", "citation_count": 2500},
    ])
    rp, _ = run("tw3c", [{"keywords": ["world model games"]}], [], sv,
                RecordingMU(), FakeCleaner(), [], PipelineConfig(use_mineru=True))
    seeds = [p for p in rp.papers if p.source == "reference_seed"]
    assert len(seeds) == 1                      # cap binds
    assert seeds[0].title != "World Models"     # existing corpus paper not re-injected
    assert "MuZero" not in seeds[0].title       # cap cut the second eligible seed


FIGURE_MD = ("# GameNGen\n\n" + "Neural game engines generate playable worlds. " * 100
             + "\n\n![Figure 1: DOOM rollout](dt=2026-06-10/ht=23/abc.jpg)\n\n"
             + "![Figure 2: Latent decoder](dt=2026-06-10/ht=23/def.jpg)\n\n"
             + "![](dt=2026-06-10/ht=23/nocaption.jpg)\n")


def test_content_figures_downloaded_to_disk(tmp_path, monkeypatch):
    from tools.models.requests import PipelineConfig
    from pathlib import Path as P

    monkeypatch.chdir(tmp_path)
    sv = MixedSV([{"unique_id": "paper:10.1/g1", "title": "GameNGen", "doc_id": "doc-g",
                   "abstract": "neural game engine", "citation_count": 50,
                   "url": "http://x/g.pdf"}],
                 fulltext_docs={"doc-g": FIGURE_MD})
    rp, pp = run("tw3d", [{"keywords": ["game engine"]}], [], sv,
                 RecordingMU(), FakeCleaner(), [], PipelineConfig(use_mineru=True))
    paper = next(p for p in rp.papers if p.paper_id == "paper:10.1/g1")
    assert paper.parse_status == "content_fulltext"
    parsed = next(p for p in pp.papers if p.paper_id == "paper:10.1/g1")
    figs = parsed.figures
    # captioned figures only (empty-alt ref skipped), assets written and non-empty
    assert [f["caption"] for f in figs] == ["Figure 1: DOOM rollout", "Figure 2: Latent decoder"]
    for f in figs:
        assert f["img_path"] and (tmp_path / f["img_path"]).read_bytes() == b"\x89PNG\r\n\x1a\nasset-bytes"
        assert f["page"] is None


# --- wave 5 (1): structured topic gate -------------------------------------------

CEPHALOPOD_TOPIC = {"display_name": "Cephalopods and Marine Biology",
                    "domain": {"display_name": "Life Sciences"},
                    "field": {"display_name": "Agricultural and Biological Sciences"}}
CS_TOPIC = {"display_name": "Distributed and Parallel Computing Systems",
            "domain": {"display_name": "Physical Sciences"},
            "field": {"display_name": "Computer Science"}}


def test_to_retrieved_extracts_topic_labels():
    rp = _to_retrieved({"unique_id": "paper:10.1/x", "title": "A", "primary_topic": CS_TOPIC})
    assert rp.topic_domain == "Physical Sciences" and rp.topic_field == "Computer Science"
    assert _to_retrieved({"unique_id": "seed:1", "title": "B"}).topic_domain == ""


def test_topic_gate_rejects_life_sciences_and_medicine():
    from tools.phases.phase3_paper_retriever import _topic_gate_ok

    moth = _to_retrieved({"unique_id": "paper:10.5281/zenodo.10630803",
                          "title": "Sora phyllopa Zhou & Chen 2024, sp. nov.",
                          "primary_topic": CEPHALOPOD_TOPIC})
    med = _to_retrieved({"unique_id": "paper:10.71943/bayer",
                         "title": "Congress presentation: BMS 2024 OASIS responder analysis",
                         "primary_topic": {"domain": {"display_name": "Health Sciences"},
                                           "field": {"display_name": "Medicine"}}})
    cs = _to_retrieved({"unique_id": "paper:10.1/g", "title": "GameNGen", "primary_topic": CS_TOPIC})
    no_topic = _to_retrieved({"unique_id": "seed:2", "title": "Legacy"})
    assert not _topic_gate_ok(moth) and not _topic_gate_ok(med)
    assert _topic_gate_ok(cs) and _topic_gate_ok(no_topic)  # missing label -> pass


def test_offtopic_drop_now_rejects_structured_junk(caplog):
    from tools.models.requests import PipelineConfig

    hits = CORE_HITS + [
        {"unique_id": "paper:10.5281/zenodo.10630803", "title": "Sora phyllopa Zhou & Chen 2024, sp. nov.",
         "abstract": "A new species is described.", "primary_topic": CEPHALOPOD_TOPIC},
        {"unique_id": "paper:10.71943/bayer", "title": "Congress presentation: BMS OASIS responder analysis",
         "abstract": "Efficacy and safety analysis with no domain word for patterns.",
         "primary_topic": {"domain": {"display_name": "Health Sciences"},
                           "field": {"display_name": "Medicine"}}},
    ]
    with caplog.at_level("WARNING", logger="tools.phases.phase3_paper_retriever"):
        rp, _ = run("tw5a", [{"keywords": ["world model games"]}], [], MixedSV(hits),
                    RecordingMU(), FakeCleaner(), [], PipelineConfig(use_mineru=True))
    titles = [p.title for p in rp.papers]
    assert not any("phyllopa" in t or "BMS" in t for t in titles)
    assert any("DreamerV3" in t for t in titles)
    assert any("topic" in r.getMessage() or "offtopic gate dropped 2" in r.getMessage()
               for r in caplog.records)


# --- wave7 topic-fit gate (embedding floor + topic-core-term exemption) ---

class _FakeEmbedder:
    """Deterministic stand-in: 'of topic' texts embed opposite to everything
    else; the topic vector is [1, 0]."""

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        import numpy as np

        out = []
        for t in texts:
            v = np.array([-1.0, 0.0]) if "facial expression" in t.lower() else np.array([1.0, 0.0])
            out.append(v)
        return out


def _gate_papers():
    from tools.models.artifacts import RetrievedPaper

    return [
        RetrievedPaper(paper_id="paper:fer", title="Advances in Facial Expression Recognition",
                       abstract="A survey of facial expression recognition methods and datasets."),
        RetrievedPaper(paper_id="paper:on", title="Procedural Puzzle Generation",
                       abstract="Generating puzzle levels with grammar-based methods."),
        RetrievedPaper(paper_id="paper:term", title="Facial Expression Recognition in Games",
                       abstract="A study of facial expression recognition during gameplay."),
    ]


def test_topic_fit_gate_drops_of_topic_paper(monkeypatch):
    from tools.phases import phase3_paper_retriever as p3

    monkeypatch.setattr(p3, "_load_embedder", _FakeEmbedder)
    kept = p3._drop_topic_mismatch(_gate_papers(), "World Models for Games: A Survey", "post-dedup")
    assert [p.paper_id for p in kept] == ["paper:on", "paper:term"]


def test_topic_fit_gate_term_hit_overrides_low_similarity(monkeypatch):
    from tools.phases import phase3_paper_retriever as p3

    monkeypatch.setattr(p3, "_load_embedder", _FakeEmbedder)
    kept = p3._drop_topic_mismatch(_gate_papers(), "World Models for Games: A Survey", "post-dedup")
    # paper:term embeds "off topic" (facial expression) but its title contains
    # the topic-core term "games", so the dual condition refuses to drop it
    assert any(p.paper_id == "paper:term" for p in kept)


def test_topic_fit_gate_fails_open_when_embedder_unavailable(monkeypatch):
    from tools.phases import phase3_paper_retriever as p3

    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(p3, "_load_embedder", boom)
    papers = _gate_papers()
    assert p3._drop_topic_mismatch(papers, "World Models for Games: A Survey", "x") == papers


def test_topic_fit_gate_skips_when_no_core_terms(monkeypatch):
    from tools.phases import phase3_paper_retriever as p3

    def never():
        raise AssertionError("embedder must not load without core terms")

    monkeypatch.setattr(p3, "_load_embedder", never)
    papers = _gate_papers()
    # every token of this "topic" is stoplisted/generic -> no gate can fire
    assert p3._drop_topic_mismatch(papers, "A Survey of Models and Systems", "x") == papers


def test_topic_core_terms_drop_generic_vocabulary():
    from tools.phases.phase3_paper_retriever import _topic_core_terms

    assert _topic_core_terms("World Models for Games: A Survey") == ["games"]
