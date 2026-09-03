import re

from tools.phases.phase3_paper_retriever import (
    run,
    dedup,
    _influence_filter_sets,
    _filter_relevant_candidates,
    _paper_relevance_score,
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
    assert widest == 5 * 3 * 40  # 5 aspects x 3 queries x page_size 40, all unique hits


def test_breadth_knobs_reach_meta_search(monkeypatch):
    monkeypatch.delenv("EVISURVEY_META_PAGE_SIZE", raising=False)
    monkeypatch.delenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", raising=False)

    sv = BreadthSV()
    run("t", _breadth_aspects(1), [], sv, FakeMU(), FakeCleaner(), [], _breadth_cfg())
    assert len(sv.calls) == 3  # 1 query x 3 influence bands
    assert sv.calls[0]["page_size"] == 15

    monkeypatch.setenv("EVISURVEY_MAX_QUERIES_PER_ASPECT", "3")
    monkeypatch.setenv("EVISURVEY_META_PAGE_SIZE", "40")
    wide_sv = BreadthSV()
    run("t", _breadth_aspects(1), [], wide_sv, FakeMU(), FakeCleaner(), [], _breadth_cfg())
    assert len(wide_sv.calls) == 9  # 3 queries x 3 influence bands
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

    assert len(rp.papers) == 5 * 3 * 4  # nothing trimmed, no reordering side effect


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


def test_aspect_coverage_floor_off_by_default(monkeypatch):
    """Delivered behavior: the pooled prefilter may still wipe a whole aspect."""
    from tools.models.requests import PipelineConfig

    monkeypatch.delenv("EVISURVEY_ASPECT_MIN_PAPERS", raising=False)
    aspects, papers_by_query = _coverage_fixture()
    rp, _ = run(
        "t", aspects, [], QuerySV(papers_by_query), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_seed_fallback=False, use_mineru=False, max_papers=1000),
    )

    counts = _corpus_counts(rp.papers)
    assert counts.get("asp0", 0) == 0
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
