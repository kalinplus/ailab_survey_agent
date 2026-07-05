from tools.phases.phase3_paper_retriever import run, dedup, _to_retrieved, pipeline_config_extra
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


# --- per-aspect resilience ---


def test_per_aspect_resilience():
    from tools.models.requests import PipelineConfig

    call_count = {"n": 0}

    class FlakySV:
        def meta_search(self, query, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("aspect 1 search failed")
            return {"results": [{"unique_id": "p2", "title": "Survived", "year": 2024, "url": "http://s"}]}

    aspects = [{"keywords": ["fail"]}, {"keywords": ["ok"]}]
    rp, pp = run(
        "t", aspects, [], FlakySV(), FakeMU(), FakeCleaner(),
        [], PipelineConfig(use_mineru=False),
    )
    assert len(rp.papers) == 1
    assert rp.papers[0].title == "Survived"


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
