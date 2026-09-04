from tools.clients.llm_fake import FakeLLMClient
from tools.phases.phase2_survey_analyzer import (
    SurveyStructure,
    analyze_surveys,
    build_expansion_candidates,
    run,
)
from tools.models.common import paper_id_from_seed


def _make_llm():
    return FakeLLMClient(responses=[
        ("Generate a paper taxonomy", {"categories": [{"name": "prelim_cat", "description": "dp"}]}),
        ("Refine this taxonomy", {"categories": [{"name": "refined_cat", "description": "dr", "incorporated_from": ["s1"]}]}),
    ])


def _surveys(paper_id="sv1"):
    return [{"paper_id": paper_id, "title": "T", "year": 2024,
             "meta_data": {"taxonomy_skeleton": ["x"], "top_referenced_papers": ["genie_2024"],
                           "key_sections": ["intro", "method"]}}]


def test_two_llm_calls_distinct_taxonomies():
    llm = _make_llm()
    s = run("t1", "world models", ["sim"], [{"aspect_name": "sim"}], _surveys(), llm)
    assert llm.calls == 2
    assert s.preliminary_taxonomy[0]["name"] == "prelim_cat"
    assert s.refined_taxonomy[0]["name"] == "refined_cat"
    assert s.preliminary_taxonomy != s.refined_taxonomy


def test_expansion_candidates_from_referenced_papers():
    llm = _make_llm()
    s = run("t1", "world models", ["sim"], [{"aspect_name": "sim"}], _surveys(), llm)
    assert len(s.expansion_candidates) == 1
    assert s.expansion_candidates[0]["paper_id_hint"] == "genie_2024"
    assert s.expansion_candidates[0]["source_survey"] == "sv1"
    assert s.expansion_candidates[0]["source_surveys"] == ["sv1"]
    assert s.expansion_candidates[0]["survey_ref_count"] == 1
    assert s.expansion_candidates[0]["priority"] == "high"


def test_analyze_surveys_dedups_by_paper_id():
    surveys = _surveys("sv1") + _surveys("sv1")
    result = analyze_surveys(surveys)
    assert len(result) == 1
    assert result[0]["paper_id"] == "sv1"


def test_build_expansion_candidates_aggregates_survey_refs():
    analyzed = [
        {"paper_id": "sv1", "referenced_paper_ids": ["genie_2024", "muzero_2020"]},
        {"paper_id": "sv2", "referenced_paper_ids": ["genie_2024", "genie_2024"]},
    ]
    expansion = build_expansion_candidates(analyzed)
    genie = expansion[0]
    assert genie["paper_id_hint"] == "genie_2024"
    assert genie["source_surveys"] == ["sv1", "sv2"]
    assert genie["survey_ref_count"] == 2


def test_expansion_candidates_are_bib_sourced_and_high_frequency_first():
    """Spec T1 acceptance 1: survey reference lists become source=bib candidates,
    ordered by how many seed surveys cite them."""
    analyzed = [
        {"paper_id": "sv1", "referenced_paper_ids": ["world_models_2018", "dreamerv3_2023"]},
        {"paper_id": "sv2", "referenced_paper_ids": ["world_models_2018", "muzero_2020"]},
    ]
    expansion = build_expansion_candidates(analyzed)
    assert [c["source"] for c in expansion] == ["bib"] * len(expansion)
    assert [c["paper_id_hint"] for c in expansion] == [
        "world_models_2018", "dreamerv3_2023", "muzero_2020",
    ]
    assert expansion[0]["survey_ref_count"] == 2


def test_analyze_surveys_extracts_reference_metadata():
    """Bib candidates carry title/authors/year; id hints keep their parsed year."""
    surveys = [{"paper_id": "sv1", "title": "T", "year": 2024, "meta_data": {
        "taxonomy_skeleton": [], "key_sections": [],
        "top_referenced_papers": [
            {"paper_id": "ha_2018", "title": "World Models", "authors": ["David Ha"], "year": 2018},
            "dreamerv3_2023",
            {"title": "MuZero", "authors": ["Julian Schrittwieser"], "year": 2020},
            "  ",
        ]}}]
    analyzed = analyze_surveys(surveys)
    assert analyzed[0]["referenced_paper_ids"] == ["ha_2018", "dreamerv3_2023", "muzero_2020"]
    entries = analyzed[0]["reference_entries"]
    assert entries[0] == {
        "paper_id_hint": "ha_2018", "title": "World Models", "authors": ["David Ha"], "year": 2018,
    }
    assert entries[1] == {
        "paper_id_hint": "dreamerv3_2023", "title": "", "authors": [], "year": 2023,
    }
    assert entries[2]["paper_id_hint"] == "muzero_2020"
    assert entries[2]["authors"] == ["Julian Schrittwieser"]


def test_paper_id_fallback():
    llm = _make_llm()
    surveys_no_pid = [{"title": "Survey On World Models", "year": 2024,
                        "meta_data": {"taxonomy_skeleton": [], "top_referenced_papers": []}}]
    s = run("t1", "wm", [], [], surveys_no_pid, llm)
    expected_id = paper_id_from_seed("Survey On World Models", 2024)
    assert s.analyzed_surveys[0]["paper_id"] == expected_id
    assert expected_id.startswith("seed:")


def test_analyzed_surveys_carries_metadata():
    llm = _make_llm()
    s = run("t1", "wm", ["sim"], [{"aspect_name": "sim"}], _surveys(), llm)
    a = s.analyzed_surveys[0]
    assert a["taxonomy_skeleton"] == ["x"]
    assert a["key_sections"] == ["intro", "method"]
    assert a["referenced_paper_ids"] == ["genie_2024"]
    assert a["key_claims"] == []


def test_empty_surveys_still_produces_taxonomy():
    llm = _make_llm()
    s = run("t1", "wm", [], [], [], llm)
    assert s.analyzed_surveys == []
    assert s.expansion_candidates == []
    assert llm.calls == 2
    assert s.preliminary_taxonomy
    assert s.refined_taxonomy


def test_analyze_surveys_direct():
    surveys = [{"paper_id": "p1", "title": "T", "year": 2023,
                "meta_data": {"taxonomy_skeleton": ["a"], "top_referenced_papers": ["r1", "r2"],
                              "key_sections": ["s1"]}}]
    result = analyze_surveys(surveys, mineru=None, cleaner=None)
    assert len(result) == 1
    assert result[0]["paper_id"] == "p1"
    assert result[0]["referenced_paper_ids"] == ["r1", "r2"]
    assert result[0]["key_claims"] == []


def test_survey_structure_defaults():
    ss = SurveyStructure(
        task_id="t", analyzed_surveys=[], preliminary_taxonomy=[],
        refined_taxonomy=[], expansion_candidates=[],
    )
    assert ss.survey_update_log == []
