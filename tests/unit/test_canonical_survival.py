"""Tests for scripts/audit_canonical_survival.py (specs/canonical存活链诊断.md).

One fixture covers the four acceptance cases: a paper that survives the whole
chain (cited), one killed by the relevance prefilter, one never found by search,
and one whose card never entered the citation whitelist. Alias judgment is
cross-checked against tools.evaluate_survey.canonical_hits on the same fixture.
"""
import json

import pytest

from scripts.audit_canonical_survival import run_audit
from tools.evaluate_survey import canonical_hits

# Canonical entries (label -> normalized-match target in the fixture below):
#   DreamerV3  -> full chain, cited as paper:d3
#   CancerMark -> raw hit exists, relevance prefilter drops it
#   GhostNet   -> never appears in any search hit
#   PlaNet     -> final corpus + card, but not in citation_ready_set
CANONICAL = {
    "papers": [
        {"title": "Mastering Diverse Domains through World Models", "alias": "DreamerV3", "year": 2023},
        {"title": "Cancer statistics 2024", "alias": "CancerMark", "year": 2024},
        {"title": "GhostNet: never retrieved", "alias": "GhostNet", "year": 2025},
        {"title": "Learning latent dynamics for planning from pixels", "alias": "PlaNet", "year": 2019},
    ]
}

SURVIVOR = {"paper_id": "paper:d3", "title": "DreamerV3: Mastering Diverse Domains through World Models"}
PREFILTERED = {"paper_id": "paper:cancer", "title": "Cancer statistics 2024"}
PLANET = {"paper_id": "paper:planet", "title": "Learning latent dynamics for planning from pixels"}

TRACE = {
    "task_id": "t",
    "cap_mode": "top-slice",
    "stages": {
        "raw_hits": [SURVIVOR, PREFILTERED, PLANET],
        "after_prefilter": [SURVIVOR, PLANET],
        "after_rank_cut": [SURVIVOR, PLANET],
        "after_corpus_cap": [SURVIVOR, PLANET],
    },
}

CARDS = {"task_id": "t", "paper_cards": [
    {"paper_id": "world_models_2018", "title": "World Models"},
    {"paper_id": "dreamerv3_2023", "title": "DreamerV3: Mastering Diverse Domains through World Models"},
    {"paper_id": "planet_2019", "title": "Learning latent dynamics for planning from pixels"},
]}

WHITELIST = {"task_id": "t", "allowed_paper_ids": ["world_models_2018", "dreamerv3_2023"],
             "items": CARDS["paper_cards"][:2]}

SURVEY_MD = (
    "# Survey\n"
    "DreamerV3 masters diverse domains [dreamerv3_2023].\n"
    "World models introduced latent play [paper:world_models_2018].\n"
)


@pytest.fixture()
def fixture_tree(tmp_path):
    paths = {
        "canonical": tmp_path / "canonical_papers.json",
        "trace": tmp_path / "p3_stage_trace.json",
        "cards": tmp_path / "paper_cards.json",
        "whitelist": tmp_path / "citation_ready_set.json",
        "survey": tmp_path / "survey.md",
        "out": tmp_path / "canonical_survival.json",
    }
    paths["canonical"].write_text(json.dumps(CANONICAL), encoding="utf-8")
    paths["trace"].write_text(json.dumps(TRACE), encoding="utf-8")
    paths["cards"].write_text(json.dumps(CARDS), encoding="utf-8")
    paths["whitelist"].write_text(json.dumps(WHITELIST), encoding="utf-8")
    paths["survey"].write_text(SURVEY_MD, encoding="utf-8")
    return paths


def _audit(paths):
    return run_audit(
        canonical_path=str(paths["canonical"]),
        trace_path=str(paths["trace"]),
        cards_path=str(paths["cards"]),
        whitelist_path=str(paths["whitelist"]),
        survey_path=str(paths["survey"]),
        out_path=str(paths["out"]),
    )


def test_stage_verdicts_over_full_chain(fixture_tree):
    results = {row["label"]: row["stage"] for row in _audit(fixture_tree)}
    assert results == {
        "DreamerV3": "cited",
        "CancerMark": "prefilter_drop",
        "GhostNet": "search_miss",
        "PlaNet": "whitelist_cut",
    }


def test_audit_writes_output_with_summary(fixture_tree):
    _audit(fixture_tree)
    report = json.loads(fixture_tree["out"].read_text(encoding="utf-8"))
    assert report["summary"]["cited"] == 1
    assert report["summary"]["prefilter_drop"] == 1
    assert report["summary"]["search_miss"] == 1
    assert report["summary"]["whitelist_cut"] == 1
    assert len(report["results"]) == 4


def test_alias_matching_agrees_with_canonical_hits(fixture_tree):
    """Spec acceptance 2: on the same fixture, canonical_hits and the audit must
    make identical found/not-found judgments (raw search hits as the corpus)."""
    _audit(fixture_tree)
    results = {row["label"]: row for row in
               json.loads(fixture_tree["out"].read_text(encoding="utf-8"))["results"]}
    ch = canonical_hits(TRACE["stages"]["raw_hits"], CANONICAL)
    assert set(ch["matched"]) == {
        label for label, row in results.items() if row["stage"] != "search_miss"
    }
    assert set(ch["missing"]) == {
        label for label, row in results.items() if row["stage"] == "search_miss"
    }


def test_alias_matches_colon_split_title_part():
    """The alias 'DreamerV3' matches the head of a colon title, exactly like
    canonical_hits' corpus-label expansion."""
    from scripts.audit_canonical_survival import _canonical_labels, _find_paper

    labels = _canonical_labels(CANONICAL["papers"][0])
    assert _find_paper(labels, [SURVIVOR]) == SURVIVOR
    assert _find_paper(_canonical_labels(CANONICAL["papers"][2]), [SURVIVOR]) is None


def test_whitelist_survivor_uncited_reports_whitelist_cut(fixture_tree):
    """In the whitelist but never cited in survey.md: the enum has no uncited
    stage, so the death is reported at the final gate with an explicit detail."""
    fixture_tree["survey"].write_text("# Survey\nNo citations here.\n", encoding="utf-8")
    results = {row["label"]: row for row in _audit(fixture_tree)}
    assert results["DreamerV3"]["stage"] == "whitelist_cut"
    assert "never cited" in results["DreamerV3"]["detail"]


def test_missing_input_fails_fast_with_clear_message(fixture_tree):
    fixture_tree["trace"].unlink()
    with pytest.raises(FileNotFoundError, match="p3 stage trace"):
        _audit(fixture_tree)


def test_trace_without_stages_fails_fast(fixture_tree):
    fixture_tree["trace"].write_text(json.dumps({"task_id": "t"}), encoding="utf-8")
    with pytest.raises(ValueError, match="stages"):
        _audit(fixture_tree)
