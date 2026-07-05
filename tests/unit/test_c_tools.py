import json
import re
from pathlib import Path

from tools import render_report, revise_survey, write_survey


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _citations(markdown: str) -> set[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    return set(re.findall(r"\[([^\]]+)\]", text_only))


def _minimal_inputs(tmp_path: Path):
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cards = {
        "task_id": "t",
        "paper_cards": [
            {
                "paper_id": "p1",
                "title": "World Models",
                "year": 2018,
                "category_id": "cat_1",
                "category": "World Models",
                "problem": "planning from learned dynamics",
                "method": "latent dynamics",
                "contribution": "imagination-based policy learning",
                "limitations": "short horizon",
            },
            {
                "paper_id": "p2",
                "title": "GameCraft",
                "year": 2024,
                "category_id": "cat_2",
                "category": "Neural Game Engine",
                "problem": "interactive simulation",
                "method": "generative world model",
                "contribution": "controllable game-like environments",
                "limitations": "evaluation remains difficult",
            },
        ],
    }
    evidence = {
        "task_id": "t",
        "evidence": [
            {"evidence_id": "e1", "paper_id": "p1", "text": "latent dynamics support planning"},
            {"evidence_id": "e2", "paper_id": "p2", "text": "interactive simulation supports game environments"},
        ],
    }
    taxonomy = {
        "task_id": "t",
        "categories": [
            {"category_id": "cat_1", "category_name": "World Models", "description": "Internal dynamics.", "paper_ids": ["p1"]},
            {"category_id": "cat_2", "category_name": "Neural Game Engine", "description": "Interactive generators.", "paper_ids": ["p2"]},
        ],
    }
    ready = {
        "task_id": "t",
        "allowed_paper_ids": ["p1", "p2"],
        "items": [
            {"paper_id": "p1", "title": "World Models", "year": 2018},
            {"paper_id": "p2", "title": "GameCraft", "year": 2024},
        ],
    }
    figure_bank = {"task_id": "t", "figures": []}
    table_bank = {"task_id": "t", "tables": []}
    for name, data in [
        ("paper_cards", cards),
        ("evidence_store", evidence),
        ("taxonomy", taxonomy),
        ("citation_ready_set", ready),
        ("figure_bank", figure_bank),
        ("table_bank", table_bank),
    ]:
        _write_json(cache / f"{name}.json", data)
    return cache, output


def test_write_survey_uses_ready_set_and_falls_back_without_generate_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    cache, output = _minimal_inputs(tmp_path)
    request = {
        "task_id": "t",
        "topic": "World Models and GameCraft",
        "language": "zh",
        "inputs": {
            "paper_cards_path": str(cache / "paper_cards.json"),
            "evidence_store_path": str(cache / "evidence_store.json"),
            "figure_bank_path": str(cache / "figure_bank.json"),
            "table_bank_path": str(cache / "table_bank.json"),
            "taxonomy_path": str(cache / "taxonomy.json"),
            "citation_ready_set_path": str(cache / "citation_ready_set.json"),
        },
        "outputs": {
            "survey_markdown_path": str(output / "survey.md"),
            "timeline_path": str(cache / "timeline.json"),
            "generated_artifact_bank_path": str(cache / "generated_artifact_bank.json"),
        },
    }
    request_path = tmp_path / "requests" / "survey_generation_request.json"
    _write_json(request_path, request)

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    survey = (output / "survey.md").read_text(encoding="utf-8")
    assert _citations(survey) <= {"p1", "p2"}
    assert len(survey) > 1800
    assert "generated_timeline_001" in survey
    assert "generated_summary_table_001" in survey
    bank = json.loads((cache / "generated_artifact_bank.json").read_text(encoding="utf-8"))
    assert len(bank["artifacts"]) >= 5
    assert all(Path(a["artifact_path"]).suffix != ".png" for a in bank["artifacts"])
    assert all(a["source_artifacts"] and a["supporting_papers"] for a in bank["artifacts"])
    assert all("placement_section" in a and "caption" in a for a in bank["artifacts"])
    claim_plan = json.loads(Path("cache/section_claim_plan.json").read_text(encoding="utf-8"))
    assert claim_plan["sections"][0]["selected_papers"]
    assert "topic_relevance_score" in claim_plan["sections"][0]["selected_papers"][0]


def test_revise_survey_removes_invalid_refs_and_rebuilds_actual_references(tmp_path):
    cache, output = _minimal_inputs(tmp_path)
    (output / "survey.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "survey.md").write_text(
        "# Survey\n\nClaim [p1]. Fake [fake].\n\n![Good](good_artifact)\n![Bad](bad_artifact)\n\n## References\n\n- p1: World Models (2018).\n- p2: GameCraft (2024).\n",
        encoding="utf-8",
    )
    _write_json(output / "citation_result.json", {"entries": [{"citation_id": "fake", "valid": False}, {"citation_id": "bad_artifact", "valid": False}]})
    _write_json(cache / "claim_map.json", {"entries": []})
    _write_json(cache / "generated_artifact_bank.json", {"artifacts": [{"artifact_id": "good_artifact"}]})
    request = {
        "task_id": "t",
        "inputs": {
            "survey_markdown_path": str(output / "survey.md"),
            "citation_result_path": str(output / "citation_result.json"),
            "claim_map_path": str(cache / "claim_map.json"),
            "citation_ready_set_path": str(cache / "citation_ready_set.json"),
            "generated_artifact_bank_path": str(cache / "generated_artifact_bank.json"),
        },
        "outputs": {"revised_survey_markdown_path": str(output / "survey_revised.md")},
    }
    request_path = tmp_path / "requests" / "revision_request.json"
    _write_json(request_path, request)

    result = revise_survey.run(str(request_path))

    assert result["status"] == "success"
    revised = (output / "survey_revised.md").read_text(encoding="utf-8")
    assert "[fake]" not in revised
    assert "![Bad](bad_artifact)" not in revised
    assert "- p1:" in revised
    assert "- p2:" not in revised
    assert "Revision Notes" in revised


def test_render_report_inlines_artifact_and_writes_review_outputs(tmp_path):
    cache, output = _minimal_inputs(tmp_path)
    (output / "survey.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "survey.md").write_text(
        "# Survey\n\n## Abstract\n\nGrounded text [p1].\n\n## Introduction\n\nWorld model and GameCraft agent simulator comparison [p2].\n\n![Summary](generated_summary_table_001)\n\n## Future Directions\n\nFuture challenge and limitation.\n\n## Conclusion\n\nDone.\n\n## References\n\n- p1: World Models (2018).\n- p2: GameCraft (2024).\n",
        encoding="utf-8",
    )
    asset = output / "generated_assets" / "generated_summary_table_001.md"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("# Summary Table\n\n| Metric | Value |\n|---|---:|\n| Papers | 2 |\n", encoding="utf-8")
    _write_json(cache / "timeline.json", {"events": [{"year": 2024, "category_name": "Neural Game Engine", "paper_ids": ["p2"], "summary": "interactive"}]})
    _write_json(cache / "generated_artifact_bank.json", {"artifacts": [{
        "artifact_id": "generated_summary_table_001",
        "artifact_type": "summary_table",
        "title": "Summary Table",
        "artifact_path": str(asset),
        "artifact_format": "markdown",
        "source_artifacts": ["cache/paper_cards.json"],
        "supporting_papers": ["p1", "p2"],
        "provenance": "generated_by_c_from_verified_artifacts",
        "usable_in_report": True,
    }]})
    _write_json(output / "citation_result.json", {"total_citations": 3, "valid_citations": 3, "invalid_citations": 0, "citation_validity_score": 1.0, "entries": []})
    _write_json(cache / "claim_map.json", {"entries": [{"claim_text": "Grounded text", "cited_paper_id": "p1", "status": "supported", "evidence_ids": ["e1"]}]})
    request = {
        "task_id": "t",
        "topic": "World Models and GameCraft",
        "inputs": {
            "survey_markdown_path": str(output / "survey.md"),
            "paper_cards_path": str(cache / "paper_cards.json"),
            "taxonomy_path": str(cache / "taxonomy.json"),
            "timeline_path": str(cache / "timeline.json"),
            "citation_result_path": str(output / "citation_result.json"),
            "claim_map_path": str(cache / "claim_map.json"),
            "generated_artifact_bank_path": str(cache / "generated_artifact_bank.json"),
            "evidence_store_path": str(cache / "evidence_store.json"),
            "citation_ready_set_path": str(cache / "citation_ready_set.json"),
        },
        "outputs": {
            "evaluation_report_path": str(output / "evaluation_report.json"),
            "html_report_path": str(output / "final_report.html"),
            "pdf_report_path": str(output / "final_report.pdf"),
        },
    }
    request_path = tmp_path / "requests" / "evaluation_render_request.json"
    _write_json(request_path, request)

    result = render_report.run(str(request_path))

    assert result["status"] == "success"
    html = (output / "final_report.html").read_text(encoding="utf-8")
    assert "Summary Table" in html
    assert "<table>" in html
    assert (output / "review_report.json").exists()
    review = json.loads((output / "review_report.json").read_text(encoding="utf-8"))
    for gate in ["topic_relevance", "section_depth", "citation_diversity", "artifact_integration"]:
        assert gate in review["hard_gates"]
    assert (output / "baseline_review_report.json").exists()
    assert (output / "baseline_comparison_report.md").exists()
    assert (output / "final_report.pdf").exists()
