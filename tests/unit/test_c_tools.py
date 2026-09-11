import json
import re
from pathlib import Path

from harness.agents.relevance import EmbeddingScorer, keyword_overlap
from harness.citation_prelock import build_citation_ready_set
from scripts.build_final_seed_papers import build_final_seed_data, write_final_seed_files
from tools import render_report, revise_survey, write_survey
from tools.clients.llm_fake import FakeLLMClient


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _citations(markdown: str) -> set[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    return set(re.findall(r"\[([^\]]+)\]", text_only))


def _isolated_root(tmp_path, monkeypatch):
    """Resolve write_survey/revise_survey root-relative cache/output writes
    inside tmp_path (config.ROOT_DIR + cwd), so unit runs never clobber real
    run artifacts (same isolation the final-seed test uses)."""
    import config

    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)


# --- citation prelock (spec T1 slice 3: recency x influence selection) -------------


def _prelock_cards():
    """Mixed pool: two canonical staples (2018 / 2021) plus fresh low-impact hits."""
    return {
        "task_id": "t",
        "paper_cards": [
            {"paper_id": "p_world_models", "title": "World Models", "year": 2018,
             "citation_count": 5000, "survey_ref_count": 3},
            {"paper_id": "p_dreamerv3", "title": "DreamerV3", "year": 2021,
             "citation_count": 800, "survey_ref_count": 2},
            {"paper_id": "p_fresh_cited", "title": "Fresh Cited World Model", "year": 2025,
             "citation_count": 2, "survey_ref_count": 0},
            {"paper_id": "p_fresh_uncited", "title": "Fresh Uncited World Model", "year": 2026,
             "citation_count": 0, "survey_ref_count": 0},
        ],
    }


def _citation_index(card_ids):
    return {"task_id": "t", "citations": [{"paper_id": pid} for pid in card_ids]}


def _prelock(card_ids, cap):
    return build_citation_ready_set(
        task_id="t",
        paper_cards=_prelock_cards(),
        citation_index=_citation_index(card_ids),
        evidence_store={"task_id": "t", "evidence": []},
        max_core_papers=cap,
    )


def test_prelock_keeps_classic_high_influence_paper_when_cap_binds():
    """Spec T1 acceptance 3: canonical staples must not be pushed out of the whitelist
    by a pure newest-first cut (they are the first casualties of that order)."""
    all_ids = ["p_world_models", "p_dreamerv3", "p_fresh_cited", "p_fresh_uncited"]
    # With the bib pin (cap binding): the highest survey_ref staple is pinned
    # first, the blend fills the rest — at cap 2 both staples survive the cut.
    assert _prelock(all_ids, 2)["allowed_paper_ids"] == ["p_world_models", "p_dreamerv3"]
    assert _prelock(all_ids, 3)["allowed_paper_ids"] == ["p_world_models", "p_dreamerv3", "p_fresh_cited"]
    assert "p_world_models" in _prelock(all_ids, 3)["allowed_paper_ids"]


def test_prelock_items_carry_doc_id():
    """Wave 3: ready-set items pass the card's doc_id through — it is the
    /content key the repair backfill grounds own-paper claims by."""
    cards = _prelock_cards()
    cards["paper_cards"][0]["doc_id"] = "doc-hash-1"
    result = build_citation_ready_set(
        task_id="t", paper_cards=cards,
        citation_index=_citation_index(["p_world_models"]),
        evidence_store={"task_id": "t", "evidence": []}, max_core_papers=5)
    item = next(i for i in result["items"] if i["paper_id"] == "p_world_models")
    assert item["doc_id"] == "doc-hash-1"
    assert all(i.get("doc_id", "") == "" for i in result["items"]
               if i["paper_id"] != "p_world_models")


def test_prelock_selection_matches_legacy_set_when_pool_below_cap():
    """pool <= cap: the whitelisted set is unchanged, now ordered by the blend."""
    all_ids = ["p_world_models", "p_dreamerv3", "p_fresh_cited", "p_fresh_uncited"]
    result = _prelock(all_ids, 10)
    assert set(result["allowed_paper_ids"]) == set(all_ids)
    assert result["allowed_paper_ids"] == [
        "p_dreamerv3", "p_fresh_cited", "p_world_models", "p_fresh_uncited",
    ]


def test_prelock_degrades_to_recency_without_influence_signals():
    """Legacy bundles carry no citation_count/survey_ref_count: pure recency order."""
    cards = {"task_id": "t", "paper_cards": [
        {"paper_id": "p_old", "title": "Old", "year": 2018},
        {"paper_id": "p_new", "title": "New", "year": 2025},
    ]}
    result = build_citation_ready_set(
        task_id="t",
        paper_cards=cards,
        citation_index=_citation_index(["p_old", "p_new"]),
        evidence_store={"task_id": "t", "evidence": []},
        max_core_papers=1,
    )
    assert result["allowed_paper_ids"] == ["p_new"]


def _bind_writer_fixture(cards, evidence):
    """Explicit, invented unit-fixture claims with real fixture block identities.

    Production never upgrades legacy card fields this way; the tests declare
    their own source roles and assertions instead of relying on that old bug.
    """
    for ev in evidence:
        card = next(c for c in cards if c["paper_id"] == ev["paper_id"])
        ev["source_type"] = "paragraph"
        ev["supports_claims"] = []
        for field, role, dimension in [("method", "own_method", "method"),
                                       ("contribution", "own_result", "key_results"),
                                       ("limitations", "own_limitation", "limitations")]:
            value = card.get(field)
            if not value:
                continue
            text = f"{card['title']} reports {field} findings on {value}"
            quote = f"We report {field} findings on {value}."
            ev["text"] += " " + quote
            ev["supports_claims"].append(dict(paper_id=ev["paper_id"], claim_text=text,
                dimension=dimension, source_role=role, source_quote=quote,
                evidence_ids=[ev["evidence_id"]], support_type="direct", confidence=.9))


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
    _bind_writer_fixture(cards["paper_cards"], evidence["evidence"])
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
    _isolated_root(tmp_path, monkeypatch)
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
    assert len(survey) > 1000  # source-constrained output omits legacy filler
    assert "publication_timeline" in survey
    assert "representative_systems" in survey
    bank = json.loads((cache / "generated_artifact_bank.json").read_text(encoding="utf-8"))
    assert len(bank["artifacts"]) >= 8
    assert all(a["source_artifacts"] and a["supporting_papers"] for a in bank["artifacts"])
    assert all("placement_section" in a and "caption" in a and "display_number" in a for a in bank["artifacts"])
    assert all(a.get("alt_text") for a in bank["artifacts"] if a.get("artifact_kind") == "figure")
    assert all(a.get("table_headers") for a in bank["artifacts"] if a.get("artifact_kind") == "table")
    claim_plan = json.loads((tmp_path / "cache" / "section_claim_plan.json").read_text(encoding="utf-8"))
    assert claim_plan["sections"][0]["selected_papers"]
    assert "topic_relevance_score" in claim_plan["sections"][0]["selected_papers"][0]


def test_revise_survey_removes_invalid_refs_and_rebuilds_actual_references(tmp_path, monkeypatch):
    _isolated_root(tmp_path, monkeypatch)
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
    # wave 5 (2a): notes live in the sidecar file, never in the delivered md
    assert "Revision Notes" not in revised
    assert "Revision Notes" in (output / "revision_notes.md").read_text(encoding="utf-8")


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
    assert "public_readiness" in review["hard_gates"]
    assert (output / "baseline_review_report.json").exists()
    assert (output / "baseline_comparison_report.md").exists()
    assert (output / "final_report.pdf").exists()
    assert (output / "citation_display_map.json").exists()
    assert (output / "survey_public.md").exists()
    assert (output / "survey.html").exists()
    assert (output / "survey.pdf").exists()
    assert (output / "survey_submission.md").exists()
    assert (output / "submission_readiness_report.json").exists()
    assert (output / "harness_audit_report.html").exists()
    public_md = (output / "survey_public.md").read_text(encoding="utf-8")
    public_html = (output / "final_report.html").read_text(encoding="utf-8")
    assert "[1]" in public_md
    assert "[p1]" not in public_md
    assert "generated_summary_table_001" not in public_md
    assert "seed:" not in public_md
    assert "CitationReadySet" not in public_md
    assert "ReviewBoard" not in public_html
    assert (output / "survey.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "final_report.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "survey.pdf").stat().st_size > 50000
    assert (output / "final_report.pdf").stat().st_size > 50000
    audit_html = (output / "harness_audit_report.html").read_text(encoding="utf-8")
    assert "artifact_id" in audit_html


def test_final_seed_papers_are_topic_relevant(tmp_path):
    data = build_final_seed_data()
    cards = data["paper_cards"]["paper_cards"]
    assert len(cards) >= 12
    text = " ".join(card["title"] + " " + card["category"] for card in cards).lower()
    for forbidden in ["diabetes", "deblurring", "indonlu", "wireless network"]:
        assert forbidden not in text
    assert all(card["topic_relevance_score"] >= 0.9 for card in cards)


def test_final_seed_write_and_render_submission_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAL_SEED_PAPERS", "1")
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    # Resolve everything inside tmp_path: tools resolve relative paths against
    # config.ROOT_DIR, and write_final_seed_files writes both <name>.json and
    # final_<name>.json — against the real repo that clobbers live cache
    # artifacts (a full-suite run once destroyed a real run's evidence store).
    import config
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    write_final_seed_files(tmp_path)
    request = {
        "task_id": "t_final",
        "topic": "World Models and GameCraft for Interactive Game Intelligence",
        "language": "en",
        "inputs": {
            "paper_cards_path": "cache/paper_cards.json",
            "evidence_store_path": "cache/evidence_store.json",
            "figure_bank_path": "cache/figure_bank.json",
            "table_bank_path": "cache/table_bank.json",
            "taxonomy_path": "cache/taxonomy.json",
            "citation_ready_set_path": "cache/citation_ready_set.json",
        },
        "outputs": {
            "survey_markdown_path": str(tmp_path / "output" / "survey.md"),
            "timeline_path": str(tmp_path / "cache" / "timeline.json"),
            "generated_artifact_bank_path": str(tmp_path / "cache" / "generated_artifact_bank.json"),
        },
    }
    request_path = tmp_path / "requests" / "survey_generation_request.json"
    _write_json(request_path, request)
    write_result = write_survey.run(str(request_path))
    assert write_result["status"] == "partial_success"  # legacy seed cards have no source-role contract

    output = tmp_path / "output"
    cache = tmp_path / "cache"
    _write_json(output / "citation_result.json", {"total_citations": 20, "valid_citations": 20, "invalid_citations": 0, "citation_validity_score": 1.0, "entries": []})
    _write_json(cache / "claim_map.json", {"entries": []})
    render_request = {
        "task_id": "t_final",
        "topic": "World Models and GameCraft for Interactive Game Intelligence",
        "inputs": {
            "survey_markdown_path": str(output / "survey.md"),
            "paper_cards_path": "cache/paper_cards.json",
            "taxonomy_path": "cache/taxonomy.json",
            "timeline_path": str(cache / "timeline.json"),
            "citation_result_path": str(output / "citation_result.json"),
            "claim_map_path": str(cache / "claim_map.json"),
            "generated_artifact_bank_path": str(cache / "generated_artifact_bank.json"),
            "evidence_store_path": "cache/evidence_store.json",
            "citation_ready_set_path": "cache/citation_ready_set.json",
        },
        "outputs": {
            "evaluation_report_path": str(output / "evaluation_report.json"),
            "html_report_path": str(output / "final_report.html"),
            "pdf_report_path": str(output / "final_report.pdf"),
        },
    }
    render_request_path = tmp_path / "requests" / "evaluation_render_request.json"
    _write_json(render_request_path, render_request)
    render_report.run(str(render_request_path))

    submission = (output / "survey_submission.md").read_text(encoding="utf-8")
    html = (output / "final_report.html").read_text(encoding="utf-8")
    for forbidden in ["seed:", "CitationReadySet", "PaperCards", "EvidenceStore", "ReviewBoard", "C module", "harness chain", "diabetes", "deblurring", "IndoNLU"]:
        assert forbidden not in submission
        assert forbidden not in html
    assert "<thead><tr><th>" in html
    assert (output / "survey.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "final_report.pdf").read_bytes().startswith(b"%PDF")
    readiness = json.loads((output / "submission_readiness_report.json").read_text(encoding="utf-8"))
    assert not readiness["pass"]  # source-poor legacy seed is not submission ready
    review = json.loads((output / "review_report.json").read_text(encoding="utf-8"))
    assert review["final_decision"] != "submission_ready"


def test_plotting_tools_force_agg_backend():
    # write_survey/_draw_*_png and render_report's PDF fallback run inside
    # ToolRegistry worker threads; a GUI backend (macosx on this machine)
    # raises "Cannot create a GUI FigureManager outside of the main thread".
    # Both modules must pin the non-interactive agg backend on import.
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


# --- S2 hybrid writer kernel (specs/写作端混合内核.md) ----------------------


_CATEGORY_SPECS = [
    ("cat_1", "World Models", "latent dynamics for planning"),
    ("cat_2", "Neural Game Engine", "generative interactive environments"),
    ("cat_3", "Game Agent Training", "self play and policy learning"),
    ("cat_4", "Open Simulator Benchmarks", "shared evaluation environments"),
    ("cat_5", "Generative Environment Design", "controllable scene synthesis"),
]


def _scaled_inputs(tmp_path: Path, papers_per_category: int, native_categories: int | None = None):
    """5 categories x N topical cards, plus one card that fits no category.

    `native_categories` limits how many categories actually own their papers,
    which forces the empty-pool sections through the bounded-reuse path.
    """
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    native_categories = native_categories or len(_CATEGORY_SPECS)
    cards: list[dict] = []
    evidence: list[dict] = []
    categories: list[dict] = []
    for cat_index, (cat_id, name, description) in enumerate(_CATEGORY_SPECS, start=1):
        categories.append({"category_id": cat_id, "category_name": name, "description": description, "paper_ids": []})
        count = papers_per_category if cat_index <= native_categories else 0
        for slot in range(count):
            pid = f"p{cat_index}_{slot}"
            cards.append(
                {
                    "paper_id": pid,
                    "title": f"{name} study {slot}",
                    "year": 2018 + slot,
                    "category_id": cat_id,
                    "category": name,
                    "problem": f"{name.lower()} problem {slot}",
                    "method": f"{description} method {slot}",
                    "contribution": f"{name.lower()} contribution {slot} for game agents",
                    "abstract": f"game agent study {slot} of {description}",

                    "limitations": f"{name.lower()} limitation {slot}",
                }
            )
            categories[-1]["paper_ids"].append(pid)
            evidence.append(
                {
                    "evidence_id": f"e_{pid}",
                    "paper_id": pid,
                    "text": f"The {name.lower()} experiment {slot} reports measured rollout behaviour over held-out episodes.",
                }
            )
    cards.append(
        {
            "paper_id": "offtopic",
            "title": "Wireless mesh spectrum scheduling",
            "year": 2021,
            "category_id": "cat_9",
            "category": "Wireless Networks",
            "problem": "spectrum allocation problem",
            "method": "wireless mesh scheduling method",
            "contribution": "spectrum allocation contribution",
            "limitations": "wireless scheduling limitation",
        }
    )
    _bind_writer_fixture(cards, evidence)
    for name, data in [
        ("paper_cards", {"task_id": "t", "paper_cards": cards}),
        ("evidence_store", {"task_id": "t", "evidence": evidence}),
        ("taxonomy", {"task_id": "t", "categories": categories}),
        ("citation_ready_set", {"task_id": "t", "allowed_paper_ids": [card["paper_id"] for card in cards]}),
        ("figure_bank", {"task_id": "t", "figures": []}),
        ("table_bank", {"task_id": "t", "tables": []}),
    ]:
        _write_json(cache / f"{name}.json", data)
    return cache, output


def _survey_request(tmp_path: Path, cache: Path, output: Path, language: str) -> Path:
    request = {
        "task_id": "t",
        "topic": "World Models and GameCraft",
        "language": language,
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
    return request_path


def _keyword_scorer():
    # no _embed and load disabled -> deterministic keyword column, no model download
    scorer = EmbeddingScorer()
    scorer._failed = True
    return scorer


def _section_body(survey: str, title: str) -> str:
    chunks = re.split(r"(?m)^##\s+", survey)
    for chunk in chunks[1:]:
        heading, _, body = chunk.partition("\n")
        if heading.strip() == title:
            return body
    raise AssertionError(f"section {title!r} missing from survey")


def test_writer_llm_gate_off_keeps_template_output_byte_identical(tmp_path, monkeypatch):
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _minimal_inputs(tmp_path)
    request_path = _survey_request(tmp_path, cache, output, "zh")
    default_run = write_survey.run(str(request_path))
    default_md = (output / "survey.md").read_text(encoding="utf-8")

    monkeypatch.setenv("EVISURVEY_WRITER_LLM", "0")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["outputs"]["survey_markdown_path"] = str(output / "survey_off.md")
    _write_json(request_path, request)
    explicit_run = write_survey.run(str(request_path))

    assert default_run["status"] == explicit_run["status"] == "success"
    assert (output / "survey_off.md").read_text(encoding="utf-8") == default_md


def test_section_selection_is_disjoint_relevance_floored_and_never_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")  # keyword column, no model download
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _scaled_inputs(tmp_path, papers_per_category=5)
    request_path = _survey_request(tmp_path, cache, output, "en")

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    plan = json.loads((tmp_path / "cache" / "section_claim_plan.json").read_text(encoding="utf-8"))
    themed = plan["sections"][: len(_CATEGORY_SPECS)]
    assert len(themed) == len(_CATEGORY_SPECS)
    # no section may lose its body to a greedier one (eval v2 empty-section trap)
    assert all(len(section["selected_papers"]) >= 1 for section in themed)
    # any two sections share at most 2 of their 5 papers
    id_sets = [{paper["paper_id"] for paper in section["selected_papers"]} for section in themed]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            assert len(id_sets[i] & id_sets[j]) <= 2
    # relevance floor (keyword fallback): no section keeps a globally bottom-fitting card
    cards = json.loads((cache / "paper_cards.json").read_text(encoding="utf-8"))["paper_cards"]
    for section, (cat_id, name, description) in zip(themed, _CATEGORY_SPECS):
        section_text = f"{name}. {description}"
        relevance = {
            card["paper_id"]: keyword_overlap(section_text, write_survey._card_text(card)) for card in cards
        }
        selected = {paper["paper_id"] for paper in section["selected_papers"]}
        assert min(relevance[paper_id] for paper_id in selected) > min(relevance.values())
        assert "offtopic" not in selected


def test_empty_pool_sections_reuse_within_bound_and_keep_their_native_papers(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _scaled_inputs(tmp_path, papers_per_category=3, native_categories=2)
    request_path = _survey_request(tmp_path, cache, output, "en")

    result = write_survey.run(str(request_path))

    # Empty taxonomy buckets have no source-bound claims after bounded reuse;
    # the writer surfaces a partial result instead of filling them from fields.
    assert result["status"] == "partial_success"
    plan = json.loads((tmp_path / "cache" / "section_claim_plan.json").read_text(encoding="utf-8"))
    themed = plan["sections"][: len(_CATEGORY_SPECS)]
    assert all(len(section["selected_papers"]) >= 1 for section in themed)
    id_sets = [{paper["paper_id"] for paper in section["selected_papers"]} for section in themed]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            assert len(id_sets[i] & id_sets[j]) <= 2
    by_title = {section["section_title"]: ids for section, ids in zip(themed, id_sets)}
    assert "p1_0" in by_title["World Models"]
    assert "p2_0" in by_title["Neural Game Engine"]


def test_template_output_has_no_meta_talk_and_cited_tail_sections(tmp_path, monkeypatch):
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _minimal_inputs(tmp_path)
    for language in ["zh", "en"]:
        request_path = _survey_request(tmp_path, cache, output, language)
        result = write_survey.run(str(request_path))
        assert result["status"] == "success"
        survey = (output / "survey.md").read_text(encoding="utf-8")
        folded = survey.casefold()
        for phrase in write_survey.BANNED_META_PHRASES:
            assert phrase not in folded, f"{language}: meta-talk leaked: {phrase}"
        tail_titles = ["开放挑战", "未来方向"] if language == "zh" else ["Open Challenges", "Future Directions"]
        for title in tail_titles:
            body = _section_body(survey, title)
            citations = _citations(body)
            assert citations <= {"p1", "p2"}, f"{language}: {title} cites outside the whitelist"
            assert body.strip(), f"{language}: {title} must state the evidence gap if all premises were already discussed"


def _llm_inputs(tmp_path: Path):
    """2 categories x 3 papers: several ids per section for the whitelist check."""
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    specs = [("cat_1", "World Models", "latent dynamics for planning"), ("cat_2", "Neural Game Engine", "generative interactive environments")]
    cards: list[dict] = []
    evidence: list[dict] = []
    categories: list[dict] = []
    for cat_index, (cat_id, name, description) in enumerate(specs, start=1):
        categories.append({"category_id": cat_id, "category_name": name, "description": description, "paper_ids": []})
        for slot in range(3):
            pid = f"p{cat_index}_{slot}"
            cards.append(
                {
                    "paper_id": pid,
                    "title": f"{name} study {slot}",
                    "year": 2018 + slot,
                    "category_id": cat_id,
                    "category": name,
                    "problem": f"{name.lower()} problem {slot}",
                    "method": f"{description} method {slot}",
                    "contribution": f"{name.lower()} contribution {slot}",
                    "limitations": f"{name.lower()} limitation {slot}",
                }
            )
            categories[-1]["paper_ids"].append(pid)
            evidence.append(
                {
                    "evidence_id": f"e_{pid}",
                    "paper_id": pid,
                    "text": f"The {name.lower()} experiment {slot} reports measured rollout behaviour.",
                }
            )
    _bind_writer_fixture(cards, evidence)
    for name, data in [
        ("paper_cards", {"task_id": "t", "paper_cards": cards}),
        ("evidence_store", {"task_id": "t", "evidence": evidence}),
        ("taxonomy", {"task_id": "t", "categories": categories}),
        ("citation_ready_set", {"task_id": "t", "allowed_paper_ids": [card["paper_id"] for card in cards]}),
        ("figure_bank", {"task_id": "t", "figures": []}),
        ("table_bank", {"task_id": "t", "tables": []}),
    ]:
        _write_json(cache / f"{name}.json", data)
    return cache, output


def _section_whitelists(cache: Path) -> dict[str, set[str]]:
    """The plan the writer will rebuild: same inputs, same deterministic selection."""
    taxonomy = json.loads((cache / "taxonomy.json").read_text(encoding="utf-8"))["categories"]
    cards = json.loads((cache / "paper_cards.json").read_text(encoding="utf-8"))["paper_cards"]
    evidence = json.loads((cache / "evidence_store.json").read_text(encoding="utf-8"))["evidence"]
    evidence_by_paper = {}
    for item in evidence:
        evidence_by_paper.setdefault(item["paper_id"], []).append(item)
    scored = write_survey._score_cards_for_topic(cards, evidence_by_paper, taxonomy, "World Models and GameCraft")
    plan = write_survey._build_section_claim_plan("t", taxonomy, scored, evidence_by_paper, scorer=_keyword_scorer())
    return {section["section_title"]: set(section["allowed_paper_ids"]) for section in plan}


def test_llm_sections_stay_inside_their_whitelist(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _llm_inputs(tmp_path)
    request_path = _survey_request(tmp_path, cache, output, "en")
    whitelists = _section_whitelists(cache)
    assert all(len(ids) >= 3 for ids in whitelists.values())

    # Task-marked replies first: FakeLLMClient returns the first response whose
    # tag occurs in the prompt, and the tail-section prompts quote paper titles
    # that also appear in the section prompts.
    replies = [("draft the introduction", "This review asks how learned worlds can be evaluated.")]
    for title, allowed in whitelists.items():
        replies.append((title, json.dumps([
            {"text": f"{title} models use learned predictions", "cite": ["P1"], "sources": ["P1S1"], "scope": "method"},
            {"text": "This sentence leaks the harness chain", "cite": ["P1"], "sources": ["P1S1"], "scope": "method"},
            {"text": "A foreign result must be rejected", "cite": ["P9"], "sources": ["P1S1"], "scope": "method"},
        ])))
    fake = FakeLLMClient(responses=replies)
    monkeypatch.setenv("EVISURVEY_WRITER_LLM", "1")
    monkeypatch.setattr(write_survey, "_writer_llm_chat", lambda cfg: fake.chat)

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    assert fake.calls >= len(whitelists), "each themed section must be drafted by the LLM"
    survey = (output / "survey.md").read_text(encoding="utf-8")
    assert "harness chain" not in survey
    assert "[offtopic]" not in survey
    for title, allowed in whitelists.items():
        body = _section_body(survey, title)
        citations = _citations(body)
        assert citations <= allowed, f"{title}: citations outside the section whitelist"
        assert "models use learned predictions" in body
        assert "Its method can be summarized as" not in body
    abstract = _section_body(survey, "Abstract")
    assert _citations(abstract) <= set().union(*whitelists.values())


def test_llm_aliases_map_back_and_leaked_ids_are_dropped():
    section = {
        "section_title": "Theme",
        "section_goal": "compare the assigned papers",
        "selected_papers": [
            {
                "paper_id": "p_alpha",
                "title": "Alpha",
                "problem": "planning",
                "method": "latent dynamics",
                "contribution": "policy learning",
                "limitations": "short horizon",
                "evidence_snippets": ["alpha evidence sentence"],
            },
            {
                "paper_id": "p_beta",
                "title": "Beta",
                "problem": "control",
                "method": "world model rollout",
                "contribution": "benchmark",
                "limitations": "narrow domain",
                "evidence_snippets": ["beta evidence sentence"],
            },
        ],
    }
    reply = "\n".join(
        [
            "[SUMMARY] Alpha studies planning from learned dynamics [P1].",
            "[COMPARISON] Beta differs in dynamics and role [P2]. A bare leak paper:10. 1016/j. fake must vanish.",
            "[LIMITATION] Unknown tags are dropped [P9]. The evidence does not establish scale [P2].",
        ]
    )
    paragraphs = [write_survey._sanitize_llm_paragraph(write_survey._map_alias_citations(part, {"P1": "p_alpha", "P2": "p_beta"}), {"p_alpha", "p_beta"}) for part in write_survey._parse_move_paragraphs(reply)]
    text = " ".join(paragraphs)

    assert "[p_alpha]" in text and "P1" not in text
    assert "[p_beta]" in text and "1016" not in text
    assert "P9" not in text and "[p_beta]" in text


def test_both_citation_styles_bind_to_the_whitelist():
    """Author-prominent and clustered tags both survive alias mapping and filtering."""
    section = {
        "section_title": "Theme",
        "section_goal": "compare the assigned papers",
        "selected_papers": [
            {
                "paper_id": "p_alpha",
                "title": "Alpha",
                "problem": "planning",
                "method": "latent dynamics",
                "contribution": "policy learning",
                "limitations": "short horizon",
                "evidence_snippets": ["alpha evidence sentence"],
            },
            {
                "paper_id": "p_beta",
                "title": "Beta",
                "problem": "control",
                "method": "world model rollout",
                "contribution": "benchmark",
                "limitations": "narrow domain",
                "evidence_snippets": ["beta evidence sentence"],
            },
        ],
    }
    reply = "\n\n".join(
        [
            "Alpha studies planning from learned dynamics [P1].",
            "Simulation scales to larger worlds. [P2]",
            "Later work agrees on the mechanism [P1] [P2].",
        ]
    )
    paragraphs = [write_survey._sanitize_llm_paragraph(write_survey._map_alias_citations(part, {"P1": "p_alpha", "P2": "p_beta"}), {"p_alpha", "p_beta"}) for part in write_survey._parse_move_paragraphs(reply)]
    text = " ".join(paragraphs)

    # author-prominent: title in subject position, tag kept
    assert "Alpha studies planning from learned dynamics [p_alpha]." in text
    # stranded tag pulled back inside its sentence (no citation-only fragment)
    assert "Simulation scales to larger worlds [p_beta]." in text
    assert "larger worlds. [p_beta]" not in text
    # information-prominent cluster binds both ids
    assert "the mechanism [p_alpha] [p_beta]." in text
    for paragraph in paragraphs:
        assert _citations(paragraph) <= {"p_alpha", "p_beta"}


def test_stranded_legacy_id_tag_is_pulled_inside_the_sentence():
    assert write_survey._map_alias_citations("The gains hold at scale. [paper:x].", {}) == "The gains hold at scale [paper:x]."
    assert write_survey._map_alias_citations("Both hold. [P1] [P2].", {"P1": "p_a", "P2": "p_b"}) == "Both hold [p_a] [p_b]."


def _prose_sentences(text: str) -> list[str]:
    prose = "\n".join(line for line in text.splitlines() if not line.strip().startswith("!["))
    return [sentence.strip() for sentence in write_survey._split_sentences(prose) if sentence.strip()]


def _citing_sentences(text: str) -> set[str]:
    return {sentence for sentence in _prose_sentences(text) if _citations(sentence)}


def test_no_sentence_repeats_across_body_sections_and_oc_fd_are_disjoint(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _scaled_inputs(tmp_path, papers_per_category=3)
    request_path = _survey_request(tmp_path, cache, output, "en")

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    survey = (output / "survey.md").read_text(encoding="utf-8")
    plan = json.loads((tmp_path / "cache" / "section_claim_plan.json").read_text(encoding="utf-8"))
    body_titles = [section["section_title"] for section in plan["sections"]]
    assert len(body_titles) >= 3

    owner: dict[str, str] = {}
    for title in body_titles:
        for sentence in _prose_sentences(_section_body(survey, title)):
            key = write_survey._norm_sentence(sentence)
            assert key not in owner, f"sentence rendered in {owner[key]!r} and {title!r}: {sentence!r}"
            owner[key] = title

    # OC and FD must not draw on the same evidence bit (S0: 0.961 similarity)
    entries = write_survey._section_paper_entries(plan["sections"])
    open_bits, direction_bits = write_survey._split_limitation_pool(write_survey._limitation_pool(entries))
    assert open_bits and direction_bits
    open_texts = {write_survey._norm_sentence(bit["text"]) for bit in open_bits}
    direction_texts = {write_survey._norm_sentence(bit["text"]) for bit in direction_bits}
    assert open_texts.isdisjoint(direction_texts)
    assert {bit["paper_id"] for bit in open_bits}.isdisjoint({bit["paper_id"] for bit in direction_bits})
    open_body = _section_body(survey, "Open Challenges")
    direction_body = _section_body(survey, "Future Directions")
    # Wave8 ③: limitation bits already rendered in the body no longer consume a
    # pool slot, so a fixture whose body consumed every bit honestly degrades
    # to the empty-branch sentence instead of a dedup-eaten bit.
    assert ("limitations discussed above" in open_body
            or "captured evidence does not yet support" in open_body)
    assert ("review proposes" in direction_body
            or "do not yet justify" in direction_body)
    assert not _citing_sentences(open_body) & _citing_sentences(direction_body)


def test_abstract_and_intro_are_differentiated(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _scaled_inputs(tmp_path, papers_per_category=3)
    request_path = _survey_request(tmp_path, cache, output, "en")

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    survey = (output / "survey.md").read_text(encoding="utf-8")
    allowed = set(json.loads((cache / "citation_ready_set.json").read_text(encoding="utf-8"))["allowed_paper_ids"])
    plan = json.loads((tmp_path / "cache" / "section_claim_plan.json").read_text(encoding="utf-8"))["sections"]

    abstract = _section_body(survey, "Abstract")
    intro = _section_body(survey, "Introduction")
    assert not _citations(abstract), "metadata inventory is not an evidence assertion"
    assert _citations(abstract) <= allowed
    for section in plan:
        assert section["section_title"] in intro, f"intro does not preview {section['section_title']!r}"
    abstract_keys = {write_survey._norm_sentence(s) for s in _prose_sentences(abstract)}
    intro_keys = {write_survey._norm_sentence(s) for s in _prose_sentences(intro)}
    assert abstract_keys and intro_keys and not abstract_keys & intro_keys
    assert _citations(intro) <= allowed


# --- T12 intro responsibility dedup (specs/Intro职责化去重.md) -----------------


def test_intro_jaccard_guard_deletes_restatement_keeps_fresh_sentences():
    """Acceptance 1: intro sentence with bag Jaccard >= 0.6 vs a body sentence
    is dropped; below threshold it stays."""
    intro = (
        "Game intelligence needs models that learn how worlds evolve. "
        "Imagined rollouts show that Dreamer learns latent dynamics for planning. "
        "Whether learned simulators can support fair evaluation remains open. "
        "Hand-written rules alone cannot capture how game worlds evolve."
    )
    body = [
        "Dreamer learns latent dynamics for planning in imagined rollouts [p_alpha].",
        "Benchmarks collect shared evaluation environments for game agents [p_beta].",
    ]
    sections = [{"section_title": "Benchmarks"}]

    out = write_survey._dedupe_intro_against_body(intro, body, sections)

    assert "Imagined rollouts show that" not in out  # 7/9 = 0.78 vs body[0]
    assert "Game intelligence needs models" in out
    assert "remains open" in out
    assert "Hand-written rules alone" in out


def test_intro_guard_keeps_everything_when_cut_would_hollow_it_out():
    """Acceptance 1: a cut leaving fewer than 3 sentences is rolled back whole."""
    intro = (
        "Dreamer learns latent dynamics for planning in imagined rollouts. "
        "Game intelligence needs grounded evidence."
    )
    body = ["Dreamer learns latent dynamics for planning in imagined rollouts [p_alpha]."]

    out = write_survey._dedupe_intro_against_body(intro, body, [{"section_title": "T"}])

    assert out == intro  # dropping the restatement would leave 1 sentence
    # no body sentences to compare against: the intro passes through untouched
    assert write_survey._dedupe_intro_against_body(intro, [], [{"section_title": "T"}]) == intro


def test_intro_roadmap_title_overlap_is_exempt_from_the_guard():
    """Acceptance 2: a map sentence overlapping a body sentence only through
    section-title words is not deleted (without the exemption its Jaccard is
    exactly 0.6)."""
    intro = (
        "Motivation one stands alone. "
        "Latent World Models follow. "
        "Evidence still stops early. "
        "Methods and metrics belong to later sections."
    )
    body = ["Latent world models learn."]
    sections = [{"section_title": "Latent World Models"}]

    out = write_survey._dedupe_intro_against_body(intro, body, sections)

    assert "Latent World Models follow." in out


class _ScriptedLLM:
    """chat stub returning one scripted reply per call, in order."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, **kwargs):
        return self.replies.pop(0)


def test_llm_intro_body_paraphrase_is_cut_after_body_render():
    """The intro renders before the body, so the guard must run once the body
    exists: an LLM intro sentence paraphrasing a body claim disappears from
    the rendered Introduction while fresh motivation and the roadmap stay."""
    sections = [
        {
            "section_id": "sec_01",
            "section_title": "Latent World Models",
            "section_goal": "latent dynamics for planning",
            "allowed_paper_ids": ["p_alpha", "p_beta"],
            "selected_papers": [
                {
                    "paper_id": "p_alpha",
                    "title": "Dreamer",
                    "problem": "planning",
                    "method": "latent dynamics",
                    "contribution": "policy learning",
                    "limitations": "short horizon",
                    "evidence_snippets": ["alpha evidence sentence"],
                },
                {
                    "paper_id": "p_beta",
                    "title": "Simulacra",
                    "problem": "control",
                    "method": "world model rollout",
                    "contribution": "benchmark breadth",
                    "limitations": "narrow domain",
                    "evidence_snippets": ["beta evidence sentence"],
                },
            ],
            "artifact_slots": [],
            "claims": [],
        }
    ]
    cards = [
        {"paper_id": "p_alpha", "title": "Dreamer", "year": 2019, "topic_relevance_score": 0.9,
         "method": "latent dynamics", "contribution": "policy learning", "limitations": "short horizon"},
        {"paper_id": "p_beta", "title": "Simulacra", "year": 2020, "topic_relevance_score": 0.8,
         "method": "world model rollout", "contribution": "benchmark breadth", "limitations": "narrow domain"},
    ]
    evidence_by_paper = {
        # claim-bound fulltext rows: papers tier fulltext_ok so the structured
        # body's limitation-scope claim passes the T8 scope gate
        "p_alpha": [{"evidence_id": "e1", "paper_id": "p_alpha", "source_type": "paragraph",
                     "text": "alpha evidence sentence",
                     "supports_claims": [{"claim_text": "c", "support_type": "direct"}]}],
        "p_beta": [{"evidence_id": "e2", "paper_id": "p_beta", "source_type": "paragraph",
                    "text": "beta evidence sentence",
                    "supports_claims": [{"claim_text": "c", "support_type": "direct"}]}],
    }
    for pid, rows in evidence_by_paper.items():
        rows[0]["text"] = "We learn latent dynamics for planning. We evaluate only short horizons."
        rows[0]["supports_claims"] = [
            dict(paper_id=pid, claim_text="Dreamer learns latent dynamics for planning in imagined rollouts",
                 source_role="own_result", dimension="key_results", source_quote="We learn latent dynamics for planning.",
                 evidence_ids=[rows[0]["evidence_id"]], support_type="direct"),
            dict(paper_id=pid, claim_text="The reported evidence stops at short horizons",
                 source_role="own_limitation", dimension="limitations", source_quote="We evaluate only short horizons.",
                 evidence_ids=[rows[0]["evidence_id"]], support_type="direct")]
    sections[0]["selected_papers"] = [write_survey._selected_paper_entry(card, evidence_by_paper) for card in cards]
    llm = _ScriptedLLM(
        [
            # abstract (aliases P1 = p_alpha)
            "This survey covers one theme over two studies [P1]. Every claim is bound to recorded evidence snippets.",
            # introduction: sentence 2 paraphrases the body claim below
            "Game intelligence needs models that learn how worlds evolve. "
            "Imagined rollouts show that Dreamer learns latent dynamics for planning. "
            "Whether learned simulators can support fair evaluation remains open.",
            # body section as structured claims
            json.dumps(
                [
                    {"text": "Dreamer learns latent dynamics for planning in imagined rollouts", "cite": ["P1"], "sources": ["P1S1"], "scope": "contribution"},
                    {"text": "The reported evidence stops at short horizons", "cite": ["P1"], "sources": ["P1S2"], "scope": "limitation"},
                ]
            ),
            # open challenges
            "Reported evidence stops at short horizons and narrow evaluation [P1].",
            # future directions
            "Follow-up work should extend the reported method to held-out environments [P1].",
        ]
    )

    survey = write_survey._render_survey(
        "World Models",
        "en",
        sections,
        [],
        cards,
        evidence_by_paper,
        llm_chat=llm.chat,
        structured_out=[],
    )

    intro = _section_body(survey, "Introduction")
    assert "Imagined rollouts show that" not in intro  # paraphrase cut
    assert "Game intelligence needs models" in intro
    assert "remains open" in intro
    assert "Latent World Models surveys" in intro  # deterministic roadmap survived
    body = _section_body(survey, "Latent World Models")
    assert "Dreamer learns latent dynamics for planning in imagined rollouts [p_alpha]." in body


def test_llm_exception_falls_back_to_template_and_still_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _minimal_inputs(tmp_path)
    request_path = _survey_request(tmp_path, cache, output, "en")

    class ExplodingLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            raise RuntimeError("intern down")

    exploding = ExplodingLLM()
    monkeypatch.setenv("EVISURVEY_WRITER_LLM", "1")
    monkeypatch.setattr(write_survey, "_writer_llm_chat", lambda cfg: exploding.chat)

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    assert exploding.calls  # the writer really tried
    survey = (output / "survey.md").read_text(encoding="utf-8")
    assert "reports method findings" in survey  # safe proposition survives without invented framing
    cited = _citations(survey)
    allowed = set(json.loads((cache / "citation_ready_set.json").read_text(encoding="utf-8"))["allowed_paper_ids"])
    assert cited <= allowed


def test_template_fallback_readable_no_fragment_symptoms(tmp_path, monkeypatch):
    """Direction-A regression: GLM-dropout template output must stay readable.

    Pins the four manually-reviewed symptoms: mid-word ellipsis stubs,
    placeholder limitation filler, doubled framing ("focuses on Focuses on"),
    and snippet-spliced abstract/meta-talk.
    """
    monkeypatch.delenv("EVISURVEY_WRITER_LLM", raising=False)
    monkeypatch.delenv("GENERATE_KEY", raising=False)
    _isolated_root(tmp_path, monkeypatch)
    cache, output = _minimal_inputs(tmp_path)
    cards_path = cache / "paper_cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    long_field = (
        "Diffusion models have emerged as a powerful new family of deep generative models. "
        "In this survey, we review the design space in detail"
    )
    for card in cards["paper_cards"]:
        card["limitations"] = ""
        card["method"] = long_field
        card["problem"] = long_field
    cards_path.write_text(json.dumps(cards), encoding="utf-8")

    request_path = _survey_request(tmp_path, cache, output, "en")
    result = write_survey.run(str(request_path))
    assert result["status"] == "success"
    md = (output / "survey.md").read_text(encoding="utf-8")

    assert not re.search(r"[A-Za-z]\.\.\.", md), "mid-word ellipsis stub leaked"
    assert "available evidence boundary" not in md
    folded = md.casefold()
    assert "focuses on focuses on" not in folded
    assert md.count("## Introduction") == 1
    assert "citation-ready studies" not in md and "carries the tag" not in md
    assert "In this survey, we review" not in md, "second sentence of a field leaked past first-sentence rule"


def test_writer_fragment_hygiene_helpers():
    from tools.write_survey import _first_sentence, _shorten, _strip_model_headings, _trim_partial_sentence

    assert _first_sentence("Diffusion models have emerged. In this survey, we") == "Diffusion models have emerged."
    assert _first_sentence("") == ""
    # Word boundary: cuts after a complete word, never mid-word ("pre...").
    assert _shorten("We present GameNGen running at high quality", 18) == "We present..."
    assert _shorten("We present GameNGen running", 12) == "We..."
    assert _shorten("We present GameNGen", 20) == "We present GameNGen"
    assert _strip_model_headings("## Introduction\n\nReal text.") == "Real text."
    assert _strip_model_headings("Plain paragraph.") == "Plain paragraph."
    assert _trim_partial_sentence("Toward real") == ""
    assert _trim_partial_sentence("First sentence. Second cut") == "First sentence."
    assert _trim_partial_sentence("Complete already.") == "Complete already."


def test_selected_paper_entry_excludes_card_claims_unlisted_in_supports_claims():
    """Wave8 ③a retraction: the claim mapper re-checks the supports_claims
    correspondence verbatim, so a card claim that failed P5.2's NLI gate must
    NOT re-enter through entry assembly (first wave8 run: 9 invalid bindings,
    grounding 0.574, coverage_fail)."""
    card = {"paper_id": "paper:muzero", "title": "MuZero", "topic_relevance_score": 1.0,
            "possible_claims": {"limitations": [{"dimension": "limitations", "source_role": "own_limitation",
                                                 "text": "MuZero does not directly address imperfect information games",
                                                 "source_quote": "we do not directly address imperfect information games",
                                                 "evidence_ids": ["paper:muzero_para_px_43"]}]}}
    ev_item = {"evidence_id": "paper:muzero_para_px_43", "paper_id": "paper:muzero", "source_type": "paragraph",
               "source_doc_id": "", "source_title": "MuZero", "source_chunk_id": "", "supports_claims": []}
    entry = write_survey._selected_paper_entry(card, {"paper:muzero": [ev_item]})
    assert entry["sources"] == []
    assert entry["limitations"] == ""


def test_limitation_pool_skips_bits_already_rendered_in_body():
    def entry(pid, text):
        return {"paper_id": pid, "title": pid, "sources": [
            {"claim_text": text, "source_role": "own_limitation", "source_quote": "distinct quote " + pid}]}

    rendered = "Distillation enables 50 FPS generation but costs simulation quality [paper:1]."
    seen = {write_survey._norm_sentence(rendered)}
    e1 = entry("paper:1", "Distillation enables 50 FPS generation but costs simulation quality")
    e2 = entry("paper:2", "Planning gains were less marked in Atari than in Go")
    pool = write_survey._limitation_pool([e1, e2], seen=seen)
    assert [bit["paper_id"] for bit in pool] == ["paper:2"]
