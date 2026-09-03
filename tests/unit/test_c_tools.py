import json
import re
from pathlib import Path

from harness.agents.relevance import EmbeddingScorer, keyword_overlap
from scripts.build_final_seed_papers import build_final_seed_data, write_final_seed_files
from tools import render_report, revise_survey, write_survey
from tools.clients.llm_fake import FakeLLMClient


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
    assert "publication_timeline" in survey
    assert "representative_systems" in survey
    bank = json.loads((cache / "generated_artifact_bank.json").read_text(encoding="utf-8"))
    assert len(bank["artifacts"]) >= 8
    assert all(a["source_artifacts"] and a["supporting_papers"] for a in bank["artifacts"])
    assert all("placement_section" in a and "caption" in a and "display_number" in a for a in bank["artifacts"])
    assert all(a.get("alt_text") for a in bank["artifacts"] if a.get("artifact_kind") == "figure")
    assert all(a.get("table_headers") for a in bank["artifacts"] if a.get("artifact_kind") == "table")
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
    root = Path.cwd()
    write_final_seed_files(root)
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
    assert write_result["status"] == "success"

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
    assert readiness["pass"]
    review = json.loads((output / "review_report.json").read_text(encoding="utf-8"))
    assert review["final_decision"] == "submission_ready"


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
    cache, output = _scaled_inputs(tmp_path, papers_per_category=5)
    request_path = _survey_request(tmp_path, cache, output, "en")

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    plan = json.loads(Path("cache/section_claim_plan.json").read_text(encoding="utf-8"))
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
    cache, output = _scaled_inputs(tmp_path, papers_per_category=3, native_categories=2)
    request_path = _survey_request(tmp_path, cache, output, "en")

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    plan = json.loads(Path("cache/section_claim_plan.json").read_text(encoding="utf-8"))
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
            assert citations, f"{language}: {title} has no citation"


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
    cache, output = _llm_inputs(tmp_path)
    request_path = _survey_request(tmp_path, cache, output, "en")
    whitelists = _section_whitelists(cache)
    assert all(len(ids) >= 3 for ids in whitelists.values())

    replies = []
    for title, allowed in whitelists.items():
        ids = sorted(allowed)
        replies.append(
            (
                title,
                "\n".join(
                    [
                        f"[SUMMARY] {ids[0]} studies the section goal [{ids[0]}].",
                        f"This sentence leaks the harness chain, CitationReadySet and invents [{ids[-1]}] plus [offtopic].",
                        f"[COMPARISON] {ids[0]} differs from {ids[1]} in method and role [{ids[0]}] [{ids[1]}].",
                        f"[LIMITATION] The reported evidence does not establish long-horizon robustness [{ids[2]}].",
                    ]
                ),
            )
        )
    fake = FakeLLMClient(responses=replies)
    monkeypatch.setenv("EVISURVEY_WRITER_LLM", "1")
    monkeypatch.setattr(write_survey, "_writer_llm_chat", lambda cfg: fake.chat)

    result = write_survey.run(str(request_path))

    assert result["status"] == "success"
    assert fake.calls == len(whitelists), "each themed section must be drafted by the LLM"
    survey = (output / "survey.md").read_text(encoding="utf-8")
    assert "harness chain" not in survey
    assert "[offtopic]" not in survey
    for title, allowed in whitelists.items():
        body = _section_body(survey, title)
        citations = _citations(body)
        assert citations <= allowed, f"{title}: citations outside the section whitelist"
        assert "differs from" in body  # LLM comparison paragraph landed
        assert "Its method can be summarized as" not in body  # template summary replaced


def test_llm_exception_falls_back_to_template_and_still_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("EVISURVEY_WRITER_EMBED", "0")
    monkeypatch.delenv("GENERATE_KEY", raising=False)
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
    assert "Its method can be summarized as" in survey  # template body survived
    cited = _citations(survey)
    allowed = set(json.loads((cache / "citation_ready_set.json").read_text(encoding="utf-8"))["allowed_paper_ids"])
    assert cited <= allowed
