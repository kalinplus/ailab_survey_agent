from tools.phases.phase6_bundle_assembler import run
from tools.models.artifacts import *
from tools.models.requests import QualityRequirements


def _empty_artifacts():
    return (
        RetrievedPapers(task_id="t", papers=[]),
        ParsedPapers(task_id="t", papers=[]),
        PaperCards(task_id="t", paper_cards=[]),
        EvidenceStore(task_id="t", evidence=[]),
        FigureBank(task_id="t", figures=[]),
        TableBank(task_id="t", tables=[]),
        Taxonomy(task_id="t", topic="wm", categories=[]),
        CitationIndex(task_id="t", citations=[]),
    )


def test_failed_status_when_no_papers():
    rp, pp, pc, es, fb, tb, tax, ci = _empty_artifacts()
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements())
    assert b.status == "failed"
    assert b.summary["paper_count"] == 0
    assert not b.quality_report["search_success"]


def test_success_status_with_enough_papers():
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}", title="T") for i in range(12)])
    pc = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id=f"p{i}", title="T", card_type="deep") for i in range(6)])
    pp, es, fb, tb, tax, ci = ParsedPapers(task_id="t", papers=[]), EvidenceStore(task_id="t", evidence=[]), FigureBank(task_id="t", figures=[]), TableBank(task_id="t", tables=[]), Taxonomy(task_id="t", topic="wm", categories=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements(min_total_papers=10, min_core_papers=5))
    assert b.status == "success"
    assert b.quality_report["warnings"] == []
    assert b.summary["paper_count"] == 12
    assert b.summary["core_paper_count"] == 6


def test_partial_success_with_warnings():
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}", title="T") for i in range(12)])
    pc = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id=f"p{i}", title="T", card_type="deep") for i in range(6)])
    pp, es, fb, tb, tax, ci = ParsedPapers(task_id="t", papers=[]), EvidenceStore(task_id="t", evidence=[]), FigureBank(task_id="t", figures=[]), TableBank(task_id="t", tables=[]), Taxonomy(task_id="t", topic="wm", categories=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, ["struct_err"], ["cov_warn"], QualityRequirements(min_total_papers=10, min_core_papers=5))
    assert b.status == "partial_success"
    assert b.quality_report["warnings"] == ["struct_err", "cov_warn"]


def test_partial_success_below_threshold():
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id="p0", title="T")])
    pc = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id="p0", title="T", card_type="shallow")])
    pp, es, fb, tb, tax, ci = ParsedPapers(task_id="t", papers=[]), EvidenceStore(task_id="t", evidence=[]), FigureBank(task_id="t", figures=[]), TableBank(task_id="t", tables=[]), Taxonomy(task_id="t", topic="wm", categories=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements(min_total_papers=10, min_core_papers=5))
    assert b.status == "partial_success"
    assert b.quality_report["search_success"] is True


def test_summary_counts():
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id="p0", title="T")])
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="p0", title="T")])
    pc = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id="p0", title="T", card_type="deep", evidence_ids=["e1"])])
    es = EvidenceStore(task_id="t", evidence=[Evidence(evidence_id="e1", paper_id="p0", text="...")])
    fb = FigureBank(task_id="t", figures=[Figure(figure_id="f1", paper_id="p0")])
    tb = TableBank(task_id="t", tables=[Table(table_id="t1", paper_id="p0")])
    tax = Taxonomy(task_id="t", topic="wm", categories=[Category(category_id="c1", category_name="C")])
    ci = CitationIndex(task_id="t", citations=[{"citation_id": "ci1"}])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements())
    assert b.summary == {
        "paper_count": 1,
        "core_paper_count": 1,
        "parsed_paper_count": 1,
        "figure_count": 1,
        "table_count": 1,
        "evidence_count": 1,
        "taxonomy_category_count": 1,
        "citation_count": 1,
    }


def test_quality_report_rates():
    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}", title="T") for i in range(4)])
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="p0", title="T")])
    pc = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="p0", title="T", card_type="deep", evidence_ids=["e1", "e2"]),
        PaperCard(paper_id="p1", title="T", card_type="deep", evidence_ids=["e3"]),
        PaperCard(paper_id="p2", title="T", card_type="shallow"),
    ])
    es = EvidenceStore(task_id="t", evidence=[])
    fb = FigureBank(task_id="t", figures=[Figure(figure_id="f1", paper_id="p0")])
    tb, tax, ci = TableBank(task_id="t", tables=[]), Taxonomy(task_id="t", topic="wm", categories=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements())
    assert b.quality_report["parse_success_rate"] == 0.25
    assert b.quality_report["paper_card_success_rate"] == 0.75
    assert b.quality_report["evidence_coverage_rate"] == round(3 / 2, 3)  # 3 evidence_ids / 2 core cards
    assert b.quality_report["has_multimodal_evidence"] is True


def test_quality_report_no_figures():
    rp, pp, pc, es = _empty_artifacts()[:4]
    fb = FigureBank(task_id="t", figures=[])
    tb, tax, ci = TableBank(task_id="t", tables=[]), Taxonomy(task_id="t", topic="wm", categories=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements())
    assert b.quality_report["has_multimodal_evidence"] is False


def test_artifacts_paths_stored_verbatim():
    rp, pp, pc, es, fb, tb, tax, ci = _empty_artifacts()
    paths = {"retrieved": "/tmp/r.json", "parsed": "/tmp/p.json"}
    b = run("t", "wm", paths, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements())
    assert b.artifacts == paths


def test_degraded_aspect_marked_when_category_below_floor():
    from tools.models.artifacts import Category

    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}", title="T") for i in range(12)])
    pc = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id=f"p{i}", title="T", card_type="deep") for i in range(6)])
    tax = Taxonomy(task_id="t", topic="wm", categories=[
        Category(category_id="c1", category_name="Latent World Models", paper_ids=["p0", "p1"], paper_count=2),
        Category(category_id="c2", category_name="Benchmarks", paper_ids=["p2"], paper_count=1),
    ])
    pp, es, fb, tb, ci = ParsedPapers(task_id="t", papers=[]), EvidenceStore(task_id="t", evidence=[]), FigureBank(task_id="t", figures=[]), TableBank(task_id="t", tables=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements(min_total_papers=10, min_core_papers=5))
    assert b.quality_report["degraded_aspects"] == ["Benchmarks"]
    assert b.coverage_report["papers_per_aspect"] == {"Latent World Models": 2, "Benchmarks": 1}
    assert b.coverage_report["undercovered_aspects"] == ["Benchmarks"]


def test_no_degraded_aspect_when_all_categories_meet_floor():
    from tools.models.artifacts import Category

    rp = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"p{i}", title="T") for i in range(12)])
    pc = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id=f"p{i}", title="T", card_type="deep") for i in range(6)])
    tax = Taxonomy(task_id="t", topic="wm", categories=[
        Category(category_id="c1", category_name="Latent World Models", paper_ids=["p0", "p1"], paper_count=2),
        Category(category_id="c2", category_name="Benchmarks", paper_ids=["p2", "p3"], paper_count=2),
    ])
    pp, es, fb, tb, ci = ParsedPapers(task_id="t", papers=[]), EvidenceStore(task_id="t", evidence=[]), FigureBank(task_id="t", figures=[]), TableBank(task_id="t", tables=[]), CitationIndex(task_id="t", citations=[])
    b = run("t", "wm", {}, rp, pp, pc, es, fb, tb, tax, ci, [], [], QualityRequirements(min_total_papers=10, min_core_papers=5))
    assert b.quality_report["degraded_aspects"] == []
    assert b.coverage_report["undercovered_aspects"] == []
