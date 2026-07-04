from tools.verify.structural import run, extract_citations, extract_figure_refs
from tools.models.artifacts import CitationIndex, FigureBank, Figure, TableBank


def _ci(*ids):
    return CitationIndex(task_id="t", citations=[{"paper_id": i} for i in ids])


def _fb(*ids):
    return FigureBank(task_id="t", figures=[Figure(figure_id=i, paper_id="dummy") for i in ids])


TB = TableBank(task_id="t", tables=[])


# --- extract_citations ---

def test_extract_skips_urls():
    assert extract_citations("see [https://x.com]") == []


def test_extract_basic():
    md = "Text [paper:a] and [paper:b]."
    assert extract_citations(md) == ["paper:a", "paper:b"]


def test_extract_empty():
    assert extract_citations("no citations here") == []


# --- extract_figure_refs ---

def test_extract_figure_refs():
    md = "![arch](paper:1_fig1) and ![result](paper:2_fig3)"
    assert extract_figure_refs(md) == ["paper:1_fig1", "paper:2_fig3"]


def test_extract_figure_refs_empty():
    assert extract_figure_refs("no figures") == []


# --- run: paper citation validation ---

def test_flags_fabricated_id():
    ci = _ci("paper:real")
    md = "Some claim [paper:real] and a fake [paper:fake]."
    res = run("t", md, ci, _fb(), TB)
    ids = {e.citation_id: e.valid for e in res.entries}
    assert ids["paper:real"] is True
    assert ids["paper:fake"] is False
    assert res.citation_validity_score == 0.5


# --- run: figure validation ---

def test_figure_ref_valid_and_invalid():
    md = "![arch](paper:1_fig1) and ![missing](paper:1_fig99)"
    res = run("t", md, _ci(), _fb("paper:1_fig1"), TB)
    ids = {e.citation_id: e.valid for e in res.entries}
    assert ids["paper:1_fig1"] is True
    assert ids["paper:1_fig99"] is False
    assert res.total_citations == 2
    assert res.valid_citations == 1
    assert res.citation_validity_score == 0.5


# --- run: combined paper + figure ---

def test_combined_paper_and_figure():
    md = "Claim [paper:a]. See ![fig](paper:1_fig1)."
    res = run("t", md, _ci("paper:a"), _fb("paper:1_fig1"), TB)
    assert len(res.entries) == 2
    ids = {e.citation_id: e.valid for e in res.entries}
    assert ids["paper:a"] is True
    assert ids["paper:1_fig1"] is True
    assert res.citation_validity_score == 1.0


def test_combined_mixed_validity():
    md = "[paper:real] and [fake] with ![good](paper:1_fig1) ![bad](paper:x)"
    res = run("t", md, _ci("paper:real"), _fb("paper:1_fig1"), TB)
    assert res.total_citations == 4
    assert res.valid_citations == 2
    assert res.invalid_citations == 2
    assert res.citation_validity_score == 0.5


# --- run: empty markdown ---

def test_empty_md():
    res = run("t", "", _ci(), _fb(), TB)
    assert res.total_citations == 0
    assert res.citation_validity_score == 1.0
    assert res.entries == []


# --- entries always have empty figures_valid / tables_valid ---

def test_entries_empty_figures_and_tables():
    md = "[paper:a] ![fig](paper:1_fig1)"
    res = run("t", md, _ci("paper:a"), _fb("paper:1_fig1"), TB)
    for e in res.entries:
        assert e.figures_valid == []
        assert e.tables_valid == []
