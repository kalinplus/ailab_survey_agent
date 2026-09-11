from tools.phases.phase5_synthesis_rest import (
    build_figure_bank, build_table_bank, build_taxonomy, build_citation_index, run,
)
from tools.models.artifacts import ParsedPapers, ParsedPaper, PaperCards, PaperCard
from tools.models.artifacts import FigureBank, TableBank, Taxonomy, CitationIndex


def test_figure_and_table_ids():
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        figures=[{"num": 1, "caption": "arch", "page": 3,
                  "img_path": "cache/mineru_assets/abc123/fig1.jpg"}],
        tables=[{"num": 2, "caption": "res", "page": 5}])])
    fb = build_figure_bank("t", pp)
    tb = build_table_bank("t", pp)
    assert fb.figures[0].figure_id == "paper:1_fig1"
    assert fb.figures[0].paper_id == "paper:1"
    assert fb.figures[0].caption == "arch"
    assert fb.figures[0].page == 3
    assert fb.figures[0].image_path == "cache/mineru_assets/abc123/fig1.jpg"
    assert tb.tables[0].table_id == "paper:1_tbl2"
    assert tb.tables[0].paper_id == "paper:1"
    assert tb.tables[0].caption == "res"
    assert tb.tables[0].page == 5


def test_figure_bank_img_path_defaults_to_none():
    # legacy/dict shapes without img_path (mock_parse, seed papers) stay valid
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="p1", title="A",
        figures=[{"num": 1, "caption": "f"}])])
    fb = build_figure_bank("t", pp)
    assert fb.figures[0].image_path is None


def test_empty_parsed_papers_gives_empty_banks():
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="No figs", figures=[], tables=[])])
    fb = build_figure_bank("t", pp)
    tb = build_table_bank("t", pp)
    assert fb.figures == []
    assert tb.tables == []


def test_multiple_papers_figure_table():
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="p1", title="A", figures=[{"num": 1, "caption": "f1"}],
                    tables=[{"num": 1, "caption": "t1"}]),
        ParsedPaper(paper_id="p2", title="B", figures=[{"num": 2, "caption": "f2"}],
                    tables=[]),
    ])
    fb = build_figure_bank("t", pp)
    tb = build_table_bank("t", pp)
    assert len(fb.figures) == 2
    assert fb.figures[0].figure_id == "p1_fig1"
    assert fb.figures[1].figure_id == "p2_fig2"
    assert len(tb.tables) == 1
    assert tb.tables[0].table_id == "p1_tbl1"


def test_taxonomy_assigns_cards():
    refined = [{"name": "Internal World Model", "description": "d"}]
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="Dreamer world model", method="latent")])
    tax = build_taxonomy("t", "wm", refined, cards, llm=None)
    assert tax.categories[0].paper_count == 1
    assert tax.categories[0].category_id.startswith("cat_")
    assert tax.categories[0].paper_ids == ["paper:1"]


def test_taxonomy_fallback_first_category():
    refined = [
        {"name": "External World Model", "description": "d1"},
        {"name": "Internal World Model", "description": "d2"},
    ]
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="Something completely unrelated", method="none")])
    tax = build_taxonomy("t", "wm", refined, cards, llm=None)
    # Wave7: no meaningful-token overlap -> unassigned (piling into the first
    # category was the 48/67 skew behind the single-bar "Uncategorized" figure)
    assert tax.categories[0].paper_count == 0
    assert all("paper:1" not in c.paper_ids for c in tax.categories)


def test_taxonomy_best_keyword_overlap():
    refined = [
        {"name": "External World Model", "description": "d1"},
        {"name": "Internal World Model", "description": "d2"},
    ]
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="Dreamer internal model", method="latent internal")])
    tax = build_taxonomy("t", "wm", refined, cards, llm=None)
    # "internal" matches both but "Internal World Model" has more keyword matches
    # "internal" matches category 2 name, and also "world"/"model" match both
    # category 1: "internal" -> 1 match
    # category 2: "internal" -> 1 match
    # Actually: split(" ") on "internal world model" -> ["internal", "world", "model"]
    # category 1 "external world model" -> ["external", "world", "model"]
    # category 2 "internal world model" -> ["internal", "world", "model"]
    # text = "dreamer internal model latent internal"
    # category 1: "external" not in text, "world" not in text, "model" in text -> 1
    # category 2: "internal" in text, "world" not in text, "model" in text -> 2
    assert tax.categories[1].paper_count == 1
    assert tax.categories[1].paper_ids == ["paper:1"]
    assert tax.categories[0].paper_count == 0


def test_taxonomy_category_ids():
    refined = [{"name": "Cat A"}, {"name": "Cat B"}, {"name": "Cat C"}]
    cards = PaperCards(task_id="t", paper_cards=[])
    tax = build_taxonomy("t", "topic", refined, cards, llm=None)
    assert tax.categories[0].category_id == "cat_001"
    assert tax.categories[1].category_id == "cat_002"
    assert tax.categories[2].category_id == "cat_003"


def test_taxonomy_refine_history():
    refined = [{"name": "Cat A"}]
    cards = PaperCards(task_id="t", paper_cards=[])
    tax = build_taxonomy("t", "topic", refined, cards, llm=None)
    # Brief spelling "prelimiminary" kept verbatim
    assert tax.refine_history == ["prelimiminary", "refined", "final"]


def test_citation_index_has_all_cards():
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A", bibtex_key="keyA", year=2023),
        PaperCard(paper_id="paper:2", title="B", bibtex_key="keyB", year=2024)])
    ci = build_citation_index("t", cards)
    assert {c["paper_id"] for c in ci.citations} == {"paper:1", "paper:2"}
    assert ci.citations[0]["title"] == "A"
    assert ci.citations[0]["year"] == 2023
    assert ci.citations[1]["bibtex_key"] == "keyB"


def test_citation_index_empty():
    cards = PaperCards(task_id="t", paper_cards=[])
    ci = build_citation_index("t", cards)
    assert ci.citations == []


def test_run_returns_4_tuple():
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="p1", title="A",
                     figures=[{"num": 1, "caption": "f"}],
                     tables=[{"num": 1, "caption": "t"}])])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="p1", title="A", method="method")])
    refined = [{"name": "Cat A", "description": "desc"}]
    fb, tb, tax, ci = run("t", "topic", pp, refined, cards, llm=None)
    assert isinstance(fb, FigureBank)
    assert isinstance(tb, TableBank)
    assert isinstance(tax, Taxonomy)
    assert isinstance(ci, CitationIndex)
    assert len(fb.figures) == 1
    assert len(tb.tables) == 1
    assert len(ci.citations) == 1


def test_run_no_file_io():
    import builtins
    original_open = builtins.open
    opened = []

    def tracking_open(*args, **kwargs):
        opened.append((args, kwargs))
        return original_open(*args, **kwargs)

    builtins.open = tracking_open
    try:
        pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="p1", title="A")])
        cards = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id="p1", title="A")])
        run("t", "topic", pp, [{"name": "C"}], cards, llm=None)
    finally:
        builtins.open = original_open
    # No file I/O: open was only called if we patched it above for tracking
    # The list should be empty (no file opens during run)
    assert len(opened) == 0


# --- wave7: taxonomy assignment under the claim contract ---

def _tax_cards():
    from tools.models.artifacts import Claim

    return PaperCards(task_id="t", paper_cards=[
        # claim text (not the empty method field) drives the match
        PaperCard(paper_id="paper:pcg", title="Wave Function Collapse Rooms",
                  possible_claims={"method": [Claim(
                      text="The generator uses procedural generation and scripting of room layouts",
                      dimension="method")]}),
        PaperCard(paper_id="paper:lat", title="Mastering Diverse Domains",
                  possible_claims={"method": [Claim(
                      text="The agent learns a latent world model for planning",
                      dimension="method")]}),
    ])


def test_build_taxonomy_matches_from_claim_text():
    refined = [
        {"name": "Procedural Generation and Scripting", "description": ""},
        {"name": "Latent Representations and World Models", "description": ""},
    ]
    tax = build_taxonomy("t", "topic", refined, _tax_cards(), None)
    by_name = {c.category_name: c for c in tax.categories}
    assert by_name["Procedural Generation and Scripting"].paper_ids == ["paper:pcg"]
    assert by_name["Latent Representations and World Models"].paper_ids == ["paper:lat"]


def test_build_taxonomy_no_meaningful_tokens_in_first_category():
    # "and"-style tokens are stoplisted: a paper matching only them is left
    # unassigned instead of piling into the first category (the 48/67 skew)
    refined = [{"name": "Methods and Systems", "description": ""},
               {"name": "Procedural Generation and Scripting", "description": ""}]
    tax = build_taxonomy("t", "topic", refined, _tax_cards(), None)
    by_name = {c.category_name: c for c in tax.categories}
    assert "paper:lat" not in by_name["Methods and Systems"].paper_ids
    assert "paper:pcg" in by_name["Procedural Generation and Scripting"].paper_ids


def test_writer_taxonomy_counts_read_category_paper_ids():
    from tools.write_survey import _taxonomy_counts, _final_table_specs

    categories = [{"category_id": "cat_1", "category_name": "Latent Representations and World Models",
                   "paper_ids": ["paper:1"]}]
    cards = [{"paper_id": "paper:1", "title": "A", "year": 2024, "method": "", "contribution": "", "limitations": ""},
             {"paper_id": "paper:2", "title": "B", "year": 2023, "method": "", "contribution": "", "limitations": ""}]
    counts = _taxonomy_counts(categories, cards)
    assert counts == {"Latent Representations and World Models": 1, "Uncategorized": 1}
    spec = next(s for s in _final_table_specs(cards, categories) if s["artifact_id"] == "representative_systems")
    rows = [line for line in spec["markdown"].splitlines() if line.startswith("|") and "paper" not in line and "---" not in line]
    cat_col_A = [r for r in rows if r.startswith("| A ")]
    assert cat_col_A and "Latent Representations" in cat_col_A[0]


def test_final_table_cells_keep_full_claim_sentences():
    from tools.write_survey import _final_table_specs

    categories = [{"category_name": "C", "paper_ids": ["paper:1"]}]
    long_claim = "The method " + "carefully " * 22 + "trains"  # 236 chars < 260 budget
    cards = [{"paper_id": "paper:1", "title": "A", "year": 2024,
              "method": long_claim, "contribution": "", "limitations": ""}]
    spec = next(s for s in _final_table_specs(cards, categories) if s["artifact_id"] == "representative_systems")
    method_cell = [line for line in spec["markdown"].splitlines()
                   if line.startswith("| A ")][0].split("|")[4]
    assert "trains" in method_cell and "..." not in method_cell
