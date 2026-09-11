from tools.models.common import paper_id_from_seed, evidence_id, figure_id, table_id, category_id


def test_seed_id_stable():
    assert paper_id_from_seed("DreamerV3", 2023) == paper_id_from_seed("DreamerV3", 2023)
    assert paper_id_from_seed("DreamerV3", 2023).startswith("seed:")
    assert paper_id_from_seed("A", 2020) != paper_id_from_seed("B", 2020)


def test_evidence_id_format():
    # source_type namespaces the id: same (page, idx) never collides across types
    assert evidence_id("paper:x", 3, 0) == "paper:x_para_p3_0"
    assert evidence_id("paper:x", 3, 0, "abstract") == "paper:x_abs_p3_0"
    assert evidence_id("paper:x", 3, 0, "caption") == "paper:x_cap_p3_0"
    assert evidence_id("paper:x", 3, 0, "agentic_chunk") == "paper:x_agentic_p3_0"
    assert evidence_id("paper:x", 3, 0, "repair") == "paper:x_repair_p3_0"
    # page=None (markdown-only parse channel) gets its own token
    assert evidence_id("paper:x", None, 7) == "paper:x_para_px_7"


def test_figure_table_category():
    assert figure_id("paper:x", 1) == "paper:x_fig1"
    assert table_id("paper:x", 2) == "paper:x_tbl2"
    assert category_id(2) == "cat_002"
