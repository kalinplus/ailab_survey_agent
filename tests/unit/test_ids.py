from tools.models.common import paper_id_from_seed, evidence_id, figure_id, table_id, category_id


def test_seed_id_stable():
    assert paper_id_from_seed("DreamerV3", 2023) == paper_id_from_seed("DreamerV3", 2023)
    assert paper_id_from_seed("DreamerV3", 2023).startswith("seed:")
    assert paper_id_from_seed("A", 2020) != paper_id_from_seed("B", 2020)


def test_evidence_id_format():
    assert evidence_id("paper:x", 3, 0) == "paper:x_p3_0"


def test_figure_table_category():
    assert figure_id("paper:x", 1) == "paper:x_fig1"
    assert table_id("paper:x", 2) == "paper:x_tbl2"
    assert category_id(2) == "cat_002"
