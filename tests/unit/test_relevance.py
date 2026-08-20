"""Unit tests for harness/agents/relevance.py (report card + scorer).

Spec §4.1 test_relevance points: n_unique dedup, pairwise Jaccard, lazy
loading + keyword degradation (logged), cosine computation. The embedding
model itself is never loaded here — tests inject vectors via ``_embed``.
"""

import logging

from harness.agents.relevance import (
    EmbeddingScorer,
    build_report_card,
    global_score,
    global_score_rows,
    hit_year,
    keyword_overlap,
    paper_identity,
    paper_text,
    render_card,
)


def _aspect(aid, name, keywords):
    return {"aspect_id": aid, "aspect_name": name,
            "description": f"about {name}", "keywords": keywords}


def _hit(uid, title, year=2023, abstract="", keywords=None):
    return {"unique_id": uid, "title": title, "year": year,
            "abstract": abstract, "keywords": keywords or []}


def _keyword_scorer():
    # no _embed and load disabled -> deterministic keyword fallback column
    scorer = EmbeddingScorer()
    scorer._failed = True
    return scorer


# --- scorer -----------------------------------------------------------------


def test_scorer_embedding_mode_without_model_load():
    scorer = EmbeddingScorer(_embed=lambda texts: [[1.0] if i == 0 else [0.0] for i in range(len(texts))])
    assert scorer.mode == "embedding"
    assert scorer._model is None  # lazy: no load attempted


def test_scorer_cosine_on_injected_vectors():
    # 2D normalized vectors: aspect [1,0], papers alternate [1,0] and [0,1]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    scorer = EmbeddingScorer(_embed=lambda texts: [vecs[i % 2] for i in range(len(texts))])
    # pairs order: [a1, p1, p2] -> texts [a1, p1, p2] indexed 0,1,2 -> [1,0],[0,1],[1,0]
    scores = scorer.score_pairs([("a", "p1"), ("a2", "p2")])
    # aspect_texts = [a, a2] -> [1,0], [0,1]; paper_texts = [p1, p2] -> [1,0], [0,1]
    # dot(a1,p1)=1, dot(a2,p2)=1 ... wait texts order is aspects+papers: [a,a2,p1,p2]
    # indexes: a=[1,0], a2=[0,1], p1=[1,0], p2=[0,1] -> cos(a,p1)=1, cos(a2,p2)=1
    assert scores == [1.0, 1.0]


def test_scorer_runtime_failure_degrades_to_keyword(caplog):
    def broken(texts):
        raise RuntimeError("encode boom")

    scorer = EmbeddingScorer(_embed=broken)
    with caplog.at_level(logging.WARNING):
        scores = scorer.score_pairs([("world model rl", "world model paper")])
    assert scores == [keyword_overlap("world model rl", "world model paper")]
    assert any("keyword fallback" in r.message for r in caplog.records)


def test_scorer_load_failure_cached_and_logged():
    # _failed=True simulates a previous load attempt that already failed:
    # every later call stays on the keyword column without retrying the load
    scorer = EmbeddingScorer(model_name="definitely/not-a-model")
    scorer._failed = True
    assert scorer.mode == "keyword"
    assert scorer.score_pairs([("a", "b")]) == [keyword_overlap("a", "b")]


def test_keyword_overlap_bounded():
    assert keyword_overlap("world model", "world model survey") == 1.0
    assert keyword_overlap("aaaa bbbb", "cccc dddd") == 0.0
    assert 0.0 <= keyword_overlap("world model rl", "latent dynamics video") <= 1.0


# --- identity / texts -------------------------------------------------------


def test_paper_identity_prefers_uid_then_doi_then_title():
    assert paper_identity({"unique_id": "x1", "doi": "10.1/z", "title": "T"}) == "uid:x1"
    assert paper_identity({"doi": "10.1/Z", "title": "T"}) == "doi:10.1/z"
    assert paper_identity({"title": "A  Paper!", "year": 2020}).startswith("title:a paper")


def test_hit_year_reads_native_and_fallback_fields():
    assert hit_year({"publication_published_year": 2019}) == 2019
    assert hit_year({"year": 2021}) == 2021
    assert hit_year({"year": "2022"}) == 2022
    assert hit_year({}) is None


def test_paper_text_takes_title_plus_two_sentences():
    text = paper_text({
        "title": "Dreamer",
        "abstract": "First sentence. Second sentence. Third sentence that must drop.",
    })
    assert text.startswith("Dreamer")
    assert "Third sentence" not in text


# --- report card ------------------------------------------------------------


def test_card_n_unique_first_claim_wins():
    aspects = [_aspect("a1", "Alpha", ["k1"]), _aspect("a2", "Beta", ["k2"])]
    hits = [
        [_hit("u1", "Shared"), _hit("u2", "Only-A")],
        [_hit("u1", "Shared"), _hit("u3", "Only-B")],
    ]
    card = build_report_card(aspects, hits, _keyword_scorer())
    rows = card["aspects"]
    assert rows[0]["n_hits"] == 2 and rows[0]["n_unique"] == 2
    assert rows[1]["n_hits"] == 2 and rows[1]["n_unique"] == 1  # u1 claimed by row 0


def test_card_overlap_jaccard():
    aspects = [_aspect("a1", "Alpha", ["k1"]), _aspect("a2", "Beta", ["k2"])]
    # A: {1,2,3}, B: {2,3,4} -> jaccard = 2/4 = 0.5 both ways
    hits = [
        [_hit(f"u{i}", f"T{i}") for i in (1, 2, 3)],
        [_hit(f"u{i}", f"T{i}") for i in (2, 3, 4)],
    ]
    card = build_report_card(aspects, hits, _keyword_scorer())
    assert card["aspects"][0]["overlap"] == 0.5
    assert card["aspects"][1]["overlap"] == 0.5


def test_card_top5_titles_and_year_span():
    aspects = [_aspect("a1", "Alpha", ["k1"])]
    hits = [_hit(f"u{i}", f"Title {i}", year=2000 + i) for i in range(1, 8)]
    card = build_report_card(aspects, [hits], _keyword_scorer())
    row = card["aspects"][0]
    assert row["top5_titles"] == [f"Title {i}" for i in range(1, 6)]
    assert row["year_span"] == [2001, 2007]


def test_card_rel_via_embedding_scorer():
    # aspect vector [1,0], every paper vector [0.5, sqrt(3)/2] -> cosine 0.5
    # score_pairs encodes aspect_texts first, then paper_texts
    import math
    paper_vec = [0.5, math.sqrt(3) / 2]

    def embed(texts):
        half = len(texts) // 2
        return [[1.0, 0.0]] * half + [paper_vec] * (len(texts) - half)

    scorer = EmbeddingScorer(_embed=embed)
    aspects = [_aspect("a1", "Alpha", ["k1"])]
    card = build_report_card(aspects, [[_hit("u1", "T1"), _hit("u2", "T2")]], scorer)
    assert card["rel_mode"] == "embedding"
    assert card["aspects"][0]["rel"] == 0.5


def test_global_score_formula():
    rows = [
        {"n_unique": 10, "rel": 0.8},   # 1.0 * 0.8
        {"n_unique": 3, "rel": 0.5},    # 0.3 * 0.5
    ]
    # mean of (0.8, 0.15) = 0.475
    assert abs(global_score_rows(rows) - 0.475) < 1e-9
    assert global_score({"aspects": rows, "global_score": 0.475}) == 0.475


def test_render_card_mentions_budget_and_rows():
    aspects = [_aspect("a1", "Alpha", ["k1"])]
    card = build_report_card(aspects, [[_hit("u1", "Some Title")]], _keyword_scorer())
    text = render_card(card, rounds_left=2, sciverse_left=10)
    assert "n_unique=1" in text
    assert "Some Title" in text
    assert "rounds_left=2" in text and "sciverse_calls_left=10" in text
