from tools.phases.phase5_cards import parse_card_response, run
from tools.models.artifacts import ParsedPapers, ParsedPaper, RetrievedPapers, RetrievedPaper
from tools.models.artifacts import Paragraph
from tools.clients.llm_fake import FakeLLMClient

SAMPLE = """## KEY RESULTS
1. Achieves SOTA on 150 domains [page 3]
2. Single config works
## METHOD
1. Recurrent state space model [page 4]
## SETUP
## LIMITATIONS
1. Short horizon only"""


def test_parse_buckets_and_evidence():
    claims = parse_card_response(SAMPLE, "paper:1")
    # key_results: 2 claims
    assert len(claims["key_results"]) == 2
    assert claims["key_results"][0].text.startswith("Achieves SOTA")
    assert claims["key_results"][0].dimension == "key_results"
    assert claims["key_results"][0].evidence_ids == ["paper:1_p3_0"]
    # uncited claim has empty evidence_ids
    assert claims["key_results"][1].text == "Single config works"
    assert claims["key_results"][1].evidence_ids == []
    # method
    assert len(claims["method"]) == 1
    assert claims["method"][0].text.startswith("Recurrent state space")
    assert claims["method"][0].evidence_ids == ["paper:1_p4_0"]
    # setup is empty
    assert claims["setup"] == []
    # limitations
    assert len(claims["limitations"]) == 1
    assert claims["limitations"][0].text == "Short horizon only"
    assert claims["limitations"][0].evidence_ids == []


def test_parse_ignores_unknown_headers():
    text = """## KEY RESULTS
1. First result
## OTHER STUFF
some text
1. This should be ignored
## METHOD
1. Real method"""
    claims = parse_card_response(text, "p:1")
    # "OTHER STUFF" is unknown, so current remains "key_results" — its numbered line
    # is collected under key_results (the brief says "does not crash (ignored)",
    # meaning the header itself is ignored but the bucket does not change).
    assert len(claims["key_results"]) == 2
    assert claims["key_results"][0].text == "First result"
    assert claims["key_results"][1].text == "This should be ignored"
    assert claims["method"][0].text == "Real method"
    # "OTHER STUFF" does NOT create a bucket in claims
    assert "other_stuff" not in claims


def test_parse_ignores_outside_bucket():
    text = """1. Orphan claim
Junk line
## KEY RESULTS
1. Valid claim"""
    claims = parse_card_response(text, "p:2")
    assert len(claims["key_results"]) == 1
    assert claims["key_results"][0].text == "Valid claim"


def test_parse_accepts_bullet_list():
    """Intern-S2 returns '-'/'*' bullets, not '1.' numbers — both must parse.

    Regression for the empty-possible_claims bug: the old regex only accepted
    '\\d+\\.' so real-LLM bullet output parsed to zero claims -> evidence_count=0.
    """
    text = """## KEY RESULTS
- LLaMA-13B outperforms GPT-3 on most benchmarks [page 1]
* asterisk bullets also work
## METHOD
- We train on trillions of tokens"""
    claims = parse_card_response(text, "paper:X")
    assert len(claims["key_results"]) == 2
    assert claims["key_results"][0].text.startswith("LLaMA-13B")
    assert claims["key_results"][0].evidence_ids == ["paper:X_p1_0"]
    assert claims["key_results"][1].text == "asterisk bullets also work"
    assert len(claims["method"]) == 1
    assert claims["method"][0].text.startswith("We train")


def test_parse_non_numbered_ignored():
    text = """## KEY RESULTS
Not a numbered line
1. This is numbered"""
    claims = parse_card_response(text, "p:3")
    assert len(claims["key_results"]) == 1
    assert claims["key_results"][0].text == "This is numbered"


def test_run_builds_cards():
    llm = FakeLLMClient(responses=[("Extract a structured paper card", SAMPLE)])
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="a",
                     paragraphs=[Paragraph(page=1, index=0, text="intro text")])
    ])
    rp = RetrievedPapers(task_id="t", papers=[
        RetrievedPaper(paper_id="paper:1", title="A", authors=["Bob"], year=2024, venue="ICML")
    ])
    cards = run("t", pp, rp, llm, [])
    # Uses paper_cards field (not cards)
    assert len(cards.paper_cards) == 1
    card = cards.paper_cards[0]
    assert card.paper_id == "paper:1"
    assert card.title == "A"
    assert card.authors == ["Bob"]
    assert card.year == 2024
    assert card.matched_aspects == []
    assert "key_results" in card.possible_claims
    assert "method" in card.possible_claims
    assert "setup" in card.possible_claims
    assert "limitations" in card.possible_claims
    assert card.possible_claims["method"][0].dimension == "method"
    assert card.possible_claims["key_results"][0].evidence_ids == ["paper:1_p3_0"]


def test_run_empty_paragraphs():
    llm = FakeLLMClient(responses=[("Extract a structured paper card", SAMPLE)])
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:2", title="B", abstract="b", paragraphs=[])
    ])
    rp = RetrievedPapers(task_id="t", papers=[
        RetrievedPaper(paper_id="paper:2", title="B")
    ])
    cards = run("t", pp, rp, llm, [])
    assert len(cards.paper_cards) == 1
    assert cards.paper_cards[0].paper_id == "paper:2"


def test_run_missing_metadata():
    llm = FakeLLMClient(responses=[("Extract a structured paper card", SAMPLE)])
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:3", title="C", abstract="c", paragraphs=[])
    ])
    rp = RetrievedPapers(task_id="t", papers=[])
    cards = run("t", pp, rp, llm, [])
    card = cards.paper_cards[0]
    assert card.authors == []
    assert card.year is None
    assert card.venue is None
