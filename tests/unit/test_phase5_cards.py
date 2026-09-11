import json

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
    assert claims["key_results"][0].evidence_ids == []  # legacy pages are not supplied block identities
    # uncited claim has empty evidence_ids
    assert claims["key_results"][1].text == "Single config works"
    assert claims["key_results"][1].evidence_ids == []
    # method
    assert len(claims["method"]) == 1
    assert claims["method"][0].text.startswith("Recurrent state space")
    assert claims["method"][0].evidence_ids == []  # legacy pages are not supplied block identities
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
    # Unknown headers end the bucket; unrelated text cannot become a result.
    assert len(claims["key_results"]) == 1
    assert claims["key_results"][0].text == "First result"
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
    assert claims["key_results"][0].evidence_ids == []  # legacy pages are not supplied block identities
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
    llm = FakeLLMClient(responses=[("Extract claim cards from the numbered source blocks", SAMPLE)])
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
    assert card.possible_claims["key_results"][0].evidence_ids == []  # legacy pages are not supplied block identities


def test_run_empty_paragraphs():
    llm = FakeLLMClient(responses=[("Extract claim cards from the numbered source blocks", SAMPLE)])
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
    llm = FakeLLMClient(responses=[("Extract claim cards from the numbered source blocks", SAMPLE)])
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:3", title="C", abstract="c", paragraphs=[])
    ])
    rp = RetrievedPapers(task_id="t", papers=[])
    cards = run("t", pp, rp, llm, [])
    card = cards.paper_cards[0]
    assert card.authors == []
    assert card.year is None
    assert card.venue is None


def test_fallback_card_has_no_placeholder_claims():
    """No limitation evidence -> empty limitations bucket, no sentinel strings.

    Regression for the L3-review placeholder leak: _fallback_card_response used
    to write "Detailed limitations require deeper paper parsing or manual
    review." (plus other fabricated setup/method filler) into every card built
    without LLM output.
    """
    llm = FakeLLMClient()
    # Phase2 sets this flag on the shared client when LLM calls keep failing.
    llm._evisurvey_card_fallback = True
    rp = RetrievedPapers(task_id="t", papers=[
        RetrievedPaper(paper_id="paper:9", title="D",
                       abstract="Real abstract describing the method and results. "
                                "A second sentence reports the evaluation protocol.")
    ])
    cards = run("t", ParsedPapers(task_id="t", papers=[]), rp, llm, [])
    card = cards.paper_cards[0]
    assert card.possible_claims["limitations"] == []
    assert card.possible_claims["setup"] == []
    assert card.possible_claims["key_results"] == []
    assert card.possible_claims["method"] == []
    assert card.extraction_status == "no_safe_claims"
    assert card.source_blocks[0]["text"].startswith("Real abstract")
    dumped = json.dumps(card.model_dump(), ensure_ascii=False)
    assert "require deeper paper parsing" not in dumped
    assert "retrieved evidence base" not in dumped
    assert "relevant to the requested survey topic" not in dumped


def test_fallback_card_without_source_leaves_all_buckets_empty():
    """No abstract and no body -> fallback emits no claims at all."""
    llm = FakeLLMClient()
    llm._evisurvey_card_fallback = True
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:10", title="E", abstract="", paragraphs=[])
    ])
    rp = RetrievedPapers(task_id="t", papers=[])
    cards = run("t", pp, rp, llm, [])
    card = cards.paper_cards[0]
    assert card.possible_claims == {
        "key_results": [], "method": [], "setup": [], "limitations": []}
    assert "require deeper paper parsing" not in json.dumps(
        card.model_dump(), ensure_ascii=False)


def test_doc_id_flows_from_retrieved_to_cards():
    """doc_id reaches both deep and shallow cards — it is the /content key the
    evidence phase grounds own-paper claims by."""
    from tools.models.artifacts import RetrievedPaper, RetrievedPapers

    llm = FakeLLMClient(responses=[("Extract claim cards from the numbered source blocks", SAMPLE)])
    retrieved = RetrievedPapers(task_id="t", papers=[
        RetrievedPaper(paper_id="paper:10.1/deep", title="D", doc_id="doc-deep", abstract=""),
        RetrievedPaper(paper_id="paper:10.1/shallow", title="S", doc_id="doc-shallow",
                       abstract="An abstract with enough content for a card."),
    ])
    parsed = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:10.1/deep", title="D", abstract="",
                    paragraphs=[Paragraph(page=1, index=0, text="body text")]),
    ])
    cards = run("t", parsed, retrieved, llm, [])
    by_pid = {c.paper_id: c.doc_id for c in cards.paper_cards}
    assert by_pid["paper:10.1/deep"] == "doc-deep"
    assert by_pid["paper:10.1/shallow"] == "doc-shallow"


def test_author_block_paragraphs_excluded_from_card_body():
    """Wave 5 (2b): leading title/author/affiliation blocks of a /content
    fulltext never feed the card LLM (live wave-4 pasted 'Jake Bruce<sup>…'
    into survey prose)."""
    from tools.phases.phase5_cards import _is_author_block

    author_block = "# Genie: Generative Interactive Environments Jake Bruce<sup>*,1</sup>, Michael Dennis<sup>*,1</sup>"
    affiliation = "Department of Computer Science, Université de Montréal"
    body = ("Genie learns a latent action interface fully unsupervised from Internet videos, "
            "enabling generation of new environments. " * 3)
    mentions_uni = ("Researchers at a University previously showed that world models help agents. " * 3)
    assert _is_author_block(author_block) and _is_author_block(affiliation)
    assert not _is_author_block(body) and not _is_author_block(mentions_uni)


def test_json_cards_require_actual_same_paper_blocks_and_claim_roles():
    quote = "Tree-based planning methods have enjoyed huge success where a perfect simulator is available."
    blocks = [dict(evidence_id="paper:mu_para_px_7", paper_id="paper:mu", text=quote)]
    row = dict(text="MuZero uses perfect simulators", dimension="method", source_role="own_method",
               source_quote=quote, evidence_ids=["paper:mu_para_px_7"])
    assert parse_card_response(json.dumps([row]), "paper:mu", blocks)["method"] == []
    row["source_role"] = "background"
    claims = parse_card_response(json.dumps([row]), "paper:mu", blocks)
    assert claims["method"][0].source_role == "background"
    assert claims["method"][0].evidence_ids == ["paper:mu_para_px_7"]
    row["evidence_ids"] = ["paper:mu_para_p0_0"]
    assert parse_card_response(json.dumps([row]), "paper:mu", blocks)["method"] == []


def test_implicit_own_method_negation_and_previous_comparison_not_background():
    from tools.verify.source_contract import role_violation
    for quote in ["MuZero plans without a perfect simulator.",
                  "Alpha improves previous methods using learned dynamics.",
                  "The model predicts policy, value and reward."]:
        assert role_violation("own_method", quote) == ""


def test_api_failure_does_not_poison_next_card_and_flags_ignored(caplog):
    class Client:
        _evisurvey_card_fallback = True
        def __init__(self):
            self.calls = 0
        def chat(self, messages, *, max_tokens=None, thinking_mode=False):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary outage")
            return json.dumps([dict(text="The model learns policy predictions", dimension="method",
                                   source_role="own_method", source_quote="We learn policy predictions.",
                                   evidence_ids=["paper:2_abs_p0_0"])])
    client = Client()
    retrieved = RetrievedPapers(task_id="t", papers=[RetrievedPaper(paper_id=f"paper:{i}", title="T",
                   abstract="We learn policy predictions.") for i in (1, 2)])
    cards = run("t", ParsedPapers(task_id="t", papers=[]), retrieved, client, []).paper_cards
    assert client.calls == 2
    assert cards[0].extraction_status == "api_or_response_failed"
    assert cards[1].extraction_status == "extracted"
    assert "temporary outage" in caplog.text
    assert cards[0].possible_claims["method"] == []


def test_prompt_source_selection_reaches_late_methods():
    class Client:
        def chat(self, messages, *, max_tokens=None, thinking_mode=False):
            prompt = messages[-1]["content"]
            assert "We propose a learned planning model." in prompt
            assert "paper:mu_para_px_35" in prompt
            assert "Bibliography entry" not in prompt
            return "[]"
    parsed = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:mu", title="MuZero",
        paragraphs=[Paragraph(page=None, index=i, text="Background details.") for i in range(35)] +
        [Paragraph(page=None, index=35, text="We propose a learned planning model."),
         Paragraph(page=None, index=36, text="Bibliography entry", role="reference")])])
    run("t", parsed, RetrievedPapers(task_id="t", papers=[]), Client(), [])


def test_card_prompt_asks_for_own_limitations():
    """Wave7: the prompt must explicitly request own_limitation claims so the
    open-challenges/future-directions sections have citable evidence."""
    from tools.phases.phase5_cards import CARD_PROMPT
    assert "own_limitation" in CARD_PROMPT
    assert "limitations" in CARD_PROMPT
