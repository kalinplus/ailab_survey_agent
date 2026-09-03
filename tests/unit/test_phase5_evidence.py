import pytest
from tools.phases.phase5_evidence import run
from tools.models.artifacts import (
    ParsedPapers, ParsedPaper, Paragraph,
    PaperCards, PaperCard, Claim,
)
from tools.nlp.nli_verifier import NLIResult


class NLIStub:
    """Minimal NLI stub returning canned result for every call."""

    def __init__(self, support_type="direct", confidence=0.85):
        self._support_type = support_type
        self._confidence = confidence

    def best_match(self, evidence_text, claim_texts):
        return NLIResult("entailment", self._support_type, self._confidence)


class NLIContradictoryStub:
    """Returns contradictory with confidence=0 so supports_claims stays empty."""

    def best_match(self, evidence_text, claim_texts):
        return NLIResult("contradiction", "contradictory", 0.0)


class NLINoClaimsStub:
    """Should never be called; placeholder for papers with no claims."""

    def best_match(self, evidence_text, claim_texts):
        raise RuntimeError("NLI called when no claims exist")


# --- tests ---

def test_evidence_id_and_supports():
    """Paragraph gets evidence_id {paper_id}_p{page}_{idx}; abstract gets {paper_id}_p0_0.
    supports_claims is populated when claims exist."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="abs",
                    paragraphs=[Paragraph(page=3, index=0, text="we show SOTA")])
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"key_results": [Claim(text="SOTA result", dimension="key_results")]})
    ])
    es = run("t", pp, cards, NLIStub())
    ids = [e.evidence_id for e in es.evidence]
    assert "paper:1_p3_0" in ids
    assert "paper:1_p0_0" in ids
    para_ev = [e for e in es.evidence if e.evidence_id == "paper:1_p3_0"][0]
    assert para_ev.supports_claims[0]["support_type"] == "direct"
    assert para_ev.supports_claims[0]["confidence"] == 0.85


def test_empty_parsed_papers():
    """Empty parsed_papers → es.evidence == []"""
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[])
    es = run("t", pp, cards, NLIStub())
    assert es.evidence == []


def test_no_card_for_paper():
    """Paper with no matching card → evidences produced with empty supports_claims."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:2", title="B", abstract="abs text",
                    paragraphs=[Paragraph(page=1, index=0, text="content")])
    ])
    cards = PaperCards(task_id="t", paper_cards=[])  # no card for paper:2
    es = run("t", pp, cards, NLINoClaimsStub())  # NLI should never be called
    assert len(es.evidence) == 2  # abstract + paragraph
    for e in es.evidence:
        assert e.supports_claims == []


def test_source_types():
    """Abstract → 'abstract', paragraph → 'paragraph', caption → 'caption'."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="the abstract",
                    paragraphs=[Paragraph(page=2, index=0, text="para text")],
                    figures=[{"num": 1, "page": 5, "caption": "fig caption"}])
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"method": [Claim(text="some claim", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIStub())
    by_id = {e.evidence_id: e for e in es.evidence}
    assert by_id["paper:1_p0_0"].source_type == "abstract"
    assert by_id["paper:1_p2_0"].source_type == "paragraph"
    assert by_id["paper:1_p5_1"].source_type == "caption"


def test_figure_without_caption_skipped():
    """Figure with no caption → no evidence produced."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="",
                    paragraphs=[],
                    figures=[{"num": 1, "page": 3}])
    ])
    cards = PaperCards(task_id="t", paper_cards=[])
    es = run("t", pp, cards, NLIStub())
    assert es.evidence == []


def test_contradictory_low_confidence_skipped():
    """Contradictory with confidence=0 → supports_claims stays empty."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="",
                    paragraphs=[Paragraph(page=1, index=0, text="text")])
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"key_results": [Claim(text="claim", dimension="key_results")]})
    ])
    es = run("t", pp, cards, NLIContradictoryStub())
    assert es.evidence[0].supports_claims == []


def test_multiple_papers():
    """Two papers, only one has a card; both produce evidences."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="p1", title="A", abstract="a1",
                    paragraphs=[Paragraph(page=1, index=0, text="t1")]),
        ParsedPaper(paper_id="p2", title="B", abstract="a2",
                    paragraphs=[Paragraph(page=2, index=0, text="t2")]),
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="p1", title="A",
                  possible_claims={"key_results": [Claim(text="c1", dimension="key_results")]})
    ])
    es = run("t", pp, cards, NLIStub())
    p1_evs = [e for e in es.evidence if e.paper_id == "p1"]
    p2_evs = [e for e in es.evidence if e.paper_id == "p2"]
    assert len(p1_evs) == 2  # abstract + paragraph
    assert len(p2_evs) == 2
    # p1 has supports_claims, p2 does not
    assert all(e.supports_claims for e in p1_evs)
    assert all(e.supports_claims == [] for e in p2_evs)


# --- agentic-search backfill ---


class FakeAgenticSV:
    """Duck-types SciVerseClient.agentic_search (real contract: hits[].chunk/doc_id/page_no/offset)."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = 0

    def agentic_search(self, query, top_k=10):
        self.calls += 1
        return {"hits": self.hits}


def test_agentic_backfill_for_unsupported_claim():
    """Claim with no parsed-paper evidence is grounded via a real SciVerse chunk."""
    sv = FakeAgenticSV(hits=[{
        "chunk": "DreamerV3 learns a world model and plans via imagination in latent space.",
        "title": "DreamerV3", "publication_published_year": 2023,
        "page_no": 2, "offset": 100, "doc_id": "abc",
    }])
    pp = ParsedPapers(task_id="t", papers=[])  # no parsed evidence -> claim unsupported
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="DreamerV3",
                  possible_claims={"method": [Claim(text="plans in imagination", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIStub(), sciverse=sv)
    agentic = [e for e in es.evidence if e.source_type == "agentic_chunk"]
    assert len(agentic) == 1
    assert agentic[0].source_page == 2
    assert agentic[0].paper_id == "paper:1"
    assert agentic[0].supports_claims[0]["claim_text"] == "plans in imagination"
    assert sv.calls == 1


def test_no_backfill_when_sciverse_none():
    """sciverse=None -> no backfill, no agentic call (existing behavior preserved)."""
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"method": [Claim(text="c", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIStub())  # sciverse defaults None
    assert es.evidence == []


def test_backfill_skipped_when_claim_already_supported():
    """Claim already grounded by parsed evidence -> agentic_search not called."""
    sv = FakeAgenticSV(hits=[])
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="abs", paragraphs=[])
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"method": [Claim(text="claim", dimension="method")]})
    ])
    run("t", pp, cards, NLIStub(), sciverse=sv)  # NLIStub supports -> abstract grounds claim
    assert sv.calls == 0


def test_backfill_skips_contradictory_chunks():
    """Chunk that contradicts the claim (conf<=0) is not added as evidence."""
    sv = FakeAgenticSV(hits=[{
        "chunk": "x", "title": "T", "publication_published_year": 2024, "page_no": 1, "offset": 0, "doc_id": "d",
    }])
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"method": [Claim(text="c", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIContradictoryStub(), sciverse=sv)
    assert [e for e in es.evidence if e.source_type == "agentic_chunk"] == []


# --- S1 depth: abstract + fulltext double layer for core papers -------------------


def test_core_paper_fulltext_yields_abstract_and_fulltext_layers():
    """A MinerU-parsed core paper contributes its abstract AND its fulltext paragraphs;
    a paper with no parsed body stays at the abstract layer."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:core", title="Core", abstract="core abstract",
                    paragraphs=[Paragraph(page=1, index=0, text="intro"),
                                Paragraph(page=2, index=0, text="method")]),
        ParsedPaper(paper_id="paper:thin", title="Thin", abstract="thin abstract"),
    ])
    cards = PaperCards(task_id="t", paper_cards=[])
    es = run("t", pp, cards, NLIStub())

    core = [e for e in es.evidence if e.paper_id == "paper:core"]
    thin = [e for e in es.evidence if e.paper_id == "paper:thin"]
    assert [e.source_type for e in core].count("abstract") == 1
    assert [e.source_type for e in core].count("paragraph") == 2  # fulltext layer
    assert [e.source_type for e in thin] == ["abstract"]
    assert len(es.evidence) == 4
