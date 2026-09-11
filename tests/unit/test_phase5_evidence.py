import logging

import pytest
from tools.phases.phase5_evidence import run
from tools.models.artifacts import (
    ParsedPapers, ParsedPaper, Paragraph,
    PaperCards, PaperCard, Claim, RetrievedPaper,
)
from tools.nlp.nli_verifier import NLIResult


class NLIStub:
    """Minimal NLI stub returning canned result for every call."""

    def __init__(self, support_type="direct", confidence=0.85):
        self._support_type = support_type
        self._confidence = confidence

    def best_match(self, claim, evidences):
        return NLIResult("entailment", self._support_type, self._confidence)


class NLIContradictoryStub:
    """Returns contradictory with confidence=0 so supports_claims stays empty."""

    def best_match(self, claim, evidences):
        return NLIResult("contradiction", "contradictory", 0.0)


class NLINoClaimsStub:
    """Should never be called; placeholder for papers with no claims."""

    def best_match(self, claim, evidences):
        raise RuntimeError("NLI called when no claims exist")


# --- tests ---

def test_evidence_id_and_supports():
    """Paragraph gets namespaced id {paper_id}_para_p{page}_{idx}; abstract gets
    {paper_id}_abs_p0_0. supports_claims is populated when claims exist."""
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
    assert "paper:1_para_p3_0" in ids
    assert "paper:1_abs_p0_0" in ids
    para_ev = [e for e in es.evidence if e.evidence_id == "paper:1_para_p3_0"][0]
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
    assert by_id["paper:1_abs_p0_0"].source_type == "abstract"
    assert by_id["paper:1_para_p2_0"].source_type == "paragraph"
    assert by_id["paper:1_cap_p5_1"].source_type == "caption"


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
    """Claim with no parsed-paper evidence is grounded via a real SciVerse chunk;
    the chunk's identity is retained in source_doc_id/source_title/source_chunk_id."""
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
    # non-DOI paper id matched via normalized title: provenance fields retained
    assert agentic[0].source_doc_id == "abc"
    assert agentic[0].source_title == "DreamerV3"
    assert agentic[0].source_chunk_id == "abc:2:100"
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


# --- wave 1B: provenance gate, NLI direction, id namespaces, reference skip ---


def test_agentic_backfill_provenance_gate(caplog):
    """Cross-paper pollution fixture: a hit whose doc_id is a different paper's
    DOI is rejected (never attached under the target paper_id, logged and
    counted); a doc_id match enters the store with all three source_* fields."""
    bad = {  # the "Resistance Avalanche" chunk that polluted AvalonBench
        "chunk": "Resistance Avalanche studies avalanches in resistor networks.",
        "title": "Resistance Avalanche", "doc_id": "10.48550/arxiv.2401.06626",
        "page_no": 1, "offset": 0,
    }
    good = {  # same paper, DOI spelled differently (prefix/case-insensitive)
        "chunk": "AvalonBench evaluates language agents on the Avalon social deduction game.",
        "title": "AvalonBench", "doc_id": "10.48550/ARXIV.2408.14837",
        "page_no": 2, "offset": 42,
    }
    sv = FakeAgenticSV(hits=[bad, good])
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:10.48550/arxiv.2408.14837", title="AvalonBench",
                  possible_claims={"key_results": [Claim(text="evaluates Avalon gameplay",
                                                         dimension="key_results")]})
    ])
    with caplog.at_level(logging.WARNING, logger="tools.phases.phase5_evidence"):
        es = run("t", pp, cards, NLIStub(), sciverse=sv)
    agentic = [e for e in es.evidence if e.source_type == "agentic_chunk"]
    assert len(agentic) == 1  # the off-topic hit never entered the store
    assert agentic[0].text == good["chunk"]
    assert agentic[0].paper_id == "paper:10.48550/arxiv.2408.14837"
    assert agentic[0].source_doc_id == "10.48550/ARXIV.2408.14837"
    assert agentic[0].source_title == "AvalonBench"
    assert agentic[0].source_chunk_id == "10.48550/ARXIV.2408.14837:2:42"
    # rejection is observable: per-hit warning + counted summary
    messages = [r.getMessage() for r in caplog.records]
    assert any("provenance gate rejected" in m and "2401.06626" in m for m in messages)
    assert any("1 hits rejected by provenance gate" in m for m in messages)


def test_hit_matches_paper_title_branch_for_doi_ids():
    """Real agentic-search doc_ids are opaque internal hashes (verified live
    2026-09-09), so a DOI-only branch never fires and would reject even the
    paper's own chunks. Title equality is an OR branch for every id form;
    different-title hits stay rejected, empty titles never match."""
    from tools.verify.provenance import hit_matches_paper

    pid = "paper:10.48550/arxiv.2408.14837"
    title = "DIFFUSION MODELS ARE REAL-TIME GAME ENGINES"
    own = {"doc_id": "c68556166f44a96bfd7dcb0f9989328585b5849efeaa5",
           "title": "Diffusion Models are Real-Time Game Engines"}
    foreign = {"doc_id": "c68556166f44a96bfd7dcb0f9989328585b5849efeaa5",
               "title": "Combat: Conditional World Models for Behavioral Agent Training"}
    assert hit_matches_paper(pid, title, own)
    assert not hit_matches_paper(pid, title, foreign)
    assert not hit_matches_paper(pid, "", own)


class SelectiveNLI:
    """Recording stub: direct support only for the one matching claim text;
    records every (claim, evidences) argument pair to pin the NLI direction."""

    def __init__(self, match):
        self.match = match
        self.calls = []

    def best_match(self, claim, evidences):
        self.calls.append((claim, list(evidences)))
        if claim == self.match:
            return NLIResult("entailment", "direct", 0.9)
        return NLIResult("contradiction", "contradictory", 0.0)


def test_nli_direction_and_per_claim_registration():
    """best_match is always called with the claim as the FIRST argument and the
    fragment as the evidence list; supports_claims registers the claim(s) that
    actually matched, not claim_texts[0]."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="",
                    paragraphs=[Paragraph(page=1, index=0, text="fragment text")])
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A", possible_claims={
            "key_results": [Claim(text="claim one", dimension="key_results"),
                            Claim(text="claim two", dimension="key_results")]})
    ])
    nli = SelectiveNLI(match="claim two")
    es = run("t", pp, cards, nli)
    # direction: each call judged ONE claim against the fragment as evidence
    assert sorted(call[0] for call in nli.calls) == ["claim one", "claim two"]
    for claim_arg, evidences_arg in nli.calls:
        assert evidences_arg == ["fragment text"]
    # registration: only the claim that actually matched is bound
    ev = es.evidence[0]
    assert [s["claim_text"] for s in ev.supports_claims] == ["claim two"]
    assert ev.supports_claims[0]["support_type"] == "direct"


def test_agentic_backfill_nli_direction():
    """Backfill judges best_match(claim.text, [chunk]) — claim first."""
    seen = []

    class DirectionSpy:
        def best_match(self, claim, evidences):
            seen.append((claim, evidences))
            return NLIResult("entailment", "direct", 0.9)

    sv = FakeAgenticSV(hits=[{
        "chunk": "chunk text", "title": "T", "doc_id": "d", "page_no": 1, "offset": 0}])
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="T",
                  possible_claims={"method": [Claim(text="the claim", dimension="method")]})
    ])
    run("t", pp, cards, DirectionSpy(), sciverse=sv)
    assert seen == [("the claim", ["chunk text"])]


def test_evidence_id_namespaces_unique_across_source_types():
    """Same (paper, page, idx) in different source types never collides — the
    caption (page, num) vs paragraph (page, index) id collision is gone."""
    from tools.models.common import evidence_id
    ids = [
        evidence_id("paper:1", 5, 3, "abstract"),
        evidence_id("paper:1", 5, 3, "paragraph"),
        evidence_id("paper:1", 5, 3, "caption"),
        evidence_id("paper:1", 5, 3, "agentic_chunk"),
        evidence_id("paper:1", 5, 3, "repair"),
    ]
    assert len(set(ids)) == len(ids)


def test_mixed_source_store_ids_unique():
    """A run-level store mixing abstract+paragraph+caption+agentic keeps every
    evidence_id unique; page=None paragraphs (markdown-only channel) work too."""
    class ChunkKeyedNLI:
        """Direct support only when the EVIDENCE side carries the fetched chunk's
        marker, so parsed fragments stay unsupported and backfill runs."""

        def best_match(self, claim, evidences):
            if any("agentic chunk" in ev for ev in evidences):
                return NLIResult("entailment", "direct", 0.9)
            return NLIResult("contradiction", "contradictory", 0.0)

    sv = FakeAgenticSV(hits=[{
        "chunk": "agentic chunk", "title": "A", "doc_id": "d",
        "page_no": 2, "offset": 0}])
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="the abstract",
                    paragraphs=[Paragraph(page=5, index=3, text="para"),
                                Paragraph(page=None, index=7, text="md-only")],
                    figures=[{"num": 3, "page": 5, "caption": "cap"}]),
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="A",
                  possible_claims={"method": [Claim(text="c", dimension="method")]})
    ])
    es = run("t", pp, cards, ChunkKeyedNLI(), sciverse=sv)
    ids = [e.evidence_id for e in es.evidence]
    assert len(ids) == len(set(ids)) == 5  # abstract + 2 paragraphs + caption + agentic
    assert any(e.source_page is None for e in es.evidence)  # page=None tolerated


def test_reference_role_paragraph_skipped():
    """role='reference' bibliography blocks never become this paper's evidence;
    a paragraph without the role behaves as before (field defaults to '')."""
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:1", title="A", abstract="",
                    paragraphs=[Paragraph(page=1, index=0,
                                          text="Smith et al. show X in earlier work [12].",
                                          role="reference"),
                                Paragraph(page=1, index=1, text="our own method")])
    ])
    cards = PaperCards(task_id="t", paper_cards=[])
    es = run("t", pp, cards, NLIStub())
    assert [e.text for e in es.evidence] == ["our own method"]


# --- wave 1.5: own-paper fulltext grounding via /content (doc_id identity) ---

CONTENT_MD = """# Real Paper

Intro paragraph.

## Method

The engine runs DOOM at playable frame rates.

## References

Someone et al. Completely other work entirely.
"""


class FakeContentSV:
    """Duck-types SciVerseClient.read_full_text (own-paper fulltext by doc_id)."""

    def __init__(self, md=CONTENT_MD, error=None):
        self.md = md
        self.error = error
        self.calls = []

    def read_full_text(self, doc_id, max_pages=8):
        self.calls.append(doc_id)
        if self.error:
            raise self.error
        return self.md


def test_content_backfill_grounds_own_paper_with_provenance():
    """A card with doc_id and an unsupported claim gets content_chunk evidence
    from the paper's OWN fulltext: namespaced id, page=None token, and the
    full provenance triple. References-section blocks stay out."""
    sv = FakeContentSV()
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:10.1/x", title="Real Paper", doc_id="doc-hash-1",
                  possible_claims={"method": [Claim(text="runs DOOM", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIStub(), sciverse=sv)
    assert sv.calls == ["doc-hash-1"]
    content = [e for e in es.evidence if e.source_type == "content_chunk"]
    texts = [e.text for e in content]
    assert "Intro paragraph." in texts
    assert "The engine runs DOOM at playable frame rates." in texts
    assert not any("Completely other work" in t for t in texts)  # references excluded
    for e in content:
        assert e.evidence_id.startswith("paper:10.1/x_content_px_")
        assert e.source_doc_id == "doc-hash-1"
        assert e.source_title == "Real Paper"
        assert e.source_chunk_id.startswith("doc-hash-1:b")
        assert e.source_page is None
        assert e.supports_claims[0]["claim_text"] == "runs DOOM"


def test_content_backfill_skips_cards_without_doc_id():
    """No identity link -> no fetch (agentic gate path still applies on its own)."""
    sv = FakeContentSV()
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="seed:abc", title="T",
                  possible_claims={"method": [Claim(text="c", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIStub(), sciverse=sv)
    assert sv.calls == []
    assert es.evidence == []


def test_content_backfill_degrades_per_paper_on_api_error(caplog):
    """A failing /content fetch skips that paper; the run keeps its parsed evidence."""
    import logging as _logging

    class HalfBrokenSV:
        def __init__(self):
            self.n = 0

        def read_full_text(self, doc_id, max_pages=8):
            self.n += 1
            raise RuntimeError("boom")

    sv = HalfBrokenSV()
    pp = ParsedPapers(task_id="t", papers=[])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:10.1/y", title="Y", doc_id="doc-2",
                  possible_claims={"method": [Claim(text="c", dimension="method")]})
    ])
    with caplog.at_level(_logging.WARNING, logger="tools.phases.phase5_evidence"):
        es = run("t", pp, cards, NLIStub(), sciverse=sv)
    assert es.evidence == []
    assert any("content fetch failed" in r.getMessage() for r in caplog.records)


def test_content_backfill_only_fetches_cards_with_unsupported_claims():
    """Claims already grounded by parsed evidence never trigger a /content call."""
    sv = FakeContentSV()
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:10.1/z", title="Z", abstract="covers it",
                    paragraphs=[])
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:10.1/z", title="Z", doc_id="doc-3",
                  possible_claims={"method": [Claim(text="c", dimension="method")]})
    ])
    es = run("t", pp, cards, NLIStub(), sciverse=sv)
    # abstract evidence grounds the claim (NLIStub always direct) -> no fetch
    assert sv.calls == []
    assert [e.source_type for e in es.evidence] == ["abstract"]


def test_content_backfill_skips_papers_already_parsed_from_content():
    """Wave 2: a paper whose /content fulltext entered P3 as parsed paragraphs
    (parse_status=content_fulltext) must not be fetched again at P5.2."""
    sv = FakeContentSV()
    pp = ParsedPapers(task_id="t", papers=[
        ParsedPaper(paper_id="paper:10.1/w", title="W", abstract="",
                    parse_status="content_fulltext",
                    paragraphs=[Paragraph(page=None, index=0, text="we show SOTA")]),
    ])
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:10.1/w", title="W", doc_id="doc-w",
                  possible_claims={"key_results": [Claim(text="SOTA", dimension="key_results")]})
    ])
    es = run("t", pp, cards, NLIStub(), sciverse=sv)
    assert sv.calls == []  # already parsed — no refetch
    assert [e.source_type for e in es.evidence] == ["paragraph"]  # from the parsed body


@pytest.mark.parametrize("support,label,confidence", [("indirect", "neutral", .9), ("contradictory", "contradiction", .99)])
def test_only_direct_support_stops_backfill_per_paper(support, label, confidence):
    from tools.phases.phase5_evidence import _agentic_backfill
    from tools.models.artifacts import Evidence
    sv = FakeAgenticSV([])
    cards = PaperCards(task_id="t", paper_cards=[PaperCard(paper_id=pid, title=pid,
        possible_claims={"method": [Claim(text="shared text", dimension="method")]}) for pid in ("paper:a", "paper:b")])
    existing = [Evidence(evidence_id="a", paper_id="paper:a", text="x",
        supports_claims=[dict(claim_text="shared text", support_type="direct")]),
        Evidence(evidence_id="b", paper_id="paper:b", text="x",
        supports_claims=[dict(claim_text="shared text", support_type=support)])]
    _agentic_backfill(cards, existing, sv, NLIStub())
    assert sv.calls == 1  # paper:a must not suppress paper:b's identical claim
    class NLI:
        def best_match(self, claim, evidences):
            return NLIResult(label, support, confidence)
    from tools.phases.phase5_evidence import _make
    assert _make("paper:b", None, 7, "x", "paragraph", ["shared text"], NLI()).supports_claims == []


def test_shallow_source_binding_retains_role_dimension_and_identity():
    from tools.phases.phase5_cards import build_shallow_card
    import json
    rp = RetrievedPaper(paper_id="paper:mu", title="MuZero", abstract="MuZero learns policy and value predictions.")
    class Client:
        def chat(self, messages, *, max_tokens=None, thinking_mode=False):
            return json.dumps([dict(text="The model predicts policy and value", dimension="method",
                source_role="own_method", evidence_ids=["paper:mu_abs_p0_0"], source_quote=rp.abstract)])
    card = build_shallow_card(rp, Client(), [])
    store = run("t", ParsedPapers(task_id="t", papers=[]), PaperCards(task_id="t", paper_cards=[card]), NLIStub())
    ev = store.evidence[0]
    row = ev.supports_claims[0]
    assert row["source_role"] == "own_method" and row["dimension"] == "method"
    assert row["paper_id"] == "paper:mu" and row["evidence_ids"] == [ev.evidence_id]
    assert ev.source_title == "MuZero" and ev.source_chunk_id == ev.evidence_id


def test_agentic_backfill_skips_declared_claims():
    """Claims with declared evidence_ids can only be grounded by their bound
    blocks (valid_support membership), so agentic search for them is dead
    weight: no search fires, no evidence is added."""
    sv = FakeAgenticSV(hits=[{
        "chunk": "anything", "title": "DreamerV3",
        "publication_published_year": 2023, "page_no": 1, "offset": 0, "doc_id": "abc",
    }])
    pp = ParsedPapers(task_id="t", papers=[])  # no parsed evidence -> claim unsupported
    cards = PaperCards(task_id="t", paper_cards=[
        PaperCard(paper_id="paper:1", title="DreamerV3",
                  possible_claims={"method": [Claim(
                      text="plans in imagination", dimension="method",
                      source_role="own_method", source_quote="plans in imagination",
                      evidence_ids=["paper:1_abs_p0_0"])]})
    ])
    es = run("t", pp, cards, NLIStub(), sciverse=sv)
    assert sv.calls == 0
    assert not [e for e in es.evidence if e.source_type == "agentic_chunk"]
