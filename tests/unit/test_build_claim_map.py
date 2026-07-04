"""Tests for tools.verify.claim_mapper — NLI-first claim mapper with LLM fallback."""

from tools.verify.claim_mapper import run, extract_claims_with_citations
from tools.models.artifacts import EvidenceStore, Evidence, ParsedPapers
from tools.nlp.nli_verifier import NLIResult
from tools.clients.llm_fake import FakeLLMClient


class NLIStub:
    """Duck-typed NLI that always returns a fixed label/confidence."""

    def __init__(self, label, conf):
        self.label = label
        self.conf = conf

    def best_match(self, claim, evidences):
        support_map = {
            "entailment": "direct",
            "neutral": "indirect",
            "contradiction": "contradictory",
        }
        return NLIResult(self.label, support_map[self.label], self.conf)


def _empty_parsed():
    return ParsedPapers(task_id="t", papers=[])


# ── extract_claims_with_citations ──────────────────────────────────

class TestExtractClaims:
    def test_sentence_with_citation(self):
        md = "DreamerV3 flies airplanes [paper:1]."
        result = extract_claims_with_citations(md)
        assert result == [("DreamerV3 flies airplanes", "paper:1")]

    def test_sentence_without_citation_skipped(self):
        md = "This sentence has no citation. Next one does [paper:2]."
        result = extract_claims_with_citations(md)
        assert len(result) == 1
        assert result[0] == ("Next one does", "paper:2")

    def test_multi_citation_uses_first(self):
        md = "Two refs here [paper:A] and [paper:B]."
        result = extract_claims_with_citations(md)
        assert result == [("Two refs here  and", "paper:A")]

    def test_chinese_period(self):
        md = "世界模型很重要 [paper:1]。"
        result = extract_claims_with_citations(md)
        assert result == [("世界模型很重要", "paper:1")]


# ── Stage 3: no evidence → unsupported ─────────────────────────────

class TestNoEvidence:
    def test_unsupported_when_no_evidence(self):
        md = "DreamerV3 flies airplanes [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[])
        cm = run("t", md, es, _empty_parsed(), NLIStub("contradiction", 0.9))
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].confidence == 0.0
        assert cm.entries[0].evidence_ids == []


# ── Stage 1: exact direct match → supported, conf 1.0 ──────────────

class TestStage1Direct:
    def test_direct_support(self):
        md = "DreamerV3 masters many domains [paper:1]."
        e1 = Evidence(
            evidence_id="paper:1_p1_0", paper_id="paper:1",
            text="We evaluate on 150 domains.",
            supports_claims=[{"claim_text": "...", "support_type": "direct"}],
        )
        e2 = Evidence(
            evidence_id="paper:1_p1_1", paper_id="paper:1",
            text="Another paragraph.",
            supports_claims=[{"claim_text": "...", "support_type": "indirect"}],
        )
        es = EvidenceStore(task_id="t", evidence=[e1, e2])
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.5))
        assert cm.entries[0].status == "supported"
        assert cm.entries[0].confidence == 1.0
        assert cm.entries[0].evidence_ids == ["paper:1_p1_0", "paper:1_p1_1"]


# ── Stage 2: NLI entailment / neutral / contradiction ──────────────

class TestStage2NLI:
    def test_supported_via_entailment(self):
        md = "DreamerV3 masters many domains [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p3_0", paper_id="paper:1",
                     text="we master 150 domains"),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("entailment", 0.85))
        assert cm.entries[0].status == "supported"
        assert cm.entries[0].confidence == 0.85
        assert cm.entries[0].evidence_ids == ["paper:1_p3_0"]

    def test_weak_via_neutral(self):
        md = "Some vague claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p2_0", paper_id="paper:1",
                     text="Partially related content."),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.7))
        assert cm.entries[0].status == "weak"
        assert cm.entries[0].confidence == 0.7

    def test_unsupported_via_contradiction(self):
        md = "DreamerV3 flies airplanes [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p1_0", paper_id="paper:1",
                     text="DreamerV3 is a robot control algorithm."),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("contradiction", 0.9))
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].confidence == 0.9


# ── Stage 2→3: LLM fallback in 0.4–0.6 band ───────────────────────

class TestLLMFallback:
    def test_llm_fallback_returns_supported(self):
        md = "Marginal claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        llm = FakeLLMClient(default_text="supported")
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.5), llm=llm)
        assert cm.entries[0].status == "supported"
        assert cm.entries[0].confidence == 0.5

    def test_llm_fallback_returns_weak(self):
        md = "Another marginal claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        llm = FakeLLMClient(default_text="weak")
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.45), llm=llm)
        assert cm.entries[0].status == "weak"
        assert cm.entries[0].confidence == 0.5

    def test_llm_fallback_returns_unsupported(self):
        md = "Bad marginal claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        llm = FakeLLMClient(default_text="unsupported")
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.4), llm=llm)
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].confidence == 0.5

    def test_fallback_band_no_llm(self):
        """0.4-0.6 band with llm=None → unsupported."""
        md = "Marginal claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.5), llm=None)
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].confidence == 0.5

    def test_below_04_unsupported(self):
        """Below 0.4 → unsupported regardless of llm."""
        md = "Very weak claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        llm = FakeLLMClient(default_text="supported")
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.3), llm=llm)
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].confidence == 0.3
