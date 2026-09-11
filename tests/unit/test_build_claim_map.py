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

    def test_image_embed_line_ignored(self):
        md = (
            "![Publication years of the selected representative papers.](publication_timeline)\n\n"
            "These models support planning [paper:1]."
        )
        result = extract_claims_with_citations(md)
        assert result == [("These models support planning", "paper:1")]

    def test_embed_caption_not_extracted_as_citation(self):
        md = "Timeline follows.\n![Figure 1. Overview.](hero_banner)\nAnd a claim [paper:2]."
        result = extract_claims_with_citations(md)
        assert ("Timeline follows", "Figure 1. Overview.") not in result
        assert result == [("And a claim", "paper:2")]


# ── shared-kernel unification (tools/verify/text_units.py) ────────

class TestKernelUnification:
    def test_math_interval_is_not_a_citation(self):
        """Brackets that are neither paper:-prefixed nor known ids — math
        intervals, figure refs — must not be mistaken for citations."""
        md = "The reward lies in [0,1] for all tasks [paper:1]."
        assert extract_claims_with_citations(md) == [
            ("The reward lies in  for all tasks", "paper:1")
        ]

    def test_bracket_without_valid_id_not_in_claim_map(self):
        """A sentence whose only bracket is not a valid id produces no entry;
        a fake bracket next to a real citation leaves the binding intact."""
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="e1", paper_id="paper:1", text="evidence")])
        cm = run("t", "See the overview [Future Matrix].", es, _empty_parsed(),
                 NLIStub("entailment", 0.85))
        assert cm.entries == []
        cm = run("t", "See the overview [Future Matrix] and a claim [paper:1].", es,
                 _empty_parsed(), NLIStub("entailment", 0.85))
        assert [e.cited_paper_id for e in cm.entries] == ["paper:1"]
        assert cm.entries[0].claim_text == "See the overview  and a claim"

    def test_seed_style_id_bound_when_in_store(self):
        """Non-paper: ids still bind when the evidence store knows them."""
        md = "World models learn dynamics [world_models_2018]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="e1", paper_id="world_models_2018", text="evidence")])
        cm = run("t", md, es, _empty_parsed(), NLIStub("entailment", 0.85))
        assert cm.entries[0].cited_paper_id == "world_models_2018"

    def test_trailing_citation_fragment_merged(self):
        """'... tag. [paper:x].' — the citation-only fragment merges back so
        the claim is kept and bound instead of dropped."""
        md = "Reward scale improves sharply. [paper:1]."
        assert extract_claims_with_citations(md) == [
            ("Reward scale improves sharply", "paper:1")
        ]

    def test_heading_line_excluded(self):
        md = "# Title mentions [paper:9]\n\nReal claim [paper:1]."
        assert extract_claims_with_citations(md) == [("Real claim", "paper:1")]

    def test_evidence_windowed_before_nli(self):
        """Long evidence reaches best_match as sentence windows (same
        evidence_units as the evaluator), not as one long-passage blob."""
        md = "GTBench is a language-driven environment [paper:1]."
        abstract = (
            "Large language models are increasingly used in games and game theory research. "
            "This paper evaluates reasoning in competitive settings across many titles. "
            "We first propose GTBench, a language-driven environment comprising "
            "10 widely recognized game-theoretic tasks. "
            "Detailed error profiles are provided for better understanding of failures."
        )
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="e1", paper_id="paper:1", text=abstract)])
        seen = []

        class WindowSpy:
            def best_match(self, claim, evidences):
                seen.append(list(evidences))
                return NLIResult("entailment", "direct", 0.85)

        cm = run("t", md, es, _empty_parsed(), WindowSpy())
        assert cm.entries[0].status == "supported"
        assert seen == [[
            "Large language models are increasingly used in games and game theory research.",
            "This paper evaluates reasoning in competitive settings across many titles.",
            "We first propose GTBench, a language-driven environment comprising "
            "10 widely recognized game-theoretic tasks.",
            "Detailed error profiles are provided for better understanding of failures.",
        ]]


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
        """Stage 1 fires only when THIS claim's text (normalized) is directly
        bound in an evidence's supports_claims; eids list the matched rows only."""
        md = "DreamerV3 masters many domains [paper:1]."
        e1 = Evidence(
            evidence_id="paper:1_p1_0", paper_id="paper:1",
            text="We evaluate on 150 domains.",
            supports_claims=[{"claim_text": "DreamerV3 masters many domains!", "support_type": "direct"}],
        )
        e2 = Evidence(
            evidence_id="paper:1_p1_1", paper_id="paper:1",
            text="Another paragraph.",
            supports_claims=[{"claim_text": "something else", "support_type": "direct"},
                             {"claim_text": "DreamerV3 masters many domains", "support_type": "indirect"}],
        )
        es = EvidenceStore(task_id="t", evidence=[e1, e2])
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.5))
        # Legacy direct labels carry no verified role; always rerun NLI.
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].nli_status == "weak"
        assert cm.entries[0].source_role_violation == "unknown_source_role"
        assert cm.entries[0].confidence == 0.5

    def test_direct_support_is_per_claim(self):
        """Claim A directly bound does not make unrelated claim B supported via
        stage 1; B falls through to NLI (stage 2) with the paper's evidence."""
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(
                evidence_id="paper:1_p1_0", paper_id="paper:1",
                text="We evaluate on 150 domains.",
                supports_claims=[{"claim_text": "claim alpha text", "support_type": "direct"}],
            ),
        ])
        cm = run("t", "Claim beta text entirely different [paper:1].", es,
                 _empty_parsed(), NLIStub("contradiction", 0.9))
        assert cm.entries[0].status == "unsupported"
        assert cm.entries[0].confidence != 1.0  # decided by NLI, not stage 1
        assert cm.entries[0].confidence == 0.9


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

class TestPureLabelVerdicts:
    """Wave 5 (3): stage-2 verdicts follow the NLI label alone — the evaluator
    L1 convention. The old 0.4-0.6 confidence band + LLM fallback was removed:
    its gate sat below the margin-compression floor (identical text ~0.56) and
    downgraded verbatim-quote claims to unsupported."""

    def test_entailment_label_supported_even_below_old_gate(self):
        md = "Verbatim quote claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("entailment", 0.56), llm=None)
        assert cm.entries[0].status == "supported"  # 0.56 < old 0.6 gate
        assert cm.entries[0].confidence == 0.56

    def test_neutral_label_weak_even_without_llm(self):
        md = "Marginal claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("neutral", 0.5), llm=None)
        assert cm.entries[0].status == "weak"

    def test_contradiction_label_unsupported(self):
        md = "Opposite claim [paper:1]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="paper:1_p0_0", paper_id="paper:1",
                     text="some evidence text"),
        ])
        cm = run("t", md, es, _empty_parsed(), NLIStub("contradiction", 0.3), llm=None)
        assert cm.entries[0].status == "unsupported"


# ── wave 1B: pair-level expansion + structured-claims freshness gate ──────────


class TestPairExpansion:
    def test_multi_citation_sentence_yields_one_entry_per_paper(self):
        """A sentence citing [A, B] becomes TWO entries (the evaluator's L1
        pair convention); the pair whose paper has zero evidence is unsupported."""
        md = "One sentence cites two papers at once [paper:A] [paper:B]."
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="eA", paper_id="paper:A", text="evidence for A")])
        cm = run("t", md, es, _empty_parsed(), NLIStub("entailment", 0.85))
        assert [(e.cited_paper_id, e.status) for e in cm.entries] == [
            ("paper:A", "supported"), ("paper:B", "unsupported")]
        assert cm.entries[0].claim_text == cm.entries[1].claim_text
        assert cm.entries[0].confidence == 0.85
        assert cm.entries[1].confidence == 0.0

    def test_structured_claim_with_multiple_cites_expands_to_pairs(self):
        structured = [{"text": "Claim spanning two studies", "cites": ["paper:A", "paper:B"]}]
        es = EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id="eA", paper_id="paper:A", text="evidence for A")])
        cm = run("t", "Claim spanning two studies [paper:A] [paper:B].", es,
                 _empty_parsed(), NLIStub("entailment", 0.85), structured_claims=structured)
        assert [(e.cited_paper_id, e.status) for e in cm.entries] == [
            ("paper:A", "supported"), ("paper:B", "unsupported")]


class TestStructuredClaimsFreshness:
    def _store(self):
        return EvidenceStore(task_id="t", evidence=[
            Evidence(evidence_id=f"e{pid}", paper_id=pid, text=f"evidence of {pid}")
            for pid in ("paper:1", "paper:2", "paper:3")])

    def test_stale_claims_drop_and_kernel_covers_the_rest(self):
        """3 structured claims, markdown keeps 2 verbatim + 1 repair-rewritten
        sentence + 1 freeform cited sentence -> 4 units; the deleted claim is
        gone; every legal citation id in the markdown is covered."""
        structured = [
            {"text": "Claim alpha text", "cites": ["paper:1"]},
            {"text": "Claim beta text", "cites": ["paper:2"]},
            {"text": "Claim gone text", "cites": ["paper:1"]},  # deleted by repair
        ]
        md = (
            "Claim alpha text [paper:1]. "
            "Claim beta text [paper:2]. "
            "Rewritten by repair into a weaker sentence [paper:2]. "
            "A freeform abstract sentence never in the list [paper:3]."
        )
        cm = run("t", md, self._store(), _empty_parsed(), NLIStub("entailment", 0.85),
                 structured_claims=structured)
        # 2 verbatim structured + rewritten kernel unit + freeform kernel unit
        assert [e.claim_text for e in cm.entries] == [
            "Claim alpha text",
            "Claim beta text",
            "Rewritten by repair into a weaker sentence",
            "A freeform abstract sentence never in the list",
        ]
        # the deleted structured claim is never verified again
        assert "Claim gone text" not in [e.claim_text for e in cm.entries]
        # coverage invariant: every legal citation id appears in >= 1 unit
        covered = {e.cited_paper_id for e in cm.entries}
        assert {"paper:1", "paper:2", "paper:3"} <= covered
        assert all(e.status == "supported" for e in cm.entries)

    def test_structured_claim_survives_only_when_text_in_markdown(self):
        """Normalized text match decides freshness: minor punctuation drift in
        the markdown still counts as present; a paraphrase does not."""
        structured = [{"text": "Verbatim claim, with punctuation", "cites": ["paper:1"]}]
        md = "Verbatim claim with punctuation [paper:1]."
        cm = run("t", md, self._store(), _empty_parsed(), NLIStub("entailment", 0.85),
                 structured_claims=structured)
        assert len(cm.entries) == 1  # present (punctuation-insensitive)

        md_paraphrased = "A quite different rendering of the idea [paper:1]."
        cm = run("t", md_paraphrased, self._store(), _empty_parsed(),
                 NLIStub("entailment", 0.85), structured_claims=structured)
        # stale structured entry dropped; the kernel unit for the actual
        # sentence is still verified
        assert [e.claim_text for e in cm.entries] == ["A quite different rendering of the idea"]

    def test_kernel_covers_citations_outside_structured_sections(self):
        """Abstract/intro/closing freeform cited sentences enter the map even
        when a structured list exists (the old code never verified them)."""
        structured = [{"text": "Body claim text", "cites": ["paper:1"]}]
        md = (
            "Body claim text [paper:1].\n\n"
            "## Conclusion\n\n"
            "Closing freeform sentence cites a study [paper:2]."
        )
        cm = run("t", md, self._store(), _empty_parsed(), NLIStub("entailment", 0.85),
                 structured_claims=structured)
        assert [e.claim_text for e in cm.entries] == [
            "Body claim text", "Closing freeform sentence cites a study"]
