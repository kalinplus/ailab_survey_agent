"""Tests for tools.evaluate_survey — post-hoc survey evaluator (AutoSurvey-style)."""

import pytest

from tools.evaluate_survey import (
    extract_claim_pairs,
    citation_quality,
    ab_comparison,
    reference_coverage,
    llm_judge,
    render_md,
)
from tools.models.artifacts import EvidenceStore, Evidence
from tools.nlp.nli_verifier import NLIResult
from tools.clients.llm_fake import FakeLLMClient


class NLIStub:
    """Duck-typed NLI that always returns a fixed label/confidence."""

    def __init__(self, label, conf):
        self.label = label
        self.conf = conf

    def judge(self, premise, hypothesis):
        support_map = {
            "entailment": "direct",
            "neutral": "indirect",
            "contradiction": "contradictory",
        }
        return NLIResult(self.label, support_map[self.label], self.conf)


def _es(*evidences):
    return EvidenceStore(task_id="t", evidence=list(evidences))


# ── extract_claim_pairs: sentence × every citation ─────────────────

class TestExtractClaimPairs:
    def test_single_citation(self):
        md = "DreamerV3 flies airplanes [paper:1]."
        assert extract_claim_pairs(md) == [("DreamerV3 flies airplanes", "paper:1")]

    def test_multi_citation_yields_pair_per_citation(self):
        md = "Two refs here [paper:A] and [paper:B]."
        assert extract_claim_pairs(md) == [
            ("Two refs here  and", "paper:A"),
            ("Two refs here  and", "paper:B"),
        ]

    def test_no_citation_skipped(self):
        md = "No citation here. Next one has it [paper:2]."
        assert extract_claim_pairs(md) == [("Next one has it", "paper:2")]

    def test_chinese_period(self):
        md = "世界模型很重要 [paper:1]。"
        assert extract_claim_pairs(md) == [("世界模型很重要", "paper:1")]

    def test_heading_lines_excluded(self):
        md = "## Introduction\n\nSome claim [paper:1]."
        assert extract_claim_pairs(md) == [("Some claim", "paper:1")]

    def test_caption_brackets_are_not_citations(self):
        md = ("See the taxonomy figure [Distribution of selected papers across "
              "the survey taxonomy.] and a claim [paper:1].")
        assert extract_claim_pairs(md) == [("See the taxonomy figure  and a claim", "paper:1")]

    def test_non_paper_id_kept_when_in_known_ids(self):
        md = "Seed style claim [world_models_2018]."
        assert extract_claim_pairs(md, known_ids={"world_models_2018"}) == [
            ("Seed style claim", "world_models_2018")
        ]


# ── citation_quality: recall / precision / coverage ────────────────

class TestCitationQuality:
    def test_all_supported(self):
        md = "DreamerV3 masters many domains [paper:1]. Pure filler sentence without citation."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="we master 150 domains"))
        res = citation_quality(md, es, NLIStub("entailment", 0.85))
        assert res["n_pairs"] == 1
        assert res["supported_pairs"] == 1
        assert res["citation_recall"] == 1.0
        assert res["citation_precision"] == 1.0
        assert res["citation_coverage"] == 0.5  # 1 cited of 2 body sentences

    def test_precision_below_recall_with_extra_unsupported_citation(self):
        """Sentence cites A (supported) and B (no evidence): sentence recalled,
        but only half of citation instances are supported."""
        md = "Claim with two refs [paper:A] plus [paper:B]."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:A", text="evidence for claim"))
        res = citation_quality(md, es, NLIStub("entailment", 0.85))
        assert res["citation_recall"] == 1.0
        assert res["citation_precision"] == 0.5
        assert res["unsupported_pairs"] == 1

    def test_no_evidence_for_cited_paper_is_unsupported(self):
        md = "Unsupported claim [paper:X]."
        res = citation_quality(md, _es(), NLIStub("entailment", 0.9))
        assert res["unsupported_pairs"] == 1
        assert res["citation_recall"] == 0.0

    def test_neutral_is_weak(self):
        md = "Vague claim [paper:1]."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="loosely related"))
        res = citation_quality(md, es, NLIStub("neutral", 0.7))
        assert res["weak_pairs"] == 1
        assert res["citation_recall"] == 0.0

    def test_low_margin_entailment_still_supported(self):
        """Verdict follows the NLI label (AutoSurvey: entail or not); the
        cross-encoder margin is compressed and must not gate support."""
        md = "Shaky claim [paper:1]."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="somewhat related"))
        res = citation_quality(md, es, NLIStub("entailment", 0.5))
        assert res["supported_pairs"] == 1

    def test_best_evidence_wins(self):
        """Contradictory evidence first, entailing evidence second — the
        pair is judged by the best-matching evidence chunk."""
        md = "A claim [paper:1]."
        es = _es(
            Evidence(evidence_id="e1", paper_id="paper:1", text="the opposite"),
            Evidence(evidence_id="e2", paper_id="paper:1", text="the support"),
        )
        calls = []

        class TwoLabelStub:
            def judge(self, premise, hypothesis):
                calls.append(premise)
                label = "entailment" if premise == "the support" else "contradiction"
                return NLIResult(label, "direct" if label == "entailment" else "contradictory", 0.9)

        res = citation_quality(md, es, TwoLabelStub())
        assert res["supported_pairs"] == 1
        assert len(calls) == 2

    def test_contradiction_is_unsupported(self):
        md = "Wrong claim [paper:1]."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="the opposite"))
        res = citation_quality(md, es, NLIStub("contradiction", 0.9))
        assert res["unsupported_pairs"] == 1

    def test_pair_details_kept_for_report(self):
        md = "DreamerV3 masters many domains [paper:1]."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="we master 150 domains"))
        res = citation_quality(md, es, NLIStub("entailment", 0.85))
        assert res["pairs"][0] == {
            "claim_text": "DreamerV3 masters many domains",
            "cited_paper_id": "paper:1",
            "status": "supported",
            "confidence": 0.85,
        }

    def test_sentence_counts_ignore_structural_lines(self):
        md = (
            "## Section Title\n\n"
            "| col | col |\n"
            "![figure](x.png)\n\n"
            "Cited sentence [paper:1]. Uncited sentence. Another uncited one."
        )
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="evidence"))
        res = citation_quality(md, es, NLIStub("entailment", 0.85))
        assert res["n_sentences"] == 3
        assert res["n_cited_sentences"] == 1

    def test_caption_only_sentence_is_not_cited(self):
        md = "Figure see [Future Matrix]. Real claim [paper:1]."
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text="evidence"))
        res = citation_quality(md, es, NLIStub("entailment", 0.85))
        assert res["n_cited_sentences"] == 1
        assert res["n_pairs"] == 1

    def test_long_evidence_is_split_into_sentence_windows(self):
        """Cross-encoder NLI fails on long-passage premises (a verbatim quote
        inside a 1700-char abstract scores neutral); evidence units must be
        sentence windows so the supporting sentence can entail."""
        md = "GTBench is a language-driven environment [paper:1]."
        abstract = (
            "Large language models are increasingly used in games. "
            "This paper evaluates reasoning in competitive settings. "
            "We first propose GTBench, a language-driven environment comprising "
            "10 widely recognized game-theoretic tasks. "
            "Detailed error profiles are provided for better understanding."
        )
        es = _es(Evidence(evidence_id="e1", paper_id="paper:1", text=abstract))
        seen_units = []

        class UnitSpy:
            def judge(self, premise, hypothesis):
                seen_units.append(premise)
                label = "entailment" if premise.startswith("We first propose") else "neutral"
                return NLIResult(label, "direct" if label == "entailment" else "indirect", 0.9)

        res = citation_quality(md, es, UnitSpy())
        assert res["supported_pairs"] == 1
        assert len(seen_units) == 4  # abstract split into 4 sentence units
        assert all(len(u) < 200 for u in seen_units)


# ── ab_comparison: repair A/B rows → recall-equivalent table ───────

class TestAbComparison:
    def test_converts_rows(self):
        rows = [
            {"config": "A", "topic": "world models", "total_claims": 61,
             "unsupported": 53, "weak": 7, "invalid_citations": 84,
             "citation_validity_score": 0.472},
            {"config": "C", "topic": "world models", "total_claims": 55,
             "unsupported": 2, "weak": 3, "invalid_citations": 0,
             "citation_validity_score": 0.98},
        ]
        out = ab_comparison(rows)
        assert out[0]["citation_recall_equiv"] == 0.016  # (61-53-7)/61
        assert out[1]["citation_recall_equiv"] == 0.909  # (55-2-3)/55
        assert out[0]["config"] == "A"
        assert out[0]["invalid_citations"] == 84
        assert out[0]["citation_validity_score"] == 0.472

    def test_zero_claims_maps_to_zero(self):
        out = ab_comparison([{"config": "A", "total_claims": 0,
                              "unsupported": 0, "weak": 0}])
        assert out[0]["citation_recall_equiv"] == 0.0


# ── reference_coverage: gold refs vs system titles ─────────────────

class ScorerStub:
    def __init__(self, score):
        self.score = score

    def score_pairs(self, pairs):
        return [self.score] * len(pairs)


class TestReferenceCoverage:
    def _gold(self, *titles):
        return {"references": [{"title": t, "year": 2020} for t in titles]}

    def test_exact_match_normalized(self):
        res = reference_coverage(
            ["World  Models!", "DreamerV3: Mastering Diverse Domains"],
            self._gold("World Models", "DreamerV3: Mastering Diverse Domains"),
        )
        assert res["exact_matches"] == 2
        assert res["reference_recall"] == 1.0
        assert set(res["matched_titles"]) == {"World Models", "DreamerV3: Mastering Diverse Domains"}

    def test_substring_match_for_malformed_bib_entries(self):
        """Bib entries extracted as full text (authors+title+venue) must still
        match a system title embedded in them."""
        gold = {"references": [{
            "title": "Julian Schrittwieser et al. Muzero: Mastering atari, go, chess "
                     "and shogi by planning with a learned model. Nature, 2020.",
            "year": 2020, "malformed": True,
        }]}
        res = reference_coverage(["MuZero: Mastering Atari"], gold)
        assert res["substring_matches"] == 1
        assert res["reference_recall"] == 1.0

    def test_substring_match_rejected_for_clean_titles(self):
        """A short system title ('World models') must NOT substring-match clean
        gold titles that merely contain the phrase."""
        res = reference_coverage(
            ["World models"],
            self._gold("Daydreamer: World models for physical robot learning",
                       "Navigation world models"),
        )
        assert res["substring_matches"] == 0
        assert res["reference_recall"] == 0.0

    def test_partial_miss_counted(self):
        res = reference_coverage(["A Paper"], self._gold("A Paper", "Other Paper"))
        assert res["exact_matches"] == 1
        assert res["reference_recall"] == 0.5
        assert res["missed_count"] == 1

    def test_case_and_punctuation_insensitive(self):
        res = reference_coverage(
            ["GENIE - Generative Interactive Environments!"],
            self._gold("Genie: Generative Interactive Environments"),
        )
        assert res["reference_recall"] == 1.0

    def test_fuzzy_match_via_scorer(self):
        res = reference_coverage(
            ["genie generative interactive envirnments"],
            self._gold("Genie: Generative Interactive Environments"),
            scorer=ScorerStub(0.9),
        )
        assert res["fuzzy_matches"] == 1
        assert res["reference_recall"] == 1.0

    def test_fuzzy_below_threshold_not_matched(self):
        res = reference_coverage(
            ["something else"],
            self._gold("Genie: Generative Interactive Environments"),
            scorer=ScorerStub(0.7),
        )
        assert res["fuzzy_matches"] == 0
        assert res["reference_recall"] == 0.0

    def test_system_in_gold_rate(self):
        res = reference_coverage(["A Paper", "Not In Gold"], self._gold("A Paper"))
        assert res["n_system_titles"] == 2
        assert res["system_in_gold_rate"] == 0.5


# ── llm_judge: rubric scoring per section ──────────────────────────

JUDGE_MD = "## Intro\n\nFirst section body.\n\n## Methods\n\nSecond section body.\n"

ALL_DIMS = {"fluency": 4, "logic": 5, "redundancy": 3, "clarity": 4,
            "accuracy": 5, "coverage": 4, "relevance": 5}


class TestLLMJudge:
    def test_per_section_and_dimension_means(self):
        llm = FakeLLMClient(default_json={"scores": dict(ALL_DIMS)})
        res = llm_judge(JUDGE_MD, llm)
        assert res["n_sections"] == 2
        assert res["dimensions"]["fluency"] == 4.0
        assert res["overall"] == round(sum(ALL_DIMS.values()) / len(ALL_DIMS), 3)
        assert len(res["per_section"]) == 2
        assert res["per_section"][0]["section"] == "Intro"

    def test_one_call_per_section(self):
        llm = FakeLLMClient(default_json={"scores": dict(ALL_DIMS)})
        llm_judge(JUDGE_MD, llm)
        assert llm.calls == 2

    def test_missing_dimension_excluded_from_mean(self):
        llm = FakeLLMClient(default_json={"scores": {"fluency": 4}})
        res = llm_judge(JUDGE_MD, llm)
        assert res["dimensions"] == {"fluency": 4.0}
        assert res["overall"] == 4.0

    def test_garbage_response_raises(self):
        llm = FakeLLMClient(default_json={"unrelated": "shape"})
        with pytest.raises(ValueError):
            llm_judge(JUDGE_MD, llm)

    def test_body_without_heading_is_a_section(self):
        llm = FakeLLMClient(default_json={"scores": dict(ALL_DIMS)})
        res = llm_judge("No heading body here.", llm)
        assert res["n_sections"] == 1
        assert res["per_section"][0]["section"] == "(untitled)"


# ── render_md: human-readable report ───────────────────────────────

def _full_report():
    return {
        "citation_quality": {
            "n_sentences": 100, "n_cited_sentences": 60, "n_pairs": 80,
            "supported_pairs": 60, "weak_pairs": 10, "unsupported_pairs": 10,
            "citation_recall": 0.8, "citation_precision": 0.75, "citation_coverage": 0.6,
            "pairs": [{"claim_text": "bad claim", "cited_paper_id": "paper:X",
                       "status": "unsupported", "confidence": 0.1}],
        },
        "reference_coverage": {
            "gold_source_title": "Gold Survey", "n_gold_refs": 340,
            "n_system_titles": 12, "exact_matches": 16, "substring_matches": 1,
            "fuzzy_matches": 1,
            "reference_recall": 0.05, "missed_count": 322,
            "matched_titles": ["A", "B"], "system_in_gold_rate": 0.5,
        },
        "llm_judge": {
            "n_sections": 8, "dimensions": {"fluency": 4.0}, "overall": 4.2,
            "per_section": [],
        },
        "ab_comparison": [
            {"config": "A", "topic": "t", "total_claims": 61, "unsupported": 53, "weak": 7,
             "invalid_citations": 84, "citation_validity_score": 0.472,
             "citation_recall_equiv": 0.016},
            {"config": "C", "topic": "t", "total_claims": 55, "unsupported": 2, "weak": 3,
             "invalid_citations": 0, "citation_validity_score": 0.98,
             "citation_recall_equiv": 0.909},
        ],
        "meta": {"judge_model": "intern-s2-preview"},
    }


class TestRenderMd:
    def test_renders_three_layers_and_numbers(self):
        md = render_md(_full_report())
        assert "Citation recall" in md and "0.8" in md
        assert "Gold Survey" in md and "0.05" in md
        assert "LLM-as-Judge" in md and "4.2" in md

    def test_renders_ab_table_rows(self):
        md = render_md(_full_report())
        assert "| A |" in md and "| C |" in md and "0.016" in md

    def test_ab_table_shows_retention_columns(self):
        md = render_md(_full_report())
        assert "unsupported" in md and "invalid" in md
        assert "| A | t | 61 | 53 | 7 | 84 |" in md

    def test_optional_layers_omitted_gracefully(self):
        report = _full_report()
        del report["reference_coverage"]
        del report["llm_judge"]
        del report["ab_comparison"]
        md = render_md(report)
        assert "Citation recall" in md
        assert "Reference" not in md

    def test_unsupported_example_shown(self):
        md = render_md(_full_report())
        assert "bad claim" in md
