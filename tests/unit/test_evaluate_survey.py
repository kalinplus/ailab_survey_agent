"""Tests for tools.evaluate_survey — post-hoc four-layer survey evaluator."""

import json
import logging
import sys
from pathlib import Path

import pytest

from tools.evaluate_survey import (
    extract_claim_pairs,
    citation_quality,
    ab_comparison,
    reference_coverage,
    academic_value,
    corpus_coverage,
    deterministic_profile,
    overall_unsupported_rate,
    render_md,
    uncited_claims,
)
from tools.models.artifacts import Category, EvidenceStore, Evidence, Taxonomy
from tools.nlp.nli_verifier import FakeNLIModel, NLIResult
from tools.clients.llm_fake import FakeLLMClient
from harness.agents.relevance import EmbeddingScorer

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "scripts" / "run_survey_eval.py"


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


# ── fixtures shared by the four-layer tests ────────────────────────

SURVEY_MD = """# Survey Title

## Introduction

World models learn compact latent dynamics [paper:a]. Agents can plan inside the learned latent space [paper:b].

![Taxonomy overview](figures/taxonomy.png)

| Method | Year |
|---|---|
| Dreamer | 2023 |

## Methods

DreamerV3 trains a world model from pixels [paper:c].

## Discussion

Open challenges remain for evaluation protocols.

## References

- paper:a: Alpha World Models (2024).
- paper:b: Beta Planning Agents (2026).
- paper:c: Gamma Dreamer (2018).
"""

PAPERS = [
    {"paper_id": "paper:a", "title": "Alpha World Models", "year": 2024},
    {"paper_id": "paper:b", "title": "Beta Planning Agents", "year": 2026},
    {"paper_id": "paper:c", "title": "Gamma Dreamer", "year": 2018},
    {"paper_id": "paper:d", "title": "Delta Robots", "year": 2025},
]
FIGURES = [{"figure_id": "f1"}, {"figure_id": "f2"}]
TABLES = [{"table_id": "t1"}]

TAXONOMY = Taxonomy(task_id="t", topic="world models", categories=[
    Category(category_id="c1", category_name="Latent dynamics",
             description="compact latent world models", paper_ids=["paper:a", "paper:c"]),
    Category(category_id="c2", category_name="Robot learning",
             description="robot manipulation policies", paper_ids=["paper:d"]),
])


class VecScorer:
    """EmbeddingScorer through its _embed test seam: controlled vectors, no model."""

    mode = "embedding"

    def __init__(self):
        self._scorer = EmbeddingScorer(_embed=self._embed)

    @staticmethod
    def _embed(texts):
        return [[1.0, 0.0] if ("latent dynamics" in t or "plan inside" in t) else [0.0, 1.0]
                for t in texts]

    def score_pairs(self, pairs):
        return self._scorer.score_pairs(list(pairs))


class PairFnScorer:
    """Constant per-(aspect, section) scores handed to corpus_coverage."""

    mode = "embedding"

    def __init__(self, fn):
        self.fn = fn

    def score_pairs(self, pairs):
        return [self.fn(a, b) for a, b in pairs]


# ── L0 deterministic profile ───────────────────────────────────────


class TestDeterministicProfile:
    def _profile(self, md=SURVEY_MD, scorer=None):
        return deterministic_profile(md, PAPERS, FIGURES, TABLES,
                                     scorer=scorer or VecScorer(), current_year=2026)

    def test_freshness_medians_offset_and_recent_shares(self):
        fr = self._profile()["freshness"]
        # cited years {2018, 2024, 2026} -> median 2024; corpus {2018,2024,2025,2026} -> 2024.5
        assert fr["citation_year_median"] == 2024.0
        assert fr["corpus_year_median"] == 2024.5
        assert fr["year_offset"] == -0.5
        # recent window is [current_year - 1, current_year] = [2025, 2026]
        assert fr["citation_recent_share"] == 0.333   # 1 of 3
        assert fr["corpus_recent_share"] == 0.5       # 2 of 4
        assert fr["n_cited_ids"] == 3
        assert fr["years_unresolved"] == []

    def test_freshness_unresolved_citation_year_excluded(self):
        # appended into a body section: a References-section line would not count as cited
        md = SURVEY_MD.replace("Open challenges remain for evaluation protocols.",
                               "Open challenges remain for evaluation protocols. "
                               "Dangling claim [paper:zzz].")
        fr = self._profile(md)["freshness"]
        assert fr["years_unresolved"] == ["paper:zzz"]
        assert fr["n_cited_ids"] == 4
        assert fr["citation_year_median"] == 2024.0  # unresolved dropped from the median

    def test_structure_balance_density_and_banks(self):
        st = self._profile()["structure"]
        assert st["n_sections"] == 3  # References section excluded
        assert st["length_cv"] == 0.501  # stdev/mean over prose chars [112, 53, 48]
        assert st["zero_citation_sections"] == ["Discussion"]
        assert st["n_citations"] == 3
        assert st["mean_citations_per_section"] == 1.0
        intro = st["sections"][0]
        assert intro["prose_chars"] == 112
        assert intro["n_citations"] == 2
        assert intro["citations_per_1000_chars"] == 17.857  # 1000 * 2 / 112
        assert st["figure_refs_in_body"] == 1
        assert st["table_refs_in_body"] == 1
        assert st["figure_bank_size"] == 2 and st["table_bank_size"] == 1
        assert st["figure_usage"] == 0.5
        assert st["table_usage"] == 1.0

    def test_redundancy_off_diagonal_exact_values(self):
        """Controlled vectors: Introduction ⊥ Methods/Discussion, the latter two
        identical -> cosines 0, 0, 1 -> mean 1/3, max 1.0."""
        rd = self._profile()["redundancy"]
        assert rd["n_section_pairs"] == 3
        assert rd["mean_similarity"] == 0.333
        assert rd["max_similarity"] == 1.0
        assert rd["top_pairs"][0] == {"a": "Methods", "b": "Discussion", "similarity": 1.0}

    def test_reference_lines_are_not_body_sections(self):
        profile = self._profile()
        assert "References" not in [s["section"] for s in profile["structure"]["sections"]]


# ── L2' corpus-grounded coverage ───────────────────────────────────


def _coverage_fn(aspect, section):
    if "Latent" in aspect and "latent dynamics" in section:
        return 0.8
    return 0.2


class TestCorpusCoverage:
    def test_coverage_depth_and_utilization(self):
        res = corpus_coverage(SURVEY_MD, TAXONOMY.categories, PAPERS,
                              scorer=PairFnScorer(_coverage_fn), cov_threshold=0.5)
        assert res["cov_threshold"] == 0.5
        assert res["n_covered"] == 1 and res["n_categories"] == 2
        assert res["category_coverage_rate"] == 0.5
        # covered: c1 (cites both of its papers) -> depth 1.0; c2 cites none of its 1 paper
        assert res["categories"][0]["citation_depth"] == 1.0
        assert res["categories"][0]["best_similarity"] == 0.8
        assert res["categories"][1]["citation_depth"] == 0.0
        assert res["weighted_citation_depth"] == 0.667  # (1.0*2 + 0.0*1) / 3
        assert res["corpus_utilization"] == 0.75        # cited {a,b,c} of 4 corpus papers
        assert res["n_cited_ids"] == 3
        assert res["n_cited_ids_in_corpus"] == 3
        assert res["id_space_mismatch"] is False
        assert res["missed_categories"] == ["Robot learning"]

    def test_disjoint_id_spaces_flagged_and_logged(self, caplog):
        """survey cites paper:<doi>, taxonomy/paper_cards use seed ids: the zeros
        must be labelled as an artifact mismatch instead of reading as a verdict."""
        papers = [{"paper_id": "seed_1", "title": "Seed One", "year": 2024}]
        taxonomy = Taxonomy(task_id="t", topic="world models", categories=[
            Category(category_id="c1", category_name="Latent dynamics",
                     description="compact latent world models", paper_ids=["seed_1"]),
        ])
        with caplog.at_level(logging.WARNING, logger="tools.evaluate_survey"):
            res = corpus_coverage(SURVEY_MD, taxonomy.categories, papers,
                                  scorer=PairFnScorer(_coverage_fn))
        assert res["id_space_mismatch"] is True
        assert res["n_cited_ids_in_corpus"] == 0
        assert res["corpus_utilization"] == 0.0
        assert res["weighted_citation_depth"] == 0.0
        assert "id-space mismatch" in caplog.text
        assert "'a'" in caplog.text and "'seed_1'" in caplog.text  # one sample per side
        note = render_md({**_full_report(), "corpus_coverage": res})
        assert "artifact id-space mismatch" in note

    def test_env_threshold_gates_coverage(self, monkeypatch):
        scorer = PairFnScorer(_coverage_fn)
        monkeypatch.setenv("EVISURVEY_EVAL_COV_MIN", "0.9")
        strict = corpus_coverage(SURVEY_MD, TAXONOMY.categories, PAPERS, scorer=scorer)
        assert strict["cov_threshold"] == 0.9
        assert strict["category_coverage_rate"] == 0.0
        monkeypatch.setenv("EVISURVEY_EVAL_COV_MIN", "0.5")
        loose = corpus_coverage(SURVEY_MD, TAXONOMY.categories, PAPERS, scorer=scorer)
        assert loose["category_coverage_rate"] == 0.5
        # explicit flag beats the env
        flag = corpus_coverage(SURVEY_MD, TAXONOMY.categories, PAPERS, scorer=scorer, cov_threshold=0.95)
        assert flag["cov_threshold"] == 0.95 and flag["n_covered"] == 0


# ── L1.5 uncited claims ────────────────────────────────────────────

UNCITED_SURVEY = """# T

## Intro

Diffusion policies scale well to new games. Quantum teleportation speeds up training. Cited fact here [paper:a].

## References

- paper:a: Alpha (2024).

## Revision Notes

- type=A delete_claim -> deleted: appendix junk that must not be searched.
"""

UNCITED_STORE = EvidenceStore(task_id="t", evidence=[
    Evidence(evidence_id="e1", paper_id="paper:a", text="alpha evidence")])


class SciverseStub:
    def __init__(self):
        self.calls = []

    def agentic_search(self, query, top_k=10, filters=None):
        self.calls.append((query, top_k))
        if "Diffusion" in query:
            return {"hits": [{"chunk": "Diffusion policies generalize across many game genres. "
                                        "They also train fast."}]}
        return {"hits": []}


class TestUncitedClaims:
    def _run(self, stub, cache_path=None, max_claims=5):
        nli = FakeNLIModel({"diffusion": "entailment"})
        return uncited_claims(UNCITED_SURVEY, UNCITED_STORE, nli, stub,
                              cache_path=cache_path, max_claims=max_claims)

    def test_statuses_and_budget_accounting(self):
        stub = SciverseStub()
        res = self._run(stub)
        # References + Revision Notes lines are not body claims
        assert res["n_uncited_total"] == 2
        assert res["n_truncated"] == 0
        assert res["supported"] == 1 and res["unsupported"] == 1
        assert res["uncited_support_rate"] == 0.5
        assert res["claims"][0] == {"claim_text": "Diffusion policies scale well to new games",
                                    "status": "supported", "confidence": 0.9,
                                    "from_cache": False, "n_hits": 1}
        assert res["claims"][1]["status"] == "unsupported"  # zero hits -> unsupported
        assert stub.calls == [("Diffusion policies scale well to new games", 3),
                              ("Quantum teleportation speeds up training", 3)]

    def test_budget_truncates_and_counts(self):
        stub = SciverseStub()
        res = self._run(stub, max_claims=1)
        assert res["n_evaluated"] == 1
        assert res["n_truncated"] == 1
        assert res["n_uncited_total"] == 2
        assert len(stub.calls) == 1

    def test_cache_resume_issues_no_second_search(self, tmp_path):
        cache = tmp_path / "uncited_claim_cache.json"
        stub = SciverseStub()
        self._run(stub, cache_path=cache)
        assert len(stub.calls) == 2
        # second run over the same fixture: every claim hits the cache
        stub2 = SciverseStub()
        res = self._run(stub2, cache_path=cache)
        assert stub2.calls == []  # zero new retrieval calls
        assert res["n_cache_hits"] == 2 and res["n_searched"] == 0
        assert res["uncited_support_rate"] == 0.5
        assert all(c["from_cache"] for c in res["claims"])
        stored = json.loads(cache.read_text(encoding="utf-8"))
        assert set(stored) == {"Diffusion policies scale well to new games",
                               "Quantum teleportation speeds up training"}

    def test_headline_overall_unsupported_rate(self):
        stub = SciverseStub()
        uc = self._run(stub)
        cq = citation_quality(UNCITED_SURVEY, UNCITED_STORE, NLIStub("contradiction", 0.9))
        assert cq["n_pairs"] == 1 and cq["unsupported_pairs"] == 1
        assert overall_unsupported_rate(cq, uc) == 0.667  # (1 + 1) / (1 + 2)
        assert overall_unsupported_rate(citation_quality("No claims at all.", UNCITED_STORE,
                                                         NLIStub("contradiction", 0.9)), None) is None


# ── academic_value: DeepSurvey three-dimension judge ───────────────


class TestAcademicValue:
    def test_three_whole_doc_calls_with_exact_scores(self):
        llm = FakeLLMClient(responses=[
            ("informational_value", {"score": 5, "rationale": "dense and precise"}),
            ("scholarly_communication_value", {"score": 3, "rationale": "readable, gaps"}),
            ("research_guidance_value", {"score": 4, "rationale": "some directions"}),
        ])
        res = academic_value("## A\n\nBody.\n\n## B\n\nMore body.\n", llm)
        assert llm.calls == 3  # one whole-doc call per dimension, not per section
        assert res["n_calls"] == 3
        assert res["dimensions"]["informational_value"] == {"score": 5, "rationale": "dense and precise"}
        assert res["dimensions"]["scholarly_communication_value"]["score"] == 3
        assert res["dimensions"]["research_guidance_value"]["score"] == 4
        assert res["overall"] == 4.0

    def test_string_score_coerced(self):
        llm = FakeLLMClient(default_json={"score": "4", "rationale": "r"})
        res = academic_value("body", llm)
        assert res["overall"] == 4.0

    def test_out_of_range_score_raises(self):
        llm = FakeLLMClient(default_json={"score": 7, "rationale": "r"})
        with pytest.raises(ValueError):
            academic_value("body", llm)

    def test_missing_rationale_raises(self):
        llm = FakeLLMClient(default_json={"score": 4})
        with pytest.raises(ValueError):
            academic_value("body", llm)

    def test_garbage_response_raises(self):
        llm = FakeLLMClient(default_json={"unrelated": "shape"})
        with pytest.raises(ValueError):
            academic_value("body", llm)


# ── render_md: human-readable four-block report ────────────────────


def _uncited_report_stub():
    return {
        "max_claims": 60, "n_uncited_total": 4, "n_evaluated": 3, "n_searched": 3,
        "n_cache_hits": 0, "n_truncated": 1, "supported": 1, "weak": 1, "unsupported": 1,
        "uncited_support_rate": 0.333,
        "claims": [{"claim_text": "rogue uncited sentence", "status": "unsupported",
                    "confidence": 0.0, "from_cache": False, "n_hits": 0}],
    }


def _full_report():
    return {
        "deterministic_profile": deterministic_profile(
            SURVEY_MD, PAPERS, FIGURES, TABLES, scorer=VecScorer(), current_year=2026),
        "citation_quality": {
            "n_sentences": 100, "n_cited_sentences": 60, "n_pairs": 80,
            "supported_pairs": 60, "weak_pairs": 10, "unsupported_pairs": 10,
            "citation_recall": 0.8, "citation_precision": 0.75, "citation_coverage": 0.6,
            "pairs": [{"claim_text": "bad claim", "cited_paper_id": "paper:X",
                       "status": "unsupported", "confidence": 0.1}],
        },
        "uncited_claims": _uncited_report_stub(),
        "overall_unsupported_rate": 0.375,
        "corpus_coverage": {
            "cov_threshold": 0.5, "n_categories": 2, "n_covered": 1,
            "category_coverage_rate": 0.5, "weighted_citation_depth": 0.62,
            "mean_citation_depth": 0.5, "corpus_utilization": 0.4,
            "missed_categories": ["Robot learning"],
            "categories": [
                {"category_id": "c1", "category_name": "Latent dynamics", "n_papers": 3,
                 "best_similarity": 0.71, "best_section": "Introduction", "covered": True,
                 "citation_depth": 0.667},
                {"category_id": "c2", "category_name": "Robot learning", "n_papers": 1,
                 "best_similarity": 0.31, "best_section": "Methods", "covered": False,
                 "citation_depth": 0.0},
            ],
        },
        "reference_coverage": {
            "gold_source_title": "Gold Survey", "n_gold_refs": 340,
            "n_system_titles": 12, "exact_matches": 16, "substring_matches": 1,
            "fuzzy_matches": 1,
            "reference_recall": 0.05, "missed_count": 322,
            "matched_titles": ["A", "B"], "system_in_gold_rate": 0.5,
        },
        "academic_value": {
            "n_calls": 3,
            "dimensions": {
                "informational_value": {"score": 5, "rationale": "dense"},
                "scholarly_communication_value": {"score": 4, "rationale": "clear"},
                "research_guidance_value": {"score": 3.6, "rationale": "some gaps"},
            },
            "overall": 4.2,
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
    def test_four_blocks_in_spec_order(self):
        md = render_md(_full_report())
        assert md.index("Deterministic Profile") < md.index("Faithfulness")
        assert md.index("Faithfulness") < md.index("Coverage")
        assert md.index("Coverage") < md.index("Academic Value")
        assert md.index("Academic Value") < md.index("A/B — repair joint")

    def test_renders_all_layers_and_numbers(self):
        md = render_md(_full_report())
        assert "Citation recall" in md and "0.8" in md
        assert "Gold Survey" in md and "0.05" in md
        assert "4.2" in md and "informational_value" in md
        assert "citation year median 2024.0" in md
        assert "Corpus-grounded coverage **0.5**" in md

    def test_headline_and_uncited_summary(self):
        md = render_md(_full_report())
        assert "overall unsupported rate **0.375**" in md
        assert "4 found, 3 verified" in md
        assert "support rate **0.333**" in md
        assert "rogue uncited sentence" in md

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
        del report["academic_value"]
        del report["ab_comparison"]
        md = render_md(report)
        assert "Citation recall" in md
        assert "Reference" not in md

    def test_unsupported_example_shown(self):
        md = render_md(_full_report())
        assert "bad claim" in md


# ── scripts/run_survey_eval.py wiring ──────────────────────────────


def _kw_scorer():
    scorer = EmbeddingScorer()
    scorer._failed = True  # keyword column: deterministic, no model download
    return scorer


def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_survey_eval_under_test", EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_artifacts(tmp_path: Path) -> dict:
    (tmp_path / "survey.md").write_text(UNCITED_SURVEY, encoding="utf-8")
    (tmp_path / "evidence_store.json").write_text(json.dumps({
        "task_id": "t",
        "evidence": [{"evidence_id": "e1", "paper_id": "paper:a", "text": "alpha evidence"}],
    }), encoding="utf-8")
    (tmp_path / "papers.json").write_text(
        json.dumps({"task_id": "t", "paper_cards": PAPERS}), encoding="utf-8")
    (tmp_path / "taxonomy.json").write_text(
        json.dumps(TAXONOMY.model_dump()), encoding="utf-8")
    (tmp_path / "figure_bank.json").write_text(
        json.dumps({"task_id": "t", "figures": FIGURES}), encoding="utf-8")
    (tmp_path / "table_bank.json").write_text(
        json.dumps({"task_id": "t", "tables": TABLES}), encoding="utf-8")
    return {"survey": tmp_path / "survey.md", "evidence": tmp_path / "evidence_store.json",
            "papers": tmp_path / "papers.json", "taxonomy": tmp_path / "taxonomy.json",
            "figures": tmp_path / "figure_bank.json", "tables": tmp_path / "table_bank.json"}


def _argv(paths, out, extra):
    return [
        "run_survey_eval.py",
        "--survey", str(paths["survey"]),
        "--evidence-store", str(paths["evidence"]),
        "--papers", str(paths["papers"]),
        "--taxonomy", str(paths["taxonomy"]),
        "--figure-bank", str(paths["figures"]),
        "--table-bank", str(paths["tables"]),
        "--gold-refs", str(paths["survey"].parent / "missing_gold.json"),
        "--ab-report", str(paths["survey"].parent / "missing_ab.json"),
        "--out", str(out),
    ] + extra


class TestRunSurveyEvalScript:
    def test_skip_uncited_never_builds_sciverse(self, tmp_path, monkeypatch):
        paths = _write_artifacts(tmp_path)
        monkeypatch.delenv("EVISURVEY_REAL_NLI", raising=False)

        class _PoisonedModule:
            """Any import/construction of the sciverse client blows up the run."""

            def __getattr__(self, name):
                raise AssertionError(f"sciverse client touched under --skip-uncited: {name}")

        monkeypatch.setitem(sys.modules, "tools.clients.sciverse_client", _PoisonedModule())
        mod = _load_script()
        monkeypatch.setattr(mod, "_embed_scorer", _kw_scorer)
        monkeypatch.setattr(sys, "argv", _argv(paths, tmp_path / "report",
                                               ["--skip-judge", "--skip-uncited"]))
        mod.main()
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert "uncited_claims" not in report
        # headline without the uncited layer = cited pairs only (0 unsupported / 1 weak pair)
        assert report["overall_unsupported_rate"] == 0.0
        assert report["deterministic_profile"]["freshness"]["citation_year_median"] == 2024.0
        assert "Deterministic Profile" in (tmp_path / "report.md").read_text(encoding="utf-8")

    def test_uncited_layer_runs_on_stub_and_resumes_from_cache(self, tmp_path, monkeypatch):
        paths = _write_artifacts(tmp_path)
        mod = _load_script()
        monkeypatch.delenv("EVISURVEY_REAL_NLI", raising=False)
        monkeypatch.setattr(mod, "_embed_scorer", _kw_scorer)
        stub = SciverseStub()
        monkeypatch.setattr(mod, "_make_sciverse", lambda: stub)
        monkeypatch.setattr(sys, "argv", _argv(paths, tmp_path / "report", [
            "--skip-judge", "--uncited-cache", str(tmp_path / "cache.json"),
            "--max-uncited-claims", "5"]))
        mod.main()
        searches_after_first_run = len(stub.calls)
        assert searches_after_first_run == 2
        mod.main()  # same artifacts + same cache -> resume, zero new retrieval
        assert len(stub.calls) == searches_after_first_run
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["uncited_claims"]["n_cache_hits"] == 2
        assert report["uncited_claims"]["n_searched"] == 0

    def test_help_lists_new_flags(self):
        import subprocess
        proc = subprocess.run([sys.executable, str(EVAL_SCRIPT), "--help"],
                              capture_output=True, text=True, cwd=str(ROOT))
        assert proc.returncode == 0
        for flag in ("--skip-uncited", "--max-uncited-claims", "--cov-threshold", "--skip-judge"):
            assert flag in proc.stdout
