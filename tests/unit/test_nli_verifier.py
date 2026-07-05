import os

import pytest

from tools.nlp.nli_verifier import NLIResult, NLIVerifier, FakeNLIModel, _support


# --- NLIResult.status ---

class TestNLIResultStatus:
    def test_supported(self):
        r = NLIResult("entailment", "direct", 0.9)
        assert r.status == "supported"

    def test_unsupported(self):
        assert NLIResult("contradiction", "contradictory", 0.2).status == "unsupported"

    def test_weak_for_neutral(self):
        assert NLIResult("neutral", "indirect", 0.5).status == "weak"

    def test_boundary_supported_at_0_6(self):
        """entailment with confidence exactly 0.6 -> supported"""
        assert NLIResult("entailment", "direct", 0.6).status == "supported"

    def test_boundary_weak_at_0_59(self):
        """entailment with confidence 0.59 -> weak (< 0.6 threshold)"""
        assert NLIResult("entailment", "direct", 0.59).status == "weak"


# --- _support ---

class TestSupport:
    def test_entailment_direct(self):
        st, conf = _support("entailment", 0.1, 0.9, 0.2)
        assert st == "direct"
        assert conf > 0

    def test_neutral_indirect(self):
        st, conf = _support("neutral", 0.1, 0.3, 0.6)
        assert st == "indirect"

    def test_contradiction(self):
        st, conf = _support("contradiction", 0.8, 0.1, 0.1)
        assert st == "contradictory"
        assert conf == 0.2

    def test_fallback_indirect(self):
        """entailment but e not > c or e not > n -> fallback"""
        st, conf = _support("entailment", 0.9, 0.1, 0.2)
        assert st == "indirect"
        assert conf == 0.2


# --- FakeNLIModel.judge ---

class TestFakeJudge:
    def test_keyword_entailment(self):
        m = FakeNLIModel(mapping={"SUPPORTS": "entailment", "CONTRADICTS": "contradiction"})
        assert m.judge("p", "this SUPPORTS the claim").label == "entailment"

    def test_keyword_contradiction(self):
        m = FakeNLIModel(mapping={"SUPPORTS": "entailment", "CONTRADICTS": "contradiction"})
        assert m.judge("p", "this CONTRADICTS the claim").label == "contradiction"

    def test_no_keyword_fallback_neutral(self):
        m = FakeNLIModel(mapping={"SUPPORTS": "entailment"})
        r = m.judge("p", "unrelated text")
        assert r.label == "neutral"
        assert r.support_type == "indirect"
        assert r.confidence == 0.5

    def test_judge_confidence_values(self):
        m = FakeNLIModel(mapping={"ENT": "entailment", "NEU": "neutral", "CON": "contradiction"})
        assert m.judge("p", "ENT").confidence == 0.9
        assert m.judge("p", "NEU").confidence == 0.5
        assert m.judge("p", "CON").confidence == 0.9


# --- FakeNLIModel.best_match ---

class TestFakeBestMatch:
    def test_keyword_found_in_evidences(self):
        m = FakeNLIModel(mapping={"SUPPORTS": "entailment", "REFUTES": "contradiction"})
        evidences = ["unrelated", "this SUPPORTS the claim", "another unrelated"]
        r = m.best_match("claim", evidences)
        assert r.label == "entailment"
        assert r.support_type == "direct"

    def test_no_keyword_fallback_neutral(self):
        m = FakeNLIModel(mapping={"SUPPORTS": "entailment"})
        r = m.best_match("claim", ["no keyword here"])
        assert r.label == "neutral"
        assert r.support_type == "indirect"
        assert r.confidence == 0.5

    def test_empty_evidences_fallback_neutral(self):
        m = FakeNLIModel(mapping={"SUPPORTS": "entailment"})
        r = m.best_match("claim", [])
        assert r.label == "neutral"
        assert r.support_type == "indirect"
        assert r.confidence == 0.5


@pytest.mark.nli_real
@pytest.mark.skipif(os.getenv("RUN_NLI_REAL") != "1", reason="set RUN_NLI_REAL=1 to load the real NLI model")
def test_real_nli_smoke():
    verifier = NLIVerifier()
    result = verifier.judge(
        "The model learns latent dynamics and plans in latent space.",
        "The paper proposes planning with a latent world model.",
    )

    assert result.label in {"entailment", "neutral", "contradiction"}
    assert result.support_type in {"direct", "indirect", "contradictory"}
    assert isinstance(result.confidence, float)
