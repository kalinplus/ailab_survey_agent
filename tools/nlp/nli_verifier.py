"""NLI verifier — adapted from SurGE's eval_relevance_* (CrossEncoder + label_mapping)."""

from dataclasses import dataclass

LABELS = ["contradiction", "entailment", "neutral"]


@dataclass
class NLIResult:
    label: str          # entailment | neutral | contradiction
    support_type: str   # direct | indirect | contradictory
    confidence: float

    @property
    def status(self) -> str:
        if self.label == "entailment" and self.confidence >= 0.6:
            return "supported"
        if self.label == "contradiction":
            return "unsupported"
        return "weak"  # neutral or low-confidence entailment


class NLIVerifier:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base"):
        from sentence_transformers import CrossEncoder  # lazy import
        self.model = CrossEncoder(model_name)

    def judge(self, premise: str, hypothesis: str) -> NLIResult:
        scores = self.model.predict([(premise, hypothesis)])[0]  # [contra, entail, neutral]
        c, e, n = float(scores[0]), float(scores[1]), float(scores[2])
        label = LABELS[int(max(range(3), key=lambda i: scores[i]))]
        return NLIResult(label, *_support(label, c, e, n))

    def best_match(self, claim: str, evidences: list[str]) -> NLIResult:
        if not evidences:
            return NLIResult("neutral", "indirect", 0.0)
        pairs = [(f"There is a paper. Content: '{ev}'", f"The paper supports: '{claim}'")
                 for ev in evidences]
        scores = self.model.predict(pairs)
        best_i = int(max(range(len(evidences)), key=lambda i: scores[i][1]))
        c, e, n = (float(x) for x in scores[best_i])
        label = LABELS[int(max(range(3), key=lambda i: scores[best_i][i]))]
        return NLIResult(label, *_support(label, c, e, n))


def _support(label, c, e, n):
    if label == "entailment" and e > c and e > n:
        return "direct", round((e - max(c, n)) / (abs(e) + abs(c) + abs(n) + 1e-8), 3)
    if label == "neutral" and n > c:
        return "indirect", round((e - c) / (abs(e) + abs(c) + 1e-8), 3)
    if label == "contradiction":
        return "contradictory", 0.2
    return "indirect", 0.2


class FakeNLIModel:
    """Duck-typed stand-in for NLIVerifier — keyword-based label selection."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def judge(self, premise: str, hypothesis: str) -> NLIResult:
        low_hypothesis = hypothesis.lower()
        for kw, label in self.mapping.items():
            if kw in hypothesis or str(kw).lower() in low_hypothesis:
                conf = {"entailment": 0.9, "neutral": 0.5, "contradiction": 0.9}[label]
                support_type = {"entailment": "direct", "neutral": "indirect", "contradiction": "contradictory"}[label]
                return NLIResult(label, support_type, conf)
        return NLIResult("neutral", "indirect", 0.5)

    def best_match(self, claim: str, evidences: list[str]) -> NLIResult:
        for ev in evidences:
            low_ev = ev.lower()
            for kw, label in self.mapping.items():
                if kw in ev or str(kw).lower() in low_ev:
                    conf = {"entailment": 0.9, "neutral": 0.5, "contradiction": 0.9}[label]
                    support_type = {"entailment": "direct", "neutral": "indirect", "contradiction": "contradictory"}[label]
                    return NLIResult(label, support_type, conf)
        return NLIResult("neutral", "indirect", 0.5)
