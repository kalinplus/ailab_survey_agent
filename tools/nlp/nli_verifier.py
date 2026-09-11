"""NLI verifier — adapted from SurGE's eval_relevance_* (CrossEncoder + label_mapping)."""

import os
from dataclasses import dataclass

LABELS = ["contradiction", "entailment", "neutral"]


def _default_device() -> str:
    """MPS first on Apple Silicon (~2.7x over 4 CPU cores for this model)."""
    override = os.getenv("EVISURVEY_NLI_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base", device=None):
        from sentence_transformers import CrossEncoder  # lazy import
        self.model = CrossEncoder(model_name, device=device or _default_device())

    def _predict(self, pairs):
        """Chunked scoring + MPS cache release: content fulltext turned
        best_match inputs into thousands of windows, and tens of thousands of
        small forward passes grow the PyTorch MPS allocator's cached blocks
        until the 20 GiB unified-memory cap OOMs (live 2026-09-10, twice).
        Chunks bound the per-forward size; empty_cache bounds accumulation."""
        if not pairs:
            return []
        chunk = int(os.getenv("EVISURVEY_NLI_BATCH", "32") or 32)
        if chunk <= 0:
            chunk = 32
        if len(pairs) <= chunk:
            out = self.model.predict(pairs)
        else:
            out = []
            for i in range(0, len(pairs), chunk):
                out.extend(self.model.predict(pairs[i:i + chunk]))
        self._release_mps_cache()
        return out

    @staticmethod
    def _release_mps_cache():
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def judge(self, premise: str, hypothesis: str) -> NLIResult:
        scores = self._predict([(premise, hypothesis)])[0]  # [contra, entail, neutral]
        c, e, n = float(scores[0]), float(scores[1]), float(scores[2])
        label = LABELS[int(max(range(3), key=lambda i: scores[i]))]
        return NLIResult(label, *_support(label, c, e, n))

    def best_match(self, claim: str, evidences: list[str]) -> NLIResult:
        if not evidences:
            return NLIResult("neutral", "indirect", 0.0)
        # Raw (evidence, claim) pairs — same convention as judge() and the
        # evaluator's L1. The old template wrapper ("There is a paper. Content:
        # '{ev}' / The paper supports: '{claim}'") broke the real cross-encoder
        # on verbatim-quote claims: live wave-4, 15/28 pairs flipped when the
        # wrapper was removed (scripts/diagnose_verifier_divergence.py).
        pairs = [(ev, claim) for ev in evidences]
        scores = self._predict(pairs)
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
