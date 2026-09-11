"""Shared claim-sentence and evidence-window kernel.

The generation-side claim mapper (claim_mapper.py) and the post-hoc evaluator
(evaluate_survey.py) must agree on how survey markdown becomes claim sentences
with bound citations and how evidence texts are windowed before NLI. Their
former private implementations diverged (bracket validation, trailing citation
fragments, structural-line exclusion), so the repair joint chased claim_map
verdicts the evaluator could not reproduce. The rules live here once.
"""

import re

SENT_SPLIT = re.compile(r"(?<=[.。])\s+")
CITATION = re.compile(r"\[([^\]]+)\]")
STRUCTURAL = ("#", "|", "!", ">")

_SENT_BOUNDARY = re.compile(r"(?<=[.!?。])\s+")
_MIN_UNIT_CHARS = 20


def normalize_text(text: str) -> str:
    """Case- and punctuation-insensitive comparison form (keeps CJK word chars).
    Shared by claim-text matching (claim mapper stage 1, freshness gate) and
    provenance title equality."""
    return re.sub(r"[\W_]+", " ", str(text).casefold()).strip()


def body_text(md: str) -> str:
    """Drop headings / tables / images / quotes so only prose is measured."""
    lines = [ln for ln in md.splitlines() if not ln.lstrip().startswith(STRUCTURAL)]
    return "\n".join(lines)


def sentences(md: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(body_text(md)) if s.strip()]


def is_citation(cited_id: str, known_ids: set[str] | None) -> bool:
    """Bracket content counts as a citation only if it is a paper id
    ('paper:...' convention) or resolvable in the evidence store — figure/
    table references like '[Future Matrix]' and math intervals like '[0,1]'
    are not citations. known_ids=None keeps the prefix rule only."""
    return cited_id.startswith("paper:") or (known_ids is not None and cited_id in known_ids)


def sentence_units(md: str, known_ids: set[str] | None = None) -> list[tuple[str, list[str]]]:
    """(claim_text, citation_ids) per logical sentence.

    Models drop the tag after the closing period ('...scale. [paper:x].'),
    which sentence splitting turns into a citation-only fragment; that
    fragment is merged back into the preceding sentence so its citations bind
    to the claim they annotate instead of being dropped (and the preceding
    sentence miscounted as uncited)."""
    units: list[list] = []
    for sentence in sentences(md):
        cites = [c for c in CITATION.findall(sentence) if is_citation(c, known_ids)]
        text = CITATION.sub("", sentence).strip().rstrip(".。").strip()
        if not text and cites and units and units[-1][0]:
            units[-1][1].extend(cites)
        elif text:
            units.append([text, cites])
    return [(text, cites) for text, cites in units]


def evidence_units(texts: list[str]) -> list[str]:
    """Sentence windows: cross-encoder NLI fails on long-passage premises
    (a verbatim quote inside a 1700-char abstract scores neutral), so each
    multi-sentence evidence text is split into sentence-level units."""
    units: list[str] = []
    for t in texts:
        parts = [p.strip() for p in _SENT_BOUNDARY.split(t)]
        if len(parts) == 1:
            units.append(t.strip())
            continue
        units.extend(p for p in parts if len(p) >= _MIN_UNIT_CHARS)
    return units or [t.strip() for t in texts]
