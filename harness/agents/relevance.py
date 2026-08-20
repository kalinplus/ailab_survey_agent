"""Report card: deterministic retrieval-quality signals for the strategy agent.

Spec: docs/RealAgent/关节1-策略Agent-方案与测试.md §2 (decision 1) + §3.5.
Relevance uses an embedding model (default BAAI/bge-small-en-v1.5 via
sentence-transformers — already a project dependency, lazy import so unit
tests never touch the network). If the model cannot load, the card keeps the
same shape and falls back to a keyword-overlap column (degradation logged).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingScorer:
    """Cosine relevance between aspect text and paper text.

    ``_embed`` is a test seam: ``(list[str]) -> list[list[float]]`` returning
    L2-normalized vectors. Without it, SentenceTransformer loads lazily on
    first use; any load/encode failure degrades to keyword overlap.
    """

    def __init__(self, model_name: str | None = None,
                 _embed: Callable[[list[str]], list[list[float]]] | None = None) -> None:
        self.model_name = model_name or os.getenv("EVISURVEY_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self._embed_fn = _embed
        self._model = None
        self._failed = False

    @property
    def mode(self) -> str:
        if self._embed_fn is not None:
            return "embedding"
        return "embedding" if self._load_model() else "keyword"

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Cosine similarity for each (aspect_text, paper_text) pair, in [0, 1]-ish."""
        if not pairs:
            return []
        if self._embed_fn is None and not self._load_model():
            return [keyword_overlap(a, p) for a, p in pairs]
        try:
            aspect_texts = [a for a, _ in pairs]
            paper_texts = [p for _, p in pairs]
            vectors = self._encode(aspect_texts + paper_texts)
            half = len(aspect_texts)
            # vectors are L2-normalized -> dot product == cosine
            return [_dot(vectors[i], vectors[half + i]) for i in range(half)]
        except Exception as exc:  # embed boundary: degrade, keep the card alive
            logger.warning(f"[relevance] embedding failed -> keyword fallback: {exc}")
            return [keyword_overlap(a, p) for a, p in pairs]

    def _load_model(self) -> bool:
        if self._embed_fn is not None:
            return True
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer  # lazy: no download at import
            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:
            self._failed = True
            logger.warning(
                f"[relevance] embed model {self.model_name!r} unavailable -> keyword fallback: {exc}")
            return False

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


_WORD_RE = re.compile(r"[a-z0-9]+")

# Tiny stoplist for the keyword fallback only (embedding path needs none).
_FALLBACK_STOP = {"the", "a", "an", "and", "of", "for", "in", "on", "with", "to", "via", "using", "based"}


def _tokens_of(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2 and t not in _FALLBACK_STOP}


def keyword_overlap(aspect_text: str, paper_text: str) -> float:
    """Overlap coefficient |A∩B| / min(|A|,|B|); deterministic fallback column."""
    a, p = _tokens_of(aspect_text), _tokens_of(paper_text)
    if not a or not p:
        return 0.0
    return len(a & p) / min(len(a), len(p))


def aspect_text(aspect: dict[str, Any]) -> str:
    keywords = " ".join(str(k) for k in aspect.get("keywords", []))
    return f"{aspect.get('aspect_name', '')}. {aspect.get('description', '')} {keywords}".strip()


def paper_text(hit: dict[str, Any]) -> str:
    """Title + first 1-2 abstract sentences (spec §2 decision 1)."""
    title = str(hit.get("title", "") or "").strip()
    abstract = str(hit.get("abstract", "") or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    lead = " ".join(sentences[:2]).strip()
    return f"{title}. {lead}".strip(". ")


def paper_identity(hit: dict[str, Any]) -> str:
    """Stable identity for cross-aspect dedup (mirrors _paper_identity in the builder)."""
    uid = str(hit.get("unique_id") or "").strip()
    if uid:
        return f"uid:{uid}"
    doi = str(hit.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = re.sub(r"\W+", " ", str(hit.get("title") or "").lower()).strip()
    if not title:
        return ""
    year = hit.get("publication_published_year") or hit.get("year") or ""
    return f"title:{title[:120]}:{year}"


def hit_year(hit: dict[str, Any]) -> int | None:
    raw = hit.get("publication_published_year") or hit.get("year")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def build_report_card(
    aspects: list[dict[str, Any]],
    per_aspect_hits: list[list[dict[str, Any]]],
    scorer: EmbeddingScorer | None = None,
) -> dict[str, Any]:
    """One row per aspect: n_hits / n_unique / overlap / rel / top5_titles / year_span.

    n_unique counts this aspect's hits not already claimed by an earlier aspect
    (first-claim-wins, mirroring P3's first_seen_rank); high hits + low unique
    means the aspect is re-finding other aspects' papers.
    """
    scorer = scorer or EmbeddingScorer()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    id_sets: list[set[str]] = []
    for aspect, hits in zip(aspects, per_aspect_hits):
        identities = [i for i in (paper_identity(h) for h in hits) if i]
        id_sets.append(set(identities))
        n_unique = sum(1 for i in identities if i not in seen)
        seen.update(identities)
        years = [y for y in (hit_year(h) for h in hits) if y]
        rows.append({
            "aspect_id": aspect.get("aspect_id", ""),
            "aspect_name": aspect.get("aspect_name", ""),
            "n_hits": len(hits),
            "n_unique": n_unique,
            "overlap": 0.0,  # filled below once all id sets are known
            "rel": 0.0,
            "top5_titles": [str(h.get("title", "")) for h in hits[:5]],
            "year_span": [min(years), max(years)] if years else None,
            "_hits": hits,
        })
    # pairwise Jaccard overlap of each aspect against every other aspect
    for index, row in enumerate(rows):
        others = [id_sets[j] for j in range(len(id_sets)) if j != index]
        jaccards = [
            len(id_sets[index] & other) / len(id_sets[index] | other)
            for other in others
            if id_sets[index] or other
        ]
        row["overlap"] = round(sum(jaccards) / len(jaccards), 3) if jaccards else 0.0
    # embedding (or fallback) relevance: aspect text vs its own hits
    pairs: list[tuple[str, str]] = []
    counts: list[int] = []
    for row, aspect in zip(rows, aspects):
        hits = row.pop("_hits")
        counts.append(len(hits))
        pairs.extend((aspect_text(aspect), paper_text(hit)) for hit in hits)
    scores = scorer.score_pairs(pairs) if pairs else []
    cursor = 0
    for row, count in zip(rows, counts):
        chunk = scores[cursor:cursor + count]
        cursor += count
        row["rel"] = round(sum(chunk) / len(chunk), 3) if chunk else 0.0
    return {
        "rel_mode": scorer.mode,
        "aspects": rows,
        "global_score": round(global_score_rows(rows), 4),
    }


def global_score_rows(rows: list[dict[str, Any]]) -> float:
    """Mean over aspects of min(n_unique,10)/10 * rel.

    Spec §3.6 writes the sum; we normalize by aspect count so a merge/delete
    edit cannot fake a regression (the rollback guarantee compares across
    rounds whose aspect counts may differ). Monotone transform of the spec
    formula when aspect count is fixed.
    """
    if not rows:
        return 0.0
    total = sum(min(r["n_unique"], 10) / 10 * r["rel"] for r in rows)
    return total / len(rows)


def global_score(card: dict[str, Any]) -> float:
    return card.get("global_score", global_score_rows(card.get("aspects", [])))


def render_card(card: dict[str, Any], *, rounds_left: int | None = None,
                sciverse_left: int | None = None) -> str:
    """Compact text block auto-injected into the agent's user messages."""
    lines = [f"report card (relevance={card['rel_mode']}, global_score={card['global_score']}):"]
    for row in card["aspects"]:
        span = f"{row['year_span'][0]}-{row['year_span'][1]}" if row["year_span"] else "n/a"
        titles = " | ".join(t[:60] for t in row["top5_titles"][:3]) or "-"
        lines.append(
            f"- {row['aspect_id']} \"{row['aspect_name']}\": n_hits={row['n_hits']} "
            f"n_unique={row['n_unique']} overlap={row['overlap']} rel={row['rel']} "
            f"years={span}\n  sample: {titles}"
        )
    budget = []
    if rounds_left is not None:
        budget.append(f"rounds_left={rounds_left}")
    if sciverse_left is not None:
        budget.append(f"sciverse_calls_left={sciverse_left}")
    if budget:
        lines.append("budget: " + " ".join(budget))
    return "\n".join(lines)
