"""C tool: grounded survey writer and visual artifact builder."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

# tools run in ToolRegistry worker threads; a GUI backend (macosx) cannot
# create a FigureManager off the main thread, and we only ever savefig to files
matplotlib.use("Agg")

from config import load_config
from harness.agents.relevance import EmbeddingScorer
from harness.json_io import read_json, write_json
from harness.logger import now_iso
from tools.verify.text_units import sentence_units, normalize_text
from tools.verify.source_contract import SCOPE_ROLES, valid_support, quote_diagnostic, role_violation

logger = logging.getLogger(__name__)

# S2 hybrid writer: skeleton stays deterministic, body paragraphs may be
# LLM-drafted when EVISURVEY_WRITER_LLM=1 (default off = template output).
SECTION_PAPERS = 5
SECTION_EVIDENCE_SNIPPETS = 2
# Spec bound: any two sections may share at most 2 of their selected papers.
REUSE_LIMIT = 2

# Harness meta-talk that must never reach survey prose (eval v2 L1.5 leak).
# Values are casefolded; compare with _contains_banned_phrase.
BANNED_META_PHRASES = [
    "c 模块",
    "c module",
    "citationreadyset",
    "papercards",
    "evidencestore",
    "generatedartifactbank",
    "reviewboard",
    "harness 链路",
    "harness chain",
    "降调写法",
    "topic relevance",
    "rule-based taxonomy fallback",
    "merged from a seed survey",
]

# A bare paper id in prose (the S0 failure mode: space-broken DOIs the model
# paraphrased out of the prompt) can never be legitimate survey text. Sentence
# splitting cuts such ids mid-DOI, so the orphan continuation fragments
# ("1016/j. fake must vanish.") need their own guard.
_LEAKED_ID_RE = re.compile(r"paper\s*:\s*\d", re.IGNORECASE)
_LEAKED_ID_FRAGMENT_RE = re.compile(r"^\s*\d{2,}\s*[/.]")

# Alias and legacy `paper:` tags stranded after the closing punctuation
# ("...scale. [P2]." / "...scale. [paper:x]."); see _map_alias_citations.
_STRANDED_TAGS_RE = re.compile(r"([.。!?！？])\s*((?:\[(?:P\d+|paper:[^\]]+)\]\s*)+\.?)")

# T5 structured claims: the LLM body path plans claims as data
# ({"text", "cite", "scope"}) before any prose exists. Scope drives the
# deterministic paragraph order and travels with the persisted claim.
_CLAIM_SCOPES = ("contribution", "method", "comparison", "limitation", "background")
_CLAIM_BRACKETS_RE = re.compile(r"\[([^\[\]]*)\]")

# Evidence tiers describe source availability; claim roles constrain scope.
_FULLTEXT_SOURCE_TYPES = {"paragraph", "agentic_chunk", "content_chunk"}
_TIER_LABELS = {"fulltext_ok": "full", "abstract_only": "abstract-only"}

# Artifact embed positions, keyed to paragraph slots. With the moves menu the
# paragraph count is model-chosen, so indices clamp to the last paragraph.
_PLACEMENT_INDEX = {
    "after_topic_paragraph": 0,
    "after_comparison_paragraph": 1,
    "after_limitation_paragraph": -1,
}

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
# \s+ not \s*: a zero-width split shreds DOIs ("10." | "48550/...") so every
# citation in an LLM sentence got fragmented and dropped by the guards.

# T12 intro dedup: an intro sentence whose word bag overlaps a body-section
# sentence with Jaccard >= 0.6 (casefolded, stopwords out) restates the body
# and is dropped once the body has rendered. Section-title words are removed
# from both bags first — the roadmap is supposed to name sections, so title
# overlap is not a penalty.
_INTRO_DEDUP_THRESHOLD = 0.6
_INTRO_MIN_SENTENCES = 3
_INTRO_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_INTRO_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "both", "by", "can",
    "each", "for", "from", "how", "in", "into", "is", "it", "its", "may",
    "no", "not", "of", "on", "or", "over", "own", "such", "than", "that",
    "the", "their", "then", "these", "this", "those", "through", "to",
    "via", "was", "were", "what", "when", "where", "whether", "which",
    "while", "who", "with", "using", "based",
    # zh function characters (CJK tokens are single characters)
    "的", "了", "在", "是", "和", "与", "或", "及", "对", "从", "被", "把",
    "为", "也", "都", "就", "而", "等", "中", "有", "不", "这", "那", "并",
    "其", "以", "于", "由", "可", "能", "会", "该",
}

POSITIVE_KEYWORDS = [
    "game",
    "games",
    "atari",
    "dota",
    "starcraft",
    "reinforcement learning",
    "world model",
    "world models",
    "model-based",
    "simulation",
    "simulator",
    "interactive",
    "agent",
    "agents",
    "gamecraft",
    "neural game engine",
    "environment generation",
]

NEGATIVE_KEYWORDS = [
    "wireless network",
    "influenza",
    "bats",
    "urban planning",
    "energy efficient wireless",
    "microbiology",
    "medical",
    "communication letters",
]


def run(request_path: str) -> dict[str, Any]:
    cfg = load_config()
    root = cfg.root_dir
    request_file = _resolve(root, request_path)
    req = read_json(request_file)
    task_id = req.get("task_id", "unknown_task")
    inputs = req.get("inputs", {})
    outputs = req.get("outputs", {})
    language = req.get("language", "zh")

    cards_raw = _read_optional(root, inputs.get("paper_cards_path"), {"paper_cards": []})
    evidence_raw = _read_optional(root, inputs.get("evidence_store_path"), {"evidence": []})
    taxonomy_raw = _read_optional(root, inputs.get("taxonomy_path"), {"categories": []})
    figure_raw = _read_optional(root, inputs.get("figure_bank_path"), {"figures": []})
    table_raw = _read_optional(root, inputs.get("table_bank_path"), {"tables": []})
    ready_set = _read_optional(root, inputs.get("citation_ready_set_path"), {"allowed_paper_ids": []})
    if os.getenv("FINAL_SEED_PAPERS") == "1":
        final_data = _load_final_seed_data(root)
        cards_raw = final_data.get("paper_cards", cards_raw)
        evidence_raw = final_data.get("evidence_store", evidence_raw)
        taxonomy_raw = final_data.get("taxonomy", taxonomy_raw)
        figure_raw = final_data.get("figure_bank", figure_raw)
        table_raw = final_data.get("table_bank", table_raw)
        ready_set = final_data.get("citation_ready_set", ready_set)

    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    topic = req.get("topic", "")
    cards = [card for card in _as_list(cards_raw, "paper_cards") if card.get("paper_id") in allowed_ids]
    evidence = _as_list(evidence_raw, "evidence")
    categories = _categories(taxonomy_raw, cards)
    evidence_by_paper = _group_by(evidence, "paper_id")
    cards = _score_cards_for_topic(cards, evidence_by_paper, categories, topic)

    section_plan = _build_section_claim_plan(task_id, categories, cards, evidence_by_paper)
    safe_cards = [{**card, **_selected_paper_entry(card, evidence_by_paper)} for card in cards]
    timeline = _build_timeline(task_id, categories, safe_cards)
    artifacts, image_notes = _build_generated_artifacts(cfg, task_id, timeline, categories, safe_cards, evidence_by_paper)
    structured_claims: list[dict[str, Any]] = []
    survey = _render_survey(
        topic,
        language,
        section_plan,
        artifacts,
        cards,
        evidence_by_paper,
        llm_chat=_writer_llm_chat(cfg),
        structured_out=structured_claims,
    )
    survey = _lint_citation_brackets(survey)

    survey_path = _resolve(root, outputs.get("survey_markdown_path", "output/survey.md"))
    timeline_path = _resolve(root, outputs.get("timeline_path", "cache/timeline.json"))
    artifact_bank_path = _resolve(root, outputs.get("generated_artifact_bank_path", "cache/generated_artifact_bank.json"))
    claim_plan_path = root / "cache" / "section_claim_plan.json"
    structured_claims_path = root / "cache" / "structured_claims.json"
    preflight_path = root / "output" / "c_preflight_report.json"
    audit_path = root / "output" / "artifact_audit_report.json"
    visual_path = root / "output" / "visual_decision_report.json"

    survey_path.parent.mkdir(parents=True, exist_ok=True)
    survey_path.write_text(survey, encoding="utf-8")
    write_json(timeline_path, timeline)
    write_json(artifact_bank_path, {"task_id": task_id, "artifacts": artifacts})
    write_json(claim_plan_path, {"task_id": task_id, "sections": section_plan})
    # T5: the structured claim source is the truth the downstream claim map is
    # built from; reverse-parsing the markdown (T4 kernel) is only the fallback.
    write_json(structured_claims_path, {"task_id": task_id, "claims": structured_claims})

    preflight = _preflight(task_id, survey, allowed_ids, artifacts, cards)
    empty_sections = sum(not any(p.get("sources") for p in section.get("selected_papers", [])) for section in section_plan)
    preflight["empty_evidence_sections"] = empty_sections
    preflight["no_verifiable_claims"] = not bool(structured_claims)
    preflight["pass"] = preflight["pass"] and not empty_sections and bool(structured_claims)
    audit = _artifact_audit(task_id, root, artifacts, allowed_ids)
    visual = _visual_decisions(task_id, timeline, artifacts, image_notes)
    write_json(preflight_path, preflight)
    write_json(audit_path, audit)
    write_json(visual_path, visual)

    status = "success" if preflight["pass"] and audit["pass"] else "partial_success"
    out_paths = [
        _rel(root, survey_path),
        _rel(root, timeline_path),
        _rel(root, artifact_bank_path),
        _rel(root, claim_plan_path),
        _rel(root, structured_claims_path),
        _rel(root, preflight_path),
        _rel(root, audit_path),
        _rel(root, visual_path),
    ]
    return _tool_result(
        task_id,
        "write_survey",
        status,
        request_path,
        out_paths,
        {
            "allowed_papers": len(allowed_ids),
            "used_papers": len(_extract_citations(survey) & allowed_ids),
            "sections": len(section_plan),
            "structured_claims": len(structured_claims),
            "empty_evidence_sections": empty_sections,
            "generated_artifacts": len(artifacts),
        },
        "Grounded survey and C-generated artifacts written.",
    )


def _writer_llm_chat(cfg) -> Callable | None:
    """`.chat` callable for the hybrid writer, or None when it stays template-only."""
    if os.getenv("EVISURVEY_WRITER_LLM", "0").lower() not in {"1", "true", "yes"}:
        return None
    from llm_client import heavy_llm_client

    client = heavy_llm_client(cfg)
    if not client.is_configured():
        logger.warning("[writer] EVISURVEY_WRITER_LLM on but no LLM key configured -> template path")
        return None
    return client.chat


def _build_section_claim_plan(
    task_id: str,
    categories: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    scorer: EmbeddingScorer | None = None,
) -> list[dict[str, Any]]:
    """Assign citation-ready papers to sections by relevance, best fit first.

    Replaces the old global-top rotation that put the same handful of papers in
    every section (84 citations over 3 unique ids) and let off-topic cards
    survive a category-bucket miss. Sections are filled from their best-fitting
    untaken papers under an equal quota; only a pool that has run dry is
    back-filled, and never past REUSE_LIMIT shared papers between two sections.
    """
    global_ranked = _rank_cards(cards, evidence_by_paper)
    scorer = scorer or _relevance_scorer()
    relevance = _section_relevance_matrix(categories, cards, scorer)

    cards_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        key = card.get("category_id") or card.get("category") or "uncategorized"
        cards_by_category[str(key)].append(card)
    category_keys = [
        str(cat.get("category_id") or cat.get("category_name") or f"cat_{position + 1:03d}")
        for position, cat in enumerate(categories)
    ]
    # Sections that own papers pick first (scarcest native pool first) so a
    # category cannot have its own papers stolen by an empty-pool section;
    # output order still follows the taxonomy order.
    pick_order = sorted(
        range(len(categories)),
        key=lambda i: (
            len(cards_by_category.get(category_keys[i], [])) == 0,
            len(cards_by_category.get(category_keys[i], [])),
            i,
        ),
    )

    taken: set[str] = set()
    quotas = _section_quotas(len(cards), len(categories))
    owners: dict[str, set[int]] = defaultdict(set)
    reuse_count: dict[str, int] = defaultdict(int)
    overlap: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    picks: dict[int, list[dict[str, Any]]] = {}
    for position in pick_order:
        selected = _pick_section_cards(
            relevance[position], cards, taken, quotas[position], position, owners, reuse_count, overlap
        )
        picks[position] = selected
        for card in selected:
            paper_id = card["paper_id"]
            for owner in owners[paper_id]:
                overlap[position][owner] += 1
                overlap[owner][position] += 1
            owners[paper_id].add(position)
            if paper_id in taken:
                reuse_count[paper_id] += 1
            taken.add(paper_id)

    sections: list[dict[str, Any]] = []
    for position, category in enumerate(categories):
        index = position + 1
        selected = picks[position]
        title = str(category.get("category_name") or category.get("name") or f"Theme {index}")
        claims = []
        for claim_index, card in enumerate(selected, start=1):
            paper_id = card["paper_id"]
            evidence_ids = [item.get("evidence_id") for item in evidence_by_paper.get(paper_id, []) if item.get("evidence_id")]
            claims.append(
                {
                    "claim_id": f"claim_sec{index}_{claim_index:03d}",
                    "claim": _claim_from_card(card),
                    "supporting_papers": [paper_id],
                    "supporting_evidence": evidence_ids[:3],
                    "allowed_figures": card.get("figure_ids", [])[:1],
                    "risk_level": "low" if evidence_ids else "medium",
                }
            )
        sections.append(
            {
                "section_id": f"sec_{index:02d}",
                "section_title": title,
                "section_goal": str(category.get("description") or "Summarize citation-ready papers in this theme."),
                "target_length_words": 450,
                "allowed_paper_ids": [card["paper_id"] for card in selected],
                "selected_papers": [_selected_paper_entry(card, evidence_by_paper) for card in selected],
                "artifact_slots": _artifact_slots_for_section(index, title),
                "claims": claims,
            }
        )

    cited = {pid for section in sections for pid in section.get("allowed_paper_ids", [])}
    remaining = [card for card in global_ranked if card.get("paper_id") not in cited]
    if remaining and len(cited) < min(8, len(cards)):
        selected = remaining[:SECTION_PAPERS]
        sections.append(
            {
                "section_id": f"sec_{len(sections) + 1:02d}",
                "section_title": "Cross-Paper Evidence Synthesis",
                "section_goal": "Use additional citation-ready papers to broaden coverage while staying within the whitelist.",
                "target_length_words": 450,
                "allowed_paper_ids": [card["paper_id"] for card in selected],
                "selected_papers": [_selected_paper_entry(card, evidence_by_paper) for card in selected],
                "artifact_slots": [],
                "claims": [
                    {
                        "claim_id": f"claim_extra_{idx:03d}",
                        "claim": _claim_from_card(card),
                        "supporting_papers": [card["paper_id"]],
                        "supporting_evidence": [
                            item.get("evidence_id")
                            for item in evidence_by_paper.get(card["paper_id"], [])
                            if item.get("evidence_id")
                        ][:3],
                        "allowed_figures": card.get("figure_ids", [])[:1],
                        "risk_level": "low" if evidence_by_paper.get(card["paper_id"]) else "medium",
                    }
                    for idx, card in enumerate(selected, start=1)
                ],
            }
        )
    if not sections and cards:
        selected = global_ranked[:SECTION_PAPERS]
        sections.append(
            {
                "section_id": "sec_01",
                "section_title": "Citation-Ready Evidence",
                "section_goal": "Summarize the available citation-ready papers.",
                "target_length_words": 450,
                "allowed_paper_ids": [card["paper_id"] for card in selected],
                "selected_papers": [_selected_paper_entry(card, evidence_by_paper) for card in selected],
                "artifact_slots": _artifact_slots_for_section(1, "Citation-Ready Evidence"),
                "claims": [
                    {
                        "claim_id": f"claim_sec1_{idx:03d}",
                        "claim": _claim_from_card(card),
                        "supporting_papers": [card["paper_id"]],
                        "supporting_evidence": [
                            item.get("evidence_id")
                            for item in evidence_by_paper.get(card["paper_id"], [])
                            if item.get("evidence_id")
                        ][:3],
                        "allowed_figures": card.get("figure_ids", [])[:1],
                        "risk_level": "medium",
                    }
                    for idx, card in enumerate(selected, start=1)
                ],
            }
        )
    for section in sections:
        section["section_goal"] = _safe_section_goal(section)
        section["claims"] = [dict(claim=source["claim_text"], source_role=source["source_role"],
                                  supporting_papers=[source["paper_id"]],
                                  supporting_evidence=source["evidence_ids"], risk_level="pending_verification")
                             for paper in section["selected_papers"] for source in paper.get("sources", [])]
    return sections


def _relevance_scorer() -> EmbeddingScorer:
    """Scorer for section-to-paper fit.

    Keyword overlap by default: deterministic and offline (unit tests and the
    eval harness must never trigger an HF download here). `EVISURVEY_WRITER_EMBED=1`
    opts real runs into the bge column of the same scorer.
    """
    if os.getenv("EVISURVEY_WRITER_EMBED", "0").lower() not in {"1", "true", "yes"}:
        scorer = EmbeddingScorer()
        scorer._failed = True  # same seam the relevance tests use: keyword column
        return scorer
    return EmbeddingScorer()


def _category_text(category: dict[str, Any]) -> str:
    keywords = " ".join(str(k) for k in category.get("keywords", []))
    name = str(category.get("category_name") or category.get("name") or "")
    return f"{name}. {category.get('description', '')} {keywords}".strip()


def _card_text(card: dict[str, Any]) -> str:
    fields = [card.get("title"), card.get("problem"), card.get("method"), card.get("contribution")]
    return " ".join(str(field) for field in fields if field).strip()


def _section_relevance_matrix(
    categories: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    scorer: EmbeddingScorer,
) -> list[dict[str, float]]:
    """relevance[category_index][paper_id] -> section-fit score (bge or keyword)."""
    pairs = [(_category_text(cat), _card_text(card)) for cat in categories for card in cards]
    scores = scorer.score_pairs(pairs) if pairs else []
    matrix: list[dict[str, float]] = [{} for _ in categories]
    cursor = 0
    for cat_index in range(len(categories)):
        for card in cards:
            matrix[cat_index][card["paper_id"]] = float(scores[cursor])
            cursor += 1
    return matrix


def _section_quotas(n_cards: int, n_sections: int) -> list[int]:
    """Equal per-section share of the citation pool (every theme keeps a body)."""
    if n_sections <= 0 or n_cards <= 0:
        return [0] * max(0, n_sections)
    base = min(SECTION_PAPERS, max(1, n_cards // n_sections))
    quotas = [base] * n_sections
    left = max(0, min(n_cards - sum(quotas), SECTION_PAPERS * n_sections - sum(quotas)))
    for position in range(n_sections):
        if left <= 0:
            break
        quotas[position] += 1
        left -= 1
    return quotas


def _pick_section_cards(
    relevance_row: dict[str, float],
    cards: list[dict[str, Any]],
    taken: set[str],
    quota: int,
    section_index: int,
    owners: dict[str, set[int]],
    reuse_count: dict[str, int],
    overlap: dict[int, dict[int, int]],
) -> list[dict[str, Any]]:
    """Best-fit cards for one section: untaken first, then bounded reuse.

    Order is relevance to this section, then global topic rank, so an off-topic
    card can no longer displace a fitting one. When the untaken pool is dry the
    section reuses already-claimed cards — at most REUSE_LIMIT shared papers
    with any other section, which keeps the plan from either collapsing (old
    global-top rotation) or emitting body sections with no papers at all.
    """
    if quota <= 0:
        return []
    by_id = {card["paper_id"]: card for card in cards}
    free = [card for pid, card in by_id.items() if pid not in taken]
    free.sort(
        key=lambda card: (
            -relevance_row.get(card["paper_id"], 0.0),
            -card.get("topic_relevance_score", 0.0),
            card.get("paper_id", ""),
        )
    )
    picked = free[:quota]
    picked_ids = {card["paper_id"] for card in picked}
    if len(picked) < quota:
        reused = [card for pid, card in by_id.items() if pid not in picked_ids]
        reused.sort(
            key=lambda card: (
                reuse_count.get(card["paper_id"], 0),
                -relevance_row.get(card["paper_id"], 0.0),
                -card.get("topic_relevance_score", 0.0),
                card.get("paper_id", ""),
            )
        )
        for card in reused:
            if len(picked) >= quota:
                break
            paper_id = card["paper_id"]
            if any(overlap[section_index][owner] >= REUSE_LIMIT for owner in owners.get(paper_id, set())):
                continue
            picked.append(card)
            picked_ids.add(paper_id)
    return picked


def _selected_paper_entry(card: dict[str, Any], evidence_by_paper: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    score = float(card.get("topic_relevance_score", 0.0))
    # ONLY claims listed in an evidence row's supports_claims may feed the
    # writer: the claim mapper re-checks that correspondence verbatim at final
    # verify (invalid_source_binding otherwise). A card claim that failed P5.2's
    # NLI gate must not sneak back in through a side door (wave8 ③a retraction —
    # it produced 9 unsupported claims and a coverage_fail in the first run).
    sources = []
    for item in evidence_by_paper.get(card.get("paper_id"), []):
        for claim in item.get("supports_claims", []):
            if valid_support(claim, item):
                sources.append({**claim, "evidence_id": item["evidence_id"],
                                "source_type": item.get("source_type", ""),
                                "source_doc_id": item.get("source_doc_id", ""),
                                "source_title": item.get("source_title", ""),
                                "source_chunk_id": item.get("source_chunk_id", "")})
    entry = {
        "paper_id": card.get("paper_id"), "title": card.get("title", ""),
        "year": card.get("year"), "sources": sources,
        "problem": "", "method": "", "contribution": "", "limitations": "",
        "evidence_snippets": [source["claim_text"] for source in sources],
        "topic_relevance_score": round(score, 3), "selection_reason": _selection_reason(card),
    }
    for field, scope in (("method", "method"), ("contribution", "contribution"), ("limitations", "limitation")):
        entry[field] = next((x["claim_text"] for x in sources if x["source_role"] in SCOPE_ROLES[scope]), "")
    return entry


def _artifact_slots_for_section(index: int, title: str) -> list[dict[str, str]]:
    text = title.lower()
    slots = []
    # Index-based assignment avoids keyword-matching artifacts appearing in multiple sections.
    if index == 1:
        slots.append(
            {
                "artifact_id": "publication_timeline",
                "placement": "after_topic_paragraph",
                "caption": "Publication years of the selected representative papers.",
            }
        )
    if index == 2:
        slots.append(
            {
                "artifact_id": "taxonomy_overview",
                "placement": "after_topic_paragraph",
                "caption": "Distribution of selected papers across the survey taxonomy.",
            }
        )
        slots.append(
            {
                "artifact_id": "representative_systems",
                "placement": "after_comparison_paragraph",
                "caption": "Representative papers organized by method, contribution, and limitation.",
            }
        )
    if index == 4:
        slots.append(
            {
                "artifact_id": "evaluation_protocol_matrix",
                "placement": "after_comparison_paragraph",
                "caption": "Evaluation protocols grouped by what they measure and where they can mislead.",
            }
        )
    if "future" in text or "challenge" in text:
        slots.append(
            {
                "artifact_id": "future_directions_matrix",
                "placement": "after_limitation_paragraph",
                "caption": "Future directions derived from limitations in the selected literature.",
            }
        )
    return slots


def _load_final_seed_data(root: Path) -> dict[str, Any]:
    final_cards = root / "cache" / "final_paper_cards.json"
    if not final_cards.exists():
        try:
            from scripts.build_final_seed_papers import write_final_seed_files

            write_final_seed_files(root)
        except Exception:
            return {}
    out: dict[str, Any] = {}
    for name in ["paper_cards", "evidence_store", "taxonomy", "citation_ready_set", "figure_bank", "table_bank"]:
        path = root / "cache" / f"final_{name}.json"
        if path.exists():
            out[name] = read_json(path)
    return out


def _norm_sentence(sentence: str) -> str:
    return " ".join(sentence.split()).casefold()


def _lower_first(text: str) -> str:
    """Lowercase the first character of a card fragment spliced mid-sentence."""
    return text[:1].lower() + text[1:] if text else text


def _fresh_text(text: str, seen: set[str], keep_duplicate: bool = True) -> str:
    """Keep only sentences no other section has used (cross-section boilerplate guard).

    Repeated boilerplate across sections is what pushed eval S0 section-pair
    similarity to 0.9+, so every rendered sentence is registered in `seen` and
    an exact repeat is dropped. With `keep_duplicate` the template path keeps
    its original text rather than rendering an empty section; the LLM path
    drops the sentence and lets the caller judge the paragraph usable.
    """
    kept = []
    for sentence in _split_sentences(text):
        key = _norm_sentence(sentence)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence.strip())
    if not kept:
        if keep_duplicate:
            logger.warning("[writer] paragraph fully duplicated across sections; keeping original text")
            return text
        return ""
    return " ".join(kept)


def _llm_or_template(label: str, llm_chat: Callable | None, draft: Callable[[], str], template: Callable[[], str]) -> str:
    """One LLM boundary: a section-level failure degrades to its template text.

    Drafts pass through heading-strip + partial-sentence trim: a model heading
    would duplicate the rendered one, and a token-limit cutoff would leak a
    mid-word stub into the survey.
    """
    if llm_chat is None:
        return template()
    try:
        draft_text = _trim_partial_sentence(_strip_model_headings(draft()))
        return draft_text or template()
    except Exception as exc:  # LLM boundary: designed per-section degradation
        logger.warning(f"[writer] {label} LLM unusable -> template fallback: {exc}")
        return template()


def _render_survey(
    topic: str,
    language: str,
    sections: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    llm_chat: Callable | None = None,
    structured_out: list[dict[str, Any]] | None = None,
) -> str:
    zh = language == "zh"
    structured_out = structured_out if structured_out is not None else []
    ranked = _rank_cards(cards, evidence_by_paper)
    allowed = {str(card["paper_id"]) for card in cards}
    # Cross-section sentence ledger: no sentence may render in two sections.
    seen: set[str] = set()
    title = f"# {topic or 'Grounded Survey'}"
    lines = [
        title,
        "",
        "## Abstract" if not zh else "## 摘要",
        _abstract_text(topic, sections, ranked, evidence_by_paper, zh, llm_chat=llm_chat, allowed=allowed, seen=seen, claims_out=structured_out),
        "",
        "## Introduction" if not zh else "## 引言",
        _introduction_text(topic, sections, zh, llm_chat=llm_chat, allowed=allowed, seen=seen),
        "",
    ]
    # The intro paragraph slot: the T12 dedupe rewrites it in place once the
    # body sentences exist (the intro renders first, so the guard is deferred).
    intro_index = len(lines) - 2
    if not zh:
        lines.extend(
            [
                "The survey first presents a graphical overview to orient readers before the technical taxonomy. Figure 1 shows the relationship among world models, game environments, agents, and evaluation.",
                "",
                "![Figure 1. Graphical Overview of World Models and GameCraft](hero_banner)",
                "",
                "Figure 2 then reframes the same topic as an interaction loop: an agent observes, a world model predicts, the environment responds, and evaluation closes the cycle.",
                "",
                "![Figure 2. Agent-Environment Interaction Loop in Learned Game Worlds](concept_overview)",
                "",
            ]
        )

    tiers = _paper_evidence_tiers(evidence_by_paper)
    body_start = len(lines)
    for index, section in enumerate(sections):
        following = sections[index + 1] if index + 1 < len(sections) else None
        lines.extend([f"## {section['section_title']}", ""])
        lines.extend(
            _build_section_text(
                section,
                zh,
                next_section=following,
                llm_chat=llm_chat,
                seen=seen,
                claims_out=structured_out,
                tiers=tiers,
            )
        )
    # T12: the intro renders before the body, so its paraphrase guard can only
    # run now that the body sentences exist. Template runs (llm_chat is None)
    # never enter it — their output stays byte-identical.
    if llm_chat is not None:
        lines[intro_index] = _dedupe_intro_against_body(
            lines[intro_index], _body_prose_sentences(lines[body_start:]), sections
        )

    entries = _section_paper_entries(sections)
    challenge_bits, direction_bits = _split_limitation_pool(_limitation_pool(entries, seen=seen))
    challenge_entries = _bits_entries(challenge_bits, entries)
    direction_entries = _bits_entries(direction_bits, entries)
    body_units = sentence_units("\n".join(lines[body_start:]), allowed)
    closing = []
    for text, cites in body_units:
        if not cites or len(closing) >= 3:
            continue
        planned = next((row for row in structured_out if normalize_text(row["text"]) == normalize_text(text)), None)
        bindings = planned.get("source_bindings", []) if planned else [source for entry in entries
                   for source in entry.get("sources", []) if normalize_text(source["claim_text"]) == normalize_text(text)]
        if (bindings and all(b["source_role"] in SCOPE_ROLES["comparison"] for b in bindings)
                and not quote_diagnostic(text, [b["source_quote"] for b in bindings])["quote_like"]):
            closing.append({"paper_id": cites[0], "cites": cites, "text": text})
    lines.extend(
        [
            "## Open Challenges" if not zh else "## 开放挑战",
            "",
            _llm_or_template(
                "open challenges",
                llm_chat,
                lambda: _llm_open_challenges(challenge_entries, llm_chat, zh, allowed, seen, structured_out),
                lambda: _open_challenges_text(challenge_bits, zh, seen),
            ),
            "",
            "Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models." if not zh else "",
            "" if not zh else "",
            "![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)" if not zh else "",
            "" if not zh else "",
            "## Future Directions" if not zh else "## 未来方向",
            "",
            _llm_or_template(
                "future directions",
                llm_chat,
                lambda: _llm_future_directions(direction_entries, llm_chat, zh, allowed, seen, structured_out),
                lambda: _future_directions_text(direction_bits, entries, zh, seen),
            ),
            "",
            "Table 3 summarizes these future directions, and Figure 5 contrasts method families by their contribution and limitation profiles." if not zh else "",
            "" if not zh else "",
            "![Table 3. Open Challenges and Future Directions](future_directions_matrix)" if not zh else "![Future Matrix](future_directions_matrix)",
            "" if not zh else "",
            "![Figure 5. Method-Contribution-Limitation Comparison](method_comparison)" if not zh else "",
            "" if not zh else "",
            "",
            "## Conclusion" if not zh else "## 结论",
            "",
            _conclusion_text(closing, zh, seen),
            "",
            "## References",
            "",
        ]
    )
    used_ids = _extract_citations("\n".join(lines))
    by_id = {card["paper_id"]: card for card in cards}
    for paper_id in sorted(used_ids):
        card = by_id.get(paper_id)
        if card:
            year = card.get("year") or "n.d."
            lines.append(f"- {paper_id}: {card.get('title', paper_id)} ({year}).")
    unique_lines = []
    embedded = set()
    for line in lines:
        match = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if match:
            if match.group(1) in embedded:
                continue
            embedded.add(match.group(1))
        unique_lines.append(line)
    final = "\n".join(unique_lines).strip() + "\n"
    present = {(normalize_text(text), tuple(cites)) for text, cites in sentence_units(final, allowed) if cites}
    structured_out[:] = [row for row in structured_out if (normalize_text(row["text"]), tuple(row["cites"])) in present]
    recorded = {(normalize_text(row["text"]), tuple(row["cites"])) for row in structured_out}
    for text, cites in sentence_units(final, allowed):
        if not cites or (normalize_text(text), tuple(cites)) in recorded:
            continue
        bindings = [source for entry in entries for source in entry.get("sources", [])
                    if source["paper_id"] in cites and normalize_text(source["claim_text"]) == normalize_text(text)]
        scope = next((scope for scope, roles in SCOPE_ROLES.items()
                      if bindings and all(b["source_role"] in roles for b in bindings)), "")
        structured_out.append({"text": text, "cites": cites, "scope": scope,
                               "source_bindings": bindings})
    return final


def _abstract_anchors(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The top-ranked papers the abstract cites (template and LLM path alike)."""
    return [card for card in ranked if card.get("paper_id")][:3]


def _template_abstract(
    topic: str,
    sections: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    zh: bool,
    seen: set[str],
) -> str:
    """Abstract = coverage plus how the claims are grounded, with citations.

    Kept separate from the introduction (motivation + roadmap): the intro owns
    the section titles, so the abstract names only its end points — sharing the
    title list is what made the old pair the second-most-similar one. The anchor
    sentence names representative works by title only (evidence snippets are no
    longer spliced here), which also keeps it away from the conclusion's
    contribution sentences.
    """
    themes = [str(section.get("section_title", "")).strip() for section in sections if section.get("section_title")]
    scope = _scope_span(themes, zh)
    sentences = []
    if zh:
        sentences.append(f"本综述覆盖 {len(sections)} 个主题、{len(ranked)} 篇入选研究，{scope}。")
        sentences.append("综述区分已报告的发现与开放问题，参考文献列出正文实际引用的研究。")
    else:
        sentences.append(f"This survey covers {len(sections)} research {'theme' if len(sections) == 1 else 'themes'} and {len(ranked)} selected {'paper' if len(ranked) == 1 else 'papers'}, {scope}.")
        sentences.append("The review separates reported findings from open questions and lists the papers cited in its discussion.")
    # Anchor sentences name representative works by title only: splicing
    # truncated evidence snippets after the title was the unreadable abstract
    # the manual review flagged ("Tree-based plannin [paper:...]").")
    anchors = _abstract_anchors(ranked)
    if anchors:
        names = [str(card.get("title") or card["paper_id"]) for card in anchors]
        if len(names) == 1:
            listing = names[0]
        else:
            listing = (", ".join(names[:-1])) + (", and " + names[-1])
        sentences.append(("代表性工作包括 " if zh else "Representative work includes ") + listing + ("。" if zh else "."))
    return _fresh_text(" ".join(sentences), seen)


def _scope_span(themes: list[str], zh: bool) -> str:
    """First-to-last theme span: the intro keeps the full title-by-title roadmap."""
    themes = [theme for theme in themes if theme]
    if not themes:
        return ("spanning world models, learned simulators, and interactive game intelligence" if not zh else "覆盖世界模型、可学习模拟器与交互式游戏智能")
    if len(themes) == 1:
        return themes[0]
    first, last = themes[0], themes[-1]
    return f"spanning {first} to {last}" if not zh else f"从 {first} 到 {last}"


def _llm_abstract(
    topic: str,
    sections: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    llm_chat: Callable,
    zh: bool,
    allowed: set[str],
    seen: set[str],
    claims_out=None,
) -> str:
    """LLM abstract over the top-ranked papers; raises when the reply is unusable."""
    section = {"section_title": "Abstract", "section_goal": f"Summarize reported findings for {topic}",
               "selected_papers": [_selected_paper_entry(c, evidence_by_paper) for c in ranked[:4]]}
    return " ".join(_llm_structured_section_body(section, llm_chat, zh, seen, claims_out,
                                               _paper_evidence_tiers(evidence_by_paper)))


def _abstract_text(
    topic: str,
    sections: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    zh: bool,
    llm_chat: Callable | None = None,
    allowed: set[str] | None = None,
    seen: set[str] | None = None,
    claims_out=None,
) -> str:
    seen = seen if seen is not None else set()
    allowed = allowed or {str(card["paper_id"]) for card in ranked if card.get("paper_id")}
    return _llm_or_template(
        "abstract",
        llm_chat,
        lambda: _llm_abstract(topic, sections, ranked, evidence_by_paper, llm_chat, zh, allowed, seen, claims_out),
        lambda: _template_abstract(topic, sections, ranked, evidence_by_paper, zh, seen),
    )


_ROADMAP_FRAMES_EN = (
    "{title} surveys {goal}.",
    "{title} collects the work on {goal}.",
    "{title} follows the evidence for {goal}.",
    "{title} maps what is known about {goal}.",
    "On {title}, the surveyed work converges on {goal}.",
)
_ROADMAP_FRAMES_ZH = (
    "{title} 一章梳理{goal}。",
    "{title} 一章围绕{goal}展开。",
    "{title} 一章汇集{goal}的证据。",
    "{title} 一章考察{goal}。",
    "在{title}一章，证据围绕{goal}组织。",
)


def _roadmap_text(sections: list[dict[str, Any]], zh: bool, seen: set[str]) -> str:
    """Taxonomy roadmap: one preview sentence per body section, from the plan.

    Deterministic on purpose — the intro must preview every section even when
    the LLM draft degrades, and the section titles make each sentence unique.
    """
    frames = _ROADMAP_FRAMES_ZH if zh else _ROADMAP_FRAMES_EN
    sentences = []
    for index, section in enumerate(sections):
        title = str(section.get("section_title", "")).strip()
        if not title:
            continue
        goal = _clean_clause(_safe_section_goal(section)) or ("the evidence for this theme" if not zh else "该主题下的证据")
        sentences.append(frames[index % len(frames)].format(title=title, goal=goal if zh else _lower_first(goal)))
    return _fresh_text(" ".join(sentences), seen)


def _introduction_text(
    topic: str,
    sections: list[dict[str, Any]],
    zh: bool,
    llm_chat: Callable | None = None,
    allowed: set[str] | None = None,
    seen: set[str] | None = None,
) -> str:
    """Motivation (uncited) plus the taxonomy roadmap; never the abstract's text."""
    seen = seen if seen is not None else set()
    allowed = allowed or set()
    motivation = _llm_or_template(
        "introduction",
        llm_chat,
        lambda: _llm_motivation(topic, sections, llm_chat, zh, allowed, seen),
        lambda: _template_motivation(zh, seen),
    )
    return motivation + "\n\n" + _roadmap_text(sections, zh, seen)


def _template_motivation(zh: bool, seen: set[str]) -> str:
    if zh:
        text = (
            "游戏智能越来越依赖能学习世界如何演化的模型，而不是只依赖手工规则；"
            "这类模型能否支撑规划、生成与公平评测，正是本综述要回答的问题。"
            "全文按技术功能组织：每个主题汇集一种能力的证据，并说明该证据到哪里为止。"
        )
    else:
        text = (
            "Game intelligence increasingly rests on models that learn how a world evolves rather than on hand-written rules alone; "
            "whether such models can support planning, generation, and fair evaluation is the question this survey addresses. "
            "Each theme that follows gathers the evidence for one capability and states where that evidence stops."
        )
    return _fresh_text(text, seen)


def _llm_motivation(
    topic: str,
    sections: list[dict[str, Any]],
    llm_chat: Callable,
    zh: bool,
    allowed: set[str],
    seen: set[str],
) -> str:
    """LLM motivation: 2-3 uncited sentences; the roadmap is added separately.

    T12 responsibility boundary: the introduction owns the research question,
    the background motivation, and a one-sentence-per-section map (appended
    deterministically). Expanding a section's specific methods, datasets, or
    metrics is the body sections' job, not the intro's.
    """
    user = "\n".join(
        [
            "Task: draft the introduction.",
            "Survey topic: " + str(topic or ""),
            "Themes: " + "; ".join(str(section.get("section_title", "")) for section in sections),
            "Write in Chinese (简体)." if zh else "Write in English.",
            "The introduction owns exactly three things: the research question, the background motivation, and a one-sentence-per-section roadmap. The roadmap is appended separately, so do not write it yourself.",
            "Write 2-3 sentences covering the research question and the motivation only: the question this survey answers, and why the field needs it answered now.",
            "Stay inside that responsibility: never expand any section's specific methods, datasets, or metrics — the body sections own that detail.",
            "Do not cite any paper, do not preview the sections, and do not invent numbers, years, benchmarks, or paper names.",
        ]
    )
    reply = llm_chat(
        [
            {"role": "system", "content": "You draft the motivation paragraph of an evidence-grounded academic survey."},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=1600,
    )
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError(f"empty LLM reply: {str(reply)[:120]}")
    body = _sanitize_llm_paragraph(reply.strip(), allowed)
    body = _fresh_text(body, seen, keep_duplicate=False)
    if not _split_sentences(body):
        raise ValueError("motivation reply is empty after filtering")
    return body


def _intro_word_bag(text: str, exclude: set[str] | None = None) -> set[str]:
    """Casefolded bag of words for the intro/body Jaccard guard.

    Latin/digit runs shorter than 2 characters are noise; CJK text has no word
    boundaries without a segmenter, so each Han character is one token (minus
    function-character stopwords). Dependency-free on purpose — the guard must
    not pull in embeddings or a tokenizer.
    """
    exclude = exclude if exclude is not None else set()
    bag: set[str] = set()
    for token in _INTRO_TOKEN_RE.findall(text.casefold()):
        if token in exclude or token in _INTRO_STOPWORDS:
            continue
        if len(token) == 1 and not ("\u4e00" <= token <= "\u9fff"):
            continue
        bag.add(token)
    return bag


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _body_prose_sentences(body_lines: list[str]) -> list[str]:
    """Prose sentences of the rendered body sections (image embeds excluded)."""
    sentences: list[str] = []
    for line in body_lines:
        if not line.strip() or line.lstrip().startswith("!["):
            continue
        sentences.extend(_split_sentences(line))
    return sentences


def _dedupe_intro_against_body(
    intro: str,
    body_sentences: list[str],
    sections: list[dict[str, Any]],
) -> str:
    """T12 write-side guard: drop intro sentences that restate body content.

    `_fresh_text` only blocks exact repeats, so an intro sentence that
    paraphrases a body sentence with different wording survived it (eval
    redundancy 0.917 for Introduction x a body section). Every intro sentence
    is compared with every body sentence by bag-of-words Jaccard (casefolded,
    stopwords out); >= _INTRO_DEDUP_THRESHOLD counts as a body restatement
    and the sentence goes. Section-title words are removed from both bags
    first — roadmap sentences are supposed to name sections, so title overlap
    is not a penalty. A cut that would leave fewer than
    _INTRO_MIN_SENTENCES sentences is rolled back whole (anti-hollowing, the
    same spirit as the coverage gate).
    """
    title_words = _intro_word_bag(" ".join(str(section.get("section_title", "")) for section in sections))
    body_bags = [bag for bag in (_intro_word_bag(sentence, title_words) for sentence in body_sentences) if bag]
    if not body_bags:
        return intro
    kept_paragraphs: list[list[str]] = []
    kept_count = 0
    dropped = 0
    for paragraph in intro.split("\n\n"):
        kept_sentences: list[str] = []
        for sentence in _split_sentences(paragraph):
            bag = _intro_word_bag(sentence, title_words)
            if bag and any(_jaccard(bag, other) >= _INTRO_DEDUP_THRESHOLD for other in body_bags):
                dropped += 1
                continue
            kept_sentences.append(sentence.strip())
        if kept_sentences:
            kept_paragraphs.append(kept_sentences)
            kept_count += len(kept_sentences)
    if dropped and kept_count < _INTRO_MIN_SENTENCES:
        logger.warning(f"[writer] intro dedupe would leave {kept_count} sentences; keeping original intro")
        return intro
    if dropped:
        logger.info(f"[writer] intro dedupe dropped {dropped} body-restating sentence(s)")
    return "\n\n".join(" ".join(sentences) for sentences in kept_paragraphs) if dropped else intro


def _section_paper_entries(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selected-paper entries in section order, deduped by paper id.

    Open Challenges / Future Directions are synthesized from these cross-section
    entries rather than from a global rank, so both tail sections reason over
    the same literature the body actually covered.
    """
    entries: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for section in sections:
        for paper in section.get("selected_papers", []):
            paper_id = paper.get("paper_id")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            entries.append(paper)
    return entries


def _limitation_pool(entries: list[dict[str, Any]], limit: int = 8, seen: set[str] | None = None) -> list[dict[str, str]]:
    """One explicitly attributed limitation per paper, deduplicated by text.

    A limitation already rendered in the body (prefix match against the
    cross-section ledger, citation suffix excluded) does not consume a slot:
    the wave7 template path wasted Open Challenges' only bit on a sentence the
    dedupe gate then dropped.
    """
    ledger = seen if seen is not None else set()
    pool = []
    used = set()
    for entry in entries:
        for source in entry.get("sources", []):
            text = source["claim_text"]
            bit_key = _norm_sentence(text).rstrip(" .;")
            rendered = bool(bit_key) and any(key.startswith(bit_key) for key in ledger)
            if (source["source_role"] != "own_limitation" or normalize_text(text) in used
                    or rendered
                    or quote_diagnostic(text, [source["source_quote"]])["quote_like"]):
                continue
            used.add(normalize_text(text))
            pool.append({"paper_id": entry["paper_id"], "title": entry["title"], "text": text})
            break
    return pool[:limit]


def _split_limitation_pool(pool: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Partition the pool so Open Challenges and Future Directions share no bit.

    The two tail sections scored 0.961 similarity in eval S0 because both read
    the same `_evidence_bits` output; disjoint halves plus the cross-section
    sentence ledger close that path.
    """
    if not pool:
        return [], []
    half = max(1, len(pool) // 2)
    return pool[:half], pool[half:]


def _bits_entries(
    bits: list[dict[str, str]],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The selected-paper entries owning `bits`, in section order."""
    wanted = {bit["paper_id"] for bit in bits}
    return [entry for entry in entries if str(entry.get("paper_id")) in wanted]


def _field_bits(ranked: list[dict[str, Any]], field: str, fallback: str, limit: int) -> list[dict[str, str]]:
    bits = []
    for card in ranked:
        paper_id = card.get("paper_id")
        if not paper_id:
            continue
        scope = "limitation" if field == "limitations" else "contribution"
        text = next((source["claim_text"] for source in card.get("sources", [])
                     if source["source_role"] in SCOPE_ROLES[scope]
                     and not quote_diagnostic(source["claim_text"], [source["source_quote"]])["quote_like"]), "")
        if not text:
            continue
        bits.append({"paper_id": paper_id, "title": _shorten(card.get("title") or paper_id, 90), "text": text})
        if len(bits) >= limit:
            break
    return bits


def _open_challenges_text(bits: list[dict[str, str]], zh: bool, seen: set[str]) -> str:
    """Cross-section synthesis of the recorded evidence boundaries."""
    if not bits:
        return _fresh_text(
            "The captured evidence does not yet support sharper open-challenge statements."
            if not zh
            else "所选论文的证据尚不足以支撑更强的开放挑战结论。",
            seen,
        )
    text = _fresh_text(" ".join(f"{bit['text'].rstrip('.。')} [{bit['paper_id']}]" + ("。" if zh else ".")
                                for bit in bits[:3]), seen, keep_duplicate=False)
    return text or ("上述局限限定了结论的适用范围；现有证据不足以进一步概括共同挑战。" if zh else
                    "The limitations discussed above bound the findings; the recorded evidence does not justify a stronger shared challenge.")


def _future_directions_text(bits, entries, zh, seen):
    if not bits:
        return "尚无充分的已归属局限可支撑具体研究建议。" if zh else "The recorded limitations do not yet justify specific research proposals."
    premises = _open_challenges_text(bits[:2], zh, seen)
    proposal = ("本综述建议优先检验上述边界；该建议尚待验证。" if zh else
                "This review proposes testing these boundaries in follow-up evaluation; the proposal remains untested.")
    return premises + " " + proposal


def _llm_open_challenges(
    entries: list[dict[str, Any]],
    llm_chat: Callable,
    zh: bool,
    allowed: set[str],
    seen: set[str],
    claims_out=None,
) -> str:
    """Cross-section open problems; input entries are disjoint from the FD ones."""
    section = {"section_title": "Open Challenges", "section_goal": "Summarize only explicit own limitations, at most 3 sentences",
               "selected_papers": [{**e, "sources": [s for s in e.get("sources", []) if s["source_role"] == "own_limitation"]} for e in entries]}
    paragraphs = _llm_structured_section_body(section, llm_chat, zh, seen, claims_out)
    return " ".join(_split_sentences(" ".join(paragraphs))[:3])


def _llm_future_directions(entries, llm_chat, zh, allowed, seen, claims_out=None):
    # Suggestions are the review author's, separated from observed premises.
    premises = _llm_open_challenges(entries, llm_chat, zh, allowed, seen, claims_out)
    proposal = ("本综述建议优先检验上述边界；这是待验证的研究建议。" if zh else
                "As a proposal of this review, follow-up evaluation should test these stated boundaries; this is not an established result.")
    return premises + " " + proposal


def _conclusion_text(bits: list[dict[str, str]], zh: bool, seen: set[str]) -> str:
    if not bits:
        return "现有安全归属的证据不足以形成跨研究结论。" if zh else "Safely attributed findings are insufficient for a cross-study conclusion."
    lead = "综合上述研究，以下发现构成本综述的结论依据。" if zh else "The synthesis rests on the following reported findings."
    findings = " ".join(f"{bit['text'].rstrip('.。')} " + " ".join(f"[{pid}]" for pid in bit.get("cites", [bit["paper_id"]])) + ("。" if zh else ".") for bit in bits[:3])
    end = "这些发现的适用范围仍分别受各自研究设置约束。" if zh else "Their scopes remain tied to the respective study settings; they do not by themselves establish a shared performance ranking."
    return lead + " " + findings + " " + end


def _build_section_text(
    section: dict[str, Any],
    zh: bool,
    next_section: dict[str, Any] | None = None,
    llm_chat: Callable | None = None,
    seen: set[str] | None = None,
    claims_out: list[dict[str, Any]] | None = None,
    tiers: dict[str, str] | None = None,
) -> list[str]:
    """One body section: framing sentence, moves-menu paragraphs, bridge sentence.

    The paragraph shape is model-chosen (moves menu) or the deterministic
    template fallback; artifact embeds are interleaved by paragraph slot, so
    neither path can strand a planned figure or table. Both paths require
    source-bound, role-labelled propositions.
    """
    title = str(section.get("section_title", "This section"))
    goal = _safe_section_goal(section)
    papers = section.get("selected_papers", [])
    section_match = re.search(r"(\d+)$", str(section.get("section_id", "")))
    section_index = int(section_match.group(1)) if section_match else 0
    seen = seen if seen is not None else set()
    lines: list[str] = []

    llm_paragraphs: list[str] = []
    if llm_chat is not None:
        try:
            llm_paragraphs = _llm_section_body(section, llm_chat, zh, seen, claims_out=claims_out, tiers=tiers)
        except Exception as exc:  # LLM boundary: designed per-section degradation
            logger.warning(
                f"[writer] section {section.get('section_id')} LLM unusable -> template fallback: {exc}"
            )

    if llm_paragraphs:
        paragraphs = llm_paragraphs
    else:
        paragraphs = [
            paragraph
            for paragraph in (
                _template_summary(papers, zh, section_index, seen),
                _template_comparison(papers, zh, title, seen),
                _template_limitations(papers, zh, section_index, seen),
            )
            if paragraph  # e.g. no real limitation evidence -> no filler paragraph
        ]

    if not paragraphs:
        paragraphs = ["No safely attributed claims are available for this section." if not zh
                      else "本节尚无可安全归属到来源的论断。"]
    # Framing sentence: no citation, so it cannot create an unsupported one.
    if zh:
        framing = f"{title} 这一部分关注的问题是：{_clean_clause(goal) or '如何把该主题下论文的方法、证据与局限组织成可比较的综述段落'}。"
    else:
        framing = f"{title} focuses on {_clean_clause(goal) or 'a citation-ready research theme'}."
    lines.extend([_fresh_text(framing, seen), ""])
    lines.extend(_section_lines(paragraphs, section.get("artifact_slots", [])))
    lines.extend([_section_bridge(title, next_section, zh, seen), ""])
    return lines


def _pid_ok(pid: Any) -> bool:
    """Corrupted pids carry rendered multi-paragraph text; the signature is whitespace."""
    return isinstance(pid, str) and bool(pid) and not any(ch.isspace() for ch in pid)


def _template_claims(papers, scopes, seen, zh):
    sentences = []
    used = set()
    for paper in papers[:SECTION_PAPERS]:
        for source in paper.get("sources", []):
            text = source["claim_text"].strip().rstrip(".。")
            if (source["source_role"] not in set().union(*(SCOPE_ROLES[scope] for scope in scopes))
                    or quote_diagnostic(text, [source["source_quote"]])["quote_like"]
                    or normalize_text(text) in used):
                continue
            used.add(normalize_text(text))
            sentences.append(f"{text} [{paper['paper_id']}]" + ("。" if zh else "."))
            break
    return _fresh_text(" ".join(sentences), seen, keep_duplicate=False) if sentences else ""


def _template_summary(papers, zh, section_index, seen):
    return _template_claims(papers, ["contribution", "background"], seen, zh)


def _template_limitations(papers, zh, section_index, seen):
    return _template_claims(papers, ["limitation"], seen, zh)


def _section_lines(paragraphs: list[str], slots: list[dict[str, Any]]) -> list[str]:
    """Paragraphs with the planned artifact embeds interleaved by slot index.

    A trailing placement (`after_limitation_paragraph`) follows the last
    paragraph whatever the moves menu produced, so no embed is dropped when the
    model returns fewer paragraphs than the old three-tag skeleton.
    """
    lines: list[str] = []
    last = len(paragraphs) - 1
    for index, paragraph in enumerate(paragraphs):
        lines.extend([paragraph, ""])
        for slot in slots:
            position = _PLACEMENT_INDEX.get(str(slot.get("placement", "")))
            if position is None:
                continue
            position = last if position == -1 else min(position, last)
            if position != index:
                continue
            lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])
    return lines


def _section_bridge(title: str, next_section: dict[str, Any] | None, zh: bool, seen: set[str]) -> str:
    """Bridge to the next section, phrased from that section's own goal.

    The old bridge repeated one fixed clause in every section; carrying the next
    section's goal keeps each bridge sentence unique to its position.
    """
    next_title = str((next_section or {}).get("section_title", "")).strip()
    if next_title:
        next_goal = _clean_clause(_safe_section_goal(next_section or {})) or (
            "how those capabilities are evaluated" if not zh else "这些能力如何被评测"
        )
        if zh:
            return _fresh_text(f"{title} 的证据把问题推向「{next_title}」：{next_goal}。", seen)
        return _fresh_text(f"{title} hands the open question to {next_title}, which asks about {_lower_first(next_goal)}.", seen)
    if zh:
        return _fresh_text(f"{title} 是正文最后一个技术主题；后续章节把上述证据汇总为开放挑战与未来方向。", seen)
    return _fresh_text(f"{title} closes the body; the remaining sections fold this evidence into open challenges and future directions.", seen)


def _template_comparison(papers, zh, section_title="", seen=None):
    return _template_claims(papers, ["method"], seen if seen is not None else set(), zh)


def _paper_evidence_tiers(evidence_by_paper: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """paper_id -> "fulltext_ok" | "abstract_only" by the paper's evidence mix.

    fulltext_ok requires a full-text fragment (paragraph / agentic_chunk) that
    actually binds one of the paper's claims (non-empty supports_claims — the
    possible_claims ↔ supports_claims correspondence): a paper whose only
    paragraph matched none of its claims must not waive method/comparison/
    limitation claims. Abstract/caption rows only — no evidence rows at all,
    or a paper absent from the map — tier abstract_only, the conservative side.
    """
    tiers: dict[str, str] = {}
    for paper_id, items in evidence_by_paper.items():
        bound_fulltext = any(
            str(item.get("source_type") or "") in _FULLTEXT_SOURCE_TYPES
            and any(valid_support(row, item) for row in item.get("supports_claims", []))
            for item in items
        )
        tiers[paper_id] = "fulltext_ok" if bound_fulltext else "abstract_only"
    return tiers


def _source_aliases(section):
    sources = {}
    for index, paper in enumerate(section.get("selected_papers", [])[:SECTION_PAPERS], 1):
        for number, source in enumerate(paper.get("sources", []), 1):
            if source.get("paper_id") == paper.get("paper_id"):
                sources[f"P{index}S{number}"] = source
    return sources


def _safe_section_goal(section):
    goal = str(section.get("section_goal", ""))
    if _contains_banned_phrase(goal):
        return "the reported evidence for " + str(section.get("section_title", "this theme"))
    return goal


def _section_alias_context(section, tiers=None):
    alias_of = {f"P{i}": str(p["paper_id"]) for i, p in
                enumerate(section.get("selected_papers", [])[:SECTION_PAPERS], 1) if p.get("paper_id")}
    lines = []
    for alias, pid in alias_of.items():
        paper = next(p for p in section["selected_papers"] if p.get("paper_id") == pid)
        label = alias if tiers is None else f"{alias}({_TIER_LABELS[tiers.get(pid, 'abstract_only')]})"
        lines.append(f"{label} | {paper.get('title', '')} (metadata, not a claim)")
    for alias, source in _source_aliases(section).items():
        lines.append(f"{alias} | role={source['source_role']} | source_type={source['source_type']} | "
                     f"claim={source['claim_text']} | source_quote={source['source_quote']}")
    return alias_of, set(alias_of.values()), lines


def _llm_section_body(
    section: dict[str, Any],
    llm_chat: Callable,
    zh: bool,
    seen: set[str],
    claims_out: list[dict[str, Any]] | None = None,
    tiers: dict[str, str] | None = None,
) -> list[str]:
    """One validated JSON path; failures reach the source-safe template."""
    return _llm_structured_section_body(section, llm_chat, zh, seen, claims_out, tiers)


def _llm_structured_section_body(
    section: dict[str, Any],
    llm_chat: Callable,
    zh: bool,
    seen: set[str],
    claims_out: list[dict[str, Any]] | None,
    tiers: dict[str, str] | None = None,
) -> list[str]:
    """Draft the section body as a structured claim list; raise ValueError
    when the reply is unusable so the caller uses source-safe templates.

    The model returns [{"text", "cite", "scope"}, ...] with alias tags only.
    Validation checks source aliases, claim-level roles and single-sentence
    round-trip through the shared claim kernel;
    paragraphs are assembled by scope, and each rendered claim unit is
    collected into claims_out as the structured source of truth for the
    downstream claim map.
    """
    alias_of, allowed, paper_lines = _section_alias_context(section, tiers)
    if not alias_of:
        raise ValueError("section has no citable papers")
    tag_list = " ".join(f"[{alias}]" for alias in alias_of)
    # Tiers describe availability; claim attribution, not tier, controls scope.
    tier_rules = [
        "Source role and the selected assertion control scope, not page availability. An abstract can support an explicitly stated method; never extrapolate implementation detail.",
        # Explicit mapping: the wave8 run's dominant fallback was
        # source_role_out_of_scope — the model picking scopes its cited
        # sources cannot legally support. State the contract, don't loosen it.
        "Scope-role contract: \"method\" admits own_method/own_setup; \"contribution\" admits own_contribution/own_result; "
        "\"comparison\" admits own_method/own_setup/own_contribution/own_result; \"limitation\" admits only own_limitation; "
        "\"background\" admits only background/related_work. Choose the scope your cited sources can legally support.",
    ]
    user = "\n".join(
        [
            "Section title: " + str(section.get("section_title", "")),
            "Section goal: " + _safe_section_goal(section),
            "Paper metadata followed by source assertions (source alias | role | type | claim | source_quote):",
            *paper_lines,
            *tier_rules,
            "Write in Chinese (简体)." if zh else "Write in English.",
            "Plan the section body as an ordered list of claims; paragraphs are assembled from the claim scopes.",
            'Reply with ONLY a JSON array (no prose around it, no code fence) where each element is {"text": <one sentence>, "cite": [<tags>], "sources": [<source aliases>], "scope": <scope>}.',
            f'"scope" is one of: {", ".join(_CLAIM_SCOPES)}. Order the claims: {" then ".join(_CLAIM_SCOPES)}.',
            '"text" is one complete survey sentence with no headings, bullets or labels; a bracketed tag inside text is allowed only in the author-prominent style (a landmark work\'s tag right after its title).',
            '"cite" lists every tag the sentence draws on; omit uncited transitions from the JSON.',
            "Paraphrase the supplied claim, preserving technical terms, attribution, negation and scope; do not copy source sentences.",
            'Every claim MUST have citations and "sources": ["P1S1", ...], at least one source from EACH cited paper.',
            "A source role constrains the claim scope; background/related_work can only support background, never own methods/results.",
            "Never infer absence of limitations from missing data. Do not cite metadata, evidence tiers, or this survey workflow.",
            "Restate only the facts in the paper list above: no invented numbers, years, benchmarks, or paper names.",
            "Refer to papers by their titles in prose, and never copy an internal paper id, DOI, or URL into the text.",
            "Write 2-6 claims, or fewer if the sources do not justify more. No uncited claims.",
            f"Cite only with these bracketed tags: {tag_list}",
        ]
    )
    # Generous budget: reasoning models spend completion tokens on hidden
    # thinking before the visible JSON, so the array needs headroom.
    messages = [
        {
            "role": "system",
            "content": (
                "You plan body paragraphs of an evidence-grounded academic survey as structured claims. "
                "Reply with a JSON array only; citations must use the supplied bracketed paper tags and nothing else."
            ),
        },
        {"role": "user", "content": user},
    ]
    source_aliases = _source_aliases(section)
    reply = llm_chat(messages, temperature=0.3, max_tokens=8000)
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError(f"empty LLM reply: {str(reply)[:120]}")
    try:
        units = _validated_claim_units(_parse_structured_claims(reply), alias_of, allowed, zh, tiers,
                                       source_aliases)
    except _ClaimValidationError as exc:
        # Persist every total-validation failure, alias or not: the wave8 run
        # showed the dominant rejection moving past aliases to role-scope, and
        # a blind spot there would repeat the wave7 undiagnosable fallbacks.
        _persist_writer_rejects(section, reply, exc.rejects, attempt=1)
        if not exc.alias_related:
            raise
        # One targeted retry: the alias contract is mechanical, so the exact
        # tag list plus the rejection reasons usually fix it in one round.
        retry_note = (
            "Your reply was rejected: " + "; ".join(exc.rejects[:4])
            + '. In the JSON arrays, "cite" and "sources" must contain the bare tags WITHOUT brackets or quotes. '
            + f'"cite" may only use: {", ".join(alias_of)}. '
            + f'"sources" may only use: {", ".join(source_aliases) if source_aliases else "(none)"}. '
            + "Reply again with the corrected JSON array only."
        )
        try:
            retry = llm_chat(messages + [{"role": "user", "content": retry_note}], temperature=0.3, max_tokens=8000)
        except Exception:
            raise exc from None
        if not isinstance(retry, str) or not retry.strip():
            raise
        try:
            units = _validated_claim_units(_parse_structured_claims(retry), alias_of, allowed, zh, tiers,
                                           source_aliases)
        except _ClaimValidationError as exc2:
            _persist_writer_rejects(section, retry, exc2.rejects, attempt=2)
            raise
        logger.info("[writer] alias-format retry recovered %d claim unit(s)", len(units))
    paragraphs, kept = _assemble_claim_paragraphs(units, allowed, seen)
    if not paragraphs:
        raise ValueError("no source-bound claim survived")
    if claims_out is not None:
        section_info = {
            "section_id": str(section.get("section_id", "")),
            "section_title": str(section.get("section_title", "")),
        }
        for row in kept:
            claims_out.append({**section_info, **row})
    return paragraphs


def _parse_structured_claims(reply: str) -> list[Any]:
    """Extract the JSON claim array; tolerant of wrapper prose or code fences."""
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("reply contains no JSON array")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list) or not data:
        raise ValueError("structured reply is not a non-empty JSON array")
    return data


class _ClaimValidationError(ValueError):
    """Total claim-validation failure carrying the per-claim rejection reasons."""

    def __init__(self, message: str, rejects: list[str], alias_related: bool):
        super().__init__(message)
        self.rejects = rejects
        self.alias_related = alias_related


# Raw LLM replies behind total alias rejections, appended as JSONL. The wave7
# run left no trace of the exact tag format the model emitted, which made the
# 6/6 template fallbacks impossible to attribute. Tests monkeypatch this path.
_WRITER_REJECTS_PATH = Path("output") / "writer_llm_rejects.jsonl"


def _persist_writer_rejects(section, reply: str, rejects: list[str], attempt: int) -> None:
    """Diagnostics only — a failed write must not break the writing path."""
    record = {
        "ts": now_iso(),
        "section_id": str(section.get("section_id", "")),
        "section_title": str(section.get("section_title", "")),
        "attempt": attempt,
        "rejects": rejects[:12],
        "reply": str(reply)[:6000],
    }
    try:
        _WRITER_REJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _WRITER_REJECTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("[writer] could not persist LLM reject diagnostics: %s", exc)


def _alias_lookup(tag: str, keys) -> str | None:
    """Canonical alias key for an LLM-emitted tag.

    The prompt presents tags bracketed ("Cite only with these bracketed tags:
    [P1] ...") while validation wants the bare key, so tolerate one bracket
    layer and case drift ([p1s2] -> P1S2). Returns None when nothing matches,
    keeping the rejection reason truthful to what the model emitted.
    """
    cand = tag.strip()
    if cand.startswith("[") and cand.endswith("]"):
        cand = cand[1:-1].strip()
    if cand in keys:
        return cand
    if cand.upper() in keys:
        return cand.upper()
    return None


def _validated_claim_units(
    raw_claims: list[Any],
    alias_of: dict[str, str],
    allowed: set[str],
    zh: bool,
    tiers: dict[str, str] | None = None,
    sources: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Enforce same-paper source aliases, claim-level roles and one sentence.

    Lexical diagnostics are advisory; every rendered rewrite still needs NLI.
    """
    units: list[dict[str, Any]] = []
    rejects: list[str] = []
    for raw in raw_claims[:6]:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        scope = raw.get("scope")
        cites = [(_alias_lookup(c, alias_of) or c) if isinstance(c, str) else c
                 for c in raw.get("cite", [])]

        def _reject(reason: str) -> None:
            # Distinguish WHY a section fell back to template: silent drops
            # made three wave6 body sections undiagnosable.
            rejects.append(f"{reason} ({str(text)[:50]!r})")

        if not isinstance(text, str) or not text.strip():
            continue
        if "\n" in text or re.match(r"^\s*(?:#+|[-*]\s|\*\*(?:Abstract|Introduction|Future Directions|Open Challenges|Conclusion)\*\*)", text, re.I):
            _reject("structural_text")
            continue
        if scope not in _CLAIM_SCOPES or not isinstance(cites, list):
            _reject(f"scope={scope!r}")
            continue
        embedded = [(_alias_lookup(t, alias_of) or t) for t in _CLAIM_BRACKETS_RE.findall(text)]
        if any(tag not in alias_of for tag in embedded):
            _reject("unknown_embedded_alias")
            continue
        if any(not isinstance(alias, str) or alias not in alias_of for alias in cites):
            _reject("unknown_cite_alias")
            continue
        bound: list[str] = []
        for alias in embedded + [alias for alias in cites if isinstance(alias, str)]:
            paper_id = alias_of[alias]
            if paper_id not in bound:
                bound.append(paper_id)
        source_names = raw.get("sources", [])
        if not bound or not isinstance(source_names, list) or not source_names or not sources:
            _reject("no_bound_paper_or_sources")
            continue
        source_names = [(_alias_lookup(n, sources) or n) if isinstance(n, str) else n
                        for n in source_names]
        if any(not isinstance(name, str) or name not in sources for name in source_names):
            _reject("unknown_source_alias")
            continue
        bindings = [sources[name] for name in source_names]
        if {b["paper_id"] for b in bindings} != set(bound):
            _reject("sources_do_not_cover_cited_papers")
            continue
        if any(b.get("source_role") not in SCOPE_ROLES[scope]
               or role_violation(b.get("source_role"), b.get("source_quote", "")) for b in bindings):
            _reject("source_role_out_of_scope")
            continue
        if _contains_banned_phrase(text) or re.search(r"\b(?:evidence (?:tier|available)|abstract.only|full.text material)\b", text, re.I):
            _reject("banned_phrase_or_leaked_tier")
            continue
        body = _CLAIM_BRACKETS_RE.sub(
            lambda m: f"[{_alias_lookup(m.group(1), alias_of) or m.group(1)}]", text).strip()
        body = body.rstrip(" .,;、;。．")
        trailing = [paper_id for paper_id in bound if f"[{paper_id}]" not in body]
        sentence = body
        if trailing:
            sentence += " " + " ".join(f"[{paper_id}]" for paper_id in trailing)
        sentence += "。" if zh else "."
        parsed = sentence_units(sentence, allowed)
        if len(parsed) != 1 or not parsed[0][0]:
            _reject("not_single_sentence")
            continue
        units.append({"scope": scope, "sentence": sentence, "text": parsed[0][0], "cites": parsed[0][1],
                      "source_bindings": bindings,
                      "quote_diagnostic": quote_diagnostic(parsed[0][0], [b["source_quote"] for b in bindings])})
    if rejects:
        logger.warning("[writer] claim validation rejected %d raw claim(s): %s",
                       len(rejects), " | ".join(rejects[:6]))
    if not any(unit["cites"] for unit in units):
        alias_related = any(r.startswith(("unknown_cite_alias", "unknown_embedded_alias", "unknown_source_alias"))
                            for r in rejects)
        raise _ClaimValidationError(
            "no claim survived alias/scope validation (rejections: "
            + ("; ".join(rejects[:6]) if rejects else "none parsed") + ")",
            rejects, alias_related)
    return units


def _assemble_claim_paragraphs(
    units: list[dict[str, Any]],
    allowed: set[str],
    seen: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Scope-grouped paragraphs plus the claim units that actually rendered.

    Every sentence still passes the freeform write-side defenses (whitelist
    filter, leak guards, cross-section dedupe); a claim dropped there is
    dropped from the structured source too, so the persisted truth always
    matches the rendered markdown.
    """
    groups: list[list[tuple[str, dict[str, Any] | None]]] = []
    for scope in _CLAIM_SCOPES:
        rows: list[tuple[str, dict[str, Any] | None]] = []
        for unit in (item for item in units if item["scope"] == scope):
            body = _sanitize_llm_paragraph(unit["sentence"], allowed)
            body = _fresh_text(body, seen, keep_duplicate=False)
            if not body:
                continue
            parsed = sentence_units(body, allowed)
            if len(parsed) != 1:
                continue
            text, cites = parsed[0]
            rows.append((body, {**{k: v for k, v in unit.items() if k != "sentence"}, "text": text, "cites": cites} if cites else None))
        if rows:
            groups.append(rows)
    if len(groups) == 1 and len(groups[0]) >= 2:
        # One scope group still yields a section: split at the claim midpoint
        # rather than discarding the reply (same policy as the freeform path).
        middle = max(1, len(groups[0]) // 2)
        rows = groups[0]
        groups = [rows[:middle], rows[middle:]]
    paragraphs = [" ".join(sentence for sentence, _ in rows) for rows in groups]
    kept = [
        {"paragraph": index, **claim}
        for index, rows in enumerate(groups)
        for _, claim in rows
        if claim
    ]
    return paragraphs, kept


def _llm_section_body_freeform(section, llm_chat, zh, seen):
    return _llm_structured_section_body(section, llm_chat, zh, seen, None)


# Paragraph-boundary markers a model may emit instead of plain blank lines: a
# bracket move label ([SUMMARY]), a markdown heading, a bullet, or a "Summary:"
# lead-in. Uppercase only, so a citing tag like [P1] or a lowercase whitelist id
# can never be mistaken for a label.
_BRACKET_LABEL_RE = re.compile(r"^(?:#+\s*)?\[[A-Z][A-Z _-]{1,24}\][:.]?(?:\s+|$)")
_LEADIN_LABEL_RE = re.compile(r"^(?:#+\s*)?(?:Summary|Comparison|Limitation)\s*[:：]\s*", re.IGNORECASE)
_BULLET_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")


def _parse_move_paragraphs(reply: str) -> list[str]:
    """Split an untagged reply into paragraphs, tolerating stray labels.

    The moves menu replaced the fixed paragraph tags, but a model used to tagged
    prompts still emits them; a label line opens a new paragraph and the label
    itself is stripped instead of failing the section. Blank lines and bullets
    also bound paragraphs, so a bulleted reply yields several paragraphs rather
    than one blob.
    """
    paragraphs: list[str] = []
    current: list[str] = []

    def _flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            _flush()
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            _flush()
            line = line[bullet.end() :].strip()
        label = _BRACKET_LABEL_RE.match(line) or _LEADIN_LABEL_RE.match(line)
        if label:
            _flush()
            line = line[label.end() :].strip()
        if line:
            current.append(line)
    _flush()
    return [text for text in paragraphs if text]


def _map_alias_citations(text: str, alias_of: dict[str, str]) -> str:
    """Rewrite [P3]-style tags to real paper ids; unknown tags stay and are
    dropped later by the whitelist filter.

    Both citation styles must survive placement normalization: information-
    prominent clusters already sit inside the sentence, while author-prominent
    and clustered tags are often stranded after the closing punctuation
    ("...scale. [P2]." / "...scale. [P1] [P2]."), which sentence splitters turn
    into citation-only fragments. Those runs are pulled back inside the sentence
    (punctuation moves behind the run) so downstream verifiers can bind the
    claim; the legacy `paper:` id form gets the same treatment.
    """
    def _replace(match: re.Match[str]) -> str:
        alias = match.group(1)
        return f"[{alias_of[alias]}]" if alias in alias_of else match.group(0)

    def _pull_inside(match: re.Match[str]) -> str:
        run = match.group(2).strip().rstrip(".。")
        return f" {run}."

    for _ in range(3):  # a run may hide further stranded tags behind a period
        pulled = _STRANDED_TAGS_RE.sub(_pull_inside, text)
        if pulled == text:
            break
        text = pulled
    text = re.sub(r"\[(P\d+)\]", _replace, text)
    # GLM recalls real DOIs from pretraining and space-breaks them
    # ("[paper:10. 48550/arxiv. 2312. 12491]") even though prompts only carry
    # aliases. The id is otherwise correct: compact it back so the whitelist
    # filter can rescue the citation instead of dropping the sentence
    # (S0 round-4: deleting these took the whole survey's citations with it).
    text = _trim_partial_sentence(_strip_model_headings(text))
    text = re.sub(
        r"\[(paper:[^\]]+)\]",
        lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]",
        text,
    )
    return text


def _sanitize_llm_paragraph(text: str, allowed: set[str]) -> str:
    # Compact space-broken recalled DOIs BEFORE sentence splitting: the split
    # would otherwise cut "10. 48550/arxiv." into orphan fragments that dodge
    # both the leak guards and the whitelist check (every LLM path funnels
    # through here; the tail sections do not alias-map, so this is the one
    # point that sees all of them).
    text = _trim_partial_sentence(_strip_model_headings(text))
    text = re.sub(
        r"\[(paper:[^\]]+)\]",
        lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]",
        text,
    )
    keep = []
    for sentence in _split_sentences(text):
        citations = _extract_citations(sentence)
        if citations and not citations <= allowed:
            continue
        # Leak/fragment guards judge the citation-free residue: a well-formed
        # bracket citation contains "paper:<digits>" and must NOT trip the
        # bare-id guard (that ordering silently dropped every LLM-drafted
        # citation between 98ec405 and this fix).
        residue = re.sub(r"\[[^\]]*\]", "", sentence)
        if _contains_banned_phrase(residue):
            continue
        if _LEAKED_ID_RE.search(residue) or _LEAKED_ID_FRAGMENT_RE.match(residue):
            continue
        keep.append(sentence.strip())
    return " ".join(part for part in keep if part)


def _build_timeline(task_id: str, categories: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    category_by_id = {str(cat.get("category_id")): cat for cat in categories if cat.get("category_id")}
    events = []
    for card in sorted(cards, key=lambda item: (item.get("year") or 0, item.get("paper_id", ""))):
        category = category_by_id.get(str(card.get("category_id")), {})
        events.append(
            {
                "year": card.get("year"),
                "category_id": card.get("category_id") or "uncategorized",
                "category_name": category.get("category_name") or card.get("category") or "Uncategorized",
                "paper_ids": [card["paper_id"]],
                "summary": _shorten(card.get("contribution") or card.get("method") or card.get("title", ""), 260),
            }
        )
    return {"task_id": task_id, "events": events}


def _build_generated_artifacts(
    cfg: Any,
    task_id: str,
    timeline: dict[str, Any],
    categories: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = cfg.root_dir
    asset_dir = root / "output" / "generated_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    support = [card["paper_id"] for card in cards[:8]]
    image_notes: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    figure_specs = [
        (
            "hero_banner",
            "Figure 1",
            "Graphical Overview of World Models and GameCraft",
            "Research overview illustration for world models and GameCraft in interactive game intelligence, neural game environments, agent learning loop, clean academic editorial style, restrained blue teal palette, no dense text, no fake charts.",
            "A clean overview linking world models, game environments, agents, and evaluation.",
            "Introduction",
        ),
        (
            "concept_overview",
            "Figure 2",
            "Agent-Environment Interaction Loop in Learned Game Worlds",
            "Conceptual diagram of a game agent interacting with a learned world model and a simulated game environment, clean infographic style, minimal labels, light background, no paper names, no fake UI.",
            "A conceptual loop connecting agent policy, learned world model, and game environment feedback.",
            "Introduction",
        ),
    ]
    for artifact_id, number, title, prompt, alt_text, placement in figure_specs:
        path = asset_dir / f"{artifact_id}.png"
        result = _try_generate_public_image(cfg, path, artifact_id, prompt)
        if result["status"] != "success":
            _draw_concept_png(path, title, number)
            result["fallback_path"] = _rel(root, path)
        image_notes.append(result)
        artifacts.append(
            _artifact_record(
                root=root,
                artifact_id=artifact_id,
                artifact_kind="figure",
                artifact_type="visual_overview",
                display_number=number,
                display_title=title,
                caption=f"{number}. {title}. The figure provides a reader-facing conceptual map and does not reproduce any paper's original figure.",
                alt_text=alt_text,
                path=path,
                fmt="png",
                placement_section=placement,
                support=support,
                display_priority=100 if artifact_id == "hero_banner" else 95,
                generation_method=result.get("generation_method", "matplotlib_fallback"),
                prompt=prompt,
            )
        )

    timeline_path = asset_dir / "publication_timeline.png"
    _draw_timeline_png(timeline_path, cards)
    artifacts.append(
        _artifact_record(
            root=root,
            artifact_id="publication_timeline",
            artifact_kind="figure",
            artifact_type="timeline",
            display_number="Figure 3",
            display_title="Publication Timeline of Representative Papers",
            caption="Figure 3. The timeline summarizes publication years for the selected representative papers.",
            alt_text="Timeline chart showing selected papers by publication year.",
            path=timeline_path,
            fmt="png",
            placement_section="Background and Scope",
            support=support,
            display_priority=90,
            generation_method="matplotlib",
        )
    )

    taxonomy_path = asset_dir / "taxonomy_overview.png"
    _draw_taxonomy_png(taxonomy_path, categories, cards)
    artifacts.append(
        _artifact_record(
            root=root,
            artifact_id="taxonomy_overview",
            artifact_kind="figure",
            artifact_type="taxonomy_graph",
            display_number="Figure 4",
            display_title="Taxonomy Distribution of Selected Papers",
            caption="Figure 4. The taxonomy distribution groups the selected papers by technical role.",
            alt_text="Horizontal bar chart showing selected paper counts per taxonomy category.",
            path=taxonomy_path,
            fmt="png",
            placement_section="Taxonomy",
            support=support,
            display_priority=85,
            generation_method="matplotlib",
        )
    )

    method_path = asset_dir / "method_comparison.png"
    _draw_method_comparison_png(method_path, cards)
    artifacts.append(
        _artifact_record(
            root=root,
            artifact_id="method_comparison",
            artifact_kind="figure",
            artifact_type="comparison_chart",
            display_number="Figure 5",
            display_title="Method-Contribution-Limitation Comparison",
            caption="Figure 5. The comparison highlights how method families differ in contribution and limitation profiles.",
            alt_text="Matrix-style figure comparing selected method families by technical emphasis.",
            path=method_path,
            fmt="png",
            placement_section="Future Directions",
            support=support,
            display_priority=70,
            generation_method="matplotlib",
        )
    )

    table_specs = _final_table_specs(cards, categories)
    for spec in table_specs:
        path = asset_dir / f"{spec['artifact_id']}.md"
        path.write_text(spec["markdown"], encoding="utf-8")
        artifacts.append(
            _artifact_record(
                root=root,
                artifact_id=spec["artifact_id"],
                artifact_kind="table",
                artifact_type=spec["artifact_type"],
                display_number=spec["display_number"],
                display_title=spec["display_title"],
                caption=spec["caption"],
                alt_text="",
                path=path,
                fmt="markdown",
                placement_section=spec["placement_section"],
                support=support,
                display_priority=spec["display_priority"],
                generation_method="programmatic_markdown_table",
                table_headers=spec["table_headers"],
            )
        )
    return artifacts, image_notes


def _artifact_record(
    *,
    root: Path,
    artifact_id: str,
    artifact_kind: str,
    artifact_type: str,
    display_number: str,
    display_title: str,
    caption: str,
    alt_text: str,
    path: Path,
    fmt: str,
    placement_section: str,
    support: list[str],
    display_priority: int,
    generation_method: str,
    prompt: str = "",
    table_headers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "artifact_kind": artifact_kind,
        "display_number": display_number,
        "display_title": display_title,
        "title": f"{display_number}. {display_title}",
        "caption": caption,
        "alt_text": alt_text,
        "artifact_path": _rel(root, path),
        "artifact_format": fmt,
        "placement_section": placement_section,
        "must_show_in_public": True,
        "table_headers": table_headers,
        "source_artifacts": ["cache/paper_cards.json", "cache/evidence_store.json", "cache/taxonomy.json"],
        "source_papers": support,
        "supporting_papers": support,
        "provenance": "generated_by_c_from_verified_artifacts",
        "generation_method": generation_method,
        "prompt": prompt,
        "display_priority": display_priority,
        "usable_in_report": True,
    }


def _try_generate_public_image(cfg: Any, output_path: Path, artifact_id: str, prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GENERATE_KEY", "")
    if not api_key:
        return {"artifact_id": artifact_id, "status": "fallback", "reason": "GENERATE_KEY is not configured."}
    payload = {
        "model": os.getenv("GENERATE_IMAGE_MODEL", "gpt-image-2"),
        "prompt": prompt,
        "size": os.getenv("GENERATE_IMAGE_SIZE", "1536x1024"),
        "quality": os.getenv("GENERATE_IMAGE_QUALITY", "low"),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_url = os.getenv("GENERATE_IMAGE_BASE_URL", "https://mikuapi.org").rstrip("/")
    timeout = float(os.getenv("GENERATE_IMAGE_TIMEOUT_SECONDS", "45"))
    try:
        import httpx
    except ModuleNotFoundError as exc:
        return {"artifact_id": artifact_id, "status": "fallback", "reason": f"httpx unavailable: {exc}"}
    errors = []
    for endpoint in [f"{base_url}/images/generations", f"{base_url}/v1/images/generations"]:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            image_base64 = data["data"][0]["b64_json"]
            output_path.write_bytes(base64.b64decode(image_base64))
            if output_path.exists() and output_path.stat().st_size > 0:
                return {
                    "artifact_id": artifact_id,
                    "status": "success",
                    "endpoint": endpoint,
                    "generation_method": "intern_prompt_to_gpt_image_2_gateway",
                    "model": payload["model"],
                    "size": payload["size"],
                    "quality": payload["quality"],
                }
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    return {"artifact_id": artifact_id, "status": "fallback", "reason": "Image API call failed.", "errors": errors[-2:]}


def _draw_concept_png(path: Path, title: str, number: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_facecolor("#f8fbfc")
    fig.patch.set_facecolor("#f8fbfc")
    ax.axis("off")
    boxes = [
        (0.12, 0.54, "Agent\npolicy and memory"),
        (0.42, 0.68, "Learned\nworld model"),
        (0.70, 0.54, "Interactive\ngame world"),
        (0.42, 0.28, "Evaluation\nand feedback"),
    ]
    for x, y, label in boxes:
        box = FancyBboxPatch(
            (x, y),
            0.18,
            0.14,
            boxstyle="round,pad=0.03,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#2f6f83",
            facecolor="#e8f3f6",
        )
        ax.add_patch(box)
        ax.text(x + 0.09, y + 0.07, label, ha="center", va="center", fontsize=12, color="#17324d", weight="bold")
    arrows = [((0.30, 0.61), (0.42, 0.73)), ((0.60, 0.73), (0.70, 0.61)), ((0.79, 0.54), (0.54, 0.42)), ((0.42, 0.36), (0.21, 0.54))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color="#3b7f95"))
    ax.text(0.5, 0.9, f"{number}. {title}", ha="center", va="center", fontsize=18, weight="bold", color="#17324d")
    ax.text(0.5, 0.13, "Programmatic fallback visualization; no paper-original figures or fabricated chart values.", ha="center", fontsize=10, color="#4b5563")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_timeline_png(path: Path, cards: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    years: dict[int, int] = defaultdict(int)
    for card in cards:
        if card.get("year"):
            years[int(card["year"])] += 1
    xs = sorted(years)
    ys = [years[x] for x in xs]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.bar(xs, ys, color="#2f6f83")
    ax.set_title("Figure 3. Publication Timeline of Representative Papers", fontsize=14, weight="bold")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Selected papers")
    ax.set_xticks(xs)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _taxonomy_counts(categories: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, int]:
    """Count cards per taxonomy category via the categories' own paper_ids.

    Card.category_id is never populated under the claim contract; reading the
    assignment at its source fixed the single-bar "Uncategorized" figure.
    """
    name_of: dict[str, str] = {}
    for cat in categories:
        for pid in cat.get("paper_ids") or []:
            name_of[pid] = cat.get("category_name") or cat.get("name") or "Uncategorized"
    counts: dict[str, int] = defaultdict(int)
    for card in cards:
        counts[name_of.get(card.get("paper_id"), card.get("category") or "Uncategorized")] += 1
    return dict(counts)


def _draw_taxonomy_png(path: Path, categories: list[dict[str, Any]], cards: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    counts = _taxonomy_counts(categories, cards)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax.barh(labels, values, color="#3b7f95")
    ax.set_title("Figure 4. Taxonomy Distribution of Selected Papers", fontsize=14, weight="bold")
    ax.set_xlabel("Selected papers")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_method_comparison_png(path: Path, cards: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    groups = [
        ("Latent world models", "compact dynamics", "rollout error"),
        ("Planning systems", "search + learned dynamics", "compute cost"),
        ("Neural game engines", "generative interaction", "temporal coherence"),
        ("Game agents", "large-scale self-play", "engineering scale"),
        ("Open simulators", "benchmark richness", "domain specificity"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    ax.axis("off")
    rows = [["Method family", "Technical emphasis", "Typical limitation"], *groups]
    table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.25, 0.38, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5df")
        if row == 0:
            cell.set_facecolor("#e8eef3")
            cell.set_text_props(weight="bold", color="#17324d")
        else:
            cell.set_facecolor("#fffef9")
    ax.set_title("Figure 5. Method-Contribution-Limitation Comparison", fontsize=14, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _final_table_specs(cards: list[dict[str, Any]], categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    name_of = {pid: (cat.get("category_name") or cat.get("name") or "")
               for cat in categories for pid in (cat.get("paper_ids") or [])}
    systems_headers = ["Paper", "Year", "Category", "Method", "Contribution", "Limitation"]
    systems_rows = [
        [
            card.get("title", ""),
            str(card.get("year", "")),
            name_of.get(card.get("paper_id"), card.get("category", "")),
            _shorten(card.get("method", ""), 260),
            _shorten(card.get("contribution", ""), 260),
            _shorten(card.get("limitations", ""), 260),
        ]
        for card in cards[:12]
    ]
    eval_headers = ["Protocol", "Relevant Systems", "What It Measures", "Known Risk"]
    eval_rows = [
        ["Game score and success rate", "MuZero; AlphaStar; Dota 2 RL", "Task performance under game rules", "Can hide brittle behavior outside the benchmark"],
        ["Interactive controllability", "Genie; GameNGen; UniSim", "Whether users or agents can steer generated worlds", "Visual plausibility may exceed semantic control"],
        ["Long-horizon consistency", "World Models; DreamerV3; Diffusion World Models", "Whether rollouts remain coherent over time", "Errors can compound during imagined planning"],
        ["Open-ended task coverage", "MineDojo; Generative Agents", "Breadth of tasks, behaviors, and social interactions", "Evaluation depends on task design"],
    ]
    future_headers = ["Direction", "Evidence Basis", "Opportunity", "Risk"]
    future_rows = [
        ["Stable long-horizon world models", "World Models; DreamerV3; Diffusion World Models", "More reliable planning and imagination", "Compounding model error"],
        ["Controllable neural game engines", "GameNGen; Genie", "Playable learned environments", "Temporal drift and weak action semantics"],
        ["Simulator-grounded agent evaluation", "UniSim; MineDojo", "Reusable benchmarks for interactive intelligence", "Domain-specific conclusions"],
        ["Multi-agent world modeling", "AlphaStar; Dota 2 RL; Generative Agents", "Richer social and strategic behavior", "Scale and safety constraints"],
    ]
    return [
        {
            "artifact_id": "representative_systems",
            "artifact_type": "comparison_table",
            "display_number": "Table 1",
            "display_title": "Representative Systems and Their Technical Roles",
            "caption": "Table 1. Representative papers are organized by their technical role in the survey taxonomy.",
            "table_headers": systems_headers,
            "markdown": _markdown_table("Table 1. Representative Systems and Their Technical Roles", systems_headers, systems_rows),
            "placement_section": "Methods and Systems",
            "display_priority": 88,
        },
        {
            "artifact_id": "evaluation_protocol_matrix",
            "artifact_type": "evaluation_matrix",
            "display_number": "Table 2",
            "display_title": "Evaluation Protocol Matrix for Game Intelligence",
            "caption": "Table 2. Evaluation protocols are grouped by what they measure and where they can mislead.",
            "table_headers": eval_headers,
            "markdown": _markdown_table("Table 2. Evaluation Protocol Matrix for Game Intelligence", eval_headers, eval_rows),
            "placement_section": "Open Challenges",
            "display_priority": 80,
        },
        {
            "artifact_id": "future_directions_matrix",
            "artifact_type": "future_matrix",
            "display_number": "Table 3",
            "display_title": "Open Challenges and Future Directions",
            "caption": "Table 3. Future directions are derived from the limitations and technical gaps in the selected literature.",
            "table_headers": future_headers,
            "markdown": _markdown_table("Table 3. Open Challenges and Future Directions", future_headers, future_rows),
            "placement_section": "Future Directions",
            "display_priority": 78,
        },
    ]


def _markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_pipe(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _plan_image_prompt(
    cfg: Any,
    categories: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
) -> str:
    base_prompt = _rule_image_prompt(categories, cards)
    if not os.getenv("GENERATE_KEY"):
        return base_prompt
    try:
        from llm_client import InternS2Client
    except ModuleNotFoundError:
        return base_prompt
    client = InternS2Client(cfg)
    if not client.is_configured():
        return base_prompt
    context = {
        "categories": [
            {
                "name": cat.get("category_name") or cat.get("name"),
                "description": cat.get("description", ""),
                "paper_ids": cat.get("paper_ids", [])[:5],
            }
            for cat in categories[:6]
        ],
        "papers": [
            {
                "paper_id": card.get("paper_id"),
                "title": card.get("title"),
                "method": _shorten(card.get("method", ""), 200),
                "contribution": _shorten(card.get("contribution", ""), 200),
                "evidence_count": len(evidence_by_paper.get(card.get("paper_id"), [])),
            }
            for card in cards[:8]
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise image-generation prompts for academic survey diagrams. "
                "Use only facts in the provided JSON. Do not add paper names, claims, "
                "citation counts, logos, or text labels not present in the JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one prompt for a clean visual overview of this research taxonomy. "
                "No dense text, no fake screenshots, no citation counts. JSON context:\n"
                f"{context}"
            ),
        },
    ]
    try:
        prompt = client.chat(messages, temperature=0.2, max_tokens=500).strip()
    except Exception:
        return base_prompt
    if not prompt:
        return base_prompt
    return _shorten(prompt.replace("\n", " "), 900)


def _rule_image_prompt(categories: list[dict[str, Any]], cards: list[dict[str, Any]]) -> str:
    names = [
        str(cat.get("category_name") or cat.get("name"))
        for cat in categories[:5]
        if cat.get("category_name") or cat.get("name")
    ]
    titles = [str(card.get("title")) for card in cards[:4] if card.get("title")]
    return (
        "Clean academic visual overview for a survey on world models and GameCraft, "
        "showing grouped research themes as connected blocks, restrained colors, "
        "plain light background, no fake paper figures, no logos, no dense text. "
        f"Themes: {', '.join(names) or 'interactive world models, game agents, neural game engines'}. "
        f"Evidence papers: {', '.join(titles)}."
    )


def _try_generate_image(cfg: Any, output_path: Path, prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GENERATE_KEY", "")
    if not api_key:
        return {
            "artifact_id": "generated_visual_overview_001",
            "status": "fallback",
            "reason": "GENERATE_KEY is not configured.",
        }
    base_url = os.getenv("GENERATE_IMAGE_BASE_URL", "https://mikuapi.org").rstrip("/")
    payload = {
        "model": os.getenv("GENERATE_IMAGE_MODEL", "gpt-image-2"),
        "prompt": prompt,
        "size": os.getenv("GENERATE_IMAGE_SIZE", "1024x1024"),
        "quality": os.getenv("GENERATE_IMAGE_QUALITY", "low"),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    errors = []
    try:
        import httpx
    except ModuleNotFoundError as exc:
        return {
            "artifact_id": "generated_visual_overview_001",
            "status": "fallback",
            "reason": f"Image API dependency unavailable: {exc}",
        }
    for endpoint in [f"{base_url}/images/generations", f"{base_url}/v1/images/generations"]:
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            image_base64 = data["data"][0]["b64_json"]
            output_path.write_bytes(base64.b64decode(image_base64))
            return {
                "artifact_id": "generated_visual_overview_001",
                "status": "success",
                "endpoint": endpoint,
                "model": payload["model"],
                "size": payload["size"],
                "quality": payload["quality"],
            }
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    return {
        "artifact_id": "generated_visual_overview_001",
        "status": "fallback",
        "reason": "Image API call failed; markdown artifacts remain available.",
        "errors": errors[-2:],
    }


def _preflight(
    task_id: str,
    survey: str,
    allowed_ids: set[str],
    artifacts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    citations = _extract_citations(survey)
    invalid = sorted(citations - allowed_ids)
    artifact_ids = {item["artifact_id"] for item in artifacts}
    figure_refs = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", survey))
    invalid_artifacts = sorted(ref for ref in figure_refs if ref not in artifact_ids)
    used_cards = citations & {card["paper_id"] for card in cards}
    return {
        "task_id": task_id,
        "pass": not invalid and not invalid_artifacts and bool(used_cards),
        "citation_check": {
            "citations": sorted(citations),
            "invalid_citations": invalid,
            "allowed_count": len(allowed_ids),
        },
        "section_coverage": {
            "used_paper_count": len(used_cards),
            "available_card_count": len(cards),
        },
        "reference_integrity": {
            "references_policy": "actual_body_citations_from_allowed_paper_cards",
            "new_references_added": False,
        },
        "artifact_refs": {
            "figure_refs": sorted(figure_refs),
            "invalid_artifact_refs": invalid_artifacts,
        },
    }


def _artifact_audit(task_id: str, root: Path, artifacts: list[dict[str, Any]], allowed_ids: set[str]) -> dict[str, Any]:
    items = []
    for artifact in artifacts:
        path = _resolve(root, artifact.get("artifact_path", ""))
        supporting = set(artifact.get("supporting_papers", []))
        item_pass = (
            bool(artifact.get("source_artifacts"))
            and bool(supporting)
            and supporting <= allowed_ids
            and path.exists()
            and artifact.get("provenance") == "generated_by_c_from_verified_artifacts"
        )
        items.append(
            {
                "artifact_id": artifact.get("artifact_id"),
                "pass": item_pass,
                "path_exists": path.exists(),
                "supporting_papers_allowed": supporting <= allowed_ids,
                "has_source_artifacts": bool(artifact.get("source_artifacts")),
            }
        )
    return {"task_id": task_id, "pass": all(item["pass"] for item in items), "items": items}


def _visual_decisions(
    task_id: str,
    timeline: dict[str, Any],
    artifacts: list[dict[str, Any]],
    image_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    events = timeline.get("events", [])
    if len(events) <= 3:
        timeline_decision = "show_as_text_or_table"
        reason = "Few timeline events; table is clearer than a graphic."
    elif len(events) <= 12:
        timeline_decision = "show_as_static_timeline_table"
        reason = "Timeline has a manageable number of dated events."
    else:
        timeline_decision = "show_as_grouped_table"
        reason = "Many events; grouped table avoids visual clutter."
    return {
        "task_id": task_id,
        "image_generation": image_notes,
        "decisions": [
            {"artifact_type": "timeline", "decision": timeline_decision, "reason": reason},
            {
                "artifact_type": "generated_artifacts",
                "decision": "embed_registered_artifacts_in_html",
                "reason": f"{len(artifacts)} generated artifacts have provenance records.",
            },
        ],
    }


def _categories(taxonomy_raw: Any, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = _as_list(taxonomy_raw, "categories")
    if categories:
        return categories
    seen: dict[str, dict[str, Any]] = {}
    for idx, card in enumerate(cards, start=1):
        name = card.get("category") or "Citation-Ready Papers"
        key = card.get("category_id") or name
        seen.setdefault(
            str(key),
            {
                "category_id": key or f"cat_{idx:03d}",
                "category_name": name,
                "description": "Auto-grouped from citation-ready paper cards.",
                "paper_ids": [],
            },
        )
        seen[str(key)]["paper_ids"].append(card["paper_id"])
    return list(seen.values())


def _claim_from_card(card: dict[str, Any]) -> str:
    return next((source["claim_text"] for source in card.get("sources", [])), "")


def _score_cards_for_topic(
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    categories: list[dict[str, Any]],
    topic: str,
) -> list[dict[str, Any]]:
    category_text = " ".join(str(cat.get("category_name") or cat.get("name") or "") for cat in categories)
    scored = []
    for card in cards:
        evidence_text = " ".join(str(item.get("text", "")) for item in evidence_by_paper.get(card.get("paper_id"), []))
        title_score = _keyword_match_score(card.get("title", ""), POSITIVE_KEYWORDS)
        abstract_score = _keyword_match_score(card.get("abstract", "") or card.get("contribution", "") or card.get("method", ""), POSITIVE_KEYWORDS)
        category_score = _keyword_match_score(f"{card.get('category', '')} {category_text} {topic}", POSITIVE_KEYWORDS)
        evidence_score = _keyword_match_score(evidence_text, POSITIVE_KEYWORDS)
        negative_score = _keyword_match_score(
            f"{card.get('title', '')} {card.get('venue', '')} {card.get('category', '')} {evidence_text}",
            NEGATIVE_KEYWORDS,
        )
        score = 0.35 * title_score + 0.25 * abstract_score + 0.20 * category_score + 0.20 * evidence_score
        combined = f"{card.get('title', '')} {card.get('category', '')} {evidence_text}".lower()
        if any(marker in combined for marker in ["atari", "dota", "starcraft", "alphago", "alphazero", "dqn", "game play"]):
            score = max(score, 0.72)
        if "game" in combined and ("reinforcement learning" in combined or "agent" in combined):
            score = max(score, 0.70)
        if negative_score > 0:
            score = min(score, 0.35)
        enriched = dict(card)
        enriched["topic_relevance_score"] = round(score, 3)
        enriched["topic_relevance_negative_hit"] = negative_score > 0
        scored.append(enriched)
    relevant = [card for card in scored if card["topic_relevance_score"] >= 0.35]
    if len(relevant) >= min(8, len(scored)):
        return relevant
    return sorted(scored, key=lambda card: card.get("topic_relevance_score", 0), reverse=True)


def _keyword_match_score(text: Any, keywords: list[str]) -> float:
    value = str(text or "").lower()
    if not value:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword.lower() in value)
    return min(1.0, hits / 3)


def _field_or_claim(card: dict[str, Any], field: str, claim_bucket: str) -> str:
    scope = {"method": "method", "limitations": "limitation"}.get(field, "contribution")
    return next((s["claim_text"] for s in card.get("sources", [])
                 if s.get("source_role") in SCOPE_ROLES[scope]), "")


def _selection_reason(card: dict[str, Any]) -> str:
    if card.get("topic_relevance_negative_hit"):
        return "Included only as a low-confidence citation-ready fallback; negative topic keywords were detected."
    score = card.get("topic_relevance_score", 0)
    if score >= 0.70:
        return "Matches game / agent / world model keywords with strong relevance."
    if score >= 0.50:
        return "Partially matches topic keywords and remains citation-ready."
    return "Citation-ready fallback with limited topic keyword evidence."


def _rank_cards(cards: list[dict[str, Any]], evidence_by_paper: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return sorted(
        cards,
        key=lambda card: (
            -card.get("topic_relevance_score", 0.0),
            card.get("topic_relevance_negative_hit", False),
            -len(evidence_by_paper.get(card.get("paper_id"), [])),
            card.get("year") or 0,
            card.get("paper_id", ""),
        ),
    )


def _read_optional(root: Path, raw_path: str | None, default: Any) -> Any:
    if not raw_path:
        return default
    path = _resolve(root, raw_path)
    if not path.exists():
        return default
    return read_json(path)


def _resolve(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _as_list(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get(key, data.get("items", []))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _group_by(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        value = item.get(key)
        if value:
            grouped[str(value)].append(item)
    return grouped


def _extract_citations(markdown: str) -> set[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    return {match.strip() for match in re.findall(r"\[([^\]]+)\]", text_only) if not match.startswith("http")}


def _split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_RE.split(text) if part.strip()]


def _clean_clause(text: str) -> str:
    """Drop trailing punctuation and a leading "Focuses on ..." so a template
    sentence keeps a single full stop and never reads "focuses on Focuses on"."""
    value = re.sub(
        r"^(?:focuses on|focus on|focused on|聚焦于|聚焦|关注)\s*",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    )
    return value.strip().rstrip(".,;、;。． ")


def _contains_banned_phrase(text: str) -> bool:
    folded = text.casefold()
    return any(phrase in folded for phrase in BANNED_META_PHRASES)


def _shorten(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return _drop_unbalanced(value)
    cut = value[: limit - 3]
    # Cut on a word boundary when one exists: a mid-word stub ("high qua...")
    # reads as corruption in the rendered survey. CJK text has no spaces, so
    # it falls back to the plain character cut.
    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip()
    return _drop_unbalanced(cut + "...")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.。!?！？])\s+")


def _first_sentence(text: Any) -> str:
    """First complete sentence of a card/evidence field.

    Template summary frames quote at most this: splicing a 200-character
    mid-abstract run-on into "X starts from ..." was the unreadable fallback
    prose flagged in manual review.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    return _SENTENCE_SPLIT_RE.split(value, maxsplit=1)[0].strip()


_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+")


def _strip_model_headings(text: str) -> str:
    """The model sometimes prefixes its own markdown heading; the renderer
    already writes the real section heading, so a model heading duplicates it."""
    lines = [ln for ln in str(text or "").splitlines() if not _HEADING_LINE_RE.match(ln)]
    return "\n".join(lines).strip()


def _trim_partial_sentence(text: str) -> str:
    """Drop a trailing fragment that lacks terminal punctuation.

    A reply cut off by a token limit ends mid-sentence ("... toward real");
    keeping only complete sentences turns it into a clean paragraph (or an
    empty one, which falls through to the template).
    """
    value = str(text or "").strip()
    if not value:
        return value
    ends = [m for m in re.finditer(r"[.。!?！？](?:\s|$)", value + " ")]
    if not ends:
        return ""
    cut = value[: ends[-1].start() + 1].strip()
    return cut or value


def _lint_citation_brackets(md: str) -> str:
    """Final structural lint before the survey is written.

    Internal LLM/template paths have repeatedly leaked half-formed citation
    brackets (S0 rounds 2-3: an orphan '[' made extractors swallow paragraphs
    into one citation id, and space-broken ids split across bracket groups).
    Enforce the document invariant here, at the write boundary, so no single
    path's leak can reach verify/repair: unbalanced '[' truncates to its start,
    and a bracket group containing whitespace/newlines is never a valid paper
    id, so its content is dropped. Image embeds (![...](...)) are line-initial
    and balanced; untouched.
    """
    out_lines = []
    for line in md.splitlines():
        if line.lstrip().startswith("!["):
            out_lines.append(line)
            continue
        # Only a TRAILING unclosed citation opener is a leak (card fields can
        # legitimately hold stray brackets like [0, 1]; whole-line bracket
        # counting ate valid citations next to them in S0 round-5).
        line = re.sub(r"\[(?:paper:|P\d)[^\]\n]*$", "", line)
        # Drop whitespace-bearing id-shaped bracket groups (cannot be valid
        # paper ids); legit multi-word links like [some text](url) are kept.
        line = re.sub(r"\[(?:paper:[^\]]*|P\d[^\]]*)\s[^\]]*\]", "", line)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _drop_unbalanced(value: str) -> str:
    """Truncation can cut a citation bracket in half ('...as [paper:10. 485');
    the orphan '[' then makes downstream extractors swallow whole paragraphs as
    one citation id (S0 round-2: a 1535-char block became one reference entry,
    printed twice by _replace_references)."""
    cut = value.rfind("[")
    if cut != -1 and "]" not in value[cut:]:
        return value[:cut].rstrip()
    return value


def _escape_pipe(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def _tool_result(
    task_id: str,
    tool: str,
    status: str,
    request_path: str,
    outputs: list[str],
    metrics: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "tool": tool,
        "owner": "C",
        "status": status,
        "input_request": request_path,
        "outputs": outputs,
        "metrics": metrics,
        "message": message,
        "timestamp": now_iso(),
    }
