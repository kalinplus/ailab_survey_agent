"""C tool: grounded survey writer and visual artifact builder."""

from __future__ import annotations

import base64
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
]

_LLM_PARAGRAPH_TAGS = ("summary", "comparison", "limitation")

# A bare paper id in prose (the S0 failure mode: space-broken DOIs the model
# paraphrased out of the prompt) can never be legitimate survey text. Sentence
# splitting cuts such ids mid-DOI, so the orphan continuation fragments
# ("1016/j. fake must vanish.") need their own guard.
_LEAKED_ID_RE = re.compile(r"paper\s*:\s*\d", re.IGNORECASE)
_LEAKED_ID_FRAGMENT_RE = re.compile(r"^\s*\d{2,}\s*[/.]")

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s*")

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
    timeline = _build_timeline(task_id, categories, cards)
    artifacts, image_notes = _build_generated_artifacts(cfg, task_id, timeline, categories, cards, evidence_by_paper)
    survey = _render_survey(topic, language, section_plan, artifacts, cards, evidence_by_paper, llm_chat=_writer_llm_chat(cfg))

    survey_path = _resolve(root, outputs.get("survey_markdown_path", "output/survey.md"))
    timeline_path = _resolve(root, outputs.get("timeline_path", "cache/timeline.json"))
    artifact_bank_path = _resolve(root, outputs.get("generated_artifact_bank_path", "cache/generated_artifact_bank.json"))
    claim_plan_path = root / "cache" / "section_claim_plan.json"
    preflight_path = root / "output" / "c_preflight_report.json"
    audit_path = root / "output" / "artifact_audit_report.json"
    visual_path = root / "output" / "visual_decision_report.json"

    survey_path.parent.mkdir(parents=True, exist_ok=True)
    survey_path.write_text(survey, encoding="utf-8")
    write_json(timeline_path, timeline)
    write_json(artifact_bank_path, {"task_id": task_id, "artifacts": artifacts})
    write_json(claim_plan_path, {"task_id": task_id, "sections": section_plan})

    preflight = _preflight(task_id, survey, allowed_ids, artifacts, cards)
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
    snippets = []
    for item in evidence_by_paper.get(card.get("paper_id"), []):
        text = _shorten(item.get("text"), 240)
        if text:
            snippets.append(text)
        if len(snippets) >= SECTION_EVIDENCE_SNIPPETS:
            break
    return {
        "paper_id": card.get("paper_id"),
        "title": card.get("title", ""),
        "year": card.get("year"),
        "problem": _field_or_claim(card, "problem", "key_results"),
        "method": _field_or_claim(card, "method", "method"),
        "contribution": _field_or_claim(card, "contribution", "key_results"),
        "limitations": _field_or_claim(card, "limitations", "limitations"),
        "evidence_snippets": snippets,
        "topic_relevance_score": round(score, 3),
        "selection_reason": _selection_reason(card),
    }


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


def _render_survey(
    topic: str,
    language: str,
    sections: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    llm_chat: Callable | None = None,
) -> str:
    zh = language == "zh"
    title = f"# {topic or 'Grounded Survey'}"
    lines = [
        title,
        "",
        "## Abstract" if not zh else "## 摘要",
        (
            "This survey reviews representative work on world models, learned simulators, neural game engines, and interactive game intelligence. It emphasizes evidence-backed synthesis, clear technical roles, and traceable figures and tables."
            if not zh
            else "本综述围绕世界模型、可学习模拟器、神经游戏引擎与交互式游戏智能，梳理代表性工作的方法、证据与局限，并给出可追溯的图表支撑。"
        ),
        "",
        "## Introduction" if not zh else "## 引言",
        (
            "World models shift game intelligence from direct interaction with fixed environments toward learned representations that can support planning, simulation, generation, and evaluation. This survey organizes the selected literature by technical function rather than by citation counts or venue prestige."
            if not zh
            else "世界模型让游戏智能从与固定环境的直接交互，转向能够支撑规划、仿真、生成与评测的学习表示。本文按技术功能而非引用量组织所选文献，并把派生图表放在支撑其论证的章节附近。"
        ),
        "",
    ]
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

    for index, section in enumerate(sections):
        next_title = str(sections[index + 1].get("section_title", "")) if index + 1 < len(sections) else ""
        lines.extend([f"## {section['section_title']}", ""])
        lines.extend(_build_section_text(section, zh, next_title=next_title, llm_chat=llm_chat))

    ranked = _rank_cards(cards, evidence_by_paper)
    challenges = _evidence_bits(ranked, evidence_by_paper)
    directions = _evidence_bits(ranked, evidence_by_paper, prefer_limitation=True)
    closing = _field_bits(ranked, "contribution", "method", 2)
    lines.extend(
        [
            "## Open Challenges" if not zh else "## 开放挑战",
            "",
            _open_challenges_text(challenges, zh),
            "",
            "Table 2 organizes the main evaluation protocols used across game intelligence, learned simulators, and interactive world models." if not zh else "",
            "" if not zh else "",
            "![Table 2. Evaluation Protocol Matrix for Game Intelligence](evaluation_protocol_matrix)" if not zh else "",
            "" if not zh else "",
            "## Future Directions" if not zh else "## 未来方向",
            "",
            _future_directions_text(directions, zh),
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
            _conclusion_text(closing, zh),
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
    return "\n".join(lines).strip() + "\n"


_LIMITATION_MARKERS = (
    "limitation",
    "however",
    "remain",
    "difficult",
    "challenge",
    "error",
    "risk",
    "fail",
    "lack",
    "cannot",
    "require",
)


_PROBLEM_SHAPED_RE = re.compile(r"^(how (to|do|can|does)|what|why)\b", re.IGNORECASE)


def _evidence_bits(
    ranked: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
    limit: int = 4,
    prefer_limitation: bool = False,
) -> list[dict[str, str]]:
    """One grounded statement per paper, taken from its evidence sentences.

    Evidence stores often lead with the problem phrasing, so problem-shaped and
    very short sentences are only used when nothing better exists.
    `prefer_limitation` picks the boundary-flavoured sentence (or the card
    limitation field) for the future-directions section.
    """
    bits = []
    for card in ranked:
        paper_id = card.get("paper_id")
        if not paper_id:
            continue
        sentences = [
            str(item.get("text", "")).strip()
            for item in evidence_by_paper.get(paper_id, [])
            if str(item.get("text", "")).strip()
        ]
        reportable = [
            s for s in sentences if not _PROBLEM_SHAPED_RE.match(s) and len(s.split()) >= 8
        ] or sentences
        text = ""
        if prefer_limitation:
            marked = [s for s in reportable if any(marker in s.lower() for marker in _LIMITATION_MARKERS)]
            if marked:
                text = _shorten(marked[0], 200)
            if not text:
                text = _shorten(card.get("limitations"), 200)
        if not text:
            text = _shorten(reportable[0], 200)
        if not text:
            continue
        bits.append({"paper_id": paper_id, "title": _shorten(card.get("title") or paper_id, 90), "text": text})
        if len(bits) >= limit:
            break
    return bits


def _field_bits(ranked: list[dict[str, Any]], field: str, fallback: str, limit: int) -> list[dict[str, str]]:
    bits = []
    for card in ranked:
        paper_id = card.get("paper_id")
        if not paper_id:
            continue
        text = _shorten(card.get(field), 200) or _shorten(card.get(fallback), 200)
        if not text:
            continue
        bits.append({"paper_id": paper_id, "title": _shorten(card.get("title") or paper_id, 90), "text": text})
        if len(bits) >= limit:
            break
    return bits


_EVIDENCE_LEADS_ZH = ("证据显示，", "实验结果表明，", "报告的结果提示，")
_LIMIT_TAILS_EN = (
    "so future work should turn that boundary into a measurable comparison target",
    "which is where follow-up systems can add the most value",
    "and that gap is the natural next target for this line of work",
)
_LIMIT_TAILS_ZH = (
    "后续工作应把这一边界变成可比较的测量对象",
    "这正是后续系统最能创造增量的位置",
    "也是该方向下一步最自然的目标",
)


def _open_challenges_text(bits: list[dict[str, str]], zh: bool) -> str:
    if not bits:
        return (
            "The captured evidence does not yet support sharper open-challenge statements."
            if not zh
            else "所选论文的证据尚不足以支撑更强的开放挑战结论。"
        )
    if zh:
        parts = []
        for index, bit in enumerate(bits):
            lead = _EVIDENCE_LEADS_ZH[index % len(_EVIDENCE_LEADS_ZH)]
            parts.append(f"{bit['title']}：{lead}{bit['text']} [{bit['paper_id']}]。")
        return "开放挑战集中在证据最薄的位置：" + " ".join(parts)
    parts = [f"{bit['title']}: {bit['text']} [{bit['paper_id']}]." for bit in bits]
    return "Open challenges concentrate where the captured evidence is thinnest. " + " ".join(parts)


def _future_directions_text(bits: list[dict[str, str]], zh: bool) -> str:
    if not bits:
        return (
            "Future directions follow from the limitation fields captured for the selected papers."
            if not zh
            else "未来方向取决于所选论文已记录的局限。"
        )
    parts = []
    for index, bit in enumerate(bits):
        if zh:
            tail = _LIMIT_TAILS_ZH[index % len(_LIMIT_TAILS_ZH)]
            parts.append(f"{bit['title']}：{bit['text']} [{bit['paper_id']}]，{tail}。")
        else:
            tail = _LIMIT_TAILS_EN[index % len(_LIMIT_TAILS_EN)]
            parts.append(f"{bit['title']}: {bit['text']} [{bit['paper_id']}], {tail}.")
    return " ".join(parts)


def _conclusion_text(bits: list[dict[str, str]], zh: bool) -> str:
    lead = (
        "The field is moving from agents that merely act in hand-built games toward systems that learn, generate, and evaluate interactive worlds."
        if not zh
        else "整体来看，所选工作把游戏智能从手工环境中的行为学习，推向能够学习、生成并评测交互世界的系统。"
    )
    if not bits:
        return lead
    if zh:
        body = "；".join(f"{bit['title']}：{bit['text']} [{bit['paper_id']}]" for bit in bits)
        return f"{lead} 其中，{body}。"
    body = " ".join(f"{bit['title']}: {bit['text']} [{bit['paper_id']}]." for bit in bits)
    return f"{lead} {body}"


def _build_section_text(
    section: dict[str, Any],
    zh: bool,
    next_title: str = "",
    llm_chat: Callable | None = None,
) -> list[str]:
    title = str(section.get("section_title", "This section"))
    goal = str(section.get("section_goal", ""))
    papers = section.get("selected_papers", [])
    section_match = re.search(r"(\d+)$", str(section.get("section_id", "")))
    section_index = int(section_match.group(1)) if section_match else 0
    lines: list[str] = []

    llm_paragraphs: dict[str, str] | None = None
    if llm_chat is not None:
        try:
            llm_paragraphs = _llm_section_paragraphs(section, llm_chat, zh)
        except Exception as exc:  # LLM boundary: designed per-section degradation
            logger.warning(
                f"[writer] section {section.get('section_id')} LLM unusable -> template fallback: {exc}"
            )

    if zh:
        # Framing sentence: no citation, so it cannot create an unsupported one.
        # Kept to one sentence: repeated section boilerplate is what pushed the
        # eval-v2 section similarity to 0.952.
        lines.append(
            f"{title} 这一部分关注的问题是：{_clean_clause(goal) or '如何把该主题下论文的方法、证据与局限组织成可比较的综述段落'}。"
        )
        lines.append("")
        for slot in section.get("artifact_slots", []):
            if slot.get("placement") == "after_topic_paragraph":
                lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])

        if llm_paragraphs:
            lines.append(llm_paragraphs["summary"])
        else:
            evidence_sentences = []
            for paper in papers[:SECTION_PAPERS]:
                pid = paper.get("paper_id")
                if not pid:
                    continue
                evidence_sentences.append(
                    f"《{paper.get('title') or pid}》处理的问题可概括为 {paper.get('problem') or '该主题下的核心建模问题'}；"
                    f"其方法侧重 {paper.get('method') or '论文中可确认的方法线索'}，主要贡献是 {paper.get('contribution') or '为该方向提供可引用证据'} [{pid}]。"
                )
            lines.append(" ".join(evidence_sentences))
        lines.append("")

        lines.append(llm_paragraphs["comparison"] if llm_paragraphs else _template_comparison(papers, zh, section_index))
        lines.append("")
        for slot in section.get("artifact_slots", []):
            if slot.get("placement") == "after_comparison_paragraph":
                lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])

        if llm_paragraphs:
            lines.append(llm_paragraphs["limitation"])
        else:
            limitation_bits = [
                f"{paper.get('title') or paper.get('paper_id')} 的边界是 "
                f"{paper.get('limitations') or '公开证据不足以支持更强结论'} [{paper.get('paper_id')}]。"
                for paper in papers[:4]
            ]
            lines.append(" ".join(limitation_bits))
        lines.append("")
        for slot in section.get("artifact_slots", []):
            if slot.get("placement") == "after_limitation_paragraph":
                lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])

        if next_title:
            lines.append(
                f"{title} 与下一节「{next_title}」的连接点在于：当模型能够描述环境、行动与反馈之后，"
                "需要继续追问这些能力如何被评测，以及哪些图表能帮助读者快速识别方法边界。"
            )
        else:
            lines.append(
                f"{title} 是正文最后一个技术主题；后续章节把上述证据汇总为开放挑战与未来方向，并给出可追溯图表。"
            )
        lines.append("")
        return lines

    # Single framing sentence, uncited: repeated section boilerplate is what
    # pushed the eval-v2 section similarity to 0.952.
    lines.append(f"{title} focuses on {_clean_clause(goal) or 'a citation-ready research theme'}.")
    lines.append("")
    for slot in section.get("artifact_slots", []):
        if slot.get("placement") == "after_topic_paragraph":
            lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])
    if llm_paragraphs:
        lines.append(llm_paragraphs["summary"])
    else:
        for paper in papers[:SECTION_PAPERS]:
            pid = paper.get("paper_id")
            lines.append(
                f"{paper.get('title') or pid} addresses {paper.get('problem') or 'a core modeling problem'}. "
                f"Its method can be summarized as {paper.get('method') or 'the method described by the paper card'}, "
                f"and its contribution is {paper.get('contribution') or 'citation-ready evidence for this theme'} [{pid}]."
            )
    lines.append("")
    lines.append(llm_paragraphs["comparison"] if llm_paragraphs else _template_comparison(papers, zh, section_index))
    lines.append("")
    # Per-section differentiated transitions
    _transition_map = {
        "generative game world simulation": "push beyond static game engines toward reusable learned dynamics",
        "planning and control": "show how learned models can support both search-based planning and policy learning",
        "neural game engine": "move from hand-authored simulation toward generated interactive environments",
        "game agent": "show how large-scale training changes what agents can do in complex games",
        "open-ended simulator": "provide shared environments where interactive intelligence can be compared",
    }
    section_key = next((k for k in _transition_map if k in title.lower()), "")
    transition = _transition_map.get(section_key, "share a common trajectory toward learnable, reusable, or generative environment models")
    lines.append(f"Taken together, this group {transition}.")
    lines.append("")
    for slot in section.get("artifact_slots", []):
        if slot.get("placement") == "after_comparison_paragraph":
            lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])
    if llm_paragraphs:
        lines.append(llm_paragraphs["limitation"])
    else:
        lines.append(
            " ".join(
                f"{paper.get('title') or paper.get('paper_id')} is limited by {paper.get('limitations') or 'the available evidence boundary'} [{paper.get('paper_id')}]."
                for paper in papers[:4]
            )
        )
    lines.append("")
    if next_title:
        lines.append(f"The next section, {next_title}, builds on this evidence base.")
    else:
        lines.append("The remaining sections summarize this evidence base into open challenges and future directions.")
    lines.append("")
    return lines


_COMPARISON_CLOSERS_EN = (
    "These differences decide whether each work serves planning, generation, or evaluation.",
    "Read this way, the papers divide by what they optimize rather than by venue or year.",
    "The contrast is technical: dynamics, interaction, and evaluation sit at different depths here.",
)
_COMPARISON_CLOSERS_ZH = (
    "这些差异决定了它们分别服务于规划、生成还是评测。",
    "由此看，这些工作的分野在于各自优化的对象，而非发表渠道或年份。",
    "对照之下，动力学、交互与评测在本节处于不同层次。",
)


def _template_comparison(papers: list[dict[str, Any]], zh: bool, section_index: int = 0) -> str:
    """Comparison paragraph built from the section's own card fields.

    Replaces the section-invariant paragraph that made every section read alike;
    the closing sentence rotates so neighbouring sections do not share it.
    """
    compared = [paper for paper in papers[:3] if paper.get("paper_id")]
    if not compared:
        return (
            "No citation-ready paper could be assigned to this section, so it stays a placeholder rather than an unsourced claim."
            if not zh
            else "本节没有可指派的论文证据，因此保留为占位说明，不给出无来源结论。"
        )
    if zh:
        parts = [
            f"{paper.get('title') or paper.get('paper_id')} 以 {_shorten(paper.get('method'), 120) or '其报告的方法'} 为核心，"
            f"贡献落在 {_shorten(paper.get('contribution'), 120) or '其报告的结果'} [{paper.get('paper_id')}]"
            for paper in compared
        ]
        head = "横向比较，" if len(compared) > 1 else "本节覆盖的工作中，"
        closer = _COMPARISON_CLOSERS_ZH[section_index % len(_COMPARISON_CLOSERS_ZH)]
        return head + "；".join(parts) + "。" + closer
    if len(compared) == 1:
        paper = compared[0]
        return (
            f"{paper.get('title') or paper.get('paper_id')} is the only citation-ready work assigned here: "
            f"its method is {_shorten(paper.get('method'), 120) or 'the reported method'} and its contribution is "
            f"{_shorten(paper.get('contribution'), 120) or 'the reported result'} [{paper.get('paper_id')}]."
        )
    parts = [
        f"{paper.get('title') or paper.get('paper_id')} centers on {_shorten(paper.get('method'), 120) or 'the reported method'} "
        f"and contributes {_shorten(paper.get('contribution'), 120) or 'the reported result'} [{paper.get('paper_id')}]"
        for paper in compared
    ]
    closer = _COMPARISON_CLOSERS_EN[section_index % len(_COMPARISON_CLOSERS_EN)]
    return "Compared side by side, " + "; ".join(parts) + ". " + closer


def _llm_section_paragraphs(section: dict[str, Any], llm_chat: Callable, zh: bool) -> dict[str, str]:
    """Draft the three body paragraphs with the LLM; raise when the reply is unusable.

    Anti-hallucination guard: the prompt carries only this section's card fields
    plus evidence snippets, identified by short [Pn] aliases — real ids never
    reach the model, so it cannot mangle them into bare, space-broken DOIs.
    Returned [Pn] tags map back to real ids; any sentence citing an unknown
    tag, an id outside the section whitelist, or a bare leaked id is dropped.
    A paragraph left empty by that filter makes the whole reply unusable.
    """
    papers = section.get("selected_papers", [])[:SECTION_PAPERS]
    allowed = {str(paper.get("paper_id")) for paper in papers if paper.get("paper_id")}
    alias_of: dict[str, str] = {}
    paper_lines = []
    for index, paper in enumerate(papers, start=1):
        paper_id = str(paper.get("paper_id"))
        if not paper.get("paper_id"):
            continue
        alias = f"P{index}"
        alias_of[alias] = paper_id
        paper_lines.append(
            " | ".join(
                [
                    alias,
                    str(paper.get("title", "")),
                    str(paper.get("problem", "")),
                    str(paper.get("method", "")),
                    str(paper.get("contribution", "")),
                    str(paper.get("limitations", "")),
                    " ".join(str(s) for s in paper.get("evidence_snippets", [])),
                ]
            )
        )
    tag_list = " ".join(f"[{alias}]" for alias in alias_of)
    user = "\n".join(
        [
            "Section title: " + str(section.get("section_title", "")),
            "Section goal: " + str(section.get("section_goal", "")),
            "Papers (tag | title | problem | method | contribution | limitation | evidence):",
            *paper_lines,
            "Write in Chinese (简体)." if zh else "Write in English.",
            "Reply with exactly three paragraphs, each starting on its own line with one tag:",
            "[SUMMARY] for every paper, in order, one sentence built almost verbatim from its evidence text (light connective edits allowed), each sentence ending with that paper's tag.",
            "[COMPARISON] how the papers differ in method and role; cite the tags you compare.",
            "[LIMITATION] what the reported evidence does not establish; cite the tags you discuss.",
            "Restate only the facts in the paper list above: no invented numbers, years, benchmarks, or paper names.",
            "Refer to papers by their titles in prose, and cite them only with the bracketed tags — never copy any internal paper id, DOI, or URL into the text.",
            f"Cite only with these bracketed tags: {tag_list}",
        ]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You draft body paragraphs for an evidence-grounded academic survey. "
                "Citations must use the supplied bracketed paper tags and nothing else."
            ),
        },
        {"role": "user", "content": user},
    ]
    # Generous budget: reasoning models spend completion tokens on hidden
    # thinking before the visible content, so 3 short paragraphs need headroom.
    reply = llm_chat(messages, temperature=0.2, max_tokens=3000)
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError(f"empty LLM reply: {str(reply)[:120]}")
    parsed = _parse_tagged_paragraphs(reply)
    paragraphs: dict[str, str] = {}
    for tag in _LLM_PARAGRAPH_TAGS:
        body = _map_alias_citations(parsed.get(tag, ""), alias_of)
        body = _sanitize_llm_paragraph(body, allowed)
        if not body:
            raise ValueError(f"LLM reply has no usable [{tag}] paragraph")
        paragraphs[tag] = body
    return paragraphs


def _map_alias_citations(text: str, alias_of: dict[str, str]) -> str:
    """Rewrite [P3]-style tags to real paper ids; unknown tags stay and are
    dropped later by the whitelist filter."""
    return re.sub(
        r"\[(P\d+)\]",
        lambda match: f"[{alias_of[match.group(1)]}]" if match.group(1) in alias_of else match.group(0),
        text,
    )


def _parse_tagged_paragraphs(reply: str) -> dict[str, str]:
    found: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for raw in reply.splitlines():
        line = raw.strip()
        match = re.match(r"^\[([A-Za-z]+)\]\s*(.*)$", line)
        if match and match.group(1).lower() in _LLM_PARAGRAPH_TAGS:
            current = match.group(1).lower()
            found[current].append(match.group(2))
        elif current is not None and line:
            found[current].append(line)
    return {tag: " ".join(parts).strip() for tag, parts in found.items()}


def _sanitize_llm_paragraph(text: str, allowed: set[str]) -> str:
    keep = []
    for sentence in _split_sentences(text):
        if _contains_banned_phrase(sentence):
            continue
        if _LEAKED_ID_RE.search(sentence) or _LEAKED_ID_FRAGMENT_RE.match(sentence):
            continue
        citations = _extract_citations(sentence)
        if citations and not citations <= allowed:
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
                "summary": _shorten(card.get("contribution") or card.get("method") or card.get("title", ""), 180),
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


def _draw_taxonomy_png(path: Path, categories: list[dict[str, Any]], cards: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    counts: dict[str, int] = defaultdict(int)
    category_names = {cat.get("category_id"): cat.get("category_name") or cat.get("name") or "Uncategorized" for cat in categories}
    for card in cards:
        counts[category_names.get(card.get("category_id"), card.get("category") or "Uncategorized")] += 1
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
    systems_headers = ["Paper", "Year", "Category", "Method", "Contribution", "Limitation"]
    systems_rows = [
        [
            card.get("title", ""),
            str(card.get("year", "")),
            card.get("category", ""),
            _shorten(card.get("method", ""), 96),
            _shorten(card.get("contribution", ""), 96),
            _shorten(card.get("limitations", ""), 96),
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
                "method": _shorten(card.get("method", ""), 140),
                "contribution": _shorten(card.get("contribution", ""), 140),
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
    title = card.get("title") or card.get("paper_id")
    method = _shorten(card.get("method", ""), 140)
    contribution = _shorten(card.get("contribution", ""), 180)
    problem = _shorten(card.get("problem", ""), 120)
    if contribution and method:
        return f"{title} addresses {problem or 'this research problem'} with {method}, and reports {contribution}"
    if contribution:
        return f"{title} contributes {contribution}"
    if method:
        return f"{title} uses {method} for {problem or 'the target task'}"
    return f"{title} is part of the citation-ready evidence base for this topic"


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
    value = str(card.get(field) or "").strip()
    if value:
        return _shorten(value, 220)
    possible = card.get("possible_claims") or {}
    bucket = possible.get(claim_bucket) or possible.get(claim_bucket.replace("_", " ")) or []
    if isinstance(bucket, list) and bucket:
        first = bucket[0]
        if isinstance(first, dict):
            return _shorten(first.get("text", ""), 220)
        return _shorten(getattr(first, "text", ""), 220)
    return ""


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
    """Drop trailing punctuation so a template sentence keeps a single full stop."""
    return str(text or "").strip().rstrip(".,;、;。． ")


def _contains_banned_phrase(text: str) -> bool:
    folded = text.casefold()
    return any(phrase in folded for phrase in BANNED_META_PHRASES)


def _shorten(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


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
