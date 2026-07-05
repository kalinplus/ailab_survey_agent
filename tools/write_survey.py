"""C tool: grounded survey writer and visual artifact builder."""

from __future__ import annotations

import base64
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import load_config
from harness.json_io import read_json, write_json
from harness.logger import now_iso

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
    survey = _render_survey(topic, language, section_plan, artifacts, cards)

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


def _build_section_claim_plan(
    task_id: str,
    categories: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    evidence_by_paper: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    cards_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_ranked = _rank_cards(cards, evidence_by_paper)
    for card in cards:
        key = card.get("category_id") or card.get("category") or "uncategorized"
        cards_by_category[str(key)].append(card)

    sections: list[dict[str, Any]] = []
    for index, category in enumerate(categories, start=1):
        category_id = str(category.get("category_id") or category.get("category_name") or f"cat_{index:03d}")
        selected = cards_by_category.get(category_id)
        if not selected:
            category_name = str(category.get("category_name") or category.get("name") or "")
            selected = [card for card in cards if card.get("category") == category_name]
        rotated = global_ranked[(index - 1) * 2 :] + global_ranked[: (index - 1) * 2]
        selected = _dedupe_cards(_rank_cards(selected or [], evidence_by_paper) + rotated)[:5]
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
        selected_papers = [_selected_paper_entry(card) for card in selected]
        sections.append(
            {
                "section_id": f"sec_{index:02d}",
                "section_title": str(category.get("category_name") or category.get("name") or f"Theme {index}"),
                "section_goal": str(category.get("description") or "Summarize citation-ready papers in this theme."),
                "target_length_words": 450,
                "allowed_paper_ids": [card["paper_id"] for card in selected],
                "selected_papers": selected_papers,
                "artifact_slots": _artifact_slots_for_section(index, str(category.get("category_name") or category.get("name") or "")),
                "claims": claims,
            }
        )
    cited = {pid for section in sections for pid in section.get("allowed_paper_ids", [])}
    remaining = [card for card in global_ranked if card.get("paper_id") not in cited]
    if remaining and len(cited) < min(8, len(cards)):
        selected = remaining[:5]
        sections.append(
            {
                "section_id": f"sec_{len(sections) + 1:02d}",
                "section_title": "Cross-Paper Evidence Synthesis",
                "section_goal": "Use additional citation-ready papers to broaden coverage while staying within the whitelist.",
                "target_length_words": 450,
                "allowed_paper_ids": [card["paper_id"] for card in selected],
                "selected_papers": [_selected_paper_entry(card) for card in selected],
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
        selected = global_ranked[:5]
        sections.append(
            {
                "section_id": "sec_01",
                "section_title": "Citation-Ready Evidence",
                "section_goal": "Summarize the available citation-ready papers.",
                "target_length_words": 450,
                "allowed_paper_ids": [card["paper_id"] for card in selected],
                "selected_papers": [_selected_paper_entry(card) for card in selected],
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


def _selected_paper_entry(card: dict[str, Any]) -> dict[str, Any]:
    score = float(card.get("topic_relevance_score", 0.0))
    return {
        "paper_id": card.get("paper_id"),
        "title": card.get("title", ""),
        "year": card.get("year"),
        "problem": _field_or_claim(card, "problem", "key_results"),
        "method": _field_or_claim(card, "method", "method"),
        "contribution": _field_or_claim(card, "contribution", "key_results"),
        "limitations": _field_or_claim(card, "limitations", "limitations"),
        "topic_relevance_score": round(score, 3),
        "selection_reason": _selection_reason(card),
    }


def _artifact_slots_for_section(index: int, title: str) -> list[dict[str, str]]:
    text = title.lower()
    slots = []
    if index == 1 or "intro" in text or "scope" in text or "foundational" in text:
        slots.append(
            {
                "artifact_id": "generated_timeline_001",
                "placement": "after_topic_paragraph",
                "caption": "Generated timeline based on citation-ready paper cards.",
            }
        )
    if "taxonomy" in text or "category" in text or index == 2:
        slots.append(
            {
                "artifact_id": "generated_taxonomy_graph_001",
                "placement": "after_topic_paragraph",
                "caption": "Generated taxonomy graph based on verified categories.",
            }
        )
    if "method" in text or "system" in text or "model" in text or index == 3:
        slots.append(
            {
                "artifact_id": "generated_comparison_table_001",
                "placement": "after_comparison_paragraph",
                "caption": "Generated comparison based on citation-ready paper cards.",
            }
        )
    if "future" in text or "challenge" in text:
        slots.append(
            {
                "artifact_id": "generated_future_matrix_001",
                "placement": "after_limitation_paragraph",
                "caption": "Generated future matrix from verified limitations and evidence.",
            }
        )
    return slots


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for card in cards:
        paper_id = card.get("paper_id")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        out.append(card)
    return out


def _render_survey(
    topic: str,
    language: str,
    sections: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
) -> str:
    zh = language == "zh"
    title = f"# {topic or 'Grounded Survey'}"
    lines = [
        title,
        "",
        "## Abstract" if not zh else "## 摘要",
        (
            "This survey is generated from citation-ready papers and verified intermediate artifacts. It emphasizes topic relevance, citation diversity, section depth, and traceable visual artifacts."
            if not zh
            else "本综述基于 CitationReadySet 中的论文、PaperCards、EvidenceStore 与可追溯派生图表生成，并在二阶段增强中加入 topic relevance 过滤、长段落模板写作、引用多样性和按章节嵌入的图表。"
        ),
        "",
        "## Introduction" if not zh else "## 引言",
        (
            "The report follows the available taxonomy instead of citation-count or impact-factor ranking. Generated artifacts are placed near the sections they support."
            if not zh
            else "本文按 taxonomy、topic relevance 与 evidence richness 组织内容，不声称已经实现引用量、影响因子或最新论文排序；派生图表会尽量放在支撑其论证的章节附近。"
        ),
        "",
    ]

    for section in sections:
        lines.extend([f"## {section['section_title']}", ""])
        lines.extend(_build_section_text(section, zh))

    lines.extend(
        [
            "## Open Challenges" if not zh else "## 开放挑战",
            "",
            (
                "Current evidence supports a conservative synthesis; stronger temporal and impact-based ranking remains future work."
                if not zh
                else "当前证据支持保守综合；更完善的时间序列排序和高影响论文排序仍属于后续工作。"
            ),
            "",
            "## Future Directions" if not zh else "## 未来方向",
            "",
            (
                "Future iterations should enrich evidence extraction, add stronger ranking signals, and improve visual summaries."
                if not zh
                else "后续应增强证据抽取、补充更可靠的排序信号，并提升派生图表的表达质量。"
            ),
            "",
            "![Future Matrix](generated_future_matrix_001)",
            "",
            "## Conclusion" if not zh else "## 结论",
            "",
            (
                "The harness-first workflow trades broad manual ranking for traceability, verification, and reproducible reporting."
                if not zh
                else "Harness 化流程以可追溯、可验证和可复盘报告为核心优势，同时保留对人工调研深度的清醒边界。"
            ),
            "",
            "![Summary Table](generated_summary_table_001)",
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


def _build_section_text(section: dict[str, Any], zh: bool) -> list[str]:
    title = str(section.get("section_title", "This section"))
    goal = str(section.get("section_goal", ""))
    papers = section.get("selected_papers", [])
    paper_ids = [paper["paper_id"] for paper in papers if paper.get("paper_id")]
    first_cite = paper_ids[0] if paper_ids else ""
    second_cite = paper_ids[1] if len(paper_ids) > 1 else first_cite
    cite_tail = f" [{first_cite}]" if first_cite else ""
    second_tail = f" [{second_cite}]" if second_cite else ""
    lines: list[str] = []

    if zh:
        lines.append(
            f"{title} 这一部分关注的问题是：{goal or '如何把可引用论文中的方法、证据和局限组织成可验证的综述段落'}。"
            f"在 World Models / GameCraft 语境下，它不是单独罗列论文，而是说明这些工作如何服务于交互环境建模、agent 学习、模拟器构建或游戏智能评测{cite_tail}。"
            "本节只使用 CitationReadySet 中的论文，并优先选择 topic relevance 较高、且能由 PaperCards 或 EvidenceStore 支撑的工作。"
        )
        lines.append("")
        for slot in section.get("artifact_slots", []):
            if slot.get("placement") == "after_topic_paragraph":
                lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])

        evidence_sentences = []
        for paper in papers[:5]:
            pid = paper.get("paper_id")
            if not pid:
                continue
            evidence_sentences.append(
                f"《{paper.get('title') or pid}》处理的问题可概括为 {paper.get('problem') or '该主题下的核心建模问题'}；"
                f"其方法侧重 {paper.get('method') or '论文中可确认的方法线索'}，主要贡献是 {paper.get('contribution') or '为该方向提供可引用证据'} [{pid}]。"
            )
        lines.append(" ".join(evidence_sentences))
        lines.append("")

        lines.append(
            f"横向比较这些工作，可以看到它们共享一个共同目标：把游戏或交互环境从固定 benchmark 推向可学习、可模拟、可复用的模型化对象。"
            f"差异在于，有的工作更强调 agent 训练和规划，有的工作更强调生成式环境表示，还有的工作更接近 benchmark 或系统工程。"
            f"因此，本节不会把它们按引用量或影响因子排序，而是按 problem-method-contribution 的证据链比较其角色{second_tail}。"
        )
        lines.append("")
        for slot in section.get("artifact_slots", []):
            if slot.get("placement") == "after_comparison_paragraph":
                lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])

        limitation_bits = []
        for paper in papers[:4]:
            limitation = paper.get("limitations") or "公开证据不足以支持更强结论"
            pid = paper.get("paper_id")
            limitation_bits.append(f"{paper.get('title') or pid} 的边界是 {limitation} [{pid}]。")
        lines.append(
            "这些证据也提示需要保守表述。"
            + " ".join(limitation_bits)
            + " 因此，C 模块在这里采用降调写法：只说明论文卡片和证据片段支持的事实，不扩展到未验证的性能、影响力或最新性判断。"
        )
        lines.append("")
        for slot in section.get("artifact_slots", []):
            if slot.get("placement") == "after_limitation_paragraph":
                lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])

        lines.append(
            f"从结构上看，{title} 与下一类问题的连接点在于：当模型能够描述环境、行动和反馈之后，综述需要继续追问这些模型如何被评测、如何与 agent 训练闭环结合，以及哪些图表能够帮助读者快速识别方法边界。"
            "这也是本报告把正文论证、CitationReadySet、GeneratedArtifactBank 和 ReviewBoard 放在同一条 harness 链路中的原因。"
        )
        lines.append("")
        return lines

    lines.append(
        f"{title} focuses on {goal or 'a citation-ready research theme'}. "
        f"In this survey, the section explains how the selected works contribute to interactive environment modeling, agent learning, simulation, or game intelligence evaluation{cite_tail}. "
        "The section uses only CitationReadySet papers and favors high topic-relevance evidence."
    )
    lines.append("")
    for slot in section.get("artifact_slots", []):
        if slot.get("placement") == "after_topic_paragraph":
            lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])
    for paper in papers[:5]:
        pid = paper.get("paper_id")
        lines.append(
            f"{paper.get('title') or pid} addresses {paper.get('problem') or 'a core modeling problem'}. "
            f"Its method can be summarized as {paper.get('method') or 'the method described by the paper card'}, "
            f"and its contribution is {paper.get('contribution') or 'citation-ready evidence for this theme'} [{pid}]."
        )
    lines.append("")
    lines.append(
        f"Taken together, these works share a move from fixed game environments toward learnable, reusable, or generative environment models. "
        f"They differ in whether they emphasize planning, simulation, benchmark construction, or environment generation{second_tail}."
    )
    lines.append("")
    for slot in section.get("artifact_slots", []):
        if slot.get("placement") == "after_comparison_paragraph":
            lines.extend([f"![{slot.get('caption', slot['artifact_id'])}]({slot['artifact_id']})", ""])
    lines.append(
        "The conservative reading is important: missing evidence, abstract-only parsing, or sparse limitations should weaken rather than strengthen the claim. "
        + " ".join(
            f"{paper.get('title') or paper.get('paper_id')} is limited by {paper.get('limitations') or 'the available evidence boundary'} [{paper.get('paper_id')}]."
            for paper in papers[:4]
        )
    )
    lines.append("")
    lines.append(
        f"This section bridges to the next theme by asking how the same evidence can support evaluation, visual comparison, and reproducible reporting inside the harness."
    )
    lines.append("")
    return lines


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
    support = [card["paper_id"] for card in cards[:5]]
    image_notes: list[dict[str, Any]] = []

    files: list[tuple[str, str, str, str, str, str, int]] = []
    timeline_md = ["# Timeline", "", "| Year | Category | Paper | Summary |", "|---:|---|---|---|"]
    for event in timeline.get("events", [])[:12]:
        timeline_md.append(
            f"| {event.get('year') or ''} | {event.get('category_name', '')} | {', '.join(event.get('paper_ids', []))} | {_escape_pipe(event.get('summary', ''))} |"
        )
    files.append(("generated_timeline_001", "timeline", "Timeline", "\n".join(timeline_md) + "\n", "Introduction / Scope", "Generated timeline based on citation-ready paper cards.", 90))

    graph_rows = ["# Taxonomy Graph", ""]
    for cat in categories:
        paper_ids = cat.get("paper_ids") or [card["paper_id"] for card in cards if card.get("category_id") == cat.get("category_id")]
        graph_rows.append(f"- {cat.get('category_name') or cat.get('name')}: {', '.join(paper_ids[:5])}")
    files.append(("generated_taxonomy_graph_001", "taxonomy_graph", "Taxonomy Graph", "\n".join(graph_rows) + "\n", "Taxonomy", "Generated taxonomy graph based on verified categories.", 80))

    comparison = ["# Comparison Table", "", "| Paper | Method | Contribution | Limitation |", "|---|---|---|---|"]
    for card in cards[:8]:
        comparison.append(
            f"| {card['paper_id']} | {_escape_pipe(_shorten(card.get('method', ''), 120))} | {_escape_pipe(_shorten(card.get('contribution', ''), 120))} | {_escape_pipe(_shorten(card.get('limitations', ''), 120))} |"
        )
    files.append(("generated_comparison_table_001", "comparison_table", "Comparison Table", "\n".join(comparison) + "\n", "Methods and Systems", "Generated comparison based on citation-ready paper cards.", 95))

    future = ["# Future Matrix", "", "| Direction | Evidence Basis | Risk |", "|---|---|---|"]
    for label in ["Long-horizon consistency", "Interactive control", "Evaluation protocols", "Agent training simulators"]:
        future.append(f"| {label} | {', '.join(support[:3])} | Requires stronger evidence and benchmarks |")
    files.append(("generated_future_matrix_001", "future_matrix", "Future Matrix", "\n".join(future) + "\n", "Future Directions", "Generated future matrix from verified limitations and evidence.", 75))

    summary = ["# Summary Table", "", "| Metric | Value |", "|---|---:|"]
    summary.append(f"| Citation-ready papers | {len(cards)} |")
    summary.append(f"| Taxonomy categories | {len(categories)} |")
    summary.append(f"| Evidence snippets | {sum(len(v) for v in evidence_by_paper.values())} |")
    files.append(("generated_summary_table_001", "summary_table", "Summary Table", "\n".join(summary) + "\n", "Conclusion / Evaluation", "Generated summary table for final report inspection.", 70))

    artifacts = []
    for artifact_id, artifact_type, title, content, placement_section, caption, display_priority in files:
        path = asset_dir / f"{artifact_id}.md"
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "title": title,
                "artifact_path": _rel(root, path),
                "artifact_format": "markdown",
                "source_artifacts": [
                    "cache/paper_cards.json",
                    "cache/evidence_store.json",
                    "cache/taxonomy.json",
                ],
                "supporting_papers": support,
                "provenance": "generated_by_c_from_verified_artifacts",
                "placement_section": placement_section,
                "caption": caption,
                "display_priority": display_priority,
                "usable_in_report": True,
            }
        )

    image_prompt = _plan_image_prompt(cfg, categories, cards, evidence_by_paper)
    image_result = _try_generate_image(cfg, asset_dir / "generated_visual_overview_001.png", image_prompt)
    image_notes.append(image_result)
    if image_result["status"] == "success":
        artifacts.append(
            {
                "artifact_id": "generated_visual_overview_001",
                "artifact_type": "taxonomy_graph",
                "title": "Visual Overview",
                "artifact_path": _rel(root, asset_dir / "generated_visual_overview_001.png"),
                "artifact_format": "png",
                "source_artifacts": [
                    "cache/paper_cards.json",
                    "cache/evidence_store.json",
                    "cache/taxonomy.json",
                ],
                "supporting_papers": support,
                "provenance": "generated_by_c_from_verified_artifacts",
                "generation_method": "intern_prompt_to_gpt_image_2_gateway",
                "prompt": image_prompt,
                "placement_section": "Taxonomy",
                "caption": "Generated visual overview from Intern-planned prompt and verified artifacts.",
                "display_priority": 60,
                "usable_in_report": True,
            }
        )
    return artifacts, image_notes


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
