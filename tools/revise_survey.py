"""C tool: survey revision after verification failure.

Two kernels behind the same request protocol:
- default (off switch): conservative deletion repair, unchanged behavior;
- EVISURVEY_REPAIR_AGENT=1: joint-2 repair agent (failure-typed actions,
  incremental NLI re-verification, repair_log) — spec
  docs/RealAgent/关节2-修复Agent-方案与测试.md.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from config import load_config
from harness.json_io import read_json, write_json
from harness.logger import now_iso

logger = logging.getLogger(__name__)


def run(request_path: str) -> dict[str, Any]:
    cfg = load_config()
    root = cfg.root_dir
    request_file = _resolve(root, request_path)
    req = read_json(request_file)
    task_id = req.get("task_id", "unknown_task")
    inputs = req.get("inputs", {})
    outputs = req.get("outputs", {})

    survey_path = _resolve(root, inputs.get("survey_markdown_path", "output/survey.md"))
    citation_result = _read_optional(root, inputs.get("citation_result_path"), {"entries": []})
    claim_map = _read_optional(root, inputs.get("claim_map_path"), {"entries": []})
    ready_set = _read_final_or_requested(root, "citation_ready_set", inputs.get("citation_ready_set_path"), {"allowed_paper_ids": [], "items": []})
    gen_bank = _read_optional(root, inputs.get("generated_artifact_bank_path"), {"artifacts": []})

    allowed_ids = set(ready_set.get("allowed_paper_ids", []))
    allowed_artifacts = {item.get("artifact_id") for item in _as_list(gen_bank, "artifacts") if item.get("artifact_id")}
    card_by_id = {item.get("paper_id"): item for item in ready_set.get("items", []) if item.get("paper_id")}

    markdown = survey_path.read_text(encoding="utf-8")

    if _repair_agent_enabled(cfg):
        result = _run_repair_agent(cfg, root, task_id, inputs, markdown, claim_map,
                                   ready_set, allowed_artifacts)
        if result is not None:
            revised, notes, metrics = result
            revised = _replace_references(revised, card_by_id)
            revised = _append_revision_notes(revised, notes)
            return _write_revised(root, task_id, request_path, outputs, revised, metrics,
                                  "Survey repaired by failure-typed actions.")

    notes: list[str] = []
    revised = _remove_invalid_entries(markdown, citation_result, allowed_ids, allowed_artifacts, notes)
    revised = _revise_unsupported_claims(revised, claim_map, allowed_ids, notes)
    revised = _replace_references(revised, card_by_id)
    revised = _append_revision_notes(revised, notes)
    return _write_revised(root, task_id, request_path, outputs, revised,
                          {"revision_notes": len(notes),
                           "remaining_citations": len(_extract_citations(revised)),
                           "remaining_artifact_refs": len(_extract_figure_refs(revised))},
                          "Survey revised conservatively without adding references.")


def _repair_agent_enabled(cfg) -> bool:
    if os.getenv("EVISURVEY_REPAIR_AGENT", "0").lower() not in {"1", "true", "yes"}:
        return False
    if not cfg.intern_api_key:
        logger.warning("[revise] EVISURVEY_REPAIR_AGENT on but INTERN_API_KEY missing -> deletion path")
        return False
    return True


def _run_repair_agent(cfg, root: Path, task_id: str, inputs: dict, markdown: str,
                      claim_map: dict, ready_set: dict, allowed_artifacts: set):
    """Delegate to the joint-2 repair agent; None means 'fall back to deletion'."""
    from harness.agents.repair_agent import run_repair
    from tools.verify_citations import _nli_model

    evidence_path = _resolve(root, inputs.get("evidence_store_path", "cache/evidence_store.json"))
    figure_bank = _read_optional(root, inputs.get("figure_bank_path", "cache/figure_bank.json"),
                                 {"figures": []})
    figure_items = [f for f in figure_bank.get("figures", []) if isinstance(f, dict)]
    evidence_store = _read_optional(root, inputs.get("evidence_store_path"),
                                    {"evidence": []})
    try:
        result = run_repair(
            task_id=task_id, round_no=_next_repair_round(root),
            survey_md=markdown, claim_map=claim_map, evidence_store=evidence_store,
            ready_set=ready_set, figure_items=figure_items,
            allowed_artifacts=allowed_artifacts,
            llm_json_chat=_make_llm_json_chat(cfg), nli=_nli_model(),
            sciverse=_make_sciverse(cfg),
            trajectory_dir=root / "logs" / "trajectory")
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        # joint boundary: a crashed repair must not lose the run — fall back to deletion
        logger.warning(f"[revise] repair agent failed -> deletion path: {exc}")
        return None

    if result["evidence_store"] != evidence_store:
        write_json(evidence_path, result["evidence_store"])
    repair_log_path = root / "output" / "repair_log.json"
    existing = read_json(repair_log_path) if repair_log_path.exists() else []
    write_json(repair_log_path, existing + result["repair_log"])
    # NOTE: no square brackets anywhere in these lines — the verifier's citation
    # extractor treats [x] as a reference id, so bracketed notes would pollute
    # the next verify round with phantom claims
    notes = [f"type={entry['failure_type']} {entry['action']} -> {entry['outcome']}: "
             f"{entry['reason']}" for entry in result["repair_log"]]
    return result["revised_md"], notes, result["metrics"]


def _next_repair_round(root: Path) -> int:
    """Outer Goal-Gate round number = how many repairs already happened."""
    repair_log_path = root / "output" / "repair_log.json"
    if not repair_log_path.exists():
        return 1
    entries = read_json(repair_log_path)
    return max((int(e.get("round", 0)) for e in entries if isinstance(e, dict)), default=0) + 1


def _make_llm_json_chat(cfg):
    from llm_client import heavy_llm_client, _extract_json_object

    client = heavy_llm_client(cfg)

    def chat(messages):
        try:
            return client.json_chat(messages, temperature=0.1, max_tokens=6000)
        except Exception:
            content = client.chat(messages, temperature=0.1, max_tokens=6000)
            extracted = _extract_json_object(content)
            if extracted is None:
                raise ValueError("model reply contained no JSON object")
            return extracted

    return chat


def _make_sciverse(cfg):
    from tools.clients.sciverse_client import SciVerseClient

    return SciVerseClient(base_url=cfg.sciverse_api_base_url, api_key=cfg.sciverse_api_token)


def _write_revised(root: Path, task_id: str, request_path: str, outputs: dict,
                   revised: str, metrics: dict, message: str) -> dict[str, Any]:
    revised_path = _resolve(root, outputs.get("revised_survey_markdown_path", "output/survey_revised.md"))
    revised_path.parent.mkdir(parents=True, exist_ok=True)
    revised_path.write_text(revised, encoding="utf-8")
    return {
        "task_id": task_id,
        "tool": "revise_survey",
        "owner": "C",
        "status": "success",
        "input_request": request_path,
        "outputs": [_rel(root, revised_path)],
        "metrics": metrics,
        "message": message,
        "timestamp": now_iso(),
    }


def _remove_invalid_entries(
    markdown: str,
    citation_result: dict[str, Any],
    allowed_ids: set[str],
    allowed_artifacts: set[str],
    notes: list[str],
) -> str:
    invalid_entries = [
        entry.get("citation_id")
        for entry in citation_result.get("entries", [])
        if isinstance(entry, dict) and entry.get("valid") is False and entry.get("citation_id")
    ]
    for item_id in invalid_entries:
        if item_id in allowed_ids or item_id in allowed_artifacts:
            continue
        before = markdown
        markdown = re.sub(rf"^.*!\[[^\]]*\]\({re.escape(item_id)}\).*$\n?", "", markdown, flags=re.MULTILINE)
        markdown = markdown.replace(f"[{item_id}]", "")
        if markdown != before:
            notes.append(f"Removed invalid citation or artifact reference: {item_id}.")

    for cite in sorted(_extract_citations(markdown) - allowed_ids):
        markdown = markdown.replace(f"[{cite}]", "")
        notes.append(f"Removed non-whitelisted citation: {cite}.")

    for artifact_id in sorted(_extract_figure_refs(markdown) - allowed_artifacts):
        markdown = re.sub(rf"^.*!\[[^\]]*\]\({re.escape(artifact_id)}\).*$\n?", "", markdown, flags=re.MULTILINE)
        notes.append(f"Removed undeclared generated artifact reference: {artifact_id}.")
    return markdown


def _revise_unsupported_claims(
    markdown: str,
    claim_map: dict[str, Any],
    allowed_ids: set[str],
    notes: list[str],
) -> str:
    entries = claim_map.get("entries")
    if entries is None:
        entries = claim_map.get("claims", [])
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status") or entry.get("evidence_status")
        cited_id = entry.get("cited_paper_id")
        if status != "unsupported" or cited_id not in allowed_ids:
            continue
        claim_text = str(entry.get("claim_text") or entry.get("claim") or "").strip()
        before = markdown
        markdown = _remove_sentence_containing(markdown, claim_text, cited_id)
        if markdown != before:
            notes.append(f"Removed unsupported claim citing {cited_id}.")
    return markdown


def _replace_references(markdown: str, card_by_id: dict[str, dict[str, Any]]) -> str:
    body = re.split(r"\n## References\b", markdown, maxsplit=1)[0].rstrip()
    used = sorted(_extract_citations(body))
    refs = ["", "## References", ""]
    for paper_id in used:
        card = card_by_id.get(paper_id, {})
        title = card.get("title") or paper_id
        year = card.get("year") or "n.d."
        refs.append(f"- {paper_id}: {title} ({year}).")
    return body + "\n" + "\n".join(refs).rstrip() + "\n"


def _append_revision_notes(markdown: str, notes: list[str]) -> str:
    if not notes:
        notes = ["No illegal citations or artifacts were found; references were normalized to actually cited allowed papers."]
    lines = [markdown.rstrip(), "", "## Revision Notes", ""]
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def _remove_sentence_containing(markdown: str, claim_text: str, cited_id: str) -> str:
    if not claim_text:
        return markdown
    fragments = re.split(r"(?<=[.。])\s+", markdown)
    kept = []
    normalized = _compact(claim_text)
    for fragment in fragments:
        if f"[{cited_id}]" in fragment and normalized and normalized[:60] in _compact(fragment):
            continue
        kept.append(fragment)
    return " ".join(kept)


def _read_optional(root: Path, raw_path: str | None, default: Any) -> Any:
    if not raw_path:
        return default
    path = _resolve(root, raw_path)
    if not path.exists():
        return default
    return read_json(path)


def _read_final_or_requested(root: Path, name: str, raw_path: str | None, default: Any) -> Any:
    if os.getenv("FINAL_SEED_PAPERS") == "1":
        final_path = root / "cache" / f"final_{name}.json"
        if final_path.exists():
            return read_json(final_path)
    return _read_optional(root, raw_path, default)


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
        data = data.get(key, [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _extract_citations(markdown: str) -> set[str]:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    return {match.strip() for match in re.findall(r"\[([^\]]+)\]", text_only) if not match.startswith("http")}


def _extract_figure_refs(markdown: str) -> set[str]:
    return {match.strip() for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)}


def _compact(text: str) -> str:
    return " ".join(text.split())

