"""Claim-level attribution and copy diagnostics, independent of NLI verdicts."""

import re
from difflib import SequenceMatcher

from tools.verify.text_units import normalize_text

SOURCE_ROLES = {"own_method", "own_contribution", "own_result", "own_setup", "own_limitation",
                "background", "related_work", "unknown"}
SCOPE_ROLES = {
    "method": {"own_method", "own_setup"},
    "contribution": {"own_contribution", "own_result"},
    "comparison": {"own_method", "own_setup", "own_contribution", "own_result"},
    "limitation": {"own_limitation"},
    "background": {"background", "related_work"},
}
SUPPORTED_TYPES = {"direct", "entailment"}
_SELF = re.compile(r"\b(?:we|our|this (?:paper|work|study)|here we)\b|本文|我们|本研究", re.I)
_OTHER = re.compile(
    r"^(?:previous|prior|earlier|existing|traditional) (?:work|methods?|studies|approaches?)\b|"
    r"^.{0,60}\bet al\.? (?:show|report|propose)|"
    r"^Tree-based planning methods have enjoyed|^The strongest chess programs are based|"
    r"^(?:已有研究|前人工作|相关工作)(?:表明|提出)", re.I)


def role_violation(role, source_quote):
    if role not in SOURCE_ROLES or role == "unknown":
        return "unknown_source_role"
    # A physical Method section may discuss prior work. Judge the selected
    # assertion's attribution, never the containing paragraph's heading.
    if role.startswith("own_") and _OTHER.search(source_quote) and not _SELF.search(source_quote):
        return "background_as_own_work"
    return ""


def valid_support(row, evidence):
    """Only a role-labelled, real source binding may feed the writer."""
    quote = row.get("source_quote", "")
    return bool(
        row.get("support_type") in SUPPORTED_TYPES
        and row.get("paper_id") == evidence.get("paper_id")
        and evidence.get("evidence_id") in (row.get("evidence_ids") or [])
        and quote and normalize_text(quote) in normalize_text(evidence.get("text", ""))
        and not role_violation(row.get("source_role", "unknown"), quote)
    )


def quote_diagnostic(text, sources):
    """Flag long copied clauses, not shared technical noun phrases.

    Length AND sentence coverage matter. Eight shared tokens alone never
    trigger a flag; this is an advisory signal, never semantic support.
    """
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold())
    cjk = sum(bool(re.fullmatch(r"[\u4e00-\u9fff]", w)) for w in words) > len(words) / 2
    best = {"quote_like": False, "longest_copy_tokens": 0, "copy_ratio": 0.0}
    for source in sources:
        other = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", source.casefold())
        size = SequenceMatcher(None, words, other, autojunk=False).find_longest_match().size
        ratio = size / max(1, len(words))
        # Technical names commonly span 8-12 tokens. A long clause covering
        # most of the sentence, or 24 consecutive words, merits review.
        clause_min, long_min = (28, 48) if cjk else (14, 24)
        flagged = (size >= clause_min and ratio >= 0.65) or size >= long_min
        if size > best["longest_copy_tokens"]:
            best = {"quote_like": flagged, "longest_copy_tokens": size, "copy_ratio": round(ratio, 3)}
        elif flagged:
            best["quote_like"] = True
    return best
