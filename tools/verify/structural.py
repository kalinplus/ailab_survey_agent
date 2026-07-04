import re

from tools.models.bundle import CitationResult, CitationEntry

CITE_RE = re.compile(r"\[([^\]]+)\]")
FIG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def extract_citations(md: str) -> list[str]:
    # strip image tags so CITE_RE doesn't match alt-text inside ![...]
    text_only = FIG_RE.sub("", md)
    return [c.strip() for c in CITE_RE.findall(text_only) if not c.strip().startswith("http")]


def extract_figure_refs(md: str) -> list[str]:
    return [fid.strip() for fid in FIG_RE.findall(md)]


def run(task_id: str, survey_md: str, citation_index, figure_bank, table_bank):
    # table-ref validation deferred (no markdown table-ref convention defined)
    ready = {c["paper_id"] for c in citation_index.citations}
    fig_set = {f.figure_id for f in figure_bank.figures}

    entries: list[CitationEntry] = []
    for cid in extract_citations(survey_md):
        entries.append(CitationEntry(
            citation_id=cid, valid=cid in ready,
            figures_valid=[], tables_valid=[]))
    for fid in extract_figure_refs(survey_md):
        entries.append(CitationEntry(
            citation_id=fid, valid=fid in fig_set,
            figures_valid=[], tables_valid=[]))

    total = len(entries)
    valid = sum(1 for e in entries if e.valid)
    score = (valid / total) if total else 1.0
    return CitationResult(
        task_id=task_id, total_citations=total, valid_citations=valid,
        invalid_citations=total - valid, weak_claims=0,
        citation_validity_score=round(score, 3), entries=entries)
