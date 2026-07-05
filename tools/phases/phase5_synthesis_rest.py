import logging
from tools.models.artifacts import Figure, FigureBank, Table, TableBank, Category, Taxonomy, CitationIndex
from tools.models.common import figure_id, table_id, category_id

logger = logging.getLogger(__name__)


def build_figure_bank(task_id, parsed_papers):
    figs = [Figure(figure_id=figure_id(p.paper_id, f["num"]), paper_id=p.paper_id,
                   caption=f.get("caption", ""), page=f.get("page", 0))
            for p in parsed_papers.papers for f in p.figures]
    return FigureBank(task_id=task_id, figures=figs)


def build_table_bank(task_id, parsed_papers):
    tbls = [Table(table_id=table_id(p.paper_id, t["num"]), paper_id=p.paper_id,
                  caption=t.get("caption", ""), page=t.get("page", 0))
            for p in parsed_papers.papers for t in p.tables]
    return TableBank(task_id=task_id, tables=tbls)


def build_taxonomy(task_id, topic, refined_taxonomy, paper_cards, llm):
    cats = []
    name_to_id = {}
    for c in refined_taxonomy:
        if c["name"] not in name_to_id:
            name_to_id[c["name"]] = category_id(len(name_to_id) + 1)
        cats.append(Category(category_id=name_to_id[c["name"]], category_name=c["name"],
                            description=c.get("description", "")))
    # assign each card to best-matching category by keyword overlap
    for card in paper_cards.paper_cards:
        best = None
        best_score = -1
        text = f"{card.title} {card.method}".lower()
        for c in refined_taxonomy:
            score = sum(1 for kw in c["name"].lower().split() if kw in text)
            if score > best_score:
                best_score = score
                best = c["name"]
        if best:
            for cat in cats:
                if cat.category_name == best:
                    cat.paper_ids.append(card.paper_id)
                    break
    for cat in cats:
        cat.paper_count = len(cat.paper_ids)
    return Taxonomy(task_id=task_id, topic=topic, taxonomy_version="v1",
                   refine_history=["prelimiminary", "refined", "final"], categories=cats)


def build_citation_index(task_id, paper_cards):
    citations = [{"paper_id": c.paper_id, "bibtex_key": c.bibtex_key,
                  "title": c.title, "year": c.year} for c in paper_cards.paper_cards]
    return CitationIndex(task_id=task_id, citations=citations)


def run(task_id, topic, parsed_papers, refined_taxonomy, paper_cards, llm):
    fb = build_figure_bank(task_id, parsed_papers)
    tb = build_table_bank(task_id, parsed_papers)
    tax = build_taxonomy(task_id, topic, refined_taxonomy, paper_cards, llm)
    ci = build_citation_index(task_id, paper_cards)
    logger.info(f"[P5.3-5] synthesis: figures={len(fb.figures)} tables={len(tb.tables)} "
                f"taxonomy={len(tax.categories)} citations={len(ci.citations)}")
    return (fb, tb, tax, ci)
