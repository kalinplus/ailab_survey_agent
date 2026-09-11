import logging
import re

from tools.models.artifacts import Figure, FigureBank, Table, TableBank, Category, Taxonomy, CitationIndex
from tools.models.common import figure_id, table_id, category_id

logger = logging.getLogger(__name__)

# Category-name tokens that match nothing distinctive; before wave7 the
# matcher counted them, piling 48/67 cards into the first category and
# leaving every card field the writer consumed empty ("Uncategorized 13").
_TAXONOMY_STOPWORDS = {"and", "or", "for", "of", "the", "a", "an", "in", "on",
                       "with", "to", "via"}


def build_figure_bank(task_id, parsed_papers):
    figs = [Figure(figure_id=figure_id(p.paper_id, f["num"]), paper_id=p.paper_id,
                   caption=f.get("caption", ""), page=f.get("page", 0),
                   image_path=f.get("img_path"))
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
    # assign each card to the best-matching category by meaningful-token
    # coverage of its title AND claim texts (the legacy method/contribution
    # fields are empty under the wave6 claim contract)
    for card in paper_cards.paper_cards:
        claim_text = " ".join(cl.text for bucket in card.possible_claims.values() for cl in bucket)
        text_tokens = set(re.findall(r"[a-z0-9]+", f"{card.title} {claim_text}".lower()))
        best, best_score = None, 0.0
        for c in refined_taxonomy:
            tokens = [t for t in re.findall(r"[a-z0-9]+", c["name"].lower())
                      if t not in _TAXONOMY_STOPWORDS]
            if not tokens:
                continue
            hits = sum(1 for t in tokens if t in text_tokens)
            score = hits / len(tokens)
            if hits and score > best_score:
                best_score, best = score, c["name"]
        if best:
            for cat in cats:
                if cat.category_name == best:
                    cat.paper_ids.append(card.paper_id)
                    break
        else:
            logger.warning("[P5.3] taxonomy: no category matched %s (%s)",
                           card.paper_id, card.title[:60])
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
