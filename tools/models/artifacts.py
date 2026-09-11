from pydantic import BaseModel


class RetrievedPaper(BaseModel):
    paper_id: str
    title: str
    # SciVerse opaque document id — the /content key. Persisted at retrieval so
    # evidence grounding can fetch the paper's OWN fulltext (unique_id alone
    # cannot: doc_id is an internal hash, not derivable from the DOI).
    doc_id: str = ""
    # OpenAlex-shaped structured topic labels (primary_topic.domain/field) —
    # the wave-5 topic gate judges relevance by them, not by title wording
    topic_domain: str = ""
    topic_field: str = ""
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    abstract: str = ""
    keywords: list[str] = []
    citation_count: int = 0
    source: str = "sciverse"
    parse_status: str = "pending"
    survey_ref_count: int = 0
    survey_ref_hints: list[str] = []


class RetrievedPapers(BaseModel):
    task_id: str
    papers: list[RetrievedPaper]


class Paragraph(BaseModel):
    # page=None means "no page information" (markdown-only parse channel);
    # index is a per-paper globally unique block number, never per-section.
    page: int | None
    index: int
    text: str
    role: str = ""  # e.g. "reference" for bibliography-section blocks


class ParsedPaper(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    sections: list[dict] = []
    paragraphs: list[Paragraph] = []
    figures: list[dict] = []
    tables: list[dict] = []
    parse_status: str = "deep"


class ParsedPapers(BaseModel):
    task_id: str
    papers: list[ParsedPaper]


class Claim(BaseModel):
    text: str
    dimension: str
    evidence_ids: list[str] = []
    source_role: str = "unknown"
    source_quote: str = ""


class PaperCard(BaseModel):
    paper_id: str
    title: str
    doc_id: str = ""  # provenance chain from RetrievedPaper (/content key)
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int = 0
    survey_ref_count: int = 0
    matched_aspects: list[dict] = []
    category_id: str | None = None
    category: str = ""
    card_type: str = "deep"
    extraction_status: str = "legacy_unverified"
    source_blocks: list[dict] = []
    problem: str = ""
    method: str = ""
    contribution: str = ""
    limitations: str = ""
    evidence_ids: list[str] = []
    figure_ids: list[str] = []
    table_ids: list[str] = []
    possible_claims: dict[str, list[Claim]] = {}
    bibtex_key: str | None = None


class PaperCards(BaseModel):
    task_id: str
    paper_cards: list[PaperCard]


class Evidence(BaseModel):
    evidence_id: str
    paper_id: str
    source_type: str = "paragraph"
    source_page: int | None = 0  # None = unknown (markdown-only source)
    source_paragraph_index: int = 0
    text: str
    supports_claims: list[dict] = []
    # Provenance of externally fetched chunks (agentic-search / repair backfill).
    # A chunk may carry paper_id=X only when these fields verify it belongs to X.
    source_doc_id: str = ""
    source_title: str = ""
    source_chunk_id: str = ""


class EvidenceStore(BaseModel):
    task_id: str
    evidence: list[Evidence]


class Figure(BaseModel):
    figure_id: str
    paper_id: str
    caption: str = ""
    image_path: str | None = None
    extracted_text: str = ""
    page: int = 0
    usable_in_report: bool = True
    generation_hint: dict = {}


class FigureBank(BaseModel):
    task_id: str
    figures: list[Figure]


class Table(BaseModel):
    table_id: str
    paper_id: str
    caption: str = ""
    page: int = 0


class TableBank(BaseModel):
    task_id: str
    tables: list[Table]


class Category(BaseModel):
    category_id: str
    category_name: str
    description: str = ""
    related_aspects: list[str] = []
    paper_ids: list[str] = []
    stage_order: int = 0
    paper_count: int = 0
    adjustment_note: str = ""


class Taxonomy(BaseModel):
    task_id: str
    topic: str
    taxonomy_version: str = "v1"
    refine_history: list[str] = []
    categories: list[Category] = []


class CitationIndex(BaseModel):
    task_id: str
    citations: list[dict] = []
