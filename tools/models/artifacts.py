from pydantic import BaseModel


class RetrievedPaper(BaseModel):
    paper_id: str
    title: str
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
    page: int
    index: int
    text: str


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


class PaperCard(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    matched_aspects: list[dict] = []
    category_id: str | None = None
    category: str = ""
    card_type: str = "deep"
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
    source_page: int = 0
    source_paragraph_index: int = 0
    text: str
    supports_claims: list[dict] = []


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
