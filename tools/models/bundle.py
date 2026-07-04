from pydantic import BaseModel


class KnowledgeBundle(BaseModel):
    task_id: str
    topic: str
    status: str
    artifacts: dict
    summary: dict
    coverage_report: dict
    quality_report: dict


class CitationEntry(BaseModel):
    citation_id: str
    valid: bool
    figures_valid: list[dict] = []
    tables_valid: list[dict] = []


class CitationResult(BaseModel):
    task_id: str
    total_citations: int
    valid_citations: int
    invalid_citations: int
    weak_claims: int = 0
    citation_validity_score: float
    entries: list[CitationEntry]


class ClaimEntry(BaseModel):
    claim_text: str
    cited_paper_id: str
    status: str
    evidence_ids: list[str] = []
    confidence: float = 0.0


class ClaimMap(BaseModel):
    task_id: str
    entries: list[ClaimEntry]
