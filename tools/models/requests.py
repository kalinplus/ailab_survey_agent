from pydantic import BaseModel


class SearchAspect(BaseModel):
    aspect_id: str
    aspect_name: str
    keywords: list[str]
    min_papers: int = 3


class TimeRange(BaseModel):
    start_year: int
    end_year: int


class SearchConstraints(BaseModel):
    max_papers: int = 40
    max_core_papers: int = 15
    time_range: TimeRange | None = None


class SearchStrategy(BaseModel):
    topic: str
    sub_domains: list[str] = []
    wide_search: dict = {}


class PipelineConfig(BaseModel):
    use_online_search: bool = True
    use_seed_fallback: bool = True
    use_mineru: bool = False
    use_mock_mineru_if_failed: bool = False  # production runs real MinerU; phase3 degrades per-paper
    update_survey_store: bool = False
    max_papers: int = 40
    max_core_papers: int = 15
    aspect_match_threshold: float = 0.6


class QualityRequirements(BaseModel):
    min_total_papers: int = 10
    min_core_papers: int = 5
    min_figures: int = 1
    min_evidence_per_core_paper: int = 1


class KnowledgeBuildRequest(BaseModel):
    task_id: str
    topic: str
    request_type: str = "knowledge_build"
    inputs: dict
    outputs: dict
    pipeline_config: PipelineConfig = PipelineConfig()
    quality_requirements: QualityRequirements = QualityRequirements()
