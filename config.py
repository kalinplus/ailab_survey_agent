"""Runtime configuration for EviSurvey."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:
    _load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    intern_api_base_url: str
    intern_api_key: str
    intern_model_name: str
    request_timeout_seconds: float
    max_llm_retries: int
    tool_timeout_seconds: float
    tool_result_max_chars: int
    default_language: str
    default_mode: str

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "cache"

    @property
    def requests_dir(self) -> Path:
        return self.root_dir / "requests"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"


def load_config() -> AppConfig:
    """Load environment-backed configuration."""

    _load_env_file(ROOT_DIR / ".env")

    return AppConfig(
        root_dir=ROOT_DIR,
        intern_api_base_url=os.getenv(
            "INTERN_API_BASE_URL",
            os.getenv("API_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"),
        ).rstrip("/"),
        intern_api_key=os.getenv("INTERN_API_KEY", os.getenv("API_KEY", "")),
        intern_model_name=os.getenv("INTERN_MODEL_NAME", "intern-s2-preview"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
        max_llm_retries=int(os.getenv("MAX_LLM_RETRIES", "1")),
        tool_timeout_seconds=float(os.getenv("TOOL_TIMEOUT_SECONDS", "300")),
        tool_result_max_chars=int(os.getenv("TOOL_RESULT_MAX_CHARS", "6000")),
        default_language=os.getenv("EVISURVEY_LANGUAGE", "zh"),
        default_mode=os.getenv("EVISURVEY_MODE", "demo"),
    )


def _load_env_file(path: Path) -> None:
    if _load_dotenv is not None:
        _load_dotenv(path)
        return
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_project_dirs(config: AppConfig) -> None:
    for path in [
        config.cache_dir,
        config.requests_dir,
        config.output_dir,
        config.logs_dir,
        config.cache_dir / "assets",
        config.cache_dir / "assets" / "papers",
        config.cache_dir / "assets" / "figures",
        config.cache_dir / "assets" / "tables",
        config.output_dir / "generated_assets",
    ]:
        path.mkdir(parents=True, exist_ok=True)
