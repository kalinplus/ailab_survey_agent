import pytest

from tools import write_survey


@pytest.fixture(autouse=True)
def _writer_rejects_to_tmp(tmp_path, monkeypatch):
    """Writer diagnostics must never append to the live output tree from tests
    (the wave8 suite polluted output/writer_llm_rejects.jsonl with fixture
    replies before this fixture existed)."""
    monkeypatch.setattr(write_survey, "_WRITER_REJECTS_PATH", tmp_path / "writer_llm_rejects.jsonl")
