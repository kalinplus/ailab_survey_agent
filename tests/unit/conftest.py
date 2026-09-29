import pytest

from tools import write_survey


@pytest.fixture(autouse=True)
def _no_jev_env(monkeypatch):
    """JEV must stay off unless a test opts in — .env keys must never turn a
    unit test into a network call."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("EVISURVEY_JEV", "off")


@pytest.fixture(autouse=True)
def _writer_rejects_to_tmp(tmp_path, monkeypatch):
    """Writer diagnostics must never append to the live output tree from tests
    (the wave8 suite polluted output/writer_llm_rejects.jsonl with fixture
    replies before this fixture existed)."""
    monkeypatch.setattr(write_survey, "_WRITER_REJECTS_PATH", tmp_path / "writer_llm_rejects.jsonl")
