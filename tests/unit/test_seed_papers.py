import json
from pathlib import Path

CATEGORIES = {
    "Game as AI Benchmark",
    "Internal World Model",
    "Neural Game Engine",
    "Foundation Interactive Game World Model",
    "LLM / MLLM Game Agent",
    "Benchmark / Evaluation / Toolkit",
}


def _load():
    return json.loads(Path("cache/seed_papers.json").read_text(encoding="utf-8"))


def test_seed_papers_cover_six_directions():
    data = _load()
    cats = {p["category"] for p in data}
    assert len(data) >= 20, f"expected >=20 papers, got {len(data)}"
    assert cats == CATEGORIES, f"missing/extra categories: {cats ^ CATEGORIES}"


def test_seed_papers_have_required_fields():
    for p in _load():
        assert p.get("title"), f"missing title: {p}"
        assert p.get("year"), f"missing year: {p}"
        assert "keywords" in p, f"missing keywords key: {p}"
        assert p.get("category") in CATEGORIES, f"bad category: {p.get('category')}"
        assert isinstance(p.get("authors"), list) and p["authors"], f"missing authors: {p}"
        assert "url" in p, f"missing url key: {p}"


def test_seed_papers_unique_bibtex_keys():
    keys = [p["bibtex_key"] for p in _load()]
    assert len(keys) == len(set(keys)), f"duplicate bibtex_keys: {keys}"
