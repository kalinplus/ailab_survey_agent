"""Tests for harness.search_strategy_builder._strategy_prompt.

The prompt drives A-side topic decomposition; these guard the field-anchor
constraint that keeps retrieval inside the topic's own domain (regression for
the off-topic retrieval bug: world-model survey pulling in LLMs / pathology /
cognitive-science education).
"""

from harness.search_strategy_builder import _strategy_prompt


def _make(topic="世界模型", probe_papers=None):
    return _strategy_prompt(
        task_id="t", topic=topic, max_papers=10, max_core_papers=5,
        end_year=2026, probe_papers=probe_papers or [],
    )


def test_prompt_anchors_keywords_to_topic_field():
    prompt = _make()
    # force every keyword to stay inside the topic's own field
    assert "inside the topic" in prompt
    # explicitly reject lexically-overlapping neighbors
    assert "lexically" in prompt
    # concrete narrowing example is present (teaches LLM what "specific" means)
    assert "dreamer world model" in prompt


def test_prompt_carries_topic():
    prompt = _make(topic="世界模型")
    assert "世界模型" in prompt


def test_prompt_surfaces_probe_titles():
    """Probe papers from the live SciVerse probe are shown to the LLM."""
    probe = [{"title": "DreamerV3: Mastering Diverse Domains", "year": 2023}]
    prompt = _make(probe_papers=probe)
    assert "DreamerV3" in prompt


# --- TOPIC_NEUTRAL experiment switch (de-biased A/B baselines) ----------------


def test_topic_neutral_disables_demo_topic_priors(monkeypatch):
    from harness import search_strategy_builder as b

    monkeypatch.setattr(b, "TOPIC_NEUTRAL", True)
    templates = b._fallback_aspect_templates("world models for games")
    assert templates == b.DEFAULT_ASPECTS          # no GameCraft aspects
    assert not any("GameNGen" in q for q in b._probe_queries("world models for games"))
    prompt = _make(topic="world models for games")
    assert "GameCraft" not in prompt and "dreamer world model" not in prompt


def test_topic_priors_on_by_default(monkeypatch):
    from harness import search_strategy_builder as b

    monkeypatch.setattr(b, "TOPIC_NEUTRAL", False)
    templates = b._fallback_aspect_templates("world models for games")
    assert templates != b.DEFAULT_ASPECTS          # GameCraft aspects present
    prompt = _make(topic="world models for games")
    assert "GameCraft" in prompt


# --- S1 breadth: aspects must carry a keyword budget for the P3 query knob ------
# P3 expands each aspect into extra queries from its own keywords
# (EVISURVEY_MAX_QUERIES_PER_ASPECT); an aspect trimmed to 1-2 keywords would
# silently disable that knob.


def test_default_strategy_keywords_feed_query_expansion():
    from harness import search_strategy_builder as b

    strategy = b._build_default_strategy(
        task_id="t", topic="world models for games",
        max_papers=60, max_core_papers=15, end_year=2026,
    )
    aspects = strategy["wide_search"]["search_aspects"]
    assert 3 <= len(aspects) <= 6
    for aspect in aspects:
        assert len(aspect["keywords"]) >= 3
        assert len(aspect["keywords"]) == len(set(aspect["keywords"]))


def test_llm_strategy_keeps_keyword_budget_for_query_expansion():
    from harness.search_strategy_builder import _coerce_strategy

    raw = {
        "main_domain": "world models",
        "organization_mode": "thematic",
        "aspects": [
            {
                "name": f"Cluster {index}",
                "description": f"aspect {index}",
                "min_papers": 3,
                "keywords": [
                    "world model cluster", "dreamer world model", "gamecraft",
                    "latent dynamics", "interactive environment benchmark",
                ],
            }
            for index in range(4)
        ],
    }
    strategy = _coerce_strategy(
        raw, task_id="t", topic="世界模型", max_papers=60, max_core_papers=15, end_year=2026,
    )
    for aspect in strategy["wide_search"]["search_aspects"]:
        assert len(aspect["keywords"]) >= 5
