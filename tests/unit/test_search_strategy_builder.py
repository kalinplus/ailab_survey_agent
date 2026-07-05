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
