"""Unit tests for the strategy agent joint (harness/agents/strategy_agent.py).

Spec §4.1 test_strategy_gate points: all-pass gate, accept rejected below
n_unique>=3, best-so-far rollback, fixpoint early stop, R0 filters=none
passthrough, R2 discovery mining — plus budget refusal, naming fallback,
invalid-edit rejection, and trajectory writing. Fake SciVerse mirrors the
real contract (meta_search -> {"results": [...]}, agentic_search -> {"hits"}).
"""

import json

from harness.agents.relevance import EmbeddingScorer
from harness.agents.strategy_agent import run_strategy_agent
from harness.search_strategy_builder import _validate_strategy


def _kw_scorer():
    scorer = EmbeddingScorer()
    scorer._failed = True  # keyword column: deterministic, no model download
    return scorer


def _hits(n, prefix, keywords=None, year=2023):
    return [{"unique_id": f"uid-{prefix}-{i}", "title": f"{prefix} paper {i}",
             "year": year, "abstract": f"{prefix} abstract about {prefix} methods",
             "keywords": keywords or [f"{prefix} kw", "shared kw"],
             "citation_count": 10, "doi": f"10.1/{prefix}.{i}"}
            for i in range(1, n + 1)]


class FakeSciverse:
    """Mirrors the real SciVerse client surface the joint uses."""

    def __init__(self, meta_by_needle=None, agentic_by_needle=None):
        self.meta_by_needle = meta_by_needle or {}
        self.agentic_by_needle = agentic_by_needle or {}
        self.meta_calls = []
        self.agentic_calls = []

    def meta_search(self, query, filters=None, page_size=25, **kw):
        self.meta_calls.append({"query": query, "filters": filters, "page_size": page_size})
        for needle, results in self.meta_by_needle.items():
            if needle in query:
                return {"results": results}
        return {"results": []}

    def agentic_search(self, query, top_k=10, filters=None):
        self.agentic_calls.append({"query": query, "top_k": top_k})
        for needle, hits in self.agentic_by_needle.items():
            if needle in query:
                return {"hits": hits}
        return {"hits": []}


class ScriptedLLM:
    """Order-scripted llm_json_chat: reply 0 is the naming call, then actions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages_seen = []

    def __call__(self, messages):
        self.messages_seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _plan(names_keywords):
    return {
        "main_domain": "World Models",
        "organization_mode": "thematic",
        "organization_reason": "technical blocks",
        "aspects": [
            {"name": name, "description": f"aspect about {name}",
             "min_papers": 3, "keywords": kws}
            for name, kws in names_keywords
        ],
    }


GOOD_PLAN = _plan([
    ("Latent World Model Learning", ["dreamer world model", "latent dynamics model",
                                     "world model reinforcement learning", "recurrent state space model",
                                     "model-based rl"]),
    ("Neural Game Engines", ["neural game engine", "video game diffusion",
                             "game generation transformer", "interactive video generation",
                             "diffusion game simulator"]),
    ("Game Agent Benchmarks", ["game agent benchmark", "minecraft agent evaluation",
                               "game playing llm benchmark", "atari world model benchmark",
                               "agent evaluation protocol"]),
])

# keyword 'deadword' matches nothing in FakeSciverse by default
DEAD_PLAN = _plan([
    ("Alive Aspect One", ["alpha query", "alpha method", "alpha model", "alpha survey", "alpha dataset"]),
    ("Alive Aspect Two", ["beta query", "beta method", "beta model", "beta survey", "beta dataset"]),
    ("Dead Aspect", ["deadword query", "deadword method", "deadword model",
                     "deadword survey", "deadword dataset"]),
])


def _run(fake_sv, llm, tmp_path, **kwargs):
    defaults = dict(
        task_id="task_test_001", topic="world models for games",
        max_papers=20, max_core_papers=8, end_year=2026,
        llm_json_chat=llm, sciverse=fake_sv,
        trajectory_dir=tmp_path / "traj", scorer=_kw_scorer(),
    )
    defaults.update(kwargs)
    return run_strategy_agent(**defaults)


def _meta_fake():
    """Probe + aspect sampling: alive aspects get unique papers, deadword gets none.

    Aspect sampling joins ALL keywords into one query, so needles must appear
    inside the joined string (first keyword of each aspect works).
    """
    return FakeSciverse(meta_by_needle={
        "survey review": _hits(6, "probe"),
        "methods benchmark dataset": _hits(4, "probe2"),
        # GOOD_PLAN aspects (matched via their first keyword)
        "dreamer": _hits(6, "gamma"),
        "neural game engine": _hits(6, "delta"),
        "game agent benchmark": _hits(6, "epsilon"),
        # DEAD_PLAN aspects
        "alpha": _hits(6, "alpha"),
        "beta": _hits(6, "beta"),
        "deadword": [],
        "rescued": _hits(6, "rescued"),
    })


# --- gate --------------------------------------------------------------------


def test_all_pass_accept_outputs_valid_strategy(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([GOOD_PLAN, {"action": "accept"}])
    strategy = _run(sv, llm, tmp_path)
    _validate_strategy(strategy, topic="world models for games")  # must not raise
    meta = strategy["strategy_agent"]
    assert meta["status"] == "finished:accept"
    assert meta["rounds"] == 1
    assert meta["scores"][0] > 0
    # every aspect passed the hard gate before accept fired
    assert all(row["n_unique"] >= 3 for row in
               [r for r in _last_card(tmp_path)["card"]["aspects"]])


def _last_card(tmp_path):
    lines = (tmp_path / "traj" / "task_test_001_strategy.jsonl").read_text().splitlines()
    cards = [json.loads(l) for l in lines if '"report_card"' in l]
    return cards[-1]


def test_accept_rejected_when_aspect_below_threshold(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([DEAD_PLAN,
                       {"action": "accept"},   # dead aspect n_unique=0 -> rejected
                       {"action": "accept"}])  # rejected again -> llm budget ends loop
    strategy = _run(sv, llm, tmp_path, max_llm_calls=3)
    meta = strategy["strategy_agent"]
    assert meta["status"] == "budget_llm_calls"
    assert meta["rounds"] == 1  # no round consumed by failed accepts
    # the rejection observation reached the model
    accepts = [m for call in llm.messages_seen[1:] for m in call
               if m["role"] == "user" and "gate rejected" in m["content"]]
    assert accepts
    _validate_strategy(strategy, topic="world models for games")


def test_rollback_returns_round1_when_round2_worse(tmp_path):
    sv = _meta_fake()
    # round 2 rewrote the alpha aspect into something that matches nothing
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "commit_edits", "edits": [
            {"op": "rewrite", "aspect_id": "aspect_001",
             "keywords": ["deadword query", "deadword method", "deadword model",
                          "deadword survey", "deadword dataset"]}]},
        {"action": "accept"},
    ])
    strategy = _run(sv, llm, tmp_path)
    meta = strategy["strategy_agent"]
    assert meta["rounds"] == 2
    # spec hard guarantee: score(final) >= score(round_1)
    assert meta["final_score"] >= meta["scores"][0]
    # rollback actually picked round 1's aspects (alpha aspect kept)
    names = [a["aspect_name"] for a in strategy["wide_search"]["search_aspects"]]
    assert "Latent World Model Learning" in names
    assert all("deadword" not in k for a in strategy["wide_search"]["search_aspects"]
               for k in a["keywords"])


def test_fixpoint_stops_early(tmp_path):
    sv = _meta_fake()
    # round 2 edits nothing that changes retrieval (same needles) -> tiny improvement
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "commit_edits", "edits": [
            {"op": "rewrite", "aspect_id": "aspect_001", "name": "Latent World Model Learning"}]},
    ])
    strategy = _run(sv, llm, tmp_path, fixpoint_epsilon=0.05)
    meta = strategy["strategy_agent"]
    assert "fixpoint" in meta["status"]
    assert meta["rounds"] == 2


# --- rescue tools ------------------------------------------------------------


def test_r0_filters_none_passthrough(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "probe_query", "query": "deadword niche query", "filters": "none"},
        {"action": "probe_query", "query": "deadword niche query", "filters": "default"},
        {"action": "accept"},
    ])
    _run(sv, llm, tmp_path)
    probe_calls = [c for c in sv.meta_calls if "deadword niche" in c["query"]]
    assert len(probe_calls) == 2
    assert probe_calls[0]["filters"] is None           # R0: filter bypassed
    assert probe_calls[1]["filters"] is not None       # default: year filter applied
    assert probe_calls[1]["filters"][0]["field"] == "publication_published_year"


def test_r2_discovery_mines_titles_and_keywords(tmp_path):
    sv = _meta_fake()
    sv.agentic_by_needle["semantic"] = [
        {"title": "Program Synthesis via Execution Feedback",
         "keywords": ["program synthesis", "neural symbolic"]},
        {"title": "Neurosymbolic Code Generation",
         "keywords": ["neurosymbolic", "program synthesis"]},
    ]
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "discover_by_description",
         "text": "semantic description of the dead subfield"},
        {"action": "accept"},
    ])
    _run(sv, llm, tmp_path)
    assert len(sv.agentic_calls) == 1
    obs = [m for m in llm.messages_seen[2]
           if m["role"] == "user" and "Observation" in m["content"]]
    text = obs[0]["content"]
    assert "Program Synthesis" in text          # titles surfaced
    assert "program synthesis" in text          # mined field keywords surfaced


def test_r3_donated_keywords_injected_for_starving_aspect(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([DEAD_PLAN, {"action": "accept"}, {"action": "accept"}])
    _run(sv, llm, tmp_path, max_llm_calls=3)
    # starving 'Dead Aspect' row got donated keywords from nearest retrieved papers
    state_texts = [m["content"] for call in llm.messages_seen[1:] for m in call
                   if m["role"] == "user" and "donated keywords" in m["content"]]
    assert state_texts
    assert "alpha kw" in state_texts[0] or "beta kw" in state_texts[0]


# --- delegated search subagent (specs/搜索SubAgent工具.md) ---------------------


def _main_observation(llm, needle):
    """Newest observation of the first MAIN-loop turn whose latest observation has needle.

    Both loops speak the same "Observation:" format and the message list grows, so a
    main-loop snapshot is told apart by its auto-injected report-card state block
    (the subagent has no state_provider); within a snapshot the model reacts to the
    LAST observation.
    """
    for snapshot in llm.messages_seen:
        if not any("[framework state]" in m["content"] and "report card" in m["content"]
                   for m in snapshot):
            continue
        observations = [m["content"] for m in snapshot
                        if m["role"] == "user" and "Observation" in m["content"]]
        if observations and needle in observations[-1]:
            return observations[-1]
    raise AssertionError(f"no main-line observation containing {needle!r}")


def _observation_payload(text):
    """Parse the delegate_search summary JSON out of a main-line observation."""
    body = text.split("Observation:\n", 1)[1].split("\n\n[framework state]")[0]
    return json.loads(body)


def _trajectory_events(tmp_path):
    lines = (tmp_path / "traj" / "task_test_001_strategy.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_delegate_search_returns_compact_summary(tmp_path):
    sv = _meta_fake()
    sv.meta_by_needle["deeper dive"] = _hits(4, "deep")
    sv.agentic_by_needle["deeper dive"] = [
        {"title": "Deeper Dive World Models", "keywords": ["deeper dive", "latent dynamics"]}]
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "delegate_search", "goal": "explore the deeper-dive sub-field",
         "queries": ["deeper dive methods"]},
        {"action": "discover", "text": "semantic description of the deeper dive sub-field"},
        {"action": "search_papers", "query": "deeper dive methods"},
        {"action": "report_findings", "notes": "sub-field exists; 'deeper dive' works"},
        {"action": "accept"},
    ])
    strategy = _run(sv, llm, tmp_path)
    meta = strategy["strategy_agent"]
    assert meta["subagent"] == {"delegations": 1, "llm_calls": 3}
    # the subagent searched through the shared counting proxy: agentic + meta, filters=none
    assert "deeper dive" in sv.agentic_calls[-1]["query"]
    assert sv.meta_calls[-1]["query"] == "deeper dive methods"
    assert sv.meta_calls[-1]["filters"] is None
    assert sv.meta_calls[-1]["page_size"] == 8
    assert meta["sciverse_calls"] == len(sv.meta_calls) + len(sv.agentic_calls)
    payload = _observation_payload(_main_observation(llm, '"n_queries"'))
    assert set(payload) == {"n_queries", "papers", "field_keywords", "notes"}
    assert payload["n_queries"] == 2
    assert payload["notes"] == "sub-field exists; 'deeper dive' works"
    assert len(payload["papers"]) <= 8 and len(payload["field_keywords"]) <= 10
    assert all(set(p) == {"title", "year"} for p in payload["papers"])
    papers_text = json.dumps(payload["papers"])
    assert "deep paper 1" in papers_text and "Deeper Dive World Models" in papers_text
    # raw subagent retrieval output never reaches the main-line observation
    text = _main_observation(llm, '"n_queries"')
    for leaked in ("unique_id", "abstract", "citation_count", "doi"):
        assert leaked not in text
    # subagent turns share the same trajectory file under their own agent name
    sub_turns = [e for e in _trajectory_events(tmp_path)
                 if e.get("agent") == "strategy_subagent" and "action" in e]
    assert [t["action"] for t in sub_turns] == ["discover", "search_papers", "report_findings"]


def test_subagent_toolbox_has_no_delegate_or_edit_tools(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "delegate_search", "goal": "scout a brand-new direction"},
        {"action": "delegate_search", "goal": "recursive delegation attempt"},
        {"action": "report_findings", "notes": "nothing usable"},
        {"action": "accept"},
    ])
    strategy = _run(sv, llm, tmp_path)
    sub_turns = [e for e in _trajectory_events(tmp_path)
                 if e.get("agent") == "strategy_subagent" and "action" in e]
    assert sub_turns[0]["action"] == "delegate_search"
    digest = sub_turns[0]["result_digest"]
    # the chassis rejection lists the real subagent toolbox: no recursion, no edit rights
    assert "unknown action 'delegate_search'" in digest
    assert "allowed actions: ['discover', 'report_findings', 'search_papers']" in digest
    assert "commit_edits" not in digest and "accept" not in digest
    # the main line only sees the summary, not the subagent's internal error
    assert strategy["strategy_agent"]["subagent"] == {"delegations": 1, "llm_calls": 2}
    payload = _observation_payload(_main_observation(llm, '"n_queries"'))
    assert payload["n_queries"] == 0 and payload["notes"] == "nothing usable"


def test_subagent_shares_sciverse_budget_and_main_loop_continues(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "delegate_search", "goal": "explore further after the sample"},
        {"action": "search_papers", "query": "alpha query"},
        {"action": "report_findings", "notes": "blocked"},
        {"action": "accept"},
    ])
    # probe(2) + sample(3) exhausts the joint budget before the subagent searches
    strategy = _run(sv, llm, tmp_path, max_sciverse_calls=5)
    text = _main_observation(llm, '"n_queries"')
    assert "SciVerse budget exhausted" in text          # error observation reaches the main line
    payload = _observation_payload(text)
    assert payload["n_queries"] == 0 and payload["papers"] == []
    assert len(sv.meta_calls) == 5                      # refused call never hit SciVerse
    assert strategy["strategy_agent"]["status"] == "finished:accept"  # main loop survived


def test_third_delegation_is_refused(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "delegate_search", "goal": "goal one"},
        {"action": "report_findings", "notes": "n1"},
        {"action": "delegate_search", "goal": "goal two"},
        {"action": "report_findings", "notes": "n2"},
        {"action": "delegate_search", "goal": "goal three"},
        {"action": "accept"},
    ])
    strategy = _run(sv, llm, tmp_path)
    assert strategy["strategy_agent"]["subagent"] == {"delegations": 2, "llm_calls": 2}
    assert "delegation budget exhausted (2 per run)" in _main_observation(llm, "delegation budget exhausted")


# --- budgets / robustness ------------------------------------------------------


def test_sciverse_budget_refusal_observation(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        GOOD_PLAN,
        {"action": "probe_query", "query": "anything at all"},
        {"action": "accept"},
    ])
    # probe(2) + sample(3) = 5 -> the 6th call must be refused with an observation
    _run(sv, llm, tmp_path, max_sciverse_calls=5)
    obs = [m for m in llm.messages_seen[2]
           if m["role"] == "user" and "Observation" in m["content"]]
    assert "SciVerse budget exhausted" in obs[0]["content"]


def test_naming_fallback_to_template(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([{"garbage": True}, {"action": "accept"}])
    strategy = _run(sv, llm, tmp_path)
    assert strategy["strategy_agent"]["naming_mode"] == "template_fallback"
    _validate_strategy(strategy, topic="world models for games")


def test_invalid_commit_rejected_without_consuming_round(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        GOOD_PLAN,
        # deleting 2 of 3 aspects leaves 1 -> invalid (3..6 required)
        {"action": "commit_edits", "edits": [
            {"op": "delete", "aspect_id": "aspect_001"},
            {"op": "delete", "aspect_id": "aspect_002"}]},
        {"action": "accept"},
    ])
    strategy = _run(sv, llm, tmp_path)
    assert strategy["strategy_agent"]["rounds"] == 1  # no round burned
    obs = [m for m in llm.messages_seen[2]
           if m["role"] == "user" and "Observation" in m["content"]]
    assert "edit rejected" in obs[0]["content"]
    assert len(strategy["wide_search"]["search_aspects"]) == 3


def test_deleted_aspect_records_lesson(tmp_path):
    sv = _meta_fake()
    plan = _plan([
        ("Alive Aspect One", ["alpha query", "alpha method", "alpha model", "alpha survey", "alpha dataset"]),
        ("Alive Aspect Two", ["beta query", "beta method", "beta model", "beta survey", "beta dataset"]),
        ("Alive Aspect Three", ["rescued query", "rescued method", "rescued model", "rescued survey", "rescued dataset"]),
        ("Dead Aspect", ["deadword query", "deadword method", "deadword model",
                         "deadword survey", "deadword dataset"]),
    ])
    llm = ScriptedLLM([
        plan,
        {"action": "commit_edits", "edits": [{"op": "delete", "aspect_id": "aspect_004"}]},
    ])
    strategy = _run(sv, llm, tmp_path)
    lessons = strategy["strategy_agent"]["lessons"]
    assert any("Dead Aspect" in lesson and "deleted" in lesson for lesson in lessons)
    assert len(strategy["wide_search"]["search_aspects"]) == 3


def test_rescue_edit_recovers_dead_aspect(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([
        DEAD_PLAN,
        {"action": "commit_edits", "edits": [
            {"op": "rewrite", "aspect_id": "aspect_003",
             "name": "Rescued Aspect", "keywords": ["rescued query", "rescued method",
                                                    "rescued model", "rescued survey",
                                                    "rescued dataset"]}]},
        {"action": "accept"},
    ])
    strategy = _run(sv, llm, tmp_path)
    meta = strategy["strategy_agent"]
    assert meta["rounds"] == 2
    assert meta["status"] == "finished:accept"
    assert meta["final_score"] >= meta["scores"][0]  # improved or rolled back
    names = [a["aspect_name"] for a in strategy["wide_search"]["search_aspects"]]
    assert "Rescued Aspect" in names  # rescue kept (it improved the card)


def test_trajectory_file_complete(tmp_path):
    sv = _meta_fake()
    llm = ScriptedLLM([GOOD_PLAN, {"action": "accept"}])
    _run(sv, llm, tmp_path)
    lines = (tmp_path / "traj" / "task_test_001_strategy.jsonl").read_text().splitlines()
    events = [json.loads(l) for l in lines]
    kinds = {e["event"] for e in events if "event" in e}
    assert {"start", "probe", "naming", "report_card", "loop_finish", "finish"} <= kinds
    turn_records = [e for e in events if e.get("action") == "accept"]
    assert turn_records and turn_records[0]["turn"] == 1


# --- memory round-trip (slice 4 acceptance: second run sees lessons) ---------


def test_lessons_persist_and_reload_for_same_topic(tmp_path):
    from harness.memory_manager import MemoryManager

    memory = MemoryManager(tmp_path, enabled=True)
    lessons = [
        "For world models for games: aspect 'Dead Aspect' starved and was deleted — "
        "this sub-area appears absent from the corpus; skip it on retry.",
    ]
    memory.append_lessons("world models for games", lessons)
    context = memory.load_for_strategy("world models for games")
    assert "Dead Aspect" in context  # second run on the same topic sees the lesson


def test_naming_fallback_anchors_generic_template_names():
    from harness.agents.strategy_agent import _name_aspects

    # non-game topic + failing LLM + no probe papers -> generic DEFAULT_ASPECTS;
    # anchoring must rename them so the strategy passes validation
    strategy, mode = _name_aspects(
        lambda messages: {"garbage": True},
        topic="vector databases", clusters=[], probe_papers=[], memory_context="",
        task_id="t", max_papers=20, max_core_papers=8, end_year=2026, cluster_count=4)
    assert mode == "template_fallback"
    _validate_strategy(strategy, topic="vector databases")
    names = [a["aspect_name"] for a in strategy["wide_search"]["search_aspects"]]
    assert all("for vector databases" in name for name in names)
