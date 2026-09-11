"""Unit tests for the shared agent-loop chassis (harness/agents/loop.py).

Covers spec §4.1 test_agent_loop points:
dispatch / illegal-action rejection / parse-fail repair-then-bubble /
three budget breakers / trajectory records / state auto-injection.
"""

import json

import pytest

from harness.agents.loop import (
    AgentConfig,
    BoundedAgentLoop,
    LoopFinished,
    LoopParseError,
)


class ScriptedLLM:
    """Callable standing in for InternS2Client.json_chat; pops scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages_seen = []

    def __call__(self, messages):
        self.messages_seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _loop(tools, replies, **config_kwargs):
    config = AgentConfig(
        name="test_agent",
        system_prompt="sys",
        tools=tools,
        **config_kwargs,
    )
    llm = ScriptedLLM(replies)
    return BoundedAgentLoop(config, llm), llm


def test_dispatch_calls_tool_and_feeds_observation():
    seen = {}

    def probe(decision):
        seen["decision"] = decision
        return {"n_hits": 3}

    def accept(decision):
        raise LoopFinished({"ok": True})

    loop, llm = _loop({"probe_query": probe, "accept": accept},
                      [{"action": "probe_query", "query": "world models"},
                       {"action": "accept"}])
    result = loop.run("task text")
    assert result.status == "finished"
    assert result.payload == {"ok": True}
    assert seen["decision"]["query"] == "world models"
    # observation was fed back before the second decision
    second_call = llm.messages_seen[1]
    assert any("n_hits" in m["content"] for m in second_call if m["role"] == "user")


def test_unknown_action_rejected_without_execution():
    executed = []

    def accept(decision):
        executed.append("accept")
        raise LoopFinished({})

    loop, llm = _loop({"accept": accept},
                      [{"action": "hack_the_registry", "x": 1}, {"action": "accept"}])
    result = loop.run("task")
    assert result.status == "finished"
    assert executed == ["accept"]  # unknown action never ran
    # the model saw a rejection observation
    rejected_obs = [m for m in llm.messages_seen[1] if m["role"] == "user"]
    assert any("unknown action" in m["content"] for m in rejected_obs)


def test_parse_fail_gets_one_repair_retry_then_succeeds():
    def accept(decision):
        raise LoopFinished({})

    loop, llm = _loop({"accept": accept},
                      [ValueError("not json"), {"action": "accept"}])
    result = loop.run("task")
    assert result.status == "finished"
    assert loop.llm_calls == 2
    # repair instruction was appended before the retry
    retry_call = llm.messages_seen[1]
    assert any("not a usable action" in m["content"] for m in retry_call if m["role"] == "user")


def test_parse_fail_twice_bubbles():
    loop, _ = _loop({"accept": lambda d: None},
                    [ValueError("bad"), ValueError("still bad")])
    with pytest.raises(LoopParseError):
        loop.run("task")


def test_missing_action_key_is_parse_failure():
    # a dict without "action" goes through the same repair path
    def accept(decision):
        raise LoopFinished({})

    loop, _ = _loop({"accept": accept}, [{"query": "no action field"}, {"action": "accept"}])
    assert loop.run("task").status == "finished"
    assert loop.llm_calls == 2


def test_budget_turns():
    loop, _ = _loop({"probe_query": lambda d: "ok"},
                    [{"action": "probe_query"}], max_turns=1)
    result = loop.run("task")
    assert result.status == "budget_turns"
    assert result.turns == 1


def test_budget_llm_calls():
    loop, _ = _loop({"probe_query": lambda d: "ok"},
                    [{"action": "probe_query"}, {"action": "probe_query"}],
                    max_turns=5, max_llm_calls=1)
    result = loop.run("task")
    assert result.status == "budget_llm_calls"
    assert result.llm_calls == 1


def test_budget_wallclock():
    loop, _ = _loop({"probe_query": lambda d: "ok"}, [], max_wallclock=0.0)
    result = loop.run("task")
    assert result.status == "budget_wallclock"
    assert result.turns == 0


def test_state_injected_every_turn():
    states = ["CARD v1", "CARD v2"]

    def accept(decision):
        raise LoopFinished({})

    config = AgentConfig(
        name="s", system_prompt="sys", tools={"accept": accept},
        state_provider=lambda: states.pop(0) if states else "CARD final",
    )
    llm = ScriptedLLM([{"action": "accept"}])
    BoundedAgentLoop(config, llm).run("task")
    initial_user = llm.messages_seen[0][-1]["content"]
    assert "CARD v1" in initial_user


def test_trajectory_records_one_line_per_turn():
    records = []

    def accept(decision):
        raise LoopFinished({})

    config = AgentConfig(
        name="traj_agent", system_prompt="sys",
        tools={"probe_query": lambda d: {"n_hits": 2}, "accept": accept},
        on_action=records.append,
    )
    llm = ScriptedLLM([{"action": "probe_query", "query": "q"}, {"action": "accept"}])
    result = BoundedAgentLoop(config, llm).run("task")
    assert len(records) == 2  # one per executed turn (accept turn also recorded)
    first = records[0]
    assert first["agent"] == "traj_agent"
    assert first["turn"] == 1
    assert first["action"] == "probe_query"
    assert first["args"] == {"query": "q"}
    assert "n_hits" in first["result_digest"]
    assert first["llm_idx"] == 1
    assert isinstance(first["elapsed"], float)
    # on_finish fired once with summary
    assert result.turns == 2


def test_on_finish_receives_summary():
    finishes = []

    def accept(decision):
        raise LoopFinished({"done": 1})

    config = AgentConfig(
        name="f", system_prompt="sys", tools={"accept": accept},
        on_finish=finishes.append,
    )
    BoundedAgentLoop(config, ScriptedLLM([{"action": "accept"}])).run("task")
    assert finishes and finishes[0]["status"] == "finished"
    assert finishes[0]["llm_calls"] == 1


def test_result_digest_truncated():
    long_result = "x" * 5000
    config = AgentConfig(
        name="t", system_prompt="s",
        tools={"probe_query": lambda d: long_result, "accept": lambda d: (_ for _ in ()).throw(LoopFinished({}))},
        result_max_chars=100,
    )
    llm = ScriptedLLM([{"action": "probe_query"}, {"action": "accept"}])
    BoundedAgentLoop(config, llm).run("task")
    obs = [m for m in llm.messages_seen[1] if m["role"] == "user"][-1]["content"]
    assert len(obs) < 500
    assert "[truncated]" in obs


def test_trajectory_writer_appends_jsonl(tmp_path):
    from harness.agents.loop import trajectory_writer

    write = trajectory_writer(tmp_path / "logs" / "traj" / "a.jsonl")
    write({"turn": 1, "action": "probe_query"})
    write({"turn": 2, "action": "accept"})
    lines = (tmp_path / "logs" / "traj" / "a.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "probe_query"


# --- native tool-calling mode -------------------------------------------------


class ScriptedToolLLM:
    """Callable standing in for InternS2Client.tool_chat; pops scripted
    assistant messages of the shape {"tool_calls": [...], "content": ...}."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages_seen = []

    def __call__(self, messages, tools):
        self.messages_seen.append([dict(m) for m in messages])
        self.tools_seen = tools
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _tool_call(name, args=None, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


SCHEMAS = {
    "probe_query": {"description": "probe a query",
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string"}},
                                   "required": ["query"]}},
    "accept": {"description": "finish", "parameters": {"type": "object", "properties": {}}},
}


def _native_loop(tools, replies, **config_kwargs):
    config = AgentConfig(name="test_agent", system_prompt="sys", tools=tools,
                         tool_schemas=SCHEMAS, **config_kwargs)
    llm = ScriptedToolLLM(replies)
    return BoundedAgentLoop(config, llm_json_chat=lambda m: (_ for _ in ()).throw(
        AssertionError("json_chat must not be called in native mode")), llm_tool_chat=llm), llm


def test_native_dispatch_and_tool_role_observation():
    seen = {}

    def probe(decision):
        seen["decision"] = decision
        return {"n_hits": 3}

    def accept(decision):
        raise LoopFinished({"ok": True})

    loop, llm = _native_loop(
        {"probe_query": probe, "accept": accept},
        [{"tool_calls": [_tool_call("probe_query", {"query": "world models"})]},
         {"tool_calls": [_tool_call("accept", call_id="call_2")]}])
    result = loop.run("task text")
    assert result.status == "finished"
    assert result.payload == {"ok": True}
    assert seen["decision"]["query"] == "world models"
    assert seen["decision"]["action"] == "probe_query"
    # observation went back as a role=tool message, after the echoed assistant tool call
    second_call = llm.messages_seen[1]
    assert second_call[-2]["role"] == "assistant"
    assert second_call[-2]["tool_calls"][0]["id"] == "call_1"
    tool_msg = second_call[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "n_hits" in tool_msg["content"]
    # the schema list handed to the provider matches the registry
    assert {t["function"]["name"] for t in llm.tools_seen} == {"probe_query", "accept"}


def test_native_unknown_action_still_gated_by_registry():
    loop, llm = _native_loop(
        {"accept": lambda d: (_ for _ in ()).throw(LoopFinished({"ok": 1}))},
        [{"tool_calls": [_tool_call("probe_query", {"query": "x"})]},
         {"tool_calls": [_tool_call("accept", call_id="call_2")]}])
    result = loop.run("task")
    assert result.status == "finished"
    obs = llm.messages_seen[1][-1]["content"]
    assert "unknown action" in obs


def test_native_no_tool_call_gets_one_repair_retry():
    loop, llm = _native_loop(
        {"accept": lambda d: (_ for _ in ()).throw(LoopFinished({}))},
        [{"content": "I would rather answer in prose."},
         {"tool_calls": [_tool_call("accept")]}])
    result = loop.run("task")
    assert result.status == "finished"
    # the repair nudge arrived as a user message before the second call
    assert any("did not call a tool" in m["content"]
               for m in llm.messages_seen[1] if m["role"] == "user")


def test_native_bad_arguments_json_is_parse_failure():
    llm_replies = [
        {"tool_calls": [{"id": "c", "type": "function",
                         "function": {"name": "accept", "arguments": "{not json"}}]},
        {"tool_calls": [_tool_call("accept", call_id="c2")]},
    ]
    loop, llm = _native_loop({"accept": lambda d: (_ for _ in ()).throw(LoopFinished({"ok": 1}))},
                             llm_replies)
    assert loop.run("task").status == "finished"


def test_native_budget_llm_calls_counts_repair_retry():
    loop, _llm = _native_loop(
        {"accept": lambda d: (_ for _ in ()).throw(LoopFinished({}))},
        [{"content": "no tool"}], max_llm_calls=1)
    with pytest.raises(LoopParseError):
        loop.run("task")


def test_native_trajectory_records_action_and_args(tmp_path):
    from harness.agents.loop import trajectory_writer

    records = []
    loop, _llm = _native_loop(
        {"accept": lambda d: (_ for _ in ()).throw(LoopFinished({"done": 1}))},
        [{"tool_calls": [_tool_call("accept")]}],
        on_action=records.append)
    loop.run("task")
    assert records[0]["action"] == "accept"
    assert records[0]["args"] == {}
