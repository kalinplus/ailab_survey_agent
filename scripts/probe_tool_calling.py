"""Probe: does the real endpoint support native Chat Completions tool calling?

Five cases per endpoint, serial with the project rate limit:
  T1 basic_tools          - structured tool_calls returned? args valid JSON?
  T2 tight_budget         - same call under max_tokens=512; does hidden thinking
                            (reasoning_content / "Thinking Process") eat the
                            budget so tool_calls comes back empty?
  T2b thinking_disabled   - Intern only: thinking {"type":"disabled"} + tools.
  T3 tools_plus_json_mode - tools + response_format json_object together: 4xx?
  T4 round_trip           - feed the tool result back as role=tool messages;
                            does the model finish the loop (content, no error)?
  T5 strict_schema        - OpenAI-only extras (strict:true, parallel_tool_calls
                            false): rejected or silently ignored?

Usage: python scripts/probe_tool_calling.py [intern|heavy|all]
Writes output/tool_calling_probe_report.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config  # noqa: E402

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search academic papers by keyword. Returns paper titles.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search keyword"},
                "max_results": {"type": "integer", "description": "max papers, 1-10"},
            },
            "required": ["query", "max_results"],
        },
    },
}

ASK_MESSAGES = [
    {
        "role": "user",
        "content": (
            "You need recent papers about world models in games. "
            "Call the search_papers tool with query \"world model reinforcement learning\" "
            "and max_results 3. Do not answer from memory; use the tool."
        ),
    }
]


def load_endpoints(which: str) -> list[dict]:
    cfg = load_config()
    targets = []
    if which in ("intern", "all"):
        targets.append({
            "name": "intern",
            "base_url": cfg.intern_api_base_url.rstrip("/"),
            "api_key": cfg.intern_api_key,
            "model": cfg.intern_model_name,
        })
    if which in ("heavy", "all"):
        base = (os.getenv("HEAVY_LLM_BASE_URL") or "").strip().rstrip("/")
        key = (os.getenv("HEAVY_LLM_API_KEY") or "").strip()
        if base and key:
            targets.append({
                "name": "heavy",
                "base_url": base,
                "api_key": key,
                "model": (os.getenv("HEAVY_LLM_MODEL") or "").strip() or "deepseek-flash",
            })
    if not targets:
        raise SystemExit("no endpoint selected/configured")
    return targets


def post_chat(target: dict, payload: dict, timeout: float = 90.0) -> dict:
    """One raw chat/completions POST; returns {"ok":..., "status":..., "data"/"error"}."""
    url = f"{target['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {target['api_key']}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        return {"ok": False, "status": None, "error": f"transport: {exc}"}
    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code, "error": resp.text[:400]}
    try:
        return {"ok": True, "status": 200, "data": resp.json()}
    except json.JSONDecodeError as exc:
        return {"ok": False, "status": 200, "error": f"bad json body: {exc}"}


def analyze(reply: dict) -> dict:
    """Extract the evidence we care about from one chat response."""
    if not reply.get("ok"):
        return {"http_ok": False, "http_status": reply.get("status"), "error": reply.get("error")}
    data = reply["data"]
    choices = data.get("choices") or [{}]
    msg = (choices[0].get("message")) or {}
    tool_calls = msg.get("tool_calls") or []
    parsed_args = None
    args_error = None
    if tool_calls:
        raw = tool_calls[0].get("function", {}).get("arguments", "")
        try:
            parsed_args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError) as exc:
            args_error = str(exc)
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {
        "http_ok": True,
        "http_status": 200,
        "finish_reason": choices[0].get("finish_reason"),
        "has_tool_calls": bool(tool_calls),
        "tool_name": tool_calls[0].get("function", {}).get("name") if tool_calls else None,
        "args": parsed_args,
        "args_valid": (
            isinstance(parsed_args, dict)
            and isinstance(parsed_args.get("query"), str)
            and isinstance(parsed_args.get("max_results"), int)
        ) if parsed_args is not None else False,
        "args_error": args_error,
        "content_chars": len(content),
        "content_preview": content[:220],
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "reasoning_preview": (reasoning[:160] if isinstance(reasoning, str) else ""),
        "usage": data.get("usage"),
    }


def run_cases(target: dict) -> list[dict]:
    results: list[dict] = []
    min_interval = 2.1
    last_call = 0.0

    def call(case: str, payload: dict) -> dict:
        nonlocal last_call
        gap = time.monotonic() - last_call
        if gap < min_interval:
            time.sleep(min_interval - gap)
        started = time.monotonic()
        last_call = time.monotonic()
        reply = post_chat(target, payload)
        row = {"endpoint": target["name"], "model": target["model"], "case": case}
        row.update(analyze(reply))
        row["elapsed"] = round(time.monotonic() - started, 1)
        results.append(row)
        print(f"[{target['name']}] {case}: "
              f"tool_calls={row.get('has_tool_calls')} finish={row.get('finish_reason')} "
              f"args_valid={row.get('args_valid')} err={row.get('error')}")
        return row

    base = {"model": target["model"], "messages": ASK_MESSAGES, "temperature": 0.0}

    t1 = call("T1_basic_tools", {**base, "tools": [SEARCH_TOOL], "tool_choice": "auto"})

    call("T2_tight_budget", {
        **base, "tools": [SEARCH_TOOL], "tool_choice": "auto", "max_tokens": 512,
    })

    if target["name"] == "intern":
        call("T2b_thinking_disabled", {
            **base, "tools": [SEARCH_TOOL], "tool_choice": "auto",
            "thinking": {"type": "disabled"},
        })

    call("T3_tools_plus_json_mode", {
        **base, "tools": [SEARCH_TOOL], "tool_choice": "auto",
        "response_format": {"type": "json_object"},
    })

    # T4: only meaningful if T1 produced a structured tool call we can echo back.
    if t1.get("has_tool_calls") and t1.get("http_ok"):
        first_call = None
        for row in results:
            if row["case"] == "T1_basic_tools":
                first_call = row
                break
        # We need the raw tool_call object, not the digest; re-derive from args.
        fake_id = "call_probe_1"
        echo_payload = {
            "model": target["model"],
            "messages": [
                ASK_MESSAGES[0],
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": fake_id,
                        "type": "function",
                        "function": {
                            "name": "search_papers",
                            "arguments": json.dumps(first_call.get("args") or {}),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": fake_id,
                    "content": json.dumps([
                        "DreamerV3: Mastering Diverse Domains through World Models",
                        "Genie: Generative Interactive Environments",
                        "World Models",
                    ]),
                },
                {"role": "user", "content": "Summarize in one sentence what you found."},
            ],
            "temperature": 0.0,
        }
        row = call("T4_round_trip", echo_payload)
        row["loop_finished_with_content"] = bool(row.get("content_chars"))
    else:
        results.append({
            "endpoint": target["name"], "case": "T4_round_trip", "skipped":
            "T1 produced no structured tool_calls",
        })
        print(f"[{target['name']}] T4_round_trip: SKIPPED (no structured tool call in T1)")

    call("T5_strict_schema", {
        **base,
        "tools": [{**SEARCH_TOOL, "strict": True}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    })
    return results


def run_smoke(endpoint_name: str) -> None:
    """Full-chain smoke: a real BoundedAgentLoop in native tool-calling mode.

    The model must drive a 3-turn plan (increment +3, increment +4, finish)
    through structured tool_calls with role=tool observations round-tripped
    against the live endpoint.
    """
    from llm_client import InternS2Client, heavy_llm_client
    from harness.agents.loop import AgentConfig, BoundedAgentLoop, LoopFinished

    cfg = load_config()
    client = (InternS2Client(cfg) if endpoint_name == "intern" else heavy_llm_client(cfg))

    counter = {"value": 0}

    def increment(decision: dict) -> dict:
        counter["value"] += int(decision.get("step", 0))
        return {"value": counter["value"]}

    def finish(decision: dict) -> dict:
        raise LoopFinished({"final_value": counter["value"]})

    config = AgentConfig(
        name="probe_smoke",
        system_prompt=(
            "You operate a counter through tool calls. Follow the task exactly: "
            "call increment with the given steps, then call finish. "
            "One tool call per turn."),
        tools={"increment": increment, "finish": finish},
        tool_schemas={
            "increment": {
                "description": "Add step to the running counter; returns the new value.",
                "parameters": {"type": "object",
                               "properties": {"step": {"type": "integer"}},
                               "required": ["step"]},
            },
            "finish": {
                "description": "Finish the task.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        max_turns=6, max_llm_calls=6, max_wallclock=180.0,
    )
    loop = BoundedAgentLoop(
        config,
        lambda messages: (_ for _ in ()).throw(AssertionError("json path hit in native smoke")),
        lambda messages, tools: client.tool_chat(messages, tools=tools),
    )
    result = loop.run("Add 3 to the counter, then add 4 more, then finish.")
    print(f"[smoke:{endpoint_name}] status={result.status} payload={result.payload} "
          f"turns={result.turns} llm_calls={result.llm_calls} elapsed={result.elapsed}")
    for record in result.history:
        print(f"  turn {record['turn']}: {record['action']} args={record['args']} -> {record['result_digest'][:80]}")
    expected = result.status == "finished" and result.payload == {"final_value": 7}
    print(f"[smoke:{endpoint_name}] {'PASS' if expected else 'FAIL'} (expected finished with final_value=7)")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "smoke":
        run_smoke(sys.argv[2] if len(sys.argv) > 2 else "intern")
        return
    targets = load_endpoints(which)
    all_rows: list[dict] = []
    for target in targets:
        print(f"=== probing {target['name']} ({target['model']} @ {target['base_url']}) ===")
        all_rows.extend(run_cases(target))
    out = Path(__file__).resolve().parent.parent / "output" / "tool_calling_probe_report.json"
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
