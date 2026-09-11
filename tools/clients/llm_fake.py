import json


class FakeLLMClient:
    """Test double matching llm_client.InternS2Client's .chat/.json_chat/.tool_chat surface."""
    def __init__(self, responses: list[tuple[str, object]] | None = None,
                 default_text: str = "ok", default_json: dict | None = None):
        self.responses = responses or []
        self.default_text = default_text
        self.default_json = default_json or {"categories": [{"name": "Default", "description": "d"}]}
        self.calls = 0

    def _last_user(self, messages):
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def chat(self, messages, temperature=0.2, response_format=None, max_tokens=None, thinking_mode=False):
        self.calls += 1
        text = self._last_user(messages)
        for tag, resp in self.responses:
            if tag in text and isinstance(resp, str):
                return resp
        return self.default_text

    def json_chat(self, messages, temperature=0.1, max_tokens=None, thinking_mode=False):
        self.calls += 1
        text = self._last_user(messages)
        for tag, resp in self.responses:
            if tag in text and isinstance(resp, dict) and "tool_calls" not in resp:
                return resp
        return self.default_json

    def tool_chat(self, messages, tools, tool_choice="auto", temperature=0.1, max_tokens=None):
        self.calls += 1
        text = self._last_user(messages) or " ".join(
            str(m.get("content") or "") for m in messages[-3:])
        for tag, resp in self.responses:
            if tag in text and isinstance(resp, dict) and "tool_calls" in resp:
                return resp
        # Default: call the first exposed tool with the default JSON as arguments.
        first = tools[0]["function"]["name"]
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_fake_1", "type": "function",
            "function": {"name": first, "arguments": json.dumps(self.default_json)},
        }]}
