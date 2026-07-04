class FakeLLMClient:
    """Test double matching llm_client.InternS2Client's .chat/.json_chat surface."""
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

    def chat(self, messages, temperature=0.2, response_format=None):
        self.calls += 1
        text = self._last_user(messages)
        for tag, resp in self.responses:
            if tag in text and isinstance(resp, str):
                return resp
        return self.default_text

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        text = self._last_user(messages)
        for tag, resp in self.responses:
            if tag in text and isinstance(resp, dict):
                return resp
        return self.default_json
