from tools.clients.llm_fake import FakeLLMClient
from llm_client import _extract_json_object

def test_json_chat_matches_tag():
    llm = FakeLLMClient(responses=[("taxonomy", {"categories": [{"name": "X"}]})])
    out = llm.json_chat([{"role": "user", "content": "build taxonomy now"}])
    assert out == {"categories": [{"name": "X"}]}

def test_chat_returns_text_default():
    llm = FakeLLMClient(default_text="hello")
    assert llm.chat([{"role": "user", "content": "anything"}], max_tokens=10) == "hello"

def test_default_json_when_no_match():
    llm = FakeLLMClient()
    assert llm.json_chat([{"role": "user", "content": "nope"}], max_tokens=10)["categories"][0]["name"] == "Default"


def test_extract_json_object_prefers_outer_object():
    text = "Thinking Process:\n{\"inner\": true}\nFinal:\n{\"items\": [{\"inner\": true}]}"
    assert _extract_json_object(text) == {"items": [{"inner": True}]}
