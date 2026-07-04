from tools.clients.llm_fake import FakeLLMClient

def test_json_chat_matches_tag():
    llm = FakeLLMClient(responses=[("taxonomy", {"categories": [{"name": "X"}]})])
    out = llm.json_chat([{"role": "user", "content": "build taxonomy now"}])
    assert out == {"categories": [{"name": "X"}]}

def test_chat_returns_text_default():
    llm = FakeLLMClient(default_text="hello")
    assert llm.chat([{"role": "user", "content": "anything"}]) == "hello"

def test_default_json_when_no_match():
    llm = FakeLLMClient()
    assert llm.json_chat([{"role": "user", "content": "nope"}])["categories"][0]["name"] == "Default"
