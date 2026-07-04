def test_tools_package_importable():
    import tools  # noqa: F401
    from tools import models, clients, nlp, indexer, phases, verify  # noqa: F401


def test_reuses_root_config_and_llm():
    from config import AppConfig, load_config
    from llm_client import InternS2Client
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert hasattr(InternS2Client(cfg), "chat")
