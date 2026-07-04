from tools.phases.phase4_rag_indexer import run
from tools.models.artifacts import ParsedPapers, ParsedPaper, Paragraph
from tools.clients.embedding_client import FakeEmbeddingClient

def test_indexes_abstract_and_paragraphs(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        abstract="abs", paragraphs=[Paragraph(page=1,index=0,text="body")])])
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"))
    hits = vs.query("abs", n=5)
    assert {h.id for h in hits} & {"paper:1_abs","paper:1_p1_0"}

def test_abstract_query_returns_abstract_id(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        abstract="The abstract text here", paragraphs=[Paragraph(page=1,index=0,text="body")])])
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"))
    hits = vs.query("abstract text", n=5)
    assert "paper:1_abs" in {h.id for h in hits}

def test_paragraph_query_returns_paragraph_id(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        abstract="abs", paragraphs=[Paragraph(page=2,index=3,text="specific paragraph content")])])
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"))
    hits = vs.query("specific paragraph content", n=5)
    assert "paper:1_p2_3" in {h.id for h in hits}

def test_extra_texts_are_indexed(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        abstract="abs", paragraphs=[Paragraph(page=1,index=0,text="body")])])
    extra = [{"id": "extra_1", "text": "extra content", "meta": {"source": "test"}}]
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"), extra_texts=extra)
    hits = vs.query("extra content", n=5)
    assert "extra_1" in {h.id for h in hits}

def test_paper_with_no_abstract_only_indexes_paragraphs(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[ParsedPaper(paper_id="paper:1", title="A",
        abstract="", paragraphs=[Paragraph(page=1,index=0,text="body")])])
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"))
    hits = vs.query("body", n=5)
    found_ids = {h.id for h in hits}
    assert "paper:1_abs" not in found_ids
    assert "paper:1_p1_0" in found_ids

def test_empty_parsedpapers_returns_vectorstore(tmp_path):
    pp = ParsedPapers(task_id="t", papers=[])
    vs = run(pp, FakeEmbeddingClient(16), path=str(tmp_path/"v"))
    assert vs is not None
    hits = vs.query("anything", n=5)
    assert len(hits) == 0
