from tools.nlp.data_cleaner import DataCleaner

def test_complete_abstract_from_md():
    p = {"abstract": "", "md_text": "title\n\nA b s t r a c t\nWe present a method. " * 5}
    out = DataCleaner().complete_abstract(p)
    assert "present a method" in out

def test_keeps_long_abstract():
    long_abs = "x" * 600
    assert DataCleaner().complete_abstract({"abstract": long_abs}) == long_abs

def test_flatten_paragraphs_from_sections():
    parsed = {"abstract": "a", "sections": [{"name":"s","paragraphs":[{"page":1,"index":0,"text":"t"}]}]}
    out = DataCleaner().clean_parsed(parsed)
    assert out["paragraphs"][0]["text"] == "t"

def test_complete_abstract_no_md_text():
    """When there is no md_text/full_text, returns existing abstract unchanged."""
    short_abs = "short abstract"
    p = {"abstract": short_abs, "md_text": "", "full_text": ""}
    out = DataCleaner().complete_abstract(p)
    assert out == short_abs

def test_complete_abstract_no_abstract_marker():
    """When md_text exists but has no abstract marker, returns md[:2000]."""
    md_content = "This is the full text content without any summary marker. " * 20
    p = {"abstract": "", "md_text": md_content}
    out = DataCleaner().complete_abstract(p)
    assert out == md_content[:2000].strip()
    assert len(out) <= 2000

def test_complete_abstract_case_whitespace_tolerant():
    """The regex is case/whitespace tolerant: matches spaced letters like 'A b s t r a c t'."""
    p = {"abstract": "", "md_text": "Title\n\nA b s t r a c t\nThis is the abstract content. " * 10}
    out = DataCleaner().complete_abstract(p)
    assert "abstract content" in out
    # Also test uppercase with spaces
    p2 = {"abstract": "", "md_text": "Title\n\nA  B  S  T  R  A  C  T\nAnother abstract. " * 10}
    out2 = DataCleaner().complete_abstract(p2)
    assert "Another abstract" in out2

def test_clean_parsed_keeps_existing_paragraphs():
    """When paragraphs already present, left as-is (not overwritten/re-flattened)."""
    existing_para = [{"page": 1, "index": 0, "text": "existing"}]
    parsed = {"abstract": "", "paragraphs": existing_para, "sections": [{"name": "s", "paragraphs": [{"page": 2, "index": 0, "text": "from_sections"}]}]}
    out = DataCleaner().clean_parsed(parsed)
    assert out["paragraphs"] == existing_para
    assert out["paragraphs"][0]["text"] == "existing"
    # Ensure it wasn't re-flattened
    assert len(out["paragraphs"]) == 1

def test_clean_parsed_no_sections_with_paragraphs():
    """When sections is missing but paragraphs present, unchanged."""
    existing_para = [{"page": 1, "index": 0, "text": "para"}]
    parsed = {"abstract": "", "paragraphs": existing_para}
    out = DataCleaner().clean_parsed(parsed)
    assert out["paragraphs"] == existing_para
    assert "sections" not in out
