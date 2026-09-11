from pathlib import Path

from tools.render_report import _markdown_to_html, _render_submission_html


def test_markdown_table_gets_thead_tbody():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    out = _markdown_to_html(md, Path("."), Path("out.html"), {"artifacts": []})
    compact = "".join(out.split())
    assert "<thead><tr><th>A</th><th>B</th></tr></thead>" in compact
    assert "<tbody><tr><td>1</td><td>2</td></tr></tbody>" in compact


def test_print_css_lets_long_tables_flow_with_row_integrity(tmp_path):
    """Wave7: whole-table break-inside:avoid failed on page-exceeding matrix
    tables (mid-row cuts, no repeated header). The print CSS must let tables
    flow (thead repeats) while keeping rows intact and images whole."""
    html_out = _render_submission_html(
        "T", "## S\n\n| A | B |\n|---|---|\n| 1 | 2 |\n", tmp_path,
        tmp_path / "out.html", {"artifacts": []})
    compact = "".join(html_out.split())
    print_css = compact.split("@mediaprint", 1)[1].split("}}", 1)[0]
    assert "table{break-inside:auto;}" in print_css
    assert "tr{break-inside:avoid;}" in print_css
    assert "img{break-inside:avoid;}" in print_css
    assert "overflow-wrap:break-word" in compact
