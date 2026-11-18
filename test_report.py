import pandas as pd
import pytest
from pypdf import PdfReader

from crew.report import build_pdf, inline, pdf_safe, split_title

SUMMARY = pd.DataFrame({"revenue": [3, 20.0, 10.0, 10, 15, 20, 25, 30]},
                       index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
DATA = pd.DataFrame({"company": ["Alpha", "Bravo", "Charlie"], "revenue": [10, 20, 30]})

MD = """# Tech Sector Q3 Analysis

## Executive Summary
Revenue grew **12%** year over year, with *strong* cloud demand. See [source](https://example.com/a).

## Key Findings
- Alpha: revenue \u2192 $10B (\u2265 target)
- Bravo: "smart quotes" \u2014 fine
1. First numbered
2. Second numbered

| Company | Revenue |
|---|---|
| Alpha | 10 |
| Bravo | 20 |

### Risks
Plain <b>tags</b> & ampersands must not break layout.
"""


def read(path):
    return PdfReader(str(path))


def test_pdf_is_written_and_valid(tmp_path):
    out = build_pdf(tmp_path / "market_report.pdf", MD, SUMMARY, DATA,
                    meta={"query": "Tech Sector Q3 Analysis", "run_id": "abc123", "manager_model": "gpt-4o"})
    assert out.exists() and out.read_bytes().startswith(b"%PDF")
    reader = read(out)
    assert len(reader.pages) >= 1
    assert abs(float(reader.pages[0].mediabox.width) - 595.27) < 1     # A4 portrait
    assert reader.metadata.title == "Tech Sector Q3 Analysis"


def test_pdf_contains_banner_tables_and_page_numbers(tmp_path):
    long_md = MD + "\n\n" + "\n\n".join(f"## Section {i}\n" + ("Lorem ipsum dolor sit amet. " * 60) for i in range(8))
    out = build_pdf(tmp_path / "r.pdf", long_md, SUMMARY, DATA, meta={"query": "Q", "run_id": "r1", "manager_model": "gpt-4o"})
    reader = read(out)
    pages = [p.extract_text() for p in reader.pages]
    total = len(pages)
    assert total >= 3
    assert "Run ID: r1" in pages[0] and "Manager LLM: gpt-4o" in pages[0]
    assert "Statistical Summary" in "\n".join(pages) and "Collected Data" in "\n".join(pages)
    for i, text in enumerate(pages, 1):
        assert f"Page {i} of {total}" in text
        assert "Verify figures before use" in text                      # footer note


def test_unicode_and_markup_do_not_raise(tmp_path):
    out = build_pdf(tmp_path / "u.pdf", "# T\n\nCJK \u4e2d\u6587 emoji \U0001f680 <script>x</script> & \u2192 ok\n", None, None, meta={"query": "q"})
    text = read(out).pages[0].extract_text()
    assert "ok" in text and "script" in text and "->" in text


def test_wide_summary_is_transposed_and_empty_inputs_ok(tmp_path):
    wide = pd.DataFrame({f"m{i}": [1.0, 2.0] for i in range(9)}, index=["mean", "max"])
    out = build_pdf(tmp_path / "w.pdf", "# T\n\nBody", wide, pd.DataFrame(), meta={"query": "q"})
    assert out.exists()
    out2 = build_pdf(tmp_path / "e.pdf", "", None, None, meta={"query": "q"})
    assert out2.exists()


def test_helpers():
    assert pdf_safe("a \u2192 b \u2265 c") == "a -> b >= c"
    assert "\u4e2d" not in pdf_safe("\u4e2d")
    assert inline("**b** *i* `c`") == "<b>b</b> <i>i</i> <font face='Courier'>c</font>"
    assert inline("a < b & c") == "a &lt; b &amp; c"
    assert split_title("# Hi\nbody", "x") == ("Hi", "body")
    assert split_title("no title", "x") == ("x", "no title")


def test_blockquote_is_rendered_without_literal_marker(tmp_path):
    out = build_pdf(tmp_path / "q.pdf", "# T\n\n> Quoted line one\n> and line two\n\nAfter text", None, None, meta={"query": "q"})
    text = read(out).pages[0].extract_text()
    assert "Quoted line one and line two" in text and "> Quoted" not in text and "After text" in text
