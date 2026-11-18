import pandas as pd
import pytest

from crew.financial import consistency_notes, load_metrics, numeric_summary, summary_markdown
from crew.tools import calculate_financial_metrics

CSV = """company,revenue_bn,net_income_bn
Alpha,10,2
Bravo,20,4
Charlie,30,6
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text(CSV)
    return p


def test_numeric_summary_values(csv_path):
    s = numeric_summary(load_metrics(csv_path))
    assert s.loc["mean", "revenue_bn"] == 20.0
    assert s.loc["min", "revenue_bn"] == 10.0
    assert s.loc["max", "revenue_bn"] == 30.0
    assert s.loc["std", "revenue_bn"] == 10.0          # sample std of 10,20,30
    assert s.loc["mean", "net_income_bn"] == 4.0


def test_tool_returns_markdown_table(csv_path):
    out = calculate_financial_metrics.run(csv_path=str(csv_path))
    assert "| mean" in out and "revenue_bn" in out
    assert "20" in out and "No consistency issues" in out


def test_tool_reports_errors_instead_of_raising(tmp_path):
    out = calculate_financial_metrics.run(csv_path=str(tmp_path / "missing.csv"))
    assert out.startswith("ERROR")


def test_rejects_non_csv(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_metrics(p)


def test_consistency_flags_missing_text_numbers_and_duplicates():
    df = pd.DataFrame({"name": ["a", "a", "b"], "rev": ["$1.2B", "$1.2B", "$3B"], "x": [1.0, 1.0, None]})
    notes = " ".join(consistency_notes(df))
    assert "missing" in notes and "looks numeric" in notes and "duplicated" in notes


def test_no_numeric_columns_raises():
    with pytest.raises(ValueError):
        numeric_summary(pd.DataFrame({"a": ["x", "y"]}))


def test_summary_markdown_contains_sections(csv_path):
    md = summary_markdown(load_metrics(csv_path))
    assert "Summary statistics" in md and "Data consistency checks" in md
