import json

import pytest
from crewai import Crew, Process
from pypdf import PdfReader

import main as cli
from crew import orchestrator
from crew.orchestrator import build_crew, clean_markdown, compile_outputs, run_pipeline


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


def test_crew_is_hierarchical_with_manager_and_three_agents(tmp_path):
    crew, tasks = build_crew("Tech Sector Q3 Analysis", tmp_path)
    assert crew.process == Process.hierarchical
    assert crew.manager_llm is not None
    assert [a.role for a in crew.agents] == ["Market Research Specialist", "Financial Data Analyst",
                                             "Technical Reports Writer"]
    assert len(crew.tasks) == 3 and set(tasks) == {"research", "analysis", "writing"}


def test_clean_markdown_strips_outer_fence():
    assert clean_markdown("```markdown\n# T\n\nbody\n```") == "# T\n\nbody"
    assert clean_markdown("# T\n```py\nx\n```") == "# T\n```py\nx\n```"


def test_compile_outputs_writes_md_and_pdf_with_statistics(tmp_path):
    (tmp_path / "metrics.csv").write_text("company,revenue_usd_bn\nA,10\nB,20\nC,30\n")
    md, pdf = compile_outputs("Q", "```markdown\n# Report\n\n## Summary\nText\n```", tmp_path,
                              {"run_id": "r1", "manager_model": "gpt-4o"})
    assert md.name == "market_report.md" and pdf.name == "market_report.pdf"
    assert "Statistical Summary (Pandas)" in md.read_text() and "| mean" in md.read_text()
    text = "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages)
    assert "Statistical Summary" in text and "revenue_usd_bn" in text and "Page 1 of" in text


def test_compile_outputs_without_csv_still_writes_pdf(tmp_path):
    _, pdf = compile_outputs("Q", "# T\n\nBody", tmp_path, {})
    assert pdf.exists()


def test_empty_report_is_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        compile_outputs("Q", "   ", tmp_path, {})


def test_failed_run_still_writes_benchmarks(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(Crew, "kickoff", boom)
    with pytest.raises(RuntimeError):
        run_pipeline("Q", tmp_path)
    data = json.loads((tmp_path / "benchmarks.json").read_text())
    assert data["status"] == "failed" and "LLM unavailable" in data["error"]


def test_stale_metrics_are_removed_before_a_run(tmp_path, monkeypatch):
    (tmp_path / "metrics.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(Crew, "kickoff", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        run_pipeline("Q", tmp_path)
    assert not (tmp_path / "metrics.csv").exists()


def test_cli_requires_configuration(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli.config, "missing_configuration", lambda: ["OPENAI_API_KEY is not set"])
    assert cli.main(["--query", "x"]) == 2


def test_cli_query_validation():
    with pytest.raises(SystemExit):
        cli.parse_args(["--query", "   "])
    assert cli.parse_args(["--query", " Tech Sector Q3 Analysis "]).query == "Tech Sector Q3 Analysis"
