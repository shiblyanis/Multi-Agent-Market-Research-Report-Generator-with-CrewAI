"""End-to-end run of the REAL CrewAI hierarchical crew against a scripted local LLM server."""
import json

import pytest
from crewai import LLM
from pypdf import PdfReader

from crew import collectors
from crew.orchestrator import run_pipeline
from tests.scripted_llm import ScriptedLLMServer


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(collectors, "search_and_scrape",
                        lambda q: "[1] Synthetic page\nSource: https://example.com/synthetic-market-data\nRevenue table")
    with ScriptedLLMServer() as server:
        manager = LLM(model="gpt-4o", base_url=server.base_url, api_key="test")
        worker = LLM(model="gpt-4o-mini", base_url=server.base_url, api_key="test")
        result = run_pipeline("Tech Sector Q3 Analysis", tmp_path, manager, worker, verbose=False)
        yield result, server, tmp_path


def test_outputs_written(run):
    result, _, out = run
    assert result.pdf_path.name == "market_report.pdf" and result.pdf_path.exists()
    assert result.markdown_path.exists() and (out / "metrics.csv").exists()
    assert (out / "benchmarks.json").exists()


def test_manager_delegated_each_task_to_the_right_agent(run):
    _, server, _ = run
    delegations = [c for c in server.calls if c["role"] == "Crew Manager" and c["action"].startswith("tool_call")]
    assert len(delegations) == 3
    roles_that_worked = [c["role"] for c in server.calls if c["role"] != "Crew Manager" and c["action"] == "final"]
    assert roles_that_worked == ["Market Research Specialist", "Financial Data Analyst", "Technical Reports Writer"]
    worker_tools = [c["action"] for c in server.calls if c["action"].startswith("tool_call:") and c["role"] != "Crew Manager"]
    assert worker_tools == ["tool_call:web_search_and_scrape", "tool_call:save_metrics_csv",
                            "tool_call:calculate_financial_metrics"]


def test_benchmarks_trace_routing_and_timings(run):
    result, _, out = run
    data = json.loads((out / "benchmarks.json").read_text())
    assert data["status"] == "completed" and data["process"] == "hierarchical"
    assert len(data["tasks"]) == 3 and all(t["seconds"] >= 0 for t in data["tasks"])
    workers = {r["agent"] for r in data["agent_runs"]}
    assert {"Market Research Specialist", "Financial Data Analyst", "Technical Reports Writer"} <= workers
    kinds = [e["event"] for e in data["routing_history"]]
    assert kinds.count("delegation") == 3
    tools = [e["detail"] for e in data["routing_history"] if e["event"] == "tool_call"]
    assert any("web_search_and_scrape" in d for d in tools) and any("calculate_financial_metrics" in d for d in tools)


def test_pdf_has_banner_statistics_table_and_page_numbers(run):
    result, _, _ = run
    pages = [p.extract_text() for p in PdfReader(str(result.pdf_path)).pages]
    full = "\n".join(pages)
    assert "Run ID:" in pages[0] and "Manager LLM: gpt-4o" in pages[0]
    assert "Statistical Summary (Pandas)" in full and "revenue_usd_bn" in full and "mean" in full
    assert f"Page 1 of {len(pages)}" in pages[0]
