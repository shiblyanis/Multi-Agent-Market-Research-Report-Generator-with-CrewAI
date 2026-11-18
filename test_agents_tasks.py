from pathlib import Path

import pytest
from crewai import LLM

from crew.agents import build_agents
from crew.tasks import build_tasks


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return LLM(model="gpt-4o-mini")


def test_agent_roles_match_brief(llm):
    agents = build_agents(llm)
    assert agents["researcher"].role == "Market Research Specialist"
    assert agents["analyst"].role == "Financial Data Analyst"
    assert agents["writer"].role == "Technical Reports Writer"


def test_workers_do_not_delegate_and_have_expected_tools(llm):
    agents = build_agents(llm)
    assert not any(a.allow_delegation for a in agents.values())
    names = {k: {t.name for t in a.tools} for k, a in agents.items()}
    assert names["researcher"] == {"web_search_and_scrape", "save_metrics_csv"}
    assert names["analyst"] == {"calculate_financial_metrics"}
    assert names["writer"] == set()


def test_tasks_are_unassigned_so_manager_delegates():
    tasks = build_tasks("Tech Sector Q3 Analysis", Path("output/metrics.csv"))
    assert all(t.agent is None for t in tasks.values())
    assert "Tech Sector Q3 Analysis" in tasks["research"].description
    assert tasks["writing"].context == [tasks["research"], tasks["analysis"]]
