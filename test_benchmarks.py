import json
from datetime import datetime, timedelta
from types import SimpleNamespace as NS

from crew.benchmarks import BenchmarkTracker

T0 = datetime(2026, 1, 1, 12, 0, 0)


def ev(seconds, **kw):
    return NS(timestamp=T0 + timedelta(seconds=seconds), **kw)


def make(tmp_path):
    return BenchmarkTracker("Tech Q3", "gpt-4o", "gpt-4o-mini", tmp_path / "benchmarks.json")


def test_task_timing_and_routing(tmp_path):
    t = make(tmp_path)
    t.on_task_started(None, ev(0, task_id="1", task_name="research", agent_role="Crew Manager"))
    t.on_tool_started(None, ev(1, agent_role="Crew Manager", tool_name="Delegate work to coworker",
                               tool_args={"coworker": "Market Research Specialist"}))
    t.on_agent_started(None, ev(1.5, agent_id="a", task_id="x", agent_role="Market Research Specialist",
                                task_name="research"))
    t.on_tool_started(None, ev(2, agent_role="Market Research Specialist", tool_name="web_search_and_scrape",
                               tool_args="{'query': 'q'}"))
    t.on_agent_completed(None, ev(6.5, agent_id="a", task_id="x", agent_role="Market Research Specialist",
                                  task_name="research"))
    t.on_task_completed(None, ev(7, task_id="1", task_name="research", agent_role="Crew Manager"))
    data = t.finish()

    assert data["tasks"] == [{"task": "research", "agent": "Crew Manager", "status": "completed", "seconds": 7.0}]
    assert data["agent_runs"][0] == {"agent": "Market Research Specialist", "task": "research", "seconds": 5.0}
    kinds = [e["event"] for e in data["routing_history"]]
    assert kinds == ["task_started", "delegation", "agent_started", "tool_call", "agent_done", "task_done"]
    assert data["routing_history"][1]["t"] == 1.0
    assert data["process"] == "hierarchical" and data["status"] == "completed"


def test_failed_task_is_recorded(tmp_path):
    t = make(tmp_path)
    t.on_task_started(None, ev(0, task_id="1", task_name="writing", agent_role="W"))
    t.on_task_failed(None, ev(2, task_id="1", task_name="writing", agent_role="W"))
    assert t.finish(status="failed", error="boom")["tasks"][0]["status"] == "failed"


def test_benchmarks_json_written_and_valid(tmp_path):
    t = make(tmp_path)
    t.finish(status="failed", error="no key")
    saved = json.loads((tmp_path / "benchmarks.json").read_text())
    assert saved["status"] == "failed" and saved["error"] == "no key"
    assert set(saved) >= {"run_id", "query", "total_seconds", "tasks", "routing_history", "agent_runs"}


def test_attach_and_detach_register_on_real_event_bus(tmp_path):
    from crewai.events import TaskStartedEvent, crewai_event_bus
    t = make(tmp_path)
    with t:
        assert TaskStartedEvent in t._handlers
        assert t._bus is crewai_event_bus
    assert t._handlers == {}


def test_real_event_shapes_underscore_delegation_and_agent_fallback(tmp_path):
    t = make(tmp_path)
    agent = NS(role="Financial Data Analyst")
    task = NS(name=None, description="Analyse the numeric metrics saved at x.csv")
    t.on_task_started(None, ev(0, task_id="1", task_name="analysis", agent_role=None))
    t.on_tool_started(None, ev(1, agent_role="Crew Manager", tool_name="delegate_work_to_coworker",
                               tool_args='{"coworker": "Financial Data Analyst", "task": "t", "context": "c"}'))
    t.on_agent_started(None, ev(1, agent_id="a", task_id="x", agent_role=None, agent=agent, task=task, task_name=None))
    t.on_agent_completed(None, ev(3, agent_id="a", task_id="x", agent_role=None, agent=agent, task=task, task_name=None))
    t.on_task_completed(None, ev(4, task_id="1", task_name="analysis", agent_role=None))
    data = t.finish()
    deleg = [e for e in data["routing_history"] if e["event"] == "delegation"]
    assert len(deleg) == 1 and deleg[0]["detail"].startswith("-> Financial Data Analyst")
    assert data["agent_runs"][0]["agent"] == "Financial Data Analyst"
    assert data["agent_runs"][0]["task"].startswith("Analyse the numeric")
    assert data["tasks"][0]["agent"] == "Crew Manager" and data["tasks"][0]["task"] == "analysis"
