"""Execution benchmarking: per task timings and task routing history -> benchmarks.json."""
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("crew.run")

DELEGATION_TOOLS = ("delegate work to coworker", "ask question to coworker")
MANAGER_ROLE = "Crew Manager"


def _role(event, default=None):
    """Agent role from the event, falling back to event.agent.role (events differ by type)."""
    role = getattr(event, "agent_role", None) or getattr(getattr(event, "agent", None), "role", None)
    return role or default


def _task_label(event) -> str:
    name = getattr(event, "task_name", None)
    if name:
        return name
    task = getattr(event, "task", None)
    if task is not None:
        return getattr(task, "name", None) or _clip(getattr(task, "description", "") or "", 80)
    return ""


def _delegate_target(tool_args) -> str:
    """Pull the coworker name out of delegation tool arguments."""
    args = tool_args
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return ""
    return str(args.get("coworker", "")) if isinstance(args, dict) else ""


def _clip(value, limit=300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class BenchmarkTracker:
    """Listens to the CrewAI event bus and records timings and routing."""

    def __init__(self, query, manager_model, worker_model, path, process="hierarchical"):
        self.run_id = uuid.uuid4().hex[:8]
        self.query, self.process = query, process
        self.manager_model, self.worker_model = manager_model, worker_model
        self.path = Path(path)
        self.tasks, self.agent_runs, self.routing_history = [], [], []
        self._task_starts, self._agent_starts = {}, {}
        self._t0 = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._perf0 = time.perf_counter()
        self._bus = None
        self._handlers = {}

    # ---- event bus wiring -------------------------------------------------
    def attach(self):
        from crewai.events import (AgentExecutionCompletedEvent, AgentExecutionStartedEvent,
                                   TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent,
                                   ToolUsageStartedEvent, crewai_event_bus)
        self._bus = crewai_event_bus
        self._handlers = {
            TaskStartedEvent: self.on_task_started,
            TaskCompletedEvent: self.on_task_completed,
            TaskFailedEvent: self.on_task_failed,
            AgentExecutionStartedEvent: self.on_agent_started,
            AgentExecutionCompletedEvent: self.on_agent_completed,
            ToolUsageStartedEvent: self.on_tool_started,
        }
        for event_type, handler in self._handlers.items():
            self._bus.on(event_type)(handler)
        return self

    def detach(self):
        if self._bus is not None:
            try:
                self._bus.flush()
            except Exception:  # flush must never hide the real result
                pass
            for event_type, handler in self._handlers.items():
                self._bus.off(event_type, handler)
        self._handlers = {}

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    # ---- helpers ----------------------------------------------------------
    def _rel(self, ts) -> float:
        if self._t0 is None:
            self._t0 = ts
        return round((ts - self._t0).total_seconds(), 3)

    def _record(self, ts, kind, agent, detail=""):
        entry = {"t": self._rel(ts), "event": kind, "agent": agent, "detail": detail}
        self.routing_history.append(entry)
        logger.info("[%7.2fs] %-14s %-26s %s", entry["t"], kind.upper(), agent or "-", detail)

    # ---- handlers (signature: source, event) -----------------------------
    def on_task_started(self, source, event):
        with self._lock:
            self._task_starts[event.task_id] = event.timestamp
            self._record(event.timestamp, "task_started", _role(event, MANAGER_ROLE),
                         _clip(_task_label(event), 120))

    def on_task_completed(self, source, event):
        with self._lock:
            start = self._task_starts.pop(event.task_id, event.timestamp)
            seconds = round((event.timestamp - start).total_seconds(), 3)
            label = _task_label(event)
            self.tasks.append({"task": label, "agent": _role(event, MANAGER_ROLE),
                               "status": "completed", "seconds": seconds})
            self._record(event.timestamp, "task_done", _role(event, MANAGER_ROLE),
                         f"{_clip(label, 80)} ({seconds:.2f}s)")

    def on_task_failed(self, source, event):
        with self._lock:
            start = self._task_starts.pop(event.task_id, event.timestamp)
            seconds = round((event.timestamp - start).total_seconds(), 3)
            label = _task_label(event)
            self.tasks.append({"task": label, "agent": _role(event, MANAGER_ROLE),
                               "status": "failed", "seconds": seconds})
            self._record(event.timestamp, "task_failed", _role(event, MANAGER_ROLE), _clip(label))

    def on_agent_started(self, source, event):
        with self._lock:
            self._agent_starts[(getattr(event, "agent_id", None), getattr(event, "task_id", None))] = event.timestamp
            self._record(event.timestamp, "agent_started", _role(event), _clip(_task_label(event), 120))

    def on_agent_completed(self, source, event):
        with self._lock:
            key = (getattr(event, "agent_id", None), getattr(event, "task_id", None))
            start = self._agent_starts.pop(key, event.timestamp)
            seconds = round((event.timestamp - start).total_seconds(), 3)
            self.agent_runs.append({"agent": _role(event), "task": _task_label(event), "seconds": seconds})
            self._record(event.timestamp, "agent_done", _role(event), f"{seconds:.2f}s")

    def on_tool_started(self, source, event):
        name = event.tool_name or ""
        is_delegation = name.lower().replace("_", " ") in DELEGATION_TOOLS
        with self._lock:
            if is_delegation:
                target = _delegate_target(event.tool_args)
                self._record(event.timestamp, "delegation", _role(event, MANAGER_ROLE),
                             f"-> {target or '?'} | {name} {_clip(event.tool_args, 160)}")
            else:
                self._record(event.timestamp, "tool_call", _role(event),
                             f"{name} {_clip(event.tool_args, 200)}")

    # ---- output -------------------------------------------------------------
    def finish(self, status="completed", error=None) -> dict:
        data = {
            "run_id": self.run_id,
            "query": self.query,
            "status": status,
            "error": error,
            "process": self.process,
            "manager_model": self.manager_model,
            "worker_model": self.worker_model,
            "started_at": self._started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_seconds": round(time.perf_counter() - self._perf0, 3),
            "tasks": self.tasks,
            "agent_runs": self.agent_runs,
            "routing_history": self.routing_history,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return data
