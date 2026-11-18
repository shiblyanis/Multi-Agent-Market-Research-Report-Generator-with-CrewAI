"""Builds the hierarchical crew, runs it with benchmarking, and compiles the final outputs."""
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from crewai import LLM, Crew, Process

from crew import config
from crew.agents import build_agents
from crew.benchmarks import BenchmarkTracker
from crew.financial import load_metrics, numeric_summary
from crew.report import build_pdf
from crew.tasks import build_tasks

logger = logging.getLogger("crew.run")


@dataclass
class RunResult:
    markdown_path: Path
    pdf_path: Path
    benchmarks_path: Path
    metrics_path: Path
    run_id: str


def build_crew(query: str, out_dir, manager_llm=None, worker_llm=None, verbose=True):
    """Crew of three agents; the manager LLM assigns tasks (Process.hierarchical)."""
    manager_llm = manager_llm or LLM(model=config.manager_model(), temperature=0.1)
    worker_llm = worker_llm or LLM(model=config.worker_model(), temperature=0.2)
    agents = build_agents(worker_llm)
    tasks = build_tasks(query, Path(out_dir) / config.METRICS_FILENAME)
    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=Process.hierarchical,
        manager_llm=manager_llm,
        verbose=verbose,
    )
    return crew, tasks


def clean_markdown(text: str) -> str:
    """Remove an outer ```markdown fence that LLMs often wrap around documents."""
    text = (text or "").strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*?)\n```$", text, re.S)
    return (m.group(1) if m else text).strip()


def compile_outputs(query: str, markdown: str, out_dir, meta: dict) -> tuple:
    """Write market_report.md and market_report.pdf. Returns (md_path, pdf_path)."""
    out_dir = Path(out_dir)
    markdown = clean_markdown(markdown)
    if not markdown:
        raise RuntimeError("The writer agent returned an empty report")

    summary, data = None, None
    csv_path = out_dir / config.METRICS_FILENAME
    if csv_path.is_file():
        try:
            data = load_metrics(csv_path)
            summary = numeric_summary(data)
        except Exception as exc:
            logger.warning("Could not summarise %s: %s", csv_path, exc)
    else:
        logger.warning("No metrics CSV at %s: PDF will not contain the statistics table", csv_path)

    md_text = markdown
    if summary is not None:
        md_text += f"\n\n## Statistical Summary (Pandas)\n\n{summary.to_markdown()}\n"
    md_path = out_dir / config.MARKDOWN_FILENAME
    md_path.write_text(md_text, encoding="utf-8")

    pdf_path = build_pdf(out_dir / config.REPORT_FILENAME, markdown, summary, data,
                         meta={"query": query, **meta})
    return md_path, pdf_path


def run_pipeline(query: str, out_dir, manager_llm=None, worker_llm=None, verbose=True) -> RunResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CREW_OUTPUT_DIR"] = str(out_dir.resolve())
    (out_dir / config.METRICS_FILENAME).unlink(missing_ok=True)   # never analyse stale data

    crew, tasks = build_crew(query, out_dir, manager_llm, worker_llm, verbose)
    manager_name = getattr(crew.manager_llm, "model", config.manager_model())
    worker_name = config.worker_model() if worker_llm is None else getattr(worker_llm, "model", "custom")
    tracker = BenchmarkTracker(query, manager_name, worker_name, out_dir / config.BENCHMARK_FILENAME)

    logger.info("Manager LLM initialising: model=%s | process=hierarchical | workers=%s",
                manager_name, [a.role for a in crew.agents])
    logger.info("Query: %s | run_id=%s", query, tracker.run_id)

    status, error = "completed", None
    try:
        with tracker:
            result = crew.kickoff()
        writing = tasks["writing"]
        raw = writing.output.raw if writing.output else getattr(result, "raw", str(result))
        md_path, pdf_path = compile_outputs(
            query, raw, out_dir,
            {"run_id": tracker.run_id, "manager_model": manager_name, "process": "hierarchical"})
        logger.info("Report written: %s", pdf_path)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        data = tracker.finish(status, error)
        logger.info("Benchmarks written: %s (total %.1fs, status=%s)",
                    tracker.path, data["total_seconds"], status)

    return RunResult(md_path, pdf_path, tracker.path, out_dir / config.METRICS_FILENAME, tracker.run_id)
