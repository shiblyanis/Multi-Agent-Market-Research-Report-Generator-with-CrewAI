"""Central configuration. Secrets come only from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPORT_FILENAME = "market_report.pdf"
MARKDOWN_FILENAME = "market_report.md"
METRICS_FILENAME = "metrics.csv"
BENCHMARK_FILENAME = "benchmarks.json"


def manager_model() -> str:
    return os.getenv("MANAGER_MODEL", "gpt-4o")


def worker_model() -> str:
    return os.getenv("WORKER_MODEL", "gpt-4o-mini")


def output_dir() -> Path:
    path = Path(os.getenv("CREW_OUTPUT_DIR", "output"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def search_provider():
    """Return 'tavily', 'serper' or None depending on configured keys."""
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("SERPER_API_KEY"):
        return "serper"
    return None


def missing_configuration() -> list:
    """Human readable list of missing required settings."""
    problems = []
    if not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set")
    if search_provider() is None:
        problems.append("Set TAVILY_API_KEY or SERPER_API_KEY for web search")
    return problems
