"""Generate the files in sample_outputs/ WITHOUT API keys.

Runs the real hierarchical CrewAI crew against a scripted local LLM server
(tests/scripted_llm.py) with SYNTHETIC data. For real research use:  python main.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

os.chdir(ROOT)  # relative output paths keep machine-specific paths out of the samples
import crew  # noqa: E402,F401
from crewai import LLM  # noqa: E402

from crew import collectors  # noqa: E402
from crew.orchestrator import run_pipeline  # noqa: E402
from tests.scripted_llm import ScriptedLLMServer  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    collectors.search_and_scrape = lambda q: (
        "[1] Synthetic market page\nSource: https://example.com/synthetic-market-data\n"
        "Synthetic quarterly revenue and net income table (demo data).")
    with ScriptedLLMServer() as server:
        manager = LLM(model="gpt-4o", base_url=server.base_url, api_key="scripted")
        worker = LLM(model="gpt-4o-mini", base_url=server.base_url, api_key="scripted")
        result = run_pipeline("Tech Sector Q3 Analysis", Path("sample_outputs"), manager, worker, verbose=False)
    # keep machine-specific absolute paths out of the committed sample files
    for f in (result.benchmarks_path, result.markdown_path):
        f.write_text(f.read_text(encoding="utf-8").replace(str(ROOT), "<project>"), encoding="utf-8")
    print(f"Wrote {result.pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
