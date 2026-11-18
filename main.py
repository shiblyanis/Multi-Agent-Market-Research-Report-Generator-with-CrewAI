"""CLI entry point:  python main.py --query 'Tech Sector Q3 Analysis'"""
import argparse
import logging
import sys

import crew  # noqa: F401  (sets CrewAI env defaults before crewai is imported)
from crew import config

DEFAULT_QUERY = "Tech Sector Q3 Analysis"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-agent market research and PDF report generator")
    p.add_argument("--query", default=DEFAULT_QUERY, help="Research topic (default: %(default)s)")
    p.add_argument("--output-dir", default=None, help="Directory for outputs (default: $CREW_OUTPUT_DIR or ./output)")
    p.add_argument("--quiet", action="store_true", help="Hide CrewAI's verbose agent output (keep the run log)")
    args = p.parse_args(argv)
    args.query = args.query.strip()
    if not args.query or len(args.query) > 200:
        p.error("--query must be 1-200 characters")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("crew.run")

    problems = config.missing_configuration()
    if problems:
        for msg in problems:
            log.error("Configuration error: %s (see .env.example)", msg)
        return 2

    from crew.orchestrator import run_pipeline  # imported late so config errors fail fast

    out_dir = args.output_dir or config.output_dir()
    try:
        result = run_pipeline(args.query, out_dir, verbose=not args.quiet)
    except Exception as exc:
        log.error("Run failed: %s", exc)
        return 1
    print(f"\nDone. run_id={result.run_id}")
    print(f"  PDF report : {result.pdf_path}")
    print(f"  Markdown   : {result.markdown_path}")
    print(f"  Benchmarks : {result.benchmarks_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
