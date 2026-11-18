"""Custom CrewAI tools (declared with the @tool decorator)."""
from crewai.tools import tool

import io
import re

import pandas as pd

from crew import collectors, config
from crew.financial import load_metrics, summary_markdown


@tool("web_search_and_scrape")
def web_search_and_scrape(query: str) -> str:
    """Searches the web (Tavily or SerperDev) for the query and scrapes the raw text of the
    top result pages. Returns numbered source blocks containing title, URL and page text."""
    try:
        return collectors.search_and_scrape(query)
    except Exception as exc:  # network / API problems are reported, not raised
        return f"ERROR: search failed for '{query}': {exc}"


@tool("save_metrics_csv")
def save_metrics_csv(csv_text: str) -> str:
    """Saves collected company metrics as CSV and returns the file path for analysis.
    csv_text must be CSV with a header row, one company/entity per row, and plain
    numeric values (no currency symbols or units inside cells; put units in column names,
    e.g. revenue_usd_bn)."""
    text = re.sub(r"^```[a-zA-Z]*\n|```\s*$", "", csv_text.strip())
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        return f"ERROR: invalid CSV: {exc}"
    if len(df) < 2:
        return "ERROR: need at least 2 data rows to compute statistics."
    if df.select_dtypes(include="number").empty:
        return "ERROR: no numeric columns found; use plain numbers (e.g. 12.5, not '$12.5B')."
    path = config.output_dir() / config.METRICS_FILENAME
    df.to_csv(path, index=False)
    return f"Saved {len(df)} rows x {len(df.columns)} columns to {path}"


@tool("calculate_financial_metrics")
def calculate_financial_metrics(csv_path: str) -> str:
    """Loads corporate metrics from a CSV file and returns summary statistics
    (count, mean, std, min, quartiles, max) plus data consistency checks as Markdown."""
    try:
        return summary_markdown(load_metrics(csv_path))
    except Exception as exc:  # returned to the agent so it can self-correct
        return f"ERROR: could not analyse '{csv_path}': {exc}"
