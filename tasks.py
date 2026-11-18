"""Task definitions. Tasks carry no fixed agent: the manager LLM delegates them dynamically."""
from pathlib import Path

from crewai import Task


def build_tasks(query: str, metrics_csv: Path) -> dict:
    research = Task(
        name="research",
        description=(
            f"Research the topic: '{query}'.\n"
            "1. Search the web and scrape pages to find recent financial metrics for the main "
            "companies in this market (for example revenue, net income, growth percentages, margins).\n"
            "2. Collect at least 4 companies with the SAME numeric columns.\n"
            "3. Save the collected metrics as CSV with the CSV-saving tool. Use plain numbers only; put "
            "units in the column names (e.g. revenue_usd_bn, yoy_growth_pct).\n"
            f"The CSV must end up at: {metrics_csv}"
        ),
        expected_output=(
            "A short summary of findings with source URLs, plus the confirmation message that "
            "the metrics CSV was saved (including its file path)."
        ),
    )
    analysis = Task(
        name="analysis",
        description=(
            f"Analyse the numeric metrics saved at {metrics_csv}. Use the financial metrics tool to "
            "compute the statistical summary and run the consistency checks, then explain what the "
            "numbers show (averages, spread, outliers, any data problems)."
        ),
        expected_output=(
            "The statistical summary as a Markdown table followed by a bullet list of key observations "
            "and any numeric consistency problems."
        ),
        context=[research],
    )
    writing = Task(
        name="writing",
        description=(
            f"Write the final market report for: '{query}'. Use ONLY the facts and numbers from the "
            "research and analysis results. Structure: Executive Summary, Market Overview, Key Metrics "
            "and Statistical Findings, Risks and Data Limitations, Conclusion, Sources. Do not invent "
            "figures. Mention source URLs in the Sources section."
        ),
        expected_output=(
            "The complete report as Markdown only (start with a '# ' title, use '## ' section "
            "headings), with no extra commentary before or after it."
        ),
        context=[research, analysis],
    )
    return {"research": research, "analysis": analysis, "writing": writing}
