"""The three specialised worker agents. Delegation is done by the crew manager."""
from crewai import Agent

from crew.tools import calculate_financial_metrics, save_metrics_csv, web_search_and_scrape


def build_agents(llm) -> dict:
    researcher = Agent(
        role="Market Research Specialist",
        goal=("Query search networks, locate financial data and scrape company metrics "
              "for the requested market topic."),
        backstory=("You are a meticulous market researcher. You only report figures that appear in "
                   "the sources you scraped, and you always keep the source URL next to each number."),
        tools=[web_search_and_scrape, save_metrics_csv],
        llm=llm,
        allow_delegation=False,
        max_iter=12,
        verbose=True,
    )
    analyst = Agent(
        role="Financial Data Analyst",
        goal=("Parse data matrices using Pandas, compute averages and statistics, and verify "
              "numeric consistency."),
        backstory=("You are a quantitative analyst. You never estimate numbers yourself; you run the "
                   "financial metrics tool on the saved CSV and interpret its output."),
        tools=[calculate_financial_metrics],
        llm=llm,
        allow_delegation=False,
        max_iter=8,
        verbose=True,
    )
    writer = Agent(
        role="Technical Reports Writer",
        goal="Synthesize research and analysis into clear, well structured report sections.",
        backstory=("You write concise, factual business reports in Markdown. You use only the figures "
                   "provided by your colleagues and never invent data."),
        tools=[],
        llm=llm,
        allow_delegation=False,
        max_iter=6,
        verbose=True,
    )
    return {"researcher": researcher, "analyst": analyst, "writer": writer}
