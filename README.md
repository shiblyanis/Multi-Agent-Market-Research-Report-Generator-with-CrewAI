# Multi-Agent Market Research & Report Generator (CrewAI)

- **Repository**: [Insert URL]
- **Author**: Shibly Anis

A hierarchical CrewAI team researches a market topic, analyses the collected metrics with Pandas, writes a
report, and compiles it into a typeset A4 PDF with ReportLab. Every run is benchmarked to `benchmarks.json`.

```
python main.py --query "Tech Sector Q3 Analysis"

 Manager LLM (Process.hierarchical)  -- assigns tasks dynamically -->
   |- Market Research Specialist : web_search_and_scrape, save_metrics_csv   -> metrics.csv
   |- Financial Data Analyst     : calculate_financial_metrics (Pandas)      -> statistics table
   '- Technical Reports Writer   : writes Markdown sections
 post-processing: market_report.md + market_report.pdf + benchmarks.json
```

## Installation
Requires Python 3.10 - 3.13.

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in:
   - `OPENAI_API_KEY` (required; used by the manager and worker LLMs)
   - `TAVILY_API_KEY` **or** `SERPER_API_KEY` (at least one; Tavily is used first when both are set)
   - optional: `MANAGER_MODEL` (default `gpt-4o`), `WORKER_MODEL` (default `gpt-4o-mini`), `CREW_OUTPUT_DIR` (default `output`)
4. Run: `python main.py --query "Tech Sector Q3 Analysis"` (options: `--output-dir DIR`, `--quiet`)
5. Tests: `pytest`

Outputs land in `output/`: `market_report.pdf`, `market_report.md`, `metrics.csv`, `benchmarks.json`.

## Verification test cases
| Brief test case | How to verify |
|---|---|
| 1. Hierarchical execution | Run `python main.py --query 'Tech Sector Q3 Analysis'`. The log starts with `Manager LLM initialising: model=... process=hierarchical`, then shows `DELEGATION Crew Manager -> <coworker>` lines, followed by the Researcher's `TOOL_CALL web_search_and_scrape`, the Analyst's `TOOL_CALL calculate_financial_metrics`, and the Writer's `AGENT_STARTED/AGENT_DONE`. |
| 2. Data aggregation & PDF | Open `output/market_report.pdf`: header banner with metadata, styled sections, "Statistical Summary (Pandas)" and "Collected Data" tables, `Page X of Y` footer on every page. |
| 3. Test suite | `pytest` (all tests pass; no network or API keys needed). |

## Tests
| File | Covers |
|---|---|
| `tests/test_financial.py` | **Pandas analysis test**: mean/min/max/std on a mock CSV, markdown output, error handling |
| `tests/test_report.py` | **PDF layout test**: valid A4 PDF, banner, tables, `Page X of Y`, unicode/markup safety |
| `tests/test_collectors.py` | Tavily/Serper search, scraper cleaning, SSRF guard, CSV saver (HTTP mocked) |
| `tests/test_benchmarks.py` | Task timings and routing history incl. real CrewAI event shapes |
| `tests/test_agents_tasks.py`, `tests/test_orchestrator.py` | Roles/tools, hierarchical crew config, outputs, failure handling, CLI |
| `tests/test_e2e_hierarchical.py` | The **real** hierarchical crew run end to end against a scripted local LLM server |

The end-to-end test uses `tests/scripted_llm.py`, a small OpenAI-compatible HTTP server that plays the manager
and worker turns, so CrewAI's real delegation, tool-calling and event machinery is exercised without API keys.
Its data is synthetic.

## Sample outputs
`sample_outputs/` holds a generated Markdown report, PDF, metrics CSV and `benchmarks.json`. They were produced with
the scripted LLM and **synthetic data** (see `sample_outputs/README.md`). Replace them with a live run by copying
the files from `output/` after running with your API keys. Regenerate the synthetic samples with
`python scripts/generate_sample_outputs.py`.

## Design notes
- **Hierarchical process**: `Crew(process=Process.hierarchical, manager_llm=...)`. Tasks have no fixed agent, so the manager decides who does what.
- **Tools** use the `@tool` decorator from `crewai.tools` (the brief's `crewai_tools.tool` import no longer exists in current CrewAI releases). Tools return `ERROR: ...` strings instead of raising, so agents can self-correct.
- **Data hand-off**: the researcher saves a CSV (`save_metrics_csv` validates it); the analyst runs `calculate_financial_metrics` on it; the PDF compiler re-reads the same CSV so the table is computed by Pandas, not typed by an LLM.
- **Benchmarking**: a `BenchmarkTracker` subscribes to the CrewAI event bus and writes per-task and per-agent durations plus the routing history (task starts, delegations, tool calls) to `benchmarks.json`. It is written even when a run fails (`status: failed`).
- **Safety**: scraper only fetches public http(s) URLs (private/loopback/link-local IPs are blocked), limits page size, strips scripts; secrets only come from `.env`; stale `metrics.csv` is deleted before each run.
- CrewAI tracing/telemetry is disabled by default (`CREWAI_TRACING_ENABLED=false`, `OTEL_SDK_DISABLED=true`).

## Limitations
- Report quality and numbers depend on what the search API returns and on the LLM; always verify figures.
- If the researcher fails to save a metrics CSV, the report is still written but without the statistics tables (a warning is logged).
- Live runs call paid APIs (OpenAI, Tavily/Serper).

## Project layout
```
main.py                 CLI entry point
crew/                   config, agents, tasks, tools, collectors, financial, benchmarks, report, orchestrator
tests/                  pytest suite (+ scripted LLM server)
scripts/                sample output generator
sample_outputs/         generated sample report, PDF, CSV, benchmarks
.env.example            required environment variables
```
