"""A tiny OpenAI-compatible HTTP server that plays scripted manager/worker LLM turns.

It lets the REAL CrewAI hierarchical machinery (manager delegation, tool calls, events)
run end to end with no API keys and no network. The data it returns is SYNTHETIC.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

SYNTHETIC_ROWS = [
    ("Example Cloud Inc", 52.4, 14.8, 11.2, 31.5),
    ("Sample Semiconductor Co", 38.1, 9.2, 8.4, 27.9),
    ("Demo Software Ltd", 24.7, 6.1, 14.9, 29.3),
    ("Placeholder Devices Corp", 61.3, 11.7, 5.6, 22.4),
    ("Mock Networks plc", 18.9, 3.4, 9.7, 18.2),
]
HEADER = "company,revenue_usd_bn,net_income_usd_bn,yoy_growth_pct,operating_margin_pct"
CSV_TEXT = HEADER + "\n" + "\n".join(",".join(str(v) for v in r) for r in SYNTHETIC_ROWS)


def _completion(model, content=None, tool_call=None):
    message = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_call:
        name, args = tool_call
        message["tool_calls"] = [{"id": f"call_{abs(hash((name, json.dumps(args)))) % 10**8}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
        finish = "tool_calls"
    return {"id": "chatcmpl-scripted", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def _text(message) -> str:
    c = message.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def report_markdown(query: str) -> str:
    revenues = [r[1] for r in SYNTHETIC_ROWS]
    avg = sum(revenues) / len(revenues)
    return f"""# {query}

> SAMPLE OUTPUT. Produced by a scripted offline LLM using SYNTHETIC company data to demonstrate the
> pipeline. These are not real market figures. Run `python main.py` with API keys for real research.

## Executive Summary
The five sample companies average **${avg:.1f}B** in quarterly revenue, with year-over-year growth between 5.6% and 14.9%.

## Market Overview
The synthetic dataset covers cloud, semiconductor, software, devices and networking businesses.

## Key Metrics and Statistical Findings
- Highest revenue: Placeholder Devices Corp (${max(revenues):.1f}B)
- Lowest revenue: Mock Networks plc (${min(revenues):.1f}B)
- Fastest growth: Demo Software Ltd (14.9% YoY)

| Company | Revenue (USD bn) | YoY growth (%) |
|---|---|---|
| Example Cloud Inc | 52.4 | 11.2 |
| Demo Software Ltd | 24.7 | 14.9 |
| Mock Networks plc | 18.9 | 9.7 |

## Risks and Data Limitations
- All figures are synthetic and for demonstration only.
- Five data points are too few for statistical significance.

## Conclusion
The pipeline collected data, computed statistics with Pandas and compiled this report.

## Sources
- https://example.com/synthetic-market-data
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            payload = self.server.script.respond(body)
            code = 200
        except Exception as exc:  # surface script bugs loudly in tests
            payload, code = {"error": {"message": f"scripted server error: {exc!r}"}}, 500
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ScriptedLLMServer:
    def __init__(self):
        self.calls = []                 # [{'role':..., 'action':...}]
        self._lock = threading.Lock()
        self._srv = None

    def __enter__(self):
        self._srv = _Server(("127.0.0.1", 0), _Handler)
        self._srv.script = self
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}/v1"

    def _log(self, role, action):
        with self._lock:
            self.calls.append({"role": role, "action": action})

    def respond(self, body: dict) -> dict:
        model = body.get("model", "scripted")
        msgs = body["messages"]
        role = re.match(r"You are ([^.\n]+)\.", _text(msgs[0])).group(1).strip()
        user = next(_text(m) for m in msgs if m["role"] == "user")
        tool_msgs = [m for m in msgs if m["role"] == "tool"]

        def call(name, args):
            self._log(role, f"tool_call:{name}")
            return _completion(model, tool_call=(name, args))

        def final(text):
            self._log(role, "final")
            return _completion(model, content=text)

        if role == "Crew Manager":
            if tool_msgs:
                return final(_text(tool_msgs[-1]))
            if "Research the topic" in user:
                coworker = "Market Research Specialist"
            elif "Analyse the numeric" in user:
                coworker = "Financial Data Analyst"
            elif "Write the final market report" in user:
                coworker = "Technical Reports Writer"
            else:
                raise ValueError(f"manager got an unknown task: {user[:120]}")
            return call("delegate_work_to_coworker",
                        {"task": user.split("\n\nThis is the expected")[0], "context": user, "coworker": coworker})

        if role == "Market Research Specialist":
            query = re.search(r"Research the topic: '([^']+)'", user)
            if len(tool_msgs) == 0:
                return call("web_search_and_scrape", {"query": f"{query.group(1) if query else 'market'} revenue net income"})
            if len(tool_msgs) == 1:
                return call("save_metrics_csv", {"csv_text": CSV_TEXT})
            return final("Found revenue, net income, growth and margin figures for 5 companies "
                         "(source: https://example.com/synthetic-market-data). " + _text(tool_msgs[-1]))

        if role == "Financial Data Analyst":
            if len(tool_msgs) == 0:
                path = re.search(r"(\S+metrics\.csv)", user).group(1).rstrip(".,")
                return call("calculate_financial_metrics", {"csv_path": path})
            return final("Statistical summary:\n\n" + _text(tool_msgs[-1]) +
                         "\n\nKey observations:\n- Revenue spread is wide (std shown above).\n- No data problems found.")

        if role == "Technical Reports Writer":
            query = re.search(r"market report for: '([^']+)'", user)
            return final(report_markdown(query.group(1) if query else "Market Report"))

        raise ValueError(f"unknown role {role}")
