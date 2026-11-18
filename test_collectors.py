import pytest

from crew import collectors
from crew.tools import save_metrics_csv, web_search_and_scrape


class FakeResp:
    def __init__(self, html="", status=200, ctype="text/html", json_data=None):
        self._html, self._json, self.headers, self.encoding = html, json_data, {"Content-Type": ctype}, "utf-8"
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise collectors.requests.HTTPError("boom")

    def iter_content(self, n):
        yield self._html.encode()

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(collectors.socket, "getaddrinfo", lambda host, port: [(0, 0, 0, "", ("93.184.216.34", 0))])


def test_is_public_url_blocks_private_and_bad_schemes(monkeypatch):
    assert collectors.is_public_url("https://example.com/x")
    assert not collectors.is_public_url("ftp://example.com")
    assert not collectors.is_public_url("file:///etc/passwd")
    monkeypatch.setattr(collectors.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", ("127.0.0.1", 0))])
    assert not collectors.is_public_url("http://localhost:8000")
    monkeypatch.setattr(collectors.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", ("169.254.169.254", 0))])
    assert not collectors.is_public_url("http://metadata.internal")


def test_scrape_strips_scripts_and_truncates(monkeypatch):
    html = "<html><script>evil()</script><body><nav>menu</nav><p>Revenue grew 12%</p>" + "x " * 5000 + "</body></html>"
    monkeypatch.setattr(collectors.requests, "get", lambda *a, **k: FakeResp(html))
    text = collectors.scrape("https://example.com", max_chars=100)
    assert "Revenue grew 12%" in text and "evil" not in text and "menu" not in text
    assert len(text) <= 100


def test_scrape_returns_empty_on_http_error_and_non_html(monkeypatch):
    monkeypatch.setattr(collectors.requests, "get", lambda *a, **k: FakeResp(status=500))
    assert collectors.scrape("https://example.com") == ""
    monkeypatch.setattr(collectors.requests, "get", lambda *a, **k: FakeResp(ctype="application/pdf"))
    assert collectors.scrape("https://example.com") == ""


def test_search_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    seen = {}

    def fake_post(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return FakeResp(json_data={"results": [{"title": "T", "url": "https://a.com", "content": "snip"}]})

    monkeypatch.setattr(collectors.requests, "post", fake_post)
    res = collectors.search("tech q3")
    assert res == [{"title": "T", "url": "https://a.com", "snippet": "snip"}]
    assert "tavily" in seen["url"] and seen["kw"]["headers"]["Authorization"] == "Bearer k"


def test_search_serper(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "s")
    monkeypatch.setattr(collectors.requests, "post", lambda url, **kw: FakeResp(
        json_data={"organic": [{"title": "S", "link": "https://b.com", "snippet": "sn"}]}))
    assert collectors.search("q")[0]["url"] == "https://b.com"


def test_search_without_keys_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        collectors.search("q")


def test_web_tool_combines_results_and_reports_errors(monkeypatch):
    monkeypatch.setattr(collectors, "search", lambda q: [{"title": "T", "url": "https://a.com", "snippet": "fallback"}])
    monkeypatch.setattr(collectors, "scrape", lambda u: "")
    out = web_search_and_scrape.run(query="x")
    assert "Source: https://a.com" in out and "fallback" in out

    def boom(q):
        raise RuntimeError("down")
    monkeypatch.setattr(collectors, "search", boom)
    assert web_search_and_scrape.run(query="x").startswith("ERROR")


def test_save_metrics_csv_validates_and_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("CREW_OUTPUT_DIR", str(tmp_path))
    ok = save_metrics_csv.run(csv_text="```csv\ncompany,rev\nA,1\nB,2\n```")
    assert ok.startswith("Saved 2 rows") and (tmp_path / "metrics.csv").exists()
    assert save_metrics_csv.run(csv_text="company,rev\nA,1\n").startswith("ERROR")
    assert save_metrics_csv.run(csv_text="company,rev\nA,$1B\nB,$2B\n").startswith("ERROR")
