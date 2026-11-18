"""Compile the writer's Markdown plus Pandas statistics into a typeset A4 PDF (ReportLab)."""
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import (CondPageBreak, HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

NAVY = colors.HexColor("#1f3a5f")
LIGHT = colors.HexColor("#e8edf5")
GREY = colors.HexColor("#6b7280")
FOOTER_NOTE = ("Generated automatically from public web sources and processed with Pandas. "
               "Verify figures before use. Not investment advice.")

_REPLACEMENTS = {"\u2192": "->", "\u2190": "<-", "\u2265": ">=", "\u2264": "<=", "\u2248": "~",
                 "\u2713": "v", "\u2717": "x", "\u2212": "-", "\u00a0": " ", "\u2009": " ",
                 "\u200b": "", "\u2011": "-", "\u2022": "\u2022"}


def pdf_safe(text: str) -> str:
    """Helvetica only has WinAnsi glyphs; map or drop anything else so no black boxes appear."""
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    out = []
    for ch in text:
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            out.append(base or "?")
    return "".join(out)


def _styles() -> dict:
    ss = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10,
                               leading=14.5, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                             textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                             textColor=NAVY, spaceBefore=14, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11,
                             textColor=colors.HexColor("#374151"), spaceBefore=8, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontName="Helvetica", fontSize=8,
                               leading=10),
        "cellh": ParagraphStyle("cellh", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=8,
                                leading=10, textColor=colors.white),
        "quote": ParagraphStyle("quote", parent=ss["BodyText"], fontName="Helvetica-Oblique", fontSize=9.5,
                                leading=13.5, leftIndent=10, borderPadding=(6, 6, 6, 8),
                                backColor=colors.HexColor("#fff7e6"), spaceBefore=4, spaceAfter=10),
        "note": ParagraphStyle("note", parent=ss["BodyText"], fontName="Helvetica-Oblique", fontSize=8,
                               textColor=GREY, leading=11),
    }


def inline(text: str) -> str:
    """Markdown inline syntax -> ReportLab paragraph markup (XML-escaped first)."""
    text = escape(pdf_safe(text))
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def grid_table(rows: list, styles: dict, width: float, first_col_bold=False) -> Table:
    """Header row + data rows rendered as a bordered, zebra-striped grid."""
    ncols = max(len(r) for r in rows)
    rows = [list(r) + [""] * (ncols - len(r)) for r in rows]
    data = [[Paragraph(inline(str(c)), styles["cellh"] if i == 0 else styles["cell"]) for c in r]
            for i, r in enumerate(rows)]
    table = Table(data, colWidths=[width / ncols] * ncols, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3cad6")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    table.setStyle(TableStyle(style))
    return table


def split_title(markdown: str, default: str):
    """Pull a leading '# Title' out of the body so it is not repeated under the banner."""
    lines = markdown.strip().splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip(), "\n".join(lines[1:])
    return default, markdown


def markdown_to_flowables(markdown: str, styles: dict, width: float) -> list:
    flow, para, bullets, ordered, table, quote = [], [], [], [], [], []

    def flush_para():
        if para:
            flow.append(Paragraph(inline(" ".join(para)), styles["body"]))
            para.clear()

    def flush_quote():
        if quote:
            flow.append(Paragraph(inline(" ".join(quote)), styles["quote"]))
            quote.clear()

    def flush_list(items, bullet):
        if items:
            flow.append(ListFlowable([ListItem(Paragraph(inline(i), styles["body"])) for i in items],
                                     bulletType="bullet" if bullet else "1",
                                     start="\u2022" if bullet else 1, leftIndent=14, bulletFontSize=9))
            flow.append(Spacer(1, 4))
            items.clear()

    def flush_table():
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table
                    if not re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", r)]
            if rows:
                flow.append(grid_table(rows, styles, width))
                flow.append(Spacer(1, 8))
            table.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if re.match(r"^\s*>", line):
            flush_para(); flush_list(bullets, True); flush_list(ordered, False); flush_table()
            quote.append(re.sub(r"^\s*>\s?", "", line).strip())
            continue
        flush_quote()
        if line.lstrip().startswith("|"):
            flush_para(); flush_list(bullets, True); flush_list(ordered, False)
            table.append(line)
            continue
        flush_table()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para(); flush_list(bullets, True); flush_list(ordered, False)
            level = min(len(m.group(1)), 3)
            flow.append(CondPageBreak(3.5 * cm))          # never strand a heading at a page bottom
            heading = Paragraph(inline(m.group(2)), styles[f"h{level}"])
            flow.append(heading)
            if level <= 2:
                flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c3cad6"),
                                       spaceAfter=4))
        elif re.match(r"^\s*[-*+]\s+", line):
            flush_para(); flush_list(ordered, False)
            bullets.append(re.sub(r"^\s*[-*+]\s+", "", line))
        elif re.match(r"^\s*\d+[.)]\s+", line):
            flush_para(); flush_list(bullets, True)
            ordered.append(re.sub(r"^\s*\d+[.)]\s+", "", line))
        elif not line.strip():
            flush_para(); flush_list(bullets, True); flush_list(ordered, False)
        elif re.match(r"^\s*(-{3,}|\*{3,})\s*$", line):
            flush_para()
            flow.append(HRFlowable(width="100%", thickness=0.5, color=GREY))
        else:
            flush_list(bullets, True); flush_list(ordered, False)
            para.append(line.strip())
    flush_para(); flush_list(bullets, True); flush_list(ordered, False); flush_table(); flush_quote()
    return flow


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas so the footer can say 'Page X of Y'."""

    footer_note = FOOTER_NOTE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, total):
        w, _ = A4
        self.setStrokeColor(colors.HexColor("#c3cad6"))
        self.line(2 * cm, 1.7 * cm, w - 2 * cm, 1.7 * cm)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(GREY)
        self.drawString(2 * cm, 1.25 * cm, pdf_safe(self.footer_note))
        self.setFont("Helvetica-Bold", 8.5)
        self.drawRightString(w - 2 * cm, 1.25 * cm, f"Page {self._pageNumber} of {total}")


def _banner(title, meta):
    def first_page(c, doc):
        w, h = A4
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, h - 3.6 * cm, w, 3.6 * cm, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#f59e0b"))
        c.rect(0, h - 3.75 * cm, w, 0.15 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 19)
        c.drawString(2 * cm, h - 1.8 * cm, pdf_safe(title)[:70])
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, h - 2.55 * cm, pdf_safe(
            f"Query: {meta.get('query', '-')}   |   Generated: {meta.get('generated_at', '-')}"))
        c.drawString(2 * cm, h - 3.1 * cm, pdf_safe(
            f"Run ID: {meta.get('run_id', '-')}   |   Manager LLM: {meta.get('manager_model', '-')}   |   "
            f"Process: {meta.get('process', 'hierarchical')}"))
        c.restoreState()

    def later_page(c, doc):
        w, h = A4
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, h - 1.2 * cm, w, 1.2 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(2 * cm, h - 0.8 * cm, pdf_safe(title)[:90])
        c.restoreState()

    return first_page, later_page


def summary_for_display(summary: pd.DataFrame, max_cols: int = 6) -> pd.DataFrame:
    """describe() has stats as rows; transpose when there are many metric columns."""
    return summary.T if summary.shape[1] > max_cols else summary


def build_pdf(path, markdown: str, summary: pd.DataFrame = None, data: pd.DataFrame = None,
              meta: dict = None, title: str = None) -> Path:
    """Write the report to `path` and return it. Raises on layout problems."""
    meta = dict(meta or {})
    meta.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    title, body = split_title(markdown, title or f"Market Report: {meta.get('query', 'Analysis')}")
    styles = _styles()
    width = A4[0] - 4 * cm

    story = [Spacer(1, 2.6 * cm)] + markdown_to_flowables(body, styles, width)

    if summary is not None and not summary.empty:
        shown = summary_for_display(summary)
        rows = [[""] + [str(c) for c in shown.columns]]
        rows += [[str(idx)] + [(f"{v:,.0f}" if str(idx) == "count" else f"{v:,.2f}")
                               if isinstance(v, (int, float)) else str(v) for v in r]
                 for idx, r in zip(shown.index, shown.values.tolist())]
        story += [CondPageBreak(6 * cm), Paragraph("Statistical Summary (Pandas)", styles["h2"]),
                  HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c3cad6"), spaceAfter=6),
                  grid_table(rows, styles, width), Spacer(1, 8)]
    if data is not None and not data.empty:
        shown = data.iloc[:15, :8]
        rows = [[str(c) for c in shown.columns]] + shown.astype(object).where(shown.notna(), "").astype(str).values.tolist()
        story += [CondPageBreak(6 * cm), Paragraph("Collected Data", styles["h2"]),
                  HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c3cad6"), spaceAfter=6),
                  grid_table(rows, styles, width), Spacer(1, 4),
                  Paragraph(f"Showing {len(shown)} of {len(data)} rows.", styles["note"])]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2.2 * cm, title=pdf_safe(title),
                            author="Market Research Crew")
    first, later = _banner(title, meta)
    doc.build(story, onFirstPage=first, onLaterPages=later, canvasmaker=NumberedCanvas)
    return path
