"""Build a Vietnamese PDF report from report/bao_cao_NLP34.md."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "bao_cao_NLP34.md"
OUTPUT = ROOT / "report" / "bao_cao_NLP34.pdf"


def inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Arial">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"https?://[^\s]+", lambda m: f'<link href="{m.group(0)}" color="#087E8B">{m.group(0).split("/")[2]}</link>', text)
    return text


def draw_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#D8E3EA"))
    canvas.line(18 * mm, 15 * mm, w - 18 * mm, 15 * mm)
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(18 * mm, 10 * mm, "NLP-34 | ABSA UIT-ViSFD")
    canvas.drawRightString(w - 18 * mm, 10 * mm, f"Trang {doc.page}")
    canvas.restoreState()


def build():
    pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

    ss = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("TitleVN", parent=ss["Title"], fontName="Arial-Bold", fontSize=20, leading=25, textColor=colors.HexColor("#123047"), alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1VN", parent=ss["Heading1"], fontName="Arial-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#087E8B"), spaceBefore=9, spaceAfter=7, keepWithNext=True),
        "h2": ParagraphStyle("H2VN", parent=ss["Heading2"], fontName="Arial-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#123047"), spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("BodyVN", parent=ss["BodyText"], fontName="Arial", fontSize=9.2, leading=13, textColor=colors.HexColor("#22313A"), alignment=TA_LEFT, spaceAfter=6),
        "bullet": ParagraphStyle("BulletVN", parent=ss["BodyText"], fontName="Arial", fontSize=9.2, leading=13, leftIndent=12, firstLineIndent=-9, spaceAfter=4),
        "small": ParagraphStyle("SmallVN", parent=ss["BodyText"], fontName="Arial", fontSize=7.4, leading=9.4, textColor=colors.HexColor("#24343E")),
        "meta": ParagraphStyle("MetaVN", parent=ss["BodyText"], fontName="Arial", fontSize=9.5, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#52616B"), spaceAfter=4),
        "quote": ParagraphStyle("QuoteVN", parent=ss["BodyText"], fontName="Arial-Italic", fontSize=9, leading=13, leftIndent=10, rightIndent=10, borderColor=colors.HexColor("#087E8B"), borderWidth=0, textColor=colors.HexColor("#425563"), spaceAfter=7),
    }

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    if lines and lines[0].startswith("# "):
        story.append(Spacer(1, 13 * mm))
        story.append(Paragraph(inline(lines[0][2:]), styles["title"]))
        story.append(HRFlowable(width="40%", thickness=2, color=colors.HexColor("#F4A261"), hAlign="CENTER", spaceBefore=0, spaceAfter=10))
        i = 1
    else:
        i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("**") and line.endswith("  "):
            story.append(Paragraph(inline(line.strip()), styles["meta"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["h2"]))
            i += 1
            continue
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    header_style = ParagraphStyle("TableHeadVN", parent=styles["small"], fontName="Arial-Bold", textColor=colors.white)
                    cell_style = header_style if not rows else styles["small"]
                    rows.append([Paragraph(inline(c), cell_style) for c in cells])
                i += 1
            if rows:
                n = len(rows[0])
                avail = A4[0] - 36 * mm
                widths = ([22, 18, 21, 20, 35, 19, 19, 20] if n == 8 else [avail / n] * n)
                widths = [w * mm for w in widths] if n == 8 else widths
                table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F7F8")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D7DD")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.extend([Spacer(1, 3), table, Spacer(1, 7)])
            continue
        if line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            wrapped = "\n".join("\n".join(textwrap.wrap(row, width=82, subsequent_indent="  ")) for row in code)
            story.append(Preformatted(wrapped, ParagraphStyle("CodeVN", fontName="Arial", fontSize=7.6, leading=10, leftIndent=8, rightIndent=8, backColor=colors.HexColor("#F3F7F8"), borderPadding=6, spaceAfter=8)))
            i += 1
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline(line[2:]), styles["quote"]))
            i += 1
            continue
        if line.startswith("- ") or re.match(r"\d+\. ", line):
            text = re.sub(r"^(?:- |\d+\. )", "", line)
            story.append(Paragraph("• " + inline(text), styles["bullet"]))
            i += 1
            continue
        if line == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D8E3EA"), spaceBefore=3, spaceAfter=6))
            i += 1
            continue
        story.append(Paragraph(inline(line), styles["body"]))
        i += 1

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=21 * mm, title="Báo cáo NLP-34 ABSA", author="Nhóm NLP-34")
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(OUTPUT)


if __name__ == "__main__":
    build()

