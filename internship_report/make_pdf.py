# -*- coding: utf-8 -*-
"""Render INTERNSHIP_REPORT.md to rapport_stage.pdf with fpdf2 (DejaVu fonts, no LaTeX needed).

Usage:  python make_pdf.py
Requires: fpdf2, pillow  (pip install fpdf2 pillow)

Arabic is transliterated (the bundled DejaVu font has no Arabic glyphs); the Markdown source
keeps the Arabic script.
"""
import os
import re
import sys

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover
    sys.exit("pip install fpdf2 pillow")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "INTERNSHIP_REPORT.md")
OUT = os.path.join(HERE, "rapport_stage.pdf")
FDIR = "/usr/share/fonts/truetype/dejavu"
F_REG = os.path.join(FDIR, "DejaVuSans.ttf")
F_BOLD = os.path.join(FDIR, "DejaVuSans-Bold.ttf")
F_ITAL = next((os.path.join(FDIR, f) for f in ("DejaVuSans-Oblique.ttf", "DejaVuSans.ttf") if os.path.exists(os.path.join(FDIR, f))), F_REG)
F_MONO = os.path.join(FDIR, "DejaVuSansMono.ttf")

# --- Arabic segments used in the report -> transliteration (diacritics render in DejaVu) ---
TRANSLIT = {
    "العنوان > الباب > الفصل": "al-ʿunwān > al-bāb > al-faṣl",
    "ما هي المرابحة؟": "mā hiya al-murābaḥa?",
    "المرابحة": "al-murābaḥa",
    "الصيغة": "al-ṣīgha",
    "العنوان": "al-ʿunwān",
    "الباب": "al-bāb",
    "الفصل": "al-faṣl",
    "غير كافٍ": "ghayr kāfin",
    "كافٍ": "kāfin",
    "متعارض": "mutaʿāriḍ",
    "محل العقد": "maḥall al-ʿaqd",
    "تمويل": "tamwīl",
    "نافذ": "nāfidh",
}
AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+[\u0600-\u06FF\s\u060C\u061B>·؟]*")


def transliterate(text: str) -> str:
    for ar, lat in sorted(TRANSLIT.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(ar, lat)
    # anything left in Arabic script (glossary column, examples) -> placeholder-free fallback
    return AR_RE.sub(lambda m: TRANSLIT.get(m.group(0).strip(), "…"), text)


class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(True, margin=18)
        self.set_margins(20, 18, 20)
        self.add_font("DejaVu", "", F_REG)
        self.add_font("DejaVu", "B", F_BOLD)
        self.add_font("DejaVu", "I", F_ITAL)
        self.add_font("Mono", "", F_MONO)
        self.set_font("DejaVu", "", 10.5)
        self.chapter_no = 0

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(120, 130, 145)
        self.cell(0, 6, transliterate("Rapport de stage — RAGLab"), align="L")
        self.cell(0, 6, str(self.page_no()), align="R")
        self.ln(8)
        self.set_text_color(0)

    def footer(self):
        pass

    # ---------- primitives ----------
    def h1(self, s):
        self.add_page()
        self.set_font("DejaVu", "B", 19)
        self.set_text_color(20, 32, 46)
        self.multi_cell(0, 10, transliterate(s), align="L")
        self.ln(3)
        self.set_draw_color(200, 210, 225)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(4)

    def h2(self, s):
        if self.get_y() > 40:
            self.ln(3)
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(22, 60, 110)
        self.multi_cell(0, 8, transliterate(s))
        self.ln(1.5)
        self.set_text_color(0)

    def h3(self, s):
        self.ln(1.5)
        self.set_font("DejaVu", "B", 11.5)
        self.multi_cell(0, 7, transliterate(s))
        self.ln(1)

    def para(self, s, indent=0.0):
        self.set_font("DejaVu", "", 10.5)
        self.set_x(self.l_margin + indent)
        self.multi_cell(self.w - self.l_margin - self.r_margin - indent, 5.6, transliterate(s))
        self.ln(1.2)

    def bullet(self, s, level=0):
        self.set_font("DejaVu", "", 10.5)
        mark = "•" if level == 0 else "–"
        x = self.l_margin + 3 + level * 6
        self.set_x(x)
        self.cell(5, 5.6, mark)
        # fpdf2 restarts wrapped lines at the LEFT MARGIN, so indent by moving the margin
        old = self.l_margin
        self.set_left_margin(x + 5)
        self.set_x(x + 5)
        self.multi_cell(0, 5.6, transliterate(s))
        self.set_left_margin(old)
        self.set_x(old)
        self.ln(0.6)

    def code(self, lines):
        self.ln(1)
        self.set_font("Mono", "", 8.2)
        self.set_fill_color(245, 247, 250)
        w = self.w - self.l_margin - self.r_margin
        for ln in lines:
            self.set_x(self.l_margin)
            self.multi_cell(w, 4.2, transliterate(ln) or " ", fill=True)
        self.ln(2.5)
        self.set_font("DejaVu", "", 10.5)

    def table(self, rows):
        if not rows:
            return
        cols = len(rows[0])
        avail = self.w - self.l_margin - self.r_margin
        widths = [avail / cols] * cols
        self.ln(1)
        for i, row in enumerate(rows):
            self.set_font("DejaVu", "B" if i == 0 else "", 8.4 if cols > 2 else 9.4)
            if i == 0:
                self.set_fill_color(236, 241, 248)
            elif i % 2 == 0:
                self.set_fill_color(250, 251, 253)
            else:
                self.set_fill_color(255, 255, 255)
            # measure row height
            heights = []
            for j, cell in enumerate(row):
                txt = transliterate(re.sub(r"\*\*(.+?)\*\*", r"\1", cell))
                txt = txt.replace("`", "")
                heights.append(len(self.multi_cell(widths[j] - 3, 4.4, txt, dry_run=True, output="LINES")) * 4.4)
            h = max(heights) + 2.4
            if self.get_y() + h > self.h - self.b_margin - 4:
                self.add_page()
            x = self.l_margin
            y = self.get_y()
            for j, cell in enumerate(row):
                txt = transliterate(re.sub(r"\*\*(.+?)\*\*", r"\1", cell)).replace("`", "")
                self.set_xy(x, y)
                self.rect(x, y, widths[j], h, style="F")
                self.set_xy(x + 1.5, y + 1.2)
                old_lm = self.l_margin
                self.set_left_margin(x + 1.5)
                self.multi_cell(widths[j] - 3, 4.4, txt)
                self.set_left_margin(old_lm)
                x += widths[j]
            self.set_xy(self.l_margin, y + h)
        self.set_draw_color(210, 216, 224)
        self.ln(2.5)
        self.set_font("DejaVu", "", 10.5)


def inline_clean(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", s)
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)          # markdown links -> label
    return s.strip()


def parse(md: str):
    """Yield (kind, payload) blocks."""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield ("code", buf)
            continue
        if re.match(r"^\|.*\|$", ln):
            rows = []
            while i < len(lines) and re.match(r"^\|.*\|$", lines[i]):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s\-:|]+$", lines[i].replace("|", "").strip()):
                    rows.append(cells)
                i += 1
            yield ("table", rows)
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").strip())
                i += 1
            yield ("quote", inline_clean(" ".join(buf)))
            continue
        if re.match(r"^!\[.*\]\(.*\)$", ln.strip()):
            i += 1
            continue
        if ln.startswith("# "):
            yield ("h1", inline_clean(ln[2:].strip()))
            i += 1
            continue
        if ln.startswith("## "):
            yield ("h2", inline_clean(ln[3:].strip()))
            i += 1
            continue
        if ln.startswith("### "):
            yield ("h3", inline_clean(ln[4:].strip()))
            i += 1
            continue
        if ln.startswith("#### "):
            yield ("h3", inline_clean(ln[5:].strip()))
            i += 1
            continue
        if ln.strip() in ("---", "***"):
            yield ("hr", None)
            i += 1
            continue
        m = re.match(r"^(\s*)[-*]\s+(.*)$", ln) or re.match(r"^(\s*)\d+[.)]\s+(.*)$", ln)
        if m:
            indent = len(m.group(1)) // 2
            buf = [m.group(2).strip()]
            i += 1
            # a Markdown list item may wrap over several indented lines: join them
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                if re.match(r"^\s*([-*]|\d+[.)])\s+", nxt):
                    break
                if nxt.startswith(("#", "|", ">", "```", "---", "![")):
                    break
                if len(nxt) - len(nxt.lstrip()) == 0:   # dedented -> new block
                    break
                buf.append(nxt.strip())
                i += 1
            yield ("bullet", (indent, inline_clean(" ".join(buf))))
            continue
        if not ln.strip():
            i += 1
            continue
        if ln.strip().startswith("*") and ln.strip().endswith("*") and len(ln.strip()) > 2:
            yield ("italic", inline_clean(ln.strip().strip("*")))
            i += 1
            continue
        buf = [ln.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (not nxt.strip() or nxt.startswith(("#", "|", ">", "```", "---", "!["))
                    or re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^\s*\d+[.)]\s+", nxt)):
                break
            buf.append(nxt.strip())
            i += 1
        yield ("para", inline_clean(" ".join(buf)))


def main():
    md = open(SRC, encoding="utf-8").read()
    # drop the "how to replace fields" front-matter table from the PDF (it is editor guidance)
    md = md.split("# Remplacement des champs (Ctrl+H)")[0] + \
         md.split("## Résumé", 1)[1].join(["## Résumé", ""]) if False else md
    pdf = Report()
    pdf.set_title("Rapport de stage — RAGLab")
    pdf.set_author("[[INTERN_NAME]]")
    pdf.add_page()

    # ---------- cover ----------
    pdf.set_font("DejaVu", "B", 13)
    pdf.ln(10)
    pdf.cell(0, 8, "République Tunisienne", align="C")
    pdf.ln(9)
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 7, "École supérieure privée d'ingénierie et de technologie", align="C")
    pdf.ln(18)
    pdf.set_font("DejaVu", "B", 15)
    pdf.multi_cell(0, 9, "Conception et évaluation d'un système de question-réponse\n"
                         "documentaire (RAG) appliqué à des documents bancaires arabes", align="C")
    pdf.ln(14)
    pdf.set_font("DejaVu", "", 11.5)
    for label, value in [("Réalisé par :", "[[INTERN_NAME]]"),
                         ("Entreprise d'accueil :", "Al Baraka Bank"),
                         ("Période :", "Du 1er août au 1er octobre 2026"),
                         ("Encadrante :", "Ahlem BENHADDOUD"),
                         ("Année universitaire :", "2025 – 2026")]:
        pdf.cell(60, 7, label, align="R")
        pdf.set_font("DejaVu", "B", 11.5)
        pdf.cell(0, 7, "  " + value)
        pdf.set_font("DejaVu", "", 11.5)
        pdf.ln(7)
    pdf.ln(10)
    pdf.set_font("DejaVu", "I", 9.5)
    pdf.multi_cell(0, 6, "Version du 5 octobre 2026 — rapport d'immersion en entreprise.\n"
                         "Les figures (schéma d'architecture et chronologie) sont fournies séparément "
                         "(architecture_schema.png, timeline.png).")
    pdf.add_page()

    skip_section = True   # skip the editor-guidance front matter
    for kind, payload in parse(md):
        if kind == "h2" and payload.startswith("Résumé"):
            skip_section = False
        if skip_section:
            continue
        if kind == "hr":
            pdf.ln(2)
        elif kind == "h1":
            pdf.h1(payload)
        elif kind == "h2":
            pdf.h2(payload)
        elif kind == "h3":
            pdf.h3(payload)
        elif kind == "para":
            pdf.para(payload)
        elif kind == "italic":
            pdf.set_font("DejaVu", "I", 9.5)
            pdf.multi_cell(0, 5.4, transliterate(payload))
            pdf.ln(1.5)
            pdf.set_font("DejaVu", "", 10.5)
        elif kind == "quote":
            pdf.set_font("DejaVu", "I", 9.5)
            pdf.set_fill_color(250, 246, 235)
            pdf.multi_cell(0, 5.2, transliterate(payload), fill=True)
            pdf.ln(2)
            pdf.set_font("DejaVu", "", 10.5)
        elif kind == "bullet":
            indent, text = payload
            pdf.bullet(text, min(indent, 2))
        elif kind == "code":
            pdf.code(payload)
        elif kind == "table":
            pdf.table(payload)

    pdf.output(OUT)
    print("written", OUT, os.path.getsize(OUT), "bytes, pages:", pdf.page_no())


if __name__ == "__main__":
    main()
