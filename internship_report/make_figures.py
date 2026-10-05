# -*- coding: utf-8 -*-
"""Generate every figure of the internship report folder.

Outputs (written next to this script):
  architecture_schema.svg   — vector architecture schema (both generations + LAST UPDATE)
  architecture_schema.png   — the same schema as a raster image
  timeline.svg / .png       — development timeline (369 commits, 19 working days)

Requirements:
  * architecture_schema.png  -> pillow
  * after that, no other dependency; the SVG files are plain text

Usage:  python make_figures.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# SVG architecture schema
# ==========================================================================
# -*- coding: utf-8 -*-
"""Self-contained SVG architecture schema for RAGLab (no external assets, no JS)."""
W, H = 1500, 1390
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

C = {
    "bg": "#ffffff", "panel": "#f7f9fc", "line": "#c8d0dc", "text": "#16202c", "muted": "#5c6775",
    "gen1_fill": "#e8f0ff", "gen1_str": "#7ba4f0", "gen1_head": "#cfe0ff",
    "pivot_fill": "#fff6e6", "pivot_str": "#dfa23c", "pivot_head": "#ffe9c4",
    "gen2_fill": "#e9f8f2", "gen2_str": "#4fae8d", "gen2_head": "#cdefe2",
    "last_fill": "#ffe9f2", "last_str": "#d6336c", "chip": "#ffffff",
}
parts = []
def add(s): parts.append(s)
def rect(x, y, w, h, fill, stroke, rx=10, sw=1.4):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def text(x, y, s, size=13, fill=None, weight="normal", anchor="start", style="normal", opacity=1.0):
    fill = fill or C["text"]
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-style="{style}" opacity="{opacity}">{s}</text>')
def chip(x, y, w, h, title, lines, stroke=None, title_size=13, body_size=11):
    rect(x, y, w, h, C["chip"], stroke or C["line"], rx=8, sw=1.2)
    text(x + 10, y + 19, title, size=title_size, weight="bold")
    yy = y + 35
    for ln in lines:
        text(x + 10, yy, ln, size=body_size, fill=C["muted"]); yy += 14.5
def arrow(x1, y1, x2, y2, color=None, sw=2.0):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color or C["muted"]}" '
        f'stroke-width="{sw}" marker-end="url(#arrow)"/>')
def vdown(x, y1, y2, color=None, sw=2.2):
    add(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color or C["muted"]}" stroke-width="{sw}" marker-end="url(#arrow)"/>')
def section(x, y, w, h, title, sub, fill, stroke, head, band=40):
    rect(x, y, w, h, fill, stroke, rx=14, sw=1.8)
    add(f'<path d="M{x+14} {y} h{w-28} a14 14 0 0 1 14 14 v{band-14} h-{w} v-{band-14} a14 14 0 0 1 14 -14 z" fill="{head}"/>')
    text(x + 22, y + 27, title, size=17, weight="bold")
    text(x + w - 22, y + 27, sub, size=12.5, fill=C["muted"], anchor="end")

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add(f'''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="{C['muted']}"/></marker></defs>''')
rect(0, 0, W, H, C["bg"], C["bg"], rx=0)

# ---------------- header ----------------
text(40, 50, "RAGLab — Architecture Schema", size=33, weight="bold")
text(40, 80, "One project, two generations, one pivot — and one current revision. Reconstructed from 387 commits (04 Sept → 05 Oct 2026), the in-repo audit trail, and the code at HEAD.",
     size=14, fill=C["muted"])
text(40, 102, "Legend:", size=12.5, weight="bold", fill=C["muted"])
lx = 100
for label, f, s in [("Generation 1 — flat lab (superseded)", C["gen1_fill"], C["gen1_str"]),
                    ("Pivot / re-foundation", C["pivot_fill"], C["pivot_str"]),
                    ("Generation 2 — governed pipeline", C["gen2_fill"], C["gen2_str"]),
                    ("CURRENT — latest revision", C["last_fill"], C["last_str"])]:
    rect(lx, 91, 18, 15, f, s, rx=4, sw=1.2)
    text(lx + 25, 103, label, size=12, fill=C["muted"]); lx += 30 + 6.3 * len(label)

# ---------------- generation 1 ----------------
section(40, 130, 1420, 250, "GENERATION 1 — 04 → 23 Sept 2026 · the flat retrieval lab (superseded)",
        "measure chunks, models and fusion strategies", C["gen1_fill"], C["gen1_str"], C["gen1_head"])
g1 = [
    ("1 · Corpus", ["fictional sample sheets", "Al Baraka web-compiled round", "(both later deleted)"]),
    ("2 · Chunk", ["size mode 220/40 default", "token-window + overlap", "reviewed chunk maps"]),
    ("3 · Embed", ["multi-provider embeddings:", "Gemini · Jina · HuggingFace", "NVIDIA NeMo-Retriever"]),
    ("4 · Store", ["local Chroma, cosine", "chunk fingerprint v4", "stale index ⇒ refuse"]),
    ("5 · Retrieve", ["vector / rrf / blend fusion", "BM25 k=60 · blend λ=0.7", "fusion tie-break policies"]),
    ("6 · Query variants", ["machine translation ON", "(gemini-3.5-flash-lite)", "best-variant merge"]),
    ("7 · Answer", ["multi-provider + fallback", "grounded-v1 JSON claims", "citation gate: quote+numeric"]),
]
bw = 183; gap = 16.5; bx = 58
for title, lines in g1:
    chip(bx, 195, bw, 100, title, lines, stroke=C["gen1_str"], title_size=12.5, body_size=10.7)
    if bx + bw + 25 < 1460:
        arrow(bx + bw + 2, 245, bx + bw + gap - 3, 245, color=C["gen1_str"], sw=1.8)
    bx += bw + gap
text(1460, 322, "Model-agnostic by design: providers, translations and retrievers were still being compared — nothing was pinned.",
     size=11.5, fill=C["muted"], anchor="end", style="italic")

# ---------------- pivot ----------------
section(40, 410, 1420, 175, "THE PIVOT — 24 → 28 Sept 2026 · from a lab that measures to a system that decides",
        "owner direction + external target-state report", C["pivot_fill"], C["pivot_str"], C["pivot_head"])
piv = [
    ("Target-state report", ["external (AR) design for a governed", "banking RAG decision system: intent ·", "evidence · sufficiency · governance"]),
    ("Reverse-engineering", ["RAGLAB_SPEC.md — every claim cited", "RAGLAB_GAP_ANALYSIS.md — current vs", "target; “the answer model never touches the query”"]),
    ("Re-foundation", ["corpus = docs/ ONLY; web round reverted", "step-and-plan protocol: a published plan", "per step, owner gate before any number"]),
    ("Corpus repair", ["4 docs, LLM-only linguistic repair:", "215 + 35 + 23 + 21 entries, machine-", "checked; adopted codices → 339 chunks"]),
]
bw2 = 321; bx = 58
for title, lines in piv:
    chip(bx, 470, bw2, 96, title, lines, stroke=C["pivot_str"], title_size=13.5, body_size=10.8)
    if bx + bw2 + 40 < 1460:
        arrow(bx + bw2 + 2, 518, bx + bw2 + 28, 518, color=C["pivot_str"], sw=1.8)
    bx += bw2 + 32

# ---------------- generations arrow ----------------
vdown(750, 380, 408, color=C["muted"], sw=2.4)
vdown(750, 585, 628, color=C["muted"], sw=2.4)

# ---------------- generation 2 layers ----------------
section(40, 630, 1090, 610, "GENERATION 2 — 24 Sept → 05 Oct 2026 · the governed, layered pipeline",
        "structure → knowledge → understanding → delivery", C["gen2_fill"], C["gen2_str"], C["gen2_head"])
layers = [
    ("LAYER 1 · Corpus & structure", "loader → restructure → fingerprint-guarded store",
     [("loader.py", ["PDF (visual order) / DOCX / MD", "Arabic normalisation built in"]),
      ("restructure.py", ["3-step structural chunking (220/40)", "adopted codices · RTL zones"]),
      ("docstore.py", ["content-hash versions; push/delete", "stale surfaced, never silent"]),
      ("store.py", ["chunk_fp + embedding_fp guards", "mixed or stale ⇒ refuse to serve"])]),
    ("LAYER 2 · Knowledge — Phase 4 (01 Oct)", "deterministic, declared data — read-only over loi 2016-48",
     [("units.py · 198/198", ["stable ids loi-2016-48:artNNN", "governed functional types"]),
      ("relations.py", ["77 internal reference edges", "+ grounding: Circulaire → art011"]),
      ("legal_numbers.py", ["53 numeric records + corrections", "→ GET /numbers (1st additive route)"]),
      ("governance.py", ["6 axes on chunk metadata", "authority · effectivity · audience"])]),
    ("LAYER 3 · Understanding — Phases 5 + 8 + 9", "deterministic first; one bounded model call last; fail-closed",
     [("intent.py", ["declared rules, first match wins", "compound split · ambiguity note"]),
      ("evidence_plan → sufficiency", ["requirements from intent", "4 explicit sufficiency states"]),
      ("decompose → guided rounds", ["micro-questions per request", "bounded guided retrieval rounds"]),
      ("interrogate.py + topic_map", ["ONE bounded model call", "corpus topics → rephrasing"]),
      ("relational_expansion (OFF)", ["intent-gated tail slots", "measured NEUTRAL → stays off"])]),
    ("LAYER 4 · Delivery & audit — Phase 6", "the answer is a contract: cited and checked, or honestly refused",
     [("answer.py — citation gate", ["verbatim quotes (normalised)", "every number in its quotes", "unit_id on law chunks"]),
      ("service.py · 1.5.0", ["27 HTTP routes · additive-only freeze", "token · 409 seals · error envelope"]),
      ("scrub.py + audit.py", ["post-gate PII scrub", "JSONL trail → GET /audit"]),
      ("REFUSALS", ["reason + referral naming the gap", "never free text, never a guess"])]),
]
ly = 690
for header, sub, chips, in [(l[0], l[1], l[2]) for l in layers]:
    n = len(chips); lh = 120 if n == 4 else 136
    rect(58, ly, 1054, lh, "#ffffff", C["gen2_str"], rx=10, sw=1.2)
    add(f'<path d="M58 {ly} h1054 v26 h-1054 z" fill="{C["gen2_head"]}" opacity="0.8"/>')
    text(70, ly + 18, header, size=13.5, weight="bold")
    text(1112 - 12, ly + 18, sub, size=10.8, fill=C["muted"], anchor="end")
    cw = (1054 - (n + 1) * 12) / n; cx = 70
    for title, lines in chips:
        chip(cx, ly + 34, cw, lh - 46, title, lines, title_size=11.5, body_size=10.4)
        cx += cw + 12
    if ly > 690:
        add(f'<line x1="585" y1="{ly-11}" x2="585" y2="{ly-2}" stroke="{C["gen2_str"]}" stroke-width="2"/>')
    ly += lh + 10

# ---------------- last update column ----------------
lx0, lw = 1148, 312
rect(lx0, 630, lw, 610, C["last_fill"], C["last_str"], rx=14, sw=2.6)
add(f'<path d="M{lx0+14} 630 h{lw-28} a14 14 0 0 1 14 14 v48 h-{lw} v-48 a14 14 0 0 1 14 -14 z" fill="{C["last_str"]}"/>')
text(lx0 + 20, 655, "LAST UPDATE", size=19, weight="bold", fill="#ffffff")
text(lx0 + 20, 682, "05 Oct 2026 · service 1.5.0 · CI green", size=11.5, fill="#ffffff")
las = [
    ("Phase 9 — question ↔ retrieval", ["262 source-grounded topic nodes;", "exact-match ids replace loose lists"]),
    ("Cross-script arm activated", ["2/2 answered with the arm,", "2/2 refused without it"]),
    ("Guided round wired into /answer", ["it was dead on the deployed path;", "found by a live run, then fixed"]),
    ("Root cause TM03 fixed", ["“Quelle” matched inside “Quelles”;", "the French question answers one-shot"]),
    ("Duplicate re-push blocked", ["POST /documents → 409 by SHA-256,", "not filename (859 dup chunks gone)"]),
    ("Re-aimed answers disclosed", ["8 colloquial→corpus rows + scope_note:", "a re-aimed answer cannot look ordinary"]),
    ("Still open (Phase 9 stages 3–8)", ["path / level / parent_id, topical", "arms, facets — none of it default"]),
]
yy = 706
for title, lines in las:
    h = 30 + 15 * len(lines) + 8
    rect(lx0 + 14, yy, lw - 28, h, "#ffffff", C["last_str"], rx=8, sw=1.1)
    text(lx0 + 24, yy + 17, "★ " + title, size=11, weight="bold", fill=C["last_str"])
    y2 = yy + 33.5
    for ln in lines:
        text(lx0 + 24, y2, ln, size=10, fill=C["muted"]); y2 += 15
    yy += h + 7

# ---------------- runtime spine ----------------
rect(40, 1262, 1420, 70, C["panel"], C["line"], rx=12, sw=1.4)
text(58, 1288, "PINNED RUNTIME SPINE — unchanged across both generations; the policy raises on any substitution", size=13, weight="bold")
sx = 58
for label in ["Embeddings — nvidia/nemotron-3-embed-1b · 2048-d · retry + bisect",
              "Index — local ChromaDB cosine · one collection per space × mode",
              "Answers — xKiro qwen3.8-max:free · grounded-v1 JSON · price check"]:
    rect(sx, 1298, 440, 24, "#ffffff", C["line"], rx=7, sw=1.1)
    text(sx + 10, 1314, label, size=10.5, fill=C["muted"]); sx += 456
text(1460, 1362, "no query translation · no provider fallback · refusal over guessing", size=11, fill=C["muted"], anchor="end", style="italic")
add('</svg>')
open('/home/user/RAGLab/internship_report/architecture_schema.svg', 'w', encoding='utf-8').write("\n".join(parts))
print("written")


# ==========================================================================
# PNG architecture schema
# ==========================================================================
# -*- coding: utf-8 -*-
"""Raster backend (PIL) for the RAGLab architecture schema — same layout as the SVG."""
from PIL import Image, ImageDraw, ImageFont

W, H, S = 1500, 1390, 2           # logical size, scale factor
DJ  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DJB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fc = {}
def font(size, bold=False):
    key = (size, bold)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(DJB if bold else DJ, int(size * S))
    return _fc[key]

AR = {"المرابحة": "al-murabaha", "كافٍ": "kafin", "غير كافٍ": "ghayr kafin"}
def clean(t):
    for k, v in AR.items(): t = t.replace(k, v)
    return t

img = Image.new("RGB", (W * S, H * S), "#ffffff")
d = ImageDraw.Draw(img)
C = {"bg":"#ffffff","panel":"#f7f9fc","line":"#c8d0dc","text":"#16202c","muted":"#5c6775",
     "gen1_fill":"#e8f0ff","gen1_str":"#7ba4f0","gen1_head":"#cfe0ff",
     "pivot_fill":"#fff6e6","pivot_str":"#dfa23c","pivot_head":"#ffe9c4",
     "gen2_fill":"#e9f8f2","gen2_str":"#4fae8d","gen2_head":"#cdefe2",
     "last_fill":"#ffe9f2","last_str":"#d6336c","chip":"#ffffff"}

def rect(x, y, w, h, fill, stroke, rx=10, sw=1.4):
    d.rounded_rectangle([x*S, y*S, (x+w)*S, (y+h)*S], radius=rx*S, fill=fill,
                        outline=stroke, width=max(1, int(sw*S)))
def text(x, y, s, size=13, fill=None, bold=False, anchor="la"):
    d.text((x*S, y*S), clean(s), font=font(size, bold), fill=fill or C["text"], anchor=anchor)
def chip(x, y, w, h, title, lines, stroke=None, title_size=13, body_size=11):
    rect(x, y, w, h, C["chip"], stroke or C["line"], rx=8, sw=1.2)
    text(x+10, y+19, title, size=title_size, bold=True)
    yy = y + 35
    for ln in lines:
        text(x+10, yy, ln, size=body_size, fill=C["muted"]); yy += 14.5
def arrow(x1, y1, x2, y2, color, sw=2.0):
    col = color or C["muted"]
    d.line([x1*S, y1*S, x2*S, y2*S], fill=col, width=max(1, int(sw*S)))
    import math
    ang = math.atan2(y2-y1, x2-x1); L = 9
    for sgn in (+1, -1):
        a = ang + sgn*math.radians(26)
        d.line([x2*S, y2*S, (x2-L*math.cos(a))*S, (y2-L*math.sin(a))*S], fill=col, width=max(1, int(sw*S)))
def vdown(x, y1, y2, color=None, sw=2.2):
    arrow(x, y1, x, y2, color or C["muted"], sw)
def section(x, y, w, h, title, sub, fill, stroke, head, band=40):
    rect(x, y, w, h, fill, stroke, rx=14, sw=1.8)
    d.rounded_rectangle([x*S, y*S, (x+w)*S, (y+band)*S], radius=14*S, fill=head)
    d.rectangle([x*S, (y+band-14)*S, (x+w)*S, (y+band)*S], fill=head)
    text(x+22, y+27, title, size=17, bold=True)
    text(x+w-22, y+27, sub, size=12.5, fill=C["muted"], anchor="ra")

# header
text(40, 50, "RAGLab — Architecture Schema", size=33, bold=True)
text(40, 80, "One project, two generations, one pivot — and one current revision. Reconstructed from 387 commits (04 Sept → 05 Oct 2026), the in-repo audit trail, and the code at HEAD.", size=14, fill=C["muted"])
text(40, 102, "Legend:", size=12.5, bold=True, fill=C["muted"])
lx = 100
for label, f, s in [("Generation 1 — flat lab (superseded)", C["gen1_fill"], C["gen1_str"]),
                    ("Pivot / re-foundation", C["pivot_fill"], C["pivot_str"]),
                    ("Generation 2 — governed pipeline", C["gen2_fill"], C["gen2_str"]),
                    ("CURRENT — latest revision", C["last_fill"], C["last_str"])]:
    rect(lx, 91, 18, 15, f, s, rx=4, sw=1.2)
    text(lx+25, 103, label, size=12, fill=C["muted"]); lx += 30 + 6.3*len(label)

# generation 1
section(40, 130, 1420, 250, "GENERATION 1 — 04 → 23 Sept 2026 · the flat retrieval lab (superseded)",
        "measure chunks, models and fusion strategies", C["gen1_fill"], C["gen1_str"], C["gen1_head"])
g1 = [
    ("1 · Corpus", ["fictional sample sheets", "Al Baraka web-compiled round", "(both later deleted)"]),
    ("2 · Chunk", ["size mode 220/40 default", "token-window + overlap", "reviewed chunk maps"]),
    ("3 · Embed", ["multi-provider embeddings:", "Gemini · Jina · HuggingFace", "NVIDIA NeMo-Retriever"]),
    ("4 · Store", ["local Chroma, cosine", "chunk fingerprint v4", "stale index ⇒ refuse"]),
    ("5 · Retrieve", ["vector / rrf / blend fusion", "BM25 k=60 · blend λ=0.7", "fusion tie-break policies"]),
    ("6 · Query variants", ["machine translation ON", "(gemini-3.5-flash-lite)", "best-variant merge"]),
    ("7 · Answer", ["multi-provider + fallback", "grounded-v1 JSON claims", "citation gate: quote+numeric"]),
]
bw, gap, bx = 183, 16.5, 58
for i, (title, lines) in enumerate(g1):
    chip(bx, 195, bw, 100, title, lines, stroke=C["gen1_str"], title_size=12.5, body_size=10.7)
    if bx + bw + 25 < 1460:
        arrow(bx+bw+2, 245, bx+bw+gap-3, 245, C["gen1_str"], sw=1.8)
    bx += bw + gap
text(1460, 322, "Model-agnostic by design: providers, translations and retrievers were still being compared — nothing was pinned.", size=11.5, fill=C["muted"], anchor="ra")

# pivot
section(40, 410, 1420, 175, "THE PIVOT — 24 → 28 Sept 2026 · from a lab that measures to a system that decides",
        "owner direction + external target-state report", C["pivot_fill"], C["pivot_str"], C["pivot_head"])
piv = [
    ("Target-state report", ["external (AR) design for a governed", "banking RAG decision system: intent ·", "evidence · sufficiency · governance"]),
    ("Reverse-engineering", ["RAGLAB_SPEC.md — every claim cited", "RAGLAB_GAP_ANALYSIS.md — current vs", "target; “the answer model never touches the query”"]),
    ("Re-foundation", ["corpus = docs/ ONLY; web round reverted", "step-and-plan protocol: a published plan", "per step, owner gate before any number"]),
    ("Corpus repair", ["4 docs, LLM-only linguistic repair:", "215 + 35 + 23 + 21 entries, machine-", "checked; adopted codices → 339 chunks"]),
]
bw2, bx = 321, 58
for title, lines in piv:
    chip(bx, 470, bw2, 96, title, lines, stroke=C["pivot_str"], title_size=13.5, body_size=10.8)
    if bx + bw2 + 40 < 1460:
        arrow(bx+bw2+2, 518, bx+bw2+28, 518, C["pivot_str"], sw=1.8)
    bx += bw2 + 32
vdown(750, 380, 408); vdown(750, 585, 628)

# generation 2
section(40, 630, 1090, 610, "GENERATION 2 — 24 Sept → 05 Oct 2026 · the governed, layered pipeline",
        "structure → knowledge → understanding → delivery", C["gen2_fill"], C["gen2_str"], C["gen2_head"])
layers = [
    ("LAYER 1 · Corpus & structure", "loader → restructure → fingerprint-guarded store",
     [("loader.py", ["PDF (visual order) / DOCX / MD", "Arabic normalisation built in"]),
      ("restructure.py", ["3-step structural chunking (220/40)", "adopted codices · RTL zones"]),
      ("docstore.py", ["content-hash versions; push/delete", "stale surfaced, never silent"]),
      ("store.py", ["chunk_fp + embedding_fp guards", "mixed or stale ⇒ refuse to serve"])]),
    ("LAYER 2 · Knowledge — Phase 4 (01 Oct)", "deterministic, declared data — read-only over loi 2016-48",
     [("units.py · 198/198", ["stable ids loi-2016-48:artNNN", "governed functional types"]),
      ("relations.py", ["77 internal reference edges", "+ grounding: Circulaire → art011"]),
      ("legal_numbers.py", ["53 numeric records + corrections", "→ GET /numbers (1st additive route)"]),
      ("governance.py", ["6 axes on chunk metadata", "authority · effectivity · audience"])]),
    ("LAYER 3 · Understanding — Phases 5 + 8 + 9", "deterministic first; one bounded model call last; fail-closed",
     [("intent.py", ["declared rules, first match wins", "compound split · ambiguity note"]),
      ("evidence_plan → sufficiency", ["requirements from intent", "4 explicit sufficiency states"]),
      ("decompose → guided rounds", ["micro-questions per request", "bounded guided retrieval rounds"]),
      ("interrogate.py + topic_map", ["ONE bounded model call", "corpus topics → rephrasing"]),
      ("relational_expansion (OFF)", ["intent-gated tail slots", "measured NEUTRAL → stays off"])]),
    ("LAYER 4 · Delivery & audit — Phase 6", "the answer is a contract: cited and checked, or honestly refused",
     [("answer.py — citation gate", ["verbatim quotes (normalised)", "every number in its quotes", "unit_id on law chunks"]),
      ("service.py · 1.5.0", ["27 HTTP routes · additive-only freeze", "token · 409 seals · error envelope"]),
      ("scrub.py + audit.py", ["post-gate PII scrub", "JSONL trail → GET /audit"]),
      ("REFUSALS", ["reason + referral naming the gap", "never free text, never a guess"])]),
]
ly = 690
for header, sub, chips in layers:
    n = len(chips); lh = 120 if n == 4 else 136
    rect(58, ly, 1054, lh, "#ffffff", C["gen2_str"], rx=10, sw=1.2)
    d.rounded_rectangle([58*S, ly*S, (58+1054)*S, (ly+26)*S], radius=6*S, fill=C["gen2_head"])
    text(70, ly+18, header, size=13.5, bold=True)
    text(1112-12, ly+18, sub, size=10.8, fill=C["muted"], anchor="ra")
    cw = (1054 - (n+1)*12)/n; cx = 70
    for title, lines in chips:
        chip(cx, ly+34, cw, lh-46, title, lines, title_size=11.5, body_size=10.4)
        cx += cw + 12
    ly += lh + 10

# last update
lx0, lw = 1148, 312
rect(lx0, 630, lw, 610, C["last_fill"], C["last_str"], rx=14, sw=2.6)
d.rounded_rectangle([lx0*S, 630*S, (lx0+lw)*S, (630+62)*S], radius=14*S, fill=C["last_str"])
d.rectangle([lx0*S, (630+48)*S, (lx0+lw)*S, (630+62)*S], fill=C["last_str"])
text(lx0+20, 655, "LAST UPDATE", size=19, bold=True, fill="#ffffff")
text(lx0+20, 682, "05 Oct 2026 · service 1.5.0 · CI green", size=11.5, fill="#ffffff")
las = [
    ("Phase 9 — question ↔ retrieval", ["262 source-grounded topic nodes;", "exact-match ids replace loose lists"]),
    ("Cross-script arm activated", ["2/2 answered with the arm,", "2/2 refused without it"]),
    ("Guided round wired into /answer", ["it was dead on the deployed path;", "found by a live run, then fixed"]),
    ("Root cause TM03 fixed", ["“Quelle” matched inside “Quelles”;", "the French question answers one-shot"]),
    ("Duplicate re-push blocked", ["POST /documents → 409 by SHA-256,", "not filename (859 dup chunks gone)"]),
    ("Re-aimed answers disclosed", ["8 colloquial→corpus rows + scope_note:", "a re-aimed answer cannot look ordinary"]),
    ("Still open (Phase 9 stages 3–8)", ["path / level / parent_id, topical", "arms, facets — none of it default"]),
]
yy = 706
for title, lines in las:
    h = 30 + 15*len(lines) + 8
    rect(lx0+14, yy, lw-28, h, "#ffffff", C["last_str"], rx=8, sw=1.1)
    text(lx0+24, yy+17, "★ " + title, size=11, bold=True, fill=C["last_str"])
    y2 = yy + 33.5
    for ln in lines:
        text(lx0+24, y2, ln, size=10, fill=C["muted"]); y2 += 15
    yy += h + 7

# spine
rect(40, 1262, 1420, 70, C["panel"], C["line"], rx=12, sw=1.4)
text(58, 1288, "PINNED RUNTIME SPINE — unchanged across both generations; the policy raises on any substitution", size=13, bold=True)
sx = 58
for label in ["Embeddings — nvidia/nemotron-3-embed-1b · 2048-d · retry + bisect",
              "Index — local ChromaDB cosine · one collection per space × mode",
              "Answers — xKiro qwen3.8-max:free · grounded-v1 JSON · price check"]:
    rect(sx, 1298, 440, 24, "#ffffff", C["line"], rx=7, sw=1.1)
    text(sx+10, 1314, label, size=10.5, fill=C["muted"]); sx += 456
text(1460, 1362, "no query translation · no provider fallback · refusal over guessing", size=11, fill=C["muted"], anchor="ra")

img = img.resize((W*2, H*2), Image.LANCZOS)
img.save(os.path.join(HERE, "architecture_schema.png"), "PNG", optimize=True)
print("png written", img.size)


# ==========================================================================
# timeline (PNG + SVG)
# ==========================================================================
# -*- coding: utf-8 -*-
"""Timeline figure: 369 non-merge commits, 04 Sept → 05 Oct 2026."""
from PIL import Image, ImageDraw, ImageFont
import math

W, H, S = 1600, 660, 2
DJ  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DJB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fc = {}
def font(size, bold=False):
    k = (size, bold)
    if k not in _fc: _fc[k] = ImageFont.truetype(DJB if bold else DJ, int(size*S))
    return _fc[k]

C = {"bg":"#ffffff","text":"#16202c","muted":"#5c6775","line":"#c8d0dc",
     "g1":"#7ba4f0","doc":"#9aa4b2","pivot":"#dfa23c","g2":"#4fae8d","last":"#d6336c"}

days = [("09-04",18,"g1"),("09-05",76,"g1"),("09-06",49,"g1"),("09-07",13,"g1"),
        ("09-08",6,"g1"),("09-11",6,"g1"),("09-14",18,"g1"),("09-15",2,"g1"),
        ("09-16",14,"g1"),("09-23",3,"doc"),("09-24",13,"doc"),("09-25",1,"doc"),
        ("09-28",14,"pivot"),("09-29",8,"pivot"),("09-30",32,"pivot"),
        ("10-01",50,"g2"),("10-02",21,"g2"),("10-04",5,"last"),("10-05",20,"last")]

img = Image.new("RGB", (W*S, H*S), "#ffffff")
d = ImageDraw.Draw(img)

def text(x, y, s, size=13, fill=None, bold=False, anchor="la", angle=0):
    f = font(size, bold)
    if angle:
        tmp = Image.new("RGBA", (int(len(s)*size*1.2*S), int(size*2.2*S)), (0,0,0,0))
        td = ImageDraw.Draw(tmp); td.text((0,0), s, font=f, fill=fill or C["text"])
        tmp = tmp.rotate(angle, expand=1)
        img.paste(tmp, (int(x*S), int(y*S)), tmp)
    else:
        d.text((x*S, y*S), s, font=f, fill=fill or C["text"], anchor=anchor)

# header
text(40, 42, "RAGLab — development timeline", size=30, bold=True)
text(40, 76, "369 non-merge commits (387 total) across 15 branches · 04 Sept → 05 Oct 2026 · the last day of work is highlighted",
     size=14, fill=C["muted"])

# phase bands
bands = [
    (0,  8,  "Generation 1 — flat lab", "g1"),
    (9,  11, "Pivot & docs", "doc"),
    (12, 14, "Corpus repair", "pivot"),
    (15, 16, "Phases 2–6", "g2"),
    (17, 18, "★ Phase 9", "last"),
]
X0, X1, YTOP, YBASE = 70, 1540, 190, 470
n = len(days)
slot = (X1 - X0) / n
band_y = 104

for a, b, label, key in bands:
    x1 = X0 + a*slot; x2 = X0 + (b+1)*slot
    d.rounded_rectangle([x1*S, band_y*S, x2*S, (band_y+22)*S], radius=6*S,
                        outline=C[key], width=max(1,int(1.2*S)))
    d.rounded_rectangle([x1*S, band_y*S, x2*S, (band_y+22)*S], radius=6*S, fill="#ffffff", outline=C[key], width=max(1,int(1.2*S)))
    text((x1+x2)/2, band_y+15, label, size=11.5, fill=C[key], bold=True, anchor="ma")

# baseline + bars
d.line([X0*S, YBASE*S, X1*S, YBASE*S], fill=C["line"], width=max(1,int(1.4*S)))
MAXX = 76.0
for i, (day, cnt, key) in enumerate(days):
    cx = X0 + i*slot + slot/2
    h = (cnt / MAXX) * (YBASE - YTOP - 60)
    bw = slot * 0.62
    col = C[key]
    d.rounded_rectangle([(cx-bw/2)*S, (YBASE-h)*S, (cx+bw/2)*S, YBASE*S], radius=4*S, fill=col)
    text(cx, YBASE - h - 8, str(cnt), size=11, fill=col, bold=True, anchor="ma")
    text(cx, YBASE + 18, day.replace("09-", "Sep ").replace("10-", "Oct "), size=10.5, fill=C["muted"], anchor="ma")

# event annotations
def annot(x, y, s, color, anchor="ma"):
    text(x, y, s, size=11.5, fill=color, bold=True, anchor=anchor)
def vline(x, y1, y2, color):
    d.line([x*S, y1*S, x*S, y2*S], fill=color, width=max(1, int(1.3*S)))

piv_x = X0 + 12*slot + slot/2
annot(piv_x, 530, "28 Sep — pivot: corpus = docs/ only,", C["pivot"]); annot(piv_x, 546, "step-and-plan protocol, owner gates", C["pivot"])
vline(piv_x, 522, 566, C["pivot"])
g2_x = X0 + 15*slot + slot/2
annot(g2_x, 585, "01 Oct — Phase 4 knowledge layer", C["g2"]); annot(g2_x, 601, "(198 units · 77 edges · 53 numbers)", C["g2"])
vline(g2_x, 576, 621, C["g2"])
last_x = X0 + 17*slot + slot/2
annot(last_x, 530, "★ 04–05 Oct — Phase 9:", C["last"]); annot(last_x, 546, "question ↔ retrieval, service 1.5.0", C["last"])
vline(last_x, 522, 566, C["last"])
g1_x = X0 + 3*slot
annot(g1_x, 530, "04–16 Sep — first pipeline: providers,", C["g1"]); annot(g1_x, 546, "translation and fusion compared by A/B", C["g1"])
vline(g1_x, 522, 566, C["g1"])
mid_x = X0 + 9*slot + slot/2
annot(mid_x, 585, "23–25 Sep — spec, gap analysis,", C["doc"]); annot(mid_x, 601, "production packaging, endpoint freeze", C["doc"])
vline(mid_x, 576, 621, C["doc"])
act_x = X0 + 16*slot + slot/2
annot(act_x, 630, "02 Oct — activation: sufficiency + interrogation ON (1.4.0)", C["g2"])
vline(act_x, 611, 641, C["g2"])

# legend
lx = 70
for label, key in [("Generation 1", "g1"), ("Pivot / docs", "doc"), ("Corpus repair", "pivot"),
                   ("Phases 2–6", "g2"), ("Phase 9 (last update)", "last")]:
    d.rounded_rectangle([lx*S, 138*S, (lx+14)*S, 151*S], radius=3*S, fill=C[key])
    text(lx+20, 150, label, size=11.5, fill=C["muted"]); lx += 30 + 6.2*len(label)

img = img.resize((W*2, H*2), Image.LANCZOS)
img.save(os.path.join(HERE, "timeline.png"), "PNG", optimize=True)
print("timeline png", img.size)

# ---- matching SVG ----
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
def st(x, y, s, size=13, fill=None, bold=False, anchor="start"):
    s = s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    w = "bold" if bold else "normal"
    svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}" fill="{fill or C["text"]}" font-weight="{w}" text-anchor="{anchor}">{s}</text>')
st(40, 50, "RAGLab — development timeline", size=30, bold=True)
st(40, 82, "369 non-merge commits (387 total) across 15 branches · 04 Sept → 05 Oct 2026 · the last day of work is highlighted", size=14, fill=C["muted"])
for a, b, label, key in bands:
    x1 = X0 + a*slot; x2 = X0 + (b+1)*slot
    svg.append(f'<rect x="{x1:.1f}" y="{band_y}" width="{x2-x1:.1f}" height="22" rx="6" fill="#ffffff" stroke="{C[key]}" stroke-width="1.2"/>')
    st((x1+x2)/2, band_y+15, label, size=11.5, fill=C[key], bold=True, anchor="middle")
svg.append(f'<line x1="{X0}" y1="{YBASE}" x2="{X1}" y2="{YBASE}" stroke="{C["line"]}" stroke-width="1.4"/>')
for i, (day, cnt, key) in enumerate(days):
    cx = X0 + i*slot + slot/2
    h = (cnt/MAXX)*(YBASE-YTOP-60); bw = slot*0.62
    svg.append(f'<rect x="{cx-bw/2:.1f}" y="{YBASE-h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="4" fill="{C[key]}"/>')
    st(cx, YBASE-h-6, str(cnt), size=11, fill=C[key], bold=True, anchor="middle")
    st(cx, YBASE+18, day.replace("09-","Sep ").replace("10-","Oct "), size=10.5, fill=C["muted"], anchor="middle")
for x, y, s, col in [(piv_x,540,"28 Sep — pivot: corpus = docs/ only,",C["pivot"]),(piv_x,556,"step-and-plan protocol, owner gates",C["pivot"]),
                     (g1_x,540,"04–16 Sep — first pipeline: providers,",C["g1"]),(g1_x,556,"translation and fusion compared by A/B",C["g1"]),
                     (last_x,540,"★ 04–05 Oct — Phase 9:",C["last"]),(last_x,556,"question ↔ retrieval, service 1.5.0",C["last"]),
                     (g2_x,595,"01 Oct — Phase 4 knowledge layer",C["g2"]),(g2_x,611,"(198 units · 77 edges · 53 numbers)",C["g2"]),
                     (mid_x,595,"23–25 Sep — spec, gap analysis,",C["doc"]),(mid_x,611,"production packaging, endpoint freeze",C["doc"]),
                     (act_x,630,"02 Oct — activation: sufficiency + interrogation ON (1.4.0)",C["g2"])]:
    st(x, y, s, size=11.5, fill=col, bold=True, anchor="middle")
for x, y1, y2, col in [(piv_x,522,566,C["pivot"]),(g1_x,522,566,C["g1"]),(last_x,522,566,C["last"]),
                       (g2_x,576,621,C["g2"]),(mid_x,576,621,C["doc"]),(act_x,611,641,C["g2"])]:
    svg.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y2}" stroke="{col}" stroke-width="1.3"/>')
lx = 70
for label, key in [("Generation 1","g1"),("Pivot / docs","doc"),("Corpus repair","pivot"),("Phases 2–6","g2"),("Phase 9 (last update)","last")]:
    svg.append(f'<rect x="{lx}" y="138" width="14" height="13" rx="3" fill="{C[key]}"/>')
    st(lx+20, 149, label, size=11.5, fill=C["muted"]); lx += 30 + 6.2*len(label)
svg.append('</svg>')
open(os.path.join(HERE, "timeline.svg"), "w", encoding="utf-8").write("\n".join(svg))
print("timeline svg written")

