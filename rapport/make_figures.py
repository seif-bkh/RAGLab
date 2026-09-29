#!/usr/bin/env python3
"""Generate the figures for the internship report (rapport/figures/*.png).

All numbers are taken from the RAGLab repository itself:
  - harness50 A/B      : README.md (restructure 47% vs size 40% hit@1)
  - free models        : raglab/FREE_MODELS_REPORT.md (5 Sept 2026)
  - retrieval judge    : raglab/HARD_HARNESS_STATUS.md (CI run 34005544576)
  - chunking measures  : raglab/README.md (chunk_maps.py measure)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})
NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
LIGHT = "#eaf2fb"
GREY = "#6c757d"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


def box(ax, xy, w, h, text, fc=LIGHT, ec=NAVY, fs=9, bold=False, alpha=1.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.5, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=NAVY, wrap=True)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", color=NAVY,
                                 lw=1.6, mutation_scale=14))


# ---------------------------------------------------------------- 1. workflow
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Flux de traitement documentaire avant le projet", pad=12)
steps = [
    "Documents bancaires\ndisperses (PDF / DOCX)\narabe, francais, anglais",
    "Lecture manuelle\npar un expert metier",
    "Recherche par mots-cles\n(Ctrl+F), sans semantique",
    "Reponse redigee a la main\nsans citation verifiable",
]
y = 8.0
for i, s in enumerate(steps):
    box(ax, (2.2, y - 1), 5.6, 1.15, s, fc="#fdebd0" if i == 3 else LIGHT)
    if i < len(steps) - 1:
        arrow(ax, (5.0, y - 1.0), (5.0, y - 1.55))
    y -= 1.75
box(ax, (0.2, 0.35), 9.6, 1.0,
    "Limites : aucune recherche translingue  |  aucune tracabilite source  |  "
    "reponse non reproductible  |  dependance totale a l'expert",
    fc="#fadbd8", ec=ORANGE, fs=8)
save(fig, "workflow_before.png")

# ---------------------------------------------------------------- 2. gantt
fig, ax = plt.subplots(figsize=(9.5, 4.6))
tasks = [
    ("Cadrage & constitution du corpus (docs/ + data/)", 0, 1.2),
    ("Ingestion : parsing, normalisation, chunking", 0.8, 3.2),
    ("Embeddings Nemotron + index ChromaDB", 2.0, 3.6),
    ("Generation citee + garde-fous (refus, PII)", 3.0, 4.4),
    ("Evaluation : harnais, juge retrieval, A/B chunking", 3.8, 5.4),
    ("Comparaison des modeles gratuits (xKiro, NVIDIA)", 4.6, 5.8),
    ("Microservice FastAPI + Docker + CI GitHub", 5.2, 6.6),
    ("Durcissement, docs, redaction", 6.0, 7.0),
]
colors = [NAVY, TEAL, TEAL, ORANGE, "#8e44ad", ORANGE, NAVY, GREY]
for i, ((label, start, end), c) in enumerate(zip(tasks, colors)):
    ax.barh(len(tasks) - 1 - i, end - start, left=start, height=0.55,
            color=c, edgecolor="white")
    ax.text(start + 0.08, len(tasks) - 1 - i, label, va="center", ha="left",
            fontsize=8, color="white", fontweight="bold")
ax.set_yticks([])
ax.set_xticks(range(0, 8))
ax.set_xticklabels([f"S{i}" if i else "" for i in range(8)])
ax.set_xlabel("Semaines de stage (S1 a S7)")
ax.set_title("Diagramme de Gantt previsionnel du stage", pad=12)
ax.set_xlim(0, 7.2)
ax.set_ylim(-0.8, len(tasks) - 0.2)
ax.grid(axis="x", linestyle=":", alpha=0.6)
ax.invert_yaxis()
save(fig, "gantt_stage.png")

# ---------------------------------------------------------------- 3. architecture
fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Architecture generale de la solution RAGLab", pad=12)
# columns
box(ax, (0.15, 6.6), 2.2, 2.4, "Clients\n\n- CLI (main.py)\n- Console (app.py)\n- HTTP (service REST)",
    fc="#d5f5e3", bold=False, fs=8)
box(ax, (2.7, 6.6), 2.6, 2.4,
    "Microservice FastAPI\n\n25 routes : /health\n/search /answer /ingest\n/documents /evaluate ...",
    fc=LIGHT, fs=8)
box(ax, (5.65, 6.6), 2.0, 2.4, "Noyau partage\n\nloader / chunker\nembedder / store\nanswer + gate",
    fc="#fef9e7", fs=8)
box(ax, (8.0, 6.6), 1.85, 2.4, "Externes\n\nNVIDIA embeddings\nxKiro Qwen 3.8",
    fc="#fdedec", fs=8)
for xa, xb in [(2.35, 2.7), (5.3, 5.65), (7.65, 8.0)]:
    arrow(ax, (xa, 7.8), (xb, 7.8))
box(ax, (0.15, 4.4), 4.55, 1.7,
    "Stockage local : ChromaDB persistant (cosinus) + BM25\n"
    "1 collection par (espace d'embedding x mode de chunking)",
    fc="#eaf2fb", fs=8)
box(ax, (5.05, 4.4), 4.8, 1.7,
    "Garde-fous : citations mot-a-mot, refus locaux\n"
    "(prive/temps reel), empreintes anti-melange",
    fc="#fef9e7", fs=8)
arrow(ax, (4.0, 6.6), (4.0, 6.1))
arrow(ax, (6.6, 6.6), (6.6, 6.1))
box(ax, (0.15, 2.2), 9.7, 1.7,
    "Corpus : 4 documents bancaires reels ar/fr (docs/) + fiches fictives (data/)\n"
    "836 chunks (tokenizer cl100k_base, 220/40)  |  3 langues : ar / fr / en",
    fc="#f4f6f7", ec=GREY, fs=8)
box(ax, (0.15, 0.3), 9.7, 1.4,
    "CI GitHub Actions : 7 workflows (tests, images Docker, free-models, hard-harness, "
    "retrieval-judge, real-test, catalogues)",
    fc="#f4f6f7", ec=GREY, fs=8)
save(fig, "architecture_diagram.png")

# ---------------------------------------------------------------- 4. pipeline
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Pipeline de traitement : ingestion et interrogation", pad=12)
ax.text(0.2, 9.0, "INGESTION (hors-ligne)", fontsize=10, fontweight="bold",
        color=NAVY)
ing = ["Fichiers\nPDF/DOCX/MD", "Lecture +\nnormalisation\n(arabe repare)",
       "Chunking\nrestructure /\nsize / manual",
       "Embeddings\nNemotron 2048d\nlots de 32",
       "Index\nChromaDB cosinus\n+ BM25"]
x = 0.2
for i, s in enumerate(ing):
    box(ax, (x, 7.2), 1.75, 1.5, s, fs=7.5)
    if i < len(ing) - 1:
        arrow(ax, (x + 1.75, 7.95), (x + 1.95, 7.95))
    x += 1.95
ax.text(0.2, 6.3, "REQUETE (en ligne)", fontsize=10, fontweight="bold",
        color=TEAL)
qry = ["Question\nar / fr / en", "Embedding\nrequete\nNemotron",
       "Top-5\ncosinus\nrequete d'origine",
       "Prompt\ngrounded-v1\n+ contexte",
       "Portillon :\ncitations exactes\nou REFUS"]
x = 0.2
for i, s in enumerate(qry):
    box(ax, (x, 4.3), 1.75, 1.6, s, fc="#e8f8f5", ec=TEAL, fs=7.5)
    if i < len(qry) - 1:
        arrow(ax, (x + 1.75, 5.1), (x + 1.95, 5.1))
    x += 1.95
box(ax, (0.2, 2.2), 9.6, 1.4,
    "Reponse livree = claims JSON portant des citations mot-a-mot extraites des chunks ;\n"
    "tout nombre doit apparaitre dans ses citations, sinon refus structure.",
    fc="#fef9e7", fs=7.5)
box(ax, (0.2, 0.3), 9.6, 1.4,
    "Contrats d'identite : empreinte de chunking + empreinte d'espace d'embedding ;\n"
    "un index obsolete est refuse (re-ingest --reset exige).",
    fc="#f4f6f7", ec=GREY, fs=7.5)
save(fig, "data_pipeline.png")

# ---------------------------------------------------------------- 5. scores
fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
fig.suptitle("Distributions des scores mesures (harnais et benchmarks)",
             fontweight="bold", fontsize=11)
# (a) harness50
ax = axes[0]
cats = ["hit@1\n(langue mel.)", "hit@1\n(verbatim)"]
x = np.arange(len(cats))
ax.bar(x - 0.2, [47, 75], 0.4, label="restructure", color=TEAL)
ax.bar(x + 0.2, [40, 42], 0.4, label="size 220/40", color=GREY)
for i, v in enumerate([47, 75]):
    ax.text(i - 0.2, v + 1, f"{v}%", ha="center", fontsize=8)
for i, v in enumerate([40, 42]):
    ax.text(i + 0.2, v + 1, f"{v}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=8)
ax.set_ylim(0, 100)
ax.set_ylabel("%")
ax.set_title("(a) A/B chunking (50 q.)", fontsize=9)
ax.legend(fontsize=7)
# (b) free models
ax = axes[1]
models = ["Qwen\n3.8-max", "MiniMax\nM3", "Mistral\nsmall", "DeepSeek\nflash"]
vals = [92.9, 78.6, 42.9, 50.0]
cols = [TEAL, NAVY, GREY, ORANGE]
bars = ax.bar(models, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center",
            fontsize=8)
ax.set_ylim(0, 110)
ax.set_title("(b) Rubrique reponse dev (14 q.)", fontsize=9)
# (c) retrieval judge
ax = axes[2]
cats = ["pret-a-repondre\ntop-5", "requetes\nsemantiques"]
x = np.arange(len(cats))
ax.bar(x - 0.2, [2.7, 0.0], 0.4, label="lexical BM25", color=GREY)
ax.bar(x + 0.2, [10.0, 62.8], 0.4, label="embeddings NVIDIA",
       color=TEAL)
for i, v in enumerate([2.7, 0.0]):
    ax.text(i - 0.2, v + 1, f"{v}%", ha="center", fontsize=8)
for i, v in enumerate([10.0, 62.8]):
    ax.text(i + 0.2, v + 1, f"{v}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=8)
ax.set_ylim(0, 80)
ax.set_title("(c) Juge retrieval (1407 q.)", fontsize=9)
ax.legend(fontsize=7)
save(fig, "score_distributions.png")

# ---------------------------------------------------------------- 6. heatmap
fig, ax = plt.subplots(figsize=(7.2, 4.6))
rows = ["Pret-a-repondre (top-5)", "Troncon >= 80 %", "Requetes semantiques",
        "AUC d'abstention"]
cols = ["Lexical (BM25)", "Embeddings (NVIDIA)"]
data = np.array([[2.7, 10.0], [9.5, 37.3], [0.0, 62.8], [55.8, 67.9]])
im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)
for i in range(len(rows)):
    for j in range(len(cols)):
        v = data[i, j]
        txt = f"{v:.1f} %" if i < 3 else f"{v/100:.3f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                fontweight="bold",
                color="white" if v > 50 else NAVY)
ax.set_title("Lexical vs semantique : matrice des metriques retrieval\n"
             "(run CI 34005544576, 469 familles / 1407 questions)", pad=12)
fig.colorbar(im, ax=ax, label="% (ou AUC x100)")
save(fig, "metrics_heatmap.png")

print("done ->", OUT)
