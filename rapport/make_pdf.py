#!/usr/bin/env python3
"""Build rapport_stage.pdf with fpdf2, mirroring rapport_stage.tex.

(The sandbox has no TeXLive and cannot download it, so this script renders
a faithful PDF directly. The .tex file remains the editable source of truth:
compile it with TeXLive/Overleaf for the canonical layout.)
"""
import os
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
FD = "/usr/share/fonts/truetype/dejavu/"

NAVY = (31, 58, 95)
GREY = (110, 110, 110)


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.add_font("sans", "", FD + "DejaVuSans.ttf")
        self.add_font("sans", "B", FD + "DejaVuSans-Bold.ttf")
        self.add_font("mono", "", FD + "DejaVuSansMono.ttf")
        self.add_font("mono", "B", FD + "DejaVuSansMono-Bold.ttf")
        self.set_auto_page_break(True, margin=25)
        self.cover_done = False

    def header(self):
        if self.cover_done and self.page_no() > 1:
            self.set_font("sans", "", 8)
            self.set_text_color(*GREY)
            self.cell(0, 8, "Rapport de stage \u2014 RAGLab : RAG multilingue pour documents bancaires",
                      align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
            self.set_text_color(0, 0, 0)

    def footer(self):
        if self.cover_done:
            self.set_y(-15)
            self.set_font("sans", "", 9)
            self.set_text_color(*GREY)
            self.cell(0, 10, f"{self.page_no()}", align="C")
            self.set_text_color(0, 0, 0)

    # -- building blocks -------------------------------------------------
    def h1(self, num, title):
        self.ln(4)
        self.set_font("sans", "B", 18)
        self.set_text_color(*NAVY)
        label = f"Chapitre {num}" if num else ""
        if label:
            self.cell(0, 10, label, new_x="LMARGIN", new_y="NEXT")
        self.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*NAVY)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin,
                  self.get_y() + 1)
        self.ln(6)
        self.set_text_color(0, 0, 0)

    def h2(self, title):
        self.ln(3)
        self.set_font("sans", "B", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def h3(self, title):
        self.ln(2)
        self.set_font("sans", "B", 12)
        self.multi_cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def p(self, text):
        self.set_font("sans", "", 11)
        self.multi_cell(0, 6.5, text, align="J", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def bullets(self, items):
        self.set_font("sans", "", 11)
        for it in items:
            x = self.get_x()
            self.cell(6, 6.5, "\u2022")
            self.multi_cell(0, 6.5, it, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def numbered(self, items):
        self.set_font("sans", "", 11)
        for i, it in enumerate(items, 1):
            self.cell(8, 6.5, f"{i}.")
            self.multi_cell(0, 6.5, it, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def code(self, text):
        self.set_font("mono", "", 9)
        self.set_fill_color(244, 246, 247)
        for line in text.split("\n"):
            self.cell(0, 5.2, "  " + line, new_x="LMARGIN", new_y="NEXT",
                      fill=True)
        self.ln(4)

    def table(self, headers, rows, caption=None, widths=None, fs=10):
        self.set_font("sans", "B", fs)
        avail = self.w - self.l_margin - self.r_margin
        if widths is None:
            widths = [avail / len(headers)] * len(headers)
        with FPDF.table(
                self,
                col_widths=tuple(w / avail for w in widths),
                text_align="LEFT", first_row_as_headings=False,
                line_height=5.5) as t:
            rh = t.row()
            for h in headers:
                rh.cell(str(h))
            self.set_font("sans", "", fs)
            for r in rows:
                row = t.row()
                for c in r:
                    row.cell(str(c))
        if caption:
            self.set_font("sans", "", 9)
            self.set_text_color(*GREY)
            self.multi_cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
        self.ln(3)

    def figure(self, name, caption, interp=None, w=170):
        self.ln(2)
        self.image(os.path.join(FIG, name), w=w, x=(210 - w) / 2)
        self.set_font("sans", "", 9)
        self.set_text_color(*GREY)
        self.multi_cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        if interp:
            self.multi_cell(0, 5, interp, align="J", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)


r = Report()
r.set_margin(25)

# ================= COVER =================
r.add_page()
r.set_font("sans", "B", 13)
r.cell(0, 8, "R\u00e9publique Tunisienne", align="C", new_x="LMARGIN",
       new_y="NEXT")
r.cell(0, 8, "\u00c9cole sup\u00e9rieure priv\u00e9e d\u2019ing\u00e9nierie et de technologie",
       align="C", new_x="LMARGIN", new_y="NEXT")
r.ln(30)
r.set_font("sans", "B", 26)
r.set_text_color(*NAVY)
r.multi_cell(0, 13, "Rapport de Stage\nd\u2019Immersion en Entreprise", align="C", new_x="LMARGIN", new_y="NEXT")
r.set_text_color(0, 0, 0)
r.ln(12)
r.set_font("sans", "", 14)
r.multi_cell(0, 9, "Sujet :", align="C", new_x="LMARGIN", new_y="NEXT")
r.set_font("sans", "B", 14)
r.multi_cell(
    0, 9, "Conception et \u00e9valuation d\u2019un syst\u00e8me de\n"
    "question-r\u00e9ponse documentaire multilingue (RAG)\n"
    "appliqu\u00e9 \u00e0 des documents bancaires", align="C", new_x="LMARGIN", new_y="NEXT")
r.ln(10)
r.set_font("sans", "", 14)
r.multi_cell(0, 9, "R\u00e9alis\u00e9 au sein de", align="C", new_x="LMARGIN", new_y="NEXT")
r.set_font("sans", "B", 16)
r.cell(0, 10, "Al Baraka Bank", align="C", new_x="LMARGIN", new_y="NEXT")
r.ln(12)
r.set_font("sans", "B", 13)
r.cell(0, 9, "Du 1er ao\u00fbt au 1er octobre 2026", align="C", new_x="LMARGIN", new_y="NEXT")
r.ln(4)
r.set_font("sans", "", 13)
r.cell(0, 9, "\u00c9labor\u00e9 par  TODO-NOM Pr\u00e9nom", align="C",
       new_x="LMARGIN", new_y="NEXT")
r.cell(0, 9, "Encadr\u00e9 par  Ahlem BENHADDOUD", align="C",
       new_x="LMARGIN", new_y="NEXT")
r.ln(4)
r.set_font("sans", "", 12)
r.cell(0, 9, "Ann\u00e9e universitaire 2025 \u2013 2026", align="C",
       new_x="LMARGIN", new_y="NEXT")
r.cover_done = True

# ================= TOC (manual) =================
r.add_page()
r.h1(None, "Table des mati\u00e8res")
r.set_font("sans", "", 11)
toc = [
    "D\u00e9dicaces", "Remerciements", "Introduction G\u00e9n\u00e9rale",
    "Chapitre 1 \u2014 Pr\u00e9sentation de l\u2019organisme d\u2019accueil et \u00e9tude pr\u00e9alable",
    "Chapitre 2 \u2014 Conception de la solution",
    "Chapitre 3 \u2014 R\u00e9alisation, traitements et r\u00e9sultats",
    "Chapitre 4 \u2014 Conclusion g\u00e9n\u00e9rale et perspectives",
]
for t in toc:
    r.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
r.ln(4)
r.p("Note : le champ TODO-NOM de la page de garde reste \u00e0 remplir (voir les "
    "\\newcommand en t\u00eate de rapport_stage.tex). "
    "Tous les chiffres du rapport sont issus des mesures du stage (harnais, "
    "bancs d\u2019essai et runs CI de septembre 2026).")

# ================= DEDICACES =================
r.add_page()
r.h1(None, "D\u00e9dicaces")
r.p("J\u2019ai le plaisir de d\u00e9dier ce travail \u00e0 :")
r.bullets([
    "Mes tr\u00e8s chers parents, pour leur amour, leur patience et leurs encouragements sans faille.",
    "Mes s\u0153urs et mon fr\u00e8re, pour leur soutien et leur pr\u00e9sence rassurante.",
    "Mes amis proches qui ont toujours \u00e9t\u00e9 l\u00e0 pour me soutenir dans ce parcours.",
    "Mes grands-parents, dont les pri\u00e8res m\u2019accompagnent chaque jour.",
    "Mes oncles et tantes qui m\u2019ont toujours soutenue, chacun \u00e0 sa mani\u00e8re.",
    "Tous mes collaborateurs et coll\u00e8gues de stage pour leur soutien technique et moral.",
])
r.p("Je vous d\u00e9die cet humble travail, fruit d\u2019un long parcours.")

# ================= REMERCIEMENTS =================
r.h1(None, "Remerciements")
r.p("Au terme de ce travail, je tiens \u00e0 remercier toutes les personnes qui "
    "ont contribu\u00e9 \u00e0 la r\u00e9alisation de ce projet.")
r.bullets([
    "Je remercie sinc\u00e8rement Ahlem BENHADDOUD, mon encadrant, pour son accompagnement, ses conseils pr\u00e9cieux et sa disponibilit\u00e9 tout au long du stage.",
    "Mes parents pour leur soutien constant.",
    "Mes s\u0153urs et mon fr\u00e8re pour leur pr\u00e9sence rassurante et leurs encouragements.",
    "Mes amis proches, qui ont toujours \u00e9t\u00e9 l\u00e0 pour moi.",
    "Tous mes collaborateurs qui m\u2019ont soutenue tout au long de la r\u00e9alisation du projet, avec leur soutien moral et technique, leur bonne humeur et leur sympathie.",
    "Tous les enseignants qui ont particip\u00e9 \u00e0 mon \u00e9volution scientifique durant les ann\u00e9es pr\u00e9c\u00e9dentes.",
    "Toutes les personnes qui ont contribu\u00e9 \u00e0 l\u2019\u00e9laboration de ce travail. Il n\u2019est malheureusement pas possible de les citer toutes ici, mais elles se reconna\u00eetront : je leur adresse mes plus profonds remerciements.",
])

# ================= INTRO =================
r.add_page()
r.h1(None, "Introduction G\u00e9n\u00e9rale")
r.p("Les grands mod\u00e8les de langage (LLM) r\u00e9digent des r\u00e9ponses fluides, "
    "mais ils ont un d\u00e9faut majeur pour un usage professionnel : ils peuvent "
    "affirmer avec assurance des faits faux ou inv\u00e9rifiables. Dans un domaine "
    "r\u00e9glement\u00e9 comme la banque, une r\u00e9ponse sans source exploitable est "
    "inutilisable. La g\u00e9n\u00e9ration augment\u00e9e par r\u00e9cup\u00e9ration (RAG, "
    "Retrieval-Augmented Generation) apporte une r\u00e9ponse \u00e0 ce probl\u00e8me : "
    "le mod\u00e8le ne r\u00e9pond qu\u2019\u00e0 partir de passages r\u00e9ellement extraits "
    "d\u2019un corpus approuv\u00e9, et chaque affirmation est rattach\u00e9e \u00e0 une "
    "citation v\u00e9rifiable.")
r.p("Ce rapport pr\u00e9sente le travail men\u00e9 au sein de l\u2019entreprise "
    "Al Baraka Bank, dont l\u2019objectif est la conception et l\u2019\u00e9valuation d\u2019un "
    "syst\u00e8me RAG multilingue (arabe, fran\u00e7ais, anglais) appliqu\u00e9 \u00e0 des "
    "documents bancaires : lois, circulaires de la Banque Centrale de Tunisie et "
    "guides internes. Le syst\u00e8me, nomm\u00e9 RAGLab, couvre toute la cha\u00eene : "
    "ingestion de documents PDF/DOCX, d\u00e9coupage s\u00e9mantique, indexation "
    "vectorielle, recherche cross-lingue et g\u00e9n\u00e9ration de r\u00e9ponses cit\u00e9es "
    "avec refus contr\u00f4l\u00e9.")
r.p("Cette \u00e9tude inclut \u00e9galement la comparaison mesur\u00e9e de strat\u00e9gies de "
    "d\u00e9coupage, le banc d\u2019essai de mod\u00e8les de r\u00e9ponse gratuits, un juge de "
    "retrieval sans LLM et l\u2019industrialisation du service (microservice REST, "
    "image Docker portable, int\u00e9gration continue). Le syst\u00e8me livr\u00e9 pendant le stage repr\u00e9sente "
    "environ 66 000 lignes de code, de mesures et de documentation, dont 39 modules Python "
    "(environ 21 300 lignes), 6 jeux de questions et 7 pipelines d\u2019int\u00e9gration continue).")

# ================= CHAPITRE 1 =================
r.add_page()
r.h1("1", "Pr\u00e9sentation de l\u2019organisme d\u2019accueil et \u00e9tude pr\u00e9alable")
r.h2("1.1  Introduction")
r.p("L\u2019entreprise Al Baraka Bank est la premi\u00e8re banque islamique de Tunisie et du Maghreb. "
    "Elle offre une gamme de produits et de services conformes aux principes de la finance "
    "islamique, et investit dans l\u2019intelligence artificielle appliqu\u00e9e \u00e0 la "
    "recherche d\u2019information pour mieux exploiter sa documentation r\u00e9glementaire et m\u00e9tier.")
r.h2("1.2  Pr\u00e9sentation de l\u2019organisme")
r.h3("Al Baraka Bank")
r.p("Cr\u00e9\u00e9e le 15 juin 1983 sous la d\u00e9nomination Beit Ettamwil Saoudi Tounsi "
    "(Best Bank), Al Baraka Bank a pris son nom actuel en 2009 dans le cadre de "
    "l\u2019unification de l\u2019identit\u00e9 du groupe. C\u2019est une filiale du groupe "
    "bancaire international Al Baraka Banking Group (si\u00e8ge \u00e0 Bahre\u00efn), "
    "acteur de r\u00e9f\u00e9rence de la finance islamique. Son si\u00e8ge social est \u00e0 "
    "Tunis et son capital s\u2019\u00e9l\u00e8ve \u00e0 120 millions de dinars.")
r.h3("Technologies mobilis\u00e9es pour le projet")
r.table(
    ["\u00c9tape", "Composant", "Technologies principales"],
    [["Ingestion", "Parsing PDF/DOCX, normalisation", "Python, pypdf, XML natif"],
     ["D\u00e9coupage", "Chunking s\u00e9mantique", "tiktoken cl100k_base, r\u00e8gles m\u00e9tier"],
     ["Indexation", "Recherche vectorielle + lexicale", "ChromaDB (cosinus), BM25"],
     ["R\u00e9ponse", "G\u00e9n\u00e9ration cit\u00e9e", "API NVIDIA, passerelle xKiro (Qwen)"],
     ["Service", "API REST conteneuris\u00e9e", "FastAPI, Uvicorn, Docker"],
     ["Qualit\u00e9", "Tests et mesures", "unittest, int\u00e9gration continue (7 pipelines)"]],
    caption="Table 1 \u2014 Briques technologiques du laboratoire RAGLab.")
r.h2("1.3  \u00c9tude de l\u2019existant")
r.h3("Description de l\u2019existant")
r.p("Avant le d\u00e9but du projet, la documentation bancaire de r\u00e9f\u00e9rence (textes "
    "de loi, circulaires, guides internes) existait sous forme de fichiers "
    "dispers\u00e9s, principalement en arabe et en fran\u00e7ais. Toute question "
    "n\u00e9cessitait une lecture manuelle par un expert m\u00e9tier : la recherche par "
    "mots-cl\u00e9s ne franchit pas la barri\u00e8re de la langue et les r\u00e9ponses "
    "produites n\u2019\u00e9taient rattach\u00e9es \u00e0 aucune citation v\u00e9rifiable "
    "automatiquement.")
r.h3("Workflow des donn\u00e9es avant projet")
r.figure("workflow_before.png",
         "Figure 1 \u2014 Flux documentaire avant le projet RAGLab.",
         "Interpr\u00e9tation : les documents \u00e9taient consult\u00e9s manuellement, sans "
         "recherche s\u00e9mantique ni tra\u00e7abilit\u00e9 \u2014 d\u00e9pendance \u00e0 l\u2019expert, "
         "absence de r\u00e9ponses translingues et de citations contr\u00f4lables.")
r.h3("Organisation et livrables du stage")
r.p("Le stage a produit un laboratoire RAG complet et document\u00e9 : 39 modules "
    "Python (environ 21 300 lignes), 4 documents bancaires r\u00e9els d\u2019\u00e9valuation, "
    "6 jeux de questions de r\u00e9f\u00e9rence, 7 pipelines d\u2019int\u00e9gration continue et "
    "une image Docker de production. Ce contenu, et les rapports de mesure qu\u2019il "
    "contient, sert de mati\u00e8re premi\u00e8re aux chapitres 2 et 3.")
r.h3("Planning pr\u00e9visionnel du stage")
r.figure("gantt_stage.png",
         "Figure 2 \u2014 Diagramme de Gantt du stage (7 semaines, S1 \u00e0 S7).",
         "Interpr\u00e9tation : planification des activit\u00e9s \u2014 corpus, ingestion, "
         "embeddings, g\u00e9n\u00e9ration cit\u00e9e, \u00e9valuation, comparaison de mod\u00e8les et "
         "industrialisation, du cadrage \u00e0 la livraison du service conteneuris\u00e9.")
r.h2("1.4  M\u00e9thodologie adopt\u00e9e")
r.numbered([
    "Cadrage et corpus : s\u00e9lection de documents bancaires r\u00e9els, avec jeux de questions gel\u00e9s par langue.",
    "Ingestion reproductible : parsing, normalisation arabe, d\u00e9coupage versionn\u00e9 (empreinte v4) et cache d\u2019embeddings resumable.",
    "Recherche contrainte et g\u00e9n\u00e9ration cit\u00e9e : top-5 cosinus sur requ\u00eate d\u2019origine, claims JSON \u00e0 citations mot-\u00e0-mot, refus en cas d\u2019\u00e9chec.",
    "Mesure syst\u00e9matique : harnais A/B, juge de retrieval sans LLM, comparaison de mod\u00e8les gratuits sur rubriques gel\u00e9es, CI qui rejoue les mesures.",
    "Industrialisation : microservice REST, image Docker portable, documentation de contrat pour les \u00e9quipes clientes.",
])
r.h2("1.5  Conclusion")
r.p("Cette premi\u00e8re \u00e9tude permet de cerner les besoins de l\u2019entreprise, de "
    "comprendre la structure du corpus bancaire trilingue et de d\u00e9finir les "
    "objectifs du projet : une r\u00e9ponse cit\u00e9e et refusable plut\u00f4t qu\u2019une "
    "r\u00e9ponse fluide mais inv\u00e9rifiable, tout en identifiant les contraintes "
    "(qualit\u00e9 du d\u00e9coupage, co\u00fbt des appels mod\u00e8les, limites du gratuit).")

# ================= CHAPITRE 2 =================
r.add_page()
r.h1("2", "Conception de la solution")
r.h2("2.1  Introduction")
r.p("La conception repose sur la d\u00e9finition des besoins, l\u2019architecture "
    "fonctionnelle et le choix du pipeline RAG. L\u2019accent a \u00e9t\u00e9 mis sur la "
    "reproductibilit\u00e9 (graines, caches, empreintes), la mesurabilit\u00e9 (jeux "
    "gel\u00e9s, harnais rejouables) et la s\u00e9curit\u00e9 des r\u00e9ponses (citations "
    "exig\u00e9es, refus local).")
r.h2("2.2  Environnement de d\u00e9veloppement")
r.bullets([
    "Python 3.11, sans framework d\u2019orchestration (ni LangChain ni LlamaIndex) et sans SDK propri\u00e9taire (HTTPS standard).",
    "ChromaDB persistant local (similarit\u00e9 cosinus) + BM25 : recherche vectorielle et lexicale sans base externe.",
    "FastAPI + Uvicorn : microservice REST de 25 routes, documentation OpenAPI \u00e0 /docs.",
    "tiktoken (cl100k_base) et pypdf : tokenisation de r\u00e9f\u00e9rence et lecture des PDF.",
    "Int\u00e9gration continue : 7 pipelines (tests, images Docker, mod\u00e8les gratuits, harnais d\u2019\u00e9valuation, juge retrieval, test r\u00e9el, catalogues).",
])
r.h3("Choix technologiques")
r.table(
    ["Besoin", "Choix retenu", "Justification"],
    [["Embeddings", "nvidia/nemotron-3-embed-1b, natif 2048 d",
      "Espace unique et stable ; le juge retrieval mesure 62,8 % sur requ\u00eates sans mot commun contre 0 % au lexical."],
     ["R\u00e9ponses", "qwen/qwen3.8-max:free via xKiro",
      "Meilleur score mesur\u00e9 (13/14 dev, 18/18 held-out) avec contr\u00f4le de gratuit\u00e9 en direct avant chaque appel."],
     ["Retrieval", "Top-5 cosinus, requ\u00eate d\u2019origine",
      "La traduction de requ\u00eate n\u2019am\u00e9liorait pas toutes les questions et d\u00e9gradait le top-1 held-out."],
     ["Cadre logiciel", "Aucun framework RAG",
      "D\u00e9pendances minimales, comportement explicite, image Docker l\u00e9g\u00e8re et auditable."]],
    caption="Table 2 \u2014 Justification des choix technologiques.")
r.h2("2.3  Besoins fonctionnels et non fonctionnels")
r.h3("Besoins fonctionnels")
r.bullets([
    "Ing\u00e9rer des documents PDF, DOCX, TXT et Markdown avec normalisation multilingue.",
    "Indexer le corpus en chunks adressables, avec une collection par couple (espace d\u2019embedding, mode de chunking).",
    "Rechercher les passages pertinents pour une question en arabe, fran\u00e7ais ou anglais, y compris translingue.",
    "G\u00e9n\u00e9rer des r\u00e9ponses cit\u00e9es ou refuser explicitement (corpus silencieux, donn\u00e9es priv\u00e9es, donn\u00e9es temps r\u00e9el).",
    "Exposer toutes les fonctions par CLI, console interactive et API REST document\u00e9e.",
])
r.h3("Besoins non fonctionnels")
r.bullets([
    "Tra\u00e7abilit\u00e9 : chaque affirmation livr\u00e9e porte des citations mot-\u00e0-mot extraites des chunks ; un nombre non cit\u00e9 entra\u00eene un refus.",
    "Reproductibilit\u00e9 : tokenizer, caches, empreintes et jeux gel\u00e9s garantissent des mesures rejouables en CI.",
    "S\u00e9curit\u00e9 : pas de r\u00e9ponse sur donn\u00e9es priv\u00e9es ou temps r\u00e9el (garde-fous locaux), masquage des cl\u00e9s, validation des vecteurs.",
    "Portabilit\u00e9 : image Docker multi-stage ex\u00e9cutable sans clone ni pip, y compris hors-ligne apr\u00e8s import.",
    "Maintenabilit\u00e9 : un noyau partag\u00e9 par les trois frontaux, 133 contr\u00f4les hors-ligne, documentation de contrat .",
])
r.h2("2.4  Architecture fonctionnelle")
r.p("L\u2019architecture se compose de quatre \u00e9l\u00e9ments principaux :")
r.table(
    ["Donn\u00e9es en entr\u00e9e", "Traitement", "Mod\u00e8les", "R\u00e9sultats"],
    [["Documents bancaires ar/fr/en",
      "Parsing, normalisation, chunking, embeddings",
      "Nemotron (recherche) + Qwen (r\u00e9daction)",
      "R\u00e9ponses cit\u00e9es ou refus motiv\u00e9s"]],
    caption="Table 3 \u2014 Les quatre blocs de l\u2019architecture.")
r.h3("Description des composants")
r.numbered([
    "Collecte : acquisition des documents (d\u00e9p\u00f4ts le corpus, t\u00e9l\u00e9versement POST /documents avec versions et purge cibl\u00e9e).",
    "Transformation : nettoyage, r\u00e9paration du texte arabe, d\u00e9coupage (restructure/size/manual) et plongements vectoriels.",
    "Application : recherche top-5 et ex\u00e9cution du mod\u00e8le de r\u00e9ponse derri\u00e8re le portillon de citations.",
    "G\u00e9n\u00e9ration : production de la r\u00e9ponse finale cit\u00e9e, ou d\u2019un refus diagnostiquable avec les extraits fournis.",
])
r.h2("2.5  Pipeline retenu : retrieval Nemotron + r\u00e9ponses Qwen")
r.h3("Description et justification")
r.p("Le pipeline est gel\u00e9 par le module pipeline_policy.py : toute substitution "
    "de mod\u00e8le sur les chemins mesur\u00e9s l\u00e8ve une erreur au lieu d\u2019un appel "
    "silencieux vers un autre mod\u00e8le. Le tableau 4 r\u00e9sume ses param\u00e8tres.")
r.table(
    ["Param\u00e8tre", "Description"],
    [["embedding_model", "nvidia/nemotron-3-embed-1b, natif 2048 dimensions"],
     ["answer_model", "qwen/qwen3.8-max:free via xKiro, prompt grounded-v1"],
     ["top_k", "5 chunks (requ\u00eate d\u2019origine, similarit\u00e9 cosinus)"],
     ["chunking", "restructure par d\u00e9faut ; 220/40, tokenizer cl100k_base"],
     ["batch_size", "32 textes par appel d\u2019embeddings, retries + bissection"],
     ["citation_gate", "Citations mot-\u00e0-mot exig\u00e9es, nombres pr\u00e9sents dans les citations"],
     ["free_price_check", "V\u00e9rification de gratuit\u00e9 en direct avant chaque appel Qwen"]],
    caption="Table 4 \u2014 Param\u00e8tres principaux du pipeline RAG.")
r.h2("2.6  Diagramme de l\u2019architecture")
r.figure("architecture_diagram.png",
         "Figure 3 \u2014 Architecture g\u00e9n\u00e9rale de la solution.",
         "Interpr\u00e9tation : les trois frontaux partagent un m\u00eame noyau ; le service "
         "REST expose 25 routes ; le stockage et les garde-fous restent locaux ; seuls "
         "les embeddings et la r\u00e9daction font appel \u00e0 des mod\u00e8les externes.")
r.h3("Pipeline de traitement des donn\u00e9es")
r.figure("data_pipeline.png",
         "Figure 4 \u2014 Pipeline d\u2019ingestion et d\u2019interrogation.",
         "Interpr\u00e9tation : l\u2019ingestion (hors-ligne) construit un index d\u00e9crit par ses "
         "empreintes ; la requ\u00eate (en ligne) suit un chemin contraint qui se termine "
         "par le portillon de citations \u2014 r\u00e9ponse cit\u00e9e ou refus.")
r.h3("Champs d\u2019un chunk et m\u00e9tadonn\u00e9es d\u2019index")
r.table(
    ["Champ", "Description"],
    [["index / source", "Position du chunk et document d\u2019origine (id source::chunk_NNNN)"],
     ["heading / language", "Titre de section et langue d\u00e9tect\u00e9e (ar/fr/en)"],
     ["token_count", "Taille mesur\u00e9e au tokenizer cl100k_base"],
     ["chunk_fp", "Empreinte du d\u00e9coupage (mode, tailles, tokenizer) \u2014 v4"],
     ["embedding_fp", "Empreinte de l\u2019espace d\u2019embeddings (mod\u00e8le, t\u00e2che, dimensions)"]],
    caption="Table 5 \u2014 Champs index\u00e9s pour chaque chunk.")
r.h2("2.7  Conclusion")
r.p("La conception fige un pipeline \u00e9troit mais justifi\u00e9 par la mesure : un seul "
    "mod\u00e8le d\u2019embeddings, un seul mod\u00e8le de r\u00e9ponses, une retrieval "
    "top-5 sur requ\u00eate d\u2019origine et un portillon de citations qui interdit "
    "l\u2019affirmation non sourc\u00e9e. Le chapitre suivant d\u00e9crit la r\u00e9alisation "
    "et les r\u00e9sultats obtenus.")

# ================= CHAPITRE 3 =================
r.add_page()
r.h1("3", "R\u00e9alisation, traitements et r\u00e9sultats")
r.h2("3.1  Chargement et exploration du corpus")
r.code("python main.py inspect --data-dir ./corpus\n"
       "python main.py ingest --reset --data-dir ./corpus\n"
       "python main.py query \"What is Murabaha?\" --query-lang en\n"
       "python main.py answer \"What is Murabaha?\" --query-lang en")
r.table(
    ["Document", "Format", "Langue", "R\u00f4le"],
    [["Loi_2016-48", "PDF (36 p. au total)", "arabe", "Corpus r\u00e9el d\u2019\u00e9valuation"],
     ["Circulaire_BCT_2019-08", "PDF", "arabe", "Corpus r\u00e9el d\u2019\u00e9valuation"],
     ["Guide_Interne_Operations_Bancaires_Islamiques", "DOCX", "arabe", "Corpus r\u00e9el d\u2019\u00e9valuation"],
     ["Madkhal_Sayrafa_Islamiya", "DOCX", "arabe", "Corpus r\u00e9el d\u2019\u00e9valuation"]],
    caption="Table 6 \u2014 Composition du corpus.")
r.table(
    ["M\u00e9trique", "Valeur"],
    [["Chunks (220/40, cl100k_base)", "836"],
     ["Pages PDF", "36"],
     ["Empreinte du manifeste", "807785db\u2026 (SHA-256)"],
     ["Couverture des preuves attendues", "16/16 (100 %)"],
     ["Chunks de m\u00e9tadonn\u00e9es de titres", "0"],
     ["Dimensions d\u2019embeddings", "2048 (natives)"]],
    caption="Table 7 \u2014 Statistiques descriptives du corpus index\u00e9.")
r.h2("3.2  Analyse d\u00e9taill\u00e9e et diagnostics")
r.h3("Couverture des preuves et qualit\u00e9 du d\u00e9coupage")
r.table(
    ["Diagnostic", "R\u00e9sultat"],
    [["Phrases-preuves joignables dans un chunk", "16/16 \u2014 tout est atteignable"],
     ["Chunks parasites (titres seuls)", "0"],
     ["Unit\u00e9s sources tenant dans un chunk de 220", "5/230 (2,2 %) \u2014 le chunker est le plafond"],
     ["Plafond honn\u00eate du top-5 joint", "12,2 % de tron\u00e7ons complets"]],
    caption="Table 8 \u2014 Diagnostics de couverture (CI et juge retrieval).")
r.p("Le diagnostic principal est structurel : seules 5 des 230 unit\u00e9s sources "
    "audit\u00e9es tiennent dans un chunk de 220 tokens : le chunker, et non le "
    "ranker, borne la qualit\u00e9 du syst\u00e8me.")
r.h2("3.3  Visualisation des distributions")
r.figure("score_distributions.png",
         "Figure 5 \u2014 Distributions des scores mesur\u00e9s.",
         "Interpr\u00e9tation : (a) restructure bat size sur hit@1 global et verbatim ; "
         "(b) Qwen domine la rubrique dev ; (c) les embeddings seuls r\u00e9solvent les "
         "requ\u00eates sans mot commun avec la preuve.")
r.h3("Matrice des m\u00e9triques retrieval")
r.figure("metrics_heatmap.png",
         "Figure 6 \u2014 Lexical vs s\u00e9mantique (run CI 34005544576).",
         "Interpr\u00e9tation : les embeddings surpassent le lexical sur les quatre axes, "
         "avec un \u00e9cart maximal sur les requ\u00eates s\u00e9mantiques (62,8 % contre 0 %).")
r.h2("3.4  Comparaison des mod\u00e8les et \u00e9valuation")
r.h3("Banc d\u2019essai des mod\u00e8les gratuits (d\u00e9veloppement, 16 cas)")
r.table(
    ["SKU xKiro", "Rubrique", "Validation", "Refus", "Temps moy."],
    [["qwen/qwen3.8-max:free", "13/14 (92,9 %)", "16/16", "2/2", "4,68 s"],
     ["minimax/minimax-m3:free", "11/14 (78,6 %)", "15/16", "2/2", "6,07 s"],
     ["mistralai/mistral-small-2603", "6/14 (42,9 %)", "11/16", "2/2", "8,27 s"],
     ["deepseek/deepseek-v4-flash", "7/14 (50,0 %)", "13/16", "2/2", "7,46 s"]],
    caption="Table 9 \u2014 Comparaison fra\u00eeche des 4 candidats (5 sept. 2026).")
r.h3("Qwen : held-out et s\u00e9curit\u00e9")
r.table(
    ["Contr\u00f4le", "R\u00e9sultat"],
    [["Rubrique r\u00e9ponses", "18/18"],
     ["Faux refus", "0/18"],
     ["Validation citations", "18/18"],
     ["Validation globale", "27/27"],
     ["Refus priv\u00e9/temps r\u00e9el", "9/9 (garde-fous locaux)"],
     ["Injections de source", "3/3"],
     ["Temps client moyen", "5,25 s (18 appels)"]],
    caption="Table 10 \u2014 Qwen held-out : 18/18 r\u00e9ponses, 9/9 refus, 3/3 injections.")
r.h3("A/B chunking : restructure contre fen\u00eatre fixe")
r.table(
    ["Mesure", "restructure", "size 220/40"],
    [["hit@1 global (50 q., BM25)", "47 %", "40 %"],
     ["hit@1 verbatim", "75 %", "42 %"],
     ["hit@1 mod\u00e8les r\u00e9els", "73-76 %", "67 %"],
     ["Gain paraphrase / questions FR", "+15 pp / +27 pp", "r\u00e9f\u00e9rence"],
     ["Rappel top-20", "inf\u00e9rieur", "parfait (100 %)"]],
    caption="Table 11 \u2014 A/B chunking : harnais-50 et test sur mod\u00e8les r\u00e9els (run 34144251576).")
r.h3("Juge de retrieval sans LLM (1407 questions)")
r.table(
    ["M\u00e9trique", "Lexical (BM25)", "Embeddings (NVIDIA)"],
    [["Tron\u00e7on entier dans un chunk du top-5", "2,7 %", "10,0 %"],
     ["Tron\u00e7on \u00e0 80 %", "9,5 %", "37,3 %"],
     ["Requ\u00eates s\u00e9mantiques (145)", "0,0 %", "62,8 %"],
     ["AUC d\u2019abstention", "0,558", "0,679"]],
    caption="Table 12 \u2014 Juge retrieval : le cross-lingue sans aucun appel g\u00e9n\u00e9ratif.")
r.h2("3.5  Exemple de r\u00e9ponse pour une question")
r.table(
    ["Question", "R\u00e9ponse du syst\u00e8me", "Statut"],
    [["What is Murabaha? (en)",
      "1 claim + 1 citation mot-\u00e0-mot du guide ; montant repris de la citation.",
      "Livr\u00e9e"],
     ["D\u00e9finition du Salam ? (fr)",
      "1 claim + 1 citation ; formulation valide mais hors rubrique \u00e9troite.",
      "Livr\u00e9e (rubrique 13/14)"],
     ["Solde de mon compte ? (fr)",
      "Refus local imm\u00e9diat, z\u00e9ro appel mod\u00e8le (question priv\u00e9e).",
      "Refus\u00e9e"]],
    caption="Table 13 \u2014 Exemples illustratifs : deux r\u00e9ponses cit\u00e9es, un refus local.")
r.h3("Exemple de reclassement des strat\u00e9gies de chunking")
r.table(
    ["Strat\u00e9gie", "Chunks", "M\u00e9diane", "Citation enti\u00e8re", "R@1 / R@5"],
    [["size 220/40", "310", "186", "45,4 %", "47,2 % / 70,9 %"],
     ["size 640/40 (gel\u00e9)", "127", "378", "58,7 %", "63,8 % / 95,0 %"],
     ["size 420/40", "173", "354", "62,3 %", "60,3 % / 89,4 %"],
     ["manuelle souple", "107", "557", "59,0 %", "59,2 % / 86,2 %"],
     ["manuelle stricte", "211", "153", "59,0 %", "55,3 % / 83,0 %"]],
    caption="Table 14 \u2014 Cinq chunkings mesur\u00e9s hors-ligne sur 282 questions.")
r.h2("3.6  Grand harnais : \u00e9tat d\u2019avancement")
r.p("Le grand harnais vise 900 familles de questions plus 100 adversariales. "
    "Sont accept\u00e9es 469 familles (1 407 paires) de cat\u00e9gorie supported, "
    "r\u00e9parties en 9 shards ; les 200 familles hors-sujet et 50 \u00e0 preuve "
    "insuffisante restent \u00e0 produire. L\u2019audit crois\u00e9 ind\u00e9pendant est de "
    "0/469 (second passage par le m\u00eame mod\u00e8le). Aucun score de comparaison "
    "de r\u00e9ponses n\u2019est gel\u00e9 : ce harnais est une infrastructure, pas un r\u00e9sultat.")
r.h2("3.7  Conclusion")
r.p("La r\u00e9alisation livre un laboratoire complet et mesur\u00e9 : corpus de 836 "
    "chunks \u00e0 couverture totale, A/B chunking gagn\u00e9 par restructure, Qwen "
    "s\u00e9lectionn\u00e9 sur banc d\u2019essai frais puis confirm\u00e9 en held-out (18/18), "
    "preuve cross-lingue apport\u00e9e sans LLM (62,8 % contre 0 %) et 133 "
    "contr\u00f4les hors-ligne au vert. Chaque chiffre est rattach\u00e9 \u00e0 son run, "
    "son git hash ou son JSON de preuve.")

# ================= CHAPITRE 4 =================
r.add_page()
r.h1("4", "Conclusion g\u00e9n\u00e9rale et perspectives")
r.h2("Conclusion")
r.p("Le stage a permis de concevoir et d\u2019\u00e9valuer un syst\u00e8me RAG multilingue "
    "complet pour documents bancaires : ingestion de 4 documents r\u00e9els en 836 "
    "chunks, indexation par embeddings natifs 2048 d, g\u00e9n\u00e9ration cit\u00e9e avec "
    "portillon de citations et refus contr\u00f4l\u00e9s, le tout expos\u00e9 en CLI, "
    "console et microservice REST conteneuris\u00e9. Les mesures montrent que le "
    "d\u00e9coupage restructur\u00e9 am\u00e9liore le hit@1 de 7 points (jusqu\u2019\u00e0 9 sur "
    "mod\u00e8les r\u00e9els), que Qwen 3.8 via xKiro atteint 92,9 % en d\u00e9veloppement "
    "et 100 % en held-out, et que les embeddings r\u00e9solvent 62,8 % des requ\u00eates "
    "sans mot commun contre 0 % au lexical. L\u2019approche a aussi identifi\u00e9 le "
    "vrai plafond du syst\u00e8me : le chunker, seules 5 unit\u00e9s sources sur 230 "
    "tenant dans un chunk de 220 tokens.")
r.h2("Limites et recommandations")
r.bullets([
    "Preuves \u00e9parpill\u00e9es par le chunking fin \u2192 passer le d\u00e9faut applicatif \u00e0 420-640 tokens et g\u00e9n\u00e9raliser l\u2019enrichissement par voisins.",
    "Serving gratuit non garanti (quotas, routage, identit\u00e9 du mod\u00e8le) \u2192 contrat fournisseur approuv\u00e9, capacit\u00e9 mesur\u00e9e et astreinte avant production.",
    "Jeux d\u2019\u00e9valuation petits et corr\u00e9l\u00e9s, audit non ind\u00e9pendant (0/469) \u2192 finir les 431 familles, ajouter les 250 n\u00e9gatives, imposer l\u2019audit crois\u00e9.",
    "Pas de garantie bancaire/juridique ni de preuve de s\u00e9curit\u00e9 \u2192 revue linguistique-m\u00e9tier, tests d\u2019injection \u00e9largis, pilote supervis\u00e9 uniquement.",
])
r.h2("Perspectives")
r.p("Trois chantiers prolongent naturellement ce travail : l\u2019enrichissement du "
    "contexte (voisins, seuils d\u2019abstention calibr\u00e9s), l\u2019ach\u00e8vement du grand "
    "harnais trilingue avec gels de versions, et la construction du dossier de "
    "production (fournisseur approuv\u00e9, revue bancaire ind\u00e9pendante, "
    "m\u00e9triques de disponibilit\u00e9). Le laboratoire est pr\u00eat \u00e0 les accueillir : "
    "chaque piste s\u2019y exprime comme une exp\u00e9rience mesur\u00e9e, compar\u00e9e et "
    "rejouable en CI.")
r.output(os.path.join(HERE, "rapport_stage.pdf"))
print("wrote rapport_stage.pdf")
