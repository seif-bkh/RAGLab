# Dossier « rapport de stage » — RAGLab

Ce dossier contient le **schéma d'architecture** (à lire avant le rapport) et le **rapport de
stage** complet, en trois formats. Tout est autonome : aucun identifiant de compte, aucune URL de
dépôt, aucun nom d'utilisateur n'y figure.

## Contenu

| Fichier | Rôle |
|---|---|
| `architecture_schema.svg` | **Schéma d'architecture** (vectoriel, qualité imprimable) — générations 1 et 2, le pivot, et l'encadré **LAST UPDATE** |
| `architecture_schema.png` | Le même schéma en image (3000 × 2780 px) |
| `ARCHITECTURE_SCHEMA.md` | Le schéma en texte + diagrammes Mermaid détaillés (couches, cycle d'une requête) et la provenance de chaque affirmation |
| `timeline.png` / `timeline.svg` | Chronologie du développement (369 commits, 19 journées de travail ; dernière journée mise en évidence) |
| `INTERNSHIP_REPORT.md` | **Rapport de stage — source Markdown** (français, avec *executive summary* anglais). C'est le fichier à éditer. |
| `rapport_stage.tex` | Le rapport en LaTeX (structure académique : page de garde, dédicaces, remerciements, résumé, chapitres, annexes) — à compiler sur Overleaf avec `pdflatex` (2 passes) |
| `rapport_stage.pdf` | Le rapport en PDF (26 pages) |
| `make_pdf.py` | Régénère `rapport_stage.pdf` depuis `INTERNSHIP_REPORT.md` (`pip install fpdf2 pillow`) |

## Remplacement des champs (Ctrl+H)

Un **seul** jeton est à remplacer :

| Jeton | Remplacement |
|---|---|
| `[[INTERN_NAME]]` | Nom et prénom de l'étudiant(e) |

Il apparaît dans `INTERNSHIP_REPORT.md` (page de garde et signature finale), `rapport_stage.tex`
(`\newcommand{\EtudiantNom}`) et `rapport_stage.pdf` (2 occurrences).

Les autres champs sont déjà renseignés — entreprise **Al Baraka Bank**, période **1er août →
1er octobre 2026**, encadrante **Ahlem BENHADDOUD**, école, année universitaire **2025–2026** — et
se modifient directement dans `INTERNSHIP_REPORT.md` et `rapport_stage.tex` (bloc `\newcommand`
en tête de fichier pour le LaTeX). Un seul complément est signalé par `TODO-ENTREPRISE` :
les éléments factuels d'entreprise (chapitre 1).

## Régénérer les livrables

```bash
# figures (schéma + chronologie) — les scripts ont servi à produire les fichiers fournis
python3 make_figures.py        # si vous modifiez les données du schéma

# PDF du rapport (équivalent fpdf2 du .tex)
pip install fpdf2 pillow
python3 make_pdf.py            # écrit rapport_stage.pdf
```

Le `.tex` se compile sur Overleaf (`pdflatex` × 2). Les termes arabes y sont translittérés pour
que la compilation ne demande aucune police arabe ; `INTERNSHIP_REPORT.md` et le PDF conservent
l'écriture arabe (le PDF joint la translittère dans le corps de texte, la police embarquée DejaVu
n'ayant pas de glyphes arabes).

## Vérifications faites avant livraison

- Aucune occurrence d'identifiant de compte, d'URL de dépôt ou de nom d'utilisateur dans les
  fichiers livrés (`grep` sur l'ensemble du dossier).
- Chiffres du rapport tracés un à un vers les documents internes du projet (annexe B du rapport) ;
  les comptages de commits (387 au total, 369 hors fusions, 15 branches) et les statistiques de
  code sont issus de l'historique Git à l'état final.
- Le PDF a été régénéré après la dernière correction de chiffres ; aucun marqueur Markdown résiduel
  (`**`, backticks) ne subsiste dans le PDF.
