# Rapport de stage — RAGLab

Rapport d'immersion en entreprise généré à partir du contenu de la branche
`arena/01a0ec17-raglab` (commit `f41e557`), en suivant la structure du rapport
précédent fourni (page de garde, dédicaces, remerciements, introduction,
4 chapitres, conclusion/perspectives).

## Contenu

| Fichier | Rôle |
|---|---|
| `rapport_stage.tex` | **Source de vérité** : rapport complet, même structure que le modèle |
| `rapport_stage.pdf` | PDF généré (miroir du `.tex`, voir § ci-dessous) |
| `figures/*.png` | 6 figures générées par `make_figures.py` |
| `make_figures.py` | Génère les figures avec matplotlib (nombres issus du dépôt) |
| `make_pdf.py` | Régénère le PDF avec fpdf2 |

## Champs à remplir (obligatoire avant rendu)

Le PDF actuel contient des placeholders visibles. Dans `rapport_stage.tex`,
renseignez le bloc `\newcommand` en tête de fichier :

- `\EtudiantNom`, `\EcoleNom`, `\EntrepriseNom`
- `\EncadrantNom`, `\StagePeriode`, `\AnneeUniv`
- Déposez `esprit.png` (logo école) à côté du `.tex` (sinon un cadre
  réservé s'affiche) — ou adaptez la page de garde.
- Adaptez le §1.2 (année de fondation, secteur) à l'entreprise réelle.

Puis régénérez le PDF (voir ci-dessous) ou recompilez le `.tex`.

## Compiler le `.tex` (mise en page canonique)

Le sandbox ne dispose pas de TeXLive (téléchargement bloqué), donc :

- **Option recommandée** : importez `rapport_stage.tex` + le dossier
  `figures/` dans **Overleaf** (ou TeXLive local : `pdflatex` × 2).
- Le `.tex` a été vérifié : `\begin{document}`/`\end{document}` uniques,
  accolades équilibrées, `%` et `&` échappés, toutes les figures présentes.

## Régénérer le PDF fourni

```bash
cd rapport
pip install fpdf2 matplotlib
python3 make_figures.py   # figures/*.png
python3 make_pdf.py       # rapport_stage.pdf (19 pages)
```

## D'où viennent les chiffres (traçabilité)

| Nombre | Source dans le dépôt |
|---|---|
| 159 fichiers, ~66 000 lignes, 39 modules, ~21 300 lignes | `git show --stat HEAD` |
| Corpus 4 docs, 836 chunks, 36 p. PDF, `807785db…` | `raglab/README.md`, `raglab/NVIDIA_REPORT.md` |
| A/B chunking 47 vs 40 %, verbatim 75 vs 42 % | `README.md` (harness50) |
| Real-test 73–76 vs 67 %, +15 pp, +27 pp, run 34144251576 | `README.md`, `AGENTS.md` §7 |
| Free models 13/14, 16/16, 2/2, 4,68 s ; held-out 18/18, 27/27, 9/9, 3/3, 5,25 s | `raglab/FREE_MODELS_REPORT.md` |
| Juge retrieval 2,7/10,0 %, 9,5/37,3 %, 0/62,8 %, AUC 0,558/0,679, run 34005544576 | `raglab/HARD_HARNESS_STATUS.md` |
| 5 chunkings (310/127/173/107/211 chunks, R@1/R@5) | `raglab/README.md` (`chunk_maps.py measure`) |
| 469 familles / 1 407 paires, 9 shards, audit 0/469 | `raglab/HARD_HARNESS_STATUS.md`, `benchmarks/…/manifest.json` |
| 133 contrôles hors-ligne, run 33979521271 | `raglab/READINESS.md` |
| 25 routes, `ghcr.io/seif-bkh/raglab-service` | `raglab/service.py`, `raglab/PROD_IMAGE.md` |
