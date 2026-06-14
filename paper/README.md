# Paper directory

LaTeX scaffold for the tinyCPG paper. Target journal:
**PLOS Computational Biology** (backup: Frontiers in Computational
Neuroscience). See `novelty_vs_zhang2021.md` for the positioning
argument.

## Layout

```
paper/
├── main.tex                    ← top-level LaTeX file
├── refs.bib                    ← bibliography (BibTeX, ~25 starter refs)
├── Makefile                    ← build helper (make pdf | quick | clean | figures)
├── README.md                   ← this file
│
├── sections/                   ← one .tex per section, \input'ed from main.tex
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── methods.tex
│   ├── results.tex
│   └── discussion.tex
│
├── figures/                    ← .png files referenced by \includegraphics
│   ├── fig2_main.png           ← from cpg_combined_summary.py (--layout landscape)
│   ├── fig3_ablation.png       ← from cpg_ablation_figure.py
│   └── fig4_speed.png          ← from cpg_speed_figure.py
│   └── fig1_schematic.png      ← MISSING — the architecture schematic (manual draw)
│
├── params.{md,tex,csv}         ← parameter table (auto-generated)
├── methods_skeleton.md         ← prose-writing notes for Methods (long-form)
└── novelty_vs_zhang2021.md     ← positioning vs. the Rybak/Zhang lineage
```

## Build

```bash
cd paper
make pdf            # full latexmk pass with bibliography
make quick          # single pdflatex pass (faster, no bibtex)
make figures        # regenerate fig2/fig3/fig4 + params.tex
make clean          # remove latex aux files
```

By default `make figures` uses `/opt/homebrew/bin/python3`; override with
`make PY=python3.11 figures`.

## Writing status

| Section | Status | Notes |
|---|---|---|
| Abstract | TODO bullet outline | write last |
| Introduction | TODO 6-paragraph outline with citations | needs prose |
| Methods | TODO subsection scaffolding with equations | longest TODO list |
| Results | TODO anchored on 3 figures + 2 metric tables | numbers already filled in |
| Discussion | TODO 6-subsection outline | anchor on `novelty_vs_zhang2021.md` |

Every `\todo{...}` macro will render as red bold "[TODO: …]" in the
compiled PDF — search the document for "TODO" to find every open item.

## Submission switch

When ready to submit, replace the generic preamble in `main.tex`:

- **PLOS Computational Biology:**
  `\documentclass{plos2015}` (from <https://journals.plos.org/ploscompbiol/s/latex>)
  and `\bibliographystyle{plos2015}`.

- **Frontiers in Computational Neuroscience:**
  Use the Frontiers Word template instead — they accept LaTeX but require
  a final Word submission via their portal. Easiest path: produce a
  clean PDF here, then paste into the Frontiers Word template.
