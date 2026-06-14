# Paper build Makefile.
# Targets:
#   make pdf      → compile main.tex with bibtex (full bibliography pass)
#   make quick    → single pdflatex pass (no bibtex, useful when iterating prose)
#   make clean    → remove all latex aux files
#   make figures  → re-run all Python plotting scripts (regenerates figures/*.png)

LATEXMK    = latexmk -pdf -interaction=nonstopmode
PDFLATEX   = pdflatex -interaction=nonstopmode
BIBTEX     = bibtex
MAIN       = main

.PHONY: pdf quick clean figures all

all: figures pdf

pdf:
	$(LATEXMK) $(MAIN).tex

quick:
	$(PDFLATEX) $(MAIN).tex

clean:
	latexmk -C
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.synctex.gz sections/*.aux

# Regenerate the three main paper figures from the latest HDF5 results.
# Assumes:
#   - cpg_bursting_paced_120s_*.h5 in repo root for the main figure
#   - cpg_ablate_*.h5             in repo root for the ablation figure
#   - cpg_speed_period*ms_*.h5    in repo root for the speed figure
PY ?= /opt/homebrew/bin/python3

figures:
	$(PY) ../cpg_combined_summary.py \
		--files ../cpg_bursting_paced_120s_*.h5 \
		--out figures/fig2_main.png \
		--representative-idx 4 --layout landscape
	$(PY) ../cpg_ablation_figure.py \
		--indir .. --out figures/fig3_ablation.png
	$(PY) ../cpg_speed_figure.py \
		--indir .. --out figures/fig4_speed.png
	$(PY) ../cpg_param_table.py \
		--src ../cpg_2legs_fast.py --format latex --out params.tex
