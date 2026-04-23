# Role Adherence

Paper proposing a metric for evaluating role consistency in multi-turn conversational AI systems, published as part of the Gaussia evaluation framework.

## Structure

```
role-adherence/
├── role-adherence-en.tex   # LaTeX source
├── role-adherence-en.pdf   # Compiled paper
├── refs.bib                # Bibliography
├── figures/                # Generated figures (PDFs)
└── experiments/            # Benchmark dataset and scoring notebooks
```

## Compiling the paper

```bash
pdflatex role-adherence-en.tex
bibtex role-adherence-en
pdflatex role-adherence-en.tex
pdflatex role-adherence-en.tex
```
