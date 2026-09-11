# Accountability

Paper proposing two metrics for whether a tool-using assistant respected human
authority over its actions: **OversightCompliance** (actions that ran without a valid
approval) and **ActionDisclosure** (what the assistant ran but never mentioned, and
what it said that the trace refutes). Attributability is reported as an admissibility
flag, not a score.

Proposed in response to [gaussia-labs/papers#24](https://github.com/gaussia-labs/papers/discussions/24).

## Structure

```
2026-09-accountability/
├── accountability.tex / accountability.pdf   # the paper
├── references.bib
└── experiments/
    ├── contradiction_checker/                # Experiment 1: which instrument verifies a claim
    └── sandbox/                              # Experiment 2: the metrics over synthetic sessions
```

## What backs each table

| Table | Reports | Artifact |
|---|---|---|
| Instrument comparison | 320 constructed cases, reranker vs. judge vs. entailment | `experiments/contradiction_checker/results/` |
| False claims caught, by failure mode | 32 cases per mode | `experiments/contradiction_checker/results/group_d.json` |
| The eighteen sandbox sessions | metrics under `gemma-4-31B` | `experiments/sandbox/results/sandbox_*.json` |
| Metric vs. rejected alternative | the denominator decision | `experiments/sandbox/results/` |

The benchmark construction and the per-group labels are documented in
`experiments/contradiction_checker/README.md`.

## Reproducing the experiments

The runners read credentials from `experiments/.env` (not committed):

```
GROQ_API_KEY=...     # probe generation
HF_TOKEN=...         # judges
HF_BILL_TO=...
```

```bash
cd experiments/contradiction_checker
python run_reranker.py      # cross-encoder reranker
python run_judge.py         # LLM judge
python run_entailment.py    # inference model
python run_group_d.py       # failure-mode breakdown

cd ../sandbox
python run_sandbox.py       # the eighteen sessions
python probe_confidence.py
```

## Compiling the paper

```bash
pdflatex accountability.tex
bibtex accountability
pdflatex accountability.tex
pdflatex accountability.tex
```
