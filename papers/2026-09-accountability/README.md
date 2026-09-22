# Accountability

Paper proposing two metrics for whether a tool-using assistant respected human
authority over its actions: **OversightCompliance** (actions that ran without a valid
approval, as a rate over actions and as a weighted figure carrying which of five checks
failed) and **ActionDisclosure** (what the assistant ran but never mentioned, and what
it said that the trace refutes). Attributability is reported as an admissibility flag,
not a score.

Proposed in response to [gaussia-labs/papers#24](https://github.com/gaussia-labs/papers/discussions/24).

## Structure

```
2026-09-accountability/
├── accountability.tex / accountability.pdf   # the paper
└── references.bib
```

The cases, the runners and the raw results for both experiments live in their own
repository: **[Alquimia-ai/experiments][exp]**, under `metrics/2026-09-accountability/`.

[exp]: https://github.com/Alquimia-ai/experiments/tree/main/metrics/2026-09-accountability

## What backs each table

| Table | Reports | Artifact |
|---|---|---|
| Instrument comparison | 320 constructed cases, reranker vs. judge vs. entailment | [`contradiction_checker/results/`][exp] |
| False claims caught, by failure mode | 32 cases per mode | [`contradiction_checker/results/group_d.json`][exp] |
| The eighteen sandbox sessions | metrics under `gemma-4-31B` | [`sandbox/results/sandbox_*.json`][exp] |
| Metric vs. rejected alternative | the denominator decision | [`sandbox/results/`][exp] |

Every deterministic figure, both OversightCompliance columns included, reproduces with
`python run_sandbox.py --judge oracle`, which needs no credentials. The benchmark
construction and the per-group labels are documented in the experiments repository.

## Compiling the paper

```bash
pdflatex accountability.tex
bibtex accountability
pdflatex accountability.tex
pdflatex accountability.tex
```
