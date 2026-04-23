# Experiments

Benchmark dataset and scoring experiments for the Role Adherence paper.

## Contents

| File | Description |
|------|-------------|
| `experiments.ipynb` | Main notebook: runs all scoring methods and generates figures |
| `benchmark.json` | 150-turn synthetic benchmark (30 conversations × 5 turns) |
| `generate_benchmark.py` | Script used to generate the benchmark dataset |

## Setup

Python 3.11 is required. Install dependencies:

```bash
pip install bert-score groq python-dotenv scipy sentence-transformers scikit-learn transformers matplotlib seaborn numpy
```

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

## Running

Open `experiments.ipynb` and run all cells. Figures are saved to `../figures/`.

The LLM judge uses `llama-3.3-70b-versatile` via Groq at `temperature=0`.
