# Roast Me — Universal Probe Library (worked example)

Runnable example of the first module of the Roast Me framework: the **Universal Probe
Library**. It takes a Knowledge Base, plugins and strategies, and a configurable retrieval
strategy, and returns a list of **probes** (red-teaming seed questions) together with their
**knowledge hooks** (provenance: which entity each probe points at, and whether that entity
exists in the KB — the ground truth for the downstream test).

Roast Me is a framework: it does not impose an implementation. This is a worked example that
grounds the theory in one case (Argentina's Ley 24.977, the Monotributo tax regime) and
measures tangible results. Self-contained folder; it only imports the `gaussia` package and
does not touch `src/`, `metrics/` or `examples/`.

> Note on language: the code, comments and docs are in English, but the LLM prompts, the
> generated probe questions and the KB stay in **Spanish**, because the example operates on a
> Spanish-language knowledge base and the probes must be Spanish to match the target domain.

## Where to start

Open **`jupyter/roastme.ipynb`** and run all cells. By default it loads the versioned
canonical datasets (instant, stable, no API key required) and walks through each engine, the
measured trade-off and sample probes. It is the guided tour of the module. To regenerate live
or change the amount of probes, set `REGENERATE = True` in the config cell.

## The core idea: how the knowledge frontier is known

Every probe needs a label (exists / does not exist). How reliable that label is depends on
how the KB's *knowledge frontier* (what exists and what does not) is known. We implement four
engines behind a single contract:

| Engine | How it knows the frontier | Absence (reliable label) | Own contribution |
|---|---|---|---|
| `deterministic` | an extractor enumerates the KB | yes (baseline, perfect label) | absence control |
| `rag` | retrieves chunks by embeddings | no | widest false-premise coverage |
| `graphrag` | full graph as an enumeration | yes (recovers it) | absence without a hand-written extractor |
| `grag` | retrieves a textual SUBGRAPH (GRAG paper) | no (doc=1) | MULTI-HOP false premise (chained relations) |

Two of the engines use knowledge graphs for different purposes: `graphrag` uses the whole
graph as a catalog (to recover absence), while `grag` implements the technique from the
*GRAG* paper (Hu et al., NAACL 2025), retrieving the subgraph relevant to a query to generate
multi-hop traps. `grag` is faithful to the paper on the retrieval side (top-N ego-graphs +
soft pruning + hierarchical text description); the "graph view" via soft prompts is out of
scope because embeddings cannot be injected into the model through a chat API (see
`engines_grag.py`).

You do not pick one engine: they **compose**. The extractor is not a branch of a cascade, it
is an extra capability. When an extractor is available all engines run and their probes are
merged with dedup (on an overlap the most reliable label wins). Without an extractor only the
LLM engines run.

## Setup

Requires a Python environment with the `gaussia` package (embedder, retriever) plus the
dependencies in `requirements.txt` (`openai`, `networkx`, `numpy`, `sentence-transformers`,
`pyyaml`, `python-dotenv`). The simplest option is to reuse the `.venv` from the `pygaussia`
repo (which already ships all of them); otherwise create an environment and install `gaussia`
+ `requirements.txt`.

Create a `.env` file in this folder with your credentials (it is gitignored):

```bash
GROQ_API_KEY=...        # for the default Groq provider
# Optional, for the multi-model comparison via HuggingFace Inference Providers:
HF_TOKEN=...            # fine-grained token with "Make calls to Inference Providers"
HF_BILL_TO=...          # org to bill serverless usage to (X-HF-Bill-To header)
```

## Usage

```bash
# PY = the python of the environment with gaussia + requirements (e.g. the pygaussia .venv).
PY=python

# Composition over the law (structured KB): produces the trade-off table.
$PY universal_probe_library.py --kb ley --engine compose

# A single engine (ablation / paper tables).
$PY universal_probe_library.py --kb ley --engine graphrag
$PY universal_probe_library.py --kb ley --engine rag

# Deterministic only, no LLM and no key.
$PY universal_probe_library.py --kb ley --engine deterministic --no-llm

# Free-text KB (no extractor -> LLM engines only).
$PY universal_probe_library.py --kb faq --engine compose

# Your own KB + configurable model.
$PY universal_probe_library.py --kb data/my_kb/ --engine compose --provider groq --model llama-3.3-70b-versatile

# Controllable amount + reproducibility: 12 false-premise and 5 absence probes per LLM engine.
$PY universal_probe_library.py --kb ley --engine compose --n-false-premise 12 --n-absence 5 --seed 42

# Restrict to a plugin/strategy subset.
$PY universal_probe_library.py --kb ley --engine compose --plugin fabrication,false_premise

# GRAG paper engine: multi-hop false premise over subgraphs.
$PY universal_probe_library.py --kb ley --engine grag --seed 42

# Via HuggingFace serverless router (set HF_TOKEN + HF_BILL_TO in .env).
$PY universal_probe_library.py --kb ley --engine grag --provider hf_router --model google/gemma-4-31B-it
```

Parameters: `--kb` (preset `ley`/`faq` or a path to a `.md`/folder), `--engine`
(`compose|deterministic|rag|graphrag|grag`), `--provider` (`groq|openai|hf|hf_router`),
`--model`, `--no-llm`, `--out`, `--n-false-premise` (cap of false-premise probes per LLM
engine), `--n-absence` (absence probes per LLM engine), `--seed` (reduces run-to-run
variation), `--plugin` / `--strategy` (comma-separated subsets). The deterministic engine
enumerates the whole KB on purpose, so its count is not capped: it is the perfect-label
baseline.

## Outputs (in `results/`)

- `dataset_<kb>_<engine>.json`: the probes + knowledge hooks.
- `summary_<kb>_<engine>.json`: per-engine breakdown, dedup and the **trade-off table**
  (probes, absence probes, per-engine absence accuracy, false-premise coverage).

`results/` is scratch for your own runs; only the canonical datasets that the notebook loads
by default are versioned (so it works after cloning, without an API key).

## Tangible result (compose over the law)

| engine | probes | absence | absence acc. | false premise |
|---|---|---|---|---|
| deterministic | 65 | 7 | 1.00 | 58 |
| graphrag | 40 | 8 | 1.00 | 32 |
| rag | 99 | 10 | 0.00 | 89 |

(Canonical run frozen in `results/dataset_ley_compose.json`, generated with `--seed 42`. The
notebook loads it by default, so it reproduces these exact numbers. When regenerating live the
counts may vary slightly due to the nature of the LLM, but the trade-off pattern — and above
all the absence accuracy — is stable.)

Reading: only the deterministic and GraphRAG engines do absence well (GraphRAG **recovers** it
with the graph); RAG, even while retrieving fragments, invents articles that do exist
(acc 0.00) but provides the widest false-premise coverage. That is the trade-off, measured on
the same KB. On the FAQ (free text) the deterministic engine does not apply and both LLM
engines generalize to qualitative false premises, with no measurable absence.

## Multi-model comparison (`compare_models.py`)

The generator LLM is configurable, so we can compare how each model generates probes.
`compare_models.py` runs an engine (default `grag`) with several models through the
`hf_router` provider (HuggingFace Inference Providers, serverless: no endpoints to create,
usage billed to the org via `HF_BILL_TO`) and produces `results/compare_<engine>.json` plus a
readable report `results/compare_<engine>.html` with the full probes of each model.

```bash
$PY compare_models.py --engine grag \
   --models "google/gemma-4-31B-it=12,zai-org/GLM-5.2=4,moonshotai/Kimi-K2.6=3"
```

Different amounts are requested per model because cost/latency is very uneven. Finding from
the canonical run (`results/compare_grag.html`):

| model | sec/probe | reading |
|---|---|---|
| `google/gemma-4-31B-it` (12B dense) | ~1s | workhorse: fast, cheap, good quality |
| `zai-org/GLM-5.2` (753B reasoning) | ~56s | most sophisticated multi-hop traps; 50x slower |
| `moonshotai/Kimi-K2.6` (1T reasoning) | ~200s | quality similar to GLM but 180x slower; hard to justify |

The table measures amount and latency; the quality of the chains is a qualitative reading
(measuring it objectively would require an LLM judge, which belongs to a later stage).

## Methodological integrity

The oracle (exact enumeration of the KB) is used **only for scoring**. It is never passed to
the LLM engine or to the verifier, which works by retrieval (RAG) or by graph membership
(GraphRAG). GraphRAG produces labels only as good as the graph: the structural pass (headers)
is reliable, the LLM pass over free text inherits the graph's noise.

## Files

- `contract.py` — durable contract (Document, KnowledgeHook, Probe, ProbeEngine, selectors).
- `probe_library.py` — DeterministicEngine + composition (merge/dedup) + config/docs loading.
- `engines_rag.py` — RAGEngine (gaussia embeddings + LLM fact twisting).
- `engines_graphrag.py` — GraphRAGEngine (full structural graph + LLM triples; absence).
- `engines_grag.py` — GRAGEngine faithful to the paper (subgraph retrieval + multi-hop false premise).
- `kb.py` — example extractor for Ley 24.977.
- `oracle.py` — honest scoring (exact enumeration of the KB, scoring only).
- `config.py` — LLM provider registry (Groq, OpenAI, HuggingFace endpoint and router).
- `config/plugins.yaml`, `config/strategies.yaml` — risk families and attack patterns.
- `universal_probe_library.py` — parameterized entrypoint (single engine or composition).
- `compare_models.py` — multi-model comparison (table + HTML report).
- `jupyter/roastme.ipynb` — guided tour of the module (start here).
- `data/` — example KBs (Ley 24.977, FAQ Aurora).
- `results/` — outputs; only the canonical datasets used by the notebook are versioned.
