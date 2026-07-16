# Roast Me — Universal Probe Library (usage example)

Executable example of the Roast Me framework. Generates **probes** (red-teaming seed
questions) from a Knowledge Base, sends them to a real assistant, and measures whether
the assistant **falls** for the trap or **resists**.

## To see the results: open the notebooks, they already have everything inside

No need to run anything or have API keys. Open in this order, with the
`Roast Me (pygaussia)` kernel:

1. **`jupyter/roastme.ipynb`** — Level 1: how probes are generated, 3 engines
   compared, the canonical dataset of 204 probes.
2. **`jupyter/compare_models.ipynb`** — comparison: how each LLM family generates
   probes (gemma / z.ai / kimi).
3. **`jupyter/profiler.ipynb`** — Level 2: the real agent evaluated, weakness
   profile, literal inferences (warning: preliminary read, see below).
4. **`jupyter/roastme_promptfoo.ipynb`** — generic risk plugins + evasion
   (inspired by promptfoo, warning: scaffold/exploratory, see below).

All four load data already frozen in `results/` and show real results
directly on opening — regenerating live is optional (last section).

## The idea: how the knowledge boundary is known

Each probe needs a label (exists / doesn't exist). How reliable that label is depends
on how the KB's knowledge boundary is known. Three engines behind the
same contract, which **compose** (you don't pick just one):

| Engine | How it knows the boundary | Absence (reliable label) | Own contribution |
|---|---|---|---|
| `deterministic` | extractor enumerates the KB | yes (baseline, perfect label) | absence control |
| `rag` | retrieves chunks via embeddings | no | maximum false-premise coverage |
| `graphrag` | FULL graph as enumeration | yes (retrieves it) | absence without a hand-built extractor |
| `grag` | retrieves a textual SUBGRAPH (GRAG paper, Hu et al. NAACL 2025) | no (doc=1) | MULTI-HOP false premise |

## Tangible result — Level 1 (compose over Ley 24.977)

| engine | probes | absence | acc. absence | false premise |
|---|---|---|---|---|
| deterministic | 65 | 7 | 1.00 | 58 |
| graphrag | 40 | 8 | 1.00 | 32 |
| rag | 99 | 10 | 0.00 | 89 |

Only the deterministic engine and GraphRAG get absence right (GraphRAG **retrieves** it
with the graph); RAG invents articles that do exist (acc 0.00) but delivers the widest
false-premise coverage. That's the trade off, measured on the same KB.

## Tangible result — multi-model generation comparison

| model | sec/probe | reading |
|---|---|---|
| `google/gemma-4-31B-it` | ~1.1s | workhorse: fast, cheap, good quality |
| `zai-org/GLM-5.2` (reasoning) | ~59.6s | more sophisticated multi-hop traps; ~54x slower |
| `moonshotai/Kimi-K2.6` (reasoning) | ~103.6s | the most elaborate (chains 2+ real facts); ~94x slower |

## Result — Level 2 (Profiler: real agent evaluated)

An **LLM judge** decides for each agent response whether it **fell** for the trap or
**resisted**, and aggregates the verdicts into an *assistant profile* (where it's most
likely to fall + which KB entities broke it).

> **Warning: preliminary read, not a paper result yet.** This is a first validation
> run of the pipeline (30 of 204 probes; the 4 judges — gemma, groq, GLM-5.2,
> Kimi-K2.6 — didn't run under equal conditions; the judge was never audited against
> human criteria; groq generates the probes AND judges; the verdicts are stochastic). The
> pipeline works end to end — that's what's proven. Before citing the numbers as a
> final result: scale to the full dataset and run the 4 judges with the
> same config.

## Result — generic risk plugins + evasion (inspired by promptfoo)

We investigated promptfoo (`plugins` = risk category, `strategies` = delivery
disguise) to see what it adds to the metric in general. Full detail and preselection
in [`promptfoo_research/plugins_and_strategies.md`](promptfoo_research/plugins_and_strategies.md).

> **Warning: scaffold/exploratory.** `GenericRiskEngine.generate_proposed()` generates
> probes via LLM (none hand-written) for 7 KB-entity-free plugins (prompt-extraction,
> excessive-agency, hallucination, etc. — 4 remain documented as `scaffolded`, not
> implemented), optionally anchored to a description of the agent under test. The
> evasion layer (`base64`/`rot13`)
> found something real: the response provider (`claude-opus-4-8`) blocks with
> `content_filter` the "decode this and answer" pattern — 8/8 probes blocked in
> both strategies, before the assistant's logic even comes into play. The
> multi-turn scaffold (`ConversationSession`) proves the session persists across turns, but
> does NOT implement a real escalation strategy (it's not Crescendo). See
> `jupyter/roastme_promptfoo.ipynb` for the detail with real data.

## Methodological integrity

The oracle (exact enumeration of the KB) is used **only for scoring**. It's never passed
to the LLM engine or to the verifier, which works via retrieval (RAG) or graph
membership (GraphRAG).

## Structure

```
roastme/
├── jupyter/              # the 4 notebooks — the tour (start here)
├── results/              # frozen data read by the notebooks (see results/README.md)
├── src/                  # all the code (plain scripts, run from the root)
├── data/                 # sample KBs (Ley 24.977, FAQ Aurora)
├── config/               # plugins/strategies (KB) + generic_plugins/evasion (promptfoo)
└── promptfoo_research/   # promptfoo plugins/strategies research (not code)
```

## Optional: regenerate from scratch

Only needed if you want to run something live (needs a `.env` file with API keys — see
`config.py`/`target_client.py` for the exact variable names). Everything is invoked
**from the `roastme/` root**:

```bash
PY=../../.venv/bin/python

# Level 1: generate the canonical dataset (the 3 composed engines)
$PY src/universal_probe_library.py --kb ley --engine compose

# Multi-model generation comparison
$PY src/compare_models.py --engine grag \
   --models "google/gemma-4-31B-it=12,zai-org/GLM-5.2=4,moonshotai/Kimi-K2.6=3"

# Direct chat with the target agent (bypassing the harness)
$PY src/target_client.py --chat

# Level 2: profile the agent with the 4 judges
$PY src/run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json \
   --judges "hf_router:google/gemma-4-31B-it,hf_router:zai-org/GLM-5.2,hf_router:moonshotai/Kimi-K2.6,groq:llama-3.3-70b-versatile" \
   --iterations 5

# Generic risk plugins (promptfoo) — no KB, templates or paraphrasing split
# across models (gemma does most of it, GLM/Kimi only a couple, they're slow reasoners)
$PY src/generic_probe_library.py --model "google/gemma-4-31B-it" --context "..."

# Evasion: wrap an already-generated dataset before profiling (writes to
# results/level2_profiler_evasion/, not results/level2_profiler/)
$PY src/run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json \
   --evasion base64 --judges "groq:llama-3.3-70b-versatile" --iterations 1 --limit 8

# Multi-turn scaffold (plumbing test, not a real strategy)
$PY src/run_multiturn_demo.py --limit 5 --turns 2
```

Each script has more parameters (`--help` lists them); `src/universal_probe_library.py`
supports a single engine, a custom KB, quantity/seed control, etc. — no need
to memorize them, they're in each file's docstring.

## Files (`src/`)

- `contract.py` — durable contract (Document, KnowledgeHook, Probe, ProbeEngine).
- `probe_library.py` — DeterministicEngine + composition (merge/dedup) + config/doc loading.
- `engines_rag.py`, `engines_graphrag.py`, `engines_grag.py` — the 3 generation engines.
- `kb.py`, `oracle.py` — example extractor for Ley 24.977 + honest scoring.
- `config.py` — LLM provider registry (Groq, HuggingFace) + logprobs + target credentials.
- `judge.py`, `profiler.py`, `run_profiler.py`, `report.py` — the Profiler (Level 2).
- `target_client.py` — client for the target assistant (Alquimia runtime, SSE; `--chat` to talk to it directly; `ConversationSession` for the multi-turn scaffold).
- `universal_probe_library.py`, `compare_models.py` — generation entrypoints (Level 1 and 1b).
- `engines_generic.py`, `generic_probe_library.py` — generic risk plugins (promptfoo, no KB).
- `evasion.py` — evasion layer (Base64/ROT13 implemented; see `config/evasion_strategies.yaml`).
- `run_multiturn_demo.py` — multi-turn scaffold (plumbing, not a real attack).
