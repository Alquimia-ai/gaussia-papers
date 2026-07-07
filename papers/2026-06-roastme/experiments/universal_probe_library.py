"""Universal Probe Library: parameterized entrypoint (Roast Me usage example).

Receives: KB (file or folder), plugins, strategies, retrieval strategy and the LLM
(provider + model). Returns a list of N probes + knowledge hooks, and a summary with
the per-engine breakdown and the trade off table (what the experiments section of the
paper cites as a tangible result).

Usage:
  python universal_probe_library.py --kb ley --engine compose
  python universal_probe_library.py --kb faq --engine rag --provider groq
  python universal_probe_library.py --kb data/mi_kb/ --engine graphrag --model llama-3.3-70b-versatile

The "engine" is the strategy for knowing the knowledge frontier:
  deterministic  extractor enumerates the KB (perfect labels, absence). Baseline.
  rag            retrieves chunks by embeddings, twists facts. Does not handle absence.
  graphrag       builds an entity graph, recovers absence with a reliable label.
  compose        runs all applicable ones and merges them with dedup (default).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import probe_library as pl
import kb
import oracle
from contract import select_engines

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Example KBs ready to use.
KB_PRESETS = {
    "ley": {"path": kb.KB_DIR, "doc_id": "ley_24977", "kind": "ley", "structured": True,
            "extractor": pl.ley_extractor},
    "faq": {"path": HERE / "data" / "faq_aurora.md", "doc_id": "faq_aurora", "kind": "faq",
            "structured": False, "extractor": None},
}


def _quiet_logs() -> None:
    for name in ("sentence_transformers", "httpx", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_engines(which: str, client, extractor, provider: str, model: str | None,
                  *, n_false_premise: int | None = None, n_absence: int | None = None,
                  seed: int | None = None):
    """Builds the engine list according to --engine (composition included).

    n_false_premise / n_absence / seed control the number of probes and
    reproducibility (the deterministic engine enumerates the full KB on purpose: it is the baseline).
    """
    det = pl.DeterministicEngine(extractor=extractor)
    engines = {"deterministic": det}
    if client is not None:
        from engines_rag import RAGEngine
        from engines_graphrag import GraphRAGEngine
        from engines_grag import GRAGEngine
        rag_kw = {"provider": provider, "model": model, "seed": seed,
                  "n_false_premise": n_false_premise}
        graphrag_kw = dict(rag_kw)
        if n_absence is not None:
            rag_kw["n_absence"] = n_absence
            graphrag_kw["n_absence"] = n_absence
        engines["rag"] = RAGEngine(client, **rag_kw)
        # graphrag: FULL graph as enumeration -> absence with a reliable label (original contribution).
        engines["graphrag"] = GraphRAGEngine(client, **graphrag_kw)
        # grag: paper technique (subgraph retrieval) -> multi-hop false premise.
        engines["grag"] = GRAGEngine(client, provider=provider, model=model, seed=seed)
    if which == "compose":
        return list(engines.values())
    if which not in engines:
        raise SystemExit(f"engine {which!r} not available (missing API key for rag/graphrag?)")
    return [engines[which]]


def tradeoff_table(probes) -> list[dict]:
    """One row per engine: absence (correct real label) and false-premise coverage.

    The absence accuracy uses the ORACLE (only for scoring): of the doc=0 probes the
    oracle can evaluate, how many point to something TRULY nonexistent.
    """
    rows = []
    for eng in sorted({p.engine for p in probes}):
        eps = [p for p in probes if p.engine == eng]
        absence = [p for p in eps if p.hook.doc == 0]
        abs_scoreable = [p for p in absence if oracle.true_doc_label(p.hook) is not None]
        abs_ok = sum(1 for p in abs_scoreable if oracle.true_doc_label(p.hook) == 0)
        false_prem = [p for p in eps if p.hook.doc == 1]
        rows.append({
            "engine": eng,
            "probes": len(eps),
            "absence_probes": len(absence),
            "absence_scoreable": len(abs_scoreable),
            "absence_accuracy": round(abs_ok / len(abs_scoreable), 3) if abs_scoreable else None,
            "false_premise_probes": len(false_prem),
        })
    return rows


def print_summary(info: dict, table: list[dict], probes) -> None:
    print("\n=== Universal Probe Library ===")
    if info:
        print(f"engines: {info.get('engines')}  raw={info.get('raw_total')} "
              f"dedup={info.get('deduped')} final={info.get('final_total')}")
    print(f"total probes: {len(probes)}")
    print("\ntrade off per engine:")
    print(f"  {'engine':<14}{'probes':>7}{'absence':>10}{'acc_abs':>9}{'false_prem':>12}")
    for r in table:
        acc = "n/a" if r["absence_accuracy"] is None else f"{r['absence_accuracy']:.2f}"
        print(f"  {r['engine']:<14}{r['probes']:>7}{r['absence_probes']:>10}"
              f"{acc:>9}{r['false_premise_probes']:>12}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal Probe Library (Roast Me)")
    ap.add_argument("--kb", default="ley", help="preset 'ley'/'faq' or path to .md/folder")
    ap.add_argument("--engine", default="compose",
                    choices=["compose", "deterministic", "rag", "graphrag", "grag"])
    ap.add_argument("--provider", default="groq",
                    help="groq | openai | hf (dedicated endpoint) | hf_router (serverless)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--plugin", default=None,
                    help="risk families to use (comma-separated); default: all")
    ap.add_argument("--strategy", default=None,
                    help="strategies to use by id (comma-separated); default: all")
    ap.add_argument("--out", default=None, help="dataset name in results/")
    ap.add_argument("--n-false-premise", type=int, default=None,
                    help="cap on false-premise probes per LLM engine (default: no cap)")
    ap.add_argument("--n-absence", type=int, default=None,
                    help="absence probes per LLM engine (default: the engine's own)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the LLM (reduces variation between runs)")
    ap.add_argument("--no-llm", action="store_true",
                    help="only engines without LLM (deterministic); does not build a client")
    args = ap.parse_args()
    _quiet_logs()

    # Resolve KB (preset or path).
    if args.kb in KB_PRESETS:
        p = KB_PRESETS[args.kb]
        docs = pl.load_documents(p["path"], doc_id=p["doc_id"], kind=p["kind"],
                                 structured=p["structured"])
        extractor = p["extractor"]
    else:
        docs = pl.load_documents(args.kb)
        extractor = None  # arbitrary KB: no extractor -> deterministic does not apply

    plugins, strategies = pl.load_config()

    # Optional filter by plugin / strategy (what Alex requested as a parameter).
    if args.plugin:
        want = {s.strip() for s in args.plugin.split(",")}
        plugins = {k: v for k, v in plugins.items() if k in want}
        strategies = [s for s in strategies if s.get("plugin") in want]
    if args.strategy:
        want = {s.strip() for s in args.strategy.split(",")}
        strategies = [s for s in strategies if s.get("id") in want]
    if not strategies:
        raise SystemExit("no strategies left after the --plugin/--strategy filter")

    # LLM client (unless deterministic-only).
    client = None
    if not args.no_llm and args.engine in ("compose", "rag", "graphrag", "grag"):
        from config import build_client
        client = build_client(args.provider)

    engines = build_engines(args.engine, client, extractor, args.provider, args.model,
                            n_false_premise=args.n_false_premise, n_absence=args.n_absence,
                            seed=args.seed)
    applicable = select_engines(docs, engines)
    print(f"KB={args.kb} engine={args.engine} -> applicable engines: "
          f"{[e.name for e in applicable]}")

    if len(applicable) > 1:
        probes, info = pl.generate_composed(docs, plugins, strategies, engines)
    else:
        probes = applicable[0].generate(docs, plugins, strategies)
        info = {}

    table = tradeoff_table(probes)
    print_summary(info, table, probes)

    # Write artifacts.
    RESULTS.mkdir(exist_ok=True)
    tag = args.out or f"{args.kb}_{args.engine}"
    ds_path = RESULTS / f"dataset_{tag}.json"
    sm_path = RESULTS / f"summary_{tag}.json"
    ds_path.write_text(json.dumps(pl.to_records(probes), ensure_ascii=False, indent=2),
                       encoding="utf-8")
    sm_path.write_text(json.dumps(
        {"kb": args.kb, "engine": args.engine, "provider": args.provider,
         "model": args.model,
         "config": {"seed": args.seed, "n_false_premise": args.n_false_premise,
                    "n_absence": args.n_absence},
         "composition": info, "tradeoff": table,
         "n_probes": len(probes),
         "by_strategy": dict(Counter(p.strategy for p in probes))},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {ds_path.name}, {sm_path.name}")


if __name__ == "__main__":
    main()
