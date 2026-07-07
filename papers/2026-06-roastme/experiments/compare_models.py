"""Multi-model comparison: runs probe generation with several LLMs and builds a table
+ a readable HTML report of how each model behaves. This is what Alex asked for the paper.

Uses the `hf_router` provider (HF Inference Providers, serverless) by default: no need to
create endpoints, usage is billed to the org via HF_BILL_TO. WATCH the cost/time: GLM and Kimi
are reasoning models (they burn many tokens and are 60-110x slower than Gemma). That's why you
can request DIFFERENT counts per model: many from Gemma, few from the reasoners.

Usage:
  PY=../../.venv/bin/python
  $PY compare_models.py --engine grag \
     --models "google/gemma-4-31B-it=12,zai-org/GLM-5.2=4,moonshotai/Kimi-K2.6=3"

Outputs in results/: compare_<engine>.json (data) and compare_<engine>.html (readable report,
opens in the browser and shows each model's full probes, untruncated).
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import logging
import os
import time
from pathlib import Path

import probe_library as pl
import oracle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Default plan: many from Gemma (viable), few from the reasoners (slow/expensive).
DEFAULT_PLAN = "google/gemma-4-31B-it=12,zai-org/GLM-5.2=4,moonshotai/Kimi-K2.6=3"


def _quiet() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for n in ("sentence_transformers", "httpx", "transformers", "openai"):
        logging.getLogger(n).setLevel(logging.ERROR)


def _max_tokens_for(model: str) -> int:
    """Reasoners need a lot of headroom (they think before the JSON); Gemma doesn't."""
    m = model.lower()
    return 4000 if ("gemma" in m or "llama" in m) else 8000


def _build_engine(which: str, client, model, embedder, seed, n, max_tokens):
    if which == "grag":
        from engines_grag import GRAGEngine
        return GRAGEngine(client, provider="hf_router", model=model, embedder=embedder,
                          seed=seed, n_probes=n, max_tokens=max_tokens)
    if which == "rag":
        from engines_rag import RAGEngine
        return RAGEngine(client, provider="hf_router", model=model, embedder=embedder,
                         seed=seed, n_false_premise=n)
    if which == "graphrag":
        from engines_graphrag import GraphRAGEngine
        return GraphRAGEngine(client, provider="hf_router", model=model, seed=seed,
                              n_false_premise=n)
    raise SystemExit(f"engine {which!r} not supported in the comparison")


def _row(model: str, probes: list, seconds: float, asked: int) -> dict:
    absence = [p for p in probes if p.hook.doc == 0]
    scoreable = [p for p in absence if oracle.true_doc_label(p.hook) is not None]
    abs_ok = sum(1 for p in scoreable if oracle.true_doc_label(p.hook) == 0)
    return {
        "model": model,
        "asked": asked,
        "probes": len(probes),
        "false_premise": sum(1 for p in probes if p.hook.doc == 1),
        "absence": len(absence),
        "absence_accuracy": round(abs_ok / len(scoreable), 3) if scoreable else None,
        "seconds": round(seconds, 1),
        "sec_per_probe": round(seconds / len(probes), 1) if probes else None,
    }


def _parse_plan(plan: str, default_n: int) -> list[tuple[str, int]]:
    out = []
    for part in plan.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            m, n = part.rsplit("=", 1)
            out.append((m.strip(), int(n)))
        else:
            out.append((part, default_n))
    return out


def _write_html(path: Path, engine: str, rows: list, samples: dict) -> None:
    """Readable report: summary table + each model's full probes (untruncated)."""
    def esc(x): return html.escape(str(x or ""))
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
         padding:0 1rem;color:#1a1a1a;line-height:1.5}
    h1{border-bottom:3px solid #333;padding-bottom:.3rem}
    table.sum{border-collapse:collapse;width:100%;margin:1rem 0}
    table.sum th,table.sum td{border:1px solid #ccc;padding:.5rem .7rem;text-align:right}
    table.sum th:first-child,table.sum td:first-child{text-align:left}
    table.sum thead{background:#f0f0f0}
    .model{margin-top:2.5rem;border-top:2px solid #666;padding-top:.5rem}
    .badge{display:inline-block;background:#eef;border-radius:4px;padding:.1rem .5rem;font-size:.85rem;margin-left:.5rem}
    .probe{border:1px solid #ddd;border-radius:8px;padding:.8rem 1rem;margin:.8rem 0;background:#fafafa}
    .q{font-weight:600;margin-bottom:.5rem}
    .chain{font-size:.9rem;margin:.2rem 0}
    .real{color:#0a6}.false{color:#c30}
    .lbl{display:inline-block;width:3.2rem;font-weight:600;color:#555}
    """
    parts = [f"<!doctype html><meta charset='utf-8'><title>Model comparison ({esc(engine)})</title>",
             f"<style>{css}</style>",
             f"<h1>Multi-model comparison &mdash; engine <code>{esc(engine)}</code></h1>",
             "<p>How each model generates probes over the same KB (Ley 24.977). Chain quality "
             "is a qualitative read; the table measures count and speed.</p>"]
    # summary table
    parts.append("<table class='sum'><thead><tr><th>model</th><th>asked</th><th>generated</th>"
                 "<th>false premise</th><th>total sec</th><th>sec/probe</th></tr></thead><tbody>")
    for r in rows:
        if "error" in r:
            parts.append(f"<tr><td>{esc(r['model'])}</td><td colspan='5'>ERROR: {esc(r['error'])}</td></tr>")
            continue
        parts.append(f"<tr><td>{esc(r['model'])}</td><td>{r['asked']}</td><td>{r['probes']}</td>"
                     f"<td>{r['false_premise']}</td><td>{r['seconds']}</td><td>{r['sec_per_probe']}</td></tr>")
    parts.append("</tbody></table>")
    # probes per model
    for model, probes in samples.items():
        parts.append(f"<div class='model'><h2>{esc(model)} <span class='badge'>{len(probes)} probes</span></h2></div>")
        for p in probes:
            parts.append("<div class='probe'>")
            parts.append(f"<div class='q'>{esc(p['query'])}</div>")
            meta = p.get("meta", {})
            if meta.get("real_chain"):
                parts.append(f"<div class='chain'><span class='lbl real'>real</span>{esc(meta['real_chain'])}</div>")
            if meta.get("false_chain"):
                parts.append(f"<div class='chain'><span class='lbl false'>false</span>{esc(meta['false_chain'])}</div>")
            parts.append("</div>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-model comparison (Roast Me)")
    ap.add_argument("--models", default=DEFAULT_PLAN,
                    help="comma-separated ids; count per model with 'id=N' (e.g. gemma=12,glm=4)")
    ap.add_argument("--engine", default="grag", choices=["grag", "rag", "graphrag"])
    ap.add_argument("--kb", default="ley")
    ap.add_argument("--provider", default="hf_router")
    ap.add_argument("--n-probes", type=int, default=4, help="count per model if not specified with =N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=100, help="probes to save per model in the report")
    args = ap.parse_args()
    _quiet()

    from config import build_client
    plan = _parse_plan(args.models, args.n_probes)

    if args.kb == "ley":
        docs = pl.load_kb_documents()
    else:
        docs = pl.load_documents(args.kb)
    plugins, strategies = pl.load_config()

    with contextlib.redirect_stderr(io.StringIO()):
        from gaussia.embedders import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    client = build_client(args.provider)

    rows, samples = [], {}
    for model, n in plan:
        mt = _max_tokens_for(model)
        print(f"\n>>> {model}  (requesting {n} probes, max_tokens={mt})", flush=True)
        eng = _build_engine(args.engine, client, model, embedder, args.seed, n, mt)
        t0 = time.perf_counter()
        try:
            probes = eng.generate(docs, plugins, strategies)
        except Exception as e:
            print(f"    FAIL: {type(e).__name__}: {str(e)[:160]}", flush=True)
            rows.append({"model": model, "error": str(e)[:200]})
            continue
        dt = time.perf_counter() - t0
        row = _row(model, probes, dt, n)
        rows.append(row)
        samples[model] = [{"query": p.query, "doc": p.hook.doc, "strategy": p.strategy,
                           "meta": p.meta} for p in probes[: args.samples]]
        print(f"    -> {row['probes']}/{n} probes, {row['seconds']}s "
              f"({row['sec_per_probe']}s/probe)", flush=True)

    print(f"\n=== comparison (engine={args.engine}) ===", flush=True)
    print(f"  {'model':<32}{'ask':>5}{'gen':>5}{'fp':>8}{'sec':>9}{'s/pr':>8}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['model']:<32}  ERROR", flush=True); continue
        print(f"  {r['model']:<32}{r['asked']:>5}{r['probes']:>5}{r['false_premise']:>8}"
              f"{r['seconds']:>9.1f}{(r['sec_per_probe'] or 0):>8.1f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    js = RESULTS / f"compare_{args.engine}.json"
    js.write_text(json.dumps({"engine": args.engine, "kb": args.kb, "provider": args.provider,
                              "seed": args.seed, "rows": rows, "samples": samples},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    hp = RESULTS / f"compare_{args.engine}.html"
    _write_html(hp, args.engine, rows, samples)
    print(f"\nwritten: {js.name}, {hp.name}", flush=True)


if __name__ == "__main__":
    main()
