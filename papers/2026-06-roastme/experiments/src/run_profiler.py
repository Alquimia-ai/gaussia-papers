"""Entrypoint for the Profiler (Roast Me, Level 2).

Fires the probe dataset at the target assistant once (freezing the transcripts),
then for each judge family and each iteration re-judges the frozen responses and
builds the assistant profile. Aggregates the per-iteration weakness rankings into
a stability table and renders the visual report.

Usage (from experiments/roastme, with the pygaussia venv):
  PY=../../.venv/bin/python
  $PY run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json \
     --judges "hf_router:google/gemma-4-31B-it,groq:llama-3.3-70b-versatile" \
     --iterations 5

Outputs in results/level2_profiler/:
  transcripts_<kb>.json            frozen assistant responses (hit the target once)
  profile_<kb>_<judge>.json        assistant profile per judge (from iteration 0)
  weakness_evolution_<kb>.json     ranked weaknesses per iteration + stability per judge
  profiler_report_<kb>.html        the readable report (open in a browser)

With --evasion <strategy_id>, every query is wrapped (see evasion.py) before hitting the
target, and ALL outputs go to results/level2_profiler_evasion/ instead — a separate
folder, not just a different filename, because profiler.ipynb globs
`profile_{KB}_*.json` in results/level2_profiler/ and that glob's `*` would also match a
merely-suffixed name (e.g. `profile_ley_compose__base64_...json`), silently mixing
evasion-wrapped profiles into the baseline view.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import probe_library as pl
import profiler
import report
from judge import DEFAULT_JUDGES, Judge, parse_judge_specs

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
RESULTS = HERE / "results" / "level2_profiler"
RESULTS_EVASION = HERE / "results" / "level2_profiler_evasion"


def _kb_name(dataset: str) -> str:
    stem = Path(dataset).stem
    return re.sub(r"^dataset_", "", stem)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# --- stability metrics ------------------------------------------------------
def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _kendall_tau(a: list, b: list) -> float:
    """Kendall's tau between two orderings of the same keys. +1 identical, -1 reversed."""
    pos = {k: i for i, k in enumerate(b)}
    seq = [pos[k] for k in a if k in pos]
    n = len(seq)
    if n < 2:
        return 1.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                c += 1
            elif seq[i] > seq[j]:
                d += 1
    return (c - d) / (n * (n - 1) / 2)


def _stability(rankings: list[list[str]], top_k: int) -> dict:
    """Pairwise-averaged stability over the per-iteration rankings."""
    pairs = list(combinations(range(len(rankings)), 2))
    if not pairs:
        return {"topk_jaccard_mean": 1.0, "kendall_tau_mean": 1.0, "rank_churn": {}}
    jac = sum(_jaccard(set(rankings[i][:top_k]), set(rankings[j][:top_k])) for i, j in pairs) / len(pairs)
    tau = sum(_kendall_tau(rankings[i], rankings[j]) for i, j in pairs) / len(pairs)
    churn: dict = {}
    all_keys = {k for r in rankings for k in r}
    for k in all_keys:
        idxs = [r.index(k) for r in rankings if k in r]
        churn[k] = max(idxs) - min(idxs) if idxs else 0
    return {"topk_jaccard_mean": round(jac, 4), "kendall_tau_mean": round(tau, 4),
            "rank_churn": churn}


def main() -> None:
    ap = argparse.ArgumentParser(description="Roast Me — Profiler (Level 2)")
    ap.add_argument("--dataset", default="results/level1_probes/dataset_ley_compose.json",
                    help="probe dataset (results/level1_probes/dataset_*.json)")
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES),
                    help="'provider:model,provider:model' judge roster")
    ap.add_argument("--iterations", type=int, default=5, help="re-judgings per judge")
    ap.add_argument("--limit", type=int, default=None, help="cap number of probes (smoke test)")
    ap.add_argument("--axis", default="by_strategy",
                    choices=["by_strategy", "by_plugin", "by_principle", "by_doc"],
                    help="weakness axis tracked in the evolution table")
    ap.add_argument("--top-k", type=int, default=3, help="top-k weaknesses for stability")
    ap.add_argument("--fallback-k", type=int, default=5, help="samples for the sampling fallback")
    ap.add_argument("--force-transcripts", action="store_true",
                    help="re-query the target even if a frozen cache exists")
    ap.add_argument("--evasion", default=None, choices=["base64", "rot13"],
                    help="wraps each query with this evasion strategy before sending "
                         "it to the target (see evasion.py); writes to "
                         "results/level2_profiler_evasion/ instead of level2_profiler/")
    args = ap.parse_args()

    results_dir = RESULTS_EVASION if args.evasion else RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    kb_name = _kb_name(args.dataset)
    probes = pl.load_dataset(args.dataset)
    print(f"dataset={args.dataset}  probes={len(probes)}  kb={kb_name}")

    if args.evasion:
        from dataclasses import replace
        import evasion
        probes = [replace(p, id=f"{p.id}--{args.evasion}",
                          query=evasion.apply(args.evasion, p.query))
                 for p in probes]
        kb_name = f"{kb_name}__{args.evasion}"
        print(f"evasion={args.evasion} applied -> kb={kb_name} (outputs in "
              f"{results_dir.relative_to(HERE)}/)")

    # 1) freeze the assistant's responses (hit the target once)
    from target_client import Target
    target = Target()
    transcripts = profiler.collect_transcripts(
        probes, target, results_dir / f"transcripts_{kb_name}.json",
        limit=args.limit, force=args.force_transcripts)

    # 2) judge x iteration sweep
    specs = parse_judge_specs(args.judges)
    per_judge: dict = {}
    profiles: dict = {}
    for provider, model in specs:
        judge = Judge(provider, model, fallback_k=args.fallback_k)
        print(f"\n=== judge {judge.key} (reasoner={judge._is_reasoner}, "
              f"try_logprobs={judge._try_logprobs}) ===")
        iters = []
        first_graded = None
        for i in range(args.iterations):
            graded = profiler.grade(probes, transcripts, judge, limit=args.limit)
            if first_graded is None:
                first_graded = graded
            ranked = profiler._rank(graded, lambda g, ax=args.axis: g[ax.replace("by_", "")])
            ranking = [str(r["key"]) for r in ranked]
            means = {str(r["key"]): r["mean"] for r in ranked}
            method = graded[0]["method"] if graded else "n/a"
            iters.append({"i": i, "ranking": ranking, "means": means, "method": method})
            print(f"  iter {i}: top={ranking[:args.top_k]} ({method})")
        per_judge[judge.key] = {
            "iterations": iters,
            "stability": _stability([it["ranking"] for it in iters], args.top_k),
        }
        # profile from iteration 0 (representative), written per judge
        profile = profiler.build_profile(first_graded or [], judge, dataset=args.dataset)
        profiles[judge.key] = profile
        pf = results_dir / f"profile_{kb_name}_{_slug(judge.key)}.json"
        pf.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  profile -> {pf.name}")

    # 3) evolution artifact
    evolution = {
        "config": {"dataset": args.dataset, "kb": kb_name, "iterations": args.iterations,
                   "mode": "fixed_transcripts", "top_k": args.top_k, "axis": args.axis},
        "per_judge": per_judge,
    }
    ev = results_dir / f"weakness_evolution_{kb_name}.json"
    ev.write_text(json.dumps(evolution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nevolution -> {ev.name}")

    # 4) readable report
    out = results_dir / f"profiler_report_{kb_name}.html"
    report.build_html(out, kb_name=kb_name, profiles=profiles, evolution=evolution)
    print(f"report -> {out.name}  (open it in your browser)")


if __name__ == "__main__":
    main()
