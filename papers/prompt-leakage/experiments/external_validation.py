"""
External validation of the PLR leak-detection signal against an independent,
labeled dataset: gabrielchua/system-prompt-leakage (HuggingFace).

That dataset gives (system_prompt, content, leakage in {0,1}). It has no safe
reference G, so we cannot run the full two-signal decision table; instead we
validate Signal 1 alone --- MaxChunkSim(content, system_prompt) --- which is the
core leak-overlap detector, against the external labels.

Caveats (reported, not hidden):
  - labels are synthetic, not human-annotated;
  - negatives are off-topic text, an easier case than the on-topic legitimate
    responses our own benchmark stresses;
  - this validates Signal 1 only, not the full pipeline.

Still, it is a genuine external check: do the reranker / cosine overlap scores
separate prompt-derived content (leak=1) from unrelated content (leak=0)?

Pulls a balanced sample via the HF datasets-server REST API (no extra deps),
scores it, and reports precision / recall / F1 at tau=0.6 plus a threshold
sweep and AUC. Writes results/external_validation.json.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import similarity_methods as sm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DS = "gabrielchua/system-prompt-leakage"
N_PER_CLASS = 80
TAU = 0.6
METHODS = ("cosine", "reranker")


def fetch_rows(offset, length=100):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DS.replace('/','%2F')}"
           f"&config=default&split=test&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def balanced_sample(n_per):
    """Collect n_per of each label by scanning pages spread across the split."""
    total = 71351
    pos, neg = [], []
    offsets = [int(total * i / 24) for i in range(24)]  # spread across the file
    for off in offsets:
        if len(pos) >= n_per and len(neg) >= n_per:
            break
        for row in fetch_rows(min(off, total - 100)):
            lab = int(row["leakage"])
            if lab == 1 and len(pos) < n_per:
                pos.append(row)
            elif lab == 0 and len(neg) < n_per:
                neg.append(row)
    return pos[:n_per] + neg[:n_per]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - m), 4), round(min(1, c + m), 4)]


def auc(scores, labels):
    """Rank-based AUC (probability a random positive outranks a random negative)."""
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return round((rank_sum - pos * (pos + 1) / 2) / (pos * neg), 4)


def prf(scores, labels, tau):
    tp = sum(1 for s, l in zip(scores, labels) if s >= tau and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= tau and l == 0)
    fn = sum(1 for s, l in zip(scores, labels) if s < tau and l == 1)
    tn = sum(1 for s, l in zip(scores, labels) if s < tau and l == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tau": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, tp + fn)}


def main():
    print(f"Sampling {N_PER_CLASS} per class from {DS} (test)...")
    rows = balanced_sample(N_PER_CLASS)
    labels = [int(r["leakage"]) for r in rows]
    print(f"  got {len(rows)} rows ({sum(labels)} leak, {len(labels)-sum(labels)} no-leak)")

    out = {"dataset": DS, "n": len(rows), "n_pos": sum(labels),
           "tau": TAU, "methods": {}}
    for m in METHODS:
        print(f"  scoring Signal 1 with {m}...")
        scores = []
        for i, r in enumerate(rows, 1):
            chunks = sm.chunk_prompt(r["system_prompt"])
            scores.append(sm.max_chunk_sim(r["content"], chunks, m))
            if i % 40 == 0:
                print(f"    {i}/{len(rows)}")
        sweep = [prf(scores, labels, round(0.3 + 0.05 * k, 2)) for k in range(13)]
        out["methods"][m] = {
            "at_tau": prf(scores, labels, TAU),
            "auc": auc(scores, labels),
            "best_f1": max(sweep, key=lambda s: s["f1"]),
            "sweep": sweep,
        }

    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "external_validation.json", "w"), ensure_ascii=False, indent=2)
    print("\n=== EXTERNAL VALIDATION (Signal 1 vs synthetic labels) ===")
    for m in METHODS:
        a = out["methods"][m]
        t = a["at_tau"]; b = a["best_f1"]
        print(f"{m:9s} AUC={a['auc']}  | @tau=0.6  P={t['precision']} R={t['recall']} F1={t['f1']}"
              f"  | best F1={b['f1']} @tau={b['tau']}")
    print(f"\n-> {RESULTS/'external_validation.json'}")


if __name__ == "__main__":
    main()
