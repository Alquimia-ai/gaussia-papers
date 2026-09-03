"""Score the 120 claim/trace cases with the framework's own ContradictionChecker.

What this measures: whether the score separates group A (claim agrees with the
trace) from group B (claim contradicts the trace using the same vocabulary). The
component asks the reranker the right question, so the open question is whether a
relevance-trained reranker can answer it. Group C (unrelated) is the control: it
shows the score is not broken in general.

The component is used as the framework uses it, defaults included, so the result
describes what a practitioner would actually get.

Usage:
    python run_reranker.py                      # canonical format only
    python run_reranker.py --format all         # all five renderings
    python run_reranker.py --threshold 0.6      # decision threshold under test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

from gaussia.core.contradiction_checker import ContradictionChecker
from gaussia.core.document_retriever import RetrievedChunk
from gaussia.rerankers import QwenReranker

from render import FORMATS, load_cases, render

RESULTS_DIR = Path(__file__).parent / "results"


def score_batch(
    checker: ContradictionChecker,
    cases: list[dict],
    fmt: str,
) -> list[dict]:
    """One reranker call per case: the claim is the query, the trace entry the document."""
    rows = []
    for i, case in enumerate(cases, 1):
        trace = render(case, fmt)
        chunk = RetrievedChunk(text=trace, source=f"trace/{case['id']}", chunk_index=0)
        ranked = checker.check(case["claim"], [chunk])[0]
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "domain": case["domain"],
                "format": fmt,
                "failure_mode": case.get("failure_mode"),
                "claim": case["claim"],
                "trace": trace,
                "expected": case["expected"],
                "score": ranked.reranker_score,
                "verdict": ranked.verdict,
            }
        )
        if i % 20 == 0:
            print(f"  {fmt}: {i}/{len(cases)}", flush=True)
    return rows


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability that a random positive outranks a random negative (ties count half).

    Here "positive" is group A and "negative" is group B, so 0.5 means the score
    carries no information about whether the claim is true.
    """
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows: list[dict], threshold: float) -> dict:
    by_group = {g: [r["score"] for r in rows if r["group"] == g] for g in ("A", "B", "C")}

    # The component's own decision rule, applied at the threshold under test.
    def verdict(score: float) -> str:
        return "SUPPORTS" if score >= threshold else "CONTRADICTS"

    correct_a = sum(verdict(s) == "SUPPORTS" for s in by_group["A"])
    correct_b = sum(verdict(s) == "CONTRADICTS" for s in by_group["B"])
    flagged_c = sum(verdict(s) == "CONTRADICTS" for s in by_group["C"])

    per_mode: dict[str, dict] = {}
    for r in rows:
        if r["group"] != "B":
            continue
        m = r["failure_mode"]
        bucket = per_mode.setdefault(m, {"n": 0, "caught": 0, "scores": []})
        bucket["n"] += 1
        bucket["caught"] += verdict(r["score"]) == "CONTRADICTS"
        bucket["scores"].append(r["score"])
    for m, b in per_mode.items():
        b["mean_score"] = round(mean(b["scores"]), 4)
        del b["scores"]

    return {
        "threshold": threshold,
        "n": len(rows),
        "score_stats": {
            g: {
                "n": len(v),
                "mean": round(mean(v), 4),
                "median": round(median(v), 4),
                "min": round(min(v), 4),
                "max": round(max(v), 4),
            }
            for g, v in by_group.items()
            if v
        },
        "auc_A_vs_B": round(auc(by_group["A"], by_group["B"]), 4),
        "auc_A_vs_C": round(auc(by_group["A"], by_group["C"]), 4),
        "accuracy": {
            "A_called_SUPPORTS": f"{correct_a}/{len(by_group['A'])}",
            "B_called_CONTRADICTS": f"{correct_b}/{len(by_group['B'])}",
            "C_called_CONTRADICTS": f"{flagged_c}/{len(by_group['C'])}",
        },
        "B_by_failure_mode": per_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="canonical", choices=[*FORMATS, "all"])
    parser.add_argument("--threshold", type=float, default=0.6, help="component default is 0.6")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="trace entries are one-liners; the component default of 8192 only pads",
    )
    opts = parser.parse_args()

    cases = load_cases()
    formats = FORMATS if opts.format == "all" else (opts.format,)

    reranker = QwenReranker(model_name=opts.model, max_length=opts.max_length)
    checker = ContradictionChecker(reranker, contradiction_threshold=opts.threshold)

    print(f"cases={len(cases)} formats={list(formats)} model={opts.model}")
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for fmt in formats:
        t0 = time.time()
        rows = score_batch(checker, cases, fmt)
        summaries[fmt] = summarize(rows, opts.threshold)
        summaries[fmt]["elapsed_s"] = round(time.time() - t0, 1)
        all_rows += rows
        s = summaries[fmt]
        print(
            f"{fmt}: AUC(A vs B)={s['auc_A_vs_B']} "
            f"AUC(A vs C)={s['auc_A_vs_C']} "
            f"B caught={s['accuracy']['B_called_CONTRADICTS']} "
            f"({s['elapsed_s']}s)",
            flush=True,
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {
            "model": opts.model,
            "threshold": opts.threshold,
            "max_length": opts.max_length,
            "instruction": reranker._instruction,  # noqa: SLF001 - recorded for reproducibility
        },
        "summaries": summaries,
        "rows": all_rows,
    }
    path = RESULTS_DIR / f"reranker_{opts.format}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
