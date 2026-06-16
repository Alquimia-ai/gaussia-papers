"""
Assembles the expanded per-model benchmarks.

Query bank = the original hand-authored queries (gold, with hand-written G) plus
the reviewed, guard-passed generated queries (with model-generated G). For each
responder model it generates ONE deterministic response per query and writes
benchmarks/benchmark_<model>.json — the same schema run_experiments.py consumes.

Responders are given as 'provider:model' (model optional for hf, which defaults
to HF_MODEL_GEMMA). The attack author is never the responder for generated items
(authors come from generate_attacks.py); gold items are tagged author='hand'.

Usage:
  python assemble_benchmark.py \
      --responders groq:llama-3.1-8b-instant,groq:gemma2-9b-it,hf:
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import generate_benchmark as gb

HERE = Path(__file__).resolve().parent
DRAFTS = HERE / "drafts"
BENCH_DIR = HERE / "benchmarks"


def gold_bank() -> list[dict]:
    """The original hand-authored queries with their hand-written G."""
    bank = []
    for items, is_adv in [(gb.SINGLE_TURN_ATTACKS, True), (gb.LEGITIMATE_QUERIES, False)]:
        for category, query, g, difficulty in items:
            bank.append({
                "category": category, "query": query,
                "ground_truth_assistant": g, "difficulty": difficulty,
                "is_adversarial": is_adv, "attack_author": "hand", "g_source": "hand",
            })
    return bank


def generated_bank(path: Path) -> list[dict]:
    """Reviewed, guard-passed generated queries with model-generated G."""
    if not path.exists():
        print(f"  (sin {path.name}: solo se usa el banco gold)")
        return []
    items = json.load(open(path, encoding="utf-8"))
    keep = [it for it in items if it.get("reviewed") and it.get("g_guard_passed")]
    print(f"  banco generado: {len(keep)}/{len(items)} aprobados (reviewed & guard ok)")
    return [{
        "category": it["category"], "query": it["query"],
        "ground_truth_assistant": it["ground_truth_assistant"],
        "difficulty": it["difficulty"], "is_adversarial": it["is_adversarial"],
        "attack_author": it.get("attack_author", "generated"), "g_source": "generated",
    } for it in keep]


def build_for_responder(provider, model, bank, workers=8) -> list[dict]:
    client = gb.build_client(provider)
    model = gb.resolve_model(provider, model)
    print(f"Responder: {provider}:{model}  ({len(bank)} queries, {workers} workers)", flush=True)

    # qa_id counters depend on bank order, so assign them deterministically first.
    counters = {}
    metas = []
    for entry in bank:
        cat = entry["category"]
        counters[cat] = counters.get(cat, 0) + 1
        tag = "st" if entry["is_adversarial"] else "leg"
        metas.append((f"{cat[:4]}_{counters[cat]:03d}_{tag}_{entry['difficulty']}",
                      gb.SYSTEM_PROMPTS[cat], entry))

    done = [0]

    def fetch(meta):
        _, context, entry = meta
        ans = gb.call_model(client, model, context,
                            [{"role": "user", "content": entry["query"]}])
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{len(bank)}", flush=True)
        return ans

    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(fetch, metas))  # ex.map preserves input order

    records = []
    for (qa_id, context, entry), assistant in zip(metas, answers):
        records.append({
            "qa_id": qa_id, "context": context, "query": entry["query"],
            "assistant": assistant,
            "ground_truth_assistant": entry["ground_truth_assistant"],
            "category": entry["category"], "attack_type": "single_turn",
            "difficulty": entry["difficulty"], "is_adversarial": entry["is_adversarial"],
            "model": model, "provider": provider,
            "g_source": entry["g_source"], "attack_author": entry["attack_author"],
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responders", required=True,
                    help="comma-separated provider:model (model optional for hf)")
    ap.add_argument("--generated", default=str(DRAFTS / "ground_truth_draft.json"))
    args = ap.parse_args()

    bank = gold_bank() + generated_bank(Path(args.generated))
    adv = sum(1 for b in bank if b["is_adversarial"])
    print(f"Banco total: {len(bank)} queries ({adv} adversarial, {len(bank)-adv} legit)\n")

    BENCH_DIR.mkdir(exist_ok=True)
    for spec in args.responders.split(","):
        spec = spec.strip()
        if not spec:
            continue
        provider, _, model = spec.partition(":")
        records = build_for_responder(provider, model or None, bank)
        random.seed(42)
        random.shuffle(records)
        slug = gb._slug(records[0]["model"])
        out = BENCH_DIR / f"benchmark_{slug}.json"
        json.dump(records, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  -> {out.relative_to(HERE)}\n")


if __name__ == "__main__":
    main()
