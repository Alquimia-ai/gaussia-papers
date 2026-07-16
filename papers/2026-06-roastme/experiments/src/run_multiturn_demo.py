"""Standalone entrypoint for the multi-turn scaffold (profiler.collect_multiturn_demo).

Proves that ConversationSession holds a session_id across several turns against the
real target. Does NOT implement a real escalation strategy (Crescendo) — the second
turn is a fixed generic follow-up, not adaptive. See target_client.ConversationSession
and profiler.collect_multiturn_demo for the detail.

Usage:
  python run_multiturn_demo.py --dataset results/level1_probes/dataset_ley_compose.json --limit 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import probe_library as pl
import profiler

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
RESULTS = HERE / "results" / "level2_profiler_multiturn"


def _kb_name(dataset: str) -> str:
    return re.sub(r"^dataset_", "", Path(dataset).stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-turn scaffold (Roast Me)")
    ap.add_argument("--dataset", default="results/level1_probes/dataset_ley_compose.json")
    ap.add_argument("--limit", type=int, default=5,
                    help="number of probes to use (scaffold, not a full run)")
    ap.add_argument("--turns", type=int, default=2, help="turns per probe")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    kb_name = _kb_name(args.dataset)
    probes = pl.load_dataset(args.dataset)

    from target_client import Target
    target = Target()
    out = profiler.collect_multiturn_demo(
        probes, target, RESULTS / f"transcripts_multiturn_{kb_name}.json",
        turns_per_probe=args.turns, limit=args.limit)
    print(f"\n{len(out)} multi-turn sessions written to "
          f"results/level2_profiler_multiturn/transcripts_multiturn_{kb_name}.json")


if __name__ == "__main__":
    main()
