#!/usr/bin/env python3
"""Read the pilot's live logs without importing or disturbing a learner."""
import argparse
import json
from pathlib import Path
import time


def show(root):
    print("\nCPU pilot — checkpoint observations (training continues between checkpoints)")
    print(f"{'Task':<13} {'Method':<18} {'Seed':>4} {'Status':<10} {'Steps':>8} {'Return':>10} {'EA draws':>9}")
    completed = failed = 0
    for path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        completed += meta["status"] == "completed"
        failed += meta["status"] == "failed"
        events = path.parent / "events.jsonl"
        latest = {}
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:  # a writer may be finishing its final line
                    continue
                if event["event"] == "evaluation":
                    latest = event
        score = f"{latest['return_mean']:.2f}" if "return_mean" in latest else "—"
        fraction = latest.get("replay_sampled_ea_fraction")
        share = f"{fraction:.1%}" if fraction is not None else "warmup"
        print(f"{meta['task']:<13} {meta['arm']:<18} {meta['seed']:>4} {meta['status']:<10} "
              f"{latest.get('env_steps', 0):>8,} {score:>10} {share:>9}")
    print(f"Completed {completed}/24; failed {failed}. Returns are native episode means; less negative is better.")
    print("EA draws = population fraction actually sampled from replay; it differs from collection share.")
    print("One seed or one checkpoint cannot establish a method advantage. Journals explain each checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilot"))
    parser.add_argument("--watch", type=float, help="Refresh interval in seconds; Ctrl-C to stop")
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error("--watch must be at least one second")
    try:
        while True:
            show(args.root)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
