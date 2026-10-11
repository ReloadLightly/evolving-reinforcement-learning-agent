#!/usr/bin/env python3
"""Audit recorded selection and learner injection in the complete CPU pilot.

This reads only small CSV/JSON files and uses the Python standard library. It
checks recorded evolutionary decisions against their training-episode fitness;
it does not infer policy quality or verify parameter tensors from checkpoints.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


TASK_HORIZONS = {"Pendulum-v1": 200, "Reacher-v5": 50}
ARMS = ("td3", "erl_pooled", "erl_balanced", "erl_no_ea_replay")
TRAINING_STEPS = 50_000
WARMUP_STEPS = 10_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(directory, metadata):
    """Return an auditable count summary for one fixed-budget pilot run."""
    run_id = directory.name
    task, arm = metadata["task"], metadata["arm"]
    horizon = TASK_HORIZONS[task]
    generation_steps = 5 * horizon
    require(metadata["horizon"] == horizon, f"{run_id}: unexpected horizon")
    require(metadata["total_training_transitions"] == TRAINING_STEPS
            and metadata["warmup"] == WARMUP_STEPS, f"{run_id}: unexpected training budget")
    episodes_path = directory / "episodes.csv"
    episodes = read_csv(episodes_path)
    require(len(episodes) == TRAINING_STEPS // horizon, f"{run_id}: wrong episode count")
    fitness = {}
    for index, episode in enumerate(episodes):
        generation, slot = divmod(index, 5)
        expected_warmup = index * horizon < WARMUP_STEPS
        require(int(episode["episode"]) == index and int(episode["generation"]) == generation
                and int(episode["slot"]) == slot, f"{run_id}: episode order/identity mismatch")
        require(int(episode["length"]) == horizon and int(episode["step"]) == (index + 1) * horizon,
                f"{run_id}: episode horizon/step mismatch")
        require(episode["warmup"] == str(expected_warmup), f"{run_id}: wrong warmup tag")
        expected_source = "ea" if arm != "td3" and slot < 4 else "learner"
        require(episode["source"] == expected_source, f"{run_id}: wrong collector source")
        emitted, native = float(episode["emitted_return"]), float(episode["native_return"])
        require(math.isfinite(emitted) and math.isfinite(native)
                and math.isclose(emitted, native, rel_tol=1e-12, abs_tol=1e-8),
                f"{run_id}: nonfinite or inconsistent episode reward")
        if expected_source == "ea" and not expected_warmup:
            fitness.setdefault((generation + 1) * generation_steps, {})[slot] = emitted

    summary = dict(run_id=run_id, task=task, arm=arm, seed=metadata["seed"],
                   training_episodes=len(episodes), checked_generations=0,
                   elite_checks=0, tournament_checks=0, injection_checks=0,
                   lineage_records=0, episodes_sha256=sha256(episodes_path))
    lineage_path = directory / "lineage.jsonl"
    if arm == "td3":
        require(not lineage_path.exists() or not lineage_path.read_text().strip(),
                f"{run_id}: standalone TD3 unexpectedly contains population events")
        return summary

    events = [json.loads(line) for line in lineage_path.read_text().splitlines() if line.strip()]
    learning_generations = (TRAINING_STEPS - WARMUP_STEPS) // generation_steps
    require(len(events) == 4 * learning_generations, f"{run_id}: wrong lineage event count")
    previous_ids = [f"initial_{slot}" for slot in range(4)]
    seen_ids = set(previous_ids)
    injection_steps = []
    for offset, step in enumerate(range(WARMUP_STEPS + generation_steps,
                                       TRAINING_STEPS + 1, generation_steps)):
        group = events[4 * offset:4 * offset + 4]
        scores = fitness[step]
        require(set(scores) == set(range(4)), f"{run_id}: incomplete population fitness")
        next_ids = []
        for slot, event in enumerate(group):
            require(event["step"] == step and event["slot"] == slot
                    and event["generation"] == step // generation_steps,
                    f"{run_id}: wrong event generation/order at {step}")
            identifier = event["individual_id"]
            require(identifier not in seen_ids and identifier not in next_ids,
                    f"{run_id}: reused lineage identifier at {step}")
            next_ids.append(identifier)
            expected_type = "elite" if slot == 0 else (
                "injection" if slot == 3 and (step - WARMUP_STEPS) % 5000 == 0 else "mutation")
            require(event["type"] == expected_type, f"{run_id}: wrong operator at {step}, slot {slot}")
            if expected_type == "injection":
                require(event["parent_index"] is None and event["parent_fitness"] is None,
                        f"{run_id}: injection incorrectly attributed to population fitness")
                require(event["parent_id"] == f"learner_g{event['generation']:06d}",
                        f"{run_id}: invalid learner parent identifier")
                require(event["discarded_offspring_parent_id"] in previous_ids,
                        f"{run_id}: discarded offspring refers to an unknown parent")
                injection_steps.append(step)
                summary["injection_checks"] += 1
                continue
            parent_index = event["parent_index"]
            require(isinstance(parent_index, int) and 0 <= parent_index < 4,
                    f"{run_id}: invalid parent index")
            require(event["parent_id"] == previous_ids[parent_index],
                    f"{run_id}: parent ID/index does not identify the previous generation")
            require(math.isclose(event["parent_fitness"], scores[parent_index],
                                 rel_tol=1e-12, abs_tol=1e-8),
                    f"{run_id}: reported parent fitness disagrees with its training episode")
            if expected_type == "elite":
                require(event["parent_fitness"] == max(scores.values()),
                        f"{run_id}: elite does not have maximum current-generation fitness")
                summary["elite_checks"] += 1
            else:
                contenders = event["tournament_indices"]
                require(len(contenders) == 2 and all(isinstance(i, int) and 0 <= i < 4 for i in contenders),
                        f"{run_id}: tournament is not two valid population indices")
                require(parent_index in contenders
                        and event["parent_fitness"] == max(scores[i] for i in contenders),
                        f"{run_id}: selected parent is not a tournament winner")
                summary["tournament_checks"] += 1
        seen_ids.update(next_ids)
        previous_ids = next_ids
        summary["checked_generations"] += 1
    require(injection_steps == list(range(15_000, 50_001, 5000)),
            f"{run_id}: expected eight injections at 15k, 20k, ..., 50k")
    summary.update(lineage_records=len(events), lineage_sha256=sha256(lineage_path))
    return summary


def audit(root):
    root = Path(root)
    expected = {(task, arm, seed) for task in TASK_HORIZONS for arm in ARMS for seed in range(3)}
    identities = set()
    runs = []
    for path in sorted(root.glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        identity = (metadata["task"], metadata["arm"], metadata["seed"])
        require(metadata["status"] == "completed", f"Incomplete run: {path.parent.name}")
        require(identity in expected and identity not in identities and metadata["reward_delay"] == 1,
                f"Unexpected or duplicate pilot condition: {identity}")
        identities.add(identity)
        runs.append(audit_run(path.parent, metadata))
    require(identities == expected and len(runs) == 24,
            "Lineage audit requires all 24 distinct completed immediate-reward pilot runs")
    totals = {key: sum(run[key] for run in runs) for key in
              ("training_episodes", "checked_generations", "elite_checks", "tournament_checks",
               "injection_checks", "lineage_records")}
    return dict(status="passed", audited_utc=datetime.now(timezone.utc).isoformat(),
                scope="Recorded evolutionary decisions in the 24-run immediate-reward CPU pilot",
                limitation="Checks logs against training fitness; does not verify policy parameter tensors or establish learning benefit.",
                completed_runs=len(runs), evolutionary_runs=sum(r["arm"] != "td3" for r in runs),
                **totals, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilot"))
    parser.add_argument("--out", type=Path, help="Default: ROOT/lineage-audit.json")
    args = parser.parse_args()
    destination = args.out or args.root / "lineage-audit.json"
    report = audit(args.root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(f"Validated {report['completed_runs']} runs, {report['checked_generations']} evolutionary generations, "
          f"and {report['injection_checks']} learner injections; wrote {destination}")


if __name__ == "__main__":
    main()
