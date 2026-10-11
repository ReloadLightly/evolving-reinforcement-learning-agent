#!/usr/bin/env python3
"""Validate the research plan and emit budgets or a reproducible CSV manifest.

Uses only the Python standard library. Does not import an RL framework, install
dependencies, create environments, run training, or contact external services.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "study.json"
FIELDS = (
    "run_id", "phase", "task", "arm", "reward_delay", "seed",
    "training_transitions", "episode_horizon", "evaluation_interval",
    "evaluation_checkpoints", "evaluation_episodes_per_checkpoint",
    "evaluation_transition_budget", "replay_sampling",
    "evolutionary_replay_fraction", "collect_evolutionary_transitions",
    "actor_injection",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value, label):
    require(type(value) is int and value > 0, f"{label} must be a positive integer")


def unique(values, label):
    require(len(values) == len(set(values)), f"Duplicate {label}")


def checkpoints(steps, interval, include_initial):
    positive_integer(steps, "Training transitions")
    positive_integer(interval, "Evaluation interval")
    require(steps % interval == 0, "Evaluation interval must divide the training budget")
    require(type(include_initial) is bool, "include_initial_checkpoint must be boolean")
    return steps // interval + int(include_initial)


def run_id(config, task, arm, delay, seed, pilot):
    phase = "pilot" if pilot else "main"
    return f"{config['study_id']}__{phase}__{task}__{arm}__d{delay:03d}__s{seed:03d}"


def build_manifest(config, stage):
    pilot = stage == "pilot"
    schedule = config["pilot"] if pilot else config["evaluation"]
    steps = (config["pilot"]["total_transitions_per_run"] if pilot else
             config["training"]["total_transitions_per_run"])
    interval = schedule["every_total_training_transitions"]
    episodes = schedule["episodes_per_checkpoint"]
    count = checkpoints(steps, interval, schedule["include_initial_checkpoint"])
    seeds = config["pilot"]["seeds"] if pilot else config["confirmatory_seeds"]
    delays = config["pilot"]["reward_delays"] if pilot else config["reward_delays"]
    rows = []
    for task in config["tasks"]:
        for arm in config["arms"]:
            first = (task["id"] in config["first_stage"]["tasks"] and
                     arm["id"] in config["first_stage"]["arms"])
            if stage == "first" and not first:
                continue
            for delay in delays:
                for seed in seeds:
                    rows.append({
                        "run_id": run_id(config, task["id"], arm["id"], delay, seed, pilot),
                        "phase": "pilot" if pilot else ("first" if first else "extension"),
                        "task": task["id"],
                        "arm": arm["id"],
                        "reward_delay": delay,
                        "seed": seed,
                        "training_transitions": steps,
                        "episode_horizon": task["episode_horizon"],
                        "evaluation_interval": interval,
                        "evaluation_checkpoints": count,
                        "evaluation_episodes_per_checkpoint": episodes,
                        "evaluation_transition_budget": count * episodes * task["episode_horizon"],
                        "replay_sampling": arm["replay_sampling"],
                        "evolutionary_replay_fraction": arm["evolutionary_replay_fraction"],
                        "collect_evolutionary_transitions": arm["collect_evolutionary_transitions"],
                        "actor_injection": arm["actor_injection"],
                    })
    return sorted(rows, key=lambda row: row["run_id"])


def budget(rows):
    return {
        "runs": len(rows),
        "training_transitions": sum(row["training_transitions"] for row in rows),
        "evaluation_transition_budget": sum(row["evaluation_transition_budget"] for row in rows),
    }


def optional_budgets(config):
    train = config["training"]["total_transitions_per_run"]
    evaluation = config["evaluation"]
    count = checkpoints(train, evaluation["every_total_training_transitions"],
                        evaluation["include_initial_checkpoint"])
    per_task = count * evaluation["episodes_per_checkpoint"]
    horizon_sum = sum(task["episode_horizon"] for task in config["tasks"])
    output = {}
    for experiment in config["optional_experiments"]:
        combinations = (len(experiment["arms"]) * len(experiment["reward_delays"]) *
                        len(config["confirmatory_seeds"]))
        runs = combinations * len(config["tasks"])
        output[experiment["id"]] = {
            "runs": runs,
            "training_transitions": runs * train,
            "evaluation_transition_budget": combinations * per_task * horizon_sum,
        }
    return output


def validate(config):
    require(config["status"] == "draft", "Expected a draft research plan")
    require(config["artifact_type"] == "research_plan_not_training", "Expected a research-plan artifact")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", config["study_id"]) is not None,
            "study_id must contain only letters, digits, underscores, or hyphens")
    for group, key in ((config["tasks"], "task"), (config["arms"], "arm")):
        require(bool(group), f"At least one {key} is required")
        unique([item["id"] for item in group], key + " IDs")
    for task in config["tasks"]:
        positive_integer(task["episode_horizon"], "Episode horizon")
    task_ids = {task["id"] for task in config["tasks"]}
    arm_ids = {arm["id"] for arm in config["arms"]}
    require(set(config["first_stage"]["tasks"]) <= task_ids, "Unknown first-stage task")
    require(set(config["first_stage"]["arms"]) <= arm_ids, "Unknown first-stage arm")
    for seeds in (config["confirmatory_seeds"], config["pilot"]["seeds"]):
        require(bool(seeds), "Seeds must not be empty")
        require(all(type(seed) is int and seed >= 0 for seed in seeds), "Seeds must be nonnegative integers")
        unique(seeds, "seeds")
    require(set(config["confirmatory_seeds"]).isdisjoint(config["pilot"]["seeds"]),
            "Pilot and confirmatory seeds must be disjoint")
    for delays in (config["reward_delays"], config["pilot"]["reward_delays"]):
        require(bool(delays), "Reward delays must not be empty")
        unique(delays, "reward delays")
        for delay in delays:
            positive_integer(delay, "Reward delay")
    for arm in config["arms"]:
        fraction = arm["evolutionary_replay_fraction"]
        require(fraction is None or (type(fraction) in (int, float) and 0 <= fraction <= 1),
                "Evolutionary replay fractions must lie in [0, 1] or be null for pooled replay")
        require((fraction is None) == (arm["replay_sampling"] == "uniform_pooled"),
                "Only uniform pooled replay has an unspecified source fraction")
    for schedule in (config["evaluation"], config["pilot"]):
        positive_integer(schedule["episodes_per_checkpoint"], "Evaluation episodes")
    warmup = config["training"]["warmup_total_transitions"]
    positive_integer(warmup, "Warmup transitions")
    require(warmup < config["pilot"]["total_transitions_per_run"], "Pilot must extend beyond warmup")
    require(warmup < config["training"]["total_transitions_per_run"], "Main study must extend beyond warmup")
    manifests = {stage: build_manifest(config, stage) for stage in ("pilot", "first", "full")}
    for stage, rows in manifests.items():
        unique([row["run_id"] for row in rows], stage + " run IDs")
        require(budget(rows) == config["expected_budgets"][stage], f"{stage} budget does not match declared counts")
    full_by_id = {row["run_id"]: row for row in manifests["full"]}
    require(all(full_by_id.get(row["run_id"]) == row for row in manifests["first"]),
            "First-stage runs must be identical members of the full study")
    options = optional_budgets(config)
    for name, counts in options.items():
        require(counts == config["expected_budgets"][name], f"{name} optional budget mismatch")
    normalization = config["random_policy_normalization"]
    positive_integer(normalization["episodes_per_task"], "Normalization episodes")
    require(normalization["seed"] not in config["confirmatory_seeds"] + config["pilot"]["seeds"],
            "Normalization seed must be independent of training seeds")
    normalization_budget = normalization["episodes_per_task"] * sum(task["episode_horizon"] for task in config["tasks"])
    require(normalization_budget == config["expected_budgets"]["random_policy_normalization"]["evaluation_transition_budget"],
            "Normalization budget mismatch")
    return manifests, options, normalization_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("pilot", "first", "full"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv", type=Path, help="Write this stage's reproducible core run manifest")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        manifests, options, normalization_budget = validate(config)
        rows = manifests[args.stage]
        counts = budget(rows)
        first_ids = {row["run_id"] for row in manifests["first"]}
        remaining = [row for row in rows if row["run_id"] not in first_ids]
        summary = {
            "study_id": config["study_id"],
            "status": config["status"],
            "artifact_type": config["artifact_type"],
            "stage": args.stage,
            **counts,
            "total_transition_budget": counts["training_transitions"] + counts["evaluation_transition_budget"],
            "shared_normalization_transition_budget": normalization_budget,
            "stage_budget_plus_shared_normalization": counts["training_transitions"] + counts["evaluation_transition_budget"] + normalization_budget,
            "optional_experiments_excluded_from_manifest": options,
            "validation": "passed",
            "runtime": "unmeasured; calibrate on the target CPU using the pilot",
            "evaluation_budget_basis": config["evaluation"]["transition_budget_basis"],
        }
        if args.stage == "full":
            summary["includes_first_stage_runs"] = len(first_ids)
            summary["additional_budget_after_first_stage"] = budget(remaining)
        if args.csv:
            require(args.csv.resolve() not in (args.config.resolve(), Path(__file__).resolve()),
                    "CSV output cannot overwrite the config or planner")
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            summary["csv_manifest"] = str(args.csv)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"study_plan: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
