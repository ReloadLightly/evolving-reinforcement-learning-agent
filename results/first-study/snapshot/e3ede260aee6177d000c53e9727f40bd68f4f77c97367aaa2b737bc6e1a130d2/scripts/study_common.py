"""Standard-library-only identities and integrity checks for the frozen study."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import sys


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/first-stage.json"
PACKAGES = ("torch", "numpy", "gymnasium", "mujoco", "matplotlib")
MANIFEST_FIELDS = (
    "run_id", "phase", "task", "arm", "reward_delay", "seed", "training_transitions",
    "warmup_transitions", "episode_horizon", "evaluation_interval", "evaluation_checkpoints",
    "evaluation_episodes_per_checkpoint", "evaluation_transition_budget", "critic_updates", "actor_updates",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return digest(Path(path).read_bytes())


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data.encode() if isinstance(data, str) else data)
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json_bytes(value))


def stable_seed(task, seed, namespace, *parts):
    """Exactly mirrors erl.envs.stable_seed without importing Gymnasium/Torch."""
    payload = json.dumps([task, int(seed), namespace, *parts], ensure_ascii=True,
                         separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def runtime_sources(repo=REPO):
    repo = Path(repo)
    return {str(path.relative_to(repo)): sha_file(path) for path in sorted((repo / "erl").glob("*.py"))}


def software_environment():
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                platform=platform.platform(), executable=sys.executable,
                packages={name: importlib.metadata.version(name) for name in PACKAGES})


def validate_config(config):
    """Reject unsupported settings instead of letting trainer defaults diverge."""
    require(config["study_id"] == "erl_cpu_delay_replay_v1" and config["stage"] == "first",
            "This runner supports the declared first study only")
    require(config["task"] == "Pendulum-v1" and config["episode_horizon"] == 200,
            "The first study is Pendulum-v1 with a 200-step horizon")
    require(config["arms"] == ["erl_pooled", "erl_balanced"] and config["reward_delays"] == [1, 20]
            and config["seeds"] == list(range(100, 110)), "Do not change the declared 40-condition design")
    expected = dict(total_transitions=200000, warmup_transitions=10000, hidden_sizes=[64, 64],
                    batch_size=128, replay_capacity=100000, critic_updates_per_transition=.25,
                    policy_delay=2, learning_rate=.0003, gamma=.99, tau=.005,
                    exploration_noise=.1, target_noise=.2, target_noise_clip=.5)
    require(config["training"] == expected, "Training settings differ from the supported frozen trainer")
    require(config["population"] == dict(size=4, elite_count=1, tournament_size=2,
                tournament_with_replacement=True, mutation_probability=.1, mutation_sigma=.02,
                injection_interval_post_warmup=5000, injection_slot=3), "Unsupported population settings")
    require(config["evaluation"] == dict(interval=10000, episodes=5, include_initial=True,
                                         policy="learner", reward="native"), "Unsupported evaluation settings")
    require(config["purpose"] == "confirmatory" and config["diagnostics"] == "full",
            "The first study requires confirmatory purpose and full diagnostics")
    require(config["execution"] == dict(device="cpu", threads_per_run=1, maximum_workers=2),
            "The supported execution budget is CPU, one thread per run, at most two workers")
    analysis = config["analysis"]
    require(analysis["analysis_seed"] == 20261012 and analysis["bootstrap_resamples"] == 10000
            and analysis["practical_effect_threshold"] == .05 and analysis["primary_delay"] == 20
            and analysis["interval_percentiles"] == [2.5, 97.5], "Analysis settings differ from the declared plan")


def build_manifest(config):
    validate_config(config)
    training, evaluation = config["training"], config["evaluation"]
    steps, warmup = training["total_transitions"], training["warmup_transitions"]
    checkpoints = steps // evaluation["interval"] + 1
    updates = int((steps - warmup) * training["critic_updates_per_transition"])
    rows = []
    for seed in config["seeds"]:
        for delay in config["reward_delays"]:
            for arm in config["arms"]:
                rows.append(dict(run_id=f"{config['study_id']}__main__{config['task']}__{arm}__d{delay:03d}__s{seed:03d}",
                    phase="first", task=config["task"], arm=arm, reward_delay=delay, seed=seed,
                    training_transitions=steps, warmup_transitions=warmup,
                    episode_horizon=config["episode_horizon"], evaluation_interval=evaluation["interval"],
                    evaluation_checkpoints=checkpoints, evaluation_episodes_per_checkpoint=evaluation["episodes"],
                    evaluation_transition_budget=checkpoints * evaluation["episodes"] * config["episode_horizon"],
                    critic_updates=updates, actor_updates=updates // training["policy_delay"]))
    return rows


def manifest_bytes(rows, fields=MANIFEST_FIELDS):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def resolved_seeds(config):
    names = ("learner_init", "warmup", "exploration", "replay", "selection", "mutation", "target_noise",
             "diagnostic_sampling", "diagnostic_target_noise")
    task = config["task"]
    generations = config["training"]["total_transitions"] // (5 * config["episode_horizon"])
    blocks = []
    for seed in config["seeds"]:
        blocks.append(dict(task=task, seed=seed,
            named_seeds={name: stable_seed(task, seed, name) for name in names},
            population_init=[stable_seed(task, seed, "population_init", slot) for slot in range(4)],
            evaluation=[stable_seed(task, seed, "evaluation", episode) for episode in range(config["evaluation"]["episodes"])],
            training_reset=[[stable_seed(task, seed, "training_reset", generation, slot) for slot in range(5)]
                            for generation in range(generations)]))
    return dict(schema_version=1, task=task, training_seeds=config["seeds"],
                seed_derivation="SHA-256 first 32 bits of JSON [task,seed,namespace,*parts], as in erl.envs.stable_seed",
                note="Each seed block is shared across both arms and both delays; evaluation is a separate namespace.",
                blocks=blocks)


def verify_freeze(root, check_worktree=True, repo=REPO):
    """Validate the sealed artifact graph and return (freeze, CSV row dicts)."""
    root, repo = Path(root), Path(repo)
    freeze = json.loads((root / "freeze.json").read_text())
    payload = {key: value for key, value in freeze.items() if key not in ("freeze_id", "snapshot")}
    require(freeze["schema_version"] == 1 and digest(canonical_bytes(payload)) == freeze["freeze_id"],
            "Freeze identity does not match its contents")
    require(freeze["snapshot"] == f"snapshot/{freeze['freeze_id']}", "Invalid snapshot path")
    snapshot = root / freeze["snapshot"]
    for name, expected_hash in freeze["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe frozen artifact path")
        require((snapshot / name).is_file() and sha_file(snapshot / name) == expected_hash,
                f"Frozen artifact changed or missing: {name}")
    validate_config(freeze["config"])
    require(freeze["config_sha256"] == freeze["files"]["configs/first-stage.json"], "Configuration hash mismatch")
    require(json.loads((snapshot / "configs/first-stage.json").read_text()) == freeze["config"],
            "Frozen configuration differs from embedded configuration")
    require(freeze["calibration_sha256"] == freeze["files"]["calibration.json"], "Calibration hash mismatch")
    for name, key in (("manifest.csv", "manifest_sha256"), ("seed-manifest.json", "seed_manifest_sha256")):
        require(sha_file(root / name) == freeze[key] == freeze["files"][name], f"Immutable {name} changed")
    require((root / "manifest.csv").read_bytes() == manifest_bytes(build_manifest(freeze["config"])),
            "Run manifest differs from the declared design")
    require(json.loads((root / "seed-manifest.json").read_text()) == resolved_seeds(freeze["config"]),
            "Resolved seed manifest differs from declared streams")
    if check_worktree:
        for name in freeze["repository_files"]:
            require((repo / name).is_file() and sha_file(repo / name) == freeze["files"][name],
                    f"Worktree changed after freeze: {name}; preserve existing runs and investigate")
        require(runtime_sources(repo) == freeze["source_sha256"], "Trainer source file set changed after freeze")
    with (root / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return freeze, rows


def validate_metadata(metadata, row, freeze, completed=False):
    expected = dict(task=row["task"], arm=row["arm"], seed=int(row["seed"]),
        reward_delay=int(row["reward_delay"]), training_budget=int(row["training_transitions"]),
        warmup=int(row["warmup_transitions"]), horizon=int(row["episode_horizon"]),
        evaluation_interval=int(row["evaluation_interval"]),
        evaluation_episodes=int(row["evaluation_episodes_per_checkpoint"]),
        source_sha256=freeze["source_sha256"], calibration_sha256=freeze["calibration_sha256"],
        protocol_sha256=freeze["freeze_id"], purpose="confirmatory", diagnostics="full", threads=1, device="cpu")
    for key, value in expected.items():
        require(metadata.get(key) == value, f"{row['run_id']}: metadata mismatch for {key}")
    if completed:
        require(metadata["status"] == "completed", f"{row['run_id']}: not completed")
        for key, value in (("total_training_transitions", int(row["training_transitions"])),
                           ("total_evaluation_transitions", int(row["evaluation_transition_budget"])),
                           ("critic_updates", int(row["critic_updates"])), ("actor_updates", int(row["actor_updates"]))):
            require(metadata.get(key) == value, f"{row['run_id']}: completed {key} mismatch")
        require(sum(metadata["sampled_transitions"].values()) == int(row["critic_updates"]) * 128,
                f"{row['run_id']}: replay sample budget mismatch")


def initial_lifecycle(freeze, rows):
    return dict(freeze_id=freeze["freeze_id"], created_utc=utc_now(), updated_utc=utc_now(),
                runs={row["run_id"]: dict(status="pending", attempts=0, pid=None) for row in rows})


def persist_lifecycle(root, state, rows):
    """Atomically replace each human/machine-readable view from one state."""
    root = Path(root)
    state["updated_utc"] = utc_now()
    counts = {status: sum(run["status"] == status for run in state["runs"].values())
              for status in ("pending", "running", "completed", "failed", "interrupted")}
    state["counts"] = counts
    write_json(root / "lifecycle.json", state)
    fields = (*MANIFEST_FIELDS, "status", "attempts", "last_error")
    lifecycle_rows = [dict(row, status=state["runs"][row["run_id"]]["status"],
                         attempts=state["runs"][row["run_id"]].get("attempts", 0),
                         last_error=state["runs"][row["run_id"]].get("last_error", "")) for row in rows]
    atomic_write(root / "lifecycle.csv", manifest_bytes(lifecycle_rows, fields))
    lines = ["# First-study execution progress", "", f"Freeze: `{state['freeze_id']}`", "",
             f"Updated: {state['updated_utc']}", "",
             "; ".join(f"{count} {status}" for status, count in counts.items()) + ".", "",
             "This reports execution and budgets. Confirmatory method comparisons wait for the complete declared study.", "",
             "The immutable plan remains in `manifest.csv`; failures keep their original seed and artifacts.", "",
             "| Run | Status | Attempts | Recorded training transitions |", "|---|---|---:|---:|"]
    for row in rows:
        run = state["runs"][row["run_id"]]
        label = f"{row['task']} · {row['arm']} · d={row['reward_delay']} · seed {row['seed']}"
        if (root / row["run_id"] / "journal.md").exists():
            label = f"[{label}]({row['run_id']}/journal.md)"
        lines.append(f"| {label} | {run['status']} | {run.get('attempts', 0)} | {run.get('observed_training_steps', 0):,} |")
    atomic_write(root / "progress.md", "\n".join(lines) + "\n")
