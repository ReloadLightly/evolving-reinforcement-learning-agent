#!/usr/bin/env python3
"""Run eight bounded development checks; these are not confirmatory experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "erl").glob("*.py"))}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_run(folder, source, calibration_hash):
    m = json.loads((folder / "metadata.json").read_text())
    expected = dict(status="completed", purpose="validation", diagnostics="full",
                    training_budget=16000, warmup=10000, evaluation_interval=2000,
                    evaluation_episodes=2, total_training_transitions=16000,
                    total_evaluation_transitions=3600, critic_updates=1500, actor_updates=750,
                    source_sha256=source, calibration_sha256=calibration_hash)
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"{folder.name}: {key} mismatch")
    with (folder / "evaluation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(int(r["step"]), int(r["evaluation_episode"])): float(r["episode_return"]) for r in rows}
    if len(rows) != 18 or set(by_key) != {(step, ep) for step in range(0, 16001, 2000) for ep in range(2)}:
        raise ValueError("Incomplete or duplicate validation evaluation grid")
    if not all(math.isfinite(v) for v in by_key.values()):
        raise ValueError("Nonfinite validation evaluation")
    for ep in range(2):
        if any(by_key[step, ep] != by_key[0, ep] for step in range(0, 10001, 2000)):
            raise ValueError("Actor changed during warmup")
    if sum(m["sampled_transitions"].values()) != 192000:
        raise ValueError("Training replay draw count mismatch")
    if m["arm"] == "erl_balanced" and m["sampled_transitions"] != {"ea": 96000, "learner": 96000}:
        raise ValueError("Balanced source quotas differ")
    for name in ("diagnostic_replay.csv", "diagnostic_critic.csv"):
        with (folder / name).open() as handle:
            diagnostic_rows = list(csv.DictReader(handle))
        if not diagnostic_rows or max(int(r["step"]) for r in diagnostic_rows) != 16000:
            raise ValueError(f"Missing final independent diagnostics: {name}")
    evidence = {p.name: sha(p) for p in sorted(folder.glob("*"))
                if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".md")}
    return dict(run_id=m["run_id"], arm=m["arm"], delay=m["reward_delay"], seed=m["seed"],
                training_transitions=16000, evaluation_transitions=3600,
                critic_updates=1500, actor_updates=750, elapsed_s=m["elapsed_s"],
                final_return=sum(by_key[16000, ep] for ep in range(2))/2,
                initial_return=sum(by_key[0, ep] for ep in range(2))/2, artifacts_sha256=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/validation/next-stage")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    source = source_hashes()
    calibration = REPO / "results/pilot/calibration.json"
    for original in (calibration, *sorted(calibration.parent.glob("*-probe.npy"))):
        target = root / original.name
        if target.exists() and sha(target) != sha(original):
            raise ValueError(f"Changed validation calibration: {target}")
        if not target.exists(): shutil.copyfile(original, target)
    calibration_hash = sha(root / "calibration.json")
    conditions = [(arm, delay, seed) for arm in ("erl_balanced", "erl_pooled")
                  for delay in (1, 20) for seed in (10, 11)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.perf_counter(); completed = []; failures = []

    def execute(condition):
        arm, delay, seed = condition
        name = f"first_stage_validation__Pendulum-v1__{arm}__d{delay:03d}__s{seed:03d}"
        folder = root / name
        if (folder / "metadata.json").exists():
            return check_run(folder, source, calibration_hash)
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "erl.train", "--task", "Pendulum-v1", "--arm", arm,
                   "--delay", str(delay), "--seed", str(seed), "--steps", "16000",
                   "--warmup", "10000", "--eval-every", "2000", "--eval-episodes", "2",
                   "--diagnostics", "full", "--purpose", "validation",
                   "--calibration", str(root / "calibration.json"), "--out", str(folder)]
        print(f"START development check: {arm}, delay {delay}, seed {seed}", flush=True)
        with (folder / "console.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name}: trainer exit {result.returncode}; inspect console.log")
        return check_run(folder, source, calibration_hash)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(execute, condition): condition for condition in conditions}
        for future in as_completed(pending):
            try:
                result = future.result(); completed.append(result)
                print(f"DONE {len(completed)}/8 {result['run_id']}", flush=True)
            except Exception as error:
                failures.append(dict(condition=pending[future], error=str(error)))
                print(f"FAILED {pending[future]}: {error}", flush=True)
    if source_hashes() != source:
        failures.append(dict(error="Training source changed during validation"))
    report = dict(status="passed" if len(completed)==8 and not failures else "failed",
                  purpose="small development validation; excluded from confirmatory inference",
                  source_sha256=source, calibration_sha256=calibration_hash,
                  workers=args.workers, wall_time_s=time.perf_counter()-started,
                  runs=sorted(completed, key=lambda r:r["run_id"]), failures=failures,
                  total_training_transitions=sum(r["training_transitions"] for r in completed),
                  total_evaluation_transitions=sum(r["evaluation_transitions"] for r in completed))
    save(root / "smoke-checks.json", report)
    lines = ["# First-stage development validation", "", "Eight small CPU checks use development seeds 10 and 11, both replay arms, and both reward delays. Each collects 16,000 training transitions, retains the 10,000-step warmup, and performs 1,500 twin-critic plus 750 actor updates. They are implementation checks, not a shortened test of H1. No method ranking or inferential interval is produced.", "", f"Status: **{report['status']}**. Completed {len(completed)}/8; failures {len(failures)}. Recorded execution time: {report['wall_time_s']/60:.2f} minutes.", "", "| Run journal | Initial return | Final return |", "|---|---:|---:|"]
    for r in report["runs"]:
        lines.append(f"| [{r['arm']} · delay {r['delay']} · seed {r['seed']}]({r['run_id']}/journal.md) | {r['initial_return']:.2f} | {r['final_return']:.2f} |")
    lines += ["", "Each return averages two evaluation episodes. The short optimization budget is not intended to establish learnability under delayed feedback. Full independent diagnostics, exact budget checks, and warmup invariance were checked in every completed run. The accompanying test-suite report separately verifies isolation, resume behavior, and the statistical pipeline."]
    (root / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":report["status"], "completed":len(completed), "failures":len(failures)}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
