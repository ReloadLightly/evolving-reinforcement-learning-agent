#!/usr/bin/env python3
"""Run the 24 development experiments with bounded CPU concurrency."""
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from study_plan import DEFAULT_CONFIG, validate


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results" / "pilot")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", action="store_true",
                        help="Only run seed-zero TD3 on each task; these runs remain part of the pilot")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("This CPU pilot supports one or two workers")
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if not (args.root / "calibration.json").exists():
        parser.error("Run python -m erl.calibrate --out ROOT/calibration.json first")
    calibration_hash = hashlib.sha256((args.root / "calibration.json").read_bytes()).hexdigest()
    config = json.loads(DEFAULT_CONFIG.read_text())
    manifests, _, _ = validate(config)
    rows = manifests["pilot"]
    if args.profile:
        rows = [r for r in rows if r["arm"] == "td3" and r["seed"] == 0]
    source = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((REPO / "erl").glob("*.py"))}
    source_id = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    snapshot = args.root / "source" / source_id
    for name in source:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(REPO / name, dest)
    for source_path, name in ((DEFAULT_CONFIG, "study.json"), (REPO / "requirements.txt", "requirements.txt")):
        target = snapshot / name
        if target.exists() and target.read_bytes() != source_path.read_bytes():
            parser.error(f"Existing snapshot input differs: {name}; preserve it and use a new output root")
        if not target.exists():
            shutil.copyfile(source_path, target)
    # Reject incompatible existing artifacts before any new run consumes compute.
    for row in rows:
        previous = args.root / row["run_id"] / "metadata.json"
        if previous.exists():
            prior = json.loads(previous.read_text())
            if prior.get("calibration_sha256") != calibration_hash:
                parser.error(f"Calibration changed for existing run {row['run_id']}; investigate before launching")
    with (args.root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests["pilot"][0]))
        writer.writeheader()
        writer.writerows(manifests["pilot"])
    invocation = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                      workers=args.workers, profile=args.profile, source_sha256=source,
                      source_snapshot=str(snapshot.relative_to(args.root)),
                      note="Already completed matching pilot runs are reused, never trained twice.")
    with (args.root / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps(invocation) + "\n")
    start = time.perf_counter()

    def execute(row):
        folder = args.root / row["run_id"]
        metadata = folder / "metadata.json"
        if metadata.exists():
            prior = json.loads(metadata.read_text())
            if prior.get("status") == "completed" and prior.get("source_sha256") == source and prior.get("calibration_sha256") == calibration_hash:
                for field, value in (("training_budget", 50_000), ("warmup", 10_000),
                                     ("evaluation_interval", 10_000), ("evaluation_episodes", 5),
                                     ("task", row["task"]), ("arm", row["arm"]), ("seed", row["seed"]),
                                     ("reward_delay", 1)):
                    if prior[field] != value:
                        raise ValueError(f"Completed run mismatch: {row['run_id']} {field}")
                return dict(run_id=row["run_id"], status="reused", elapsed_s=prior["elapsed_s"])
            raise ValueError(f"Existing incomplete or changed-source run needs explicit investigation: {folder}")
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "erl.train", "--task", row["task"],
                   "--arm", row["arm"], "--seed", str(row["seed"]), "--delay", "1",
                   "--steps", "50000", "--out", str(folder),
                   "--calibration", str(args.root / "calibration.json")]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MPLCONFIGDIR="/tmp/erl-mplcache")
        print(f"START {row['task']} / {row['arm']} / seed {row['seed']} → {folder.name}/journal.md", flush=True)
        with (folder / "console.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = (folder / "console.log").read_text()[-3000:]
            raise RuntimeError(f"{row['run_id']} failed with exit {result.returncode}:\n{tail}")
        final = json.loads(metadata.read_text())
        return dict(run_id=row["run_id"], status=final["status"], elapsed_s=final["elapsed_s"])

    completed = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print(f"DONE {len(completed)}/{len(rows)} {result['run_id']}: "
                      f"{result['status']} in {result['elapsed_s']:.1f}s", flush=True)
            except Exception as error:
                failures.append(dict(run_id=row["run_id"], error=str(error)))
                print(f"FAILED {row['run_id']}: {error}", flush=True)
    summary = dict(completed=completed, failures=failures,
                   wall_time_s=time.perf_counter() - start, workers=args.workers, profile=args.profile)
    name = "profile_execution.json" if args.profile else "pilot_execution.json"
    (args.root / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(completed=len(completed), failures=len(failures),
                         wall_time_s=summary["wall_time_s"]), indent=2), flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
