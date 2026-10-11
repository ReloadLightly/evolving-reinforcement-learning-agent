#!/usr/bin/env python3
"""Execute the immutable 40-run study, retaining failures and attempt costs.

The runner never chooses settings or seeds from evaluation performance. Use
--max-runs for an explicitly bounded invocation; it does not shorten the plan.
An interrupted run resumes only with --resume-incomplete and its own checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from study_common import (REPO, initial_lifecycle, persist_lifecycle, require,
                          software_environment, utc_now, validate_metadata, verify_freeze, write_json)


def append_event(path, record):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def observed_work(folder, checkpoint_only=False):
    """Use durable log rows, not Torch checkpoint deserialization, for the ledger."""
    checkpoint = 0
    for row in read_rows(folder / "checkpoint_manifest.csv"):
        if row.get("kind") == "latest_full" and (row.get("step") or "").isdigit():
            checkpoint = max(checkpoint, int(row["step"]))
    training = checkpoint if checkpoint_only else 0
    if not checkpoint_only:
        for row in read_rows(folder / "episodes.csv"):
            if (row.get("step") or "").isdigit():
                training = max(training, int(row["step"]))
    evaluation = 0
    for row in read_rows(folder / "evaluation.csv"):
        if (row.get("step") or "").isdigit() and (row.get("episode_length") or "").isdigit():
            if not checkpoint_only or int(row["step"]) <= checkpoint:
                evaluation += int(row["episode_length"])
    return dict(training_steps=training, evaluation_steps=evaluation, checkpoint_steps=checkpoint)


def command_for(row, freeze, root, resume=False, python=sys.executable):
    command = [str(python), "-m", "erl.train", "--task", row["task"], "--arm", row["arm"],
        "--seed", str(row["seed"]), "--delay", str(row["reward_delay"]),
        "--steps", str(row["training_transitions"]), "--warmup", str(row["warmup_transitions"]),
        "--eval-every", str(row["evaluation_interval"]),
        "--eval-episodes", str(row["evaluation_episodes_per_checkpoint"]),
        "--out", str(Path(root) / row["run_id"]),
        "--calibration", str(Path(root) / freeze["snapshot"] / "calibration.json"),
        "--purpose", "confirmatory", "--diagnostics", "full", "--protocol-sha256", freeze["freeze_id"]]
    if resume:
        command.append("--resume")
    return command


def validate_environment(freeze):
    current = software_environment()
    for key in ("python", "implementation", "platform", "packages"):
        require(current[key] == freeze["software"][key],
                f"Software {key} differs from the frozen environment; investigate before execution")


def classify_existing(root, row, freeze, resume_incomplete):
    folder = root / row["run_id"]
    path = folder / "metadata.json"
    if not path.exists():
        require(not (folder / "checkpoints/latest.pt").exists(),
                f"Checkpoint without metadata requires investigation: {folder}")
        return "pending", False, None
    metadata = json.loads(path.read_text())
    validate_metadata(metadata, row, freeze, completed=metadata.get("status") == "completed")
    if metadata["status"] == "completed":
        return "completed", False, metadata
    # Numerical failures are experimental outcomes, not infrastructure retries.
    if metadata.get("error_type") in ("FloatingPointError", "AssertionError", "ValueError"):
        return "failed", False, metadata
    if resume_incomplete:
        require((folder / "checkpoints/latest.pt").is_file(),
                f"Cannot resume {row['run_id']}: no durable checkpoint; preserve artifacts and investigate")
        return "pending", True, metadata
    return ("failed" if metadata.get("status") == "failed" else "interrupted"), False, metadata


def cumulative_ledger(root):
    records = []
    path = root / "attempts.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    closed = [record for record in records if record["event"] in ("attempt_finished", "attempt_interrupted")]
    return dict(attempts_started=sum(r["event"] == "attempt_started" for r in records),
        attempts_closed=len(closed),
        recorded_training_transitions=sum(r.get("recorded_training_transitions", 0) for r in closed),
        recorded_evaluation_transitions=sum(r.get("recorded_evaluation_transitions", 0) for r in closed),
        recorded_process_wall_seconds=sum(r.get("elapsed_s", 0.0) for r in closed),
        interrupted_attempts=sum(r["event"] == "attempt_interrupted" for r in closed),
        has_incomplete_cost_observations=any(r.get("cost_is_lower_bound", False) for r in closed),
        note="Costs sum attempts, including repeated work after resume. Abruptly lost/unlogged work is unknown; flagged totals are lower bounds. Concurrent process times overlap.")


def recover_abandoned_attempt(root, row, run, metadata):
    if run.get("status") != "running":
        return
    require(not process_alive(run.get("pid")),
            f"A previous worker is still alive for {row['run_id']} (PID {run['pid']}); do not launch a duplicate")
    work = observed_work(root / row["run_id"])
    append_event(root / "attempts.jsonl", dict(event="attempt_interrupted", timestamp_utc=utc_now(),
        run_id=row["run_id"], attempt_id=run.get("attempt_id"),
        status="completed" if metadata and metadata.get("status") == "completed" else "interrupted",
        recorded_training_transitions=max(0, work["training_steps"] - run.get("attempt_base_training", 0)),
        recorded_evaluation_transitions=max(0, work["evaluation_steps"] - run.get("attempt_base_evaluation", 0)),
        elapsed_s=run.get("attempt_observed_elapsed_s", 0.0), cost_is_lower_bound=True,
        note="Recovered after controller interruption; last logged counters and elapsed heartbeat retained."))
    run.update(status="interrupted", pid=None)


def execute(root, workers=1, max_runs=None, resume_incomplete=False, dry_run=False):
    root = Path(root).resolve()
    freeze, rows = verify_freeze(root)
    validate_environment(freeze)
    require(type(workers) is int and 1 <= workers <= 2, "Use one or two CPU workers")
    require(max_runs is None or type(max_runs) is int and max_runs > 0, "max-runs must be positive")
    lifecycle_path = root / "lifecycle.json"
    state = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else initial_lifecycle(freeze, rows)
    require(state["freeze_id"] == freeze["freeze_id"]
            and set(state["runs"]) == {r["run_id"] for r in rows}, "Lifecycle does not match immutable manifest")
    # Validate every existing run before launching anything, even a bounded prefix.
    candidates, classifications = [], []
    for row in rows:
        status, resume, metadata = classify_existing(root, row, freeze, resume_incomplete)
        run = state["runs"][row["run_id"]]
        if run.get("status") == "running":
            require(not process_alive(run.get("pid")), f"Worker remains active for {row['run_id']}")
        classifications.append((row, status, resume, metadata))
        if status == "pending":
            candidates.append((row, resume))
    if max_runs is not None:
        candidates = candidates[:max_runs]
    if dry_run:
        result = dict(freeze_id=freeze["freeze_id"], planned_runs=len(rows), launching=len(candidates),
                      workers=workers, dry_run=True, commands=[command_for(row, freeze, root, resume) for row, resume in candidates])
        print(json.dumps(result, indent=2))
        return 0

    lock_path = root / ".runner.lock"
    if lock_path.exists():
        prior = json.loads(lock_path.read_text())
        require(prior.get("host") == socket.gethostname() and not process_alive(prior.get("pid")),
                "Another study controller may be active; inspect .runner.lock before continuing")
        lock_path.unlink()
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), started_utc=utc_now()), handle)
    invocation_id = utc_now()
    started = time.perf_counter()
    active = {}
    interrupted = False
    try:
        for row, status, resume, metadata in classifications:
            run = state["runs"][row["run_id"]]
            recover_abandoned_attempt(root, row, run, metadata)
            run.update(status=status, pid=None)
            if metadata:
                work = observed_work(root / row["run_id"])
                run["observed_training_steps"] = work["training_steps"]
                if status == "failed":
                    run["last_error"] = metadata.get("error", "Recorded failure; retained without replacement")
        persist_lifecycle(root, state, rows)
        append_event(root / "invocations.jsonl", dict(event="invocation_started", invocation_id=invocation_id,
            timestamp_utc=utc_now(), freeze_id=freeze["freeze_id"], workers=workers,
            max_runs=max_runs, resume_incomplete=resume_incomplete, selected_runs=len(candidates)))
        queue = list(candidates)
        last_refresh = 0.0
        while queue or active:
            while queue and len(active) < workers:
                # Guards catch edits between launches; loaded workers keep their original code.
                verify_freeze(root)
                row, resume = queue.pop(0)
                folder = root / row["run_id"]
                folder.mkdir(exist_ok=True)
                run = state["runs"][row["run_id"]]
                run["attempts"] = run.get("attempts", 0) + 1
                attempt_id = f"{row['run_id']}__attempt{run['attempts']:03d}"
                baseline = observed_work(folder, checkpoint_only=True) if resume else dict(training_steps=0, evaluation_steps=0)
                log = (folder / f"attempt-{run['attempts']:03d}.log").open("w")
                command = command_for(row, freeze, root, resume)
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           MPLCONFIGDIR="/tmp/erl-mplcache")
                attempt_started = time.perf_counter()
                run.update(status="running", pid=None, attempt_id=attempt_id,
                           attempt_base_training=baseline["training_steps"],
                           attempt_base_evaluation=baseline["evaluation_steps"], attempt_observed_elapsed_s=0.0,
                           started_utc=utc_now(), last_error="")
                append_event(root / "attempts.jsonl", dict(event="attempt_started", timestamp_utc=utc_now(),
                    invocation_id=invocation_id, run_id=row["run_id"], attempt_id=attempt_id,
                    resumed=resume, baseline=baseline, command=command))
                try:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                except OSError as failure:
                    log.close()
                    run.update(status="failed", pid=None, last_error=str(failure))
                    append_event(root / "attempts.jsonl", dict(event="attempt_finished", timestamp_utc=utc_now(),
                        invocation_id=invocation_id, run_id=row["run_id"], attempt_id=attempt_id,
                        status="failed", error=str(failure), exit_code=None,
                        elapsed_s=time.perf_counter() - attempt_started, cost_is_lower_bound=False,
                        recorded_training_transitions=0, recorded_evaluation_transitions=0))
                    persist_lifecycle(root, state, rows)
                    continue
                run["pid"] = process.pid
                active[row["run_id"]] = (row, process, log, attempt_started)
                persist_lifecycle(root, state, rows)
                print(f"START {row['run_id']} {'(resume)' if resume else ''}; teaching journal: {folder / 'journal.md'}", flush=True)
            for run_id, (row, process, log, attempt_started) in list(active.items()):
                run = state["runs"][run_id]
                work = observed_work(root / run_id)
                run.update(observed_training_steps=work["training_steps"],
                           observed_evaluation_steps=work["evaluation_steps"],
                           attempt_observed_elapsed_s=time.perf_counter() - attempt_started)
                code = process.poll()
                if code is None:
                    continue
                log.close()
                # Read again after exit: a writer may have finished its last CSV
                # rows between the preceding observation and poll().
                work = observed_work(root / run_id)
                run.update(observed_training_steps=work["training_steps"],
                           observed_evaluation_steps=work["evaluation_steps"])
                status, error = "completed", ""
                try:
                    require(code == 0, f"Worker exited with code {code}; inspect attempt-{run['attempts']:03d}.log")
                    metadata = json.loads((root / run_id / "metadata.json").read_text())
                    validate_metadata(metadata, row, freeze, completed=True)
                except Exception as failure:
                    status, error = "failed", str(failure)
                append_event(root / "attempts.jsonl", dict(event="attempt_finished", timestamp_utc=utc_now(),
                    invocation_id=invocation_id, run_id=run_id, attempt_id=run["attempt_id"], status=status,
                    exit_code=code, error=error, elapsed_s=run["attempt_observed_elapsed_s"],
                    recorded_training_transitions=max(0, work["training_steps"] - run["attempt_base_training"]),
                    recorded_evaluation_transitions=max(0, work["evaluation_steps"] - run["attempt_base_evaluation"]),
                    cost_is_lower_bound=status != "completed"))
                run.update(status=status, pid=None, last_error=error, finished_utc=utc_now())
                del active[run_id]
                persist_lifecycle(root, state, rows)
                print(f"{status.upper()} {run_id}; {run['observed_training_steps']:,} recorded training transitions", flush=True)
            if time.perf_counter() - last_refresh >= 5:
                persist_lifecycle(root, state, rows)
                last_refresh = time.perf_counter()
            if active:
                time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupt received; preserving each worker's last durable checkpoint.", flush=True)
    finally:
        # Also stop workers if an integrity check fails between launches.
        for run_id, (row, process, log, attempt_started) in active.items():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            run = state["runs"][run_id]
            work = observed_work(root / run_id)
            append_event(root / "attempts.jsonl", dict(event="attempt_interrupted", timestamp_utc=utc_now(),
                invocation_id=invocation_id, run_id=run_id, attempt_id=run["attempt_id"], status="interrupted",
                elapsed_s=time.perf_counter() - attempt_started, cost_is_lower_bound=True,
                recorded_training_transitions=max(0, work["training_steps"] - run["attempt_base_training"]),
                recorded_evaluation_transitions=max(0, work["evaluation_steps"] - run["attempt_base_evaluation"])))
            run.update(status="interrupted", pid=None, observed_training_steps=work["training_steps"])
        persist_lifecycle(root, state, rows)
        summary = dict(freeze_id=freeze["freeze_id"], updated_utc=utc_now(), planned_runs=len(rows),
                       counts=state["counts"], cumulative_attempt_costs=cumulative_ledger(root))
        write_json(root / "execution.json", summary)
        append_event(root / "invocations.jsonl", dict(event="invocation_finished", invocation_id=invocation_id,
            timestamp_utc=utc_now(), elapsed_s=time.perf_counter() - started,
            counts=state["counts"], interrupted=interrupted))
        lock_path.unlink(missing_ok=True)
    return 130 if interrupted else int(any(r["status"] in ("failed", "interrupted") for r in state["runs"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/first-study")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without creating execution attempts")
    parser.add_argument("--max-runs", type=int, help="Bound this invocation; the immutable 40-run manifest remains unchanged")
    parser.add_argument("--resume-incomplete", action="store_true", help="Resume the same interrupted seeds from their own checkpoints")
    args = parser.parse_args()
    return execute(args.root, args.workers, args.max_runs, args.resume_incomplete, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
