#!/usr/bin/env python3
"""Seal the first study's protocol, executable sources, seeds, and calibration.

Freezing is a local, immutable declaration, not independent preregistration.
This command never launches training and never replaces an existing freeze.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from study_common import (REPO, atomic_write, build_manifest, canonical_bytes, digest,
                          initial_lifecycle, json_bytes, manifest_bytes, persist_lifecycle,
                          require, resolved_seeds, runtime_sources, sha_file,
                          software_environment, utc_now, validate_config, verify_freeze, write_json)


REQUIRED_FILES = (
    "configs/first-stage.json", "requirements.txt", "docs/protocol.md", "docs/first-stage-protocol.md",
    "scripts/study_common.py", "scripts/freeze_study.py", "scripts/run_study.py", "scripts/analyze_first.py",
    "scripts/validate_next_stage.py",
)


def freeze_study(root, validation_path, calibration_path, repo=REPO, environment=None):
    root, repo = Path(root), Path(repo)
    validation_path, calibration_path = Path(validation_path), Path(calibration_path)
    require(validation_path.is_file(), "A completed validation JSON report is required before freezing")
    require(calibration_path.is_file(), "The independent calibration artifact is required before freezing")
    if (root / "freeze.json").exists():
        freeze, _ = verify_freeze(root, repo=repo)
        require(sha_file(validation_path) == freeze["validation_sha256"], "Existing freeze has different validation evidence")
        require(sha_file(calibration_path) == freeze["calibration_sha256"], "Existing freeze has different calibration")
        return freeze
    require(not (root / "lifecycle.json").exists() and not list(root.glob("*/metadata.json")),
            "Cannot freeze a directory containing pre-existing study execution artifacts")
    source = runtime_sources(repo)
    require(source and "erl/train.py" in source, "No trainer source was found")
    validation = json.loads(validation_path.read_text())
    require(validation.get("status") == "passed", "Validation report must have status='passed'")
    require(validation.get("source_sha256") == source,
            "Validation source hashes do not match the trainer being frozen; revalidate changes first")
    config = json.loads((repo / "configs/first-stage.json").read_text())
    validate_config(config)
    files = {}
    repository_files = set(REQUIRED_FILES) | set(source)
    repository_files.update(str(p.relative_to(repo)) for p in (repo / "tests").glob("test_*.py"))
    for name in sorted(repository_files):
        path = repo / name
        require(path.is_file(), f"Required source/protocol file is not ready: {name}")
        files[name] = path.read_bytes()
    checked_files = validation.get("validated_files_sha256", {})
    for name in repository_files:
        if not name.startswith("docs/"):
            require(checked_files.get(name) == digest(files[name]),
                    f"Executable/configuration changed after validation or was not checked: {name}")
    files["validation.json"] = validation_path.read_bytes()
    files["calibration.json"] = calibration_path.read_bytes()
    calibration = json.loads(files["calibration.json"])
    tasks = calibration.get("tasks", calibration)
    require(config["task"] in tasks, "Calibration is missing Pendulum")
    for task, record in tasks.items():
        mean = record["random_return_mean"]
        require(math.isfinite(mean) and mean < 0, f"Invalid calibration anchor: {task}")
        require(record["episodes"] == 20 and len(record["returns"]) == 20,
                f"Calibration must preserve all twenty independent random-policy episodes: {task}")
        probe_name = record["probe_path"]
        require(Path(probe_name).name == probe_name and probe_name not in files,
                "Calibration probe must be a distinct sibling filename")
        probe_path = calibration_path.parent / probe_name
        require(probe_path.is_file() and sha_file(probe_path) == record["probe_sha256"],
                f"Calibration probe hash mismatch: {task}")
        files[probe_name] = probe_path.read_bytes()
    rows = build_manifest(config)
    files["manifest.csv"] = manifest_bytes(rows)
    files["seed-manifest.json"] = json_bytes(resolved_seeds(config))
    hashes = {name: digest(data) for name, data in sorted(files.items())}
    payload = dict(schema_version=1, created_utc=utc_now(), config=config,
        config_sha256=hashes["configs/first-stage.json"], source_sha256=source,
        files=hashes, repository_files=sorted(repository_files),
        calibration_sha256=hashes["calibration.json"], validation_sha256=hashes["validation.json"],
        manifest_sha256=hashes["manifest.csv"], seed_manifest_sha256=hashes["seed-manifest.json"],
        software=software_environment() if environment is None else environment,
        planned_runs=len(rows), total_training_transitions=sum(r["training_transitions"] for r in rows),
        total_evaluation_transitions=sum(r["evaluation_transition_budget"] for r in rows),
        declaration="Locally frozen before confirmatory execution; not an independent preregistration")
    freeze_id = digest(canonical_bytes(payload))
    freeze = dict(payload, freeze_id=freeze_id, snapshot=f"snapshot/{freeze_id}")
    snapshot = root / freeze["snapshot"]
    for name, data in files.items():
        destination = snapshot / name
        if destination.exists():
            require(destination.read_bytes() == data, f"Refusing to replace existing snapshot artifact: {name}")
        else:
            atomic_write(destination, data)
    # Verify the worktree stayed still throughout capture before publishing the seal.
    for name in repository_files:
        require(sha_file(repo / name) == hashes[name], f"Source changed while freezing: {name}")
    atomic_write(root / "manifest.csv", files["manifest.csv"])
    atomic_write(root / "seed-manifest.json", files["seed-manifest.json"])
    write_json(root / "freeze.json", freeze)
    persist_lifecycle(root, initial_lifecycle(freeze, rows), rows)
    verify_freeze(root, repo=repo)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/first-study")
    parser.add_argument("--validation", type=Path, required=True,
                        help="Successful checks.json whose source_sha256 matches current erl/*.py")
    parser.add_argument("--calibration", type=Path, default=REPO / "results/pilot/calibration.json")
    args = parser.parse_args()
    freeze = freeze_study(args.root, args.validation, args.calibration)
    print(f"Frozen {freeze['planned_runs']} runs: {freeze['freeze_id']}")
    print(f"Training: {freeze['total_training_transitions']:,}; evaluation: {freeze['total_evaluation_transitions']:,}")
    print(f"Snapshot: {args.root / freeze['snapshot']}")
    print("No training has been launched by this command.")


if __name__ == "__main__":
    main()
