#!/usr/bin/env python3
"""Package retained, hash-verified checkpoints for a versioned release."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilot"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="pilot-v1-checkpoints")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    root.relative_to(repo)
    files, runs = [], []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata["status"] != "completed":
            raise ValueError(f"Incomplete run: {metadata_path.parent.name}")
        folder = metadata_path.parent
        with (folder / "checkpoint_manifest.csv").open() as handle:
            # latest.pt is replaced at every checkpoint; only its final hash applies.
            retained = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
        for name, expected in retained.items():
            if Path(name).name != name:
                raise ValueError("Checkpoint manifest contains a nonlocal filename")
            path = folder / "checkpoints" / name
            actual = digest(path)
            if actual != expected:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            files.append({"path": str(path.relative_to(repo)), "sha256": actual,
                          "bytes": path.stat().st_size, "kind": "checkpoint"})
        log = folder / "console.log"
        if log.exists():
            files.append({"path": str(log.relative_to(repo)), "sha256": digest(log),
                          "bytes": log.stat().st_size, "kind": "console_log"})
        runs.append({"run_id": metadata["run_id"], "source_sha256": metadata["source_sha256"]})
    if not runs:
        raise ValueError("No completed runs found")
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / f"{args.name}.tar.gz"
    manifest = args.out / f"{args.name}-manifest.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError("Use a new output directory or release name; archives are immutable")
    manifest.write_text(json.dumps({"runs": runs, "files": files,
                                   "restore": "Extract the archive from the repository root."}, indent=2) + "\n")
    with tarfile.open(archive, "w:gz", compresslevel=1) as output:
        for record in files:
            output.add(repo / record["path"], arcname=record["path"], recursive=False)
        output.add(manifest, arcname=manifest.name, recursive=False)
    checksums = args.out / "SHA256SUMS"
    checksums.write_text("".join(f"{digest(p)}  {p.name}\n" for p in (archive, manifest)))
    print(json.dumps({"runs": len(runs), "files": len(files), "archive": str(archive),
                      "archive_mib": archive.stat().st_size / 2**20,
                      "manifest": str(manifest), "checksums": str(checksums)}))


if __name__ == "__main__":
    main()
