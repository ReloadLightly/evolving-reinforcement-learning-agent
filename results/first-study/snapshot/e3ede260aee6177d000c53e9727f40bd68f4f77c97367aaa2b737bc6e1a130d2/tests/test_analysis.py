"""Synthetic fixtures live only in temporary directories, never research results."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_first import (ANALYSIS_SEED, ARMS, DELAYS, SEEDS, STEPS, TASK,
                           auc, audit_runs, completion_report, paired_statistics)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    freeze = dict(source_sha256={"erl/train.py": "synthetic-source"},
                  calibration_sha256="synthetic-anchor", freeze_id="synthetic-freeze")
    manifest = []
    for seed in SEEDS:
        for delay in DELAYS:
            for arm in ARMS:
                run_id = f"synthetic_{arm}_{delay}_{seed}"
                row = dict(run_id=run_id, task=TASK, arm=arm, reward_delay=delay, seed=seed)
                manifest.append(row)
                directory = root / run_id
                directory.mkdir()
                metadata = dict(row, status="completed", source_sha256=freeze["source_sha256"],
                                calibration_sha256=freeze["calibration_sha256"], protocol_sha256=freeze["freeze_id"],
                                purpose="confirmatory", diagnostics="full", device="cpu", training_budget=200000,
                                total_training_transitions=200000, warmup=10000, critic_updates=47500,
                                actor_updates=23750, evaluation_interval=10000, evaluation_episodes=5,
                                total_evaluation_transitions=21000, horizon=200, threads=1, batch_size=128,
                                replay_capacity=100000, sampled_transitions={"ea": 3040000, "learner": 3040000})
                (directory / "metadata.json").write_text(json.dumps(metadata))
                evaluation = [dict(task=TASK, arm=arm, delay=delay, seed=seed, step=int(step),
                                   evaluation_episode=episode, reset_seed=seed * 10 + episode,
                                   episode_return=-500 + max(0, int(step) - 10000) / 1000 - episode,
                                   episode_length=200) for step in STEPS for episode in range(5)]
                write_csv(directory / "evaluation.csv", evaluation)
                episodes = [dict(episode=episode, length=200, source="ea" if episode % 5 < 4 else "learner",
                                 native_return=-10, emitted_return=-10) for episode in range(1000)]
                write_csv(directory / "episodes.csv", episodes)
                generations = [dict(step=step, critic_updates=max(0, step - 10000) // 4,
                                    actor_updates=max(0, step - 10000) // 8) for step in range(1000, 200001, 1000)]
                write_csv(directory / "generations.csv", generations)
    return freeze, manifest


class MathematicsTests(unittest.TestCase):
    def test_auc_includes_initialization_and_warmup(self):
        np.testing.assert_allclose(auc([[0, 0, 10], [-100, -50, 0]], [0, 1, 2]), [2.5, -50])
        with self.assertRaises(ValueError):
            auc([1, np.nan, 3], [0, 1, 2])

    def test_bootstrap_preserves_seed_arm_and_delay_pairing(self):
        curves = np.empty((10, 2, 2, len(STEPS)))
        for seed in range(10):
            for delay in range(2):
                # Large shared seed effects must cancel before contrasts; correlated
                # per-seed effects must also cancel in the delay interaction.
                curves[seed, delay, 0] = -1000 - seed * 200
                curves[seed, delay, 1] = curves[seed, delay, 0] + seed + delay * 10
        stats = paired_statistics(curves, -100)
        np.testing.assert_allclose(stats["effects"]["H1_delayed"]["per_seed"], np.arange(10) / 100 + .1)
        np.testing.assert_allclose(stats["effects"]["H2_delay_interaction"]["per_seed"], .1, atol=1e-12)
        np.testing.assert_allclose(stats["effects"]["H2_delay_interaction"]["ci95"], [.1, .1], atol=1e-12)
        expected_indices = np.random.default_rng(ANALYSIS_SEED).integers(0, 10, size=(10000, 10))
        np.testing.assert_array_equal(stats["indices"], expected_indices)
        expected = (np.arange(10)[expected_indices] / 100 + .1).mean(axis=1)
        np.testing.assert_allclose(stats["effects"]["H1_delayed"]["ci95"], np.quantile(expected, [.025, .975]))

    def test_normalization_and_inconclusive_direction(self):
        curves = np.full((10, 2, 2, 21), -50.)
        for seed in range(10):
            curves[seed, :, 1] += (-1 if seed < 5 else 1) * 5
        stats = paired_statistics(curves, -100)
        np.testing.assert_allclose(stats["normalized_auc"][:, :, 0], .5)
        self.assertEqual(stats["effects"]["H1_delayed"]["direction"], "inconclusive")
        self.assertFalse(stats["effects"]["H1_delayed"]["clears_practical_threshold"])
        with self.assertRaises(ValueError):
            paired_statistics(curves, 0)


class StudyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="erl-analysis-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.freeze, self.manifest = fixture(self.root)
        self.directory = self.root / self.manifest[0]["run_id"]

    def metadata_change(self, **values):
        path = self.directory / "metadata.json"
        data = json.loads(path.read_text())
        data.update(values)
        path.write_text(json.dumps(data))

    def check_rejected(self, reason):
        completion, records = audit_runs(self.root, self.freeze, self.manifest)
        self.assertFalse(completion["complete"])
        self.assertFalse(completion["inferential_analysis_allowed"])
        self.assertLess(len(records), 40)
        self.assertTrue(any(reason in row["reason"] for row in completion["runs"]), completion)
        return completion

    def test_complete_manifest_and_declared_evaluation_seeds(self):
        resets = {(seed, ep): seed * 10 + ep for seed in SEEDS for ep in range(5)}
        completion, records = audit_runs(self.root, self.freeze, self.manifest, resets)
        self.assertTrue(completion["complete"])
        self.assertEqual(len(records), 40)
        resets[100, 0] += 1
        completion, _ = audit_runs(self.root, self.freeze, self.manifest, resets)
        self.assertFalse(completion["complete"])

    def test_missing_failed_runs_emit_no_inferential_claim(self):
        (self.directory / "metadata.json").unlink()
        second = self.root / self.manifest[1]["run_id"] / "metadata.json"
        meta = json.loads(second.read_text()); meta.update(status="failed", error="synthetic numerical failure")
        second.write_text(json.dumps(meta))
        completion = self.check_rejected("No metadata")
        self.assertEqual(len(completion["runs"]), 40)
        self.assertIn("failed", {row["status"] for row in completion["runs"]})
        completion_report(self.root, completion)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertIsNone(summary["effects"])
        self.assertIn("withheld", (self.root / "report.md").read_text())

    def test_duplicate_and_missing_evaluation_rows_rejected(self):
        path = self.directory / "evaluation.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        write_csv(path, rows + [rows[0]])
        self.check_rejected("Missing or duplicate evaluation")
        write_csv(path, rows[:-1])
        self.check_rejected("Missing or duplicate evaluation")

    def test_mixed_source_and_protocol_hashes_rejected(self):
        self.metadata_change(source_sha256={"erl/train.py": "changed"})
        self.check_rejected("source hashes")
        self.metadata_change(source_sha256=self.freeze["source_sha256"], protocol_sha256="changed")
        self.check_rejected("protocol/configuration")

    def test_reset_seed_and_update_budget_mismatch_rejected(self):
        path = self.directory / "evaluation.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows[5]["reset_seed"] = int(rows[5]["reset_seed"]) + 1
        write_csv(path, rows)
        self.check_rejected("reset seeds")
        self.metadata_change(critic_updates=47499)
        self.check_rejected("critic_updates")

    def test_duplicate_manifest_identity_rejected(self):
        manifest = self.manifest[:-1] + [self.manifest[0]]
        with self.assertRaisesRegex(ValueError, "40 identities"):
            audit_runs(self.root, self.freeze, manifest)


if __name__ == "__main__":
    unittest.main()
