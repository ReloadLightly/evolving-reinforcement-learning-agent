"""Fast integrity and orchestration checks; no scientific training is launched."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import freeze_study
import run_study
import study_common as common


class FrozenStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="erl-study-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        self.root = self.base / "study"
        self.calibration = self.base / "calibration.json"
        self.validation = self.base / "validation.json"
        self.config = json.loads((REPO / "configs/first-stage.json").read_text())
        for name in (*freeze_study.REQUIRED_FILES, "erl/train.py", "erl/core.py", "tests/test_fixture.py"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Frozen fixture, never executed.\n")
        common.write_json(self.repo / "configs/first-stage.json", self.config)
        probe = self.base / "Pendulum-v1-probe.npy"
        probe.write_bytes(b"test fixture for byte-preservation checks; not loaded as an array")
        common.write_json(self.calibration, {"Pendulum-v1": {
            "random_return_mean": -1000.0, "episodes": 20, "returns": [-1000.0] * 20,
            "probe_path": probe.name, "probe_sha256": common.sha_file(probe),
        }})
        checked = {str(path.relative_to(self.repo)): common.sha_file(path)
                   for path in self.repo.rglob("*") if path.is_file()
                   and not str(path.relative_to(self.repo)).startswith("docs/")}
        common.write_json(self.validation, dict(status="passed", source_sha256=common.runtime_sources(self.repo),
                                              validated_files_sha256=checked, tests=dict(passed=1)))
        self.environment = dict(python="fixture", implementation="fixture", packages={})

    def freeze(self):
        return freeze_study.freeze_study(self.root, self.validation, self.calibration,
                                         repo=self.repo, environment=self.environment)

    def verify(self, root):
        return common.verify_freeze(root, repo=self.repo)

    def metadata(self, row, freeze, status="completed"):
        return dict(status=status, task=row["task"], arm=row["arm"], seed=int(row["seed"]),
            reward_delay=int(row["reward_delay"]), training_budget=int(row["training_transitions"]),
            warmup=int(row["warmup_transitions"]), horizon=int(row["episode_horizon"]),
            evaluation_interval=int(row["evaluation_interval"]),
            evaluation_episodes=int(row["evaluation_episodes_per_checkpoint"]),
            source_sha256=freeze["source_sha256"], calibration_sha256=freeze["calibration_sha256"],
            protocol_sha256=freeze["freeze_id"], purpose="confirmatory", diagnostics="full",
            threads=1, device="cpu", total_training_transitions=200000,
            total_evaluation_transitions=21000, critic_updates=47500, actor_updates=23750,
            sampled_transitions={"ea": 3040000, "learner": 3040000}, elapsed_s=1.0)

    def test_manifest_and_seed_blocks_preserve_full_design(self):
        rows = common.build_manifest(self.config)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["run_id"] for row in rows}), 40)
        self.assertEqual(sum(row["training_transitions"] for row in rows), 8000000)
        self.assertEqual(sum(row["evaluation_transition_budget"] for row in rows), 840000)
        self.assertEqual(sum(row["critic_updates"] for row in rows), 1900000)
        self.assertEqual(sum(row["actor_updates"] for row in rows), 950000)
        seeds = common.resolved_seeds(self.config)
        self.assertEqual(len(seeds["blocks"]), 10)
        for block in seeds["blocks"]:
            self.assertEqual(len(block["evaluation"]), 5)
            self.assertEqual(len(block["training_reset"]), 200)
            self.assertTrue(all(len(generation) == 5 for generation in block["training_reset"]))
            training_seeds = {seed for generation in block["training_reset"] for seed in generation}
            self.assertFalse(set(block["evaluation"]).intersection(training_seeds))
            self.assertIn("diagnostic_sampling", block["named_seeds"])
            self.assertIn("diagnostic_target_noise", block["named_seeds"])
        modified = copy.deepcopy(self.config)
        modified["seeds"].pop()
        with self.assertRaises(ValueError):
            common.build_manifest(modified)

    def test_freeze_requires_validation_of_current_sources(self):
        common.write_json(self.validation, {"status": "failed", "source_sha256": common.runtime_sources(self.repo)})
        with self.assertRaisesRegex(ValueError, "status='passed'"):
            self.freeze()
        common.write_json(self.validation, {"status": "passed", "source_sha256": {}})
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.freeze()
        self.assertFalse((self.root / "freeze.json").exists())

    def test_freeze_is_idempotent_and_copies_calibration_without_changes(self):
        freeze = self.freeze()
        first_bytes = (self.root / "freeze.json").read_bytes()
        second = self.freeze()
        self.assertEqual(freeze["freeze_id"], second["freeze_id"])
        self.assertEqual(first_bytes, (self.root / "freeze.json").read_bytes())
        self.assertEqual((self.root / freeze["snapshot"] / "calibration.json").read_bytes(),
                         self.calibration.read_bytes())
        verified, rows = self.verify(self.root)
        self.assertEqual(verified["freeze_id"], freeze["freeze_id"])
        self.assertEqual(len(rows), 40)
        lifecycle = json.loads((self.root / "lifecycle.json").read_text())
        self.assertEqual(lifecycle["counts"]["pending"], 40)

    def test_analysis_change_after_validation_requires_rechecking(self):
        (self.repo / "scripts/analyze_first.py").write_text("# Altered inference after validation.\n")
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            self.freeze()
        self.assertFalse((self.root / "freeze.json").exists())

    def test_changed_source_cannot_execute_but_archive_remains_verifiable(self):
        self.freeze()
        (self.repo / "erl/train.py").write_text("# Changed after validation/freeze.\n")
        with self.assertRaisesRegex(ValueError, "Worktree changed"):
            self.verify(self.root)
        common.verify_freeze(self.root, check_worktree=False, repo=self.repo)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_snapshot_and_manifest_tampering_are_rejected(self):
        freeze = self.freeze()
        path = self.root / freeze["snapshot"] / "docs/first-stage-protocol.md"
        original = path.read_bytes()
        path.write_text("Changed protocol")
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            self.verify(self.root)
        path.write_bytes(original)
        manifest = self.root / "manifest.csv"
        manifest.write_text(manifest.read_text().replace("200000", "100000", 1))
        with self.assertRaisesRegex(ValueError, "manifest.csv changed"):
            self.verify(self.root)

    def test_freeze_identity_detects_rewritten_configuration(self):
        self.freeze()
        path = self.root / "freeze.json"
        value = json.loads(path.read_text())
        value["config"]["analysis"]["practical_effect_threshold"] = 0.0
        common.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "Freeze identity"):
            self.verify(self.root)

    def test_dry_run_is_bounded_without_mutating_plan_or_lifecycle(self):
        self.freeze()
        before = {name: (self.root / name).read_bytes()
                  for name in ("manifest.csv", "freeze.json", "lifecycle.json")}
        output = io.StringIO()
        with mock.patch.object(run_study, "verify_freeze", side_effect=self.verify), \
             mock.patch.object(run_study, "validate_environment"), \
             mock.patch.object(run_study.subprocess, "Popen") as launch, \
             contextlib.redirect_stdout(output):
            code = run_study.execute(self.root, workers=2, max_runs=3, dry_run=True)
        self.assertEqual(code, 0)
        launch.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["planned_runs"], 40)
        self.assertEqual(plan["launching"], 3)
        for command in plan["commands"]:
            self.assertIn("confirmatory", command)
            self.assertIn("full", command)
            self.assertIn("--protocol-sha256", command)
        for name, original in before.items():
            self.assertEqual((self.root / name).read_bytes(), original)
        self.assertFalse((self.root / "attempts.jsonl").exists())

    def test_bounded_execution_updates_lifecycle_and_accounts_attempts(self):
        freeze = self.freeze()
        _, rows = self.verify(self.root)
        by_id = {row["run_id"]: row for row in rows}
        executed = []

        def fake_worker(command, **kwargs):
            folder = Path(command[command.index("--out") + 1])
            row = by_id[folder.name]
            executed.append(folder.name)
            common.write_json(folder / "metadata.json", self.metadata(row, freeze))
            (folder / "episodes.csv").write_text("step,length\n200000,200000\n")
            (folder / "evaluation.csv").write_text("step,episode_length\n200000,21000\n")
            process = mock.Mock()
            process.pid = 999999999
            process.poll.return_value = 0
            return process

        with mock.patch.object(run_study, "verify_freeze", side_effect=self.verify), \
             mock.patch.object(run_study, "validate_environment"), \
             mock.patch.object(run_study.subprocess, "Popen", side_effect=fake_worker), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_study.execute(self.root, workers=2, max_runs=2), 0)
            self.assertEqual(run_study.execute(self.root, workers=1, max_runs=1), 0)
        self.assertEqual(len(executed), 3)
        self.assertEqual(len(set(executed)), 3)
        state = json.loads((self.root / "lifecycle.json").read_text())
        self.assertEqual(state["counts"]["completed"], 3)
        self.assertEqual(state["counts"]["pending"], 37)
        self.assertEqual(len(self.verify(self.root)[1]), 40)
        ledger = json.loads((self.root / "execution.json").read_text())["cumulative_attempt_costs"]
        self.assertEqual(ledger["attempts_started"], 3)
        self.assertEqual(ledger["recorded_training_transitions"], 600000)
        self.assertEqual(ledger["recorded_evaluation_transitions"], 63000)

    def test_resume_uses_same_checkpoint_and_retains_numerical_failures(self):
        freeze = self.freeze()
        row = self.verify(self.root)[1][0]
        folder = self.root / row["run_id"]
        folder.mkdir()
        metadata = self.metadata(row, freeze, status="running")
        common.write_json(folder / "metadata.json", metadata)
        status, resume, _ = run_study.classify_existing(self.root, row, freeze, False)
        self.assertEqual((status, resume), ("interrupted", False))
        with self.assertRaisesRegex(ValueError, "no durable checkpoint"):
            run_study.classify_existing(self.root, row, freeze, True)
        (folder / "checkpoints").mkdir()
        (folder / "checkpoints/latest.pt").write_bytes(b"not deserialized by orchestration")
        status, resume, _ = run_study.classify_existing(self.root, row, freeze, True)
        self.assertEqual((status, resume), ("pending", True))
        self.assertIn("--resume", run_study.command_for(row, freeze, self.root, resume=True))
        metadata.update(status="failed", error_type="FloatingPointError")
        common.write_json(folder / "metadata.json", metadata)
        self.assertEqual(run_study.classify_existing(self.root, row, freeze, True)[:2], ("failed", False))

    def test_worker_launch_failure_is_recorded_without_replacing_seed(self):
        self.freeze()
        with mock.patch.object(run_study, "verify_freeze", side_effect=self.verify), \
             mock.patch.object(run_study, "validate_environment"), \
             mock.patch.object(run_study.subprocess, "Popen", side_effect=OSError("fixture launch failure")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_study.execute(self.root, max_runs=1), 1)
        state = json.loads((self.root / "lifecycle.json").read_text())
        self.assertEqual(state["counts"]["failed"], 1)
        self.assertEqual(state["counts"]["pending"], 39)
        ledger = run_study.cumulative_ledger(self.root)
        self.assertEqual(ledger["attempts_started"], 1)
        self.assertEqual(ledger["attempts_closed"], 1)
        self.assertEqual(ledger["recorded_training_transitions"], 0)

    def test_mismatched_existing_run_is_rejected_before_launch(self):
        freeze = self.freeze()
        row = self.verify(self.root)[1][-1]
        folder = self.root / row["run_id"]
        folder.mkdir()
        metadata = self.metadata(row, freeze)
        metadata["calibration_sha256"] = "0" * 64
        common.write_json(folder / "metadata.json", metadata)
        with mock.patch.object(run_study, "verify_freeze", side_effect=self.verify), \
             mock.patch.object(run_study, "validate_environment"), \
             mock.patch.object(run_study.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "calibration_sha256"):
                run_study.execute(self.root, max_runs=1)
        launch.assert_not_called()

    def test_cost_ledger_counts_replayed_work_after_interruption(self):
        self.root.mkdir()
        for record in (
            dict(event="attempt_started"),
            dict(event="attempt_interrupted", recorded_training_transitions=15000,
                 recorded_evaluation_transitions=2000, elapsed_s=10.0, cost_is_lower_bound=True),
            dict(event="attempt_started"),
            dict(event="attempt_finished", recorded_training_transitions=190000,
                 recorded_evaluation_transitions=19000, elapsed_s=100.0, cost_is_lower_bound=False),
        ):
            run_study.append_event(self.root / "attempts.jsonl", record)
        ledger = run_study.cumulative_ledger(self.root)
        self.assertEqual(ledger["recorded_training_transitions"], 205000)
        self.assertEqual(ledger["recorded_process_wall_seconds"], 110.0)
        self.assertTrue(ledger["has_incomplete_cost_observations"])


if __name__ == "__main__":
    unittest.main()
