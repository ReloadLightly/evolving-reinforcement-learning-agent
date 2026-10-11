"""Small subprocess integration checks; these are not scientific study runs.

Each subprocess owns Torch's one-time inter-op thread configuration. Temporary
artifacts disappear after the tests. The interruption occurs after learning has
begun, so restart checks cover optimizer, mutation, and target-noise state.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = r'''
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from erl.train import Experiment

mode, output, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
task, warmup = sys.argv[4], int(sys.argv[5])
args = SimpleNamespace(task=task, arm="erl_balanced", seed=73,
    delay=20, steps=total, warmup=warmup, eval_every=warmup, eval_episodes=2,
    out=output, resume=mode == "resume")
experiment = Experiment(args)

if mode == "interrupted":
    original_evaluate = experiment.evaluate
    def evaluate_then_interrupt():
        original_evaluate()
        if experiment.steps == 2000:
            raise SystemExit(0)
    experiment.evaluate = evaluate_then_interrupt

if mode == "isolation":
    experiment.collect_generation()
    def snapshot():
        return copy.deepcopy(dict(
            agent=experiment.agent.state_dict(), replay=experiment.replay.state_dict(),
            population=[p.state_dict() for p in experiment.population],
            rngs={k: v.bit_generator.state for k, v in experiment.rngs.items()},
            target_generator=experiment.target_generator.get_state(),
            torch_rng=torch.random.get_rng_state(),
            env_rng=experiment.env.unwrapped.np_random.bit_generator.state,
            env_state=experiment.env.unwrapped.state,
            steps=experiment.steps, episode=experiment.episode,
            generation=experiment.generation, update_credit=experiment.update_credit,
        ))
    before = snapshot()
    experiment.evaluate()
    after = snapshot()
    torch.save(dict(before=before, after=after), Path(output) / "isolation.pt")
    experiment.env.close()
    experiment.eval_env.close()
else:
    experiment.run()
'''


class TrainingIntegrationTests(unittest.TestCase):
    def run_subprocess(self, mode, directory, steps=2000, task="Pendulum-v1", warmup=1000):
        completed = subprocess.run(
            [sys.executable, "-c", RUNNER, mode, str(directory), str(steps), task, str(warmup)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def assert_nested_equal(self, first, second, path="state"):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second), path)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second, err_msg=path)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys(), path)
            for key in first:
                self.assert_nested_equal(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second), path)
            for index, (a, b) in enumerate(zip(first, second)):
                self.assert_nested_equal(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(first, second, path)

    def test_balanced_delayed_smoke_has_exact_budgets_and_reward_accounting(self):
        with tempfile.TemporaryDirectory(prefix="erl-smoke-") as temporary:
            out = Path(temporary) / "run"
            self.run_subprocess("normal", out)
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["total_training_transitions"], 2000)
            self.assertEqual(metadata["total_evaluation_transitions"], 1200)
            self.assertEqual(metadata["critic_updates"], 250)
            self.assertEqual(metadata["actor_updates"], 125)
            self.assertEqual(metadata["sampled_transitions"], {"ea": 16000, "learner": 16000})
            state = torch.load(out / "checkpoints/latest.pt", weights_only=False)
            replay = state["replay"]
            np.testing.assert_array_equal(replay["counts"], [400, 1600])
            self.assertEqual(replay["total_inserted"], 2000)
            self.assertEqual(state["update_credit"], 0.0)
            # Pending rewards emit every 20th transition; 200-step episodes have
            # no extra partial block. Native rewards must never leak into replay.
            rewards = replay["data"]["rewards"][:2000].reshape(-1)
            self.assertTrue(np.all(rewards[np.arange(2000) % 20 != 19] == 0.0))
            self.assertTrue(np.all(rewards[np.arange(2000) % 20 == 19] < 0.0))
            with (out / "episodes.csv").open() as handle:
                episodes = list(csv.DictReader(handle))
            self.assertEqual(len(episodes), 10)
            for episode in episodes:
                self.assertAlmostEqual(float(episode["native_return"]),
                                       float(episode["emitted_return"]), places=9)
            with (out / "evaluation.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            by_step = {step: [float(r["episode_return"]) for r in rows if int(r["step"]) == step]
                       for step in (0, 1000, 2000)}
            self.assertEqual(by_step[0], by_step[1000])

    def test_reacher_carries_fractional_update_credit_and_flushes_partial_blocks(self):
        with tempfile.TemporaryDirectory(prefix="erl-reacher-smoke-") as temporary:
            out = Path(temporary) / "run"
            self.run_subprocess("normal", out, steps=750, task="Reacher-v5", warmup=250)
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["total_training_transitions"], 750)
            self.assertEqual(metadata["total_evaluation_transitions"], 400)
            self.assertEqual(metadata["critic_updates"], 125)
            self.assertEqual(metadata["actor_updates"], 62)
            with (out / "generations.csv").open() as handle:
                generations = list(csv.DictReader(handle))
            self.assertEqual([int(row["critic_updates"]) for row in generations], [0, 62, 125])
            self.assertEqual([float(row["update_credit"]) for row in generations], [0.0, 0.5, 0.0])
            state = torch.load(out / "checkpoints/latest.pt", weights_only=False)
            rewards = state["replay"]["data"]["rewards"][:750].reshape(-1)
            # Reacher's 50-step horizon ends with a ten-step partial block.
            emitted = np.isin(np.arange(750) % 50, [19, 39, 49])
            self.assertTrue(np.all(rewards[~emitted] == 0.0))
            self.assertTrue(np.all(rewards[emitted] < 0.0))
            with (out / "episodes.csv").open() as handle:
                for episode in csv.DictReader(handle):
                    self.assertAlmostEqual(float(episode["native_return"]),
                                           float(episode["emitted_return"]), places=9)

    def test_evaluation_does_not_change_training_state_or_training_randomness(self):
        with tempfile.TemporaryDirectory(prefix="erl-isolation-") as temporary:
            self.run_subprocess("isolation", temporary)
            states = torch.load(Path(temporary) / "isolation.pt", weights_only=False)
            self.assert_nested_equal(states["before"], states["after"])

    def test_checkpoint_after_learning_resumes_exactly(self):
        with tempfile.TemporaryDirectory(prefix="erl-resume-") as temporary:
            baseline = Path(temporary) / "baseline"
            resumed = Path(temporary) / "resumed"
            self.run_subprocess("normal", baseline, 3000)
            self.run_subprocess("interrupted", resumed, 3000)
            interrupted = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
            self.assertEqual(interrupted["steps"], 2000)
            self.assertEqual(interrupted["agent"]["updates"], 250)
            # Emulate interruption after logs advanced but before the next
            # durable checkpoint. The replayed generation must replace these
            # records rather than leave duplicate lineage or misleading prose.
            marker = "discarded_future_test_record"
            lineage = json.loads((resumed / "lineage.jsonl").read_text().splitlines()[-1])
            lineage.update(step=3000, individual_id=marker)
            with (resumed / "lineage.jsonl").open("a") as handle:
                handle.write(json.dumps(lineage) + "\n")
            event = json.loads((resumed / "events.jsonl").read_text().splitlines()[-1])
            event.update(env_steps=3000, event=marker, observe=marker)
            with (resumed / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            with (resumed / "journal.md").open("a") as handle:
                handle.write(f"\n### {marker}\n\n{marker}\n")
            self.run_subprocess("resume", resumed, 3000)
            first = torch.load(baseline / "checkpoints/latest.pt", weights_only=False)
            second = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
            for key in ("agent", "replay", "population", "lineages", "rngs", "target_generator",
                        "steps", "eval_steps", "generation", "episode", "update_credit", "last_diagnostics"):
                self.assert_nested_equal(first[key], second[key], key)
            self.assertEqual((baseline / "evaluation.csv").read_text(),
                             (resumed / "evaluation.csv").read_text())
            self.assertEqual((baseline / "episodes.csv").read_text(),
                             (resumed / "episodes.csv").read_text())
            self.assertEqual((baseline / "lineage.jsonl").read_text(),
                             (resumed / "lineage.jsonl").read_text())
            self.assertNotIn(marker, (resumed / "events.jsonl").read_text())
            self.assertNotIn(marker, (resumed / "journal.md").read_text())


if __name__ == "__main__":
    unittest.main()
