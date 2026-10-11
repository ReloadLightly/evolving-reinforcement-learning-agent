"""Tests for causal-comparison invariants, not learned performance."""

import copy
import unittest

import numpy as np
import torch

from erl.core import ReplayBuffer, TD3


def transition(buffer, step, source):
    buffer.add(np.array([step, -step], np.float32), np.array([0.1], np.float32),
               float(-step), np.array([step + 1, -step - 1], np.float32),
               False, step % 10 == 0, source, step, step // 10, step // 50, step <= 5)


def batch():
    rng = np.random.default_rng(13)
    return dict(obs=rng.normal(size=(8, 2)).astype(np.float32),
                actions=rng.uniform(-1, 1, size=(8, 1)).astype(np.float32),
                rewards=rng.normal(size=(8, 1)).astype(np.float32),
                next_obs=rng.normal(size=(8, 2)).astype(np.float32),
                terminated=np.array([True, False] * 4),
                truncated=np.array([False, True] * 4),
                sources=np.array([0, 1] * 4))


def generator(seed=23):
    return torch.Generator(device="cpu").manual_seed(seed)


class ReplayTests(unittest.TestCase):
    def test_global_fifo_source_quota_and_pool_consistency(self):
        replay = ReplayBuffer(11, 2, 1)
        for step in range(1, 101):
            transition(replay, step, "learner" if step % 5 == 0 else "ea")
            replay.validate()
            expected = list(range(max(1, step - 10), step + 1))
            self.assertEqual(sorted(replay.metadata()["steps"].tolist()), expected)
        self.assertEqual(replay.source_counts, {"learner": 3, "ea": 8})
        sample = replay.sample(128, "balanced", np.random.default_rng(3))
        self.assertEqual(int(sample["sources"].sum()), 64)
        self.assertEqual(replay.total_sample_counts, {"learner": 64, "ea": 64})
        self.assertEqual(int(replay.reuse_counts.sum()), 128)
        self.assertTrue(np.all(sample["ages"] == 100 - sample["steps"]))
        learner = replay.sample(128, "learner", np.random.default_rng(4))
        self.assertTrue(np.all(learner["sources"] == 0))
        self.assertEqual(replay.source_counts["ea"], 8, "Blocked sampling must retain EA occupancy")

    def test_pooled_is_uniform_and_diagnostics_do_not_record_reuse(self):
        replay = ReplayBuffer(5, 2, 1)
        for step in range(1, 6):
            transition(replay, step, "learner" if step == 5 else "ea")
        before = replay.state_dict()
        sample = replay.sample(50_000, "pooled", np.random.default_rng(5), record=False)
        frequencies = np.bincount(sample["indices"], minlength=5) / 50_000
        np.testing.assert_allclose(frequencies, np.full(5, 0.2), atol=0.006)
        np.testing.assert_array_equal(replay.reuse_counts, before["reuse_counts"])
        self.assertEqual(replay.total_sample_counts, {"learner": 0, "ea": 0})

    def test_sampling_fails_when_intervention_cannot_be_met(self):
        replay = ReplayBuffer(5, 2, 1)
        with self.assertRaises(ValueError):
            replay.sample(128, "pooled", np.random.default_rng(1))
        transition(replay, 1, "ea")
        with self.assertRaises(ValueError):
            replay.sample(128, "balanced", np.random.default_rng(1))
        with self.assertRaises(ValueError):
            replay.sample(128, "learner", np.random.default_rng(1))
        transition(replay, 2, "learner")
        with self.assertRaises(ValueError):
            replay.sample(127, "balanced", np.random.default_rng(1))

    def test_replay_checkpoint_continues_eviction_sampling_and_reuse(self):
        original = ReplayBuffer(11, 2, 1)
        for step in range(1, 16):
            transition(original, step, step % 2)
        rng = np.random.default_rng(93)
        original.sample(8, "balanced", rng)
        restored = ReplayBuffer(11, 2, 1)
        restored.load_state_dict(original.state_dict())
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        for step in range(16, 21):
            for replay in (original, restored):
                transition(replay, step, step % 2)
        left, right = original.sample(20, "balanced", rng), restored.sample(20, "balanced", restored_rng)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        original.validate()
        restored.validate()


class TD3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_models_equal(self, left, right):
        for name in ("actor", "actor_target", "critic1", "critic2", "critic1_target", "critic2_target"):
            for key, value in getattr(left, name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(right, name).state_dict()[key]), f"{name}/{key}")

    def test_initialization_and_target_noise_preserve_global_rng(self):
        global_state = torch.get_rng_state().clone()
        left, right = TD3(2, 1, 10), TD3(2, 1, 10)
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        self.assert_models_equal(left, right)
        self.assertFalse(torch.equal(left.critic1.net[0].weight, left.critic2.net[0].weight))
        left.update(batch(), generator())
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))

    def test_target_uses_smaller_critic_and_bootstraps_time_limit(self):
        learner = TD3(2, 1, 10)
        with torch.no_grad():
            for critic, value in ((learner.critic1_target, 3.0), (learner.critic2_target, 2.0)):
                for parameter in critic.parameters():
                    parameter.zero_()
                critic.net[-1].bias.fill_(value)
        data = batch()
        data["rewards"] = np.ones((8, 1), np.float32)
        targets = learner.compute_targets(data, generator()).numpy().reshape(-1)
        np.testing.assert_allclose(targets[::2], 1.0)
        np.testing.assert_allclose(targets[1::2], 1 + 0.99 * 2)

    def test_actor_delay_and_polyak_targets(self):
        learner = TD3(2, 1, 10)
        before = learner.state_dict()
        rng = generator()
        first = learner.update(batch(), rng)
        self.assertIsNone(first["actor_loss"])
        self.assertEqual(learner.actor_updates, 0)
        for key, value in before["actor"].items():
            self.assertTrue(torch.equal(value, learner.actor.state_dict()[key]))
        for key, value in before["critic1_target"].items():
            self.assertTrue(torch.equal(value, learner.critic1_target.state_dict()[key]))
        second = learner.update(batch(), rng)
        self.assertIsNotNone(second["actor_loss"])
        self.assertEqual(learner.actor_updates, 1)
        for key, value in learner.critic1_target.state_dict().items():
            expected = before["critic1_target"][key] * 0.995 + learner.critic1.state_dict()[key] * 0.005
            torch.testing.assert_close(value, expected)
        self.assertEqual(second["source_fraction_ea"], 0.5)

    def test_checkpoint_restores_optimizers_targets_and_delay_parity(self):
        original = TD3(2, 1, 10)
        rng = generator()
        original.update(batch(), rng)
        saved = original.state_dict()
        restored = TD3(2, 1, 99)
        restored.load_state_dict(saved)
        restored_rng = generator(8)
        restored_rng.set_state(rng.get_state().clone())
        for _ in range(3):
            left, right = original.update(batch(), rng), restored.update(batch(), restored_rng)
            self.assertEqual(left, right)
            self.assert_models_equal(original, restored)
        self.assertEqual(saved["updates"], 1, "Checkpoint objects must not change as training continues")


if __name__ == "__main__":
    unittest.main()
