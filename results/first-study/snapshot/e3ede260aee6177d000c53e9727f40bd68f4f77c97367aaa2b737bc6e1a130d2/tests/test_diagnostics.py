"""Diagnostic scientific meaning and isolation from all training state."""

import copy
import unittest

import numpy as np
import torch

from erl.core import ReplayBuffer, TD3
from erl.diagnostics import AGE_BIN_COLUMNS, collect_diagnostics, critic_rows, replay_rows


def add_transition(replay, step, source, reward=-1., phase=1):
    replay.add(np.array([1., 0.], np.float32), np.array([0.2], np.float32), reward,
               np.array([0., 1.], np.float32), False, step % 50 == 0,
               source, step, step // 50, step // 250, False, reward_phase=phase)


def tensor_generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_full_diagnostics_preserve_parameters_optimizers_gradients_replay_and_training_rngs(self):
        agent, replay = TD3(2, 1, 7), ReplayBuffer(200, 2, 1)
        for step in range(1, 201):
            add_transition(replay, step, int(step % 5 != 0), phase=(step - 1) % 20 + 1)
        training_rng, target_rng = np.random.default_rng(3), tensor_generator(4)
        # Populate Adam moments and existing parameter .grad tensors.
        for _ in range(2):
            agent.update(replay.sample(128, "balanced", training_rng), target_rng)
        parameters = list(agent.actor.parameters()) + list(agent.critic1.parameters()) + list(agent.critic2.parameters())
        def snapshot():
            return copy.deepcopy(dict(agent=agent.state_dict(), replay=replay.state_dict(),
                                      gradients=[p.grad for p in parameters],
                                      requires_grad=[p.requires_grad for p in parameters],
                                      training_rng=training_rng.bit_generator.state,
                                      target_rng=target_rng.get_state(),
                                      global_torch=torch.get_rng_state(), global_numpy=np.random.get_state()))
        before = snapshot()
        with torch.no_grad():
            replay_result, critic_result = collect_diagnostics(
                agent, replay, 200, 20, "balanced", np.random.default_rng(8), tensor_generator(9))
        self.assert_nested_equal(before, snapshot())
        samples = [row for row in replay_result if row["population"] == "diagnostic_sample"]
        self.assertEqual([row["n"] for row in samples], [64, 64])
        self.assertEqual(len(critic_result), 43)
        for source in ("ea", "learner"):
            group = [row for row in critic_result if row["source"] == source]
            self.assertEqual(group[0]["n"], 128)
            self.assertEqual(sum(row["n"] for row in group[1:]), 128)
        cosine = critic_result[-1]
        self.assertEqual(cosine["gradient_status"], "ok")
        self.assertTrue(-1 <= cosine["gradient_cosine"] <= 1)

    def test_joint_critic_gradient_cosine_distinguishes_agreement_and_conflict(self):
        for opposite, expected in ((False, 1.), (True, -1.)):
            agent, replay = TD3(2, 1, 7), ReplayBuffer(4, 2, 1)
            with torch.no_grad():
                for critic in (agent.critic1, agent.critic2, agent.critic1_target, agent.critic2_target):
                    for parameter in critic.parameters():
                        parameter.zero_()
            add_transition(replay, 1, "ea", reward=1.)
            add_transition(replay, 2, "learner", reward=-1. if opposite else 1.)
            rows = critic_rows(agent, replay, 2, 1, np.random.default_rng(8), tensor_generator(9))
            self.assertAlmostEqual(rows[-1]["gradient_cosine"], expected, places=12)
            self.assertEqual(rows[-1]["gradient_status"], "ok")
            self.assertAlmostEqual(rows[0]["td_abs_q50"], 1.)

    def test_fixed_age_bins_include_boundary_values_and_reuse_is_resident_accounting(self):
        replay = ReplayBuffer(12, 2, 1)
        ages = [0, 999, 1000, 4999, 5000, 10000, 25000, 50000, 100000, 199999, 200000, 250000]
        for index, age in enumerate(ages):
            add_transition(replay, 250000 - age, index % 2)
        rows = replay_rows(replay, 250000, "pooled", np.random.default_rng(2))
        residents = [row for row in rows if row["population"] == "resident"]
        counts = [sum(row[column] for row in residents) for column in AGE_BIN_COLUMNS]
        self.assertEqual(counts, [2, 2, 1, 1, 1, 1, 2, 2])
        self.assertEqual(sum(row["n"] for row in residents), 12)
        self.assertTrue(all(row["reuse_q100"] == 0 for row in residents))
        self.assertEqual(replay.total_sample_counts, {"ea": 0, "learner": 0})

    def test_checkpointed_diagnostic_streams_reproduce_next_records_exactly(self):
        agent, replay = TD3(2, 1, 7), ReplayBuffer(200, 2, 1)
        for step in range(1, 201):
            add_transition(replay, step, int(step % 5 != 0), phase=(step - 1) % 20 + 1)
        rng, target = np.random.default_rng(8), tensor_generator(9)
        collect_diagnostics(agent, replay, 200, 20, "pooled", rng, target)
        saved_rng, saved_target = copy.deepcopy(rng.bit_generator.state), target.get_state().clone()
        expected = collect_diagnostics(agent, replay, 200, 20, "pooled", rng, target)
        restored_rng, restored_target = np.random.default_rng(), tensor_generator(100)
        restored_rng.bit_generator.state = saved_rng
        restored_target.set_state(saved_target)
        observed = collect_diagnostics(agent, replay, 200, 20, "pooled", restored_rng, restored_target)
        self.assert_nested_equal(expected, observed)

    def test_empty_and_single_source_are_explicit_not_fabricated_zero_cosines(self):
        agent, replay = TD3(2, 1, 7), ReplayBuffer(2, 2, 1)
        for populated in (False, True):
            if populated:
                add_transition(replay, 1, "learner")
            rows, critics = collect_diagnostics(agent, replay, 1, 1, "learner", np.random.default_rng(2), tensor_generator(3))
            self.assertIsNone(critics[-1]["gradient_cosine"])
            self.assertEqual(critics[-1]["gradient_status"], "missing_source")
            ea = [row for row in rows if row["source"] == "ea"]
            self.assertTrue(all(row["n"] == 0 and row["age_q50"] is None for row in ea))


if __name__ == "__main__":
    unittest.main()
