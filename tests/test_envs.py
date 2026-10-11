"""The reward intervention preserves episode accounting, not discounting."""

import unittest

import gymnasium as gym
import numpy as np

from erl.envs import RewardAggregation, make_env, stable_seed


class SequenceEnv(gym.Env):
    observation_space = gym.spaces.Box(-100.0, 100.0, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def __init__(self, rewards, ending="terminated"):
        self.rewards = rewards
        self.ending = ending
        self.position = 0
        self.last_info = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return np.array([0.0], dtype=np.float32), {"reset_marker": True}

    def step(self, action):
        reward = self.rewards[self.position]
        self.position += 1
        done = self.position == len(self.rewards)
        self.last_info = {"base_marker": True}
        return (
            np.array([self.position], dtype=np.float32), reward,
            done and self.ending == "terminated",
            done and self.ending == "truncated", self.last_info,
        )


def rollout(wrapper):
    wrapper.reset(seed=7)
    rewards, infos = [], []
    for _ in wrapper.unwrapped.rewards:
        observation, reward, terminated, truncated, info = wrapper.step(np.array([0.0]))
        rewards.append(reward)
        infos.append(info)
    return rewards, infos, observation, terminated, truncated


class RewardAggregationTests(unittest.TestCase):
    def test_delay_one_is_exact_identity(self):
        native = [-0.2, 0.3, -8.75, 1.0]
        rewards, infos, observation, terminated, truncated = rollout(
            RewardAggregation(SequenceEnv(native), 1)
        )
        self.assertEqual(rewards, native)
        self.assertEqual([info["native_reward"] for info in infos], native)
        np.testing.assert_array_equal(observation, [4.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_full_blocks_and_partial_termination_flush(self):
        env = RewardAggregation(SequenceEnv([1.0, 2.0, 3.0, 4.0, 5.0]), 2)
        rewards, infos, _, terminated, truncated = rollout(env)
        self.assertEqual(rewards, [0.0, 3.0, 0.0, 7.0, 5.0])
        self.assertEqual(sum(rewards), 15.0)
        self.assertEqual(infos[-1]["native_episode_return"], 15.0)
        self.assertEqual([info["reward_block_phase"] for info in infos], [1, 2, 1, 2, 1])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.pending_reward, 0.0)
        self.assertEqual(env.block_steps, 0)
        self.assertNotIn("native_reward", env.unwrapped.last_info)

    def test_truncation_flush_preserves_bootstrap_flag_and_final_observation(self):
        rewards, infos, observation, terminated, truncated = rollout(
            RewardAggregation(SequenceEnv([-1.0, -2.0, -3.0], "truncated"), 20)
        )
        self.assertEqual(rewards, [0.0, 0.0, -6.0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        np.testing.assert_array_equal(observation, [3.0])
        self.assertTrue(infos[-1]["reward_block_emitted"])

    def test_reset_discards_pending_reward_from_abandoned_episode(self):
        env = RewardAggregation(SequenceEnv([1.0, 2.0, 3.0]), 2)
        env.reset()
        env.step(np.array([0.0]))
        observation, info = env.reset(seed=18)
        self.assertEqual(env.pending_reward, 0.0)
        self.assertEqual(env.native_episode_return, 0.0)
        self.assertEqual(env.block_steps, 0)
        self.assertEqual(info, {"reset_marker": True})
        np.testing.assert_array_equal(observation, [0.0])
        self.assertEqual(env.step(np.array([0.0]))[1], 0.0)
        self.assertEqual(env.step(np.array([0.0]))[1], 3.0)

    def test_reward_delay_does_not_change_pendulum_trajectory_or_episode_sum(self):
        immediate, delayed = make_env("Pendulum-v1", 1), make_env("Pendulum-v1", 20)
        try:
            observation_a, _ = immediate.reset(seed=22)
            observation_b, _ = delayed.reset(seed=22)
            np.testing.assert_array_equal(observation_a, observation_b)
            total_a = total_b = 0.0
            for _ in range(200):
                result_a = immediate.step(np.array([0.2], dtype=np.float32))
                result_b = delayed.step(np.array([0.2], dtype=np.float32))
                np.testing.assert_array_equal(result_a[0], result_b[0])
                self.assertEqual(result_a[2:4], result_b[2:4])
                total_a += result_a[1]
                total_b += result_b[1]
            self.assertAlmostEqual(total_a, total_b, places=10)
            self.assertEqual(result_a[2:4], (False, True))
        finally:
            immediate.close()
            delayed.close()

    def test_invalid_configuration_is_rejected(self):
        for delay in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                RewardAggregation(SequenceEnv([1.0]), delay)
        with self.assertRaises(ValueError):
            make_env("CartPole-v1")


class SeedTests(unittest.TestCase):
    def test_namespaces_are_stable_and_distinct(self):
        seed = stable_seed("Pendulum-v1", 0, "evaluation", 3)
        self.assertEqual(seed, stable_seed("Pendulum-v1", 0, "evaluation", 3))
        variants = [
            seed,
            stable_seed("Pendulum-v1", 1, "evaluation", 3),
            stable_seed("Reacher-v5", 0, "evaluation", 3),
            stable_seed("Pendulum-v1", 0, "training", 3),
            stable_seed("Pendulum-v1", 0, "evaluation", 4),
        ]
        self.assertEqual(len(set(variants)), len(variants))
        for value in variants:
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 2 ** 32)

    def test_argument_boundaries_cannot_alias(self):
        self.assertNotEqual(stable_seed("a", 1, "b", "cd", "e"),
                            stable_seed("a", 1, "b", "c", "de"))
        self.assertEqual(stable_seed("a", np.int64(1), "b"), stable_seed("a", 1, "b"))


if __name__ == "__main__":
    unittest.main()
