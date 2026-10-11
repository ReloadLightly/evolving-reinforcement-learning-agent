"""Task creation, reward accounting, and reproducible named seed streams.

Native rewards in ``info`` are diagnostics only. The learner must train on the
reward returned by ``step``; otherwise delayed feedback would be bypassed.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any

import gymnasium as gym


TASKS = ("Pendulum-v1", "Reacher-v5")


def stable_seed(task: str, seed: int, namespace: str, *parts: Any) -> int:
    """Resolve a named, process-independent 32-bit seed using SHA-256.

    JSON encoding keeps adjacent arguments unambiguous. Algorithm arm and reward
    delay should be omitted for streams intended to match across paired runs.
    ``parts`` must contain JSON-serializable values; no process-randomized Python
    ``hash()`` participates in the result.
    """
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    payload = json.dumps(
        [task, int(seed), namespace, *parts],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class RewardAggregation(gym.Wrapper):
    """Emit undiscounted native reward sums in blocks of ``delay`` steps.

    Partial blocks flush on either termination or time-limit truncation. The two
    flags are preserved: TD3 may bootstrap through truncation, but not through a
    true terminal state. ``reset`` starts a new block and episode accumulator.

    This preserves completed *undiscounted* episode returns. With a discount
    factor below one it changes the learning objective, and the hidden pending
    reward makes the feedback process history dependent.
    """

    def __init__(self, env: gym.Env, delay: int = 1):
        if not isinstance(delay, Integral) or isinstance(delay, bool) or delay < 1:
            raise ValueError("delay must be a positive integer")
        super().__init__(env)
        self.delay = int(delay)
        self.pending_reward = 0.0
        self.block_steps = 0
        self.native_episode_return = 0.0

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self.pending_reward = 0.0
        self.block_steps = 0
        self.native_episode_return = 0.0
        return observation, info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        native_reward = float(reward)
        self.pending_reward += native_reward
        self.native_episode_return += native_reward
        self.block_steps += 1
        block_phase = self.block_steps
        emit = self.block_steps == self.delay or terminated or truncated
        emitted_reward = self.pending_reward if emit else 0.0
        if emit:
            self.pending_reward = 0.0
            self.block_steps = 0
        diagnostics = dict(info)
        diagnostics["native_reward"] = native_reward
        diagnostics["native_episode_return"] = self.native_episode_return
        diagnostics["reward_block_phase"] = block_phase
        diagnostics["reward_block_emitted"] = bool(emit)
        return observation, emitted_reward, terminated, truncated, diagnostics


def make_env(task: str, delay: int = 1) -> RewardAggregation:
    """Create one declared task with the accounting wrapper outermost."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {TASKS}")
    if not isinstance(delay, Integral) or isinstance(delay, bool) or delay < 1:
        raise ValueError("delay must be a positive integer")
    return RewardAggregation(gym.make(task), delay)
