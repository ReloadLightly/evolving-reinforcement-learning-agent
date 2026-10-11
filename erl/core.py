"""Small CPU TD3 learner and one globally evicted, provenance-aware replay store.

Actions are normalized to [-1, 1] throughout this module. Environment action
scaling belongs at the simulator boundary. Collector source 0 means learner and
1 means evolutionary population; warmup is separately tagged behavior metadata.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, actions), dim=-1))


class TD3:
    """Twin Delayed DDPG with independently initialized critics.

    Network construction restores the caller's global CPU torch RNG. Training
    consumes only the explicitly supplied target-noise generator. Critic targets
    bootstrap through time limits and stop only at genuine terminations.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, seed: int, hidden: int = 64,
        lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
        policy_delay: int = 2, target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        if policy_delay < 1 or not 0 <= gamma <= 1 or not 0 <= tau <= 1:
            raise ValueError("Invalid TD3 discount, target rate, or policy delay")
        self.config = dict(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden,
                           lr=lr, gamma=gamma, tau=tau, policy_delay=policy_delay,
                           target_noise=target_noise, noise_clip=noise_clip)
        self.gamma, self.tau = gamma, tau
        self.policy_delay, self.target_noise, self.noise_clip = policy_delay, target_noise, noise_clip
        self.initialization_seeds = [
            int(s.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))
            for s in np.random.SeedSequence(seed).spawn(3)
        ]
        modules = []
        for module_type, module_seed in zip((Actor, Critic, Critic), self.initialization_seeds):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(module_seed)
                modules.append(module_type(obs_dim, act_dim, hidden))
        self.actor, self.critic1, self.critic2 = modules
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic1_target = copy.deepcopy(self.critic1).requires_grad_(False)
        self.critic2_target = copy.deepcopy(self.critic2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr,
        )
        self.updates = 0
        self.actor_updates = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self.actor(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy().copy()

    @torch.no_grad()
    def compute_targets(self, batch: dict[str, np.ndarray], target_generator: torch.Generator) -> torch.Tensor:
        """y = r + gamma * (1 - terminated) * min(Q1', Q2')."""
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32).reshape(-1, 1)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32).reshape(-1, 1)
        actions = self.actor_target(next_obs)
        noise = torch.randn(actions.shape, generator=target_generator, dtype=actions.dtype)
        noise = (noise * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
        actions = (actions + noise).clamp(-1, 1)
        next_q = torch.minimum(self.critic1_target(next_obs, actions), self.critic2_target(next_obs, actions))
        return rewards + self.gamma * (1 - terminated) * next_q

    def update(self, batch: dict[str, np.ndarray], target_generator: torch.Generator) -> dict[str, Any]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        targets = self.compute_targets(batch, target_generator)
        q1, q2 = self.critic1(obs, actions), self.critic2(obs, actions)
        loss = ((q1 - targets).square() + (q2 - targets).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TD3 critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.updates += 1

        actor_loss_value = None
        if self.updates % self.policy_delay == 0:
            # The actor needs dQ/da, but the critic's parameters are fixed here.
            self.critic1.requires_grad_(False)
            try:
                actor_loss = -self.critic1(obs, self.actor(obs)).mean()
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Nonfinite TD3 actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
            finally:
                self.critic1.requires_grad_(True)
            actor_loss_value = float(actor_loss.detach())
            self.actor_updates += 1
            with torch.no_grad():
                for live, target in (
                    (self.actor, self.actor_target), (self.critic1, self.critic1_target),
                    (self.critic2, self.critic2_target),
                ):
                    for parameter, target_parameter in zip(live.parameters(), target.parameters()):
                        target_parameter.lerp_(parameter, self.tau)

        residual = ((q1.detach() - targets).abs() + (q2.detach() - targets).abs()) * 0.5
        disagreement = (q1.detach() - q2.detach()).abs()
        diagnostics: dict[str, Any] = dict(
            critic_loss=float(loss.detach()), actor_loss=actor_loss_value,
            td_abs_mean=float(residual.mean()), q_disagreement=float(disagreement.mean()),
            target_mean=float(targets.mean()), updates=self.updates,
            actor_updates=self.actor_updates,
        )
        if "sources" in batch:
            sources = np.asarray(batch["sources"]).reshape(-1)
            diagnostics["source_fraction_ea"] = float(np.mean(sources == 1))
            for name, source in (("learner", 0), ("ea", 1)):
                mask = torch.as_tensor(sources == source)
                diagnostics[f"td_abs_{name}"] = float(residual[mask].mean()) if mask.any() else None
                diagnostics[f"q_disagreement_{name}"] = float(disagreement[mask].mean()) if mask.any() else None
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        # Independent copies also make in-memory checkpoints safe to retain.
        state: dict[str, Any] = dict(config=self.config, initialization_seeds=self.initialization_seeds,
                                     updates=self.updates, actor_updates=self.actor_updates)
        for name in ("actor", "critic1", "critic2", "actor_target", "critic1_target", "critic2_target",
                     "actor_optimizer", "critic_optimizer"):
            state[name] = getattr(self, name).state_dict()
        return copy.deepcopy(state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != self.config:
            raise ValueError("Checkpoint TD3 configuration differs from this learner")
        for name in ("actor", "critic1", "critic2", "actor_target", "critic1_target", "critic2_target",
                     "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(copy.deepcopy(state[name]))
        self.initialization_seeds = list(state["initialization_seeds"])
        self.updates, self.actor_updates = int(state["updates"]), int(state["actor_updates"])


class ReplayBuffer:
    """Global FIFO with O(1) insertion/eviction and O(batch size) source sampling.

    Source pools are dense arrays of resident slot IDs. Eviction removes an ID
    by swapping the pool's final ID into its position. Both source pools point
    into the *same* FIFO; neither has an independent retention policy.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.obs_dim, self.act_dim = capacity, obs_dim, act_dim
        self.data = {
            "obs": np.zeros((capacity, obs_dim), dtype=np.float32),
            "actions": np.zeros((capacity, act_dim), dtype=np.float32),
            "rewards": np.zeros((capacity, 1), dtype=np.float32),
            "next_obs": np.zeros((capacity, obs_dim), dtype=np.float32),
            "terminated": np.zeros((capacity, 1), dtype=np.bool_),
            "truncated": np.zeros((capacity, 1), dtype=np.bool_),
            "sources": np.zeros(capacity, dtype=np.int8),
            "steps": np.zeros(capacity, dtype=np.int64),
            "episodes": np.zeros(capacity, dtype=np.int64),
            "generations": np.zeros(capacity, dtype=np.int64),
            "warmup": np.zeros(capacity, dtype=np.bool_),
            # Phase is diagnostic metadata only; zero denotes an unspecified
            # phase for callers predating the delayed-reward diagnostics.
            "reward_phases": np.zeros(capacity, dtype=np.int32),
        }
        self.reuse_counts = np.zeros(capacity, dtype=np.int64)
        self._pools = np.zeros((2, capacity), dtype=np.int64)
        self._positions = np.zeros(capacity, dtype=np.int64)
        self._counts = np.zeros(2, dtype=np.int64)
        self._sample_counts = np.zeros(2, dtype=np.int64)
        self.size, self.cursor, self.total_inserted = 0, 0, 0
        self.latest_step = 0

    def __len__(self) -> int:
        return self.size

    @property
    def source_counts(self) -> dict[str, int]:
        return {"learner": int(self._counts[0]), "ea": int(self._counts[1])}

    @property
    def total_sample_counts(self) -> dict[str, int]:
        return {"learner": int(self._sample_counts[0]), "ea": int(self._sample_counts[1])}

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float,
        next_obs: np.ndarray, terminated: bool, truncated: bool,
        source: str | int, step: int, episode: int, generation: int, warmup: bool,
        *, reward_phase: int = 0,
    ) -> None:
        source = {"learner": 0, "rl": 0, "ea": 1}.get(source, source)
        if source not in (0, 1):
            raise ValueError("Source must be learner/0 or ea/1")
        source = int(source)
        index = self.cursor
        if self.size == self.capacity:
            old_source = self.data["sources"][index]
            position = self._positions[index]
            last_index = self._pools[old_source, self._counts[old_source] - 1]
            self._pools[old_source, position] = last_index
            self._positions[last_index] = position
            self._counts[old_source] -= 1
        else:
            self.size += 1
        self._positions[index] = self._counts[source]
        self._pools[source, self._counts[source]] = index
        self._counts[source] += 1
        for key, value in (("obs", obs), ("actions", action), ("rewards", reward),
                           ("next_obs", next_obs), ("terminated", terminated), ("truncated", truncated),
                           ("sources", source), ("steps", step), ("episodes", episode),
                           ("generations", generation), ("warmup", warmup),
                           ("reward_phases", reward_phase)):
            self.data[key][index] = value
        self.reuse_counts[index] = 0
        self.cursor = (index + 1) % self.capacity
        self.total_inserted += 1
        self.latest_step = int(step)

    def sample(
        self, batch_size: int, mode: str, rng: np.random.Generator,
        current_step: int | None = None, *, record: bool = True,
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or self.size == 0:
            raise ValueError("A positive batch size and nonempty replay are required")
        if mode == "pooled":
            indices = rng.integers(self.size, size=batch_size)
        elif mode in ("balanced", "learner", "ea"):
            required = (0, 1) if mode == "balanced" else (1,) if mode == "ea" else (0,)
            if any(self._counts[source] == 0 for source in required):
                raise ValueError("Required replay source is empty; no fallback is allowed")
            if mode == "balanced" and batch_size % 2:
                raise ValueError("Balanced batches require an even batch size")
            learner_count = batch_size // 2 if mode == "balanced" else batch_size
            first_source = 1 if mode == "ea" else 0
            indices = self._pools[first_source, rng.integers(self._counts[first_source], size=learner_count)]
            if mode == "balanced":
                ea_indices = self._pools[1, rng.integers(self._counts[1], size=batch_size - learner_count)]
                indices = np.concatenate((indices, ea_indices))
                rng.shuffle(indices)
        else:
            raise ValueError(f"Unknown replay sampling mode: {mode}")
        if record:
            np.add.at(self.reuse_counts, indices, 1)
            self._sample_counts += np.bincount(self.data["sources"][indices], minlength=2)
        batch = {key: value[indices].copy() for key, value in self.data.items()}
        batch["indices"] = indices.copy()
        batch["ages"] = (self.latest_step if current_step is None else current_step) - batch["steps"]
        batch["reuse_counts"] = self.reuse_counts[indices].copy()
        return batch

    def metadata(self, current_step: int | None = None) -> dict[str, np.ndarray]:
        """Copy resident provenance for diagnostics without sampling or mutation."""
        result = {key: self.data[key][:self.size].copy() for key in
                  ("sources", "steps", "episodes", "generations", "warmup", "reward_phases")}
        result["ages"] = (self.latest_step if current_step is None else current_step) - result["steps"]
        result["reuse_counts"] = self.reuse_counts[:self.size].copy()
        return result

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(
            capacity=self.capacity, obs_dim=self.obs_dim, act_dim=self.act_dim,
            data=self.data, reuse_counts=self.reuse_counts, pools=self._pools,
            positions=self._positions, counts=self._counts, sample_counts=self._sample_counts,
            size=self.size, cursor=self.cursor, total_inserted=self.total_inserted,
            latest_step=self.latest_step,
        ))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state["capacity"], state["obs_dim"], state["act_dim"]) != (self.capacity, self.obs_dim, self.act_dim):
            raise ValueError("Checkpoint replay dimensions differ from this buffer")
        state = copy.deepcopy(state)
        if set(state["data"]) != set(self.data):
            raise ValueError("Checkpoint replay fields differ")
        for key, array in self.data.items():
            if state["data"][key].shape != array.shape or state["data"][key].dtype != array.dtype:
                raise ValueError(f"Invalid checkpoint replay field: {key}")
        self.data = state["data"]
        for key in ("reuse_counts", "size", "cursor", "total_inserted", "latest_step"):
            setattr(self, key, state[key])
        for key in ("pools", "positions", "counts", "sample_counts"):
            setattr(self, "_" + key, state[key])
        self.validate()

    def validate(self) -> None:
        """Audit FIFO/source index consistency; intended for tests/checkpoints."""
        if not 0 <= self.size <= self.capacity or self.cursor != self.total_inserted % self.capacity:
            raise ValueError("Invalid FIFO counters")
        if self.size != min(self.total_inserted, self.capacity) or self._counts.sum() != self.size:
            raise ValueError("Invalid source counts")
        all_indices = []
        for source in (0, 1):
            indices = self._pools[source, :self._counts[source]]
            if np.any(indices < 0) or np.any(indices >= self.size):
                raise ValueError("Nonresident replay source index")
            if not np.all(self.data["sources"][indices] == source):
                raise ValueError("Incorrect replay source index")
            if not np.array_equal(self._positions[indices], np.arange(len(indices))):
                raise ValueError("Incorrect replay reverse index")
            all_indices.extend(indices.tolist())
        if len(set(all_indices)) != self.size:
            raise ValueError("Duplicate or missing resident replay index")
