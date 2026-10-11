"""Bounded, independent measurements of replay and critic behavior.

These are descriptive diagnostics, not value-error ground truth or new learner
inputs. Native rewards are never read. A 128-draw replay sample estimates the
arm's current sampling-age distribution; it is not a historical training ledger.
Critic diagnostics use a separate 128-draw sample per available source. All
sampling is with replacement, using caller-owned diagnostic RNGs. No replay
reuse counter, optimizer state, parameter, or parameter .grad is modified.
"""

from __future__ import annotations

import numpy as np
import torch

from .core import ReplayBuffer, TD3


QUANTILES = (0, 25, 50, 75, 90, 100)
AGE_BIN_EDGES = (0, 1000, 5000, 10000, 25000, 50000, 100000, 200000, np.inf)
AGE_BIN_COLUMNS = tuple(f"age_bin_{int(left)}_{int(right)}" if np.isfinite(right)
                        else f"age_bin_{int(left)}_plus"
                        for left, right in zip(AGE_BIN_EDGES, AGE_BIN_EDGES[1:]))
SAMPLES_PER_SOURCE = 128


def quantiles(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Nonfinite {prefix} diagnostic")
    result = np.quantile(values, np.asarray(QUANTILES) / 100, method="linear") if len(values) else [None] * len(QUANTILES)
    return {f"{prefix}_q{q}": float(value) if value is not None else None
            for q, value in zip(QUANTILES, result)}


def replay_rows(replay: ReplayBuffer, step: int, mode: str, rng: np.random.Generator) -> list[dict]:
    resident = replay.metadata(current_step=step)
    if len(replay):
        sampled = replay.sample(SAMPLES_PER_SOURCE, mode, rng, current_step=step, record=False)
    else:
        sampled = resident
    rows = []
    for population, data in (("resident", resident), ("diagnostic_sample", sampled)):
        for name, source in (("ea", 1), ("learner", 0)):
            mask = data["sources"] == source
            ages, reuse = data["ages"][mask], data["reuse_counts"][mask]
            if np.any(ages < 0):
                raise ValueError("Replay contains a collection step after the diagnostic checkpoint")
            histogram, _ = np.histogram(ages, bins=AGE_BIN_EDGES)
            rows.append(dict(step=step, population=population, source=name, n=int(mask.sum()),
                             resident_count=replay.source_counts[name],
                             sampling_mode="resident" if population == "resident" else mode,
                             **quantiles(ages, "age"), **quantiles(reuse, "reuse"),
                             **{key: int(value) for key, value in zip(AGE_BIN_COLUMNS, histogram)}))
    return rows


def critic_rows(
    agent: TD3, replay: ReplayBuffer, step: int, delay: int,
    rng: np.random.Generator, target_generator: torch.Generator,
) -> list[dict]:
    if delay < 1:
        raise ValueError("Diagnostic reward delay must be positive")
    rows, gradients, source_counts = [], {}, {}
    parameters = tuple(agent.critic1.parameters()) + tuple(agent.critic2.parameters())
    empty = np.array([], dtype=np.float64)
    for name in ("ea", "learner"):
        batch = replay.sample(SAMPLES_PER_SOURCE, name, rng, current_step=step, record=False) if replay.source_counts[name] else None
        if batch is None:
            phases, residuals, disagreement = empty, empty, empty
        else:
            phases = batch["reward_phases"]
            if np.any(phases < 1) or np.any(phases > delay):
                raise ValueError("Full diagnostics require recorded reward phases in [1, delay]")
            # Explicitly enable gradients even when called inside an evaluation
            # no_grad context. autograd.grad returns values without accumulating
            # into .grad, unlike backward(). Target computation remains no_grad.
            with torch.enable_grad():
                obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
                actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
                targets = agent.compute_targets(batch, target_generator)
                q1, q2 = agent.critic1(obs, actions), agent.critic2(obs, actions)
                loss = ((q1 - targets).square() + (q2 - targets).square()).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diagnostic critic loss")
                parts = torch.autograd.grad(loss, parameters, create_graph=False)
                gradients[name] = torch.cat([part.detach().reshape(-1) for part in parts]).to(torch.float64)
                if not torch.isfinite(gradients[name]).all():
                    raise FloatingPointError("Nonfinite diagnostic critic gradient")
            residuals = ((q1.detach() - targets).abs() + (q2.detach() - targets).abs()).numpy().reshape(-1) / 2
            disagreement = (q1.detach() - q2.detach()).abs().numpy().reshape(-1)
        source_counts[name] = len(phases)
        for phase in range(delay + 1):
            mask = np.ones(len(phases), dtype=np.bool_) if phase == 0 else phases == phase
            rows.append(dict(step=step, source=name, phase=phase, n=int(mask.sum()),
                             **quantiles(residuals[mask], "td_abs"),
                             **quantiles(disagreement[mask], "q_disagreement"),
                             gradient_cosine=None, gradient_norm_ea=None,
                             gradient_norm_learner=None, gradient_status=""))
    norms = {name: float(torch.linalg.vector_norm(gradient)) for name, gradient in gradients.items()}
    cosine, status = None, "missing_source"
    if len(gradients) == 2:
        norm_product = norms["ea"] * norms["learner"]
        status = "zero_norm" if norm_product == 0 else "ok"
        if norm_product:
            cosine = float(torch.dot(gradients["ea"], gradients["learner"]) / norm_product)
            cosine = max(-1.0, min(1.0, cosine))
    rows.append(dict(step=step, source="cross_source", phase=0, n=sum(source_counts.values()),
                     **quantiles(empty, "td_abs"), **quantiles(empty, "q_disagreement"),
                     gradient_cosine=cosine, gradient_norm_ea=norms.get("ea"),
                     gradient_norm_learner=norms.get("learner"), gradient_status=status))
    return rows


def collect_diagnostics(
    agent: TD3, replay: ReplayBuffer, step: int, delay: int, mode: str,
    rng: np.random.Generator, target_generator: torch.Generator,
) -> tuple[list[dict], list[dict]]:
    """Return replay/critic CSV rows while leaving all training state untouched."""
    return replay_rows(replay, step, mode, rng), critic_rows(agent, replay, step, delay, rng, target_generator)
