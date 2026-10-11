"""Collect independent random-policy reference returns and fixed behavioral probes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .envs import make_env, stable_seed


def calibrate(destination, episodes=20):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Calibration already exists; reuse it instead of silently replacing anchors")
    result = {}
    for task in ("Pendulum-v1", "Reacher-v5"):
        env = make_env(task, 1)
        rng = np.random.default_rng(stable_seed(task, 900000, "calibration_actions"))
        observations, returns, lengths = [], [], []
        for episode in range(episodes):
            obs, _ = env.reset(seed=stable_seed(task, 900000, "calibration_reset", episode))
            total, length = 0.0, 0
            while True:
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(length)
        env.close()
        probe_rng = np.random.default_rng(stable_seed(task, 900000, "calibration_probe"))
        indices = probe_rng.choice(len(observations), size=128, replace=False)
        probe_path = destination.parent / f"{task}-probe.npy"
        np.save(probe_path, np.asarray(observations)[indices])
        mean = float(np.mean(returns))
        if mean >= 0:
            raise ValueError("Invalid nonnegative random-policy anchor")
        result[task] = dict(random_return_mean=mean, reward_upper_bound=0.0,
                            returns=returns, episode_lengths=lengths, interactions=sum(lengths),
                            seed=900000, episodes=episodes, probe_path=probe_path.name,
                            probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest())
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {"random_return_mean": v["random_return_mean"],
                         "interactions": v["interactions"]} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/pilot/calibration.json")
    args = parser.parse_args()
    calibrate(args.out)
