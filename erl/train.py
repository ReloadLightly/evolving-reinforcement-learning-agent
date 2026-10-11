"""CPU ERL/TD3 runner with isolated evaluation and an explanatory event journal."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
from datetime import datetime, timezone

import numpy as np
import torch

from .core import Actor, ReplayBuffer, TD3
from .envs import make_env, stable_seed
from .evolution import evolve_population


ARMS = {"td3": "learner", "erl_pooled": "pooled",
        "erl_balanced": "balanced", "erl_no_ea_replay": "learner"}
MEANING = {
    "td3": "All five rollout slots use the learner. Every replay sample comes from its own experience.",
    "erl_pooled": "Four evolutionary actors collect four episodes; the learner collects one. Pooled replay follows the resulting roughly 80:20 data mix.",
    "erl_balanced": "The same 80:20 collection mix is sampled 50:50. This changes exposure to data, not how much data is collected.",
    "erl_no_ea_replay": "Population collection, selection and injection still happen, but the learner never samples population transitions. This isolates direct experience transfer.",
}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha_file(p) for p in sorted((root / "erl").glob("*.py"))}


def append_csv(path, rows):
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def env_action(normalized, env):
    low, high = env.action_space.low, env.action_space.high
    return (low + (np.asarray(normalized) + 1.0) * 0.5 * (high - low)).astype(np.float32)


class Experiment:
    def __init__(self, args):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.args = args
        if args.eval_episodes < 2:
            raise ValueError("At least two evaluation episodes are required for the reported episode SD")
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.out / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)
        self.env = make_env(args.task, args.delay)
        self.eval_env = make_env(args.task, 1)
        self.horizon = int(self.env.spec.max_episode_steps)
        self.generation_steps = 5 * self.horizon
        if args.steps % self.generation_steps or args.warmup % self.generation_steps:
            raise ValueError("Training and warmup budgets must end on generation boundaries")
        if args.eval_every % self.generation_steps or args.steps % args.eval_every:
            raise ValueError("Evaluation checkpoints must be complete generation boundaries")
        if args.warmup >= args.steps or args.warmup <= 0:
            raise ValueError("Run must include a positive warmup followed by learning")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.agent = TD3(self.obs_dim, self.act_dim,
                         stable_seed(args.task, args.seed, "learner_init"))
        self.population = []
        for slot in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(stable_seed(args.task, args.seed, "population_init", slot))
                self.population.append(Actor(self.obs_dim, self.act_dim))
        self.lineages = [f"initial_{i}" for i in range(4)]
        self.replay = ReplayBuffer(100_000, self.obs_dim, self.act_dim)
        self.rngs = {name: np.random.default_rng(stable_seed(args.task, args.seed, name))
                     for name in ("warmup", "exploration", "replay", "selection", "mutation")}
        self.target_generator = torch.Generator(device="cpu")
        self.target_generator.manual_seed(stable_seed(args.task, args.seed, "target_noise"))
        self.steps = self.eval_steps = self.generation = self.episode = 0
        self.update_credit = 0.0
        self.elapsed_before = 0.0
        self.started = time.perf_counter()
        self.last_diagnostics = {}
        self.probe = None
        calibration_path = getattr(args, "calibration", None)
        if calibration_path:
            calibration_path = Path(calibration_path)
            calibration = json.loads(calibration_path.read_text())[args.task]
            probe_path = calibration_path.parent / calibration["probe_path"]
            if sha_file(probe_path) != calibration["probe_sha256"]:
                raise ValueError("Behavioral probe hash mismatch")
            self.probe = torch.as_tensor(np.load(probe_path), dtype=torch.float32)
        self.metadata = {
            "status": "running", "purpose": "development pilot; not confirmatory evidence",
            "task": args.task, "arm": args.arm, "seed": args.seed, "reward_delay": args.delay,
            "training_budget": args.steps, "warmup": args.warmup,
            "evaluation_interval": args.eval_every, "evaluation_episodes": args.eval_episodes,
            "horizon": self.horizon, "hidden_sizes": [64, 64], "batch_size": 128,
            "critic_updates_per_transition": 0.25, "replay_capacity": 100_000,
            "injection_interval_post_warmup": 5000, "threads": 1, "device": "cpu",
            "python": platform.python_version(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "mujoco")},
            "source_sha256": source_hashes(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "initialization": "PyTorch Linear defaults; separate stable learner/population seeds",
            "run_id": self.out.name,
            "calibration_sha256": sha_file(calibration_path) if calibration_path else None,
        }
        if args.resume:
            self.restore()
        elif (self.out / "metadata.json").exists():
            raise ValueError("Run directory already exists; use --resume or a new output directory")
        write_json(self.out / "metadata.json", self.metadata)

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def event(self, event, observe, explain, check, next_step, **measurements):
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), event=event,
                      run_id=self.out.name, task=self.args.task, arm=self.args.arm,
                      seed=self.args.seed, delay=self.args.delay, env_steps=self.steps,
                      eval_steps=self.eval_steps, critic_updates=self.agent.updates,
                      actor_updates=self.agent.updates // 2, elapsed_s=self.elapsed(),
                      observe=observe, explain=explain, check=check, next=next_step, **measurements)
        with (self.out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        with (self.out / "journal.md").open("a") as handle:
            handle.write(f"\n### {event} · {self.steps:,} training transitions\n\n")
            for label, value in (("Observe", observe), ("Explain", explain),
                                 ("Check", check), ("Next", next_step)):
                handle.write(f"**{label}:** {value}\n\n")
        print(f"\n[{self.out.name} | {self.steps:,}/{self.args.steps:,}]\n"
              f"OBSERVE {observe}\nEXPLAIN {explain}\nCHECK   {check}\nNEXT    {next_step}", flush=True)

    @torch.no_grad()
    def policy_action(self, policy, obs):
        return policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    def evaluate(self):
        rows = []
        for episode in range(self.args.eval_episodes):
            reset_seed = stable_seed(self.args.task, self.args.seed, "evaluation", episode)
            obs, _ = self.eval_env.reset(seed=reset_seed)
            episode_return = 0.0
            length = 0
            while True:
                action = self.agent.select_action(obs)
                obs, reward, terminated, truncated, _ = self.eval_env.step(env_action(action, self.eval_env))
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            self.eval_steps += length
            rows.append(dict(task=self.args.task, arm=self.args.arm, seed=self.args.seed,
                             delay=self.args.delay, step=self.steps, evaluation_episode=episode,
                             reset_seed=reset_seed, episode_return=episode_return, episode_length=length))
        append_csv(self.out / "evaluation.csv", rows)
        returns = np.array([r["episode_return"] for r in rows])
        counts = self.replay.total_sample_counts
        sampled_total = sum(counts.values())
        sampled_fraction = counts["ea"] / sampled_total if sampled_total else None
        resident = self.replay.source_counts
        resident_fraction = resident["ea"] / len(self.replay) if len(self.replay) else None
        reward_mean, reward_sd = float(returns.mean()), float(returns.std(ddof=1))
        finite = bool(np.isfinite(returns).all())
        if not finite:
            raise FloatingPointError("Nonfinite evaluation return")
        self.diagnostics(sampled_fraction)
        if self.steps <= self.args.warmup:
            explanation = ("Warmup fills replay with random actions. No gradient update has occurred, "
                           "so the deterministic learner should match its initial evaluation.")
        else:
            explanation = (f"The actor is tested without exploration noise on the same held-out starts. "
                           f"A less negative return is better. Evaluation episodes do not enter replay. "
                           f"{self.agent.updates:,} critic updates have trained from stored experience.")
        share_text = "no minibatches yet" if sampled_fraction is None else f"{sampled_fraction:.1%} EA samples"
        self.event("evaluation",
                   f"Mean native return {reward_mean:.2f} (episode SD {reward_sd:.2f}); "
                   f"{self.eval_steps:,} separate evaluation interactions.",
                   explanation,
                   f"Finite returns; {share_text}; {self.agent.updates // 2:,} actor updates. "
                   "Episode SD describes evaluation starts, not uncertainty across training seeds.",
                   "Complete the run and audit costs." if self.steps == self.args.steps else
                   f"Continue to the next {self.args.eval_every:,}-transition checkpoint.",
                   return_mean=reward_mean, return_sd=reward_sd, finite=finite,
                   replay_sampled_ea_fraction=sampled_fraction,
                   replay_resident_ea_fraction=resident_fraction)
        self.save()

    @torch.no_grad()
    def diagnostics(self, sampled_fraction):
        disagreement = None
        if self.probe is not None and self.args.arm != "td3":
            actions = torch.stack([p(self.probe) for p in self.population])
            distances = [float(((actions[i] - actions[j]) / 2).square().mean().sqrt())
                         for i in range(4) for j in range(i + 1, 4)]
            disagreement = float(np.mean(distances))
        resident = self.replay.metadata(current_step=self.steps)
        row = dict(step=self.steps, action_disagreement=disagreement,
                   sampled_ea_fraction=sampled_fraction)
        for label, source in (("ea", 1), ("learner", 0)):
            mask = resident["sources"] == source
            row[f"age_{label}_mean"] = float(resident["ages"][mask].mean()) if mask.any() else None
            row[f"reuse_{label}_mean"] = float(resident["reuse_counts"][mask].mean()) if mask.any() else None
            row[f"td_abs_{label}"] = self.last_diagnostics.get(f"td_abs_{label}")
        row["q_disagreement"] = self.last_diagnostics.get("q_disagreement")
        append_csv(self.out / "diagnostics.csv", [row])

    def save(self):
        actor_path = self.checkpoints / f"actor_{self.steps:07d}.pt"
        torch.save(self.agent.actor.state_dict(), actor_path)
        state = {
            "agent": self.agent.state_dict(), "replay": self.replay.state_dict(),
            "population": [p.state_dict() for p in self.population],
            "lineages": self.lineages, "rngs": {k: v.bit_generator.state for k, v in self.rngs.items()},
            "target_generator": self.target_generator.get_state(),
            "steps": self.steps, "eval_steps": self.eval_steps, "generation": self.generation,
            "episode": self.episode, "update_credit": self.update_credit,
            "elapsed_s": self.elapsed(), "last_diagnostics": self.last_diagnostics,
            "metadata": self.metadata,
        }
        temporary = self.checkpoints / "latest.tmp"
        torch.save(state, temporary)
        temporary.replace(self.checkpoints / "latest.pt")
        append_csv(self.out / "checkpoint_manifest.csv", [
            {"step": self.steps, "kind": "actor", "path": actor_path.name, "sha256": sha_file(actor_path)},
            {"step": self.steps, "kind": "latest_full", "path": "latest.pt",
             "sha256": sha_file(self.checkpoints / "latest.pt")},
        ])

    def restore(self):
        state = torch.load(self.checkpoints / "latest.pt", map_location="cpu", weights_only=False)
        prior = state["metadata"]
        for key in ("task", "arm", "seed", "reward_delay", "training_budget", "warmup",
                    "evaluation_interval", "evaluation_episodes", "source_sha256", "calibration_sha256"):
            if prior[key] != self.metadata[key]:
                raise ValueError(f"Resume configuration/source mismatch: {key}")
        self.agent.load_state_dict(state["agent"])
        self.replay.load_state_dict(state["replay"])
        for policy, weights in zip(self.population, state["population"]):
            policy.load_state_dict(weights)
        self.lineages = state["lineages"]
        for key, rng in self.rngs.items():
            rng.bit_generator.state = state["rngs"][key]
        self.target_generator.set_state(state["target_generator"])
        for key in ("steps", "eval_steps", "generation", "episode", "update_credit", "last_diagnostics"):
            setattr(self, key, state[key])
        self.elapsed_before = state["elapsed_s"]
        self.metadata = prior
        self.metadata["status"] = "running"
        discarded = []
        # Remove rows written after the last durable checkpoint before replaying lost work.
        for name, field in (("evaluation.csv", "step"), ("generations.csv", "step"),
                            ("episodes.csv", "step"), ("checkpoint_manifest.csv", "step"),
                            ("diagnostics.csv", "step")):
            path = self.out / name
            if path.exists():
                with path.open() as handle:
                    reader = csv.DictReader(handle)
                    fields = reader.fieldnames
                    all_rows = list(reader)
                    rows = [r for r in all_rows if int(r[field]) <= self.steps]
                    discarded.extend(dict(file=name, record=r) for r in all_rows if int(r[field]) > self.steps)
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        for name, field in (("lineage.jsonl", "step"), ("events.jsonl", "env_steps")):
            path = self.out / name
            if path.exists():
                records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                kept = [r for r in records if int(r[field]) <= self.steps]
                discarded.extend(dict(file=name, record=r) for r in records if int(r[field]) > self.steps)
                path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        if discarded:
            with (self.out / "discarded_after_interruption.jsonl").open("a") as handle:
                for record in discarded:
                    handle.write(json.dumps(record) + "\n")
        history = self.out / "restart_history.jsonl"
        with history.open("a") as handle:
            handle.write(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                        restored_step=self.steps, discarded_logged_records=len(discarded),
                                        note="Discarded records are preserved; partially unlogged lost work may be unknown.")) + "\n")
        events_path = self.out / "events.jsonl"
        if events_path.exists():
            with (self.out / "journal.md").open("w") as journal:
                for line in events_path.read_text().splitlines():
                    record = json.loads(line)
                    journal.write(f"\n### {record['event']} · {record['env_steps']:,} training transitions\n\n")
                    for field in ("observe", "explain", "check", "next"):
                        journal.write(f"**{field.title()}:** {record[field]}\n\n")

    def collect_generation(self):
        warmup = self.steps < self.args.warmup
        fitness = []
        episode_rows = []
        for slot in range(5):
            source = "ea" if self.args.arm != "td3" and slot < 4 else "learner"
            policy = self.population[slot] if source == "ea" else self.agent.actor
            reset_seed = stable_seed(self.args.task, self.args.seed, "training_reset", self.generation, slot)
            obs, _ = self.env.reset(seed=reset_seed)
            emitted_total = native_total = 0.0
            length = 0
            while True:
                if warmup:
                    normalized = self.rngs["warmup"].uniform(-1, 1, size=self.act_dim).astype(np.float32)
                else:
                    normalized = self.policy_action(policy, obs)
                    if source == "learner":
                        normalized = normalized + self.rngs["exploration"].normal(0, 0.1, self.act_dim)
                    normalized = np.clip(normalized, -1, 1).astype(np.float32)
                new_obs, reward, terminated, truncated, info = self.env.step(env_action(normalized, self.env))
                self.steps += 1
                length += 1
                self.replay.add(obs, normalized, reward, new_obs, terminated, truncated,
                                source, self.steps, self.episode, self.generation, warmup)
                emitted_total += float(reward)
                native_total += float(info["native_reward"])
                obs = new_obs
                if terminated or truncated:
                    break
            if length != self.horizon:
                raise ValueError("Unexpected horizon: fixed-budget collection assumption failed")
            if not np.isclose(emitted_total, native_total, rtol=1e-6, atol=1e-5):
                raise ValueError("Reward aggregation changed the episode total")
            if slot < 4:
                fitness.append(emitted_total)
            episode_rows.append(dict(step=self.steps, generation=self.generation, episode=self.episode,
                                     slot=slot, source=source, warmup=warmup, reset_seed=reset_seed,
                                     emitted_return=emitted_total, native_return=native_total, length=length))
            self.episode += 1
        append_csv(self.out / "episodes.csv", episode_rows)
        return warmup, fitness

    def run(self):
        try:
            if not self.args.resume:
                self.event("run_start", f"Starting {self.args.task}, {self.args.arm}, seed {self.args.seed}.",
                           MEANING[self.args.arm] + " A seed labels an independent repeat, not a skill level.",
                           f"One CPU thread; {self.args.steps:,} training transitions; "
                           f"{self.args.warmup:,} warmup transitions; separate evaluation RNG.",
                           "Measure the untrained policy, then fill replay with random actions.")
                self.evaluate()
            else:
                self.event("resume", f"Restored checkpoint at {self.steps:,} steps.",
                           "Replay, networks, optimizers, RNG states, and update credits were restored.",
                           "Source hashes and run configuration match.", "Resume at the next complete generation.")
            while self.steps < self.args.steps:
                warmup, fitness = self.collect_generation()
                metrics = {}
                if not warmup:
                    self.update_credit += 0.25 * self.generation_steps
                    updates = int(self.update_credit)
                    for _ in range(updates):
                        batch = self.replay.sample(128, ARMS[self.args.arm], self.rngs["replay"],
                                                   current_step=self.steps)
                        metrics = self.agent.update(batch, self.target_generator)
                    self.update_credit -= updates
                    self.last_diagnostics = metrics
                    if self.args.arm != "td3":
                        inject = (self.steps - self.args.warmup) % 5000 == 0
                        old_population = self.population
                        self.population, self.lineages, events = evolve_population(
                            self.population, fitness, self.rngs["selection"], self.rngs["mutation"],
                            self.agent.actor, inject, self.generation + 1, self.lineages)
                        if self.probe is not None:
                            with torch.no_grad():
                                for event in events:
                                    child = self.population[event["slot"]](self.probe)
                                    parent_policy = self.agent.actor if event["parent_index"] is None else old_population[event["parent_index"]]
                                    parent = parent_policy(self.probe)
                                    event["action_rms_from_parent"] = float(((child - parent) / 2).square().mean().sqrt())
                        with (self.out / "lineage.jsonl").open("a") as handle:
                            for event in events:
                                handle.write(json.dumps(dict(step=self.steps, **event)) + "\n")
                self.generation += 1
                samples = self.replay.total_sample_counts
                row = dict(step=self.steps, generation=self.generation, warmup=warmup,
                           critic_updates=self.agent.updates, actor_updates=self.agent.updates // 2,
                           update_credit=self.update_credit, elapsed_s=self.elapsed(),
                           fitness_mean=float(np.mean(fitness)), fitness_max=float(np.max(fitness)),
                           resident_ea=self.replay.source_counts["ea"],
                           resident_learner=self.replay.source_counts["learner"],
                           sampled_ea=samples["ea"], sampled_learner=samples["learner"])
                for key in ("critic_loss", "actor_loss", "td_abs_mean", "q_disagreement"):
                    row[key] = metrics.get(key)
                append_csv(self.out / "generations.csv", [row])
                if self.steps % self.args.eval_every == 0:
                    self.evaluate()
            expected = int((self.args.steps - self.args.warmup) * 0.25)
            if self.agent.updates != expected:
                raise ValueError(f"Update budget mismatch: {self.agent.updates} != {expected}")
            totals = self.replay.total_sample_counts
            if sum(totals.values()) != expected * 128:
                raise ValueError("Replay sample count mismatch")
            self.metadata.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                                 total_training_transitions=self.steps, total_evaluation_transitions=self.eval_steps,
                                 critic_updates=self.agent.updates, actor_updates=self.agent.updates // 2,
                                 sampled_transitions=totals, elapsed_s=self.elapsed(),
                                 peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                 checkpoint_bytes=sum(p.stat().st_size for p in self.checkpoints.glob("*")))
            write_json(self.out / "metadata.json", self.metadata)
            self.event("run_complete", f"Completed {self.steps:,} training transitions in {self.elapsed():.1f} seconds.",
                       "This run is one noisy observation. Comparing several seeds helps separate a method's pattern from luck.",
                       f"Budget verified: {expected:,} twin-critic steps, {expected // 2:,} actor steps; "
                       f"{self.eval_steps:,} separately counted evaluation steps.",
                       "Compare matched pilot runs and assess learnability and CPU cost; do not test H1 on this pilot.")
        except Exception as error:
            self.metadata.update(status="failed", error_type=type(error).__name__, error=str(error),
                                 total_training_transitions=self.steps, elapsed_s=self.elapsed())
            write_json(self.out / "metadata.json", self.metadata)
            raise
        finally:
            self.env.close()
            self.eval_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("Pendulum-v1", "Reacher-v5"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration", help="Independent calibration JSON with fixed probe observations")
    parser.add_argument("--resume", action="store_true")
    Experiment(parser.parse_args()).run()


if __name__ == "__main__":
    main()
