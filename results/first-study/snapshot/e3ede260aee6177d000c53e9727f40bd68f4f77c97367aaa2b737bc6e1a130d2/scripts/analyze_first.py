#!/usr/bin/env python3
"""Audit and analyze the frozen, 40-run Pendulum confirmatory experiment."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

TASK = "Pendulum-v1"
ARMS = ("erl_pooled", "erl_balanced")
DELAYS = (1, 20)
SEEDS = tuple(range(100, 110))
STEPS = np.arange(0, 200001, 10000)
ANALYSIS_SEED = 20261012
BOOTSTRAPS = 10000
THRESHOLD = .05
COLORS = ("#E69F00", "#009E73")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else np.nan


def auc(curves, steps=STEPS):
    """Time-average a complete trajectory, including initialization and warmup."""
    values = np.asarray(curves, dtype=float)
    steps = np.asarray(steps, dtype=float)
    require(values.shape[-1] == len(steps) and len(steps) > 1, "AUC grid mismatch")
    require(np.isfinite(values).all() and np.all(np.diff(steps) > 0), "Invalid AUC trajectory/grid")
    return np.sum((values[..., :-1] + values[..., 1:]) * .5 * np.diff(steps), axis=-1) / (steps[-1] - steps[0])


def paired_statistics(curves, anchor, analysis_seed=ANALYSIS_SEED, resamples=BOOTSTRAPS):
    """Resample whole seed records jointly across arms, delays, and checkpoints."""
    curves = np.asarray(curves, dtype=float)
    require(curves.ndim == 4 and curves.shape[1:3] == (2, 2), "Expected [seed,delay,arm,checkpoint]")
    require(len(curves) >= 2 and np.isfinite(anchor) and anchor < 0, "Invalid seed count or anchor")
    raw_auc = auc(curves)
    normalized_auc = (raw_auc - anchor) / -anchor
    differences = normalized_auc[:, :, 1] - normalized_auc[:, :, 0]
    indices = np.random.default_rng(analysis_seed).integers(0, len(curves), size=(resamples, len(curves)))
    boot_differences = differences[indices].mean(axis=1)
    interaction = differences[:, 1] - differences[:, 0]
    boot_interaction = boot_differences[:, 1] - boot_differences[:, 0]

    def effect(values, boot):
        interval = np.quantile(boot, [.025, .975])
        direction = "positive" if interval[0] > 0 else "negative" if interval[1] < 0 else "inconclusive"
        return dict(mean=float(np.mean(values)), ci95=[float(v) for v in interval],
                    direction=direction, clears_practical_threshold=bool(interval[0] > THRESHOLD),
                    per_seed=[float(v) for v in values])

    return dict(raw_auc=raw_auc, normalized_auc=normalized_auc, differences=differences,
                indices=indices, bootstrap_differences=boot_differences,
                effects={"H1_delayed": effect(differences[:, 1], boot_differences[:, 1]),
                         "immediate_descriptive": effect(differences[:, 0], boot_differences[:, 0]),
                         "H2_delay_interaction": effect(interaction, boot_interaction)})


def read_evaluation(directory, metadata, paired_resets):
    rows = read_csv(directory / "evaluation.csv")
    keys = [(int(r["step"]), int(r["evaluation_episode"])) for r in rows]
    expected = {(int(step), episode) for step in STEPS for episode in range(5)}
    require(len(keys) == len(set(keys)) == 105 and set(keys) == expected,
            "Missing or duplicate evaluation rows; interpolation and complete-case deletion are forbidden")
    matrix, local_resets = np.empty((21, 5)), {}
    for row in rows:
        require((row["task"], row["arm"], int(row["delay"]), int(row["seed"])) ==
                (TASK, metadata["arm"], metadata["reward_delay"], metadata["seed"]), "Evaluation identity mismatch")
        require(int(row["episode_length"]) == 200, "Unexpected evaluation horizon")
        key = (int(row["seed"]), int(row["evaluation_episode"]))
        reset = int(row["reset_seed"])
        require(local_resets.setdefault(key, reset) == reset and paired_resets.get(key, reset) == reset,
                "Evaluation reset seeds are not paired across checkpoints, arms, and delays")
        matrix[int(row["step"]) // 10000, key[1]] = float(row["episode_return"])
    require(np.isfinite(matrix).all(), "Nonfinite evaluation return")
    require(np.array_equal(matrix[0], matrix[1]), "Learner evaluation changed during random-action warmup")
    paired_resets.update(local_resets)
    return matrix


def audit_runs(root, freeze, manifest, expected_evaluation_resets=None):
    """Keep every planned run in the completion ledger, even on validation failure."""
    expected = {(TASK, arm, delay, seed) for arm in ARMS for delay in DELAYS for seed in SEEDS}
    identities = [(r["task"], r["arm"], int(r["reward_delay"]), int(r["seed"])) for r in manifest]
    require(len(manifest) == 40 and len(set(identities)) == 40 and set(identities) == expected,
            "Frozen manifest must contain exactly the planned 40 identities")
    require(len({r["run_id"] for r in manifest}) == 40, "Duplicate run ID in manifest")
    ledger, records, paired_resets = [], {}, dict(expected_evaluation_resets or {})
    initial_evaluations = {}
    planned = {row["run_id"] for row in manifest}
    unexpected = sorted(path.parent.name for path in root.glob("*/metadata.json") if path.parent.name not in planned)
    for row, identity in zip(manifest, identities):
        directory = root / row["run_id"]
        entry = dict(run_id=row["run_id"], task=identity[0], arm=identity[1], delay=identity[2], seed=identity[3])
        path = directory / "metadata.json"
        if not path.exists():
            ledger.append(dict(entry, status="missing", reason="No metadata.json"))
            continue
        try:
            meta = json.loads(path.read_text())
            require((meta["task"], meta["arm"], meta["reward_delay"], meta["seed"]) == identity, "Run identity mismatch")
            require(meta["run_id"] == row["run_id"], "Run ID mismatch")
            require(meta["source_sha256"] == freeze["source_sha256"], "Mixed or unfrozen trainer source hashes")
            require(meta["calibration_sha256"] == freeze["calibration_sha256"], "Calibration hash mismatch")
            require(meta["protocol_sha256"] == freeze["freeze_id"], "Frozen protocol/configuration identity mismatch")
            require(meta["purpose"] == "confirmatory" and meta["diagnostics"] == "full", "Wrong run purpose/diagnostic mode")
            require(meta["device"] == "cpu", "Unexpected training device")
            if meta["status"] != "completed":
                ledger.append(dict(entry, status=meta["status"], reason=meta.get("error", "Run has not completed")))
                continue
            for key, value in (("training_budget", 200000), ("total_training_transitions", 200000),
                               ("warmup", 10000), ("critic_updates", 47500), ("actor_updates", 23750),
                               ("evaluation_interval", 10000), ("evaluation_episodes", 5),
                               ("total_evaluation_transitions", 21000), ("horizon", 200),
                               ("threads", 1), ("batch_size", 128), ("replay_capacity", 100000)):
                require(meta[key] == value, f"Incorrect {key}: {meta[key]}")
            samples = meta["sampled_transitions"]
            require(sum(samples.values()) == 6080000, "Incorrect replay-sample budget")
            if identity[1] == "erl_balanced":
                require(samples["ea"] == samples["learner"] == 3040000, "Balanced replay quota mismatch")
            candidate_resets = dict(paired_resets)
            matrix = read_evaluation(directory, meta, candidate_resets)
            require(identity[3] not in initial_evaluations or
                    np.array_equal(matrix[0], initial_evaluations[identity[3]]),
                    "Initial learner evaluations differ across arms or delays")
            episodes = read_csv(directory / "episodes.csv")
            require(len(episodes) == 1000 and len({int(r["episode"]) for r in episodes}) == 1000,
                    "Missing or duplicate collection episodes")
            require(sum(int(r["length"]) for r in episodes) == 200000 and
                    sum(int(r["length"]) for r in episodes if r["source"] == "ea") == 160000,
                    "Collected budget/source split mismatch")
            for r in episodes:
                require(np.isfinite(float(r["native_return"])) and np.isclose(float(r["native_return"]),
                        float(r["emitted_return"]), rtol=1e-6, atol=1e-5), "Native/emitted episode totals differ")
            generations = read_csv(directory / "generations.csv")
            require([int(r["step"]) for r in generations] == list(range(1000, 200001, 1000)),
                    "Generation grid is incomplete or duplicated")
            for r in generations:
                updates = max(0, int(r["step"]) - 10000) // 4
                require(int(r["critic_updates"]) == updates and int(r["actor_updates"]) == updates // 2,
                        "Generation update count mismatch")
            records[(identity[3], identity[2], identity[1])] = dict(metadata=meta, directory=directory,
                    curve=matrix.mean(axis=1), evaluation=matrix, generations=generations)
            paired_resets.update(candidate_resets)
            initial_evaluations[identity[3]] = matrix[0].copy()
            ledger.append(dict(entry, status="validated", reason="All prespecified run checks passed"))
        except (KeyError, ValueError, OSError, TypeError) as error:
            ledger.append(dict(entry, status="invalid", reason=str(error)))
    complete = len(records) == 40 and not unexpected
    return dict(complete=complete, expected_runs=40, validated_runs=len(records), runs=ledger,
                unexpected_run_directories=unexpected, inferential_analysis_allowed=complete), records


def validate_study(root):
    from study_common import verify_freeze
    freeze, manifest = verify_freeze(root, check_worktree=False)
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == freeze["files"]["scripts/analyze_first.py"],
            "Executing analysis script differs from its frozen version")
    config = freeze["config"]
    expected_analysis = dict(analysis_seed=ANALYSIS_SEED, bootstrap_resamples=BOOTSTRAPS,
                             practical_effect_threshold=THRESHOLD, primary_delay=20,
                             interval_percentiles=[2.5, 97.5])
    require(all(config["analysis"].get(key) == value for key, value in expected_analysis.items()),
            "Unexpected frozen analysis settings")
    seed_manifest = json.loads((root / "seed-manifest.json").read_text())
    blocks = seed_manifest["blocks"]
    require(len(blocks) == 10 and {b["seed"] for b in blocks} == set(SEEDS), "Unexpected frozen seed blocks")
    declared_resets = {}
    for block in blocks:
        require(block["task"] == TASK and len(block["evaluation"]) == 5, "Invalid frozen evaluation reset table")
        declared_resets.update({(block["seed"], episode): reset for episode, reset in enumerate(block["evaluation"])})
    completion, records = audit_runs(root, freeze, manifest, declared_resets)
    return freeze, completion, records


def completion_report(root, completion):
    dump(root / "analysis-validation.json", completion)
    lines = ["# Confirmatory experiment completion audit", "",
             f"Validated **{completion.get('validated_runs', 0)} of 40** planned runs.", "",
             "Inference requires every planned arm × delay × seed record. Failed, missing, or invalid runs are not dropped or replaced.", "",
             "| Run | Status | Reason |", "|---|---|---|"]
    for row in completion.get("runs", []):
        lines.append(f"| {row['run_id']} | {row['status']} | {row['reason'].replace('|', '/')} |")
    for error in completion.get("global_errors", []):
        lines += ["", f"Validation error: {error}"]
    if completion.get("unexpected_run_directories"):
        lines += ["", "Unexpected directories: " + ", ".join(completion["unexpected_run_directories"])]
    text = "\n".join(lines) + "\n"
    (root / "completion.md").write_text(text)
    if not completion["complete"]:
        (root / "report.md").write_text(text + "\n**H1/H2 estimates and confidence intervals are withheld. No primary inferential claim is made.**\n")
        dump(root / "summary.json", dict(status="incomplete_or_invalid", completion=completion, effects=None))


def save_figure(fig, root, name, caption, figures):
    import matplotlib.pyplot as plt
    fig.tight_layout(rect=(0, .075, 1, .94))
    fig.text(.015, .012, caption, fontsize=8, va="bottom", wrap=True)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    figures.append(dict(name=name, caption=caption))
    plt.close(fig)


def make_figures(root, records, curves, stats):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/erl-mplcache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "svg.fonttype": "none"})
    figures = []
    bootstrap_curves = curves[stats["indices"]].mean(axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for di, delay in enumerate(DELAYS):
        for ai, arm in enumerate(ARMS):
            ax = axes[di, ai]
            for line in curves[:, di, ai]:
                ax.plot(STEPS / 1000, line, color=COLORS[ai], alpha=.22, lw=.8)
            lo, hi = np.quantile(bootstrap_curves[:, di, ai], [.025, .975], axis=0)
            ax.fill_between(STEPS / 1000, lo, hi, color=COLORS[ai], alpha=.17)
            ax.plot(STEPS / 1000, curves[:, di, ai].mean(axis=0), color=COLORS[ai], lw=2)
            ax.axvspan(0, 10, color="grey", alpha=.08)
            ax.set(title=f"{'Pooled' if ai == 0 else 'Balanced'} replay · delay {delay}", xlabel="Training transitions (thousands)", ylabel="Native evaluation return ↑")
    fig.suptitle("Pendulum learning curves: all ten training seeds")
    save_figure(fig, root, "learning-curves", "Thin lines: seeds; thick lines: seed mean. Shading: pointwise 95% percentile intervals from 10,000 whole-seed bootstrap resamples, not simultaneous bands. Five held-out episodes per checkpoint; no smoothing.", figures)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for di, delay in enumerate(DELAYS):
        effect = stats["effects"]["H1_delayed" if delay == 20 else "immediate_descriptive"]
        ax = axes[di]
        ax.scatter(range(10), stats["differences"][:, di], c="#0072B2")
        ax.plot([11, 11], effect["ci95"], c="#D55E00", lw=3)
        ax.scatter(11, effect["mean"], c="#D55E00", marker="D", s=55)
        ax.axhline(0, color="black", lw=.8)
        ax.axhline(THRESHOLD, color="#009E73", ls="--", label="Practical threshold +0.05")
        ax.set(xticks=list(range(10)) + [11], xticklabels=[str(s) for s in SEEDS] + ["Mean"],
               ylabel="Balanced − pooled normalized AUC", title=f"Delay {delay}" + (" · primary H1" if delay == 20 else " · reference"))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("Paired effects retain the independent training seed as the unit of inference")
    save_figure(fig, root, "paired-effects", "Each blue point compares matched trained agents; the diamond and line show the paired mean and 95% percentile interval. Normalization uses the frozen independent random-policy anchor. Ten whole-seed records, 10,000 paired resamples.", figures)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for values in stats["differences"]:
        axes[0].plot([0, 1], values, "o-", alpha=.4, color="#0072B2")
    axes[0].plot([0, 1], stats["differences"].mean(axis=0), "D-", color="#D55E00", lw=2)
    axes[0].set(xticks=[0, 1], xticklabels=["Immediate", "20-step aggregation"], ylabel="Balanced − pooled normalized AUC")
    effect = stats["effects"]["H2_delay_interaction"]
    axes[1].scatter(range(10), effect["per_seed"], color="#0072B2")
    axes[1].plot([11, 11], effect["ci95"], color="#D55E00", lw=3)
    axes[1].scatter(11, effect["mean"], color="#D55E00", marker="D")
    axes[1].set(xticks=list(range(10)) + [11], xticklabels=[str(s) for s in SEEDS] + ["Mean"], ylabel="Delayed effect − immediate effect")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.axhline(0, color="black", lw=.8)
    fig.suptitle("Secondary H2: does aggregation change the replay effect?")
    save_figure(fig, root, "delay-interaction", "Left: connected contrasts from the same seed across delays. Right: per-seed difference of differences and its mean/95% interval. The same bootstrap seed indices are used jointly across both delays; H2 is secondary.", figures)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for di, delay in enumerate(DELAYS):
        for si in range(10):
            axes[di].plot([0, 1], curves[si, di, :, -1], color="grey", alpha=.25, lw=.7)
        for ai in range(2):
            axes[di].scatter(ai + np.linspace(-.06, .06, 10), curves[:, di, ai, -1], color=COLORS[ai], alpha=.8)
        axes[di].set(xticks=[0, 1], xticklabels=["Pooled", "Balanced"], title=f"Delay {delay}", ylabel="Final native return ↑")
    fig.suptitle("Final performance at exactly 200,000 training transitions")
    save_figure(fig, root, "final-returns", "One point per training seed; grey lines connect matched seeds. Each point averages five evaluation episodes. Final return is secondary; primary AUC includes the complete trajectory and warmup.", figures)
    diagnostic_figures(root, records, figures)
    return figures


def diagnostic_figures(root, records, figures):
    """Diagnostics explain measured processes; they are not additional hypothesis tests."""
    import matplotlib.pyplot as plt
    conditions = [(delay, arm) for delay in DELAYS for arm in ARMS]
    tables = {}
    for key, record in records.items():
        tables[key] = {name: read_csv(record["directory"] / name) if (record["directory"] / name).exists() else []
                       for name in ("diagnostic_replay.csv", "diagnostic_critic.csv", "diagnostics.csv")}
    age_fields = ("age_bin_0_1000", "age_bin_1000_5000", "age_bin_5000_10000", "age_bin_10000_25000",
                  "age_bin_25000_50000", "age_bin_50000_100000", "age_bin_100000_200000", "age_bin_200000_plus")
    if all(t["diagnostic_replay.csv"] for t in tables.values()):
        fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True, sharey=True)
        for ci, (delay, arm) in enumerate(conditions):
            for sj, source in enumerate(("ea", "learner")):
                arrays = []
                for seed in SEEDS:
                    rows = tables[seed, delay, arm]["diagnostic_replay.csv"]
                    data = np.full((len(STEPS), 8), np.nan)
                    for row in rows:
                        if row["population"] == "diagnostic_sample" and row["source"] == source and int(row["n"]) > 0:
                            data[int(row["step"]) // 10000] = [number(row, key) / int(row["n"]) for key in age_fields]
                    arrays.append(data)
                data = finite_mean(arrays)
                axes[ci, sj].imshow(data.T, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="cividis", extent=(-5, 205, -.5, 7.5))
                axes[ci, sj].set(title=f"{arm.replace('erl_', '')} · delay {delay} · {source}",
                                 yticks=range(8), yticklabels=["0–1k", "1–5k", "5–10k", "10–25k", "25–50k", "50–100k", "100–200k", "200k+"],
                                 xlabel="Training transitions (thousands)", ylabel="Transition age bin")
        fig.suptitle("Which ages appear in an independent diagnostic sample? Color: within-source fraction (0–1)")
        save_figure(fig, root, "replay-age-heatmaps", "Each column averages ten seeds' within-source age fractions from a bounded 128-draw diagnostic batch using the arm's sampler. Colors use a common 0–1 scale; missing observations are blank. These are not historical training-sample counts.", figures)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, (delay, arm) in zip(axes.flat, conditions):
            for sj, source in enumerate(("ea", "learner")):
                arrays = []
                for seed in SEEDS:
                    rows = tables[seed, delay, arm]["diagnostic_replay.csv"]
                    data = np.full((len(STEPS), 3), np.nan)
                    for row in rows:
                        if row["population"] == "resident" and row["source"] == source and int(row["n"]) > 0:
                            data[int(row["step"]) // 10000] = [number(row, f"reuse_q{q}") for q in (25, 50, 75)]
                    arrays.append(data)
                mean = finite_mean(arrays)
                ax.fill_between(STEPS / 1000, mean[:, 0], mean[:, 2], color=COLORS[sj], alpha=.18)
                ax.plot(STEPS / 1000, mean[:, 1], color=COLORS[sj], label=source)
            ax.set(title=f"{arm.replace('erl_', '')} · delay {delay}", xlabel="Training transitions (thousands)", ylabel="Resident transition reuse count")
            ax.legend()
        fig.suptitle("Replay reuse after FIFO eviction begins")
        save_figure(fig, root, "replay-reuse-quantiles", "Lines: mean of ten seeds' resident reuse medians; shading: mean resident 25th–75th percentiles, NOT uncertainty intervals. Each resident record's count includes draws while it was retained. Evicted transitions are absent.", figures)
    if all(t["diagnostic_critic.csv"] for t in tables.values()):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (delay, arm) in zip(axes.flat, conditions):
            for sj, source in enumerate(("ea", "learner")):
                lines = []
                for seed in SEEDS:
                    rows = tables[seed, delay, arm]["diagnostic_critic.csv"]
                    vals = {int(r["phase"]): number(r, "td_abs_q50") for r in rows if int(r["step"]) == 200000 and r["source"] == source}
                    line = [vals.get(phase, np.nan) for phase in range(1, delay + 1)]
                    ax.plot(range(1, delay + 1), line, "o-", color=COLORS[sj], alpha=.2, ms=2)
                    lines.append(line)
                ax.plot(range(1, delay + 1), finite_mean(lines), "o-", color=COLORS[sj], label=source, ms=4)
            ax.set(title=f"{arm.replace('erl_', '')} · delay {delay}", xlabel="Reward-block phase", ylabel="Absolute TD residual median")
            ax.legend()
        fig.suptitle("Final critic residuals by source and reward-block phase")
        save_figure(fig, root, "critic-reward-phase", "Thin lines: seeds; thick: mean of available seed medians. Each source uses 128 independent diagnostic draws, then phase subsets; empty subsets are missing. Residuals average both critics' absolute errors against emitted-reward TD targets, not true value error.", figures)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
        for ax, (delay, arm) in zip(axes.flat, conditions):
            lines = []
            for seed in SEEDS:
                rows = tables[seed, delay, arm]["diagnostic_critic.csv"]
                vals = {int(r["step"]): number(r, "gradient_cosine") for r in rows if r["source"] == "cross_source" and r["gradient_status"] == "ok"}
                line = [vals.get(int(step), np.nan) for step in STEPS]
                ax.plot(STEPS / 1000, line, color="#0072B2", alpha=.2)
                lines.append(line)
            ax.plot(STEPS / 1000, finite_mean(lines), color="#0072B2", lw=2)
            ax.axhline(0, color="black", lw=.7)
            ax.set(ylim=(-1.05, 1.05), title=f"{arm.replace('erl_', '')} · delay {delay}", xlabel="Training transitions (thousands)", ylabel="Critic-gradient cosine")
        fig.suptitle("Do the two replay sources suggest aligned critic updates?")
        save_figure(fig, root, "gradient-alignment", "Cosine compares joint twin-critic loss gradients from independent source minibatches. +1: aligned; −1: opposed; zero/missing gradients are omitted, not set to zero. Thin lines: seeds; thick: available-seed mean. Alignment alone cannot establish causal benefit.", figures)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    for ci, (delay, arm) in enumerate(conditions):
        for seed in SEEDS:
            record = records[seed, delay, arm]
            generations = [r for r in record["generations"] if int(r["step"]) > 10000]
            axes[0, ci].plot([int(r["step"]) / 1000 for r in generations], [number(r, "fitness_mean") for r in generations], color=COLORS[ARMS.index(arm)], alpha=.2)
            rows = tables[seed, delay, arm]["diagnostics.csv"]
            axes[1, ci].plot([int(r["step"]) / 1000 for r in rows], [number(r, "action_disagreement") for r in rows], color=COLORS[ARMS.index(arm)], alpha=.35)
        axes[0, ci].set(title=f"{arm.replace('erl_', '')} · delay {delay}", ylabel="Mean population fitness")
        axes[1, ci].set(ylabel="Normalized action disagreement", xlabel="Training transitions (thousands)")
    fig.suptitle("Population training behavior and fixed-probe diversity")
    save_figure(fig, root, "population-diagnostics", "Each trace is one seed. Fitness averages the four population training episodes and is not held-out learner evaluation. Diversity is pairwise RMS action disagreement on a fixed calibration probe; differences may be unhelpful exploration.", figures)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ci, (delay, arm) in enumerate(conditions):
        for si, seed in enumerate(SEEDS):
            meta = records[seed, delay, arm]["metadata"]
            for ax, (field, divisor, label) in zip(axes, (("elapsed_s", 60, "Per-run wall minutes"), ("peak_rss_mib", 1, "Peak process memory (MiB)"), ("total_diagnostic_seconds", 1, "Diagnostic wall seconds"))):
                ax.scatter(ci + (si - 4.5) * .018, meta.get(field, np.nan) / divisor, color=COLORS[ARMS.index(arm)], s=20)
                ax.set(xticks=range(4), xticklabels=[f"{a[4:]} d{d}" for d, a in conditions], ylabel=label)
                ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Measured cost of all forty runs")
    save_figure(fig, root, "compute-cost", "Each point is one run, including evaluation/checkpoint overhead. Per-run wall times overlap when workers run concurrently and are sensitive to contention. Peak RSS is per process; diagnostic timing is part of total runtime, not an additional cost.", figures)


def finite_mean(values):
    data = np.asarray(values, dtype=float)
    count = np.isfinite(data).sum(axis=0)
    return np.divide(np.nansum(data, axis=0), count, out=np.full(data.shape[1:], np.nan), where=count > 0)


def write_outputs(root, freeze, completion, records, figures_enabled=True):
    calibration = json.loads((root / freeze["snapshot"] / "calibration.json").read_text())
    anchor = calibration[TASK]["random_return_mean"]
    curves = np.array([[[records[seed, delay, arm]["curve"] for arm in ARMS] for delay in DELAYS] for seed in SEEDS])
    stats = paired_statistics(curves, anchor)
    rows = []
    for si, seed in enumerate(SEEDS):
        for di, delay in enumerate(DELAYS):
            for ai, arm in enumerate(ARMS):
                rows.append(dict(seed=seed, delay=delay, arm=arm, initial_return=float(curves[si, di, ai, 0]),
                                 final_return=float(curves[si, di, ai, -1]), raw_auc=float(stats["raw_auc"][si, di, ai]),
                                 normalized_auc=float(stats["normalized_auc"][si, di, ai])))
    with (root / "seed-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paired = [dict(seed=seed, immediate=float(stats["differences"][si, 0]), delayed=float(stats["differences"][si, 1]),
                   interaction=float(stats["differences"][si, 1] - stats["differences"][si, 0])) for si, seed in enumerate(SEEDS)]
    with (root / "paired-effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    figures = make_figures(root, records, curves, stats) if figures_enabled else []
    summary = dict(status="complete", freeze_id=freeze["freeze_id"], task=TASK, training_seeds=list(SEEDS),
                   config_sha256=freeze["config_sha256"], source_sha256=freeze["source_sha256"],
                   calibration_sha256=freeze["calibration_sha256"],
                   analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   normalization_anchor=anchor, normalization_uncertainty="conditional on frozen independent calibration",
                   analysis_seed=ANALYSIS_SEED, bootstrap_resamples=BOOTSTRAPS, practical_effect_threshold=THRESHOLD,
                   effects=stats["effects"], completion=completion, per_seed=rows, figures=figures,
                   training_transitions=8000000, evaluation_transitions=840000,
                   sum_run_wall_seconds=sum(r["metadata"]["elapsed_s"] for r in records.values()))
    dump(root / "summary.json", summary)
    lines = ["# First confirmatory study: Pendulum replay balance", "", "All 40 prespecified runs passed identity, hash, budget, reward-accounting, and paired-evaluation audits. This is a one-task, fixed-budget comparison; the pilot is excluded.", "", "| Contrast | Mean normalized AUC difference | 95% paired bootstrap interval | Direction |", "|---|---:|---:|---|"]
    for name, effect in stats["effects"].items():
        lines.append(f"| {name} | {effect['mean']:+.5f} | [{effect['ci95'][0]:+.5f}, {effect['ci95'][1]:+.5f}] | {effect['direction']} |")
    primary = stats["effects"]["H1_delayed"]
    lines += ["", f"H1 is the sole primary contrast: balanced minus pooled normalized learning-curve AUC under 20-step reward aggregation. Its interval {'clears' if primary['clears_practical_threshold'] else 'does not clear'} the prespecified +0.05 practical-effect threshold. An interval spanning zero is inconclusive, not evidence of equivalence. H2 is the secondary difference between delayed and immediate replay effects.", "", f"Intervals use 10,000 percentile bootstrap resamples of the ten complete training-seed records, jointly across arms, delays, and checkpoints (analysis seed {ANALYSIS_SEED}). Evaluation episodes/checkpoints are not independent trained replicates. No p-values or post hoc seed exclusions are used. The normalization anchor is {anchor:.15g}; inference conditions on this independently measured frozen reference.", "", "AUC uses the trapezoidal integral over all 21 checkpoints, divided by 200,000 interactions, including initialization and warmup. Each checkpoint averages five deterministic held-out episodes. [Raw per-seed endpoints](seed-results.csv) and [paired contrasts](paired-effects.csv) retain every seed.", "", f"The planned and verified cost is 8,000,000 training plus 840,000 evaluation interactions. Summed per-run wall time is {summary['sum_run_wall_seconds']/3600:.2f} hours; concurrent runs overlap, so this is not study elapsed time. Diagnostic quantities are exploratory and cannot establish the causal mechanism of a performance difference.", "", "Reward aggregation preserves undiscounted episode totals, but changes the discounted objective and introduces hidden reward-history dependence. Results do not isolate credit assignment, establish an optimal replay fraction, or generalize beyond this task/implementation. No-EA replay and standalone TD3 are absent from this first confirmatory stage, so it cannot establish that evolutionary experience is necessary or that hybridization beats TD3.", "", "[Full completion audit](completion.md) · [Machine-readable results](summary.json)"]
    for figure in figures:
        name = figure["name"]
        lines += ["", f"![{name}](figures/{name}.png)", "", figure["caption"], "", f"[SVG](figures/{name}.svg) · [PDF](figures/{name}.pdf)"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/first-study"))
    parser.add_argument("--validate-only", action="store_true", help="Audit only; do not estimate effects or render figures")
    parser.add_argument("--no-figures", action="store_true", help="Generate validated numerical results without plots")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        freeze, completion, records = validate_study(args.root)
    except (ValueError, KeyError, OSError, TypeError) as error:
        completion = dict(complete=False, validated_runs=0, expected_runs=40, global_errors=[str(error)],
                          inferential_analysis_allowed=False, runs=[])
    completion_report(args.root, completion)
    if not completion["complete"]:
        print("No inferential analysis: incomplete or invalid study. See completion.md.")
        return 2
    if not args.validate_only:
        write_outputs(args.root, freeze, completion, records, figures_enabled=not args.no_figures)
    print(f"Validated all 40 frozen runs; {'audit only' if args.validate_only else 'analysis complete'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
