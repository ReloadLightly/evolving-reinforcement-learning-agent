#!/usr/bin/env python3
"""Validate a complete CPU pilot and build descriptive figures and learning notes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/erl-mplcache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

TASKS = ("Pendulum-v1", "Reacher-v5")
ARMS = ("td3", "erl_pooled", "erl_balanced", "erl_no_ea_replay")
LABELS = ("TD3", "Pooled ERL", "Balanced ERL", "No EA replay")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
GRID = np.arange(0, 50001, 10000)
EXPLANATIONS = {
    "td3": "All five collection slots belong to the learner; all replay samples come from learner slots.",
    "erl_pooled": "Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix.",
    "erl_balanced": "The population collects 80% of training data, but each update uses 64 population and 64 learner samples.",
    "erl_no_ea_replay": "The population still collects, evolves, and receives learner injections; updates exclude its transitions.",
}


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else np.nan


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_runs(root):
    runs, hashes = [], set()
    evaluation_reset_seeds = {}
    calibration_path = root / "calibration.json"
    calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest() if calibration_path.exists() else None
    for path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        require(meta["status"] == "completed", f"Incomplete run: {path.parent.name}")
        run = dict(meta, directory=path.parent)
        identity = (meta["task"], meta["arm"], meta["seed"], meta["reward_delay"])
        require(identity[0] in TASKS and identity[1] in ARMS and identity[2] in (0, 1, 2)
                and identity[3] == 1, f"Unexpected pilot condition: {identity}")
        for key, expected in (("total_training_transitions", 50000), ("warmup", 10000),
                              ("critic_updates", 10000), ("actor_updates", 5000),
                              ("evaluation_episodes", 5), ("evaluation_interval", 10000)):
            require(meta[key] == expected, f"{path.parent.name}: invalid {key}")
        hashes.add(json.dumps(meta["source_sha256"], sort_keys=True))
        require(meta.get("calibration_sha256") == calibration_hash,
                f"Calibration artifact/hash mismatch: {path.parent.name}")
        require(meta["threads"] == 1, f"Expected one computational thread: {path.parent.name}")
        rows = csv_rows(path.parent / "evaluation.csv")
        keys = [(int(row["step"]), int(row["evaluation_episode"])) for row in rows]
        require(len(keys) == 30 and len(set(keys)) == 30 and
                set(keys) == {(int(step), ep) for step in GRID for ep in range(5)},
                f"Incomplete or duplicate evaluation grid: {path.parent.name}")
        for row in rows:
            require((row["task"], row["arm"], int(row["seed"]), int(row["delay"])) == identity,
                    f"Evaluation identity mismatch: {path.parent.name}")
            reset_key = (row["task"], int(row["seed"]), int(row["evaluation_episode"]))
            reset_seed = int(row["reset_seed"])
            expected_reset = evaluation_reset_seeds.setdefault(reset_key, reset_seed)
            require(reset_seed == expected_reset,
                    f"Evaluation reset seed differs across checkpoints/arms for {reset_key}: {path.parent.name}")
        by_key = {(int(r["step"]), int(r["evaluation_episode"])): float(r["episode_return"]) for r in rows}
        run["evaluation_matrix"] = np.array([[by_key[int(step), ep] for ep in range(5)] for step in GRID])
        require(np.isfinite(run["evaluation_matrix"]).all(), f"Nonfinite return: {path.parent.name}")
        require(np.array_equal(run["evaluation_matrix"][0], run["evaluation_matrix"][1]),
                f"Learner changed during warmup: {path.parent.name}")
        run["curve"] = run["evaluation_matrix"].mean(axis=1)
        run["raw_auc"] = float(np.sum((run["curve"][:-1] + run["curve"][1:]) / 2 * np.diff(GRID)) / GRID[-1])
        eval_steps = sum(int(row["episode_length"]) for row in rows)
        require(eval_steps == meta["total_evaluation_transitions"] == 30 * meta["horizon"],
                f"Evaluation budget mismatch: {path.parent.name}")
        run["generations"] = csv_rows(path.parent / "generations.csv")
        episodes = csv_rows(path.parent / "episodes.csv")
        collected = sum(int(row["length"]) for row in episodes)
        ea = sum(int(row["length"]) for row in episodes if row["source"] == "ea")
        require(collected == 50000 and ea == (0 if meta["arm"] == "td3" else 40000),
                f"Collection split mismatch: {path.parent.name}")
        for row in episodes:
            require(np.isclose(float(row["native_return"]), float(row["emitted_return"]), rtol=1e-6, atol=1e-5),
                    f"Reward accounting mismatch: {path.parent.name}")
        last = run["generations"][-1]
        run["collected_ea"] = ea / collected
        run["resident_ea"] = int(last["resident_ea"]) / (int(last["resident_ea"]) + int(last["resident_learner"]))
        counts = meta["sampled_transitions"]
        require(sum(counts.values()) == 1280000, f"Replay draw count mismatch: {path.parent.name}")
        run["sampled_ea"] = counts["ea"] / sum(counts.values())
        exact = {"td3": 0, "erl_balanced": .5, "erl_no_ea_replay": 0}.get(meta["arm"])
        require(exact is None or run["sampled_ea"] == exact, f"Sample mix mismatch: {path.parent.name}")
        diagnostics = path.parent / "diagnostics.csv"
        run["diagnostics"] = csv_rows(diagnostics) if diagnostics.exists() else []
        runs.append(run)
    expected = {(task, arm, seed) for task in TASKS for arm in ARMS for seed in range(3)}
    require(len(runs) == 24 and {(r["task"], r["arm"], r["seed"]) for r in runs} == expected,
            "Full report requires exactly 24 distinct completed pilot runs (2 tasks × 4 arms × 3 seeds)")
    require(len(hashes) == 1, "Mixed trainer source hashes: archive invalid runs and rerun consistently")
    return runs


def selected(runs, task, arm):
    return sorted((r for r in runs if r["task"] == task and r["arm"] == arm), key=lambda r: r["seed"])


def scheduling_note(root):
    profile, sweep = root / "profile_execution.json", root / "pilot_execution.json"
    if profile.exists() and sweep.exists():
        p, s = json.loads(profile.read_text()), json.loads(sweep.read_text())
        return (f"The two seed-zero TD3 runs were profiled with {p['workers']} worker(s), then reused in the "
                f"24-run pilot; the remaining runs used {s['workers']} concurrent worker(s). "
                "Each run used one computational thread. Different contention makes per-method wall-time rankings unfair.")
    return ("Each run used one computational thread. Invocation-level scheduling is not fully recorded here; "
            "per-run timing may reflect different contention and should not rank methods by speed.")


def save_figure(fig, root, name, caption, figures):
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.text(.015, .014, caption, ha="left", va="bottom", fontsize=8, wrap=True)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    figures.append(dict(name=name, caption=caption))
    plt.close(fig)


def make_figures(runs, root):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "svg.fonttype": "none"})
    figures = []
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), sharex=True, sharey="row")
    for ti, task in enumerate(TASKS):
        for ai, arm in enumerate(ARMS):
            ax = axes[ti, ai]
            curves = np.array([r["curve"] for r in selected(runs, task, arm)])
            for curve in curves:
                ax.plot(GRID / 1000, curve, color=COLORS[ai], alpha=.35, linewidth=1)
            ax.plot(GRID / 1000, curves.mean(axis=0), "o-", color=COLORS[ai], linewidth=2, markersize=4)
            ax.axvspan(0, 10, color="grey", alpha=.08)
            ax.set_title(f"{task} · {LABELS[ai]}")
            if ai == 0:
                ax.set_ylabel("Native return ↑")
            if ti == 1:
                ax.set_xlabel("Training transitions (thousands)")
    fig.suptitle("Learning trajectories: every seed remains visible")
    save_figure(fig, root, "learning-curves", "Thin lines: 3 training seeds; thick line: seed mean. Each checkpoint averages 5 evaluation episodes. Shading: random-action warmup. No smoothing or confidence bands.", figures)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ti, task in enumerate(TASKS):
        ax, labels = axes[ti], []
        for ai, arm in enumerate(ARMS):
            for run in selected(runs, task, arm):
                y = ai * 4 + run["seed"]
                first, final = run["curve"][[0, -1]]
                ax.plot([first, final], [y, y], color=COLORS[ai], alpha=.65)
                ax.scatter(first, y, facecolors="white", edgecolors=COLORS[ai], zorder=3)
                ax.scatter(final, y, color=COLORS[ai], zorder=3)
                labels.append((y, f"{LABELS[ai]} · seed {run['seed']}"))
        ax.set(yticks=[y for y, _ in labels], yticklabels=[label for _, label in labels], title=task, xlabel="Native evaluation return ↑")
        ax.invert_yaxis()
    fig.suptitle("Initial → final performance of every trained agent")
    save_figure(fig, root, "initial-final", "Open point: untrained policy; filled point: 50,000 training steps. A movement to the right is improvement. Same 5 evaluation resets within each run; 3 independent training seeds per arm.", figures)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        for ai, arm in enumerate(ARMS):
            values = [r["raw_auc"] for r in selected(runs, task, arm)]
            ax.scatter(ai + np.array([-.13, 0, .13]), values, color=COLORS[ai], s=45)
            ax.plot([ai - .25, ai + .25], [np.mean(values)] * 2, color=COLORS[ai], linewidth=2)
        ax.set(xticks=range(4), xticklabels=LABELS, ylabel="Time-averaged native return ↑", title=task)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Area under the learning curve, with all pilot seeds")
    save_figure(fig, root, "raw-auc", "Points: individual training-seed AUC, trapezoidal integral divided by 50,000 steps, including initialization and warmup. Bars: mean of 3 seeds. Raw scales differ by task; these are descriptive results.", figures)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        for j, (key, label, hatch) in enumerate((("collected_ea", "Collected", ""), ("resident_ea", "Resident", "//"), ("sampled_ea", "Sampled", ".."))):
            vals = [np.mean([r[key] for r in selected(runs, task, arm)]) for arm in ARMS]
            ax.bar(np.arange(4) + (j - 1) * .24, vals, width=.22, color=[COLORS[i] for i in range(4)], alpha=.4 + j * .25, hatch=hatch, label=label)
            for ai, arm in enumerate(ARMS):
                ax.scatter(np.full(3, ai + (j - 1) * .24), [r[key] for r in selected(runs, task, arm)], color="black", s=8, zorder=4)
        ax.set(xticks=range(4), xticklabels=LABELS, ylim=(0, 1), title=task, ylabel="Evolutionary-source fraction")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)
    fig.suptitle("What we collect is not necessarily what we learn from")
    save_figure(fig, root, "replay-provenance", "Measured final collected/resident shares and cumulative sampled shares. Bars: mean of 3 seeds; black points: seeds (often overlapping). Warmup source tags identify collector slots, not evolved behavior.", figures)

    available = any(r["diagnostics"] for r in runs)
    if available:
        for name, title, keys, ylabel in (
            ("replay-age", "The age of experience retained in replay", ("age_ea_mean", "age_learner_mean"), "Resident age (training transitions)"),
            ("replay-reuse", "How often stored experience has been sampled", ("reuse_ea_mean", "reuse_learner_mean"), "Mean draws per resident transition"),
            ("population-diversity", "Population behavior on a fixed observation probe", ("action_disagreement",), "Mean normalized action distance"),
        ):
            arms = ARMS[1:] if name == "population-diversity" else ARMS
            fig, axes = plt.subplots(2, len(arms), figsize=(13, 7), sharex=True, sharey="row", squeeze=False)
            for ti, task in enumerate(TASKS):
                for ai, arm in enumerate(arms):
                    ax = axes[ti, ai]
                    for ki, key in enumerate(keys):
                        trajectories = []
                        for run in selected(runs, task, arm):
                            rows = run["diagnostics"]
                            steps = np.array([int(row["step"]) for row in rows])
                            vals = np.array([number(row, key) for row in rows])
                            if len(vals) and np.isfinite(vals).any():
                                ax.plot(steps / 1000, vals, color=("#D55E00", "#0072B2")[ki], alpha=.3)
                                trajectories.append((steps, vals))
                        if trajectories and all(np.array_equal(trajectories[0][0], t[0]) for t in trajectories):
                            data = np.array([t[1] for t in trajectories])
                            count = np.isfinite(data).sum(axis=0)
                            mean = np.divide(np.nansum(data, axis=0), count, out=np.full(data.shape[1], np.nan), where=count > 0)
                            ax.plot(trajectories[0][0] / 1000, mean, "o-", color=("#D55E00", "#0072B2")[ki], label=("Population", "Learner")[ki] if len(keys) == 2 else "Population diversity")
                    ax.set_title(f"{task} · {LABELS[ARMS.index(arm)]}")
                    if ai == 0:
                        ax.set_ylabel(ylabel)
                    if ti == 1:
                        ax.set_xlabel("Training transitions (thousands)")
                    if ax.lines:
                        ax.legend(fontsize=7)
            fig.suptitle(title)
            caption = "Thin lines: training seeds; thick lines: available-seed mean. Diagnostics do not consume training randomness. Missing source measurements are absent, not zero."
            caption += " Mean over all resident transitions by source; the 100,000-slot buffer never fills in this pilot." if name.startswith("replay-") else " Probe coverage is limited; diverse actions need not be useful actions."
            save_figure(fig, root, name, caption, figures)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ti, task in enumerate(TASKS):
        for ci, (key, divisor, label) in enumerate((("elapsed_s", 60, "Per-run wall time (minutes)"), ("peak_rss_mib", 1, "Peak process memory (MiB)"), ("checkpoint_bytes", 2**20, "Retained checkpoints (MiB)"))):
            ax = axes[ti, ci]
            for ai, arm in enumerate(ARMS):
                vals = [r[key] / divisor for r in selected(runs, task, arm)]
                ax.scatter(ai + np.array([-.13, 0, .13]), vals, color=COLORS[ai])
            ax.set(xticks=range(4), xticklabels=LABELS, ylabel=label, title=task)
            ax.tick_params(axis="x", rotation=15)
    fig.suptitle("CPU and storage cost: every pilot run")
    save_figure(fig, root, "compute-cost", "Each point is one run, including evaluation and checkpoint writes. " + scheduling_note(root) + " Wall times overlap; peak RSS is per process.", figures)

    run = selected(runs, TASKS[0], "erl_balanced")[0]
    path = run["directory"] / "lineage.jsonl"
    if path.exists():
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        coords, fig = {}, plt.figure(figsize=(13, 4.5))
        ax = fig.subplots()
        for event in events:
            x, y = event["step"] / 1000, event["slot"]
            parent = coords.get(event["parent_id"])
            if parent:
                ax.plot([parent[0], x], [parent[1], y], color="#777777", alpha=.22, linewidth=.8)
            kind = event["type"]
            color = {"elite": "#0072B2", "mutation": "#999999", "injection": "#D55E00"}[kind]
            ax.scatter(x, y, color=color, marker="*" if kind == "injection" else "o", s=80 if kind == "injection" else 15, zorder=3)
            coords[event["individual_id"]] = (x, y)
        ax.set(xlabel="Training transitions (thousands)", ylabel="Destination population slot", yticks=range(4), title="Pendulum · balanced replay · seed 0: selection, mutation, and learner injection")
        ax.set_ylim(-.15, 3.5)
        ax.legend(handles=[Line2D([0], [0], marker="*" if kind == "injection" else "o", linestyle="", color=color, label=kind.capitalize()) for kind, color in (("elite", "#0072B2"), ("mutation", "#999999"), ("injection", "#D55E00"))], loc="upper left", ncol=3)
        save_figure(fig, root, "lineage", "Illustrative run chosen by task/arm/seed, not performance. Lines connect recorded parents to children; stars are learner copies. Fitness is a training diagnostic; lineage survival alone does not establish causal benefit.", figures)
    return figures


def write_reports(runs, root, figures):
    calibration_path = root / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    calibration = calibration.get("tasks", calibration)
    summaries = []
    for run in runs:
        reference = calibration.get(run["task"], {}).get("random_return_mean")
        require(reference is None or np.isfinite(reference) and reference < 0, "Invalid random-policy normalization anchor")
        summaries.append(dict(run_id=run["run_id"], task=run["task"], arm=run["arm"], seed=run["seed"],
                              initial_return=float(run["curve"][0]), final_return=float(run["curve"][-1]),
                              return_change=float(run["curve"][-1] - run["curve"][0]), raw_auc=run["raw_auc"],
                              normalized_auc=(run["raw_auc"] - reference) / -reference if reference is not None else None,
                              sampled_ea_fraction=run["sampled_ea"], elapsed_s=run["elapsed_s"], peak_rss_mib=run["peak_rss_mib"]))
    groups = []
    for task in TASKS:
        for arm in ARMS:
            group = [r for r in summaries if r["task"] == task and r["arm"] == arm]
            groups.append(dict(task=task, arm=arm, seeds=3, final_mean=float(np.mean([r["final_return"] for r in group])),
                               final_sd=float(np.std([r["final_return"] for r in group], ddof=1)),
                               raw_auc_mean=float(np.mean([r["raw_auc"] for r in group])),
                               elapsed_mean_s=float(np.mean([r["elapsed_s"] for r in group]))))
    summary = dict(scope="Immediate-reward development pilot; descriptive, not a test of H1", runs=summaries, groups=groups,
                   analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   normalization_anchors={task: calibration.get(task, {}).get("random_return_mean") for task in TASKS},
                   verified_runs=len(runs), total_training_transitions=sum(r["total_training_transitions"] for r in runs),
                   total_evaluation_transitions=sum(r["total_evaluation_transitions"] for r in runs),
                   total_critic_updates=sum(r["critic_updates"] for r in runs), total_actor_updates=sum(r["actor_updates"] for r in runs),
                   sum_per_run_wall_seconds=sum(r["elapsed_s"] for r in runs),
                   scheduling_note=scheduling_note(root), figures=figures)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = ["# CPU pilot results", "", "All 24 planned runs completed and passed the analysis budget/grid checks. These are immediate-reward development runs with three training seeds per task/method. They measure correctness, baseline learnability, and CPU cost; **they do not test the delayed-reward hypothesis H1**.", "", "| Task | Method | Final native return, mean ± seed SD | Mean raw AUC | Mean run time |", "|---|---|---:|---:|---:|"]
    for group in groups:
        report.append(f"| {group['task']} | {LABELS[ARMS.index(group['arm'])]} | {group['final_mean']:.2f} ± {group['final_sd']:.2f} | {group['raw_auc_mean']:.2f} | {group['elapsed_mean_s']/60:.2f} min |")
    report += ["", "Raw AUC is the trapezoidal learning-curve integral divided by the 50,000-step budget, so its units are native episode return. Less negative return/AUC is better. Seed SD describes variation among three trained agents; it is not a confidence interval. See [individual runs](run-guide.md) and [machine-readable summaries](summary.json).", "", f"Verified totals: **{summary['total_training_transitions']:,} training transitions**, **{summary['total_evaluation_transitions']:,} evaluation transitions**, **{summary['total_critic_updates']:,} twin-critic updates**, and **{summary['total_actor_updates']:,} actor updates**. Every run has 30 evaluation episodes, an unchanged learner across random-action warmup, finite returns, and the intended replay sample count. All runs share the same recorded trainer source hashes and match the calibration artifact hash.", "", f"Summed per-run wall time is {summary['sum_per_run_wall_seconds']/60:.2f} minutes. Concurrent runs overlap, so this is not suite elapsed time. The two profiling runs are included in the 24-run totals. Calibration, smoke checks, installation, and analysis are outside these run totals. One critic update trains both critics.", "", scheduling_note(root), "", "A complete immediate-reward run cannot establish delayed-reward robustness. These small-seed comparisons must not choose whichever arm happens to win as a new hypothesis. Freeze any justified implementation/settings revisions before using the disjoint confirmatory seeds. The 100,000-slot replay buffer is not filled by a 50,000-step pilot, so separate tests must establish eviction correctness."]
    for figure in figures:
        name = figure["name"]
        report += ["", f"![{name}](figures/{name}.png)", "", figure["caption"], "", f"[SVG](figures/{name}.svg) · [PDF](figures/{name}.pdf)"]
    (root / "report.md").write_text("\n".join(report) + "\n")
    guide = ["# What happened in each pilot run", "", "Each row is a fresh trained agent. Initial and final returns average the same five held-out starting conditions within that run. Positive change means less negative final return, not a statistically established method advantage. A seed labels an independent training repeat.", "", "| Run and journal | Initial return | Final return | Change | Sampled EA share | Minutes |", "|---|---:|---:|---:|---:|---:|"]
    for run in summaries:
        guide.append(f"| [{run['task']} · {LABELS[ARMS.index(run['arm'])]} · seed {run['seed']}]({run['run_id']}/journal.md) | {run['initial_return']:.2f} | {run['final_return']:.2f} | {run['return_change']:+.2f} | {run['sampled_ea_fraction']:.1%} | {run['elapsed_s']/60:.2f} |")
    guide += ["", "Every run spent its first 10,000 transitions collecting random actions, then performed 10,000 twin-critic and 5,000 actor updates. Its evaluation episodes stayed outside replay. During each collection generation, policies were frozen; learning occurred after five complete episodes. The [learning guide](../../docs/learning-guide.md) explains the equations and operators."]
    for run in summaries:
        direction = "higher" if run["return_change"] > 0 else "lower" if run["return_change"] < 0 else "unchanged"
        guide += ["", f"## {run['task']} · {LABELS[ARMS.index(run['arm'])]} · seed {run['seed']}", "", f"**OBSERVE.** Final evaluation was {direction} than initialization by {abs(run['return_change']):.2f} native-return units. The full-trajectory raw AUC was {run['raw_auc']:.2f}; measured runtime was {run['elapsed_s']:.1f} seconds.", "", f"**EXPLAIN.** {EXPLANATIONS[run['arm']]} The observed cumulative evolutionary-source sample fraction was {run['sampled_ea_fraction']:.3%}.", "", "**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.", "", "**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1."]
    (root / "run-guide.md").write_text("\n".join(guide) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    runs = load_runs(args.root)
    figures = make_figures(runs, args.root)
    write_reports(runs, args.root, figures)
    print(f"Verified {len(runs)} complete runs; wrote {len(figures)} figure families and report to {args.root}")


if __name__ == "__main__":
    main()
