# Research progress

The repository is the public record of this project. Publish coherent, verified milestones as work advances, including code, measurements, figures, and explanations. Large trained checkpoints are distributed as release assets; environments and temporary files are not research artifacts.

## 12 October 2026 — CPU pilot complete; next-stage implementation begins

The [CPU pilot](../results/pilot/report.md) completed all 24 runs with zero failures. Every learner improved from initialization. Replay-composition effects were mixed across the two tasks, and the delayed-reward hypothesis remains untested. The [scientific README](../README.md), [learning guide](learning-guide.md), and [individual-run journals](../results/pilot/run-guide.md) explain the evidence.

The next milestone implements the first confirmatory stage: balanced versus pooled replay on Pendulum, delays 1 and 20, and ten disjoint training seeds per condition. Its full budget is 40 runs and 8 million training transitions. Implementation and small development checks precede any confirmatory execution.

Work in this milestone:

- Add independent replay and critic diagnostics without changing training randomness.
- Freeze the exact protocol, configuration, normalization references, source, and resolved seed table before scientific runs.
- Add a bounded runner with progress reports, strict reuse/resume checks, and explicit run statuses.
- Add paired-seed AUC analysis, bootstrap intervals, and delay-interaction figures.
- Validate with development seeds and publish the resulting checks and learning notes.

**Published:** [pilot and learning lab, `3718c43`](https://github.com/ReloadLightly/evolving-reinforcement-learning-agent/commit/3718c43), followed by byte-preserving artifact settings in `5cf9938`.

The [pilot-v1 release](https://github.com/ReloadLightly/evolving-reinforcement-learning-agent/releases/tag/pilot-v1) also contains all 144 actor snapshots, 24 final resumable checkpoints, and 24 console logs. Its 192 archived files were verified against SHA-256 manifests after compression.

**Diagnostics milestone:** independent replay/critic diagnostics and reward-phase logging were implemented and passed eighteen targeted tests. The expanded validation below supersedes that intermediate check count while preserving the completed pilot's original evidence.

## 12 October 2026 — Next-stage implementation validated; scientific runs pending

**Completed:** the unified suite passed **55 tests**, and **eight development runs completed with zero failures**. Those runs cover pooled/balanced replay, delays 1/20, and development seeds 10/11. They used 128,000 training and 28,800 evaluation transitions in total; their execution took 1.30 minutes, excluding the separately recorded test suite. The [validation report](../results/validation/next-stage/report.md), [checks and hashes](../results/validation/next-stage/checks.json), and [test report](../results/validation/next-stage/test-suite.json) preserve the evidence.

Each short run retained the 10,000-step warmup, then trained for only 6,000 additional transitions. Some final returns deteriorated; all remain visible. These checks validate execution, diagnostics, budgets, and replay/evaluation isolation. They do not establish delayed-reward performance or answer H1.

The next-stage implementation now includes:

- Independent replay-age/reuse and critic diagnostics, reward-block phases, and gradient alignment.
- An immutable protocol/source/configuration/calibration/seed snapshot with strict identity checks.
- A bounded runner that retains failures, distinguishes pending/running/completed/interrupted states, and resumes the same seed from its own checkpoint.
- Complete-manifest analysis with 10,000 jointly paired bootstrap resamples, per-seed results, and ten figure families. Missing or invalid runs prevent primary inference.

The [next-stage learning guide](next-stage-guide.md) explains each addition. The [freeze record](../results/first-study/freeze.json) and [execution ledger](../results/first-study/progress.md) are the audit artifacts for the planned study. Inspect its commands without launching agents:

```bash
.venv/bin/python scripts/run_study.py --root results/first-study --workers 2 --dry-run
```

**Current status:** next-stage code and development validation are complete; **all 40 confirmatory runs remain pending**. The full study is a separate compute commitment. The following commands document deliberate execution and subsequent analysis; recording them here does not mean they have run:

```bash
.venv/bin/python scripts/run_study.py --root results/first-study --workers 2
.venv/bin/python scripts/analyze_first.py --root results/first-study --validate-only
.venv/bin/python scripts/analyze_first.py --root results/first-study
```

The 40-run design requires 8 million training plus 840,000 evaluation interactions. Its actual runtime has not been measured. The original [pilot-v1 release](https://github.com/ReloadLightly/evolving-reinforcement-learning-agent/releases/tag/pilot-v1) remains the versioned completed experiment; current commits add validated machinery without rewriting that evidence. Coherent milestones continue to be published as work advances.

The [lab notebook](lab-notebook.md) contains detailed observations and decisions. Future entries here will link verified commits and result artifacts, and distinguish development checks from scientific results.
