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

**Current status:** pilot artifacts are being published; next-stage implementation is in progress. No confirmatory outcome is available.

The [lab notebook](lab-notebook.md) contains detailed observations and decisions. Future entries here will link verified commits and result artifacts, and distinguish development checks from scientific results.
