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

**Diagnostics milestone:** independent replay/critic diagnostics and reward-phase logging are implemented. Eighteen targeted tests passed, including exact training-trajectory equivalence with diagnostics enabled, diagnostic checkpoint restoration, and unchanged training randomness. Eight short development runs are now exercising both methods and both delays with seeds 10 and 11. The [first-stage specification](first-stage-protocol.md) fixes the scientific question and separates these checks from its 40 reserved-seed runs.

**Current status:** the pilot is available on GitHub; next-stage implementation and small development validation are in progress. No confirmatory outcome is available. The full 40-run execution is a separate compute commitment, provisionally around 2–3 hours on the measured CPU; its actual runtime has not been measured.

The [lab notebook](lab-notebook.md) contains detailed observations and decisions. Future entries here will link verified commits and result artifacts, and distinguish development checks from scientific results.
