# Lab notebook

This is the human-readable record of decisions, checks, and observations. Raw event logs and run artifacts remain the evidence for numerical claims. The [learning guide](learning-guide.md) explains the terminology; the [protocol](protocol.md) defines the comparisons.

Each entry uses **OBSERVE → EXPLAIN → CHECK → NEXT**. Observations describe recorded facts, explanations connect those facts to methods, checks name a verified condition or an outstanding test, and next steps state a decision. An outstanding check is not a passed check.

## 2026-10-11 — CPU pilot authorized; implementation begins

**OBSERVE.** The user authorized the CPU pilot and requested educational progress reporting and an informative repository About description. The existing design specifies 24 runs: four arms × two tasks × development seeds 0, 1, and 2. All pilot runs use immediate rewards, 50,000 training transitions, and five evaluation episodes at each of six checkpoints. The repository initially contains the research manuscript, literature review, protocol, configuration, and run planner. At this entry, no pilot result is recorded here.

**EXPLAIN.** This stage establishes implementation correctness, basic learning behavior, and measured CPU cost. Three ERL arms change the source of replay samples while retaining the same evolutionary collection and exchange schedule. Standalone TD3 supplies a practical reference. Development seeds are separate from confirmatory seeds so pilot-driven repairs do not reuse the later test data.

The run budget includes 10,000 random-action warmup transitions with no optimization. The remaining 40,000 transitions earn 10,000 twin-critic updates and 5,000 actor updates. Evaluation is additional work: it measures the current learner without adding those episodes to training replay. “The run is collecting data” and “the policy is being updated” therefore mean different things during warmup.

**CHECK — outstanding at entry creation.** Verify reward accounting, termination versus truncation, replay sampling and retention, independent random streams, equal interaction/update budgets, finite TD3 updates, and checkpoint behavior before interpreting training outcomes. Measure runtime rather than infer that a small task is automatically cheap. The delayed-feedback hypothesis remains outside this immediate-feedback pilot's scope.

**NEXT.** Implement and check the trainer, run a bounded smoke test, profile complete runs for both tasks, then execute the pilot manifest with measured concurrency. Preserve failures and deviations, summarize every run, and update the manuscript from actual artifacts. Publish the About description when the pilot is complete, as requested.

**Decision sources:** [study configuration](../configs/study.json), [protocol §3–7](protocol.md), and the user's authorization in this conversation.

## Event conventions for subsequent entries

At a **run start**, record the run ID, task, method, delay, seed, code/config identity, planned counts, and the comparison this run belongs to. Explain the method once, then refer back to the learning guide.

At a **checkpoint**, record training and evaluation steps separately, critic and actor update counts, elapsed time, mean evaluation return with episode spread, and replay-source diagnostics. Explain the current phase. Check quantities whose intended values are known, such as zero updates at warmup completion and a 50% sampled population share for the balanced arm.

At a **run end**, record status, actual counts, final evaluation, artifact locations, and total cost. Distinguish successful execution from evidence of learning: an implementation may finish cleanly while its policy fails to improve.

At a **failure or deviation**, record the observed symptom, affected runs, evidence, category, repair, and whether artifacts remain valid. Preserve the original failure record. Do not replace an inconvenient seed or silently present a rerun as the original.

At a **pilot synthesis**, report all runs and variability, evaluate feasibility and baseline learning, and decide whether implementation or settings require revision before freezing the confirmatory protocol. Keep exploratory explanations separate from observed measurements. No entry should call the pilot a test of the delayed-reward hypothesis.

## Proposed machine-readable event fields

The implementation can map these concepts to its actual schema. Field names below describe the logging contract; they are not a claim that a particular file has already been emitted.

| Group | Contents | Why it matters |
|---|---|---|
| Identity | Timestamp, event type, run ID, task, arm, delay, seed, code/config hashes | Connect each observation to one executable experiment. |
| Budget | Training steps, evaluation steps, generation, critic updates, actor updates | Audit fairness and separate new experience from replay reuse. |
| Replay | Resident counts by source, sampled counts by source, age summaries | Establish that the intervention occurred and expose likely mechanisms. |
| Performance | Checkpoint, native evaluation returns, mean and episode spread | Separate measured behavior from training loss and population fitness. |
| Compute | Elapsed time, measured throughput, CPU threads, worker concurrency, peak memory | Turn “CPU-sized” into an empirical statement. |
| Integrity | Lifecycle status, finite-value status, exception category, artifact paths | Preserve failures and trace claims to evidence. |
| Teaching | `observe`, `explain`, `check`, `next` messages | Make progress legible without discarding the underlying numbers. |

Keep machine-readable episode and diagnostic records alongside concise console explanations. Avoid per-gradient prose: it increases volume without helping a reader see phase changes or scientific evidence.

## 2026-10-12 — Implementation checks and first profiling run

*Dates in this notebook use the user's Europe/Berlin timezone; machine event timestamps use UTC.*

**OBSERVE.** The trainer, population operators, replay samplers, reward wrapper, calibration tool, and bounded pilot runner now exist. The implementation test suite passed 27 checks, including training-budget integration, isolated evaluation, and checkpoint-resume equivalence. Independent random-action calibration recorded 20 episodes per task: mean return −1,229.24 for Pendulum and −43.10 for Reacher, using 5,000 additional interactions in total. These are normalization references, not learned-agent scores. The seed-zero TD3 Pendulum profiling run has started; no completed profile time is asserted in this entry.

**EXPLAIN.** Profiling is built into the pilot: the seed-zero standalone TD3 runs on each task count toward the planned 24, so they will not be retrained merely to fill the manifest. Each uses one computational thread. Complete-run timing includes evaluation and checkpoint writes, giving a more useful estimate than a fast isolated network benchmark. Calibration also supplies a fixed 128-observation behavior probe per task without drawing observations from the methods being compared.

**CHECK.** Replay tests cover eviction even though a 50,000-step pilot cannot fill the 100,000-slot production buffer. The analysis script rejects incomplete manifests, duplicate or missing evaluation rows, mixed trainer source hashes, wrong budgets, warmup policy changes, and incorrect exact sampling fractions. Its complete-pilot report still requires all 24 runs. Actual checkpoint diagnostics use resident-buffer mean age/reuse and the last training minibatch's critic residuals; the latter are a documented pilot simplification, not independent value-accuracy measurements.

**NEXT.** Finish the two full profiling runs, use their measurements to bound worker concurrency, and complete the remaining manifest. Then generate the descriptive report, individual-run guide, and figures from the recorded artifacts. Keep H1 untested until the separate delayed-reward experiment.

**Evidence:** [implementation checks](../tests/), [trainer](../erl/train.py), [pilot runner](../scripts/run_pilot.py), [calibration](../results/pilot/calibration.json), and [analysis script](../scripts/analyze_pilot.py).

## 2026-10-12 — Pilot completed, audited, and interpreted

**OBSERVE.** All 24 planned runs completed without failures or seed replacements. Every final learner return improved from its own initialization. Verified totals are 1.2 million training transitions, 90,000 evaluation transitions, 240,000 twin-critic updates, and 120,000 actor updates. Independent calibration adds 5,000 interactions; implementation smoke checks remain outside these scientific-run totals. The two execution invocations took 20.73 minutes combined, excluding installation, calibration, analysis, and the pause between invocations. The two individually profiled TD3 runs are included among the 24, not extra experiments. Peak measured process memory was 680.7 MiB.

**EXPLAIN.** Learning is visible on both tasks, but the method comparison is mixed. Balanced minus pooled raw AUC is −10.01 on Pendulum and +0.56 on Reacher. The highest mean final return and highest mean AUC belong to different methods on each task. AUC rewards improvement earlier in training; a final score answers what the learner achieved at the fixed endpoint. Three development seeds do not establish a general ranking.

The replay audit makes the mechanism concrete: balanced replay ends with a resident mean reuse of 64 for learner records and 16 for population records, compared with approximately 25.6 for each source under pooled replay. These are counts of repeated draws from stored experience, not new environment interactions. They verify the intervention without proving its benefit. One fixed illustrative run—Pendulum, balanced replay, seed 0—improved from −1,550.52 at initialization to −1,458.07 at 20k, −325.43 at 30k, and −151.95 at 50k. The large intermediate improvement is an observation about that trajectory, not evidence assigning credit to mutation or injection.

**CHECK.** All 27 implementation tests passed before the sweep. Analysis verified all 24 identities, budgets, 720 evaluation-episode rows, paired resets, finite returns, warmup invariance, source/calibration hashes, and sampling fractions. The separate lineage audit verified 1,800 evolutionary generations and 144 learner injections. Every run's underlying CSV/JSONL and journal remain available. The nine figure families use actual measurements and export PNG/SVG/PDF. Figure review corrected shared diagnostic axes and prevented the lineage legend from hiding early injection markers. Per-run wall times reflect different contention and are not a fair speed ranking. A 50k run does not reach the 100k FIFO capacity; dedicated tests cover eviction.

**NEXT.** Keep the bounded research question and hypotheses unchanged. Before the delayed-reward study, implement its richer prespecified diagnostics, freeze settings and software, and use the disjoint confirmatory seeds. No confirmatory runs were launched as part of this pilot. Read the [individual-run guide](../results/pilot/run-guide.md), use the [offline results explorer](../results/pilot/explorer.html), and try the [learning exercises](learning-guide.md) to connect the evidence with the equations.

**Repository About.** The [GitHub repository](https://github.com/ReloadLightly/evolving-reinforcement-learning-agent) now describes the CPU ERL lab, replay-composition question, reproducible experiments, scientific figures, and guided explanations. Topics identify reinforcement learning, evolutionary algorithms, TD3, and reproducible research. Code, documentation, and results remain in the local workspace pending normal version-control publication.

**Evidence:** [complete report](../results/pilot/report.md), [numeric summary](../results/pilot/summary.json), [lineage audit](../results/pilot/lineage-audit.json), [execution record](../results/pilot/pilot_execution.json), and [environment manifest](../results/pilot/software-environment.json).
