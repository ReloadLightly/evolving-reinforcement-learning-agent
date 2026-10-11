# First-stage protocol: does replay balance help under delayed feedback?

**Implementation-stage specification, 12 October 2026.** This document becomes part of an immutable, hashed study bundle when `scripts/freeze_study.py` succeeds. A repository commit and local freeze are an auditable plan, not independent third-party preregistration. No confirmatory outcomes have been observed.

## Question and scope

On Pendulum-v1, with equal training-interaction and gradient-update budgets, does a 50:50 mixture of population and learner replay improve the TD3 learner's normalized return AUC under 20-step reward aggregation relative to uniform pooled replay?

The original [protocol](protocol.md) supplies the motivation, full-study extension, and algorithm. This document fixes the smaller first-stage commitment. The [pilot](../results/pilot/report.md) established implementation feasibility and showed mixed method rankings; it does not contribute training seeds or observations to these estimates.

## Fixed experiment

| Item | Setting |
|---|---|
| Task | Pendulum-v1; native 200-step time limit |
| Methods | `erl_balanced`, `erl_pooled` |
| Reward feedback | Native each step (`d=1`); accumulated blocks of 20 (`d=20`) |
| Independent training seeds | 100–109; paired across both arms and delays |
| Planned runs | 2 methods × 2 delays × 10 seeds = 40 |
| Training per run | 200,000 total environment transitions, including population rollouts |
| Warmup | First 10,000 transitions use random actions; no updates or evolution |
| Post-warmup generation | Four population episodes and one learner episode; actors frozen during collection |
| TD3 | Two 64-unit ReLU hidden layers; batch 128; Adam 0.0003; discount 0.99; target rate 0.005 |
| Update schedule | 0.25 twin-critic updates per collected post-warmup transition; actor every two updates |
| Noise | Learner collection noise 0.1; target-action noise 0.2, clipped to ±0.5; normalized actions clipped to ±1 |
| Evolution | Four actors; one elite; two-candidate tournaments with replacement; no crossover |
| Mutation | Independent mask probability 0.1; Gaussian parameter noise standard deviation 0.02, including biases |
| Learner injection | Fixed nonelite slot 3 every 5,000 post-warmup transitions |
| Replay | One global FIFO of 100,000 transitions; pooled uniform draws or exactly 64 samples per source |
| Evaluation | Deterministic learner at 0, 10k, …, 200k; five fixed held-out resets per training seed |
| Per-run updates | 47,500 twin-critic and 23,750 actor updates; 6,080,000 training replay draws |
| Stage interactions | 8,000,000 training + 840,000 evaluation; normalization references reused |
| Compute | CPU; one computational thread per run; at most two concurrent workers |

All population fitness interactions count toward training. Evaluation never enters replay, mutation selection, policy updates, or stopping decisions. Source-aware sampling changes exposure to the same global storage policy; it does not create separate eviction rules.

The reward wrapper flushes incomplete blocks at both termination and truncation, preserving native undiscounted episode totals. TD3 bootstraps through time limits. Aggregation changes the discounted objective and introduces unobserved reward history. This is an aggregated-feedback intervention, not an isolated causal manipulation of temporal credit assignment.

## Outcomes and decision rules

The primary outcome is the learner's normalized evaluation-return area under its complete learning curve. Use the frozen pilot calibration file, which contains a separately sampled 20-episode random-policy reference. The Pendulum reference is approximately −1,229.2367547471829; the exact serialized value and file hash are authoritative. The upper reference is zero. Normalize without clipping:

$$z_k=(R_k-L)/(0-L), \qquad A=\frac{1}{200000}\sum_{k=1}^{20}\frac{z_{k-1}+z_k}{2}(t_k-t_{k-1}).$$

For each training seed, retain its complete four-condition record. Define paired differences $D_{s,d}=A_{s,\mathrm{balanced},d}-A_{s,\mathrm{pooled},d}$.

- **H1, primary:** the mean of $D_{s,20}$ is positive.
- **H2, secondary:** the mean of $D_{s,20}-D_{s,1}$ is positive.
- **Practical-effect threshold:** 0.05 normalized AUC units, retained from the original proposal.

Use 10,000 paired percentile-bootstrap resamples, with analysis seed **20261012**. Each resample selects ten whole training-seed records with replacement, using the same selected records for both delays and methods. Report the mean effect, 2.5th/97.5th percentiles, and every original seed contrast. Do not resample checkpoints or evaluation episodes as independent training replicates. Intervals are conditional on the frozen normalization reference and this fixed task.

An H1 interval entirely above zero supports the predicted direction in this implementation/task; entirely below zero contradicts it; an interval including zero is inconclusive. Report separately whether its lower endpoint exceeds 0.05. Ten seeds are a compute choice and do not guarantee adequate power. No p-values, optional stopping, or additional primary hypotheses are specified. Raw AUC, final return, H2, and mechanism diagnostics remain separately labeled summaries. H3 requires the later full-study extension and is not tested here.

## Independent diagnostics and explicit refinements after the pilot

Diagnostics observe learning without consuming training randomness or changing replay's training-sample counters, parameters, optimizer state, or gradients. Their dedicated random streams are saved and restored with checkpoints.

1. **Resident replay:** per-source age and reuse quantiles describe all retained records. The 200k experiment crosses the 100k FIFO capacity; these summaries describe survivors, not all historical experience.
2. **Sampled replay:** a bounded 128-record diagnostic draw follows the arm's actual sampler without incrementing training reuse counts. Fixed age bins are 0–1k, 1k–5k, 5k–10k, 10k–25k, 25k–50k, 50k–100k, 100k–200k, and 200k+. These are checkpoint estimates of the sampling distribution, not a census of all training minibatches. Report that distinction in every heatmap caption.
3. **Critic diagnostics:** independent batches of up to 128 draws per source supply TD-residual and twin-critic-disagreement quantiles, overall and by reward-block phase. Targets use emitted rewards only. Small phase groups are explicitly counted; missing groups remain absent. These measurements are not ground-truth value errors.
4. **Gradient agreement:** report the cosine between the two source-conditioned gradients of the combined twin-critic loss, with gradient norms. Positive values indicate local alignment of these sampled update directions; they do not prove beneficial transfer. A zero norm makes the cosine undefined, not zero.
5. **Evolution:** retain population fitness, ancestry, mutation/injection records, and action disagreement on the fixed 128-observation calibration probe. Training fitness remains distinct from held-out learner return.

The pilot used resident means and its last training minibatch for critic diagnostics. These richer observations replace that shortcut before the first-stage freeze. The bounded sampled-age estimate is an explicit clarification of the original visualization plan. Learning hyperparameters and research hypotheses are retained; instrumentation must pass isolation and restart checks before the freeze.

## Integrity, failures, and publication

Freeze the source files, executable analysis, configuration, protocol documents, pinned requirements, calibration/probe files, validation evidence, planned manifest, and resolved evaluation/named-stream seeds before reserved seeds run. The runner must reject a changed source, calibration, or protocol bundle. Preserve lifecycle and attempt records. Completed matching runs may be reused; interrupted runs require explicit checkpoint resume under the same identity.

Keep every planned seed in the public status table. An incomplete or failed study receives a completion report without a primary inferential conclusion; no silent complete-case estimate or replacement seed is allowed. Technical corruption is investigated separately from numerical divergence or poor return. A change that invalidates a comparison requires a newly identified experiment and documented deviation.

Publish coherent verified milestones and progress logs to GitHub. Interim confirmatory outcomes may diagnose technical problems but cannot select hyperparameters, additional seeds, alternate outcomes, or a stopping time. The next implementation milestone uses development seeds **10 and 11**, short budgets, and explicitly labeled validation output. Those checks do not enter the 40-run scientific dataset.
