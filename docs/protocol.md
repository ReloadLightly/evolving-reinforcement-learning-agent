# Experimental protocol: replay provenance under delayed rewards

**Status: CPU pilot complete; confirmatory outcomes remain prospective.** On 12 October 2026, all 24 immediate-reward pilot runs completed with zero failures. The pilot implementation passed 27 unit/integration tests; the completed dataset passed the analysis budget/grid/hash checks and the recorded-lineage audit. This does not test H1. The [README](../README.md) carries the manuscript; the [pilot report](../results/pilot/report.md) contains measured results. The [first-stage specification](first-stage-protocol.md) fixes the next experiment, and its freeze tool records the final executable bundle before reserved seeds run. The scientific contribution remains a controlled comparison on a small task suite, not a claim of a new replay mechanism or state-of-the-art performance.

## Pilot implementation record and declared differences

The pilot uses the algorithm and hyperparameters in §4, immediate reward feedback, development seeds `0, 1, 2`, and 50,000 training transitions per run. Its 10,000-step warmup is followed by 10,000 twin-critic and 5,000 actor updates. Each run has six checkpoints and 30 held-out evaluation episodes. This implements the bounded development stage; it does not freeze or begin the 200,000-step confirmatory experiment.

| Item | Implemented pilot behavior | Interpretation or remaining requirement |
|---|---|---|
| Software and device | Python 3.10.12, PyTorch 2.7.1 on CPU, NumPy 2.2.6, Gymnasium 1.4.0, MuJoCo 3.15.0; one computational thread per run | Exact packages and source/calibration hashes are recorded in each run's metadata; [requirements](../requirements.txt) pins the selected packages. |
| Replay age and reuse | Checkpoints record mean age and cumulative sample-reuse counts over all **resident** transitions, separately by source | This simplifies the sampled-age distributions/quantiles proposed in §5 and §9. A resident mean is not the age distribution of the learner's sampled minibatches. |
| Critic diagnostics | Source-conditioned absolute TD residuals and twin-critic disagreement come from the **last training minibatch** before a checkpoint | This simplifies the independent diagnostic batch proposed in §9. Measurements depend on the arm's sampler, are noisy, and do not measure ground-truth value accuracy. |
| Behavioral probe | A fixed, hashed set of 128 observations per task from independent random-policy calibration | Population disagreement averages pairwise root-mean-square action differences, normalized by action range, over that probe. Coverage is limited to these calibration observations; this makes the pilot's RMS reduction explicit relative to §9's general action-distance description. |
| Pilot inference | Individual seed trajectories and points, means, and descriptive seed standard deviations; **no bootstrap confidence intervals or hypothesis tests** | Three development seeds and immediate rewards do not test the delayed-reward H1. The ten-seed confirmatory analysis in §8 remains prospective. |
| FIFO boundary | A 50,000-step pilot stores fewer records than the 100,000-slot capacity | Eviction and source-index consistency are covered by dedicated replay tests, not exercised by full pilot runs reaching capacity. |
| CPU profiling | Seed-zero TD3 runs on both tasks are profiled individually and reused as two of the 24 pilot runs; the remaining runs use two concurrent workers | All runs use one computational thread. Different contention prevents interpreting per-method wall-time differences as a fair speed ranking. Execution metadata records actual scheduling. |

The independent calibration consists of 20 random-action episodes per task, adding 5,000 interactions outside the pilot's training/evaluation totals. Logs preserve both native and emitted returns; only emitted rewards enter learning. Educational `journal.md` and `events.jsonl` files explain run phases without feeding their measurements back into training. The [learning guide](learning-guide.md) and [lab notebook](lab-notebook.md) explain these artifacts and decisions.

The [analysis script](../scripts/analyze_pilot.py) requires all 24 planned runs, complete evaluation grids, equal budgets, matching trainer/calibration hashes, unchanged evaluation during random-action warmup, and evaluation reset seeds paired across checkpoints and arms. It records its own SHA-256 and exact normalization anchors. The completed 24-run pilot passed those checks; [the lineage audit](../results/pilot/lineage-audit.json) additionally verified recorded evolutionary selection and scheduled learner injections. Any additional changes required before confirmatory execution must be documented and frozen explicitly; the pilot differences above do not silently replace the prospective requirements below.

## 1. Question, hypothesis, and scope

**Research question.** At a fixed environment-interaction budget and gradient-update budget, does balancing evolutionary and learner-generated replay improve the sample efficiency of the reinforcement-learning policy in a small ERL system, and does the answer change when rewards are delivered in blocks?

The core intervention changes only replay sampling within an otherwise shared ERL implementation. An evolutionary population supplies 80% of collected transitions. Ordinary sampling from the pooled buffer therefore tends to put 80% evolutionary transitions in the learner's minibatches. The treatment instead samples 50% from each source, without changing which transitions are retained.

**Primary hypothesis, H1.** With rewards aggregated over 20 steps, balanced replay produces higher normalized native-return area under the learning curve than pooled replay, evaluated on the learner policy over 200,000 total training transitions.

This is a directional, falsifiable prediction, not an expected result entered into a chart. It could fail if the population provides better data than the learner, if repeated learner samples are harmful, if a population of four is insufficient, or if both variants solve these tasks too quickly to distinguish their learning curves.

**Secondary questions.** Is the balanced-minus-pooled effect different between immediate and aggregated rewards? Does allowing evolutionary transitions help compared with blocking them? Does fitness-based selection add value beyond a population that collects data and receives learner injections? These questions diagnose the result; they do not create additional primary hypotheses after seeing the data.

The first confirmatory stage concerns **Pendulum only**. A later, predeclared extension adds Reacher and the two controls. The extension reuses the first-stage runs and is therefore not an independent replication. Neither stage supports claims about locomotion, sparse-goal exploration, quality-diversity archives, large networks, or all ERL algorithms.

The original ERL work motivates exchanging population experience and learner parameters. EvoRainbow demonstrates that hybridization has multiple interacting design choices. Experience filtering and dynamic mixed replay have already been proposed, so this project tests a deliberately small mechanism rather than claiming to invent source-aware replay. [Khadka and Tumer, 2018](https://arxiv.org/abs/1805.07917); [Li et al., 2024](https://proceedings.mlr.press/v235/li24cp.html); [Zheng et al., 2025](https://doi.org/10.1109/TNNLS.2025.3596553).

## 2. Task suite and reward intervention

| Task | Role | Default horizon | Observation / action dimensions | Evaluation reward |
|---|---|---:|---:|---|
| `Pendulum-v1` | First-stage question and implementation check | 200 | 3 / 1 | Unmodified native episode return |
| `Reacher-v5` | Second task in the optional full study | 50 | 10 / 2 | Unmodified native episode return |

Both tasks have continuous actions, short episodes, and nonpositive native rewards. They offer a tractable learning study on a laptop. Pendulum supplies a small interpretable control problem; Reacher adds a robot arm and random target locations. Their theoretical reward upper bound of zero is useful for a fixed normalization, although that bound need not be attainable. Pin exact Gymnasium and MuJoCo versions before freezing the protocol; old environment versions and their published scores are not interchangeable. [Pendulum documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/); [Reacher documentation](https://gymnasium.farama.org/environments/mujoco/reacher/).

There are two training conditions: block length `d = 1` and `d = 20`. The wrapper retains the original observations and actions. It accumulates native rewards, emits zero until the end of a block, then emits their sum. Flush a partial block whenever an episode terminates or truncates. Reset the accumulator at every reset.

```text
on reset:
    pending_reward = 0
    block_steps = 0

on step(action):
    obs, native_reward, terminated, truncated, info = base_env.step(action)
    pending_reward += native_reward
    block_steps += 1
    if block_steps == d or terminated or truncated:
        emitted_reward = pending_reward
        pending_reward = 0
        block_steps = 0
    else:
        emitted_reward = 0
    log native_reward separately
    return obs, emitted_reward, terminated, truncated, info
```

For a completed episode, the sum of emitted rewards must equal the sum of native rewards to numerical tolerance. Evolutionary fitness uses the completed episode return, so this redistribution preserves the fitness of a fixed trajectory. The learner receives only emitted rewards; native reward and reward-component diagnostics must not enter its observations, updates, prioritization, selection, or stopping rules. Evaluation always reports native returns.

**Interpretation limit.** This wrapper is a delayed-feedback intervention, not a pure isolated manipulation of temporal credit assignment. The pending reward depends on unobserved history, so the reward process is generally non-Markovian with respect to the unchanged observation. At the proposed discount factor `gamma = 0.99`, preserving undiscounted episode return does not preserve the discounted objective. Block length also represents different fractions of the two episode horizons: 10% for Pendulum and 40% for Reacher. Analyze each task separately and do not attribute every interaction effect to one mechanism. Delayed-reward RL explicitly recognizes this history dependence. [Han et al., 2021](https://arxiv.org/abs/2106.11854).

Do not silently add the pending reward to the observation: that would change what information is available and define a different experiment. A recurrent learner, discount-compensated reward redistribution, or state augmentation can be a subsequent study, not an unreported repair to this one.

TD targets mask true termination and ordinarily bootstrap through time-limit truncation, using the final observation before reset. Flushing at a truncation is required for reward accounting; it does not turn that truncation into a true termination. Since `d = 20` divides the default Pendulum horizon but not Reacher's, the latter has a final 10-step block. Record this boundary effect as part of the wrapper specification. [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## 3. Experimental arms

| Arm identifier | Data collection after warmup | Replay used for learner updates | Purpose |
|---|---|---|---|
| `td3` | Learner collects all transitions | Uniform over learner replay | Standalone gradient-learning reference |
| `erl_pooled` | Four population episodes and one learner episode per generation | Uniform over the global FIFO buffer | Primary comparator |
| `erl_balanced` | Same ERL schedule | Exactly half the minibatch from each source | Primary treatment |
| `erl_no_ea_replay` | Same ERL schedule; population still evolves and receives injections | Learner-origin samples only | Replay-transfer ablation |

The three ERL arms have the same population size, retention policy, actor/critic architecture, exploration noise, injection schedule, collection budget, and number of gradient updates. Their trajectories will diverge as their learner policies change; matching the algorithm and random-number schedules does not imply they receive an identical realized dataset.

`erl_no_ea_replay` is not pure TD3: it still spends 80% of its interaction budget on the population. It deliberately removes one communication channel while retaining its cost. The standalone TD3 arm spends that same total interaction budget collecting learner data. This distinction is essential when interpreting those two controls.

The following extensions are optional and do not belong to the first-stage claim:

| Extension | Conditions | Interpretation |
|---|---|---|
| `erl_balanced_no_selection` | Balanced replay, `d = 20`, both tasks | Choose survivors and parents without using fitness, retaining mutations, collection, and injections |
| Replay fractions `alpha = 0.25` and `0.75` | `d = 20`, both tasks | Examine the response to evolutionary sampling share beyond the primary comparison |

The no-selection arm still has mutation and learner injection. Calling it “no evolution” would overstate what was removed. It is a fitness-selection ablation. A pure evolutionary algorithm would answer a different question and is not required for the replay-specific primary contrast.

## 4. Implementation used in the pilot; candidate confirmatory settings

These values are implemented for the CPU pilot and remain candidate defaults for the confirmatory experiment. The pilot may identify implementation failures or unsuitable learning settings. Record all changes and their reasons before creating a frozen confirmatory run manifest. Do not choose settings from confirmatory evaluation returns.

| Item | Proposed value / rule |
|---|---|
| Learner | TD3 with two critics, target-policy smoothing, delayed actor updates |
| Actor and each critic | Two 64-unit ReLU hidden layers; tanh actor output scaled to the action bounds |
| Numeric precision | Float32 network and replay arrays; float64 accumulated return statistics |
| Optimizer | Adam, actor and critics learning rate `3e-4` |
| Discount / target update | `gamma = 0.99`, `tau = 0.005` |
| Minibatch size | 128 |
| Replay capacity | 100,000 transitions in one global FIFO store |
| Warmup | First 10,000 total training transitions use uniformly random actions; no gradients |
| Critic update-to-data ratio | 0.25 per total post-warmup transition, including population transitions |
| Actor frequency | One actor update per two twin-critic update steps |
| Learner action noise | Independent Gaussian standard deviation 0.1 in normalized action units, clipped to action bounds |
| Target-policy noise | Gaussian standard deviation 0.2, clipped to `[-0.5, 0.5]` before bounding actions |
| Observation / reward processing | No learned normalization, clipping, reward scaling, or prioritized replay |
| Population | Four actors, one elite retained each generation |
| Parent selection | Tournament size two, sampled with replacement; ties broken with the selection RNG |
| Offspring mutation | Independently select each weight/bias with probability 0.1; add Gaussian noise with standard deviation 0.02 to selected entries |
| Crossover | None |
| Learner injection | Every 5,000 post-warmup total transitions, replace offspring slot 3 with the current learner actor |
| CPU configuration | One computational thread per run; measure throughput before choosing concurrent worker count |

TD3 is a well-defined off-policy baseline, but this compact ERL implementation is an **ERL-style TD3 hybrid**, not a claimed reproduction of every detail of the original ERL paper. The small network and reduced update ratio are compute choices whose adequacy must be checked during the pilot. [Fujimoto et al., 2018](https://proceedings.mlr.press/v80/fujimoto18a.html).

Initialize the learner identically across arms within a task/seed block. Use an explicit, recorded network initialization procedure and separate initialization streams for its two critics and the population. Initialize the four population actors independently. Do not change the learner's optimizer when it is copied into the population. Population actors have no critic or optimizer state.

The mutation scale and mask probability are hypotheses about a usable baseline, not literature-established optima. Log action differences before and after mutation on a fixed probe set during the pilot. Saturated or virtually unchanged actions indicate a reason to revise the proposed operator before the protocol is frozen.

### 4.1. Warmup, collection, and scheduling

During warmup, use the same episode allocation as the later collection phase: four population-source episodes and one learner-source episode per ERL generation. All actions are random, so also tag these transitions `behavior = warmup`; “source” names the collector slot rather than asserting that evolution generated that behavior. Do not select parents using these random-action returns. Do not perform mutation or injection during warmup. The standalone learner uses random actions for the same 10,000 total transitions.

After warmup, freeze all behavior actors during each generation. Collect one complete episode from each of the four population actors and one from the learner. Population actors use their deterministic actions; the learner uses its action noise. The native tasks have no ordinary early termination, so a generation has exactly 1,000 Pendulum transitions or 250 Reacher transitions and the collection split is exactly 80:20.

Preassign the reset seeds to the five rollout slots, using independent seeds for the slots and identical schedules across experimental arms. Each population fitness estimate therefore uses one training episode and is noisy. Re-evaluate the elite every generation; do not compare a stale elite score with fresh offspring scores. Extra fitness episodes are not free and cannot be introduced in one arm without changing its budget.

The standalone TD3 reference also freezes its actor for five episodes, then performs the corresponding update block. It uses the same five-slot environment-seed schedule. This matches update cadence to the ERL implementation, while the reference still collects all five episodes with the learner. Document this cadence when comparing it with implementations that train after every individual environment step.

After collecting a generation, add `0.25 × newly collected post-warmup transitions` to an update-credit accumulator and execute its integer part. Carry the fractional remainder forward. Reacher alternates 62 and 63 update steps per generation; rounding 62.5 down on every generation would produce an avoidable budget mismatch. Actor-update parity also persists across generations.

At 200,000 total training transitions, each arm has made **47,500 twin-critic update steps and 23,750 actor updates**, assuming the frozen warmup and update ratio above. One twin-critic step updates both critics; counting it twice in one report and once in another would obscure compute use. At batch size 128, this is 6,080,000 sampled transitions, with replacement.

### 4.2. Population and learner exchange

```text
initialize learner, target networks, four population actors, global replay
save initial actor; evaluate isolated learner at total_training_steps = 0
collect 10,000 random-action warmup transitions; keep source and behavior tags
save checkpoint; evaluate isolated learner at total_training_steps = 10,000
update_credit = 0
post_warmup_steps = 0

while total_training_steps < 200,000:
    snapshot learner and all population actors for collection
    for each of four population slots:
        collect one full episode; store emitted rewards and source = EA
        fitness[slot] = sum of emitted rewards for that episode
    collect one learner episode with action noise; store source = RL

    update_credit += 0.25 * generation_steps
    perform floor(update_credit) TD3 updates with the arm's replay sampler
    subtract performed updates from update_credit

    next_population[0] = copy current highest-fitness population actor
    for offspring_slot in [1, 2, 3]:
        parent = winner of a size-two tournament in the current population
        next_population[offspring_slot] = masked_gaussian_mutation(copy parent)

    post_warmup_steps += generation_steps
    if post_warmup_steps is divisible by 5,000:
        next_population[3] = copy updated learner actor
    population = next_population

    if a declared evaluation checkpoint was reached:
        save checkpoint; evaluate learner using an isolated evaluator
```

The learner is copied into a fixed nonelite offspring slot. Its test performance does not decide whether injection occurs. For the no-selection ablation, select the retained actor uniformly and choose parents without their fitness. Keep the copied slot, number of mutations, and injection schedule identical. Continue collecting complete fitness episodes even when their scores are unused.

## 5. Replay provenance and retention

Keep **one physical FIFO buffer** with source metadata. On insertion, globally evict the oldest transition once capacity is reached, regardless of source. Secondary index sets may speed sampling, but separate source buffers with independent capacities would change sample age and retention and therefore define a different intervention.

`erl_no_ea_replay` also inserts and retains population transitions in this same store. Its learner-origin samples occupy the same fraction of retained capacity as in the other ERL arms. Deleting evolutionary samples from its store would leave learner samples resident much longer and confound the intended comparison.

The pooled sampler draws uniformly with replacement from all resident slots. Its evolutionary proportion is measured, not forced to equal 0.8 in each batch. The balanced sampler draws exactly 64 transitions from resident EA slots and 64 from resident RL slots, then shuffles the batch. The blocked sampler draws all 128 from resident RL slots. The optional doses use 32 or 96 EA transitions, respectively. Sampling with replacement is allowed within each source.

The scheduler must fill both sources during warmup before gradients begin. Treat an unexpectedly empty source as an implementation error. Silently substituting the other source would change the experiment. Record actual source counts so that startup, FIFO eviction, and warmup behavior can be audited.

For each sampled transition, make it possible to recover collector source, behavior type, collection step, population generation, and episode identifier. Log sampling-age distributions and cumulative sample reuse by source. Global FIFO controls the residence rule, but different sources can still have different age distributions or reuse rates; balanced replay intentionally reuses the scarcer learner data more often. Those are possible mechanisms of its effect, not errors to hide.

## 6. Budget ladder for a laptop

Run the stages sequentially and report measured throughput before scaling. A laptop-compatible implementation does not make a 160-run study inexpensive. The main budget below is in interactions; actual time depends on the processor, simulator, implementation, and worker count. Do not convert it to hours before measuring.

| Stage | Design | Training transitions | Evaluation transitions | Total |
|---|---|---:|---:|---:|
| Implementation checks | Short wrapper/replay/learner smoke runs, both delays | Explicitly capped and logged separately | Logged separately | Not evidence for H1 |
| Immediate-reward pilot | 4 arms × 2 tasks × 3 development seeds × 50,000 steps | 1,200,000 | 90,000 | 1,290,000 |
| Independent normalization | Random-action policy, 20 episodes per task | 0 | 5,000 | 5,000 |
| First confirmatory stage | 2 primary arms × Pendulum × 2 delays × 10 new seeds × 200,000 steps | 8,000,000 | 840,000 | 8,840,000 |
| Additional full-study runs | Remaining 120 runs: Pendulum controls plus all Reacher arms | 24,000,000 | 1,260,000 | 25,260,000 |
| Full confirmatory study, including first stage | 4 arms × 2 tasks × 2 delays × 10 seeds × 200,000 steps | 32,000,000 | 2,100,000 | 34,100,000 |

The full-study row includes the first-stage row; do not add them twice. Pilot plus normalization plus the full confirmatory study costs **35,395,000 interactions**, before smoke checks, diagnostics needing extra rollouts, or optional ablations. Pilot plus normalization plus the first confirmatory stage costs **10,135,000 interactions** on the same basis.

Main evaluation occurs at steps `0, 10,000, ..., 200,000`: 21 checkpoints, five episodes each. This adds 21,000 steps to each Pendulum run and 5,250 to each Reacher run. Pilot evaluation uses `0, 10,000, ..., 50,000`: six checkpoints, five episodes each. The table includes evaluation for every arm, even when an initial policy could theoretically be reused. Evolutionary fitness rollouts are already included in the training counts.

| Optional delayed-reward extension | Runs | Training | Evaluation | Total |
|---|---:|---:|---:|---:|
| No-selection control, two tasks, ten seeds | 20 | 4,000,000 | 262,500 | 4,262,500 |
| Two additional replay fractions, two tasks, ten seeds | 40 | 8,000,000 | 525,000 | 8,525,000 |

Use CPU execution as the baseline and one thread per run. Measure at least one complete 50,000-step pilot run per task, including evaluation and file writes. Record steps per second, update time, peak memory, checkpoint/output bytes, hardware, thread count, and concurrency. Estimate the next stage from those measurements, including observed variability. Try limited parallel workers only if throughput and memory measurements support it. Changing device, thread count, or software midway through confirmatory runs requires a recorded deviation and may compromise exact reproducibility.

The pilot is for correctness, throughput, and basic learnability. It is not a smaller confirmatory study, and its three seeds do not justify choosing whichever primary arm happens to win. If the frozen design is unaffordable, reduce scope before beginning a confirmatory stage and clearly narrow the question. Do not quietly shorten selected runs or drop seeds after observing outcomes.

## 7. Seeds, isolation, and protocol freeze

The pilot uses development seeds `0, 1, 2` and calibration seed namespace `900000`; confirmatory training seeds `100–109` remain reserved for the prospective study. The exact resolved seed table belongs in the frozen manifest. Derive named random streams from the task and training-seed block for network initialization, environment resets, exploration noise, replay sampling, mutation, parent selection, and evaluation. Use a stable hash or explicit seed sequence, not a language runtime's process-dependent hash.

Pair the same training-seed identifiers across all arms and both delays within a task. Use independent task namespaces so that identically numbered seeds in different tasks are not artificially treated as coupled runs. Separate streams ensure that changing replay sampling does not consume mutation or environment randomness. Logical differences in an algorithm will still make trajectories diverge; exact sample equality is neither promised nor required.

For each task/training-seed block, predeclare five evaluation reset seeds and reuse them across arms, delay conditions, and checkpoints. Different training-seed blocks receive different evaluation starts. This creates paired comparisons with controlled evaluation noise. These starts are independent of training, pilot, and calibration starts. Five episodes are a compute compromise; they do not turn a single trained policy into five independent training replications.

Evaluation must use a separate environment and RNG. Disable action noise, gradient updates, replay insertion, population selection, reward-based stopping, and learner injection. Evaluate the **learner actor**, not the best-scoring member selected on evaluation returns. Any champion-population analysis requires training-only selection, a separately labeled endpoint, and extra evaluation budget.

Before running confirmatory seeds, archive:

1. The protocol version, chosen settings, allowed comparisons, seed table, and normalization constants.
2. The code commit or content hash, dependency lockfile, environment IDs and kwargs, simulator versions, and hardware/thread settings.
3. Pilot logs, every tested configuration, the reason for each change, and an explanation of why the implementation can answer the question.
4. A run manifest listing every planned task × delay × arm × seed combination.

At that point the document can be called a frozen protocol. A local timestamped commit is not an independently registered study. If formal preregistration is desired, register the frozen document separately and report that fact accurately.

After freezing, intermediate confirmatory plots may diagnose technical corruption but cannot choose settings, extra seeds, alternate outcomes, or the stopping point. A numerical divergence is a result to report, not an invitation to replace that seed. An infrastructure interruption may resume the same run from an exact checkpoint, with the restart recorded.

## 8. Primary estimand and statistical analysis

### 8.1. Normalize without using confirmatory outcomes

For each task `e`, obtain the mean native return `L_e` from 20 independent random-action episodes before the main study. Fix the theoretical reward upper bound `U_e = 0`. Archive those episode returns and the anchor hash. Both environments penalize deviations and/or action magnitude, making zero an upper bound on native episode return under their declared defaults.

For evaluation episode mean `R`, define the normalized score as:

$$
Z = \frac{R - L_e}{0 - L_e}.
$$

A score of zero denotes the sampled random-policy reference and one denotes the theoretical upper bound, not an empirically established optimum. Scores may be negative. Do not clip them to `[0, 1]`. Reject an invalid nonnegative random anchor as an implementation/calibration issue before freezing rather than silently inserting a small denominator. Report raw-return figures and task-specific effects alongside normalized summaries. The primary interval is conditional on the frozen anchors; their finite-sample uncertainty remains a limitation.

### 8.2. One outcome per trained seed

Let `Z_{e,d,a,s,k}` be the mean normalized return over five evaluation episodes at checkpoint `k`, after `t_k` total training transitions. With `B = 200,000`, compute the trapezoidal normalized area under the learning curve:

$$
A_{e,d,a,s}
= \frac{1}{B}\sum_{k=1}^{20}
\frac{Z_{e,d,a,s,k-1}+Z_{e,d,a,s,k}}{2}
(t_k-t_{k-1}).
$$

Include the zero-step and warmup checkpoints. Do not choose a favorable post-warmup window, best checkpoint, smoothing bandwidth, or final-episode subset after looking at results. Native-return AUC is also computed and reported within each task.

For the first stage, the primary effect is the mean paired difference on Pendulum at `d = 20`:

$$
\widehat{\Delta}_{P}
= \frac{1}{10}\sum_{s=1}^{10}
\left(A_{P,20,balanced,s}-A_{P,20,pooled,s}\right).
$$

For the full two-task extension, report each task's effect and their equally weighted mean:

$$
\widehat{\Delta}_{suite}
= \frac{1}{2}\left(\widehat{\Delta}_{P}+\widehat{\Delta}_{R}\right).
$$

The extension includes the first-stage data. It refines the scope of the evidence and cannot be counted as an independent second confirmation of the Pendulum finding. Treat the two tasks as fixed choices; ten training seeds per task do not establish performance on an unobserved population of environments.

### 8.3. Uncertainty and decision rules

Use 10,000 paired bootstrap resamples of **complete training-seed records** within each task. A resampled record includes both arms, both delays, all checkpoints, and the evaluation-episode observations belonging to that training seed. Compute the 2.5th and 97.5th percentiles of the resulting effect estimates. This preserves the pairing and temporal dependence. Do not bootstrap checkpoints or evaluation episodes as if they were independent trained agents.

Report the point estimate, its 95% interval, all ten paired seed differences, and raw task-specific effects. The prespecified interpretation is:

- A 95% interval entirely above zero supports a positive effect for this task suite, budget, and implementation.
- An interval entirely below zero provides evidence against the predicted direction.
- An interval spanning zero is inconclusive about direction; it does not establish equivalence or absence of an effect.

Before freezing, retain or revise a proposed practical-effect threshold of **0.05 normalized AUC units**. This is a judgment about a useful effect, not a literature-derived constant. Report whether the interval clears that threshold separately from whether it clears zero. Do not change the threshold after seeing confirmatory data. Ten seeds are a resource allocation, not a demonstrated power guarantee; intervals may remain wide.

These choices follow the principle that finite-run uncertainty, implementation sensitivity, hyperparameter selection, and experimenter decisions must be visible. [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264); [Patterson et al., 2024](https://www.jmlr.org/papers/v25/23-0183.html).

### 8.4. Secondary and mechanistic analysis

The delay interaction is a paired difference of differences:

$$
I_e = \Delta_{e,d=20}-\Delta_{e,d=1}.
$$

Resample the same seed records jointly across delays when estimating its interval. Additional outcomes are final return at exactly 200,000 steps, the balanced-minus-blocked replay contrast, the full-study TD3 comparison, and any declared no-selection or dose extension. Keep these labeled secondary. If formal significance tests are added, predeclare their family and adjust for multiple testing; do not treat every diagnostic interval as a new confirmatory claim.

An interquartile mean of final normalized scores and a performance profile may be helpful secondary summaries once the full suite exists. With only one or two tasks, neither substitutes for the task-specific seed plots. A probability-of-improvement estimate must say whether it compares paired seed blocks or two independently sampled runs; those are different quantities.

Population action diversity, replay age, sampling reuse, critic disagreement, and fitness rank are diagnostics. Correlation with return does not show that a diagnostic caused an improvement. The no-selection ablation offers a controlled test of a specific component; an attractive scatter plot does not.

### 8.5. Failures and missing data

Publish the manifest with a status for every run. Keep infrastructure failures, nonfinite optimization, simulator errors, and intentional cancellations distinct. Never replace a failed training seed with a new seed or report only successful runs. Resume an interrupted run from its exact checkpoint when possible. If a code bug invalidates a comparison, archive the invalid runs and rerun the affected comparison consistently under a new protocol/code version.

A missing checkpoint prevents the prespecified AUC calculation for that run. Do not silently interpolate or carry forward its last score in the primary analysis. Report failure counts and any complete-case estimate with its reduced denominator and selection limitation. A study with unexplained differential missingness is not a clean confirmation of H1. Any additional imputation or worst-case sensitivity analysis must be labeled separately.

## 9. Figures and what each should establish

Pilot empirical panels are generated only from completed, validated run artifacts. The following table remains the prospective confirmatory visualization plan; the pilot's simpler diagnostics and descriptive treatment are recorded above. Synthetic curves must never be presented as results, and captions must identify the task, units, sample size, error-band meaning where applicable, and data source.

| Figure | Data and display | Scientific purpose |
|---|---|---|
| Mechanism diagram | Population, learner, global replay, source sampler, injection arrows, budget counters | Define the intervention and information flow |
| Learning curves | Native returns versus total training transitions; task × delay facets; seed traces plus mean and seed-bootstrap bands | Show when differences appear, variability, and plateaus |
| Primary effect plot | Paired seed AUC differences and bootstrap interval; zero and proposed practical-effect lines | Answer H1 without hiding run-level outcomes |
| Delay interaction | Paired difference-of-differences by task, with interval | Test whether the replay effect changes under aggregation |
| Final-return distribution | All seed values, restrained density/box summary, no bars alone | Reveal skew, collapse, and multimodality |
| Replay provenance audit | Collected share, resident share, and actually sampled share over training | Verify that the intended intervention occurred |
| Replay age and reuse | Source-specific empirical CDFs or quantiles at fixed checkpoints | Separate current behavior quality from old-data/reuse explanations |
| Behavioral diversity | Normalized pairwise action distance on a frozen observation probe set; seed summaries over time | Assess behavior differences without equating parameter distance with diversity |
| Critic diagnostics | Twin-critic absolute disagreement and TD residual by source and reward-block phase | Identify instability; neither metric is ground-truth value error |
| Population transfer | Training-only fitness ranks, survival counts, and injected-policy lineage over generations | Make the evolutionary component and learner transfer visible |
| Compute tradeoff | Return/AUC against measured wall time and total interactions, including evaluation | Show the cost of an apparent sample-efficiency gain |
| Optional replay-dose response | Raw task effects at measured sampling fractions, with seed-level uncertainty | Examine whether the main result is specific to a 50:50 batch |

Build the probe set from the independent random-policy calibration trajectories, using a fixed number of uniformly selected observations per task. Freeze and hash it before main training. Normalize action distances by each action range and average across action dimensions. Do not mix different task observation spaces into a single embedding. A fixed random-policy probe has limited coverage; report that limitation rather than claiming to measure all reachable behavioral diversity.

For confirmatory critic plots, sample a bounded diagnostic batch using an independent diagnostic RNG so logging cannot change training samples. The pilot instead records the last training minibatch, as declared above. Native rewards can appear in analysis-only diagnostics, never in the learning target. If estimating actual value calibration with fresh rollout returns is later desired, budget those rollouts separately; ordinary TD residuals and twin-critic disagreement are not that measurement.

Use consistent colors for algorithms across panels, accessible line styles, and clear raw units. In the confirmatory analysis, draw confidence bands from raw seed records; the three-seed pilot displays individual trajectories and means without confidence bands. Any display smoothing must be declared in the caption and must not change the values entering inference. Export standalone SVG/PDF plus a PNG preview and the tabular data needed to reproduce each scientific plot.

## 10. Minimal reproducibility artifacts

The pilot now emits per-run metadata, training/evaluation tables, diagnostics, lineage records, educational logs, and resumable checkpoints. Calibration, source snapshots, and the planned pilot manifest are also recorded. Full-pilot summaries and figures have now been generated after all 24 runs passed analysis checks. The following table states the reproducibility requirements for the eventual confirmatory results paper. The first-stage freeze tool seals these inputs before execution; confirmatory outcomes remain unavailable until the corresponding runs finish:

| Artifact | Required contents |
|---|---|
| Frozen protocol and configuration | Values, stage scope, seed table, comparison plan, date, code/config hashes |
| Run manifest | Planned combinations, lifecycle status, exception category, restart history |
| Environment manifest | Environment kwargs, library/simulator versions, package lock, wrapper order |
| Evaluation table | One row per episode with task, delay, arm, seed, checkpoint, reset seed, native return, length |
| Training/diagnostic table | Total environment steps, fitness steps, update counts, buffer/source statistics, timing, memory, finite-value checks |
| Checkpoints | Learner and targets, critics and targets, optimizers, population, FIFO contents/indices, RNG states, update credit, counters |
| Calibration artifact | Random-policy episodes, anchors, observation probes, seeds, hash |
| Analysis outputs | Script/config, machine-readable summaries and paired effects, figure source data, exported figures |

At each evaluation checkpoint, save an immutable actor-only snapshot and atomically replace the latest full resumable checkpoint. Retain the final full checkpoint for each run; archiving every intermediate replay-bearing snapshot is optional and must be included in the disk budget. Full checkpoints occur at generation boundaries after updates and population exchange, where no rollout is partially completed. Warmup ends on such a boundary for both tasks. Record snapshot type, retention status, and SHA-256 hash in the manifest. Verify resume equivalence during implementation checks. Actor-only snapshots support evaluation, not reproducible training restart. Count any replayed work after an interruption in the actual compute ledger.

Record training and evaluation transitions separately, then also report their sum. Count all population fitness episodes as training interactions. Record gradient-update counts and CPU time rather than assuming equal transition budgets imply equal compute. Additional video episodes, normalization episodes, and diagnostic rollouts belong in the project-wide cost ledger.

## 11. Implementation acceptance checks

Before a scientific run, verify the properties that make the comparison interpretable:

- `d = 1` emits exactly the original rewards; block emission preserves completed native episode totals; termination, truncation, and reset correctly clear pending reward.
- Time-limit transitions use the correct final observation and bootstrap mask; an automatic environment reset cannot replace the stored successor with a reset observation.
- Global FIFO eviction and source indexes remain consistent; blocked replay retains EA occupancy; balanced batches contain exactly 64 samples of each source.
- Environment, replay, mutation, and evaluation RNG streams are isolated; evaluating a checkpoint does not change the next training transition or sampled update.
- Every primary arm spends the same total training transitions and update count; Reacher's fractional update credits are preserved.
- TD3's twin critics, target smoothing, delayed actor update, and soft targets behave as specified; injected policies copy only intended actor parameters.
- A saved generation-boundary checkpoint resumes to the same subsequent state under the same deterministic settings, or any platform nondeterminism is documented.
- Analysis rejects duplicate evaluation rows, incomplete checkpoint grids, mixed config hashes, and seed omissions rather than silently averaging them.

The implemented [test suite](../tests/) passed **27 tests**, covering reward accounting and boundary handling, replay quotas and FIFO eviction, TD3 targets and update timing, population selection/mutation/injection, state independence, equal-budget integration, isolated evaluation, and exact checkpoint-resume equivalence under the tested deterministic CPU settings. This establishes the tested properties, not universal correctness or empirical support for H1. Final manifest, pairing, hash, and evaluation-grid checks are performed separately by the analysis script after all pilot runs complete. New changes or failures require relevant revalidation before confirmatory execution.

## 12. Interpretation and eventual paper

A positive outcome would support a small, specific statement: under the tested implementation, tasks, delay wrapper, and budgets, changing the sampled source mix improved the learner's learning curve. It would not prove that balancing replay is universally best or that more evolutionary diversity caused the gain.

A negative or mixed outcome is still informative. It can show that the population's experience is already useful at its natural share, that balancing mainly oversamples weak learner data, that the delay wrapper overwhelms this learner, or that these compact tasks do not expose the proposed tradeoff. Diagnostics and prespecified controls help distinguish these explanations, but some will require follow-up experiments.

The final README should report the protocol version and deviations, state the measured primary effect with uncertainty, include raw per-task evidence, disclose total compute and failed runs, and separate limitations from conclusions. Validated pilot measurements may be reported as development results; claims about H1 and the confirmatory results must remain explicitly prospective until those experiments exist. This project succeeds by producing a reproducible answer to a bounded question, including an inconclusive or contrary answer.
