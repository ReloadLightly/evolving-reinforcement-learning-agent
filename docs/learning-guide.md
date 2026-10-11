# Learning alongside the agent

This guide explains what the CPU pilot does, how to read its logs, and why the experiment is designed this way. The [README](../README.md) is the scientific manuscript; the [protocol](protocol.md) is the specification; the [lab notebook](lab-notebook.md) records decisions and observations. This guide connects the machinery to the ideas.

**The pilot question is: does this implementation behave correctly, learn useful behavior, and fit our CPU budget?** Our later scientific question is whether balancing replay improves learning under delayed rewards. The pilot uses immediate rewards, development seeds, and shorter runs, so it cannot answer that later question.

## 1. What one run means

A **run** trains a fresh agent under one combination of task, method, reward delay, and random seed. It starts from new networks, a new population where applicable, and empty replay. It does not inherit a successful agent from the preceding run.

For example, `Pendulum-v1 / erl_balanced / delay 1 / seed 0` means:

- The task is to control a pendulum with continuous torque.
- The method combines four evolving policies with one TD3 learner.
- Rewards arrive each step, because the delay is one.
- Each learning minibatch contains equal numbers of population-source and learner-source transitions.
- Seed zero identifies one reproducible stochastic realization. It does not identify a difficulty level or a random policy that stays fixed throughout training.

The pilot repeats all four methods on two tasks with three development seeds: **4 × 2 × 3 = 24 runs**. Each run collects 50,000 training transitions, including 10,000 warmup transitions. Its six evaluation checkpoints occur at 0, 10,000, 20,000, 30,000, 40,000, and 50,000 training steps, with five evaluation episodes each.

| Method | What happens in this run | What comparison teaches us |
|---|---|---|
| `td3` | The learner collects all five rollout slots and learns from its own replay. | Can the gradient learner learn this task at this budget? Is hybridization useful enough to pay for its extra collection? |
| `erl_pooled` | Four population actors and the learner each collect one episode; updates draw uniformly from all stored transitions. | What happens with the hybrid's natural, population-heavy replay composition? |
| `erl_balanced` | Collection and evolution follow the same schedule; each batch draws 64 population and 64 learner transitions. | Does changing which experience the learner sees alter learning? |
| `erl_no_ea_replay` | The population still collects, evolves, and receives learner copies, but updates sample learner-source transitions only. | What changes when we block the population-to-learner replay channel while retaining its collection cost? |

The last method is an **ablation**: we remove one component to study its contribution. It is not standalone TD3, because it still spends four fifths of its training interactions collecting population episodes.

## 2. The basic reinforcement-learning loop

An **environment** is a simulator that accepts an action and returns an observation, reward, and episode-ending flags. The **policy** is the rule that chooses actions. Here the policy is a small neural network, called the **actor**.

At time step $t$:

$$
a_t=\pi_\theta(o_t),\qquad
(o_{t+1},r_t)=\operatorname{environment.step}(a_t).
$$

$o_t$ is an observation, $a_t$ an action, $r_t$ a reward, and $\theta$ the actor's weights and biases. The observation is what the agent can see; it need not contain every hidden simulator variable. A **transition** is the recorded one-step experience $(o_t,a_t,r_t,o_{t+1},\text{ending flags})$. A **trajectory** is a sequence of these transitions. An **episode** is a trajectory between reset and termination or a time limit.

For Pendulum, the observation encodes angle and angular velocity; the action is torque. Reacher has a two-joint arm and a target, with two action dimensions. The networks have two hidden layers of 64 units. The actor's final `tanh` bounds its output, which is scaled to the environment's action range.

The native episode **return** is the sum of rewards:

$$
R=\sum_{t=0}^{T-1}r_t.
$$

Both tasks use penalties, so less negative return is better. Moving from −1,200 to −600 is an improvement; moving from −600 to −1,200 is a deterioration. Zero is a theoretical upper bound, not a score we assume is attainable.

A reward says how desirable a particular outcome was. It is not a label telling the network which action it should have taken. Learning must connect actions to their later consequences.

## 3. Why we need a critic

The **critic** $Q_\phi(o,a)$ estimates the discounted future reward after taking action $a$ and then following a policy. Its parameters are $\phi$; they are separate from the actor's parameters $\theta$.

$$
Q^\pi(o_t,a_t)
\approx \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid o_t,a_t,\pi\right].
$$

The **discount factor** $\gamma=0.99$ gives a reward one step later 99% of the weight of an immediate reward, two steps later $0.99^2$, and so on. The critic's discounted prediction and the evaluation episode's undiscounted return are different quantities.

The Bellman idea is that the value of an action can be decomposed into **the immediate reward plus the discounted value of what comes next**. Instead of waiting for every future consequence, we use another value estimate for the future. This is **bootstrapping**.

That makes learning efficient, but it creates a feedback loop: the network learns from targets that partly depend on network predictions. A biased or unstable estimate can therefore propagate. TD3 adds several mechanisms to make this process more reliable.

## 4. TD3, one update at a time

**TD3** means *Twin Delayed Deep Deterministic Policy Gradient*. Its actor chooses a definite continuous action for a given observation. “Deterministic” describes the actor, not the entire stochastic training process.

### 4.1 Twin critics and a conservative target

We train two critics, $Q_{\phi_1}$ and $Q_{\phi_2}$. Separate **target networks**, indicated by primes, provide slowly changing predictions for the next observation. The update target is:

$$
\begin{aligned}
\epsilon &\sim \mathcal{N}(0,0.2^2), \\
\widetilde a' &= \operatorname{bound}\left(\pi_{\theta'}(o')+
\operatorname{clip}(\epsilon,-0.5,0.5)\right), \\
y &= r+\gamma m\min\left(Q_{\phi'_1}(o',\widetilde a'),
Q_{\phi'_2}(o',\widetilde a')\right).
\end{aligned}
$$

Noise and clipping here are expressed in normalized action units; the implementation must apply the appropriate action scaling. $m$ is the **bootstrap mask**: zero at a true terminal state and one otherwise. Taking the minimum of two critics reduces the chance of building the target around one critic's optimistic error. It can also introduce underestimation; two critics do not guarantee correct values.

Adding small noise to the target action is **target-policy smoothing**. It discourages the learner from exploiting an implausibly sharp peak in the critic at one exact action. This differs from **exploration noise**, which perturbs the learner's actions during actual collection. Target smoothing changes a learning target; exploration noise changes what the agent experiences.

### 4.2 A numerical example

Suppose a replay transition has reward $r=-1$, $\gamma=0.99$, no true termination, and target-critic predictions −4 and −5. Then:

$$
y=-1+0.99\min(-4,-5)=-5.95.
$$

If the current first critic predicts −4, its residual, using prediction minus target, is $-4-(-5.95)=1.95$. Its squared error for this example is $1.95^2=3.8025$. Gradient descent changes its parameters to reduce that error. If this were a true terminal transition, $m=0$ and the target would instead be just −1.

A time limit is normally a **truncation**, not a true terminal state: we stopped observing because the evaluation or training episode reached its cap. We still bootstrap from the final observation before reset. Accidentally using the next episode's reset observation would connect two unrelated trajectories.

Each critic minimizes its average squared error across a minibatch:

$$
L_{Q_i}=\frac{1}{B}\sum_{j=1}^B
\left(Q_{\phi_i}(o_j,a_j)-y_j\right)^2.
$$

A low critic loss says predictions agree with these bootstrapped targets. It does not prove that the predictions agree with actual future returns.

### 4.3 Updating the actor

Every second twin-critic update, we update the actor to favor actions that the first critic values highly:

$$
L_\pi=-\frac{1}{B}\sum_{j=1}^B Q_{\phi_1}(o_j,\pi_\theta(o_j)).
$$

Minimizing the negative value amounts to increasing the predicted value. A **gradient** describes how a small change in a parameter changes this objective. Automatic differentiation follows the actor-through-critic computation; we do not differentiate through the simulator. Adam converts these gradients into parameter updates using running estimates of their scale and direction.

The delay gives the critics more opportunities to adjust before the actor changes again. The targets then move a little toward the current networks:

$$
\theta'\leftarrow(1-\tau)\theta'+\tau\theta,
\qquad \tau=0.005,
$$

with the corresponding operation for critic targets. This **soft update** retains 99.5% of the old target parameters and mixes in 0.5% of the current parameters. It is a parameter average, not an average of rewards.

## 5. Replay: the intervention we actually care about

**Experience replay** stores old transitions and reuses them for updates. A **minibatch** is a sampled set of 128 transitions. **Off-policy** learning means that transitions may have been collected by behavior different from the policy currently being improved: an earlier learner, noisy learner actions, random warmup actions, or an evolutionary actor.

One global **FIFO** (*first in, first out*) buffer stores up to 100,000 transitions. When full, a new insertion evicts the oldest transition regardless of source. All ERL arms have the same storage rule. The 50,000-step pilot does not fill this buffer, so actual FIFO eviction must be checked separately; successful pilot training alone cannot validate that boundary.

Three shares answer different questions:

| Share | Question |
|---|---|
| Collected share | Who generated the interactions? |
| Resident share | Which sources occupy replay right now? |
| Sampled share | Which sources actually contributed to gradient updates? |

With four population episodes and one learner episode of equal length, 80% of ERL collection belongs to population slots. Suppose the buffer contains 40,000 population and 10,000 learner transitions. Uniform pooled replay gives an expected **102.4 population / 25.6 learner** samples per 128-item batch. These are averages across batches, not fractional records in an individual batch.

Balanced replay instead draws **64 population / 64 learner** samples. The population still collected 80% of the data. We only changed the exposure during learning. Relative to pooled replay, each learner transition is sampled **2.5 times as often on average** ($64/25.6$), and each population transition 0.625 times as often ($64/102.4$). A learner transition is four times as likely as a population transition to be sampled in the balanced arm because learner data is four times scarcer.

This is both an opportunity and a tradeoff. Learner experience may be more relevant to the current actor, but oversampling it may repeat a small set of weak behaviors. Population data may broaden exploration, but may also describe actions the learner will rarely take. The hypothesis must be tested; neither intuition decides the outcome.

Sampling is with replacement, so a transition can appear multiple times. **Replay age** is how many training interactions have occurred since collection. **Reuse** is how often a record has been sampled. These are different: a recent learner transition can be reused heavily, while an old population transition can be sampled rarely.

During warmup all actions are uniformly random. Source tags name collector slots then; a record tagged population-source is not evidence of an evolved behavior. The separate warmup behavior tag prevents that misunderstanding.

## 6. How evolution learns without gradients

The population contains four actors. Each actor is a candidate solution, and its neural-network parameters are its **genotype**. The resulting behavior is its **phenotype**. Two nearby parameter vectors need not behave similarly, which is why action disagreement on a fixed set of observations is useful alongside parameter measurements.

A **generation** evaluates the current population, learns from collected experience, and creates the next population:

1. Each population actor collects one episode with its own deterministic actions. Its total episode reward is its **fitness**. Each receives a different training reset, so fitness is noisy and partly reflects episode conditions.
2. The learner collects one episode with exploration noise. Population and learner actors are frozen during this collection block.
3. The TD3 learner updates using the chosen replay sampler.
4. **Elitism** copies the current highest-fitness population actor unchanged into the next generation. Its performance is evaluated again next generation; retaining an actor does not guarantee its next score improves.
5. For each remaining slot, **tournament selection** compares two randomly drawn population candidates and picks the one with higher current fitness. Sampling is with replacement, so the two draws can name the same candidate.
6. **Mutation** independently selects each parameter with probability 0.1 and adds Gaussian noise of standard deviation 0.02 to selected parameters. This is parameter-space exploration; it differs from perturbing actions at individual time steps.
7. Every 5,000 post-warmup training transitions, **injection** replaces a fixed nonelite offspring slot with a copy of the updated learner actor.

There is no crossover in this implementation. The evolutionary actors have no critics or optimizers. Their direct improvement comes from selection and mutation, plus periodic learner injection.

ERL therefore has two directions of transfer: **population → learner through experience**, and **learner → population through actor parameters**. Balanced replay changes the first channel. The no-EA-replay arm blocks that channel while keeping the second. A surviving injected lineage shows parameter transfer occurred, but does not by itself prove that transfer caused better learner performance.

## 7. What happens during the 50,000 steps

| Phase | What the software does | What to learn from the log |
|---|---|---|
| Initialization, step 0 | Creates fresh models, replay, random streams, and population; evaluates the untrained learner. | This is the starting policy, not a random-action reference. An untrained network can prefer nonuniform actions. |
| Warmup, steps 1–10,000 | Collects uniformly random actions; no gradient updates, mutation, selection, or injection. | Buffer growth is progress even if policy performance stays unchanged. |
| Learning, steps 10,001–50,000 | Alternates whole collection generations, TD3 updates, and population replacement. | Check data and update budgets, replay shares, finite losses, and measured evaluation returns. |
| Checkpoints, every 10,000 steps | Evaluates the isolated learner for five episodes and records artifacts. | These episodes measure behavior; they do not teach the learner. |
| Completion | Records final counters, runtime, status, and retained artifacts. | A complete run is one experimental replicate, even though it contains many episodes. |

For Pendulum, five 200-step episodes make 1,000 steps per generation. A post-warmup generation earns $1,000\times0.25=250$ twin-critic updates. For Reacher, five 50-step episodes make 250 steps, earning 62.5 updates. A carried fractional credit alternates 62 and 63, avoiding a systematic rounding error. The standalone TD3 method uses the same five-episode collection cadence.

The accounting per complete pilot run is:

$$
\begin{aligned}
N_{\text{critic steps}}&=(50,000-10,000)\times0.25=10,000,\\
N_{\text{actor steps}}&=10,000/2=5,000,\\
N_{\text{sample draws}}&=10,000\times128=1,280,000.
\end{aligned}
$$

One twin-critic step updates **both** critics. The sample-draw count is reuse of stored data, not 1.28 million new environment interactions. Across 24 runs, the planned training count is 1.2 million transitions, with 90,000 additional evaluation transitions. Independent random-policy calibration and implementation checks have separate costs.

## 8. Reading an educational progress log

Use a short four-part rhythm at run starts, evaluation checkpoints, failures, and run ends:

| Label | Meaning | Example of the kind of statement |
|---|---|---|
| **OBSERVE** | What the program measured or which event occurred. | “Warmup ended at 10,000 training transitions; critic updates = 0.” |
| **EXPLAIN** | The relevant concept, in ordinary language. | “Warmup fills replay before the critics start fitting targets.” |
| **CHECK** | A concrete invariant or interpretive limit. | “Both replay sources are present; no policy updates should have occurred yet.” |
| **NEXT** | The next operation and why it matters. | “Start collection/update generations and inspect the next held-out evaluation.” |

Those lines are an illustrative template, not a report of a completed run. The durable [lab notebook](lab-notebook.md) uses the same structure. Machine-readable event records should carry the actual counters and artifact paths; prose should interpret them without replacing them.

The implemented trainer writes a `journal.md` and `events.jsonl` inside each run directory under `results/pilot/`. The journal provides the teaching narrative; the event file preserves its counters. `evaluation.csv` records one row per held-out episode, `episodes.csv` records collection episodes, `generations.csv` records learning and population statistics, and `lineage.jsonl` records parent–child relationships. `diagnostics.csv` contains checkpoint measurements. `metadata.json` identifies software, source hashes, budgets, runtime, and lifecycle status. The generated `results/pilot/run-guide.md` compares every completed run and links to its journal.

In this pilot, replay-age and reuse diagnostics summarize **all resident records**, separately by source. They are not the ages of an independently drawn diagnostic minibatch. Source-conditioned TD residuals and twin-critic disagreement come from the **last training minibatch** before each checkpoint. This keeps the pilot inexpensive, but makes those critic measurements noisy and dependent on each arm's actual sampler. They are not independent estimates of true value-prediction error. Population action disagreement uses a fixed 128-observation probe from independent random-policy calibration.

Before each run, identify its task, arm, delay, seed, budget, and role in the comparison. During it, explain the first occurrence of warmup completion, learning, evaluation, and injection. Thereafter summarize at checkpoints so the important changes stay readable. After it, report starting and final returns, the learning trajectory, actual compute, failures, and what remains uncertain. A table containing every run helps prevent the most exciting run from becoming the whole story.

Distinguish three kinds of statement: **observation** (“sampled EA share was 50%”), **inference** (“learner-data reuse may explain the difference”), and **decision** (“retain this setting for the next stage because the checks passed”). A plausible explanation is not a measured cause.

## 9. Measuring learning without teaching to the test

During collection, learner action noise encourages exploration. During evaluation, the focal learner takes deterministic actions, using separate environment seeds. Evaluation transitions never enter replay, fitness selection, or gradient updates. Separate random streams ensure that adding an evaluation does not change the next training action, mutation, or sampled minibatch.

Evaluation answers: **how does this learner act now, without collection noise, on these held-out resets?** Population fitness answers a different question about the population's recent training episodes. Their returns can differ without either being wrong.

Five evaluation episodes at a checkpoint average over initial-condition variability. They are not five independently trained agents. The three training seeds per pilot condition provide three training replicates. Likewise, six checkpoints from one run describe one learning history; they are not six independent confirmations.

**Sample efficiency** means performance achieved per environment interaction. **Compute efficiency** means performance achieved per measured computing cost. Equal training interactions and equal update counts do not guarantee equal runtime: simulation, evolution, diagnostics, and artifact writes also cost time.

The later study uses **area under the learning curve** (AUC) to measure performance throughout training, not just the final checkpoint. With evenly spaced checkpoints it is a trapezoidal average of consecutive returns. A policy that learns early can have higher AUC than one reaching the same final score much later.

Normalization uses a separately measured random-action return $L<0$:

$$
z=\frac{R-L}{0-L}.
$$

If $L=-1,000$ and a policy scores −500, its normalized score is 0.5. This is halfway between that random reference and the theoretical zero bound; it is not “50% solved.” Scores below zero are allowed. These numbers illustrate the formula and are not pilot measurements.

## 10. Why immediate rewards come first

The pilot uses native rewards every step. This checks basic learnability before introducing the delayed-feedback intervention. With delay 20, the later study will emit zero on most steps and the accumulated reward at the end of each block, flushing any remainder at an episode ending.

For example, native rewards $(-1,-2,-3)$ aggregated in blocks of three become $(0,0,-6)$. The undiscounted total stays −6. With $\gamma=0.99$, the original discounted total is −5.9203, while the aggregated total is −5.8806. They are different objectives. Moreover, the next emitted reward depends on an unobserved accumulated total. Keeping the observation unchanged introduces history dependence.

This makes the manipulation scientifically interesting but limits the interpretation. A future effect could reflect altered credit assignment, discounting, hidden reward history, or their combination. Successful immediate-reward pilot runs do not validate the delayed-reward hypothesis.

## 11. Reading the plots critically

| Visualization | First question to ask | Temptation to avoid |
|---|---|---|
| Learning curves with individual seed traces | Do improvements repeat across fresh training runs, and when do they occur? | Treating one best run or a smoothed curve as the typical result. |
| Replay-source audit | Did the intended sampled mixture actually occur? | Confusing population collection share with sampled share. |
| Replay-age distribution | Does the learner see older experience from one source? | Assuming equal storage rules produce identical age or reuse distributions. |
| Population fitness and diversity | Are candidates finding better or different behaviors? | Equating parameter distance or noisy training fitness with useful exploration. |
| Critic losses, disagreement, and residuals | Are learning signals finite and behaving plausibly? | Calling small residuals proof of an accurate value function. |
| Runtime versus return | What performance did our measured CPU cost buy? | Equating fewer environment steps with lower wall-clock cost. |
| Paired differences across seeds | How consistent is the method contrast? | Treating a shared seed as an identical realized dataset. |

Pilot summaries are descriptive. Three seeds and immediate rewards cannot establish H1, and a small apparent winner is not permission to tune the confirmatory protocol around that winner. Before the main study, freeze settings and use the disjoint confirmatory seeds. Inconclusive, negative, or mixed outcomes are still answers worth documenting.

## 12. Three small investigations to do yourself

Open the [measured-results explorer](../results/pilot/explorer.html) locally and keep the [math lab](learning-lab.html) beside it. Predict an answer before moving a control or opening a journal.

1. **Follow one learning history.** Choose Pendulum, balanced replay, seed 0. Why are the evaluations at 0 and 10,000 steps identical? At 30,000 steps, how many critic and actor updates have occurred? Compare your arithmetic with that run's journal, then inspect whether its final score is also its best observed score.
2. **Follow one stored transition.** Suppose the buffer contains four population records for every learner record. Under balanced sampling, how much more likely is a particular learner record to be drawn than a particular population record? Use the replay-reuse figure to examine the consequence across an entire run.
3. **Count independent evidence.** One task/method has three trained agents, each evaluated five times at six checkpoints. Does this give 3, 15, or 90 independent training replicates? What would go wrong if every evaluation episode became an independent point in a method-comparison confidence interval?

<details>
<summary>Check your reasoning</summary>

1. The actor is unchanged during random-action warmup, and evaluation resets are repeated. By 30,000 transitions it has earned `(30,000 − 10,000) × 0.25 = 5,000` twin-critic updates and `2,500` actor updates. A best checkpoint and a final checkpoint answer different questions; selecting the best after inspection can exaggerate performance.
2. Four times as likely: equal total source probability is spread over four times as many population records. This is an expected per-draw ratio, not a promise that every record is sampled exactly four times as often. Pooled replay gives every resident record equal draw probability.
3. Three. Episodes and checkpoints share a trained policy and learning history. Treating them as independently trained agents understates uncertainty; this is called pseudoreplication.

</details>

## Further reading

The original [ERL paper](https://arxiv.org/abs/1805.07917) explains the two-way exchange between population search and RL. The [TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a.html) motivates the twin critics, delayed actor updates, and target-policy smoothing. Our [literature review](literature-review.md) places these ideas among more recent hybrids; the [protocol](protocol.md) records the exact choices and their limitations for this experiment.
