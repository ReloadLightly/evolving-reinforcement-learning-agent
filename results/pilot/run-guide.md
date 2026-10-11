# What happened in each pilot run

Each row is a fresh trained agent. Initial and final returns average the same five held-out starting conditions within that run. Positive change means less negative final return, not a statistically established method advantage. A seed labels an independent training repeat.

| Run and journal | Initial return | Final return | Change | Sampled EA share | Minutes |
|---|---:|---:|---:|---:|---:|
| [Pendulum-v1 · Balanced ERL · seed 0](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_balanced__d001__s000/journal.md) | -1550.52 | -151.95 | +1398.58 | 50.0% | 1.19 |
| [Pendulum-v1 · Balanced ERL · seed 1](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_balanced__d001__s001/journal.md) | -1657.17 | -292.96 | +1364.21 | 50.0% | 1.19 |
| [Pendulum-v1 · Balanced ERL · seed 2](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_balanced__d001__s002/journal.md) | -1360.62 | -70.62 | +1290.00 | 50.0% | 1.16 |
| [Pendulum-v1 · No EA replay · seed 0](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_no_ea_replay__d001__s000/journal.md) | -1550.52 | -146.05 | +1404.47 | 0.0% | 1.12 |
| [Pendulum-v1 · No EA replay · seed 1](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_no_ea_replay__d001__s001/journal.md) | -1657.17 | -274.88 | +1382.29 | 0.0% | 1.19 |
| [Pendulum-v1 · No EA replay · seed 2](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_no_ea_replay__d001__s002/journal.md) | -1360.62 | -73.83 | +1286.78 | 0.0% | 1.19 |
| [Pendulum-v1 · Pooled ERL · seed 0](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_pooled__d001__s000/journal.md) | -1550.52 | -146.99 | +1403.53 | 80.0% | 1.31 |
| [Pendulum-v1 · Pooled ERL · seed 1](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_pooled__d001__s001/journal.md) | -1657.17 | -142.77 | +1514.40 | 80.0% | 1.30 |
| [Pendulum-v1 · Pooled ERL · seed 2](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__erl_pooled__d001__s002/journal.md) | -1360.62 | -71.33 | +1289.28 | 80.0% | 1.34 |
| [Pendulum-v1 · TD3 · seed 0](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__td3__d001__s000/journal.md) | -1550.52 | -150.18 | +1400.34 | 0.0% | 0.83 |
| [Pendulum-v1 · TD3 · seed 1](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__td3__d001__s001/journal.md) | -1657.17 | -126.51 | +1530.66 | 0.0% | 1.31 |
| [Pendulum-v1 · TD3 · seed 2](erl_cpu_delay_replay_v1__pilot__Pendulum-v1__td3__d001__s002/journal.md) | -1360.62 | -72.18 | +1288.44 | 0.0% | 1.25 |
| [Reacher-v5 · Balanced ERL · seed 0](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_balanced__d001__s000/journal.md) | -33.11 | -6.44 | +26.68 | 50.0% | 1.43 |
| [Reacher-v5 · Balanced ERL · seed 1](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_balanced__d001__s001/journal.md) | -21.16 | -9.32 | +11.84 | 50.0% | 1.82 |
| [Reacher-v5 · Balanced ERL · seed 2](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_balanced__d001__s002/journal.md) | -32.78 | -8.74 | +24.04 | 50.0% | 1.82 |
| [Reacher-v5 · No EA replay · seed 0](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_no_ea_replay__d001__s000/journal.md) | -33.11 | -6.09 | +27.03 | 0.0% | 2.02 |
| [Reacher-v5 · No EA replay · seed 1](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_no_ea_replay__d001__s001/journal.md) | -21.16 | -8.78 | +12.39 | 0.0% | 1.97 |
| [Reacher-v5 · No EA replay · seed 2](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_no_ea_replay__d001__s002/journal.md) | -32.78 | -8.57 | +24.21 | 0.0% | 1.96 |
| [Reacher-v5 · Pooled ERL · seed 0](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_pooled__d001__s000/journal.md) | -33.11 | -8.60 | +24.51 | 80.0% | 1.98 |
| [Reacher-v5 · Pooled ERL · seed 1](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_pooled__d001__s001/journal.md) | -21.16 | -8.78 | +12.38 | 79.9% | 2.12 |
| [Reacher-v5 · Pooled ERL · seed 2](erl_cpu_delay_replay_v1__pilot__Reacher-v5__erl_pooled__d001__s002/journal.md) | -32.78 | -9.96 | +22.82 | 80.0% | 2.17 |
| [Reacher-v5 · TD3 · seed 0](erl_cpu_delay_replay_v1__pilot__Reacher-v5__td3__d001__s000/journal.md) | -33.11 | -7.55 | +25.56 | 0.0% | 0.92 |
| [Reacher-v5 · TD3 · seed 1](erl_cpu_delay_replay_v1__pilot__Reacher-v5__td3__d001__s001/journal.md) | -21.16 | -9.33 | +11.83 | 0.0% | 2.18 |
| [Reacher-v5 · TD3 · seed 2](erl_cpu_delay_replay_v1__pilot__Reacher-v5__td3__d001__s002/journal.md) | -32.78 | -8.61 | +24.17 | 0.0% | 2.14 |

Every run spent its first 10,000 transitions collecting random actions, then performed 10,000 twin-critic and 5,000 actor updates. Its evaluation episodes stayed outside replay. During each collection generation, policies were frozen; learning occurred after five complete episodes. The [learning guide](../../docs/learning-guide.md) explains the equations and operators.

## Pendulum-v1 · Balanced ERL · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 1398.58 native-return units. The full-trajectory raw AUC was -867.20; measured runtime was 71.5 seconds.

**EXPLAIN.** The population collects 80% of training data, but each update uses 64 population and 64 learner samples. The observed cumulative evolutionary-source sample fraction was 50.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · Balanced ERL · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 1364.21 native-return units. The full-trajectory raw AUC was -960.47; measured runtime was 71.7 seconds.

**EXPLAIN.** The population collects 80% of training data, but each update uses 64 population and 64 learner samples. The observed cumulative evolutionary-source sample fraction was 50.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · Balanced ERL · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 1290.00 native-return units. The full-trajectory raw AUC was -746.85; measured runtime was 69.5 seconds.

**EXPLAIN.** The population collects 80% of training data, but each update uses 64 population and 64 learner samples. The observed cumulative evolutionary-source sample fraction was 50.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · No EA replay · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 1404.47 native-return units. The full-trajectory raw AUC was -941.14; measured runtime was 67.5 seconds.

**EXPLAIN.** The population still collects, evolves, and receives learner injections; updates exclude its transitions. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · No EA replay · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 1382.29 native-return units. The full-trajectory raw AUC was -1079.87; measured runtime was 71.4 seconds.

**EXPLAIN.** The population still collects, evolves, and receives learner injections; updates exclude its transitions. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · No EA replay · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 1286.78 native-return units. The full-trajectory raw AUC was -752.73; measured runtime was 71.5 seconds.

**EXPLAIN.** The population still collects, evolves, and receives learner injections; updates exclude its transitions. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · Pooled ERL · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 1403.53 native-return units. The full-trajectory raw AUC was -855.09; measured runtime was 78.6 seconds.

**EXPLAIN.** Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix. The observed cumulative evolutionary-source sample fraction was 80.015%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · Pooled ERL · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 1514.40 native-return units. The full-trajectory raw AUC was -946.62; measured runtime was 78.3 seconds.

**EXPLAIN.** Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix. The observed cumulative evolutionary-source sample fraction was 80.003%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · Pooled ERL · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 1289.28 native-return units. The full-trajectory raw AUC was -742.79; measured runtime was 80.2 seconds.

**EXPLAIN.** Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix. The observed cumulative evolutionary-source sample fraction was 80.021%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · TD3 · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 1400.34 native-return units. The full-trajectory raw AUC was -919.38; measured runtime was 49.7 seconds.

**EXPLAIN.** All five collection slots belong to the learner; all replay samples come from learner slots. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · TD3 · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 1530.66 native-return units. The full-trajectory raw AUC was -966.78; measured runtime was 78.4 seconds.

**EXPLAIN.** All five collection slots belong to the learner; all replay samples come from learner slots. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Pendulum-v1 · TD3 · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 1288.44 native-return units. The full-trajectory raw AUC was -762.32; measured runtime was 75.2 seconds.

**EXPLAIN.** All five collection slots belong to the learner; all replay samples come from learner slots. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · Balanced ERL · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 26.68 native-return units. The full-trajectory raw AUC was -14.71; measured runtime was 85.8 seconds.

**EXPLAIN.** The population collects 80% of training data, but each update uses 64 population and 64 learner samples. The observed cumulative evolutionary-source sample fraction was 50.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · Balanced ERL · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 11.84 native-return units. The full-trajectory raw AUC was -12.63; measured runtime was 108.9 seconds.

**EXPLAIN.** The population collects 80% of training data, but each update uses 64 population and 64 learner samples. The observed cumulative evolutionary-source sample fraction was 50.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · Balanced ERL · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 24.04 native-return units. The full-trajectory raw AUC was -17.32; measured runtime was 108.9 seconds.

**EXPLAIN.** The population collects 80% of training data, but each update uses 64 population and 64 learner samples. The observed cumulative evolutionary-source sample fraction was 50.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · No EA replay · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 27.03 native-return units. The full-trajectory raw AUC was -15.34; measured runtime was 121.2 seconds.

**EXPLAIN.** The population still collects, evolves, and receives learner injections; updates exclude its transitions. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · No EA replay · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 12.39 native-return units. The full-trajectory raw AUC was -13.45; measured runtime was 117.9 seconds.

**EXPLAIN.** The population still collects, evolves, and receives learner injections; updates exclude its transitions. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · No EA replay · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 24.21 native-return units. The full-trajectory raw AUC was -16.43; measured runtime was 117.3 seconds.

**EXPLAIN.** The population still collects, evolves, and receives learner injections; updates exclude its transitions. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · Pooled ERL · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 24.51 native-return units. The full-trajectory raw AUC was -16.33; measured runtime was 118.9 seconds.

**EXPLAIN.** Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix. The observed cumulative evolutionary-source sample fraction was 80.015%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · Pooled ERL · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 12.38 native-return units. The full-trajectory raw AUC was -12.72; measured runtime was 127.1 seconds.

**EXPLAIN.** Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix. The observed cumulative evolutionary-source sample fraction was 79.931%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · Pooled ERL · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 22.82 native-return units. The full-trajectory raw AUC was -17.28; measured runtime was 130.0 seconds.

**EXPLAIN.** Four population episodes accompany one learner episode. Uniform sampling follows the resident source mix. The observed cumulative evolutionary-source sample fraction was 79.963%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · TD3 · seed 0

**OBSERVE.** Final evaluation was higher than initialization by 25.56 native-return units. The full-trajectory raw AUC was -16.00; measured runtime was 55.4 seconds.

**EXPLAIN.** All five collection slots belong to the learner; all replay samples come from learner slots. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · TD3 · seed 1

**OBSERVE.** Final evaluation was higher than initialization by 11.83 native-return units. The full-trajectory raw AUC was -12.85; measured runtime was 130.8 seconds.

**EXPLAIN.** All five collection slots belong to the learner; all replay samples come from learner slots. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.

## Reacher-v5 · TD3 · seed 2

**OBSERVE.** Final evaluation was higher than initialization by 24.17 native-return units. The full-trajectory raw AUC was -16.65; measured runtime was 128.7 seconds.

**EXPLAIN.** All five collection slots belong to the learner; all replay samples come from learner slots. The observed cumulative evolutionary-source sample fraction was 0.000%.

**CHECK.** Training/evaluation budgets, warmup invariance, finite returns, sample count, and complete checkpoint grid passed the analysis checks. A finished run alone does not prove every algorithmic component correct.

**NEXT.** Compare this learning curve with the other two development seeds and matched methods on the same task. Retain adverse outcomes; use the future delayed-reward study to address H1.
