# First-stage development validation

Eight small CPU checks use development seeds 10 and 11, both replay arms, and both reward delays. Each collects 16,000 training transitions, retains the 10,000-step warmup, and performs 1,500 twin-critic plus 750 actor updates. They are implementation checks, not a shortened test of H1. No method ranking or inferential interval is produced.

Status: **passed**. Completed 8/8; failures 0. Recorded execution time: 1.30 minutes.

| Run journal | Initial return | Final return |
|---|---:|---:|
| [erl_balanced · delay 1 · seed 10](first_stage_validation__Pendulum-v1__erl_balanced__d001__s010/journal.md) | -1323.78 | -1866.11 |
| [erl_balanced · delay 1 · seed 11](first_stage_validation__Pendulum-v1__erl_balanced__d001__s011/journal.md) | -1402.66 | -1662.74 |
| [erl_balanced · delay 20 · seed 10](first_stage_validation__Pendulum-v1__erl_balanced__d020__s010/journal.md) | -1323.78 | -1320.82 |
| [erl_balanced · delay 20 · seed 11](first_stage_validation__Pendulum-v1__erl_balanced__d020__s011/journal.md) | -1402.66 | -1637.38 |
| [erl_pooled · delay 1 · seed 10](first_stage_validation__Pendulum-v1__erl_pooled__d001__s010/journal.md) | -1323.78 | -1888.71 |
| [erl_pooled · delay 1 · seed 11](first_stage_validation__Pendulum-v1__erl_pooled__d001__s011/journal.md) | -1402.66 | -1757.11 |
| [erl_pooled · delay 20 · seed 10](first_stage_validation__Pendulum-v1__erl_pooled__d020__s010/journal.md) | -1323.78 | -1534.80 |
| [erl_pooled · delay 20 · seed 11](first_stage_validation__Pendulum-v1__erl_pooled__d020__s011/journal.md) | -1402.66 | -1528.09 |

Each return averages two evaluation episodes. The short optimization budget is not intended to establish learnability under delayed feedback. Full independent diagnostics, exact budget checks, and warmup invariance were checked in every completed run. The accompanying test-suite report separately verifies isolation, resume behavior, and the statistical pipeline.
