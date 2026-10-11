
### run_start · 0 training transitions

**Observe:** Starting Pendulum-v1, erl_pooled, seed 11.

**Explain:** Four evolutionary actors collect four episodes; the learner collects one. Pooled replay follows the resulting roughly 80:20 data mix. A seed labels an independent repeat, not a skill level.

**Check:** One CPU thread; 16,000 training transitions; 10,000 warmup transitions; separate evaluation RNG.

**Next:** Measure the untrained policy, then fill replay with random actions.


### evaluation · 0 training transitions

**Observe:** Mean native return -1402.66 (episode SD 91.43); 400 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 2,000 training transitions

**Observe:** Mean native return -1402.66 (episode SD 91.43); 800 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 4,000 training transitions

**Observe:** Mean native return -1402.66 (episode SD 91.43); 1,200 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 6,000 training transitions

**Observe:** Mean native return -1402.66 (episode SD 91.43); 1,600 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 8,000 training transitions

**Observe:** Mean native return -1402.66 (episode SD 91.43); 2,000 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 10,000 training transitions

**Observe:** Mean native return -1402.66 (episode SD 91.43); 2,400 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 12,000 training transitions

**Observe:** Mean native return -1599.68 (episode SD 69.23); 2,800 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 500 critic updates have trained from stored experience.

**Check:** Finite returns; 80.0% EA samples; 250 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 14,000 training transitions

**Observe:** Mean native return -1618.26 (episode SD 84.68); 3,200 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 1,000 critic updates have trained from stored experience.

**Check:** Finite returns; 80.1% EA samples; 500 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 16,000 training transitions

**Observe:** Mean native return -1757.11 (episode SD 103.93); 3,600 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 1,500 critic updates have trained from stored experience.

**Check:** Finite returns; 80.1% EA samples; 750 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Complete the run and audit costs.


### run_complete · 16,000 training transitions

**Observe:** Completed 16,000 training transitions in 13.1 seconds.

**Explain:** This run is one noisy observation. Comparing several seeds helps separate a method's pattern from luck.

**Check:** Budget verified: 1,500 twin-critic steps, 750 actor steps; 3,600 separately counted evaluation steps.

**Next:** Audit diagnostic isolation, delayed feedback, and the complete validation manifest before freezing a study.

