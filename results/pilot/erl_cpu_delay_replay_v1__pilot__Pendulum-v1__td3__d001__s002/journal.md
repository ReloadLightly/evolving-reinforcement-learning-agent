
### run_start · 0 training transitions

**Observe:** Starting Pendulum-v1, td3, seed 2.

**Explain:** All five rollout slots use the learner. Every replay sample comes from its own experience. A seed labels an independent repeat, not a skill level.

**Check:** One CPU thread; 50,000 training transitions; 10,000 warmup transitions; separate evaluation RNG.

**Next:** Measure the untrained policy, then fill replay with random actions.


### evaluation · 0 training transitions

**Observe:** Mean native return -1360.62 (episode SD 166.86); 1,000 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 10,000-transition checkpoint.


### evaluation · 10,000 training transitions

**Observe:** Mean native return -1360.62 (episode SD 166.86); 2,000 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 10,000-transition checkpoint.


### evaluation · 20,000 training transitions

**Observe:** Mean native return -1411.24 (episode SD 117.62); 3,000 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 2,500 critic updates have trained from stored experience.

**Check:** Finite returns; 0.0% EA samples; 1,250 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 10,000-transition checkpoint.


### evaluation · 30,000 training transitions

**Observe:** Mean native return -221.56 (episode SD 209.77); 4,000 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 5,000 critic updates have trained from stored experience.

**Check:** Finite returns; 0.0% EA samples; 2,500 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 10,000-transition checkpoint.


### evaluation · 40,000 training transitions

**Observe:** Mean native return -101.81 (episode SD 107.90); 5,000 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 7,500 critic updates have trained from stored experience.

**Check:** Finite returns; 0.0% EA samples; 3,750 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 10,000-transition checkpoint.


### evaluation · 50,000 training transitions

**Observe:** Mean native return -72.18 (episode SD 65.64); 6,000 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 10,000 critic updates have trained from stored experience.

**Check:** Finite returns; 0.0% EA samples; 5,000 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Complete the run and audit costs.


### run_complete · 50,000 training transitions

**Observe:** Completed 50,000 training transitions in 75.2 seconds.

**Explain:** This run is one noisy observation. Comparing several seeds helps separate a method's pattern from luck.

**Check:** Budget verified: 10,000 twin-critic steps, 5,000 actor steps; 6,000 separately counted evaluation steps.

**Next:** Compare matched pilot runs and assess learnability and CPU cost; do not test H1 on this pilot.

