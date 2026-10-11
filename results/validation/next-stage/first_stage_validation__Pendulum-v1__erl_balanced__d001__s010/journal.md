
### run_start · 0 training transitions

**Observe:** Starting Pendulum-v1, erl_balanced, seed 10.

**Explain:** The same 80:20 collection mix is sampled 50:50. This changes exposure to data, not how much data is collected. A seed labels an independent repeat, not a skill level.

**Check:** One CPU thread; 16,000 training transitions; 10,000 warmup transitions; separate evaluation RNG.

**Next:** Measure the untrained policy, then fill replay with random actions.


### evaluation · 0 training transitions

**Observe:** Mean native return -1323.78 (episode SD 290.24); 400 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 2,000 training transitions

**Observe:** Mean native return -1323.78 (episode SD 290.24); 800 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 4,000 training transitions

**Observe:** Mean native return -1323.78 (episode SD 290.24); 1,200 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 6,000 training transitions

**Observe:** Mean native return -1323.78 (episode SD 290.24); 1,600 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 8,000 training transitions

**Observe:** Mean native return -1323.78 (episode SD 290.24); 2,000 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 10,000 training transitions

**Observe:** Mean native return -1323.78 (episode SD 290.24); 2,400 separate evaluation interactions.

**Explain:** Warmup fills replay with random actions. No gradient update has occurred, so the deterministic learner should match its initial evaluation.

**Check:** Finite returns; no minibatches yet; 0 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 12,000 training transitions

**Observe:** Mean native return -1693.38 (episode SD 197.76); 2,800 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 500 critic updates have trained from stored experience.

**Check:** Finite returns; 50.0% EA samples; 250 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 14,000 training transitions

**Observe:** Mean native return -1876.56 (episode SD 40.95); 3,200 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 1,000 critic updates have trained from stored experience.

**Check:** Finite returns; 50.0% EA samples; 500 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Continue to the next 2,000-transition checkpoint.


### evaluation · 16,000 training transitions

**Observe:** Mean native return -1866.11 (episode SD 38.77); 3,600 separate evaluation interactions.

**Explain:** The actor is tested without exploration noise on the same held-out starts. A less negative return is better. Evaluation episodes do not enter replay. 1,500 critic updates have trained from stored experience.

**Check:** Finite returns; 50.0% EA samples; 750 actor updates. Episode SD describes evaluation starts, not uncertainty across training seeds.

**Next:** Complete the run and audit costs.


### run_complete · 16,000 training transitions

**Observe:** Completed 16,000 training transitions in 14.2 seconds.

**Explain:** This run is one noisy observation. Comparing several seeds helps separate a method's pattern from luck.

**Check:** Budget verified: 1,500 twin-critic steps, 750 actor steps; 3,600 separately counted evaluation steps.

**Next:** Audit diagnostic isolation, delayed feedback, and the complete validation manifest before freezing a study.

