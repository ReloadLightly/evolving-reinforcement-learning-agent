# Project working agreement

- The user wants to follow work directly on GitHub. Commit and push coherent, verified progress milestones, including code, documentation, raw scientific evidence, and generated figures. Do not defer all publication to the final response.
- Update `docs/progress.md` and the lab notebook when the research stage or evidence changes. Explain observations, methods, checks, and next steps so the project remains a learning experience.
- Keep large trained checkpoints in versioned release assets with checksums; keep virtual environments, caches, credentials, and temporary files out of Git.
- Preserve completed experimental source snapshots and raw evidence. Never overwrite an old experiment with changed training code or silently replace an adverse seed.
- Confirmatory settings and analyses must be frozen before their reserved seeds run. Development validation and pilot outcomes are not confirmatory evidence.
- Respect the CPU/laptop compute budget. Use one computational thread per run and at most two concurrent workers unless the user changes the budget.
- Use normal non-forced pushes. Do not rewrite shared history or publish credentials.
