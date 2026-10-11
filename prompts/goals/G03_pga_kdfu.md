# G03 — PGA + KD-FU (timeboxed) + cost-to-recover

/goal G03 — follow prompts/goal_protocol.md with prompts/goals/G03_pga_kdfu.md

- **Window:** W5–W6 (09/11 → 22/11).
- **Entry check:** FedEraser change notes ≥ L1.

**Lane A**
1. PGA (Halimi et al., arXiv 2207.05521): gradient ascent on the target client, projection onto the L2 ball around the reference model, then recovery rounds with non-target clients; all hyperparameters in config; never tuned on the test split.
2. Cost-to-recover accounting shared by all methods (bytes and wall-clock until utility reaches the retrain level on the validation split).
3. KD-FU (Wu et al., arXiv 2201.09441): attempt reproduction; **stop after 1 week of effort** if it does not run, and record why in PROGRESS.md.

**Lane B:** KD-FU details (what server-side data it needs, how they evaluate); verify the venue of Halimi et al.

**Lane C:** critique PGA radius/recovery choices and the cost-to-recover definition.

**Done when:** PGA (and KD-FU, or a recorded stop) smoke-tested on MNIST IID and α=0.1.

**Tutor queue:** PGA change notes; `Dạy mình paper arXiv 2207.05521 theo 3 vòng`; L3 modify exercise on the projection radius.
