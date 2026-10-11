# G02 — FedEraser + canonical retrain grid (MNIST)

/goal G02 — follow prompts/goal_protocol.md with prompts/goals/G02_federaser_and_retrain_grid.md

- **Window:** W3–W4 (26/10 → 08/11).
- **Entry check:** G01 done; decisions on rounds and seed mode recorded in PROGRESS.md.

**Lane A**
1. FedEraser (Liu et al., IWQoS 2021; local `articles/5.*`): calibration with stored updates, calibration interval and ratio from the config, every step citing the paper section; `# DEVIATION` where the arXiv/IWQoS versions differ.
2. Cost accounting for FedEraser: storage bytes, calibration rounds, communication bytes.
3. Canonical MNIST runs (only if the used functions are ≥ L2): original M + M_retrain^(-A) + second retrain, 4 α × T1–T3 × 3 seeds, one process at a time; verifier after each.

**Lane B**
- Re-check `literature/federaser_2021_six_questions.md` against the IWQoS version (calibration formula, storage cost, datasets, forgetting metric).
- FedQUIT (arXiv 2408.07587): extract the cost-to-recover protocol. Nguyen et al. (PMLR v222): extract their FU evaluation setup.

**Lane C:** critique FedEraser implementation vs paper; critique the retrain/noise-floor table once it exists.

**Done when:** FedEraser smoke-tested on MNIST IID and α=0.1; retrain-vs-retrain table (noise floor) for MNIST if canonical was allowed.

**Tutor queue:** FedEraser change notes; `Dạy mình paper FedEraser theo 3 vòng`; `Backfill: buổi 4.3, 5.1`; number audit of the noise-floor table.
