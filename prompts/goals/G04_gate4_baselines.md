# G04 — Gate 4: baseline comparison + backdoor probe + M4 report

/goal G04 — follow prompts/goal_protocol.md with prompts/goals/G04_gate4_baselines.md

- **Window:** W7–W8 (23/11 → 06/12). **Gate 4: 06/12.**
- **Entry check:** FedEraser and PGA ≥ L2 (canonical runs need it).

**Lane A**
1. Backdoor probe: target client poisons part of its data with a trigger; ASR measured on M, M_retrain^(-A), M_FU.
2. Canonical comparison on MNIST × {IID, 1.0, 0.5, 0.1} × T1–T3 × 3 seeds: Retrain, FedEraser, PGA (+KD-FU).
3. Tables/figures for RQ1–RQ4 (gap vs noise floor per split and class group, utility, cost) with `<!-- src: -->` on every number.

**Main session:** draft `reports/month4_baseline_reproduction.md` ("Reproduction Study of FU Methods", original plan Deliverable M4) and create `reports/gate4_self_check.md` with questions only (no answers).

**Lane C:** critique the report; draft-check its text.

**Done when:** report drafted, failure modes at α=0.1 listed (input for the proposed method), gate questions ready.

**Tutor queue:** number audit of each table; `Gate 4: ...`; rehearsal "why does FU get harder at α=0.1?".

**Gate rule (original plan):** if no FU baseline is reproduced, do not propose a new algorithm; switch to an applied comparative study.
