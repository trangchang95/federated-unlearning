# G01 — Hoàn tất Week 13–14: gold standard sẵn sàng cho canonical

/goal G01 — follow prompts/goal_protocol.md with prompts/goals/G01_week13-14_finish.md

- **Window:** plan v4 W1–W2 (12/10 → 25/10). Gate 3 xác nhận trước 25/10.
- **Entry check:** worktree clean (commit Week 13–14 lane output first).

**Lane A — experiment-engineer**
1. Target-client rule T1/T2/T3 (plan §6) computed from the partition and written into per-target configs; tests on a toy partition.
2. Original-run history: the run that trains M must save what later baselines need — per-round global state and per-client updates (FedEraser), the last-round global state (PGA reference). Record storage bytes. One change note.
3. Replace `Ke_hoach_v3_FU_AI_tutor.md` references in docstrings/notes with `Ke_hoach_v4_FU.md` §6.
4. Smoke-time MNIST: t_train per round for the 10-client SimpleMLP; report the projected cost of the canonical retrain grid (4 α × 3 targets × 3 seeds).
5. Week 13 centralized retrain: smoke only (reference, not the gold standard).

**Lane B — lit-scout**
- Verify from primary sources: GDFA (CVPR 2026), arXiv 2510.13606, FedCCCU (2510.07022), Lethe (2608.01094). For each: one gap-table row against the *proposed Non-IID-aware method* (plan v4 §4), not against the evaluation protocol.

**Lane C — reviewer (critique)**
- Gold-standard design before canonical runs: `number_of_rounds = 5` (enough for convergence and for FedEraser calibration?), noise-floor seed mode, CPU-only device, target rule.

**Done when:** all of the above smoke-tested; canonical configs complete except for the open decisions; no canonical run yet unless FedAvg, Dirichlet, federated retrain and forgetting metrics are ≥ L2.

**Tutor queue:** `Backfill: buổi 2.2.` · `Backfill: buổi 3.2.` · `Backfill: buổi 4.2.` · `Walkthrough notes/changes/2026-10-10_federated_retrain.md.` · `..._forgetting_metrics.md` · `..._noise_floor.md` · `Gate 3: hỏi từng câu, chấm bài mình viết trong reports/gate3_self_check.md.`

**Decisions needed from me:** number of rounds · noise-floor seed mode (`training` vs `training_and_init`) — ask the supervisor.
