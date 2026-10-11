# Progress Log

Running record of what's been done on this thesis, in plain language.
Read [`../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md`](../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md)
for the full 6-month plan this log tracks against. Each entry below is
meant to be understandable on its own, without needing to remember
earlier entries.

---

Older entries: `PROGRESS_archive.md`.

## 2026-09-30 — Gate 3 closed (PASS after follow-ups)

### What was decided

The student's answers in `reports/gate3_self_check.md` were reviewed five
times by the `gate-examiner` subagent. The final result is **PASS after
follow-ups**. The result was recorded at the student's request, and a human
supervisor may re-examine it.

- **Solid (7 of 8):** statistical vs. systems heterogeneity, client drift,
  the cross-entropy effect on absent classes, FedProx's proximal term,
  FedProx's negative result scoped to this setup, a variable-epoch
  experiment design, and communication cost.
- **Partial (1):** Q4, on why the α-vs-accuracy drop is sharp. Its
  explanation is plausible but doesn't yet account for the sharpness.

The final Q5 answer was rewritten by the student in their own words, after
the earlier Q5 text was accidentally deleted.

### Open follow-ups (non-blocking)

1. **Q4:** explain why accuracy drops 0.76 pp from α=1.0 to 0.5 but
   18.23 pp from 0.5 to 0.1. Work it out from the per-client class
   histograms, including digits 1, 4 and 9 at α=0.1. This is recommended
   before the first unlearning experiment on a Non-IID partition.
2. **Q8 (optional):** write out S = 407,080 bytes.

**Gate 3 status:** CLOSED

### What comes next

**Month 4, Week 13: SISA-style delete-and-retrain.** Train on the full
dataset, delete a subset, retrain, and measure accuracy and time. The
student writes the core logic. The measurement code from the previous entry
(`evaluation/efficiency.py`, `evaluation/three_way.py`) is ready to use.

### Known issue, not caused by this change

`reports/verify_week8_fedavg.py`, and therefore
`reports/verify_gate2_readiness.py`, fails on `main` too. The Mac's OS
version changed from macOS 26.5.1 to 26.6.2, and the verifier compares the
saved runtime snapshot exactly. Python and every package version still
match. Either re-run Week 8 on the current OS or relax the platform check;
that decision is open.

---

## 2026-10-06 — Month 4, Week 13 preparation; run not yet performed

### Prepared

- Added a config-driven MNIST full-retraining baseline that trains on the
  complete training split, removes one deterministic client shard, and
  retrains from the same initial weights without that shard. Epoch monitoring
  uses the held-out validation split; the test set is reserved for final
  evaluation.
- The runner saves metrics, model tensor schema, and both checkpoints. The
  verifier reconstructs the partition and recomputes evaluation metrics from
  the saved checkpoints; no result values are recorded before training.
- Static preflight passed: the three Python files compile and the JSON config
  parses. The training run and report verifier have not been executed.
- This is the exact-retraining reference. It does not implement SISA's
  shard/slice checkpointing or a Federated Unlearning method.

### Blocked on student

- **What:** explicitly authorize a commit of the intended Gate 3 closure and
  Week 13 preparation files, then create the configured `month4-week13` tag.
- **Why:** repository rules reserve commits for the student's explicit
  `commit`, and experiments/verifiers require a clean worktree and the
  producing Git tag at `HEAD`.
- **Files:** `PROGRESS.md`, `README.md`, `reports/gate3_self_check.md`,
  `configs/month4_week13_mnist_full_retraining.json`,
  `experiments/unlearning/month4_week13_mnist_full_retraining.py`,
  `reports/build_month4_week13_report.py`, and
  `reports/verify_month4_week13_retraining.py`. The untracked `.codex/`
  agent definitions were left untouched.
- **What to say next:** `commit`.

No experiment or verifier has been run, so Week 13 results remain pending.

---

## 2026-10-10 — Month 4, Week 13–14 (plan W3–W4): preflight only; lanes not started

> **Note added 2026-10-11 — plan v3 is OBSOLETE.** This entry was written
> under `Ke_hoach_v3_FU_AI_tutor.md`; "plan W3–W4" and the file list below
> refer to that version. v3 was replaced by `Ke_hoach_v4_FU.md` on
> 2026-10-11 and removed from the repo root (archived, marked OBSOLETE, in
> `../_old_agent_config_2026-10-10/`). v3 changed the thesis title and
> goals; its title, goals and contribution do not describe this thesis. The
> fixed goals are those of the original plan,
> `Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md`.

### What happened

- Preflight for the milestone "federated retrain gold standard
  M_retrain^(-A) + noise floor + forgetting metrics" was run by the main
  session. It stopped at the first check: the worktree is dirty, and the
  milestone rule is to stop there with one batched commit request.
- Nothing was built, nothing was run, and no lane (experiment-engineer,
  lit-scout, reviewer) was started. This entry is the only file change.
- `notes/changes/` contains no change notes, so none are at L0.
- Checked what is waiting to be committed: the test suite passes (122 tests,
  `~/.venvs/federated-unlearning-mse-ai/bin/python -m unittest discover -s tests -p 'test_*.py'`),
  the three Week 13 Python files compile, and the Week 13 config parses. The
  Week 13 centralized retraining run and its verifier have still not been
  executed.

### Draft plan for the milestone (to confirm before the lanes start)

- **Lane A — experiment-engineer.** Three changes, each one concept, under
  150 lines, with toy-case tests and a change note:
  1. `exclude_clients` federated retrain runner, config and verifier, giving
     M_retrain^(-A) from the same initial weights without the target client;
  2. `evaluation/forgetting.py`: accuracy gap on the three splits,
     prediction agreement, shared-class vs target-dominant classes;
  3. noise floor: the gap between two retrains with different seeds.
  Done means: smoke test passes and t_train is measured. Three notes at L0
  is the WIP limit, so Lane A does not run ahead into PGA.
- **Lane B — lit-scout.** Halimi (PGA): verify, six_questions, setup
  extraction, 3 tutor questions, matrix row. Re-check the existing
  `literature/federaser_2021_six_questions.md` against the same checklist.
- **Lane C — reviewer.** Critique of the retrain and noise-floor design, then
  of the smoke results; Fatal/Major findings become fix tasks.

### Known blockers inside the milestone

- **Canonical runs are not allowed yet.** Every code row in
  `notes/understanding_map.md` is `?`, including FedAvg aggregation and the
  Dirichlet partition, which the federated retrain uses. Until those reach
  L2 the milestone ends at smoke tests, without the retrain-vs-retrain table.
- **Lane B limit.** `literature/` has 13 six_questions files and
  `notes/learning_log.md` has no tutor session logged, against a limit of 2
  papers without a session. Read literally, Lane B may not add Halimi.

## Blocked on student

1. **Commit request (one batch).** Say `commit` (or `commit, ignore .codex`).
   - Modified: `PROGRESS.md`, `README.md`, `reports/gate3_self_check.md`
     (your Q5 answer and the recorded review result; not edited by any agent).
   - New: `Ke_hoach_v3_FU_AI_tutor.md`, `docs/huong_dan_MCP_doc_paper.md`,
     `notes/understanding_map.md`, `notes/curriculum_from_start.md`,
     `notes/learning_log.md`, `notes/changes/README.md`,
     `notes/tutor/README.md`, `prompts/goal_milestone.md`,
     `prompts/tutor_session.md`,
     `configs/month4_week13_mnist_full_retraining.json`,
     `experiments/unlearning/month4_week13_mnist_full_retraining.py`,
     `reports/build_month4_week13_report.py`,
     `reports/verify_month4_week13_retraining.py`.
   - To decide: `.codex/agents/` (4 `.toml` files). Recommended: add `.codex`
     to `.gitignore`, as is already done for `.claude`, `CLAUDE.md` and
     `AGENTS.md`; otherwise they are committed with the rest.
2. **Tutor queue** (paste into `claude --agent tutor`; these unblock the
   canonical retrain runs and do not depend on the commit):
   1. FedAvg aggregation (needs L2 for canonical runs, L3 for the defense):
      `Backfill: buổi 2.2.`
   2. Dirichlet partition (same requirement): `Backfill: buổi 3.2.`
   3. Three-way evaluation, utility and efficiency metrics (L2):
      `Backfill: buổi 4.2.`
3. **Questions waiting for your answer** (no gate questions are open):
   1. Confirm or change the draft plan above.
   2. Lane B limit: count only papers added from now on (Halimi goes ahead),
      or keep Lane B idle until the existing papers have tutor sessions?

---

## 2026-10-11 — Month 4, Week 13–14: federated retrain, forgetting metrics and noise floor built; smoke only

Supersedes the `## Blocked on student` list of the 2026-10-10 entry. The
plan changed from v3 to `Ke_hoach_v4_FU.md` while the lanes were running;
the work below was checked against v4 §6 and its docstrings point to v4.
The new Lane A/B tasks listed in `prompts/goals/G01_week13-14_finish.md`
(target rule T1/T2/T3, saved history for FedEraser, GDFA/FedCCCU/Lethe)
were **not** started in this sitting.

### What was built (Lane A, experiment-engineer)

Three changes, each with a change note at **L0**:

1. **Federated retrain M_retrain^(-A)** —
   `notes/changes/2026-10-10_federated_retrain.md`.
   `algorithms/federated_retrain.py` filters the target client out before
   the unmodified FedAvg server is built, so n_k/n renormalises over the
   non-target clients. Client IDs and the initial weights are kept. Runner:
   `experiments/unlearning/month4_week14_mnist_federated_retrain.py`.
2. **Forgetting metrics** —
   `notes/changes/2026-10-10_forgetting_metrics.md`.
   `evaluation/forgetting.py`: accuracy gap and prediction agreement against
   M_retrain^(-A) on global test, non-target clients and target client, with
   the target split divided into shared-class and target-dominant classes.
   No parameter distances.
3. **Noise floor and verifier** — `notes/changes/2026-10-10_noise_floor.md`.
   k retrains per cell, all pairwise gaps stored, summary per row; runner
   `experiments/unlearning/month4_week14_mnist_noise_floor.py`; verifier
   `reports/verify_month4_week14_federated_retrain.py`.

Tests: `tests/test_federated_retrain.py`, `tests/test_forgetting.py`,
`tests/test_noise_floor.py`. The suite passes with 167 tests (122 before),
run by the main session with
`~/.venvs/federated-unlearning-mse-ai/bin/python -m unittest discover -s tests -p 'test_*.py'`.

The existing FedAvg code (`algorithms/fedavg.py`,
`server/fedavg_server.py`, `clients/federated_client.py`) was not modified.

Size: the changes are larger than the "< 150 lines" guideline. After the
fix round the two runners are 362 and 232 lines and the verifier 298.

### What ran (smoke only; not thesis evidence)

2 rounds, 5,000 training and 1,000 test examples, 10 clients, target
client 0 (arbitrary), 3 retrains, CPU. Run on a dirty worktree, as the
smoke exception in `CLAUDE.md` §4 allows; the result JSON records this.

| Config | Result directory | Verifier |
|---|---|---|
| `configs/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain.json` | `results/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain/` | PASS |
| `configs/smoke_month4_week14_v2_mnist_iid_federated_retrain.json` | `results/smoke_month4_week14_v2_mnist_iid_federated_retrain/` | PASS |

The main session re-ran the verifier on both. Its PASS line reads: "PASS
(consistency check, nothing retrained): config/seed/device/runtime/commit/
code hash recorded; partition, target exclusion, initial weights, accuracy,
class groups, pairwise counts, summary and flags match the checkpoints."

- **PASS is a consistency check.** It shows the JSON matches the saved
  checkpoints, recounted independently of `evaluation/forgetting.py`. It
  does not show the checkpoints came from the recorded training.
- **t_train (smoke, CPU):** α=0.1: M 0.47 s, retrains 0.31 / 0.34 / 0.32 s;
  IID: M 0.43 s, retrains 0.32 / 0.35 / 0.31 s.
  <!-- src: results/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain/metrics.json, noise_floor.json; results/smoke_month4_week14_v2_mnist_iid_federated_retrain/metrics.json, noise_floor.json -->
  These say nothing about canonical cost.
- **Many rows are uninformative in smoke.** The verifier marks a row
  UNINFORMATIVE when the original model M, which never unlearned, is itself
  within the floor. That holds for every accuracy-gap row under IID and for
  four of five agreement rows at α=0.1.
  <!-- src: results/smoke_month4_week14_v2_mnist_*_federated_retrain/noise_floor.json -->
  The models are undertrained after 2 rounds, so this is a property of the
  smoke setting; whether it persists in canonical runs is unknown.
- **The first smoke runs are superseded.** `results/smoke_month4_week14_mnist_*`
  and their two configs come from the single-pair design and are refused by
  the current scripts. They were left in place.
- **No canonical run.** No code row in `notes/understanding_map.md` is at
  L2, and the canonical configs carry null for the open decisions below, so
  the runners refuse to start.

### Literature (Lane B, lit-scout)

- New: `literature/halimi_2022_six_questions.md` (PGA), with setup
  extraction, 3 tutor questions and a tutor key; one row and one Sources
  entry in `literature/literature_matrix.md`. Source: arXiv 2207.05521 v3
  via the arXiv MCP; verified as an arXiv preprint, no peer-reviewed version
  found (DBLP not reached).
- Checked: `literature/federaser_2021_six_questions.md` against arXiv v3.
  Nothing corrected; a scout check, 3 tutor questions and a list of places
  where the paper's own numbers disagree were added.
- Neither paper reports seeds, variance or a second retrain, so neither has
  a noise floor (stated in each note with section references).

### Reviewer findings and status (Lane C)

| Review | Findings | Status |
|---|---|---|
| Draft-check of the literature notes | 0 Fatal, 7 Major, 18 Minor: wording that outran the evidence, unlabelled scout inferences, one use of "forgotten" without a retrain comparison | All 25 fixed by Lane B; not re-reviewed |
| Critique of retrain / metrics / noise floor | Retrain: no target leak found. Noise floor: 1 Fatal (single pair, verdicts decided by exact ties), 5 Major (float ties, no positive control, runtime not checked, client-ID test gap, verifier sharing code with the runner), 10 Minor | Fatal, Major and 7 Minor fixed by Lane A in the three existing changes; D9 (runner wiring tests) and D12 (half-written result directories) not done; `notes/understanding_map.md` rows (D15) left to the tutor; fixes not re-reviewed |
| Critique: design decisions S1–S10 | Belong to the student and supervisor | Open; listed below |

## Blocked on student

1. **Commit request (one batch).** Say `commit`.
   - Lane A: `algorithms/federated_retrain.py`, `evaluation/forgetting.py`,
     `experiments/unlearning/month4_week14_mnist_federated_retrain.py`,
     `experiments/unlearning/month4_week14_mnist_noise_floor.py`,
     `reports/verify_month4_week14_federated_retrain.py`,
     `tests/test_federated_retrain.py`, `tests/test_forgetting.py`,
     `tests/test_noise_floor.py`, the six `configs/*month4_week14*` files,
     and the three notes `notes/changes/2026-10-10_*.md`.
   - Lane B: `literature/halimi_2022_six_questions.md`,
     `literature/federaser_2021_six_questions.md`,
     `literature/literature_matrix.md`.
   - Main session: `PROGRESS.md`.
   - Changed outside this goal (yours, the tutor's or another session's; not
     reviewed here): `Ke_hoach_v4_FU.md`, deletion of
     `Ke_hoach_v3_FU_AI_tutor.md`, `PROGRESS_archive.md`,
     `prompts/goal_milestone.md`, `prompts/goal_protocol.md`,
     `prompts/goals/` (12 files), `notes/understanding_map.md`,
     `notes/learning_log.md`, `notes/curriculum_from_start.md`,
     `.codex/agents/` (4 files). Say if any of these should stay out.
2. **Tutor queue** (paste into `claude --agent tutor`):
   1. `Backfill: buổi 2.2.` — FedAvg aggregation; needed at L2 before any
      canonical retrain.
   2. `Backfill: buổi 3.2.` — Dirichlet partition; same.
   3. `Backfill: buổi 4.2.` — three-way evaluation, utility, efficiency.
   4. `Walkthrough notes/changes/2026-10-10_federated_retrain.md. Bắt đầu từ "why".`
   5. `Walkthrough notes/changes/2026-10-10_forgetting_metrics.md. Bắt đầu từ "why".`
   6. `Walkthrough notes/changes/2026-10-10_noise_floor.md. Bắt đầu từ "why".`
   7. `Dạy mình paper Halimi (PGA), dùng literature/halimi_2022_six_questions.md và arXiv MCP để mở đúng section. Đi từ intuition đến figure/table chính.`
   8. `Dạy mình paper FedEraser, dùng literature/federaser_2021_six_questions.md và arXiv MCP để mở đúng section. Đi từ intuition đến figure/table chính.`
   The tutor also needs to add the three new items to
   `notes/understanding_map.md` at L0 (rows 23–24 still say "to build").
3. **Gate and design questions waiting for your answers.**
   - Gate 3: plan v4 lists it as awaiting the tutor's confirmation —
     `Gate 3: hỏi từng câu, chấm bài mình viết trong reports/gate3_self_check.md.`
   - Decisions for you and your supervisor; the canonical configs stay null
     until they are set (options are in §7 of the three change notes):
     1. Which randomness the noise floor varies: minibatch order only,
        also the initial weights, or both reported; and what "3 seeds"
        means given the four seed keys in the configs.
     2. The rule that turns pairwise gaps into a floor (max of pairs,
        mean + c·std, other), inclusive or strict at ties, and how many
        retrains per cell.
     3. What to report on target-dominant rows, where the accuracy-gap
        floor is expected to be near zero by construction.
     4. Whether M_FU is compared with one reference retrain or with all.
     5. Whether RQ1 gets one combined verdict from accuracy gap and
        agreement, or the two are reported separately.
     6. The non-target split is the non-target clients' training data:
        keep and name it so, or hold out a per-client test fraction.
     7. Number of rounds (R=5 is inherited, unconfirmed) and whether to use
        the validation split to fix it.
     8. CPU-only timing: the existing FedAvg server rejects MPS.
     9. How the target client is chosen and recorded (T1/T2/T3 is not built).
