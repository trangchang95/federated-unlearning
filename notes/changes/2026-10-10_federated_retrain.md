# Federated retrain M_retrain^(-A)

**Level: L0** (AI-built, not yet walked through)

Revised after the reviewer's critique (same change, same level). Revision list at the end of §5.

## 1. What and why (which RQ/H)

A config-driven runner that trains two FedAvg models on MNIST with 10 clients:

- **M**: all clients take part.
- **M_retrain^(-A)**: the target client A is left out of every round.

Both start from the same initial weights w_0, use the same client partition, seed, number of rounds and hyperparameters. M_retrain^(-A) is the gold standard of the thesis: every later statement about a method's effect on client A is a comparison with this model (plan v4 §6, CLAUDE.md §9). It serves RQ1–RQ3 (reference model) and RQ4 (t_train is the `Time_retrain` in Speedup).

This is the **federated** retrain. `experiments/unlearning/month4_week13_mnist_full_retraining.py` is a centralized retrain and was not touched.

## 2. Files/functions touched

New (no existing file was modified by this change):

| File | Content | Lines |
|---|---|---:|
| `algorithms/federated_retrain.py` | `remaining_client_datasets`, `train_fedavg_excluding` | 103 (about 45 of code) |
| `experiments/unlearning/month4_week14_mnist_federated_retrain.py` | `CODE_FILES`, `SPLIT_DEFINITIONS`, `check_supported_config`, `require_decided`, `set_random_seed`, `current_code_state`, `code_provenance`, `load_mnist`, `build_client_partition`, `make_initial_state`, `evaluation_loaders`, `train_and_evaluate`, `prepare_run`, `main` | 362 |
| `configs/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain.json`, `configs/smoke_month4_week14_v2_mnist_iid_federated_retrain.json` | current smoke configs (2 rounds, 5,000 train / 1,000 test examples) | — |
| `configs/smoke_month4_week14_mnist_alpha0.1_federated_retrain.json`, `configs/smoke_month4_week14_mnist_iid_federated_retrain.json` | first smoke configs. **Superseded**: they lack the new noise-floor keys, so the current noise-floor script and verifier refuse them. Kept only because `results/smoke_month4_week14_mnist_*` name them. | — |
| `configs/month4_week14_mnist_alpha0.1_federated_retrain_seed42.json`, `configs/month4_week14_mnist_iid_federated_retrain_seed42.json` | canonical configs, **prepared but not run**; open decisions are null | — |
| `tests/test_federated_retrain.py` | 11 toy tests | 210 |
| `reports/verify_month4_week14_federated_retrain.py` | verifier (shared with the noise-floor change) | 298 |

The runner is larger than the 150-line guideline. It is infrastructure (data loading, partition records, provenance, config guards, JSON); the concept itself is the ~45 lines in `algorithms/federated_retrain.py`.

**Not modified:** `algorithms/fedavg.py`, `server/fedavg_server.py`, `clients/federated_client.py`. Their levels are not reset by this change.

## 3. Formula ↔ code map

| Idea | Where |
|---|---|
| Remaining clients = all clients minus A, IDs kept | `algorithms/federated_retrain.py:49-53` |
| Retrain = unchanged `FedAvgServer` built on the remaining clients | `algorithms/federated_retrain.py:89-98` |
| The server passes only the counts of the clients it trained to the aggregation | `server/fedavg_server.py:202-205` (unchanged) |
| Renormalised weight n_k / Σ_{j≠A} n_j: total = sum of the counts received, weight = count / total | `algorithms/fedavg.py:143` and `:165` (unchanged) |
| (`server/fedavg_server.py:209` is a different total: it only weights the recorded local loss) | — |
| Each client trains a deep copy of the global model | `clients/federated_client.py:98` (unchanged) |
| Shuffle seed of a client depends on (seed, round, client ID) | `server/fedavg_server.py:58` (unchanged) |
| One timed phase per training round → t_train | `algorithms/federated_retrain.py:100-102` |
| Same initial weights: w_0 is created once … | `experiments/unlearning/month4_week14_mnist_federated_retrain.py:224-227`, called at `:324` |
| … and a copy is loaded into a fresh model for each run | same file `:257` |
| M uses `exclude_clients=()`, M_retrain^(-A) uses `(target,)`, same `random_seed` | same file `:326-333` |
| Per-client class histogram and near-empty flag | same file `:208-220` |
| Descriptive config fields must equal what the code runs | same file `:77-95` |
| A null design decision stops the run | same file `:98-105` |
| Git commit, worktree status, SHA-256 of the code files | same file `:117-141` |
| Smoke/canonical guard | same file `:144-156` |
| Checkpoints | same file `:352-354` |

No paper equation is implemented. The protocol comes from plan v4 §6 ("Gold standard").

## 4. Design choices and rejected alternatives

1. **Exclusion by filtering the client dictionary before the server is built.** Rejected: adding an `exclude_clients` argument to `FedAvgServer.run_round` or to `weighted_average_model_states`. Filtering needs no change to code that is required at L3, and the renormalisation follows from the existing line "total = sum of the counts I received".
2. **Client IDs are kept (not renumbered).** The local shuffle seed is derived from the client ID, so a remaining client sees its minibatches in the same order in M and in M_retrain^(-A). Rejected: renumbering to 0..K-2, which would silently change every client's data order. Pinned by a test that excludes client 0.
3. **"Same initial weights" means the same tensor values**, not only the same seed: one state dict w_0 is created after seeding with `model_initialization_seed`, saved as `initial_state.pt`, and cloned into each model. Rejected: re-seeding and constructing the model twice.
4. **Same training seed for M and M_retrain^(-A)** (`random_seed`), as the plan requires.
5. **`run_kind` in the config.** A canonical run refuses a dirty worktree; a smoke run records the dirty state and must write to a `smoke_*` directory (CLAUDE.md §4 now states this exception).
6. **`target_client` is an explicit ID in the config**; the canonical configs carry `null` and the runner refuses to start. Rejected for now: implementing the target rule here (a second concept; see §7).
7. **Device is `cpu`.** `FedAvgServer` and `train_client` accept only CPU or CUDA. Rejected: changing them to accept MPS (resets FedAvg to L0, and MPS is not bit-reproducible).
8. **No per-round evaluation and no plots**, to keep the runner short.
9. **The 15% validation split is still carved off and unused**, so the training set is the same 51,000 examples as in Months 2–3.
10. **Aggregation / BatchNorm:** `SimpleMLP` has no BatchNorm layers, so there are no running statistics or integer buffers to aggregate.
11. **Descriptive config fields are asserted, not trusted.** The code always runs MNIST, SimpleMLP, plain SGD, CrossEntropyLoss, ToTensor; `check_supported_config` stops the run if `dataset`, `model`, `architecture`, `number_of_classes`, `preprocessing`, `optimizer`, `optimizer_details` or `loss_function` say anything else. Rejected: reading them to choose a model/optimizer (more code than this milestone needs).
12. **`code_sha256`**: SHA-256 over the bytes of the 14 files in `CODE_FILES`. Added because a commit hash identifies nothing on a dirty worktree.
13. **Result JSON labels.** The unlabelled top-level `accuracy`, `f1`, `unlearning_time`, `communication_cost` were removed from `metrics.json`; the measured values are under `original` and `retrain_without_target`. The config still carries these four keys as placeholders because `load_json_config` requires them; `placeholder_fields_note` in the config says so. `split_definitions` states that the target and non-target splits are **training** data.

## 5. How it was verified

- Unit tests, `tests/test_federated_retrain.py` (11 tests), on a 3-client toy federation (n = 2, 6, 4) with a 2×2 linear model:
  - the target is removed and IDs are kept; excluding nobody keeps everyone; an unknown ID or excluding everyone is an error;
  - arithmetic only (`FilterThenAverageArithmeticTests`): scalar weights 0, 4, 100 with n = 1, 3, 6: all clients give 61.2, filtered then averaged gives (0·1 + 4·3)/4 = 3.0. This test does not call the training function;
  - excluding client 2: in every round the selected clients are (0, 1) and n = 8, not 12; the timer is entered once per round;
  - excluding client 2: one round equals 0.25·(client 0 model) + 0.75·(client 1 model), the two local models trained separately;
  - **excluding client 0**: remaining keys `[1, 2]`, selected `(1, 2)`, one round equals 0.6·(client 1 model) + 0.4·(client 2 model) where each local model is trained with the seed of its ORIGINAL id; training client 1's data with the seed of id 0 gives a different model;
  - replacing the target's data leaves the retrain bit-identical, but changes the original model;
  - retrain ≠ original; same seed gives the same result; a different training seed gives a different result;
  - runner's `train_and_evaluate` on toy 28×28 data: the shared `initial_state` is unchanged after the M run and after the retrain run; both runs moved away from it and ended in different places.
- Full suite: 167 tests OK (122 before the three changes).
- Smoke runs (SMOKE numbers, not canonical, not thesis evidence; CPU, 2 rounds, 5,000 training examples, 10 clients, target client 0 chosen arbitrarily):
  - `configs/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain.json` → `results/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain/metrics.json`: t_train original 0.47 s, retrain 0.31 s;
  - `configs/smoke_month4_week14_v2_mnist_iid_federated_retrain.json` → `results/smoke_month4_week14_v2_mnist_iid_federated_retrain/metrics.json`: t_train original 0.43 s, retrain 0.32 s.
  - In the α=0.1 smoke partition clients 7 (21 examples) and 9 (73 examples) are flagged near-empty (< 128).
- CPU determinism (checked on the first smoke run): retraining again in the same environment gave a checkpoint bit-identical to the saved `retrain_without_target.pt`.
- Both canonical configs were started once to check the guards: they stopped before training (dirty worktree; null design decisions) and wrote nothing.
- Verifier `reports/verify_month4_week14_federated_retrain.py`: PASS on both v2 smoke directories (see `2026-10-10_noise_floor.md` §5).

Revisions after review: tests for excluding client 0 and for the runner's initial weights (two tests renamed to say what they test); `git status` output no longer loses the leading space of its first line (`:127-130`); `code_sha256`; `check_supported_config`; `require_decided`; unlabelled top-level result fields removed; note line references corrected.

## 6. Three questions a committee could ask

1. Why is M_retrain^(-A) started from the same initial weights and seed as M? What would change in your conclusions if it used a fresh initialisation?
2. Show in the code where the aggregation weights become n_k / Σ_{j≠A} n_j. You did not write a new aggregation function — why is the result still correct?
3. If only a fraction of clients is sampled per round (C < 1), is your retrain still "the original run without A"? Why or why not?

## 7. Known limitations / open risks

- **Smoke runs were made on a dirty worktree** (the new files are uncommitted; only the student commits). The recorded HEAD `75401d3…` does not contain the new code; `code_sha256` identifies the 14 code files instead, but those files cannot be recovered from it.
- **C < 1:** the per-round sample is drawn from the remaining clients, so it is not the original sample minus A (marked `# DEVIATION` in the code). All configs use C = 1.0.
- **No resume-from-checkpoint** inside a run.
- Only MNIST + `SimpleMLP`. CIFAR-10 needs a model/dataset switch in the runner.
- t_train covers the training rounds only (no data loading, no evaluation).
- The canonical grid is blocked until the core functions are at L2 in `notes/understanding_map.md`.

Open questions for the student/supervisor (not decided here):

- **Target-selection rule:** not implemented and not recorded in the result. Options: implement the rule and record rule + chosen ID in the JSON / set the ID by hand per config and record the rule name as text.
- **R=5, E=1, B=128, lr=0.1** in the canonical configs are copied from Months 2–3. Options: keep / choose by a convergence check on the 10-client setting.
- **Unused 15% validation split.** Options: keep it (same 51,000 training examples as earlier months) / give it back to training / use it for model selection.
- **CPU-only timing.** Options: accept CPU t_train for Speedup (all methods on the same device) / extend the server to MPS (resets FedAvg to L0).
- **Meaning of "3 seeds" in the plan's grid**, given four seed keys (`split_seed`, `train_partition_seed`, `model_initialization_seed`, `random_seed`): which of them change from seed to seed. See also `2026-10-10_noise_floor.md` §7.

## Tutor findings
