# Noise floor (k M_retrain^(-A) runs, different seeds)

**Level: L0** (AI-built, not yet walked through)

Revised after the reviewer's critique (same change, same level). The first version used one pair of retrains and compared floats; the reviewer showed that its verdicts were decided by exact ties. This version replaces it. Revision list at the end of §5.

## 1. What and why (which RQ/H)

Retrains without the target client never saw the target's data, yet they are not identical if their random seed differs. How far apart they are is the **noise floor**: the difference that exists without any "remembering". A model M_x is only said to match M_retrain^(-A) when its gap is not larger than this (plan v4 §2 row 2 and §6, CLAUDE.md §9–§10).

This change adds:

- `pairwise_retrain_rows(...)`: every pair of k retrains compared with the same `compare_to_retrain` used for any other model;
- `summarise_pairs(...)`: per row, the count of pairs and mean / std / max of the pairwise |accuracy gap| and mean / std / min of the pairwise agreement;
- `within_noise_floor(..., decision_rule)`: per row, is M_x inside the floor, under a rule named in the config;
- a positive control: the same verdict for the original model M, which was never unlearned;
- a script that trains the extra retrains and writes `noise_floor.json`;
- the verifier for the retrain + noise-floor result files.

It serves RQ1 (gap vs noise floor), H1–H2 (phrased relative to the noise floor) and Requirement 3.

## 2. Files/functions touched

| File | Content | Lines |
|---|---|---:|
| `evaluation/forgetting.py` | `DECISION_RULES`, `pairwise_retrain_rows`, `summarise_pairs`, `within_noise_floor` | lines 159–267 (about 60 of code) |
| `experiments/unlearning/month4_week14_mnist_noise_floor.py` | `NOISE_FLOOR_FIELDS`, `retrain_seed_plan`, `evaluate_checkpoint`, `main` | 232 |
| `reports/verify_month4_week14_federated_retrain.py` | consistency check of `metrics.json` + `noise_floor.json` | 298 |
| `tests/test_noise_floor.py` | 18 toy tests | 210 |
| Week 14 configs | keys `noise_floor_number_of_retrains`, `noise_floor_extra_seeds`, `noise_floor_seed_mode`, `noise_floor_decision_rule`, `target_dominant_threshold` | — |

## 3. Formula ↔ code map

R_0 … R_{k−1} are the k retrains without A (R_0 is the one from the first runner, trained with `random_seed`). For a pair p = (i, j), i < j, on a row with N examples: d_p = |correct(R_j) − correct(R_i)| and a_p = #{examples on which R_i and R_j predict the same label}. Both are integers.

| Formula | Where |
|---|---|
| all pairs i < j, each compared with `compare_to_retrain(R_j, R_i)` | `evaluation/forgetting.py:177-182` |
| d_p and a_p for every pair | `evaluation/forgetting.py:212-213` |
| max_abs_correct_difference = max_p d_p ; min_agree = min_p a_p (integers) | `evaluation/forgetting.py:217-218` |
| abs_accuracy_gap mean / std / max = mean, population std, max of d_p, divided by N | `evaluation/forgetting.py:219-221` |
| agreement mean / std / min = mean, population std, min of a_p, divided by N | `evaluation/forgetting.py:222-224` |
| rule must be one of `DECISION_RULES`; no default | `evaluation/forgetting.py:162`, `:249-250` |
| "max_of_pairs_inclusive": \|correct_x − correct_ref\| ≤ max_p d_p ; agree(M_x, ref) ≥ min_p a_p | `evaluation/forgetting.py:259-262` |
| "max_of_pairs_strict": the same with < and > | `evaluation/forgetting.py:264-265` |
| seeds and checkpoint name of every retrain | `experiments/unlearning/month4_week14_mnist_noise_floor.py:82-112` |
| null noise-floor key → stop | same file `:84` |
| mode "training": reuse the saved w_0; otherwise draw new w_0 | same file `:162-165` |
| extra retrains trained one after the other, same exclusion | same file `:161-172` |
| pairs, summary | same file `:179-180` |
| positive control: M vs R_0, its within-floor flags, and their negation | same file `:185-199` |
| runtime environment validated and compared with the first run | same file `:134`, `:144-145` |
| commit and code hash must equal the first run's | same file `:141-143` |

## 4. Design choices and rejected alternatives

1. **k retrains and all pairs, not one pair.** One pair gives one number, and a verdict against one number is decided by which seed happened to be drawn (and, at a tie, by `≤` vs `<`). k retrains give k(k−1)/2 pairwise gaps. The result file stores a list `retrains` and a list `pairs`; nothing in the layout assumes k = 2 or one pair per directory. Rejected: keeping one pair per directory and aggregating across directories later.
2. **Decisions on integer counts.** d_p, a_p and M_x's counts are integers over the same N, so "equal" is exact. Rejected: comparing float gaps (the first version did; equality there depended on rounding).
3. **The decision rule is a required config key with no default** (`noise_floor_decision_rule`). Only the two simplest rules are implemented: the floor is the largest pairwise difference (and the smallest pairwise agreement), with ties counted as inside (`max_of_pairs_inclusive`) or outside (`max_of_pairs_strict`). The smoke configs use `max_of_pairs_inclusive` as an **arbitrary** value. Not implemented: mean + c·std, a quantile of the pairwise gaps, a statistical test.
4. **Which seed is varied is a required config key with no default** (`noise_floor_seed_mode`):
   - `"training"`: only the FedAvg training seed changes. **With `client_fraction = 1.0` (all Week 14 configs) every client is selected in every round whatever the seed, and `SimpleMLP` has no dropout, so the only thing that differs between the retrains is the order of the minibatches inside each client.** Initial weights are the saved `initial_state.pt`.
   - `"training_and_init"`: the training seed changes and new initial weights are drawn with the extra seed.
   - **Never varied:** `split_seed` and `train_partition_seed`. The partition decides which examples belong to the target; another partition is another unlearning request.
   - The smoke configs use `"training"` as an **arbitrary** value. The choice is not neutral: a floor from minibatch order alone is expected to be smaller than one that also varies w_0. A smaller floor is stricter for "FU is within the floor" (H1) and more lenient for "the gap exceeds the floor" (H2).
5. **Number of retrains and their seeds are required config keys** (`noise_floor_number_of_retrains`, `noise_floor_extra_seeds`); smoke uses 3 and `[43, 44]`, arbitrary. The seeds are listed explicitly instead of derived by a hidden rule.
6. **Population standard deviation over the pairs** (divide by the number of pairs). The pairs share retrains (3 pairs from 3 retrains), so they are not independent and the std is a description of spread, not a standard error.
7. **Positive control.** The original model M trained on the target and was not unlearned. `noise_floor.json` stores, per row and metric, whether M is **outside** the floor (`positive_control_original_outside_floor`). If M is inside, the row cannot separate "unlearned" from "not unlearned" in that setting, and the verifier prints it as UNINFORMATIVE. Rejected: leaving the reader to notice this from the numbers.
8. **Two separate answers per row** (accuracy gap, agreement), no combined verdict (see open questions).
9. **M is compared with R_0** (`reference_retrain_index: 0`), the retrain with the same seeds as M, i.e. the gold standard as the plan defines it. See open questions.
10. **The extra retrains are a separate script writing into the same results directory.** It refuses to run if the config, partition, commit, code hash or runtime environment differ from the first run, and never overwrites `noise_floor.json`.
11. **The verifier recounts independently.** It computes predictions from the checkpoints with a forward pass written in the verifier and recounts every count, class share, summary statistic and flag with numpy, without importing `evaluation/forgetting.py`. Rejected: calling the same functions as the runner (a shared bug would pass).

## 5. How it was verified

- Unit tests, `tests/test_noise_floor.py` (18 tests):
  - three hand-written retrains with 3/5, 4/5, 5/5 correct on the target split → pairs (0,1), (0,2), (1,2); d = 1, 2, 1; a = 4, 3, 4; max 2; min agree 3; mean (1+2+1)/3/5; population std √(2/9)/5; global, non-target and dominant rows checked by hand; one retrain alone is an error; an empty class group has only its counts;
  - `within_noise_floor` against floor (max difference 2, min agree 8) on 10 examples: inside, outside, and the two answers being independent; exactly at the floor is inside under the inclusive rule and outside under the strict rule; a missing rule and mismatched example counts are errors; an empty row gives `None`;
  - **float trap:** 9/10 vs 6/10 and 7/10 vs 4/10 are both a gap of 3 answers; the test asserts that `0.9 - 0.6 > 0.7 - 0.4` in floating point and that the integer comparison still says "at the floor";
  - `retrain_seed_plan`: mode "training" → (training seed, w_0 seed) = (42,42), (43,42), (44,42); mode "training_and_init" → (42,42), (43,43), (44,44); checkpoint names; each of the four null keys stops the run; wrong number of seeds, repeated seed, seed equal to the first retrain's, unknown mode and k = 1 are errors;
  - `check_supported_config` accepts the supported values and rejects another optimizer, model or architecture.
- Full suite: 167 tests OK (122 before the three changes).
- Smoke runs (SMOKE numbers, not canonical, not thesis evidence; CPU; 2 rounds; 5,000 training / 1,000 test examples; target client 0; 3 retrains with training seeds 42, 43, 44; mode "training"; rule "max_of_pairs_inclusive"; all of these arbitrary):
  - `results/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain/noise_floor.json` — t_train of the extra retrains 0.34 s and 0.32 s; target-dominant classes of client 0: `[4]`;
  - `results/smoke_month4_week14_v2_mnist_iid_federated_retrain/noise_floor.json` — t_train of the extra retrains 0.35 s and 0.31 s; no target-dominant class.
- Verifier on both v2 smoke configs: `PASS (consistency check, nothing retrained)`. Its per-row output:
  - α=0.1: accuracy gap informative on global, target, non-target and shared rows, UNINFORMATIVE on the dominant row; agreement informative only on the target row.
  - IID: accuracy gap UNINFORMATIVE on every non-empty row; agreement informative on every non-empty row; dominant row has no examples.
- Negative checks (one saved value changed in memory, files untouched): a pair's `agree` + 1 → `FAIL: pair (0, 1) target_client: agree is 552, recount is 551`; a positive-control flag flipped → `FAIL: target_client: positive control flags`; runtime changed → `FAIL: noise-floor runtime differs from the first run`; a summary std changed → `FAIL: global_test: abs_accuracy_gap_std`.
- Both canonical configs stop with `Config field 'noise_floor_number_of_retrains' is null. It is a design decision with no default; …`.

**What was found about the target-dominant row at α=0.1** (source: the four checkpoints in `results/smoke_month4_week14_v2_mnist_alpha0.1_federated_retrain/`, predictions recomputed over all 6,000 smoke examples): after 2 rounds none of the four models (M, R_0, R_1, R_2) predicts class 4 for any example, nor class 2 or 7. Class 4 is the target-dominant class (client 0 holds 459 of its 464 training examples). So every model, including M which trained on those 459 examples, has 0 correct on that row: correct = 0 vs 0 in all three pairs and for M vs R_0. The accuracy gap is 0 for everybody and cannot tell the models apart. On agreement, M agrees with R_0 on 385 of 459 examples and the three pairs agree on 385, 402 and 442, so M sits exactly at the minimum (a tie, inside under the inclusive rule). The smoke models are undertrained (global test accuracy about 0.37–0.41), so this says nothing about a canonical run; it does show that a row can be uninformative and why the positive control is needed.

Revisions after review: one pair → k retrains with all pairs and a summary; integer comparisons; decision rule, seed mode, number of retrains and seeds are required config keys with no default; positive control and UNINFORMATIVE marking; runtime environment validated and saved by the noise-floor script; guard on code hash, not only on the commit; verifier rewritten as an independent recount with a docstring stating what PASS does and does not mean.

## 6. Three questions a committee could ask

1. In your retrains only the minibatch order differs. Is your noise floor then too small? What would you expect if the initial weights also changed, and how would that move your conclusions for H1 and H2?
2. Your rule says "inside the floor if the gap is at most the largest gap between retrains". What happens to the verdict at a tie, and why did you compare integer counts instead of accuracies?
3. On a row where the original model M is itself inside the floor, what can you conclude about an unlearning method that is also inside? What did you do with such rows?

## 7. Known limitations / open risks

- **Near-zero accuracy-gap floor on target-dominant rows, by construction (expectation, not yet tested on canonical runs).** On a class held almost entirely by the target, every retrain has seen almost no example of that class and is expected to score near zero on it whatever the seed. The pairwise accuracy gap on that row is then near zero for a structural reason, not because training is stable, and any method that retains some accuracy on the class falls outside the floor. In the 2-round smoke run the situation is more extreme: no model predicts the class at all (see §5), so even M is inside.
- **A row where M is inside the floor is uninformative**; in the smoke runs this is most rows for one of the two metrics. Whether that persists with trained models is not known.
- With k = 3 the summary rests on 3 dependent pairs. The std is descriptive only. Do not use the word "significant".
- With deterministic CPU training the same seed gives a bit-identical retrain, so the floor exists only because a seed is changed on purpose. On MPS there would be extra, uncontrolled noise.
- **The "same commit" guard means nothing on a dirty worktree.** In the first smoke run the retrain step and the noise-floor step recorded the same HEAD but different worktree status lists, and the guard passed. The guard now also compares `code_sha256` (14 code files), which does detect a code change between the two steps; it still cannot restore the code that ran.
- The verifier is a consistency check: nothing is retrained; data and partition are rebuilt with the runner's `prepare_run`; loss and macro-F1 are recomputed with the runner's evaluation code. Its docstring lists this.
- The verifier needs the noise-floor step to have run; it fails on a directory that only has the first runner's output. The first smoke directories (`results/smoke_month4_week14_mnist_*`) were produced by the superseded one-pair version and cannot be checked by the current verifier.
- The smoke runs were made on a dirty worktree; the canonical configs are prepared but not run, and are blocked until the core functions are at L2.

Open questions for the student/supervisor (not decided here; the canonical configs carry null for the first three and the scripts refuse to start):

- **Which randomness the floor varies** (`noise_floor_seed_mode`): minibatch order only / also the initial weights w_0 / report both floors side by side.
- **What "3 seeds" in the plan's grid means**, given four seed keys (`split_seed`, `train_partition_seed`, `model_initialization_seed`, `random_seed`), and how many retrains per (config, target) the floor uses (`noise_floor_number_of_retrains`, `noise_floor_extra_seeds`): the grid's seeds double as the floor's retrains / separate extra retrains per grid cell.
- **Decision rule and tie convention** (`noise_floor_decision_rule`): max of pairs / mean + c·std / a quantile / a statistical test; ties inside or outside.
- **Target-dominant rows with a near-zero floor by construction:** report the row but give no verdict / use agreement only on that row / use a different reference (e.g. chance level) / report the positive control next to every verdict.
- **Reference for M_FU:** compare with one retrain (R_0, same seeds as M) / with every retrain and report the range / with the majority prediction of the retrains.
- **RQ1 verdict:** keep two separate answers (accuracy gap, agreement) / define one combined verdict (both inside, either inside).
- **Std convention:** population std over pairs (current) / sample std / std over retrains instead of over pairs.

## Tutor findings
