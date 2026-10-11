# Forgetting metrics against M_retrain^(-A)

**Level: L0** (AI-built, not yet walked through)

Revised after the reviewer's critique (same change, same level): every row now carries integer counts; `classify_classes` validates its input.

## 1. What and why (which RQ/H)

`evaluation/forgetting.py` compares any model M_x with the gold standard M_retrain^(-A):

- accuracy of both models and the **accuracy gap** on the three splits (global test, target client, non-target clients);
- **prediction agreement**: the fraction of examples on which both models predict the same label;
- the target split divided into **shared-class** and **target-dominant** classes.

It serves RQ1 (gap per split), RQ2 (shared vs target-dominant), RQ3 (non-target and global rows) and RQ5 (accuracy gap and agreement are two metrics that can disagree). CLAUDE.md §9: a claim about forgetting is only ever a comparison with M_retrain^(-A), never a low target accuracy alone.

Parameter distances are not computed (plan v4 §6).

## 2. Files/functions touched

| File | Content | Lines |
|---|---|---:|
| `evaluation/forgetting.py` (was a one-line placeholder) | `SPLITS`, `classify_classes`, `prediction_agreement`, `compare_predictions`, `compare_to_retrain` | lines 1–157 (about 80 of code) |
| `tests/test_forgetting.py` | 16 toy tests | 188 |

Reused, not modified: `evaluation/three_way.py` (`ThreeWayEvaluation` holds the labels and predictions of each split), `evaluation/utility.py`.

## 3. Formula ↔ code map

N examples (x_i, y_i); p_x(i), p_r(i) are the predicted labels of M_x and M_retrain^(-A); n_{k,c} is the number of training examples of class c at client k; A is the target.

| Formula | Where |
|---|---|
| input check: one count per class for every client, no negative count | `evaluation/forgetting.py:63-67` |
| share_c = n_{A,c} / Σ_k n_{k,c} (0 if nobody holds class c) | `evaluation/forgetting.py:72-73` |
| class c is target-dominant ⇔ share_c ≥ threshold; otherwise shared | `evaluation/forgetting.py:75-78` |
| correct_x = #{i : p_x(i) = y_i} (integer) | `evaluation/forgetting.py:107` |
| correct_retrain = #{i : p_r(i) = y_i} (integer) | `evaluation/forgetting.py:108` |
| agree = #{i : p_x(i) = p_r(i)} (integer) | `evaluation/forgetting.py:109` |
| accuracy_x = correct_x / N, accuracy_retrain = correct_retrain / N | `evaluation/forgetting.py:115-116` |
| accuracy_gap = (correct_x − correct_retrain) / N (signed) | `evaluation/forgetting.py:117` |
| agreement = agree / N | `evaluation/forgetting.py:118` (stand-alone version: `:82-89`) |
| same examples in the same order for both models (guard) | `evaluation/forgetting.py:142-143` |
| target examples grouped by their TRUE label into dominant / shared | `evaluation/forgetting.py:148-156` |

The threshold is the config key `target_dominant_threshold` (0.5 in all Week 14 configs, from plan v4 §2 row 3: "target holds ≥ 50% of the examples of the class").

## 4. Design choices and rejected alternatives

1. **One primitive (`compare_predictions`) used for five rows**: three splits plus the two class groups of the target split.
2. **Works on two `ThreeWayEvaluation` objects**, which already contain labels and predictions. Rejected: running the models again inside `forgetting.py`.
3. **Integer counts are stored next to the fractions** (`examples`, `correct_x`, `correct_retrain`, `agree`). Every later decision is taken on the counts; the floats are for display. Reason: two gaps that are both "3 answers out of 10" can differ as floats (0.9 − 0.6 > 0.7 − 0.4 in floating point). Rejected: comparing floats with a tolerance (a tolerance is one more number to defend).
4. **Signed gap** (M_x minus M_retrain); the absolute value is taken only where a comparison with the noise floor needs it.
5. **Threshold rule `share ≥ threshold`, inclusive.** Exactly half counts as dominant, as written in the plan. The share is computed over **training** counts of all clients **including** the target.
6. **Class groups use the true label of each target example**, not the predicted label.
7. **Empty group → `None` for every fraction, counts 0.** Under IID there is no target-dominant class; an accuracy of 0.0 would look like a result.
8. **Agreement ignores the true label.** Two models making the same mistake agree.
9. Rejected: macro-F1 gap and KL divergence of outputs. Not requested for this milestone.

## 5. How it was verified

- Unit tests, `tests/test_forgetting.py` (16 tests):
  - class counts `[6,1,0] / [2,1,5] / [2,2,5]`: target 0 has shares 0.6, 0.25, 0.0 → dominant `[0]`, shared `[1,2]`; a share of exactly 0.5 is dominant; threshold 0.25 gives `[0,1]`; equal shares give no dominant class; a class nobody holds is shared; unknown target, thresholds 0.0 / 1.5, histograms of unequal length and a negative count are errors;
  - agreement of `[0,1,2,2]` and `[0,1,1,2]` is 0.75; unequal lengths and empty input are errors;
  - four examples: counts (correct_x, correct_retrain, agree) = (3, 2, 3), accuracy 0.75 vs 0.5, gap +0.25, agreement 0.75 (one position where both are wrong still counts as agreement);
  - an empty group gives counts 0 and `None` fractions;
  - five-row comparison on hand-written predictions (global gap 0.5, target gap 0.4, dominant row 2 examples, shared row 3 examples);
  - a model compared with itself: gap 0, agreement 1; evaluations on different examples are rejected;
  - integration with a real `evaluate_three_way` call on an identity model.
- Full suite: 167 tests OK.
- Independent recount: the verifier recomputes predictions from the checkpoints with its own forward pass and recounts every `examples / correct_x / correct_retrain / agree` with numpy, without importing this module; PASS on both v2 smoke runs (`2026-10-10_noise_floor.md` §5).

## 6. Three questions a committee could ask

1. Two models have the same accuracy on the target split. Can their agreement be below 1? Give an example and say which of the two metrics tells you more about whether M_x behaves like the retrain, and why.
2. Why 50% as the threshold for target-dominant, and why is the target's own data included in the denominator? What happens to your RQ2 result if the threshold is 30% or 70%?
3. The target client's data was training data for M but not for M_retrain^(-A). Why is the target split still a fair place to compare them?

## 7. Known limitations / open risks

- With the inclusive rule and threshold 0.5, **two clients can both "dominate" the same class** (each holding exactly half).
- The threshold is a single fixed number; no sensitivity analysis is built.
- Small groups: a class group with very few target examples gives coarse accuracies (steps of 1/N). The row reports `examples`.
- "Shared" includes classes of which the target holds zero examples; those contribute no examples to the shared row.
- The functions trust that both evaluations used the same unshuffled loaders. The label-equality guard catches different examples with different labels, not two orderings that happen to have equal label sequences.
- No confidence-based metric (KL, entropy) and no backdoor ASR here.
- `evaluation/forgetting.py` must reach L3 before the defense.

Open questions for the student/supervisor (not decided here):

- **Naming of the non-target split.** The `target_client` and `non_target_clients` rows are the clients' **training** examples (only `global_test` is held-out data). The key names come from `evaluation/three_way.py` and were not changed; the result files carry a `split_definitions` block. Options: keep keys + definitions block / rename keys in `three_way.py` (resets it to L0) / add held-out per-client test splits.
- **Empty dominant group (IID, or a target with no dominant class).** When results are averaged over seeds and targets, the dominant row has fewer contributing cells than the other rows. Options: report the number of contributing cells next to every mean / report the dominant row only for settings where it is non-empty.

## Tutor findings
