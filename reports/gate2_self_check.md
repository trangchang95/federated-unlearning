# Gate 2 Beginner Self-Check — FedAvg and the First Multi-Client Run

**Current status:** awaiting the student's answers and two controlled config
changes. Week 8's technical experiment has passed verification, but Gate 2 is
not closed merely because code ran successfully.

Read
[`month2_week8_centralized_vs_fedavg.md`](month2_week8_centralized_vs_fedavg.md)
before answering. Use your own words. It is fine to use short sentences, but
each answer must explain *why*, not only repeat a definition.

## Evidence card

These are saved-run facts that you may use in your explanations:

| Item | Centralized SGD | Hand-written FedAvg |
|---|---:|---:|
| MNIST test accuracy | 94.75% | 90.99% |
| MNIST test macro F1 | 94.68% | 90.86% |
| Best validation step | epoch 5 | round 5 |
| Optimizer steps | 1,995 | 2,000 local steps |
| Training-example exposures | 255,000 | 255,000 |
| Estimated model-transfer payload | 0 bytes | 20,354,000 bytes |

The configured protocol is `K=5`, `C=1.0`, `E=1`, `B=128`, and `R=5` with
plain SGD at learning rate 0.1. Both paths start from the same model and use
the same 51,000 training examples. The final test set is evaluated only after
validation selects a checkpoint.

## Part A — Explain the completed run

1. Trace one complete FedAvg round. Where does `optimizer.step()` change
   weights, and where does sample-weighted aggregation change weights?

   **Answer:**

2. Why must every selected client in one round start from the same global
   model? What would go wrong if each client started from an independent
   random model?

   **Answer:**

3. Explain `K`, `C`, `E`, `B`, and `R` in your own words, then state their
   values in the saved run.

   **Answer:**

4. The two paths process the same 255,000 example exposures, but centralized
   SGD uses 1,995 optimizer steps and FedAvg uses 2,000. Explain why the five
   extra steps do **not** mean FedAvg received extra training examples.

   **Answer:**

5. State the measured accuracy result and the 3.76-percentage-point
   difference. Why is it correct to report that FedAvg underperformed in this
   run, but incorrect to conclude that centralized learning is universally
   better?

   **Answer:**

6. Both validation-accuracy curves rise and both validation-loss curves fall
   across all five passes. What does this support? Why does it not prove
   mathematical convergence or a stable final limit?

   **Answer:**

7. Reconstruct the communication estimate using:

   ```text
   model payload × two directions × selected clients × rounds
   ```

   What does the 20,354,000-byte value include, and which real-network costs
   does it omit?

   **Answer:**

8. The report includes five IID client test partitions. Why should we inspect
   per-client utility in addition to global accuracy? Also explain why one
   fixed seed gives repeatability but not statistical equivalence.

   **Answer:**

## Part B — Required hands-on changes

Do not overwrite the canonical config or its result directory. Each exercise
must use a copied JSON config, a new `experiment_name`, and a new
`output_subdirectory`. Every executed variant must still be committed/tagged
before training because the runner rejects unversioned experiment claims.

First write your proposed field changes and predictions below. After review,
create and run the two variants. The exact commands and saved outputs will be
recorded during the review; hand-copying numbers into a spreadsheet is not
allowed.

9. **Change the number of clients.** Propose a `K=10` variant while keeping
   `C=1`, `E=1`, `B=128`, and `R=5`.

   - Which JSON fields and names will you change?
   - How many training examples should each equal-sized client receive?
   - How many clients participate per round?
   - Predict the dense communication bytes using the formula in Question 7.
   - Predict whether the total local optimizer-step count remains the same or
     changes, and explain the minibatch arithmetic.

   **Proposed changes and prediction:**

   **Saved run/result after review:**

10. **Change local epochs.** Propose an `E=2` variant with `K=5`, `C=1`,
    `B=128`, and `R=5`.

    - Which JSON fields and names will you change?
    - Why must `centralized_epochs` become `R × E` for the matched-budget
      comparison?
    - Predict the training-example exposures and optimizer-step totals.
    - Predict whether communication bytes change when `K` and `R` stay fixed.
    - After the run, compare its validation curve with the canonical `E=1`
      curve without claiming that one seed proves a general rule.

    **Proposed changes and prediction:**

    **Saved run/result after review:**

## Pre-run proposal review

This review happens after the two predictions are written but before either
config is created, tagged, or executed. Its committed marker makes that order
auditable: the producing Git tag must already contain this approval.

**Review date:** NOT YET REVIEWED

**Result:** NOT YET REVIEWED

**Reviewer notes:** The reviewer will check that only the intended `K` or `E`
field changes, the centralized exposure budget stays matched, names/output
directories are new, and all predicted arithmetic is explained.

## Passing standard

Gate 2 passes only when the student can:

- distinguish client-side optimizer updates from server aggregation;
- correctly explain the saved result and convergence plot;
- report the negative centralized-versus-FedAvg difference honestly;
- explain matched exposure versus different optimizer trajectories;
- change `K` and `E` through separate reproducible configs and interpret the
  resulting utility, step-count, and communication changes; and
- avoid claims about Non-IID behavior, privacy, unlearning, real network
  speed, or statistical equivalence that this experiment did not test.

## Review result

**Review date:** NOT YET REVIEWED

**Result:** NOT YET REVIEWED

**Reviewer notes:** The reviewer will assess the explanations and predictions
for conceptual correctness, then verify both saved variant runs. This field is
not an automated substitute for that review.

Week 8 is technically complete. Gate 2 remains open until the answers and both
hands-on changes above are reviewed. Month 3 Non-IID work and FedProx remain
blocked until then; Federated Unlearning remains blocked by Gates 2–3 and the
later baseline requirements.
