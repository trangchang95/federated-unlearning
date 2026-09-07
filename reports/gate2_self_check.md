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
   At the beginning of round t, the server has the global model w_t and sends a copy of it to the selected clients. Each client trains its local copy using its own data with the normal training loop: forward pass, loss calculation, backward pass, and optimizer.step(). The optimizer.step() changes the local model weights on each client. After local training, the clients send their updated models back to the server. The server then performs sample-weighted aggregation, which changes the global weights and produces the next global model w_{t+1}.

2. Why must every selected client in one round start from the same global
   model? What would go wrong if each client started from an independent
   random model?

   **Answer:**
   Every selected client should start from the same global model so that their local models represent different updates from the same starting point. If each client started from an independent random model, the differences between their final weights would come from both random initialization and local data. Averaging those models would therefore not represent the intended FedAvg update and could make training unstable or meaningless.
                    same starting point
                       wt
                    /  |  \
                   /   |   \
                  A    B    C 
                  ↓    ↓    ↓
                 wA   wB   wC
                  \    |   /
                   \   |  /
                    FedAvg
                       ↓
                      wt+1

3. Explain `K`, `C`, `E`, `B`, and `R` in your own words, then state their
   values in the saved run.

   **Answer:**
   K is the total number of available clients. (  "number_of_clients": 5) 
   C is the fraction of clients selected in each communication round. ("client_fraction": 1.0)
   E is the number of local epochs each selected client performs before sending its model back. ("local_epochs": 1) 
   B is the local batch size used for each optimizer step. ("batch_size": 128)
   R is the total number of communication rounds. (  "number_of_rounds": 5)
   In the saved run, there are 5 clients, all 5 clients participate in each round, 1 local epoch, batch size 128, and 5 communication rounds.

4. The two paths process the same 255,000 example exposures, but centralized
   SGD uses 1,995 optimizer steps and FedAvg uses 2,000. Explain why the five
   extra steps do **not** mean FedAvg received extra training examples.

   **Answer:**
   The number of optimizer steps and the number of example exposures are different quantities. Both approaches process the same 255,000 example exposures, so FedAvg did not receive additional training data. The difference of five optimizer steps comes from how the examples are grouped into batches and how the final batches are handled across the centralized and federated training procedures. Therefore, five additional optimizer steps do not imply five additional examples or an additional amount of training data equal to five examples.

5. State the measured accuracy result and the 3.76-percentage-point
   difference. Why is it correct to report that FedAvg underperformed in this
   run, but incorrect to conclude that centralized learning is universally
   better?

   **Answer:**
   In this run, centralized training achieved 94.75%  while FedAvg achieved 90.99% , giving a difference of 3.76 percentage points in favor of centralized training. It is correct to report that FedAvg underperformed in this particular experiment because that is what the measured result shows. However, one fixed configuration and one run are not enough to conclude that centralized learning is universally better than FedAvg. The result can depend on the number of clients, local epochs, client data distribution, optimization settings, number of rounds, and random seed.

6. Both validation-accuracy curves rise and both validation-loss curves fall
   across all five passes. What does this support? Why does it not prove
   mathematical convergence or a stable final limit?

   **Answer:**
The increasing validation accuracy and decreasing validation loss across all five passes support the conclusion that both centralized training and FedAvg are learning useful patterns and improving on the validation data during this experiment. However, this does not prove mathematical convergence or that either method has reached a stable final limit. Five passes are only a finite observation, and the curves could behave differently with more rounds or different settings.

7. Reconstruct the communication estimate using:

   ```text
   model payload × two directions × selected clients × rounds
   ```

   What does the 20,354,000-byte value include, and which real-network costs
   does it omit?

   **Answer:**
   The estimate is calculated as:
model payload × two directions × selected clients × communication rounds.
The two directions represent sending the global model from the server to clients and sending the updated local model back to the server. The resulting 20,354,000 bytes therefore estimates the total model-payload communication for the saved FedAvg experiment. It does not represent the complete cost of a real network. It omits protocol headers, encryption or security overhead, connection setup, acknowledgements, retries, compression effects, latency, and other network or system overheads.

8. The report includes five IID client test partitions. Why should we inspect
   per-client utility in addition to global accuracy? Also explain why one
   fixed seed gives repeatability but not statistical equivalence.

   **Answer:**
   Global accuracy can hide differences between clients. Even if the global model has good accuracy, one client may perform much worse than the others, especially when client data distributions differ. Therefore, the experiment should report per-client accuracy or other client-level utility metrics in addition to global accuracy. A fixed random seed makes the experiment repeatable under the same configuration, but it does not show that the result would be the same across different seeds. One seed provides repeatability, not statistical evidence that the observed result is representative of the broader experiment distribution.

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

   **Proposed changes and prediction:**

   - Which JSON fields and names will you change?
     federated-unlearning/configs/month2_week8_mnist_iid_fedavg_vs_centralized.json
     "number_of_clients": 5 -> 10
   - How many training examples should each equal-sized client receive? 
   51,000 / 10 = 5,100
   - How many clients participate per round?
   10 × 1 = 10
   - Predict the dense communication bytes using the formula in Question 7.
   canonical communication is 20,354,000 bytes, so doubling the participating clients from 5 to 10 should double the model-payload communication:
   20,354,000 × (10 / 5) = 40,708,000 bytes
   - Predict whether the total local optimizer-step count remains the same or
     changes, and explain the minibatch arithmetic.
     remains 2,000, not increases. Each client gets half as many examples (5,100 instead of 10,200), so each client needs half as many batches (40 instead of 80). But there are twice as many clients, so:
   10 × 40 × 5 = 2,000.

   **Saved run/result after review:**

10. **Change local epochs.** Propose an `E=2` variant with `K=5`, `C=1`,
    `B=128`, and `R=5`.

    **Proposed changes and prediction:**

    - Which JSON fields and names will you change?
         federated-unlearning/configs/month2_week8_mnist_iid_fedavg_vs_centralized.json
      "local_epochs": 1 -> 2
    - Why must `centralized_epochs` become `R × E` for the matched-budget
      comparison?
      This keeps the centralized and federated procedures matched in training-example exposures. Otherwise the centralized model would receive only half as many passes through the training set, making the comparison unfair.
    - Predict the training-example exposures and optimizer-step totals.
    K remains 5, so each client still has 10,200 examples.
    With batch size 128, this gives ceil(10,200 / 128) = 80 minibatches per epoch. 
    With two local epochs, each client performs 160 optimizer steps per round, so FedAvg performs 5 × 160 × 5 = 4,000 local optimizer steps in total.
    The centralized run is expected to have 3,990 optimizer steps because ceil(51,000 / 128) = 399 steps per epoch and 399 × 10 = 3,990.
    - Predict whether communication bytes change when `K` and `R` stay fixed.
    Communication should remain 20,354,000 dense model-payload bytes because the number of clients, participating clients, model size, and communication rounds are unchanged.
    - After the run, compare its validation curve with the canonical `E=1`
      curve without claiming that one seed proves a general rule.

    **Saved run/result after review:**

## Pre-run proposal review

This review happens after the two predictions are written but before either
config is created, tagged, or executed. Its committed marker makes that order
auditable: the producing Git tag must already contain this approval.

**Review date:** 2026-09-07

**Result:** APPROVED TO RUN

**Reviewer notes:** Both proposals change only the intended field
(`number_of_clients` for K=10; `local_epochs`/`centralized_epochs` for E=2),
keep every other field identical to the canonical config, and use new
`experiment_name`/`output_subdirectory`/`code_revision` values that do not
alias the canonical run. The K=10 arithmetic is correct: 5,100 examples per
client, 10 participating clients, 40,708,000 predicted communication bytes,
and 2,000 total local steps with the minibatch reasoning (40 steps x 10
clients x 5 rounds) spelled out. The E=2 arithmetic is correct: matched-budget
reasoning for `centralized_epochs = R x E = 10` is explained, 4,000 predicted
FedAvg local steps and 3,990 predicted centralized steps both check against
the runner's step formula, and unchanged 20,354,000 communication bytes is
correctly attributed to `K`/`R` staying fixed. One precision note, not
blocking: the E=2 answer does not state the training-example-exposures number
itself (510,000 for both methods, from `51,000 x 10`) even though the
step-count numbers it does give are consistent with that value — add it when
convenient, no re-approval needed. Approved to create and run both tagged
configs.

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
