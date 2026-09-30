# Gate 3 Self-Check — Non-IID Federated Learning

**Current status:** awaiting the student's answers. Weeks 9-12's technical
experiments have passed verification, but Gate 3 is not closed merely
because the code ran successfully — the plan (§9, §15) requires the student
to explain statistical heterogeneity, client drift, why Non-IID hurts FL,
and what problem FedProx addresses.

Read
[`month3_week9_iid_vs_noniid.md`](month3_week9_iid_vs_noniid.md),
[`month3_week10_dirichlet_benchmark.md`](month3_week10_dirichlet_benchmark.md),
[`month3_week11_fedprox.md`](month3_week11_fedprox.md), and
[`month3_week12_benchmark.md`](month3_week12_benchmark.md) before answering.
Use your own words. Each answer must explain *why*, not only repeat a
definition.

## Evidence card

| Setting | FedAvg accuracy | FedProx accuracy (μ=0.01) |
|---|---:|---:|
| IID | 90.99% | 90.88% |
| Dirichlet α=1.0 | 90.59% | 90.54% |
| Dirichlet α=0.5 | 89.83% | 89.72% |
| Dirichlet α=0.1 | 71.60% | 71.43% |
| Pathological 2-shard (Week 9) | 64.01% | — |

All rows share `K=5, C=1.0, E=1, B=128, R=5`, the same MNIST split, and the
same initial weights. FedProx's `mu=0` is verified by test to reproduce
FedAvg exactly, so any FedProx-vs-FedAvg difference above comes only from
`mu=0.01`, not a divergent reimplementation.

## Part A — Explain the concepts

1. In your own words, what is **statistical heterogeneity**, and how is it
   different from **systems heterogeneity**? Which one did every experiment
   in Weeks 9-12 actually vary?

   **Answer:**
Statistical Heterogeneity: Refers to non-identical data distributions across clients, such as label skew (where certain digit classes are over- or under-represented on different devices) or feature skew.Systems Heterogeneity: Arises from differences in hardware capabilities, network connectivity, compute power, or variable local computation budgets (such as clients running different numbers of local epochs or stragglers submitting partial work).Which was varied in Weeks 9–12: Every experiment in Weeks 9–12 specifically varied statistical heterogeneity (by testing pathological 2-shard and Dirichlet $\alpha$ partitions).Workload Note: While parameters governing systems heterogeneity were held uniform (every client was configured with $E=1$ or $E=5$ local epochs), under Dirichlet partitioning at $\alpha=0.1$, client dataset sizes varied substantially (e.g., Client 0 had 6,131 samples while Client 3 had 14,848 samples). Consequently, clients executed different total numbers of local SGD minibatch steps per round, even though statistical heterogeneity was the single controlled variable.

2. What is **client drift**? Trace it through one FedAvg round: which step
   causes client models to move apart, and which step pulls them back
   together?

   **Answer:**
Client Drift: The phenomenon where individual client models diverge toward narrow local optima during local training because each client optimizes its parameters on its own skewed local data distribution rather than the global population distribution.Step that causes models to move apart: Local SGD optimization. When selected clients receive the global model $w_t$ and perform local SGD steps on their private datasets, their local gradients point toward minimizing their respective local loss functions. Because each client's dataset is statistically skewed, these local updates push local parameter vectors in conflicting directions.Step that pulls them back together: Server aggregation. After local training completes, clients send their updated weights back to the central server. The server calculates a sample-weighted average of all returned model parameters to construct the new global model $w_{t+1}$, pulling the divergent local models back into a single aggregate state.

3. The evidence card shows FedAvg falling from 90.99% (IID) to 71.60%
   (α=0.1). Explain *why* Non-IID data causes this drop — connect your
   answer to what each client's local gradient is actually pointing toward
   when its data is skewed toward one or two classes.

   **Answer:**
Under severe label skew ($\alpha=0.1$), a client's local dataset is heavily dominated by one or two classes while other classes are nearly absent.Logit Suppression via Softmax Cross-Entropy: Under softmax cross-entropy loss, missing or under-represented classes do produce a gradient signal—specifically, one that actively suppresses (drives down) the output logits for those unrepresented classes while boosting the logits for the local majority classes. As a result, local SGD shapes parameter weights to suppress predictions for missing classes.Conflicting Local Trajectories: Because different clients possess different dominant classes, their local gradients point in opposing directions (e.g., Client A suppresses class X while Client B boosts class X).Impact on Aggregation & Horizon: When the server averages these local models, combining mutually conflicting parameter states distorts global decision boundaries. Furthermore, the recorded 71.60% test accuracy represents performance at round 5 ($R=5$) while accuracy was still actively rising (progressing from 67.6% in earlier rounds), rather than an asymptotic plateau.

4. Week 10 found the α-vs-accuracy relationship is *nonlinear* (α=1.0 and
   α=0.5 stay close to IID; α=0.1 drops sharply). Propose one reason a
   moderate label skew might be nearly harmless while a severe one is not.

   **Answer:**
A moderate label skew can be nearly harmless because each client may still see enough different classes to learn updates that are broadly aligned with the global model. With severe skew, clients see much narrower label distributions, so their local updates move in very different directions (client drift); averaging those conflicting updates can hurt the global model much more.
This is consistent with Week 10: α=1.0 and α=0.5 stayed close to IID, while α=0.1 caused a much larger accuracy drop

6. The benchmark shows FedProx (μ=0.01) *slightly below* FedAvg at every
   setting, including IID. Explain why this is not evidence that FedProx is
   a bad method in general — what two conditions does Li et al. (2020)'s
   paper report its clearest gains under, and which of those does this
   project's protocol (`E=1`, `R=5`, uniform local epochs) *not* create?

   **Answer:**
Why FedProx underperformed at $\alpha=0.1$ and lost further at $E=5$:Under extreme label skew ($\alpha=0.1$), clients need significant parameter movement to adapt to local data. Penalizing movement away from $w_{global}$ with $\mu > 0$ acts as an artificial drag on necessary local learning.Increasing local computation to $E=5$ compounds this issue: more local minibatch steps accumulate a larger total proximal penalty, creating a stronger drag that widens the gap between FedProx and FedAvg (at $E=1$, $\mu=0.01$ lags by 0.17 pp; at $E=5$, $\mu=0.01$ lags by 0.95 pp and $\mu=1.0$ lags by 18.98 pp).The 5-Round Horizon Limit:The $R=5$ round limit is a short-horizon limitation of our own experimental setup, not a claim from Li et al. (2020). Over 5 early rounds, weight movement reflects essential initial learning; penalizing this movement prematurely hurts accuracy before drift-control benefits can accumulate over longer horizons.Two Conditions for Clearest Gains in Li et al. (2020):Systems Heterogeneity: Environments where clients perform variable amounts of local work due to hardware stragglers or variable local epochs.High Statistical Heterogeneity: Where unconstrained local SGD leads to divergence.Our project protocol enforced uniform local epochs ($E=1$ or $E=5$) across all clients, eliminating systems heterogeneity—the primary scenario where FedProx's design excels.

7. If you were to design one additional experiment most likely to make
   FedProx outperform FedAvg, what single setting would you change, and
   why would you expect that specific change to help?

   **Answer:**
Committed Baseline: A Variable-Epoch FedAvg baseline, where local epochs vary across participating clients (e.g., stragglers or low-power clients run $E_k = 1$, while faster clients run $E_k = 5$ or $E_k = 10$).Why variable local epochs differ from Week 11's uniform $E=5$:In Week 11's uniform $E=5$ setup, all clients executed the exact same number of local epochs and submitted complete updates, allowing standard FedAvg aggregation to function normally.In a variable-epoch setup ($E_k$), clients running more epochs ($E_k=10$) drift much further and over-influence the global model, while clients running fewer epochs ($E_k=1$) contribute under-developed updates, leading to severe update imbalance in standard FedAvg.FedProx's proximal term directly addresses this by restricting local drift regardless of how many local steps a client executes. It prevents fast clients from drifting into incompatible parameter regions while allowing slow clients to contribute safe partial updates. Thus, FedProx is expected to outperform the Variable-Epoch FedAvg baseline under systems heterogeneity.

8. Communication cost is identical for FedAvg and FedProx at every setting
   in the benchmark (20,354,000 bytes). Explain why algorithm choice does
   not affect this number — what does it actually depend on?

   **Answer:**
Why algorithm choice does not affect communication cost: FedProx modifies only the client-side local objective function by adding a scalar regularization penalty during loss computation. It does not alter the model architecture, parameter count, tensor shapes, or data precision. In both FedAvg and FedProx, the server downloads and uploads the exact same dense parameter tensor to each selected client in every round.What communication cost actually depends on:Model payload size ($S$): Total parameter elements multiplied by bytes per element (e.g., 4 bytes per float32 parameter).Client participation per round ($m$): $m = C \times K$ (here, $1.0 \times 5 = 5$ clients).Communication rounds ($R$): Total global rounds (here, $R=5$).Formula: $\text{Total Bytes} = 2 \times m \times S \times R$ (accounting for 1 download and 1 upload per selected client per round).

## Passing standard

Gate 3 passes only when the student can:

- correctly distinguish statistical from systems heterogeneity, and
  correctly state which one this project's experiments varied;
- explain client drift mechanistically (local gradient direction under
  skewed data vs. server aggregation), not just name it;
- connect the measured nonlinear α-vs-accuracy relationship to a plausible
  mechanism, not just restate the numbers;
- explain what FedProx's proximal term does and why it does not prevent
  drift outright;
- correctly interpret FedProx's negative result here as scope-specific
  (short horizon, no systems heterogeneity) rather than either dismissing
  FedProx or overstating what this benchmark disproves;
- avoid claims about privacy, unlearning, or statistical significance across
  seeds that these experiments did not test.

## Review result

**Review date:** NOT YET REVIEWED

**Result:** NOT YET REVIEWED

**Reviewer notes:** The reviewer will assess the explanations for
conceptual correctness against the passing standard above. This field is
not an automated substitute for that review.

Weeks 9-12 are technically complete and verified. Gate 3 remains open until
the answers above are reviewed. Month 4 (Machine Unlearning, SISA, full
retraining, FedEraser, KD-based FU) remains blocked until then.
