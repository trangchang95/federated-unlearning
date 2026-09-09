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

2. What is **client drift**? Trace it through one FedAvg round: which step
   causes client models to move apart, and which step pulls them back
   together?

   **Answer:**

3. The evidence card shows FedAvg falling from 90.99% (IID) to 71.60%
   (α=0.1). Explain *why* Non-IID data causes this drop — connect your
   answer to what each client's local gradient is actually pointing toward
   when its data is skewed toward one or two classes.

   **Answer:**

4. Week 10 found the α-vs-accuracy relationship is *nonlinear* (α=1.0 and
   α=0.5 stay close to IID; α=0.1 drops sharply). Propose one reason a
   moderate label skew might be nearly harmless while a severe one is not.

   **Answer:**

5. What specific problem does FedProx's proximal term
   `(mu/2)*||w - w_global||^2` target? Explain what happens to a client's
   local loss as its local model drifts further from the global model,
   and why that discourages (but does not prevent) drift.

   **Answer:**

6. The benchmark shows FedProx (μ=0.01) *slightly below* FedAvg at every
   setting, including IID. Explain why this is not evidence that FedProx is
   a bad method in general — what two conditions does Li et al. (2020)'s
   paper report its clearest gains under, and which of those does this
   project's protocol (`E=1`, `R=5`, uniform local epochs) *not* create?

   **Answer:**

7. If you were to design one additional experiment most likely to make
   FedProx outperform FedAvg, what single setting would you change, and
   why would you expect that specific change to help?

   **Answer:**

8. Communication cost is identical for FedAvg and FedProx at every setting
   in the benchmark (20,354,000 bytes). Explain why algorithm choice does
   not affect this number — what does it actually depend on?

   **Answer:**

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
