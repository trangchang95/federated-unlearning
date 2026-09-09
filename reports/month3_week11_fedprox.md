# Month 3, Week 11 — FedProx

## Why this week exists

Weeks 9-10 measured FedAvg's accuracy falling as data heterogeneity rose.
Li et al. (2020)'s FedProx adds a proximal term specifically to limit the
client drift responsible for that fall. Week 11 asks the direct question:
**at the heterogeneity level where FedAvg struggled most, does FedProx
actually help?**

## What was built

`clients/fedprox_client.py` and `server/fedprox_server.py` add FedProx as a
strict generalization of the existing hand-written FedAvg: the local loss
becomes `task_loss + (mu/2) * ||w - w_global||^2`, and `mu=0` is verified
(by test) to reproduce plain FedAvg's local training and full server round
*exactly* — same weights, same loss, same optimizer-step count. This means
any difference measured below comes only from `mu`, not from a
reimplementation drifting from the original FedAvg semantics.

`experiments/noniid/month3_week11_mnist_fedprox.py` reuses Week 10's exact
Dirichlet(α=0.1) partition, seeds, and initial weights — the most severe
realistic heterogeneity measured so far (FedAvg: 71.60%) — and sweeps
`mu ∈ {0.01, 0.1, 1.0}`, training FedAvg (`mu=0`) in the same run as the
reference point.

## Measured result

### E=1 (the Week 8-10 canonical local-epoch count)

| Method | Test accuracy | Gap vs. FedAvg |
|---|---:|---:|
| FedAvg (μ=0) | 71.60% | — |
| FedProx (μ=0.01) | 71.43% | −0.17pp |
| FedProx (μ=0.1) | 69.65% | −1.95pp |
| FedProx (μ=1.0) | 60.15% | −11.45pp |

### E=5 (extra check: does more local computation change the answer?)

| Method | Test accuracy | Gap vs. FedAvg |
|---|---:|---:|
| FedAvg (μ=0) | 76.28% | — |
| FedProx (μ=0.01) | 75.33% | −0.95pp |
| FedProx (μ=0.1) | 69.65% | −6.63pp |
| FedProx (μ=1.0) | 57.30% | −18.98pp |

**FedProx did not beat FedAvg at any swept μ, in either local-epoch
setting.** Accuracy falls monotonically as μ rises in both cases. This is
reported plainly rather than searching for a μ that would flip the result;
the plan's own risk-management guidance is explicit that a negative result
must remain reportable, not hidden.

## Why this is a real, informative negative result — not a bug

Three checks before trusting this negative result:

1. **Aggregation and mu=0 equivalence are tested**, not assumed (five
   synthetic tests, including a full server-round comparison with FedAvg).
   A silent implementation bug would need to coincidentally reproduce exact
   equality at μ=0 and only diverge at μ>0 in a *directionally consistent*
   way across two different local-epoch settings — implausible.
2. **The round-by-round histories differ between E=1 and E=5** at every μ
   (checked directly against saved `metrics.json`), so the two experiments
   are not accidentally reusing cached results.
3. One coincidence worth naming rather than hiding: μ=0.1's final-round
   accuracy is 69.65% in *both* E=1 and E=5 — verified as a genuine
   coincidence in the underlying round-by-round trajectories (which differ
   throughout rounds 1-4 and only meet at round 5), not a copy-paste error.

## Plausible explanation

FedProx's own paper reports its clearest gains under **systems
heterogeneity** — clients contributing genuinely different amounts of local
work (stragglers, variable epochs) — and over longer training horizons.
This experiment instead uses:

- a **short horizon** (5 communication rounds), where early updates are
  mostly legitimate, necessary learning rather than harmful drift, so a term
  that penalizes *any* movement away from the initial global model has more
  cost than benefit early on;
- **uniform** local epochs across all clients (no systems heterogeneity),
  removing exactly the condition FedProx's partial-work tolerance targets;
- a **small model and easy task** (SimpleMLP on MNIST), where FedAvg's plain
  aggregation may already handle the tested heterogeneity adequately within
  5 rounds, leaving little drift for the proximal term to usefully correct.

None of this contradicts Li et al. (2020) — it narrows *which* of FedProx's
claimed benefits this project's specific setup would need to reproduce.

## What this does not show

- **Not proof FedProx never helps.** One model, one dataset, one partition,
  one seed, a short 5-round budget, and uniform local epochs. Systems
  heterogeneity (varying local work per client) was not tested at all.
- **Not a fully tuned μ search.** Three values were swept; accuracy fell
  monotonically across all three, so there is no evidence a smaller μ than
  0.01 would exceed FedAvg either, but the search was not exhaustive.
- **Not the Gate 3 benchmark.** Week 12 assembles the full FedAvg-vs-FedProx
  × α table; this negative finding is itself one input to that table, not a
  reason to exclude FedProx from it.

## Reproduce

```bash
python experiments/noniid/month3_week11_mnist_fedprox.py --config configs/month3_week11_mnist_fedprox_mu_sweep.json
python experiments/noniid/month3_week11_mnist_fedprox.py --config configs/month3_week11_mnist_fedprox_mu_sweep_e5.json
```
