# Month 3, Week 12 — FedAvg vs. FedProx Benchmark

## Why this week exists

This is the plan's Gate 3 evidence (plan §9, Week 12 table): one benchmark
comparing FedAvg and FedProx across the same four data distributions Weeks
9-10 already characterized for FedAvg alone. FedProx uses a single fixed
`mu=0.01` — the least-detrimental value found in Week 11's sweep at
`alpha=0.1` — across all four settings, rather than re-tuning `mu` per
setting, which would make the comparison about hyperparameter search
instead of about the method.

## The benchmark table

| Setting | FedAvg accuracy | FedProx accuracy (μ=0.01) | Rounds | FedProx time | Communication |
|---|---:|---:|---:|---:|---:|
| IID | 90.99% | 90.88% | 5 | 4.50s | 20,354,000 bytes |
| Dirichlet α=1.0 | 90.59% | 90.54% | 5 | 4.57s | 20,354,000 bytes |
| Dirichlet α=0.5 | 89.83% | 89.72% | 5 | 4.53s | 20,354,000 bytes |
| Dirichlet α=0.1 | 71.60% | 71.43% | 5 | — (see Week 11) | 20,354,000 bytes |

FedAvg numbers are cited from already-verified evidence, not rerun: IID from
Week 8/9 (`results/month2_week8_mnist_iid_comparison`), and the three
Dirichlet settings from Week 10. Communication is identical across every
row because it depends only on model size, client count, and rounds — none
of which differ between FedAvg and FedProx, or across settings. FedProx
training time is a single-process CPU simulation figure, not a
distributed-network measurement, consistent with every other timing number
in this project.

## Reading the result

**FedProx (μ=0.01) is very slightly below FedAvg at every one of the four
settings**, including IID and the two mildest heterogeneity levels where
client drift should be smallest. The gap is small (0.11-0.17 percentage
points at three of four settings) but consistently in FedAvg's favor, not
mixed. This is the same direction Week 11 already found at α=0.1 across a
wider μ sweep, now shown to hold across the full heterogeneity range at the
one μ value carried forward.

This is a coherent, honestly reported negative result for FedProx *in this
project's specific setup* — not a contradiction of Li et al. (2020). Their
paper's own clearest reported gains come from two conditions this benchmark
does not create:

1. **Systems heterogeneity** — clients contributing genuinely different
   amounts of local work (stragglers, variable epochs). Every client here
   always completes the same `E=1` local epoch.
2. **Longer training horizons.** Every setting here trains for the same 5
   communication rounds used throughout Months 2-3. FedProx's proximal term
   trades faster early adaptation for reduced drift; over a short horizon,
   that trade can show up as a small accuracy cost before any drift-control
   benefit would accumulate.

## What this does show

- Statistical heterogeneity alone (label skew, no systems heterogeneity)
  degrades FedAvg sharply as α falls (90.99% → 71.60%), replicated across
  Weeks 9-10 and unchanged by this benchmark.
- Adding FedProx's proximal term at a small, literature-plausible μ neither
  helps nor meaningfully hurts under mild heterogeneity (IID, α=1.0, α=0.5:
  within 0.15pp of FedAvg), and does not recover FedAvg's severe-heterogeneity
  drop at α=0.1 either.
- Communication cost is unaffected by which of the two algorithms is used,
  since both aggregate the same dense model state the same number of times.

## What this does not show

- **Not proof FedProx never helps.** μ was fixed at one value across all
  four settings deliberately, and systems heterogeneity was never
  introduced — Li et al.'s clearest reported gains come from exactly that
  condition.
- **Not multi-seed.** Every number is one fixed-seed run, per this
  project's standing practice.
- **Not the final thesis benchmark.** This closes Gate 3 (Month 3's
  required FedAvg/FedProx/IID/Non-IID evidence); Month 6's final benchmark
  will be broader (more datasets, client counts, and settings per plan §12).

## Reproduce

```bash
python experiments/noniid/month3_week12_fedprox_benchmark.py --config configs/month3_week12_iid_fedprox.json
python experiments/noniid/month3_week12_fedprox_benchmark.py --config configs/month3_week12_alpha1.0_fedprox.json
python experiments/noniid/month3_week12_fedprox_benchmark.py --config configs/month3_week12_alpha0.5_fedprox.json
```

The α=0.1 FedProx (μ=0.01) row reuses Week 11's saved result
(`results/month3_week11_mnist_fedprox_mu_sweep/metrics.json`,
`fedprox["0.01"]`) rather than rerunning it, since nothing about that
setting changed.
