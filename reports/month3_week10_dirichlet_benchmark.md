# Month 3, Week 10 — Dirichlet(α) Partitioning

## Why this week exists

Week 9 showed that Non-IID data hurts FedAvg using one pathological extreme
(each client dominated by exactly 1-2 digits). Real deployments are rarely
that extreme or that uniform — heterogeneity comes in degrees. Week 10
introduces **Dirichlet(α) label-skew partitioning** (Hsu, Qi & Brown, 2019),
which lets severity be dialed with a single number instead of being an
all-or-nothing choice.

## How the partition works

`clients/dirichlet_partition.py`: for each of MNIST's ten digit classes
independently, draw a proportion vector from `Dirichlet(α, α, ..., α)` over
the 5 clients, then split that class's examples across clients according to
those proportions (largest-remainder rounding keeps the split exact). A
large α makes every class's proportions close to uniform across clients
(nearly IID); a small α concentrates each class onto very few clients. Unlike
Week 9's shard scheme, the same code produces every severity level just by
changing one number.

Three configs were run — `alpha=1.0`, `alpha=0.5`, `alpha=0.1` — each
training hand-written FedAvg and a centralized SGD baseline from **identical
initial weights**, on MNIST's same 51,000-example split, under the same
`K=5, C=1, E=1, B=128, R=5`, SGD `lr=0.1` protocol as the Week 8 canonical
run and Week 9. Only the client partition changes between configs.

## Measured result

| Setting | Centralized | FedAvg | Gap (pp) | Client training-set sizes |
|---|---:|---:|---:|---|
| IID (Week 8/9 canonical) | 94.76% | 90.99% | −3.77 | ~10,200 each |
| Dirichlet α=1.0 | 94.76% | 90.59% | −4.17 | 8,874 / 12,895 / 6,437 / 10,879 / 11,915 |
| Dirichlet α=0.5 | 94.76% | 89.83% | −4.93 | 9,228 / 8,359 / 7,440 / 16,714 / 9,259 |
| Dirichlet α=0.1 | 94.76% | 71.60% | −23.16 | 6,131 / 11,200 / 12,025 / 14,848 / 6,796 |
| Pathological 2-shard (Week 9) | 94.76% | 64.01% | −30.75 | ~10,200 each (label-sorted, not random) |

The centralized number is identical across every row (94.76%) — it is
retrained fresh each time from the same weights and data for a
self-contained comparison, but it does not depend on the client partition at
all, so an unchanged number here is the expected sanity check, not a
coincidence. FedAvg communication is identical across all three Dirichlet
runs (20,354,000 bytes) for the same reason communication was unchanged
across Week 8's `E=2` variant: it depends only on model size, client count,
and rounds, none of which changed.

## Reading the severity gradient

FedAvg's accuracy falls as α falls, exactly as the literature predicts, but
**not linearly**: α=1.0 and α=0.5 are both close to the IID number (within
about 1.4 points), while α=0.1 causes a much larger 19-point drop from α=0.5.
This matches Hsu et al.'s own finding that Dirichlet skew's effect on
accuracy is highly nonlinear in α — moderate heterogeneity is often nearly
harmless for FedAvg, while severe heterogeneity is not. Note also that
Dirichlet also skews client *quantity*, not just label mix (client training
sizes above range from about 6,100 to 16,700 examples even though every
class has roughly the same total count) — this is a real, literature-typical
side effect of the per-class Dirichlet draw, not a bug, and it compounds with
label skew rather than replacing it.

Placing Week 9's pathological shard result in the same table is intentional:
at α=0.1, Dirichlet skew (71.60%) is meaningfully less severe than the
2-shard scheme (64.01%), even though both are commonly called "severe
Non-IID" informally. This is a useful, measured reminder that "Non-IID"
names a spectrum, not one condition — a claim later chapters can point back
to with saved numbers instead of restating it as an assertion.

## What this does not show

- **Not a FedProx result.** Week 11 introduces FedProx specifically to
  address the drift this table documents; nothing here evaluates whether it
  helps.
- **Not multi-seed.** Every number above is one fixed-seed run, per this
  project's standing practice — the qualitative ordering (IID > mild
  Dirichlet > severe Dirichlet > pathological shards) is the literature-
  consistent finding; the exact percentage-point gaps are specific to this
  seed, architecture, and client count.
- **Not the Gate 3 benchmark.** Week 12 assembles the full FedAvg-vs-FedProx
  × α table that closes Gate 3; this week only establishes the FedAvg side
  of that table.

## Reproduce

```bash
python experiments/noniid/month3_week10_mnist_dirichlet_fedavg.py --config configs/month3_week10_mnist_alpha1.0_fedavg.json
python experiments/noniid/month3_week10_mnist_dirichlet_fedavg.py --config configs/month3_week10_mnist_alpha0.5_fedavg.json
python experiments/noniid/month3_week10_mnist_dirichlet_fedavg.py --config configs/month3_week10_mnist_alpha0.1_fedavg.json
```
