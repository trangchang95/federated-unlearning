# Month 3, Week 9 — IID vs. Non-IID Federated Learning

## Why this week exists

Every FL experiment so far (Weeks 6-8) used IID clients: each client's local
data looked like a small random sample of the whole dataset. Real federated
deployments rarely look like that — a hospital's patient records, a phone's
photo library, or a keyboard's typed words reflect *that* client's particular
situation, not the population average. Week 9 asks a concrete question before
any new algorithm is introduced: **what happens to plain FedAvg when clients
stop being interchangeable random samples?**

## Two vocabulary terms from Kairouz et al. (2019/2021)

- **Statistical heterogeneity**: clients hold different data *distributions*
  — some digits, words, or diagnoses are over- or under-represented on some
  clients relative to others. This is what this week's experiment tests.
- **Systems heterogeneity**: clients differ in compute power, connectivity,
  and availability. This thesis does not simulate that axis; every client
  here runs the same number of local epochs on the same hardware.

Keeping these separate matters: a result caused by statistical heterogeneity
should not be blamed on stragglers or dropped connections, and vice versa.

## What was built

`clients/noniid_partition.py` implements the classic **pathological Non-IID**
scheme from McMahan et al. (2017, Section 3): sort all 51,000 training
examples by digit label, cut the sorted sequence into equal shards, then give
each of the 5 clients 2 shards from a seeded shuffle of shard order. Because
MNIST's ten classes are nearly balanced, a shard is almost always dominated
by one label, so each client ends up with mostly 1-2 digits instead of a
representative mix.

`experiments/noniid/month3_week9_mnist_noniid_fedavg.py` trains hand-written
FedAvg twice from **the same initial weights** and the same protocol as the
Week 8 canonical config (`K=5`, `C=1`, `E=1`, `B=128`, `R=5`, plain SGD,
`lr=0.1`) — the only thing that changes between the two runs is how the
51,000 training examples are split across the 5 clients.

## Client class distributions (the actual saved partitions)

| Client | IID class histogram (digits 0-9) | Non-IID class histogram (digits 0-9) |
|---|---|---|
| 0 | ~1,000 examples per digit | 4,969 fives + 5,064 sixes, ~0 elsewhere |
| 1 | ~1,000 examples per digit | 5,100 ones + 4,463 fours + 637 threes |
| 2 | ~1,000 examples per digit | 4,646 fives + 5,036 sixes, ~0 elsewhere |
| 3 | ~1,000 examples per digit | 5,037 zeros + 5,100 sevens, ~0 elsewhere |
| 4 | ~1,000 examples per digit | 5,067 twos + 4,572 threes, ~0 elsewhere |

(Digit indices above follow the saved `class_histogram` arrays in
`results/month3_week9_mnist_iid_vs_noniid/metrics.json`; see
`class_distribution.png` for the visual version of this table.) Every IID
client's bars are roughly flat across all ten digits — a small random sample
of the whole dataset. Every Non-IID client's bars spike on one or two digits
and are near zero everywhere else — visible statistical heterogeneity, not a
description we have to take on faith.

## Measured result

| Partition | Test accuracy | Test macro F1 | Communication bytes |
|---|---:|---:|---:|
| IID | 90.99% | 90.86% | 20,354,000 |
| Non-IID (2 shards/client) | 64.01% | 57.80% | 20,354,000 |

The IID number here is not a new measurement — it reproduces Week 8's
canonical hand-written FedAvg result exactly (90.99%/90.86%), which is
expected: same seed, same initial weights, same IID partition function, same
hyperparameters. That exact match is itself a sanity check that this new
runner is wired correctly. Communication bytes are identical between the two
rows because they depend only on model size, client count, and rounds, none
of which changed — only *which* 5,100-ish examples each client received
changed.

## Why Non-IID hurts FedAvg here: client drift

Every selected client starts each round from the same global model, but then
trains only on its own narrow slice of digits. A client that has seen almost
no examples of "3" will happily push its local weights toward whatever makes
its own 2 digits easiest to separate, with no signal at all about the other 8
classes. When the server averages five such locally-drifted models, the
result is a compromise between five very different local optima rather than
five noisy estimates of the same optimum — this divergence between local
update directions is what "client drift" means. The FedAvg paper's own
experiments show the same pattern: pathological Non-IID splits need more
rounds, and sometimes plateau at a noticeably lower accuracy than IID splits
under an identical budget, which is exactly what the confusion matrices in
`noniid_confusion_matrix.png` versus `iid_confusion_matrix.png` show directly
— the Non-IID model confuses far more digit pairs.

## What this result does not show

- **Not a Dirichlet sweep.** This is one Non-IID severity (2 shards/client),
  a pathological extreme from the original FedAvg paper. Week 10 introduces
  Dirichlet(α) partitioning, which can produce milder, more realistic
  heterogeneity at α=1.0/0.5/0.1.
- **Not a FedProx comparison.** FedProx (Week 11) adds a proximal term meant
  specifically to reduce client drift under heterogeneity; this week only
  shows that plain FedAvg is vulnerable, not how much a fix would help.
- **Not a general claim about "2 shards".** One seed, one architecture, one
  shard count. The direction (Non-IID hurts) is well-established in the
  literature; the exact 26.98-point gap is specific to this configuration.

## Gate 3 relevance

This experiment gives concrete, measured grounding for two of Gate 3's
required explanations: what statistical heterogeneity looks like in saved
data (the class-histogram table above), and why it produces client drift
(the divergence argument above, backed by the accuracy and confusion-matrix
evidence). The remaining Gate 3 requirements — Dirichlet-controlled severity
and what FedProx does about it — are Weeks 10 and 11.
