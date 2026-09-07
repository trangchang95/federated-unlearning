# Month 2, Week 8 — Centralized SGD vs Hand-Written FedAvg

> Generated deterministically from [`results/month2_week8_mnist_e2_variant/metrics.json`](../results/month2_week8_mnist_e2_variant/metrics.json). Do not copy result numbers into this report by hand.

## How to read the evidence

- **Saved-result fact:** a value stored by the config-driven experiment.
- **Derived fact:** arithmetic recomputed by this builder from saved values.
- **Interpretation:** a deliberately limited explanation, not a new measurement.
- **Limitation:** what this run cannot establish.

## 1. Run identity and integrity

**Saved-result facts**

| Item | Recorded value |
|---|---|
| Dataset | `MNIST` — 51,000 train / 9,000 validation / 10,000 test examples |
| Config | [`configs/month2_week8_mnist_e2_fedavg_vs_centralized.json`](../configs/month2_week8_mnist_e2_fedavg_vs_centralized.json) |
| Resolved run config | [`results/month2_week8_mnist_e2_variant/resolved_config.json`](../results/month2_week8_mnist_e2_variant/resolved_config.json) |
| Configured code revision | `month2-week8-e2` |
| Resolved Git commit / clean HEAD | `1315eea929972d2844173e5984588e5638226646` / `True` |
| Environment manifest | [`environment/month2_cpu_runtime.json`](../environment/month2_cpu_runtime.json) |
| Resolved-config SHA-256 | `1fdd67ab9776c713a7e2da49d3bb6af700ef5d2510dd25d82838b7b3f81ed363` |
| Metrics SHA-256 | `ea68277ae025592a50e7decffcffe59f20bef43a73a95f7e6022f0884a728f79` |
| Common initial-model SHA-256 | `f1a6fbfc1590ec7bfef00d6df509b9d5b4eee93b4814397af11b6f4f223d2125` |
| Partition | `iid_seeded_equal_size` |
| Fixed seeds | `random_seed=42`, `split_seed=42`, `train_partition_seed=1042`, `test_partition_seed=2042`, `centralized_loader_seed=3042`, `model_initialization_seed=42` |

**Derived integrity result:** PASS. Before rendering, the builder checked that every resolved input setting is unchanged in the metrics (only the declared `accuracy`, `f1`, and `communication_cost` output placeholders may be filled), the configured Git revision resolved to a clean HEAD, the mandatory top-level metrics belong to `handwritten_fedavg`, the initialization was common, data-pass counts matched, exactly one validation checkpoint was selected per method, optimizer-step arithmetic was correct, per-client test data covered the test set, and every round's upload/download byte totals agreed. A mismatch makes the builder stop instead of producing a report.

## 2. Beginner protocol card: K, C, E, B, R

**Saved/derived facts**

| Symbol | Meaning | This run |
|---|---|---:|
| `K` | Total number of clients | 5 |
| `C` | Fraction of clients selected each round | 1.00 (5 of 5 selected) |
| `E` | Local epochs completed by each selected client before upload | 2 |
| `B` | Local/central minibatch size | 128 |
| `R` | FedAvg communication rounds | 5 |
| `η` | SGD learning rate | 0.1 |
| Central epochs | Full centralized passes through the training split | 10 |

With full participation (`C=1`), `R × E = 5 × 2 = 10` FedAvg passes over all client training examples. This matches the 10 centralized passes.

## 3. What is matched, and what is still different

**Saved/derived facts**

| Matched on purpose | Evidence |
|---|---|
| Data | Same MNIST training split; each path processed 510,000 training-example exposures |
| Starting model | All three recorded initialization checksums equal `f1a6fbfc1590ec7bfef00d6df509b9d5b4eee93b4814397af11b6f4f223d2125` |
| Optimization settings | `SGD`, learning rate 0.1, batch size 128 |
| Training budget | 10 centralized full passes vs 10 FedAvg full passes |
| Model selection | Each method's best validation checkpoint was selected before the test set was evaluated |

| Mechanically different | Why it matters |
|---|---|
| Centralized SGD | One model processes the complete shuffled training split and updates after every central minibatch. |
| FedAvg | 5 IID client shards train separate local copies; the server performs a sample-weighted model average once per round. |
| Minibatch boundaries | Splitting 51,000 examples across clients changes where partial final batches occur, even though total example exposures match. |
| Communication | Centralized simulation records 0 model-transfer bytes; FedAvg counts one download and one upload per selected client per round. |
| Data order and update trajectory | `Centralized SGD follows one continuous trajectory; FedAvg follows separate local trajectories and averages model weights. Partial final batches also make the optimizer-step counts differ.` |

**Interpretation:** this is a controlled *matched-budget comparison*, not a claim that the two training algorithms are numerically identical.

## 4. Final test results

**Saved-result facts**

| Method | Test loss | Test accuracy | Test macro F1 | Best validation checkpoint | Train + validation-loop time |
|---|---:|---:|---:|---:|---:|
| Centralized SGD | 0.134718 | 96.17% | 96.13% | epoch 9 (95.71%) | 19.535 s |
| Hand-written FedAvg | 0.263875 | 92.31% | 92.21% | round 5 (91.90%) | 18.205 s |

**Derived differences:** FedAvg minus centralized is **-3.86 percentage points** for accuracy and **-3.93 percentage points** for macro F1.

**Interpretation — report the negative result honestly:** FedAvg underperformed the centralized reference by **3.86 percentage points** in test accuracy; its macro-F1 difference was **-3.93 percentage points**. This is the honest outcome of this saved run, not evidence that one method is generally superior. The two optimization procedures are mechanically different, and only one fixed-seed run is available.

The saved visual evidence is the [convergence plot](../results/month2_week8_mnist_e2_variant/convergence_comparison.png), [centralized confusion matrix](../results/month2_week8_mnist_e2_variant/centralized_confusion_matrix.png), and [FedAvg confusion matrix](../results/month2_week8_mnist_e2_variant/fedavg_confusion_matrix.png).

## 5. Validation behavior and convergence

**Saved-result facts**

| Cumulative full-data-equivalent passes | Central validation accuracy | Central validation loss | FedAvg validation accuracy | FedAvg validation loss |
|---:|---:|---:|---:|---:|
| 0 | 10.42% | 2.297378 | 10.42% | 2.297378 |
| 1 | 89.83% | 0.356070 | — | — |
| 2 | 91.69% | 0.289488 | 88.13% | 0.459458 |
| 3 | 92.68% | 0.250824 | — | — |
| 4 | 93.68% | 0.223018 | 89.72% | 0.360461 |
| 5 | 94.29% | 0.202904 | — | — |
| 6 | 94.86% | 0.179512 | 90.83% | 0.323419 |
| 7 | 95.23% | 0.170137 | — | — |
| 8 | 95.43% | 0.161243 | 91.38% | 0.299283 |
| 9 | 95.71% | 0.149862 | — | — |
| 10 | 95.66% | 0.146038 | 91.90% | 0.282253 |

- Centralized best: epoch 9, validation accuracy 95.71%; final epoch accuracy 95.66%.
- FedAvg best: round 5 (pass 10), validation accuracy 91.90%; final round accuracy 91.90%.

**Interpretation:** Both methods improved from their common untrained starting point. The best validation gains were +85.29 percentage points for centralized SGD and +81.48 percentage points for FedAvg. This supports a narrow claim that training progressed; 10 data passes are not enough to claim mathematical convergence or a stable asymptote.

## 6. Per-client test utility

These are IID test partitions used to expose whether the overall average hides a weak client. They are not non-IID results.

**Saved/derived facts**

| Client | Test examples | Central accuracy | FedAvg accuracy | FedAvg − central | Central macro F1 | FedAvg macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,000 | 95.75% | 91.85% | -3.90 percentage points | 95.68% | 91.68% |
| 1 | 2,000 | 96.40% | 93.00% | -3.40 percentage points | 96.35% | 92.84% |
| 2 | 2,000 | 96.45% | 92.15% | -4.30 percentage points | 96.42% | 92.03% |
| 3 | 2,000 | 96.10% | 92.45% | -3.65 percentage points | 96.05% | 92.30% |
| 4 | 2,000 | 96.15% | 92.10% | -4.05 percentage points | 96.16% | 92.14% |

- Centralized client accuracy range: 95.75% to 96.45%.
- FedAvg client accuracy range: 91.85% to 93.00%.

**Interpretation:** client-level differences are descriptive checks for this one IID split. They do not estimate performance for a population of real clients and must not be called non-target-client preservation (no unlearning target exists in Week 8).

## 7. Why the optimizer-step totals differ

**Derived facts**

- Centralized: `ceil(51,000 / 128) × 10 = 399 × 10 = 3,990` optimizer steps.
- FedAvg: sum `ceil(client examples / 128) × E` over every selected client and round = **4,000** optimizer steps.

Thus the saved totals are **3,990 centralized steps vs 4,000 FedAvg local steps**, even though both paths process 510,000 example exposures. The difference comes from rounding each client's final partial minibatch separately; it is not extra training data.

## 8. Communication accounting

**Derived fact**

`407,080 payload bytes × 2 directions × 25 client-round participations = 20,354,000 bytes = 19.411 MiB`.

For this full-participation run, `25 = K × R = 5 × 5`. The two directions are one global-model download and one locally trained model upload. `E` does not multiply communication because all 2 local epoch(s) occur between those two transfers.

**Limitation:** this is a dense tensor-payload estimate. It excludes protocol headers, serialization overhead, retries, latency, compression, and network contention. Timing was measured as `sequential CPU process; not distributed wall-clock time`, so the CPU seconds above are not a real distributed-system speed benchmark.

## 9. Conclusion and limits

**Interpretation:** the saved run is sufficient to check that the hand-written multi-client FedAvg workflow trains, logs its mechanics, and can be compared with a deliberately matched centralized reference. FedAvg underperformed the centralized reference by **3.86 percentage points** in test accuracy; its macro-F1 difference was **-3.93 percentage points**. This is the honest outcome of this saved run, not evidence that one method is generally superior. The two optimization procedures are mechanically different, and only one fixed-seed run is available.

**Limitations**

- This is **one fixed-seed run**. It provides no across-seed variance, confidence interval, or statistical-equivalence test.
- The IID partition is an introductory Month 2 setting; it says nothing yet about Non-IID client drift.
- The clients run sequentially in one CPU process. This validates learning and accounting logic, not deployment throughput, privacy, or network behavior.
- The result establishes neither Federated Unlearning nor forgetting; those remain later-gate work.

For exact reruns, use the saved and resolved configs, code revision `month2-week8-e2` (commit `1315eea929972d2844173e5984588e5638226646`), and environment manifest `environment/month2_cpu_runtime.json`. The recorded seed set is `random_seed=42`, `split_seed=42`, `train_partition_seed=1042`, `test_partition_seed=2042`, `centralized_loader_seed=3042`, `model_initialization_seed=42`.
