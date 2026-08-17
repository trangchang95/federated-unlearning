# Month 2, Week 8 — Centralized SGD vs Hand-Written FedAvg

> Generated deterministically from [`results/month2_week8_mnist_iid_comparison/metrics.json`](../results/month2_week8_mnist_iid_comparison/metrics.json). Do not copy result numbers into this report by hand.

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
| Config | [`configs/month2_week8_mnist_iid_fedavg_vs_centralized.json`](../configs/month2_week8_mnist_iid_fedavg_vs_centralized.json) |
| Resolved run config | [`results/month2_week8_mnist_iid_comparison/resolved_config.json`](../results/month2_week8_mnist_iid_comparison/resolved_config.json) |
| Configured code revision | `month2-week8` |
| Resolved Git commit / clean HEAD | `2fb17f49b27a3b0d57a3e3edafa8e2bf514af549` / `True` |
| Environment manifest | [`environment/month2_cpu_runtime.json`](../environment/month2_cpu_runtime.json) |
| Resolved-config SHA-256 | `e403a826841ba675d0d7cf32c9d575afd51b59ac8089333586acd531598247cc` |
| Metrics SHA-256 | `03e341d553ef898d8fced8f316e28d92f58b732487b4a549860a2e73bd7b4858` |
| Common initial-model SHA-256 | `17ed89a630e81e954b7a7a36d8dc78330f5bb48685c2152c4e49e80e046b288b` |
| Partition | `iid_seeded_equal_size` |
| Fixed seeds | `random_seed=42`, `split_seed=42`, `train_partition_seed=1042`, `test_partition_seed=2042`, `centralized_loader_seed=3042`, `model_initialization_seed=42` |

**Derived integrity result:** PASS. Before rendering, the builder checked that every resolved input setting is unchanged in the metrics (only the declared `accuracy`, `f1`, and `communication_cost` output placeholders may be filled), the configured Git revision resolved to a clean HEAD, the mandatory top-level metrics belong to `handwritten_fedavg`, the initialization was common, data-pass counts matched, exactly one validation checkpoint was selected per method, optimizer-step arithmetic was correct, per-client test data covered the test set, and every round's upload/download byte totals agreed. A mismatch makes the builder stop instead of producing a report.

## 2. Beginner protocol card: K, C, E, B, R

**Saved/derived facts**

| Symbol | Meaning | This run |
|---|---|---:|
| `K` | Total number of clients | 5 |
| `C` | Fraction of clients selected each round | 1.00 (5 of 5 selected) |
| `E` | Local epochs completed by each selected client before upload | 1 |
| `B` | Local/central minibatch size | 128 |
| `R` | FedAvg communication rounds | 5 |
| `η` | SGD learning rate | 0.1 |
| Central epochs | Full centralized passes through the training split | 5 |

With full participation (`C=1`), `R × E = 5 × 1 = 5` FedAvg passes over all client training examples. This matches the 5 centralized passes.

## 3. What is matched, and what is still different

**Saved/derived facts**

| Matched on purpose | Evidence |
|---|---|
| Data | Same MNIST training split; each path processed 255,000 training-example exposures |
| Starting model | All three recorded initialization checksums equal `17ed89a630e81e954b7a7a36d8dc78330f5bb48685c2152c4e49e80e046b288b` |
| Optimization settings | `SGD`, learning rate 0.1, batch size 128 |
| Training budget | 5 centralized full passes vs 5 FedAvg full passes |
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
| Centralized SGD | 0.187003 | 94.75% | 94.68% | epoch 5 (94.31%) | 90.148 s |
| Hand-written FedAvg | 0.319598 | 90.99% | 90.86% | round 5 (90.28%) | 88.971 s |

**Derived differences:** FedAvg minus centralized is **-3.76 percentage points** for accuracy and **-3.82 percentage points** for macro F1.

**Interpretation — report the negative result honestly:** FedAvg underperformed the centralized reference by **3.76 percentage points** in test accuracy; its macro-F1 difference was **-3.82 percentage points**. This is the honest outcome of this saved run, not evidence that one method is generally superior. The two optimization procedures are mechanically different, and only one fixed-seed run is available.

The saved visual evidence is the [convergence plot](../results/month2_week8_mnist_iid_comparison/convergence_comparison.png), [centralized confusion matrix](../results/month2_week8_mnist_iid_comparison/centralized_confusion_matrix.png), and [FedAvg confusion matrix](../results/month2_week8_mnist_iid_comparison/fedavg_confusion_matrix.png).

## 5. Validation behavior and convergence

**Saved-result facts**

| Cumulative full-data-equivalent passes | Central validation accuracy | Central validation loss | FedAvg validation accuracy | FedAvg validation loss |
|---:|---:|---:|---:|---:|
| 0 | 10.42% | 2.297378 | 10.42% | 2.297378 |
| 1 | 89.83% | 0.356070 | 84.73% | 0.682182 |
| 2 | 91.69% | 0.289488 | 88.12% | 0.459437 |
| 3 | 92.67% | 0.250805 | 89.29% | 0.392832 |
| 4 | 93.68% | 0.223047 | 89.88% | 0.360449 |
| 5 | 94.31% | 0.202929 | 90.28% | 0.340381 |

- Centralized best: epoch 5, validation accuracy 94.31%; final epoch accuracy 94.31%.
- FedAvg best: round 5 (pass 5), validation accuracy 90.28%; final round accuracy 90.28%.

**Interpretation:** Both methods improved from their common untrained starting point. The best validation gains were +83.89 percentage points for centralized SGD and +79.86 percentage points for FedAvg. This supports a narrow claim that training progressed; 5 data passes are not enough to claim mathematical convergence or a stable asymptote.

## 6. Per-client test utility

These are IID test partitions used to expose whether the overall average hides a weak client. They are not non-IID results.

**Saved/derived facts**

| Client | Test examples | Central accuracy | FedAvg accuracy | FedAvg − central | Central macro F1 | FedAvg macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,000 | 94.10% | 90.80% | -3.30 percentage points | 93.95% | 90.60% |
| 1 | 2,000 | 95.50% | 91.50% | -4.00 percentage points | 95.42% | 91.28% |
| 2 | 2,000 | 94.70% | 90.95% | -3.75 percentage points | 94.62% | 90.81% |
| 3 | 2,000 | 94.75% | 91.35% | -3.40 percentage points | 94.65% | 91.18% |
| 4 | 2,000 | 94.70% | 90.35% | -4.35 percentage points | 94.72% | 90.40% |

- Centralized client accuracy range: 94.10% to 95.50%.
- FedAvg client accuracy range: 90.35% to 91.50%.

**Interpretation:** client-level differences are descriptive checks for this one IID split. They do not estimate performance for a population of real clients and must not be called non-target-client preservation (no unlearning target exists in Week 8).

## 7. Why the optimizer-step totals differ

**Derived facts**

- Centralized: `ceil(51,000 / 128) × 5 = 399 × 5 = 1,995` optimizer steps.
- FedAvg: sum `ceil(client examples / 128) × E` over every selected client and round = **2,000** optimizer steps.

Thus the saved totals are **1,995 centralized steps vs 2,000 FedAvg local steps**, even though both paths process 255,000 example exposures. The difference comes from rounding each client's final partial minibatch separately; it is not extra training data.

## 8. Communication accounting

**Derived fact**

`407,080 payload bytes × 2 directions × 25 client-round participations = 20,354,000 bytes = 19.411 MiB`.

For this full-participation run, `25 = K × R = 5 × 5`. The two directions are one global-model download and one locally trained model upload. `E` does not multiply communication because all 1 local epoch(s) occur between those two transfers.

**Limitation:** this is a dense tensor-payload estimate. It excludes protocol headers, serialization overhead, retries, latency, compression, and network contention. Timing was measured as `sequential CPU process; not distributed wall-clock time`, so the CPU seconds above are not a real distributed-system speed benchmark.

## 9. Conclusion and limits

**Interpretation:** the saved run is sufficient to check that the hand-written multi-client FedAvg workflow trains, logs its mechanics, and can be compared with a deliberately matched centralized reference. FedAvg underperformed the centralized reference by **3.76 percentage points** in test accuracy; its macro-F1 difference was **-3.82 percentage points**. This is the honest outcome of this saved run, not evidence that one method is generally superior. The two optimization procedures are mechanically different, and only one fixed-seed run is available.

**Limitations**

- This is **one fixed-seed run**. It provides no across-seed variance, confidence interval, or statistical-equivalence test.
- The IID partition is an introductory Month 2 setting; it says nothing yet about Non-IID client drift.
- The clients run sequentially in one CPU process. This validates learning and accounting logic, not deployment throughput, privacy, or network behavior.
- The result establishes neither Federated Unlearning nor forgetting; those remain later-gate work.

For exact reruns, use the saved and resolved configs, code revision `month2-week8` (commit `2fb17f49b27a3b0d57a3e3edafa8e2bf514af549`), and environment manifest `environment/month2_cpu_runtime.json`. The recorded seed set is `random_seed=42`, `split_seed=42`, `train_partition_seed=1042`, `test_partition_seed=2042`, `centralized_loader_seed=3042`, `model_initialization_seed=42`.
