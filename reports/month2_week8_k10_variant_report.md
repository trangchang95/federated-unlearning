# Month 2, Week 8 — Centralized SGD vs Hand-Written FedAvg

> Generated deterministically from [`results/month2_week8_mnist_k10_variant/metrics.json`](../results/month2_week8_mnist_k10_variant/metrics.json). Do not copy result numbers into this report by hand.

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
| Config | [`configs/month2_week8_mnist_k10_fedavg_vs_centralized.json`](../configs/month2_week8_mnist_k10_fedavg_vs_centralized.json) |
| Resolved run config | [`results/month2_week8_mnist_k10_variant/resolved_config.json`](../results/month2_week8_mnist_k10_variant/resolved_config.json) |
| Configured code revision | `month2-week8-k10` |
| Resolved Git commit / clean HEAD | `b601ac48d77f5194f039a67c06087804c1e3132f` / `True` |
| Environment manifest | [`environment/month2_cpu_runtime.json`](../environment/month2_cpu_runtime.json) |
| Resolved-config SHA-256 | `4f418279547d6cf4adde22b5a880de9308b01e64f51a56ccf5eb28365712c05d` |
| Metrics SHA-256 | `c316e78ac25794d4b62be88477337ccaf2d9679accdf9b4c0f139b4f766a9983` |
| Common initial-model SHA-256 | `f1a6fbfc1590ec7bfef00d6df509b9d5b4eee93b4814397af11b6f4f223d2125` |
| Partition | `iid_seeded_equal_size` |
| Fixed seeds | `random_seed=42`, `split_seed=42`, `train_partition_seed=1042`, `test_partition_seed=2042`, `centralized_loader_seed=3042`, `model_initialization_seed=42` |

**Derived integrity result:** PASS. Before rendering, the builder checked that every resolved input setting is unchanged in the metrics (only the declared `accuracy`, `f1`, and `communication_cost` output placeholders may be filled), the configured Git revision resolved to a clean HEAD, the mandatory top-level metrics belong to `handwritten_fedavg`, the initialization was common, data-pass counts matched, exactly one validation checkpoint was selected per method, optimizer-step arithmetic was correct, per-client test data covered the test set, and every round's upload/download byte totals agreed. A mismatch makes the builder stop instead of producing a report.

## 2. Beginner protocol card: K, C, E, B, R

**Saved/derived facts**

| Symbol | Meaning | This run |
|---|---|---:|
| `K` | Total number of clients | 10 |
| `C` | Fraction of clients selected each round | 1.00 (10 of 10 selected) |
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
| Starting model | All three recorded initialization checksums equal `f1a6fbfc1590ec7bfef00d6df509b9d5b4eee93b4814397af11b6f4f223d2125` |
| Optimization settings | `SGD`, learning rate 0.1, batch size 128 |
| Training budget | 5 centralized full passes vs 5 FedAvg full passes |
| Model selection | Each method's best validation checkpoint was selected before the test set was evaluated |

| Mechanically different | Why it matters |
|---|---|
| Centralized SGD | One model processes the complete shuffled training split and updates after every central minibatch. |
| FedAvg | 10 IID client shards train separate local copies; the server performs a sample-weighted model average once per round. |
| Minibatch boundaries | Splitting 51,000 examples across clients changes where partial final batches occur, even though total example exposures match. |
| Communication | Centralized simulation records 0 model-transfer bytes; FedAvg counts one download and one upload per selected client per round. |
| Data order and update trajectory | `Centralized SGD follows one continuous trajectory; FedAvg follows separate local trajectories and averages model weights. Partial final batches also make the optimizer-step counts differ.` |

**Interpretation:** this is a controlled *matched-budget comparison*, not a claim that the two training algorithms are numerically identical.

## 4. Final test results

**Saved-result facts**

| Method | Test loss | Test accuracy | Test macro F1 | Best validation checkpoint | Train + validation-loop time |
|---|---:|---:|---:|---:|---:|
| Centralized SGD | 0.187009 | 94.76% | 94.69% | epoch 5 (94.29%) | 9.794 s |
| Hand-written FedAvg | 0.396494 | 89.55% | 89.37% | round 5 (88.89%) | 9.945 s |

**Derived differences:** FedAvg minus centralized is **-5.21 percentage points** for accuracy and **-5.32 percentage points** for macro F1.

**Interpretation — report the negative result honestly:** FedAvg underperformed the centralized reference by **5.21 percentage points** in test accuracy; its macro-F1 difference was **-5.32 percentage points**. This is the honest outcome of this saved run, not evidence that one method is generally superior. The two optimization procedures are mechanically different, and only one fixed-seed run is available.

The saved visual evidence is the [convergence plot](../results/month2_week8_mnist_k10_variant/convergence_comparison.png), [centralized confusion matrix](../results/month2_week8_mnist_k10_variant/centralized_confusion_matrix.png), and [FedAvg confusion matrix](../results/month2_week8_mnist_k10_variant/fedavg_confusion_matrix.png).

## 5. Validation behavior and convergence

**Saved-result facts**

| Cumulative full-data-equivalent passes | Central validation accuracy | Central validation loss | FedAvg validation accuracy | FedAvg validation loss |
|---:|---:|---:|---:|---:|
| 0 | 10.42% | 2.297378 | 10.42% | 2.297378 |
| 1 | 89.83% | 0.356070 | 78.27% | 1.184959 |
| 2 | 91.69% | 0.289488 | 84.66% | 0.679512 |
| 3 | 92.68% | 0.250824 | 86.86% | 0.530195 |
| 4 | 93.68% | 0.223018 | 88.08% | 0.459561 |
| 5 | 94.29% | 0.202904 | 88.89% | 0.418942 |

- Centralized best: epoch 5, validation accuracy 94.29%; final epoch accuracy 94.29%.
- FedAvg best: round 5 (pass 5), validation accuracy 88.89%; final round accuracy 88.89%.

**Interpretation:** Both methods improved from their common untrained starting point. The best validation gains were +83.87 percentage points for centralized SGD and +78.47 percentage points for FedAvg. This supports a narrow claim that training progressed; 5 data passes are not enough to claim mathematical convergence or a stable asymptote.

## 6. Per-client test utility

These are IID test partitions used to expose whether the overall average hides a weak client. They are not non-IID results.

**Saved/derived facts**

| Client | Test examples | Central accuracy | FedAvg accuracy | FedAvg − central | Central macro F1 | FedAvg macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,000 | 95.40% | 90.20% | -5.20 percentage points | 95.24% | 89.93% |
| 1 | 1,000 | 92.80% | 89.10% | -3.70 percentage points | 92.59% | 88.80% |
| 2 | 1,000 | 96.50% | 91.20% | -5.30 percentage points | 96.50% | 91.01% |
| 3 | 1,000 | 94.40% | 88.90% | -5.50 percentage points | 94.25% | 88.55% |
| 4 | 1,000 | 95.30% | 90.10% | -5.20 percentage points | 95.16% | 89.76% |
| 5 | 1,000 | 94.30% | 87.90% | -6.40 percentage points | 94.25% | 87.74% |
| 6 | 1,000 | 93.40% | 87.90% | -5.50 percentage points | 93.31% | 87.50% |
| 7 | 1,000 | 96.10% | 90.90% | -5.20 percentage points | 96.00% | 90.78% |
| 8 | 1,000 | 94.50% | 89.00% | -5.50 percentage points | 94.55% | 89.12% |
| 9 | 1,000 | 94.90% | 90.30% | -4.60 percentage points | 94.94% | 90.25% |

- Centralized client accuracy range: 92.80% to 96.50%.
- FedAvg client accuracy range: 87.90% to 91.20%.

**Interpretation:** client-level differences are descriptive checks for this one IID split. They do not estimate performance for a population of real clients and must not be called non-target-client preservation (no unlearning target exists in Week 8).

## 7. Why the optimizer-step totals differ

**Derived facts**

- Centralized: `ceil(51,000 / 128) × 5 = 399 × 5 = 1,995` optimizer steps.
- FedAvg: sum `ceil(client examples / 128) × E` over every selected client and round = **2,000** optimizer steps.

Thus the saved totals are **1,995 centralized steps vs 2,000 FedAvg local steps**, even though both paths process 255,000 example exposures. The difference comes from rounding each client's final partial minibatch separately; it is not extra training data.

## 8. Communication accounting

**Derived fact**

`407,080 payload bytes × 2 directions × 50 client-round participations = 40,708,000 bytes = 38.822 MiB`.

For this full-participation run, `50 = K × R = 10 × 5`. The two directions are one global-model download and one locally trained model upload. `E` does not multiply communication because all 1 local epoch(s) occur between those two transfers.

**Limitation:** this is a dense tensor-payload estimate. It excludes protocol headers, serialization overhead, retries, latency, compression, and network contention. Timing was measured as `sequential CPU process; not distributed wall-clock time`, so the CPU seconds above are not a real distributed-system speed benchmark.

## 9. Conclusion and limits

**Interpretation:** the saved run is sufficient to check that the hand-written multi-client FedAvg workflow trains, logs its mechanics, and can be compared with a deliberately matched centralized reference. FedAvg underperformed the centralized reference by **5.21 percentage points** in test accuracy; its macro-F1 difference was **-5.32 percentage points**. This is the honest outcome of this saved run, not evidence that one method is generally superior. The two optimization procedures are mechanically different, and only one fixed-seed run is available.

**Limitations**

- This is **one fixed-seed run**. It provides no across-seed variance, confidence interval, or statistical-equivalence test.
- The IID partition is an introductory Month 2 setting; it says nothing yet about Non-IID client drift.
- The clients run sequentially in one CPU process. This validates learning and accounting logic, not deployment throughput, privacy, or network behavior.
- The result establishes neither Federated Unlearning nor forgetting; those remain later-gate work.

For exact reruns, use the saved and resolved configs, code revision `month2-week8-k10` (commit `b601ac48d77f5194f039a67c06087804c1e3132f`), and environment manifest `environment/month2_cpu_runtime.json`. The recorded seed set is `random_seed=42`, `split_seed=42`, `train_partition_seed=1042`, `test_partition_seed=2042`, `centralized_loader_seed=3042`, `model_initialization_seed=42`.
