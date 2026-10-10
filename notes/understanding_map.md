# Understanding map

Levels: **L0** AI-built · **L1** walked-through · **L2** explain-back · **L3** modify (predicted, changed, ran, explained).
Required before the defense: core (`algorithms/`, `evaluation/forgetting.py`) **L3**; everything else **L2**.
The tutor updates this file; only levels she has demonstrated are recorded. `?` = not yet assessed.

## Code
| Item | File | Required | Level | Last checked | Note |
|---|---|---|---|---|---|
| Logistic/MLP baseline | experiments/month1_week1_mnist_baseline.py, experiments/month1_week2_mnist_mlp.py, models/simple_mlp.py | L2 | ? | — | curriculum 1.1 |
| CNN | models/small_cnn.py, experiments/month1_week3_cnn.py | L2 | ? | — | curriculum 1.2 |
| Experiment utils / configs / verifiers | experiments/experiment_utils.py, reports/verify_*.py | L2 | ? | — | curriculum 1.3 |
| FedAvg aggregation (n_k/n) | algorithms/fedavg.py, server/fedavg_server.py, clients/federated_client.py | L3 | ? | — | curriculum 2.2 |
| IID partition | clients/iid_partition.py | L2 | ? | — | curriculum 2.2 |
| Week 8 FedAvg vs centralized | experiments/iid/month2_week8_mnist_fedavg.py | L2 | ? | — | curriculum 2.3 |
| Shard Non-IID partition | clients/noniid_partition.py | L2 | ? | — | curriculum 3.1 |
| Dirichlet partition | clients/dirichlet_partition.py | L3 | ? | — | curriculum 3.2 |
| FedProx proximal term | clients/fedprox_client.py, server/fedprox_server.py (algorithms/fedprox.py is a stub) | L3 | ? | — | curriculum 3.3 |
| Three-way evaluation | evaluation/three_way.py | L2 | ? | — | curriculum 4.2 |
| Utility metrics | evaluation/utility.py | L2 | ? | — | curriculum 4.2 |
| Efficiency / timing / storage bytes | evaluation/efficiency.py | L2 | ? | — | curriculum 4.2 |
| Week 13 centralized retrain | experiments/unlearning/month4_week13_mnist_full_retraining.py | L2 | ? | — | curriculum 4.3; not the federated gold standard |
| Federated retrain M_retrain^(-A) | (to build) | L3 | — | — | |
| Forgetting metrics + noise floor | evaluation/forgetting.py | L3 | — | — | not built yet |
| PGA | algorithms/federated_unlearning.py | L3 | — | — | not built yet |
| FedEraser | algorithms/federated_unlearning.py | L3 | — | — | not built yet |
| Improvement | algorithms/proposed_method.py | L3 | — | — | not built yet |

## Results (each table/figure used in the thesis)
| Item | Source | Required | Level | Note |
|---|---|---|---|---|

## Thesis sections
| Section | File | Required | Level | Note |
|---|---|---|---|---|
