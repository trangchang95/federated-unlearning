# Federated Unlearning — Thesis Prototype

Prototype and experiment codebase for the 6-month thesis plan in
[`../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md`](../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md).

**Topic:** Federated Unlearning for heterogeneous (Non-IID) Federated Learning.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

| Folder | Purpose |
|---|---|
| `data/` | Raw/partitioned datasets (MNIST, CIFAR-10, ...) — gitignored |
| `models/` | Model architectures (CNNs) |
| `clients/` | Client-side training logic |
| `server/` | Server-side aggregation logic |
| `algorithms/` | `fedavg.py`, `fedprox.py`, `federated_unlearning.py`, `proposed_method.py` |
| `experiments/` | Experiment scripts, grouped by `iid/`, `noniid/`, `unlearning/`, `ablation/` |
| `evaluation/` | `utility.py`, `forgetting.py`, `efficiency.py`, `privacy.py` metrics |
| `configs/` | YAML/JSON experiment configs (see logging convention below) |
| `results/` | Logged experiment outputs — gitignored |
| `notebooks/` | Exploratory analysis |

## Experiment logging convention

Every run must record (see plan §14):
`dataset, number_of_clients, client_fraction, local_epochs, learning_rate, batch_size, alpha, random_seed, algorithm, target_client, number_of_rounds, accuracy, f1, unlearning_time, communication_cost`

Use a config file per run (`configs/`) and write results to `results/` automatically — no manual copy-paste into spreadsheets.

## Current milestone

**Month 1 — ML/DL Foundation** (see plan §9, Gate 1 in §15): centralized CNN baseline on MNIST and CIFAR-10.

## Status

- [ ] Gate 1 (Month 1): CNN + centralized baseline
- [ ] Gate 2 (Month 2): FedAvg from scratch, multi-client FL
- [ ] Gate 3 (Month 3): IID/Non-IID + FedProx benchmark
- [ ] Gate 4 (Month 4): FU baseline reproduced, retraining comparison
- [ ] Gate 5 (Month 5): Research gap + proposed method + preliminary results
- [ ] Gate 6 (Month 6): Final experiments, thesis, reproducible code
