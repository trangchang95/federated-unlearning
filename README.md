# Federated Unlearning — Thesis Prototype

Prototype and experiment codebase for the 6-month thesis plan in
[`../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md`](../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md).

**Topic:** Federated Unlearning for heterogeneous (Non-IID) Federated Learning.

## Setup

For ordinary development, install the project dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For exact Month 1 CPU reproduction, use Python 3.10.20, check out the
`month1-gate1` revision, and install the locked direct dependencies:

```powershell
git checkout month1-gate1
python -m pip install -r environment\month1_cpu_requirements.txt
```

Every Month 1 JSON config names that code revision, the CPU device, its runner,
dataset version, and `environment/month1_cpu_runtime.json`. The runners stop
before training when strict environment validation detects a different Python
or package version; this prevents an accidental machine-dependent rerun from
being presented as the same experiment.

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
| `environment/` | Exact runtime manifest and dependency locks for reproduced milestones |
| `literature/` | Six-question paper notes and the literature matrix |
| `reports/` | Report builders, verification scripts, and gate-review learning material |
| `output/` | Versioned final artifacts such as the verified Month 1 PDF report |
| `results/` | Logged experiment outputs — gitignored |
| `notebooks/` | Exploratory analysis |

## Experiment logging convention

Every run must record (see plan §14):
`dataset, number_of_clients, client_fraction, local_epochs, learning_rate, batch_size, alpha, random_seed, algorithm, target_client, number_of_rounds, accuracy, f1, unlearning_time, communication_cost`

Use a config file per run (`configs/`) and write results to `results/` automatically — no manual copy-paste into spreadsheets.

## Current milestone

**Month 1 — ML/DL Foundation** (see plan §9, Gate 1 in §15): centralized CNN baseline on MNIST and CIFAR-10.

### Month 1 breakdown

- [x] Week 1: Python/NumPy/Pandas/Matplotlib and logistic-regression MNIST baseline
- [x] Week 2: PyTorch neural-network fundamentals and MLP MNIST baseline
- [x] Week 3: CNN baselines on MNIST and CIFAR-10
- [x] Week 4: evaluation methodology, SISA concepts, and Month 1 report

The next permitted unit is the **Gate 1 beginner explanation check** in
[`reports/gate1_self_check.md`](reports/gate1_self_check.md). The technical
artifacts are ready, but the Gate 1 checkbox stays open until the student can
explain the training loop, loss behavior, CNN feature maps, validation/test
separation, and the observed results in their own words. Federated Learning,
Non-IID, FedProx, and Federated Unlearning remain blocked by the gates below.

## Status

- [ ] Gate 1 (Month 1): CNN + centralized baseline
- [ ] Gate 2 (Month 2): FedAvg from scratch, multi-client FL
- [ ] Gate 3 (Month 3): IID/Non-IID + FedProx benchmark
- [ ] Gate 4 (Month 4): FU baseline reproduced, retraining comparison
- [ ] Gate 5 (Month 5): Research gap + proposed method + preliminary results
- [ ] Gate 6 (Month 6): Final experiments, thesis, reproducible code

## Reproducing completed experiments on this machine

The verified local environment is the Conda environment `mse-ai` (Python
3.10.20, CPU PyTorch). Run experiments from this directory:

```powershell
conda run -n mse-ai python experiments\month1_week1_mnist_baseline.py --config configs\month1_week1_mnist_logistic_regression.json
conda run -n mse-ai python experiments\month1_week2_mnist_mlp.py --config configs\month1_week2_mnist_mlp.json
conda run -n mse-ai python experiments\month1_week3_cnn.py --config configs\month1_week3_mnist_cnn.json
conda run -n mse-ai python experiments\month1_week3_cnn.py --config configs\month1_week3_cifar10_cnn.json
```

The first run downloads MNIST into `data/`. Code and configurations are
versioned; downloaded data, model checkpoints, plots, and metrics under
`data/` and `results/` are intentionally gitignored.

## Month 1 report and Gate 1 review

- Final report: [`output/pdf/month1_experiment_report.pdf`](output/pdf/month1_experiment_report.pdf)
- SISA six-question note: [`literature/sisa_2021_six_questions.md`](literature/sisa_2021_six_questions.md)
- Literature matrix: [`literature/literature_matrix.md`](literature/literature_matrix.md)
- Beginner Gate 1 study guide: [`reports/gate1_study_guide.md`](reports/gate1_study_guide.md)
- Gate 1 evidence checklist: [`reports/gate1_evidence.md`](reports/gate1_evidence.md)
- Student explanation check: [`reports/gate1_self_check.md`](reports/gate1_self_check.md)

Rebuild and structurally verify the report from saved experiment artifacts:

```powershell
conda run -n mse-ai python reports\build_month1_report.py
conda run -n mse-ai python reports\verify_month1_report.py
```

Verify all non-student Gate 1 evidence without rerunning training. This command
does not assess understanding or close the gate:

```powershell
conda run -n mse-ai python reports\verify_gate1_artifacts.py
```
