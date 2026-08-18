# Federated Unlearning — Thesis Prototype

Prototype and experiment codebase for the 6-month thesis plan in
[`Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md`](Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md).

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

For Month 2, use Python 3.10.20 and the separate direct-dependency lock that
adds the Python-3.10-compatible Flower version used in Week 7:

```powershell
python -m pip install -r environment\month2_cpu_requirements.txt
```

The matching runtime record is `environment/month2_cpu_runtime.json`. Flower
is pinned to 1.30.0 because 1.31 and newer require Python 3.11; this preserves
the verified Python 3.10 environment rather than silently changing runtimes.

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
| `tests/` | Deterministic synthetic checks for hand-written algorithms and workflow invariants |
| `output/` | Versioned final artifacts such as the verified Month 1 PDF report |
| `results/` | Logged experiment outputs — gitignored |
| `notebooks/` | Exploratory analysis |

## Experiment logging convention

Every run must record (see plan §14):
`dataset, number_of_clients, client_fraction, local_epochs, learning_rate, batch_size, alpha, random_seed, algorithm, target_client, number_of_rounds, accuracy, f1, unlearning_time, communication_cost`

Use a config file per run (`configs/`) and write results to `results/` automatically — no manual copy-paste into spreadsheets.

## Current milestone

**Month 2, Gate 2 review — explain and vary the completed IID experiment**
(see plan §9): the canonical centralized-versus-FedAvg run is verified. The
student must now explain it and perform the controlled `K` and `E` changes in
`reports/gate2_self_check.md` before Gate 2 can close.

### Month 1 breakdown

- [x] Week 1: Python/NumPy/Pandas/Matplotlib and logistic-regression MNIST baseline
- [x] Week 2: PyTorch neural-network fundamentals and MLP MNIST baseline
- [x] Week 3: CNN baselines on MNIST and CIFAR-10
- [x] Week 4: evaluation methodology, SISA concepts, and Month 1 report

The student's ten answers in
[`reports/gate1_self_check.md`](reports/gate1_self_check.md) were reviewed on
2026-08-16 and passed. This closes Gate 1: the centralized baselines are
reproducible, and the student can explain the training loop, loss behavior,
accuracy changes, CNN features, evaluation splits, and measured results.

### Month 2 breakdown

- [x] Week 5: FL concepts and six-question reading of McMahan et al. (2017)
- [x] Week 6: hand-implemented FedAvg
- [x] Week 7: inspect/use an FL framework only after hand-written FedAvg works
- [x] Week 8: first MNIST FL experiment and centralized-versus-FedAvg report

The Week 5 learning package is prepared:

- beginner guide:
  [`reports/month2_week5_fl_concepts.md`](reports/month2_week5_fl_concepts.md);
- fixed six-question paper note:
  [`literature/fedavg_2017_six_questions.md`](literature/fedavg_2017_six_questions.md);
- student check:
  [`reports/month2_week5_self_check.md`](reports/month2_week5_self_check.md).

The student's ten Week 5 answers were reviewed on 2026-08-18 and passed. The
student correctly traced one round, separated local optimizer updates from
server aggregation, calculated sample-weighted averaging, explained the
`C`/`E`/`B` trade-offs, and preserved the privacy caveat.

The Week 6 hand-written implementation is now complete:

- pure sample-count-weighted aggregation:
  [`algorithms/fedavg.py`](algorithms/fedavg.py);
- isolated local SGD responsibility:
  [`clients/federated_client.py`](clients/federated_client.py);
- deterministic server round orchestration:
  [`server/fedavg_server.py`](server/fedavg_server.py);
- beginner explanation:
  [`reports/month2_week6_fedavg_implementation.md`](reports/month2_week6_fedavg_implementation.md);
- synthetic tests: [`tests/test_fedavg.py`](tests/test_fedavg.py).

Verify it without downloading data or running an experiment:

```powershell
conda run -n mse-ai python reports\verify_week6_fedavg.py
```

The verifier passed 13 tests on 2026-08-18 and confirmed that no FL framework
is imported. These checks prove the weighting and round mechanics on synthetic
data; they do not claim MNIST accuracy or convergence. Week 7 framework work
was then completed transparently:

- Flower adapter: [`server/flower_compat.py`](server/flower_compat.py);
- compatibility tests: [`tests/test_flower_compat.py`](tests/test_flower_compat.py);
- beginner/source mapping:
  [`reports/month2_week7_flower_compatibility.md`](reports/month2_week7_flower_compatibility.md);
- exact environment: `environment/month2_cpu_runtime.json` and
  `environment/month2_cpu_requirements.txt`.

Verify Week 7 with:

```powershell
conda run -n mse-ai python reports\verify_week7_flower.py
```

Flower 1.30.0 and all 20 Week 6/7 tests passed on 2026-08-18. The comparison
found matching sample-weighted aggregation and an explicitly recorded client-
selection rounding difference.

Week 8 is now technically complete. Its canonical config uses five IID MNIST
clients, full participation, one local epoch, batch size 128, and five rounds.
Centralized SGD receives the same initial model, data split, learning rate,
batch size, and five full-data passes. The measured result is:

| Method | Test accuracy | Macro F1 | Optimizer steps |
|---|---:|---:|---:|
| Centralized SGD | 94.75% | 94.68% | 1,995 |
| Hand-written FedAvg | 90.99% | 90.86% | 2,000 local steps |

FedAvg is 3.76 percentage points lower in this saved run. That negative result
is reported directly; it is not evidence that centralized training is always
better. Both methods improve throughout the five recorded passes, but their
optimizer trajectories and minibatch boundaries differ. FedAvg's estimated
dense model-transfer payload is 20,354,000 bytes (19.411 MiB), not measured
network traffic.

The complete beginner report is
[`reports/month2_week8_centralized_vs_fedavg.md`](reports/month2_week8_centralized_vs_fedavg.md).
The canonical run was repeated once: every deterministic JSON field and both
checkpoint hashes were identical; only CPU timings changed. Verify the
already-saved evidence without training or downloading data with:

```powershell
conda run -n mse-ai python reports\build_month2_week8_report.py --check
conda run -n mse-ai python reports\verify_week8_fedavg.py
```

The canonical config names tag `month2-week8`. The runner intentionally
refuses to claim an exact rerun when the checked-out `HEAD` is a newer commit.
For an intentional training rerun, first save or commit your current work,
then use the producing tag and return to `main` before rebuilding the report:

```powershell
git switch --detach month2-week8
conda run -n mse-ai python experiments\iid\month2_week8_mnist_fedavg.py --config configs\month2_week8_mnist_iid_fedavg_vs_centralized.json --overwrite
git switch main
conda run -n mse-ai python reports\build_month2_week8_report.py
conda run -n mse-ai python reports\verify_week8_fedavg.py
```

The rerun replaces only the gitignored canonical result directory. CPU timing
normally changes, so rebuilding the tracked report may produce a documentation
diff even when all deterministic values and checkpoint hashes repeat.

Gate 2 remains open because its learning check is not yet reviewed. The
student must answer
[`reports/gate2_self_check.md`](reports/gate2_self_check.md), change the number
of clients in a separate config, change local epochs in another matched-budget
config, and explain the resulting convergence curves. Non-IID, FedProx, and
Federated Unlearning remain blocked by later gates.

Check Gate 2 readiness with:

```powershell
conda run -n mse-ai python reports\verify_gate2_readiness.py --allow-waiting
```

The current expected result is `WAITING`, not `PASS`. This means the canonical
technical evidence is valid but required student evidence is still absent.
`--allow-waiting` prevents Conda from displaying an error footer for this
expected beginner state; it never turns `WAITING` into `READY`. Automated gate
checks should omit the flag so `WAITING` exits with code 2. The checker reads
[`reports/gate2_variant_evidence.json`](reports/gate2_variant_evidence.json)
and clearly lists each missing item. It returns `FAIL` for corrupted or
contradictory evidence and returns `READY` only after all of the following:

1. the student answers Questions 1–8 and writes the two predictions in
   Questions 9–10;
2. a reviewer approves the proposed changes before execution;
3. separate `K=10` and `E=2` configs are committed/tagged and run into new,
   non-canonical output directories;
4. deterministic reports are built from both saved `metrics.json` files;
5. the student interprets both results; and
6. a human reviewer records a date and `PASS` after checking conceptual
   correctness.

The pre-run approval must already be present in the Git tag that produces the
two runs. The checker hashes the approved answers, predictions, and pre-run
review and requires the same snapshot in both producing tags and the final
self-check; predictions cannot be rewritten after seeing results. Final
closure also requires the exact line
`**Gate 2 status:** CLOSED` in the dated `PROGRESS.md` entry; general words such
as “Week 8 complete” do not count as a gate decision.

The checker confirms only that answer fields are non-empty; it does not grade
the prose by keywords or replace the human review. It also does not require a
variant to beat the canonical accuracy, because negative results must remain
reportable.

One saved metric has a narrower audit limit: global accuracy/F1, per-class
accuracy, per-client counts, and weighted per-client accuracy are recomputed
from logged evidence, but each client's macro F1 is only range-checked. The
frozen Week 8 runner did not log a confusion matrix per client, so that value
cannot be independently reconstructed without changing the protected
experiment. Variant reports must treat it as a runner-reported descriptive
value, not a separately cross-validated statistic.

## Status

- [x] Gate 1 (Month 1): CNN + centralized baseline
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
- Literature sanity check:
  [`reports/month1_literature_sanity_check.md`](reports/month1_literature_sanity_check.md)
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

Verify the recorded Gate 1 technical evidence and completion status without
rerunning training. The script checks the saved review marker; it does not
replace the conceptual review of the student's answers:

```powershell
conda run -n mse-ai python reports\verify_gate1_artifacts.py
```

The 2026-08-17 literature check classifies all four Month 1 accuracies as
`plausible / consistent with literature`. It is a source audit only: the
neural-network values still come from one fixed seed each, and differences in
architectures, splits, preprocessing, optimizers, epochs, and seeds prevent
claims of direct comparison or statistical equivalence.
