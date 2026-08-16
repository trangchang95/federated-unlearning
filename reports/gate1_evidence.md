# Gate 1 Evidence Checklist

This checklist separates two different questions:

1. **Are the Month 1 technical artifacts complete and reproducible?** This can
   be checked automatically from files, metrics, checkpoints, and Git.
2. **Can the student explain the concepts?** This requires the student's own
   answers and cannot be passed by a script or by an AI assistant.

## Technical evidence

| Month 1 requirement | Evidence | Status |
|---|---|---|
| Python, NumPy, Pandas, Matplotlib | Week 1 runner, class-distribution CSV, sample plot, and confusion matrix | Proven |
| PyTorch neural-network training | Week 2 MLP with explicit forward, cross-entropy, backward, and optimizer steps | Proven |
| CNN concepts | `SmallCNN` contains learned convolutions, ReLU, pooling, adaptive pooling, and a classifier | Proven |
| Centralized MNIST CNN | Config-driven run, checkpoint, curves, feature maps, confusion matrix, and measured test metrics | Proven |
| Centralized CIFAR-10 CNN | Config-driven run, checkpoint, curves, feature maps, class metrics, confusion matrix, and archive checksum | Proven |
| Evaluation methodology | Accuracy, precision, recall, macro F1, validation selection, test evaluation, and confusion matrices | Proven |
| SISA motivation and concept | Fixed six-question note plus the required literature-matrix row | Proven |
| Experiment logging | Every config and result includes all 15 mandatory fields | Proven |
| Reproducible configuration | Each config pins CPU, seed, runner, dataset version, exact environment manifest, and code tag | Proven |
| Exact software environment | Python 3.10.20 manifest and exact Month 1 dependency lock | Proven |
| Git repository and versioned evidence | All Month 1 source, configs, literature, documentation, verifier, and final report are committed under tag `month1-gate1`; data and result outputs remain ignored | Proven |
| Experiment report | Eight-page PDF built from saved metrics and visually inspected page by page | Proven |

Run the automated audit from the project root:

```powershell
conda run -n mse-ai python reports\verify_gate1_artifacts.py
```

The checker deliberately fails if a required field or artifact is missing, a
numeric confusion matrix does not match the test-set size, the CIFAR-10 archive
checksum changes, report values differ from saved metrics, Month 1 files are
not committed, the Git worktree is dirty, or `month1-gate1` does not identify
the checked commit.

## Human-understanding evidence

| Requirement | Evidence | Status |
|---|---|---|
| Explain how a model is trained | Student's own Gate 1 answers | Waiting for student |
| Explain how loss works | Student's own Gate 1 answers | Waiting for student |
| Explain why accuracy changes | Student's own Gate 1 answers using measured Month 1 evidence | Waiting for student |

Use [`gate1_study_guide.md`](gate1_study_guide.md) to learn the concepts, then
answer [`gate1_self_check.md`](gate1_self_check.md) without copying. Gate 1
must remain unchecked until those answers are reviewed.
