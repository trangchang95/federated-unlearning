# Gate 1 Evidence Checklist

This checklist separates two different questions:

1. **Are the Month 1 technical artifacts complete and reproducible?** This can
   be checked automatically from files, metrics, checkpoints, and Git.
2. **Can the student explain the concepts?** This requires the student's own
   answers and a separate conceptual review; a script cannot infer genuine
   understanding from experiment files.

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
| Git repository and versioned evidence | The reproduced Month 1 experiment source is fixed under tag `month1-gate1`; the reviewed Gate 1 closure is fixed under tag `gate1-complete`; data and result outputs remain ignored | Proven |
| Experiment report | Eight-page PDF built from saved metrics and visually inspected page by page | Proven |

Run the automated audit from the project root:

```powershell
conda run -n mse-ai python reports\verify_gate1_artifacts.py
```

The checker deliberately fails if a required field or artifact is missing, a
numeric confusion matrix does not match the test-set size, the CIFAR-10 archive
checksum changes, report values differ from saved metrics, Month 1 files are
not committed, or the Git worktree is dirty. It also checks that
`month1-gate1` is an ancestor containing the reproduced experiment state and,
after closure, that `gate1-complete` identifies the reviewed state.

## Human-understanding evidence

| Requirement | Evidence | Status |
|---|---|---|
| Explain how a model is trained | Student correctly separated forward, loss, backward, and optimizer weight-update steps | Passed on 2026-08-16 |
| Explain how loss works | Student correctly distinguished confidence-sensitive cross-entropy from top-class accuracy | Passed on 2026-08-16 |
| Explain why accuracy changes | Student correctly used learning behavior, dataset difficulty, augmentation, dropout, and measured Month 1 evidence | Passed on 2026-08-16 |

The complete question-by-question decision is recorded in
[`gate1_self_check.md`](gate1_self_check.md). The Month 1 PDF now states the
dated Gate 1 outcome and points readers to `README.md` and `PROGRESS.md`; the
experiment values remain tied to the original reproduced run, while the
completion tag records the later conceptual review.
