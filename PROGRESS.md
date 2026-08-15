# Progress Log

Running record of what's been done on this thesis, in plain language.
Read [`../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md`](../Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md)
for the full 6-month plan this log tracks against. Each entry below is
meant to be understandable on its own, without needing to remember
earlier entries.

---

## 2026-08-16 — Repo scaffold

**What:** Created the `federated-unlearning/` project folder with the
structure the plan calls for (`algorithms/`, `evaluation/`,
`experiments/`, `configs/`, `results/`, etc.), a starter
`requirements.txt`, and initialized git.

**Why:** Every later week adds code into this structure instead of
starting from scratch, so it needs to exist before Week 1 coding begins.

**Status:** Done. Nothing to verify yet — no experiments have run.

---

## 2026-08-16 — Week 1: Python + ML basics, MNIST baseline

### What "Week 1" means here

The plan's Week 1 (see plan file, section "MONTH 1", "Week 1 — Python +
ML") is explicitly **not** about neural networks yet — that starts in
Week 2. Week 1 is about the everyday tools every experiment will use:

| Topic | What it means | Where it shows up in the code |
|---|---|---|
| **NumPy** | Arrays of numbers, and math on them without slow Python loops | Pixel values are stored as one big NumPy array; normalizing `pixels / 255.0` is a NumPy operation on all 70,000 images at once |
| **Pandas** | Tables (like a spreadsheet) for counting/summarizing data | Counting how many images of each digit (0-9) are in each split |
| **Matplotlib** | Drawing plots/images | Showing example digit images, drawing the confusion matrix |
| **train/validation/test split** | Don't test a model on the same data it learned from | MNIST's 70,000 images are split 70% train / 15% validation / 15% test |
| **Git** | Version control | Already set up in the previous entry; this week's code is a git commit |

### What was built

**File:** [`experiments/month1_week1_mnist_baseline.py`](experiments/month1_week1_mnist_baseline.py)

A script that:
1. Downloads the MNIST dataset (70,000 handwritten digit images, 28x28
   pixels each, labeled 0-9) from OpenML.
2. Splits it into train (70%) / validation (15%) / test (15%).
3. Uses Pandas to print/save a table of how many images of each digit
   ended up in each split (sanity check — should be roughly even).
4. Uses Matplotlib to save one example image per digit (sanity check —
   confirms the images loaded correctly and labels match).
5. Trains a **logistic regression** classifier — this is a simple
   "draw a line between classes" model, not a neural network. It's the
   right level for Week 1 because neural networks (forward/backward
   propagation) aren't introduced until Week 2, and CNNs not until Week 3.
   Using logistic regression now means: when a CNN is introduced in
   Week 3, its accuracy can be compared against this baseline to show
   the improvement is real.
6. Reports accuracy, precision, recall, F1 on the held-out test set,
   and saves a confusion matrix image (shows which digits get confused
   with which, e.g. 4 vs 9).

### Why a plain classifier and not a neural net

A thesis needs baselines to compare against. If Week 3's CNN gets 99%
accuracy, that number means nothing without knowing "99% compared to
what?". Logistic regression (expected ~92% on MNIST) is the simplest
possible baseline — it establishes the floor.

### How to reproduce

```bash
cd federated-unlearning
pip install -r requirements.txt
python experiments/month1_week1_mnist_baseline.py
```

Outputs land in `results/month1_week1/`:
- `class_distribution.csv` — digit counts per split
- `sample_digits.png` — one example image per digit
- `confusion_matrix.png` — test-set error pattern
- `metrics.json` — accuracy/precision/recall/F1 + exact run config (dataset,
  model, random seed, split sizes) so the run is reproducible

### Results

| Metric | Value |
|---|---|
| Split sizes | train 49,000 / val 10,500 / test 10,500 (70/15/15) |
| Validation accuracy | 91.71% |
| Test accuracy | 92.20% |
| Test macro F1 | 92.09% |
| Train time | 24.8s (CPU, this machine) |

92% is in the expected range for plain logistic regression on MNIST
(the commonly cited range is ~90-92%). This becomes the number every
later model (Week 2 neural net, Week 3 CNN, and eventually the FL/FU
models) should beat — if a fancier model scores *below* ~92%, that's a
bug, not a good result.

See `results/month1_week1/confusion_matrix.png` for which digits the
model confuses most (typically 4↔9 and 3↔5↔8 for this kind of model).

### Gate M1 check (plan section 15)

Gate 1 requires: PyTorch, CNN, centralized baseline. This entry covers
the **centralized baseline** piece only (using scikit-learn, not PyTorch
yet — PyTorch and neural nets start Week 2). Not yet done: PyTorch, CNN
(planned for Weeks 2-3).

### Known risk for later weeks

This machine runs **Python 3.14**, which is very new — PyTorch wheels
may not yet support it when Week 2 (neural networks in PyTorch) starts.
If `pip install torch` fails then, the fix is installing a second Python
version (e.g. 3.11 or 3.12) in a virtual environment just for this
project, since PyTorch typically lags newest Python releases by several
months.
