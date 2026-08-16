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

---

## 2026-08-16 — Experiment logging compliance update

### What changed

The Week 1 script now reads all of its settings from
`configs/month1_week1_mnist_logistic_regression.json`. A shared helper,
`experiments/experiment_utils.py`, checks that every configuration contains
the 15 fields required by the thesis rules: dataset, client settings,
training settings, random seed, algorithm, output metrics, unlearning time,
and communication cost.

Some of those fields do not apply yet because this is centralized learning,
not Federated Learning. They are still recorded explicitly: one client,
zero communication rounds, no target client, zero communication cost, and
zero unlearning time. Using `null` or zero is intentional and clearer than
silently omitting a field.

The OpenML cache was also moved from the Windows user directory into
`data/openml/`. This matters because `data/` is the repository's designated,
gitignored location for downloaded datasets. The experiment no longer
depends on a hidden cache elsewhere on one particular computer.

### Verification

The saved configuration was rerun successfully. It produced 92.19% test
accuracy and 92.08% macro F1, effectively reproducing the original Week 1
result. The tiny difference from the earlier entry (92.20% / 92.09%) is only
rounding; it is not a meaningful model change.

---

## 2026-08-16 — Week 2: PyTorch neural-network fundamentals

### What was built

**Model:** `models/simple_mlp.py`

The first PyTorch model is a small **multilayer perceptron (MLP)**. It turns
each 28×28 image into 784 input numbers, sends them through a hidden layer of
128 units, and produces 10 output scores—one for each possible digit. It has
101,770 trainable parameters. A parameter is simply a number the model is
allowed to adjust while learning.

This is intentionally not a CNN. Week 2 is about seeing the mechanics of a
neural network clearly; convolution and pooling belong to Week 3.

**Experiment:** `experiments/month1_week2_mnist_mlp.py`

The training loop makes the five Week 2 concepts explicit:

1. **Forward propagation:** images pass through the model and produce 10
   scores (called logits) per image.
2. **Loss:** cross-entropy converts “how wrong were those scores?” into one
   number. Lower loss means better predictions.
3. **Backpropagation:** PyTorch calculates how much every parameter
   contributed to the error.
4. **Optimizer:** Adam changes the parameters in the direction expected to
   reduce the next error.
5. **Learning rate:** `0.001` limits the size of each update. Too large can
   make learning unstable; too small can make it unnecessarily slow.

The 60,000 official MNIST training images are split deterministically into
51,000 training images and 9,000 validation images. The official 10,000 test
images remain untouched until the end. This distinction is important:
training data teaches the model, validation data helps select the best
training epoch, and test data estimates final performance.

### Reproducible configuration and outputs

The complete run is defined by
`configs/month1_week2_mnist_mlp.json`: random seed 42, batch size 128, five
epochs, Adam optimizer, and learning rate 0.001. It can be reproduced with:

```powershell
conda run -n mse-ai python experiments\month1_week2_mnist_mlp.py --config configs\month1_week2_mnist_mlp.json
```

Automatic outputs in `results/month1_week2/` are:

- `metrics.json`: configuration, per-epoch history, timings, accuracy, and F1;
- `learning_curve.png`: loss and accuracy across the five epochs;
- `confusion_matrix.png`: which digit pairs the MLP confuses;
- `best_model.pt`: model parameters from the best validation epoch.

### Results and beginner interpretation

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.4437 | 88.45% | 0.2561 | 92.64% |
| 2 | 0.2060 | 94.12% | 0.1883 | 94.67% |
| 3 | 0.1499 | 95.71% | 0.1513 | 95.68% |
| 4 | 0.1171 | 96.62% | 0.1380 | 95.96% |
| 5 | 0.0950 | 97.26% | 0.1205 | 96.43% |

| Final test metric | Value |
|---|---:|
| Accuracy | 96.80% |
| Macro F1 | 96.77% |
| CPU training time | 105.9 seconds |

The falling loss shows that the optimizer is successfully reducing error.
Validation accuracy also rises instead of collapsing, so the model is
learning patterns that transfer to unseen images rather than merely
memorizing the training set. At epoch 5, training accuracy is only 0.83
percentage points above validation accuracy; this is not evidence of severe
overfitting in this run.

Compared with Week 1 logistic regression (92.19%), the MLP reaches 96.80%, an
improvement of 4.61 percentage points. This is the first experimental evidence
in the project that learned hidden features help beyond a single linear
classifier.

### Gate status and next step

Week 2 is complete, but Gate 1 remains open. Gate 1 specifically requires a
PyTorch **CNN** and centralized baselines, including MNIST and CIFAR-10. The
next allowed unit is therefore Week 3: learn convolution, pooling, and feature
maps, then implement and measure the two CNN baselines. FedAvg and all later
Federated Learning/Unlearning work remain out of scope until Gate 1 closes.

---

## 2026-08-16 — Week 3: CNN baselines on MNIST and CIFAR-10

### What a CNN adds

Week 2's MLP flattened an image into one long row of numbers. That discards
the fact that nearby pixels form local shapes. A **convolutional neural
network (CNN)** preserves this spatial structure:

1. A small 3×3 **filter** slides across the image.
2. At each position, it produces a response to a local pattern such as an
   edge, corner, color transition, or texture.
3. All responses from one filter form a **feature map**.
4. **Pooling** reduces each feature map's width and height, retaining strong
   responses while lowering computation.
5. A classifier combines the learned feature maps into ten class scores.

The saved `feature_maps.png` files make this visible. For the MNIST example,
different filters emphasize different strokes of the digit 7. For the
CIFAR-10 example, different filters respond to the cat's outline, texture,
and contrast with its background. A feature map is therefore not another
input image; it is a map of where one learned filter responded strongly.

### What was built

**Model:** `models/small_cnn.py`

The same deliberately small CNN is used for both datasets:

```text
Input image
  → 3×3 convolution (32 feature maps) → ReLU → 2×2 max pooling
  → 3×3 convolution (64 feature maps) → ReLU → 2×2 max pooling
  → adaptive average pooling to 4×4
  → 128-unit fully connected layer
  → 10 class scores
```

MNIST has one input channel (grayscale), so the model has 151,306 trainable
parameters. CIFAR-10 has three input channels (red, green, blue), so only the
first convolution changes and the model has 151,882 parameters. Keeping the
rest of the architecture the same makes the comparison easier to explain.

**Experiment:** `experiments/month1_week3_cnn.py`

One config-driven runner supports both datasets and automatically saves:

- `metrics.json` with every required thesis field, split sizes, per-epoch
  history, best epoch, per-class accuracy, and CPU training time;
- `learning_curve.png` for training/validation loss and accuracy;
- `confusion_matrix.png` for class-by-class errors;
- `feature_maps.png` showing the first eight convolution responses;
- `best_model.pt` containing the parameters from the best validation epoch.

The runner validates input channels, normalization values, validation
fraction, and model channel settings before starting an expensive run. It
uses fixed random seed 42 and deterministic PyTorch operations.

### Dataset protocol

| Dataset | Training | Validation | Test | Training augmentation |
|---|---:|---:|---:|---|
| MNIST | 51,000 | 9,000 | 10,000 | None |
| CIFAR-10 | 45,000 | 5,000 | 10,000 | Random crop + horizontal flip |

The validation indices are selected once from the official training set using
seed 42. Augmentation applies only to CIFAR-10 training images. Validation and
test images use stable transformations so their metrics remain comparable
from epoch to epoch.

Normalization changes each color channel to a more convenient numerical
scale. It does not add information or change labels. Random crop and
horizontal flip create slightly altered training examples, teaching the
model that a small shift or left/right orientation should not change an
object's class.

### Reproducible configurations

```powershell
conda run -n mse-ai python experiments\month1_week3_cnn.py --config configs\month1_week3_mnist_cnn.json
conda run -n mse-ai python experiments\month1_week3_cnn.py --config configs\month1_week3_cifar10_cnn.json
```

Both use Adam with learning rate 0.001 and batch size 128. MNIST runs for five
epochs. CIFAR-10 runs for ten epochs and additionally uses dropout 0.25,
weight decay 0.0001, and training augmentation because natural color images
are much more varied than centered handwritten digits.

### MNIST results

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.4532 | 85.38% | 0.1266 | 96.20% |
| 2 | 0.1303 | 95.99% | 0.0877 | 97.31% |
| 3 | 0.0925 | 97.14% | 0.0568 | 98.30% |
| 4 | 0.0757 | 97.61% | 0.0538 | 98.23% |
| 5 | 0.0664 | 98.00% | 0.0487 | 98.54% |

| Final MNIST metric | Value |
|---|---:|
| Best epoch | 5 |
| Test accuracy | 98.60% |
| Test macro F1 | 98.60% |
| CPU training time | 430.8 seconds (7.2 minutes) |

The CNN improves test accuracy by 1.80 percentage points over Week 2's MLP
(96.80%) and by 6.41 points over Week 1 logistic regression (92.19%). This is
experimental evidence for the key Week 3 idea: preserving local image
structure helps image classification.

### CIFAR-10 results

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.7462 | 35.73% | 1.4194 | 47.66% |
| 2 | 1.4559 | 46.96% | 1.2661 | 54.18% |
| 3 | 1.3348 | 51.55% | 1.1789 | 58.64% |
| 4 | 1.2335 | 55.28% | 1.0631 | 62.66% |
| 5 | 1.1642 | 58.49% | 1.0003 | 64.70% |
| 6 | 1.1156 | 60.09% | 0.9504 | 67.26% |
| 7 | 1.0695 | 61.76% | 0.9103 | 68.60% |
| 8 | 1.0307 | 63.48% | 0.8749 | 69.82% |
| 9 | 1.0104 | 64.23% | 0.8501 | 70.98% |
| 10 | 0.9799 | 65.36% | 0.8210 | 72.24% |

| Final CIFAR-10 metric | Value |
|---|---:|
| Best epoch | 10 |
| Test accuracy | 71.38% |
| Test macro F1 | 71.07% |
| CPU training time | 1,078.2 seconds (18.0 minutes) |

Validation accuracy is higher than training accuracy here, but this is not a
data leak. During training, random crop/flip makes images harder and dropout
temporarily disables part of the classifier. Both are turned off during
validation. The two curves improve throughout all ten epochs and validation
loss keeps falling, so this run does not show late-epoch overfitting.

MNIST and CIFAR-10 percentages should not be compared as if they were the
same task. MNIST contains centered grayscale digits on plain backgrounds.
CIFAR-10 contains small color objects with different backgrounds, positions,
textures, and viewpoints. A 71.38% small-CNN baseline on CIFAR-10 is useful
because later methods have a realistic, harder reference rather than because
it is expected to match MNIST's 98.60%.

### CIFAR-10 class-level findings

| Class | Accuracy | Class | Accuracy |
|---|---:|---|---:|
| Airplane | 78.7% | Automobile | 87.6% |
| Bird | 53.6% | Cat | 54.5% |
| Deer | 61.7% | Dog | 55.1% |
| Frog | 85.3% | Horse | 79.8% |
| Ship | 78.3% | Truck | 79.2% |

The easiest classes are automobile and frog. Bird, cat, and dog are the
hardest. The confusion matrix shows 233 of 1,000 dogs predicted as cats and
110 cats predicted as dogs. This is a plausible weakness: those animal
classes often share fur, body shapes, and natural backgrounds at only 32×32
pixels. Reporting this weakness is more informative than giving only one
overall percentage.

### Dataset download integrity and compute note

The standard Toronto CIFAR-10 host stalled during the first attempt. That
incomplete download never reached training, its two leftover processes were
stopped, and only the incomplete archive was removed. The same archive was
then obtained from the Zenodo CIFAR-10 record. Before use it was verified as
exactly 170,498,071 bytes with MD5
`c58f30108f718f92721af3b95e74349a`, which is the checksum expected by
TorchVision. Therefore the mirror changed only transfer reliability, not the
dataset contents.

The CPU runtimes—7.2 minutes for MNIST and 18.0 minutes for CIFAR-10—also
confirm the plan's compute warning. Later federated experiments should begin
with small client counts and modest local epochs rather than multiplying
CIFAR-10 work prematurely.

### Gate status and next step

Week 3 is complete: PyTorch CNNs now run on both required centralized
datasets. Gate 1 is deliberately still open because the Month 1 deliverable
also requires Week 4 experimental methodology, SISA motivation/concepts, and
a 5–8 page experiment report. The next allowed unit is Week 4. FedAvg remains
blocked until Gate 1 is fully verified and checked off.

---

## 2026-08-16 — Week 4: methodology, SISA reading, and Month 1 report

### Why Week 4 exists

Working code is not enough for a thesis. A result is scientifically useful
only when another person can understand what was measured, reproduce the
run, and distinguish evidence from interpretation. Week 4 therefore connects
the Month 1 programming exercises to the research habits needed later for
Federated Learning and Federated Unlearning.

For a beginner, the central experiment lifecycle is:

```text
Saved config and seed
  → split train / validation / test data
  → train only on the training split
  → use validation results to choose the best checkpoint
  → evaluate the chosen checkpoint once on the test split
  → automatically save metrics, plots, timing, and the exact config
```

The three data splits have different jobs. The training set changes the model
weights. The validation set helps choose among checkpoints without changing
those weights. The test set estimates performance after all model choices are
finished. Repeatedly choosing changes based on test accuracy would leak test
information into development and make the final number too optimistic.

### Metrics in beginner language

- **Accuracy** is the fraction of all examples classified correctly.
- **Precision** asks: among examples predicted as one class, how many truly
  belong to that class?
- **Recall** asks: among all examples that truly belong to one class, how many
  did the model find?
- **Macro F1** balances precision and recall separately for every class, then
  gives all classes equal weight in the average.
- A **confusion matrix** counts each true-class/predicted-class pair, exposing
  errors hidden by one overall percentage.
- **Cross-entropy loss** uses the full predicted probability distribution.
  Loss can fall while accuracy stays unchanged because the model becomes more
  confident in already-correct answers or less confident in wrong answers.

These distinctions already matter in Month 1. The CIFAR-10 CNN reaches 71.38%
overall accuracy, but its class-level accuracy ranges from 53.6% for bird to
87.6% for automobile. Later Non-IID clients may contain very different class
mixtures, so reporting only global accuracy could hide serious client-level
weaknesses.

### SISA concept review

The first unlearning paper was recorded in
`literature/sisa_2021_six_questions.md` using the plan's fixed six questions:
Problem, Gap, Idea, Assumption, Evaluation, and Limitation. A matching row was
added to `literature/literature_matrix.md`, which is the plan-permitted
Markdown equivalent of `literature_matrix.xlsx`.

SISA means **Sharded, Isolated, Sliced, and Aggregated** training:

1. Split training data into separate shards.
2. Train one constituent model per shard, keeping shards isolated.
3. Divide each shard into chronological slices and save checkpoints.
4. Aggregate the constituent models' predictions.
5. When one point must be forgotten, retrain only the affected shard from a
   checkpoint before that point instead of retraining everything.

The important lesson is conceptual: an unlearning request can be cheaper when
the original training process is designed to localize data influence. The
limitations are equally important. More shards can reduce accuracy,
checkpoints consume storage, the method assumes this structure exists before
training, and the original work studies centralized point-level unlearning
rather than this thesis's eventual client-level federated setting. Week 4
reads the method only for motivation; no unlearning code was implemented.

### Month 1 report

The final deliverable is
`output/pdf/month1_experiment_report.pdf`, an eight-page report generated from
the saved JSON metrics and plots rather than by manually copying numbers. It
contains:

- the Month 1 learning progression and exact measured headline results;
- the data-splitting, checkpoint-selection, and reproducibility protocol;
- plain-language explanations of loss, accuracy, macro F1, and confusion
  matrices;
- Week 1 logistic-regression and Week 2 MLP comparison;
- MNIST CNN architecture, results, and learned feature maps;
- CIFAR-10 learning curves, per-class results, confusion matrix, and compute
  limitations;
- the six-question SISA concept table;
- honest conclusions, limitations, references, and Gate 1 status.

The source is `reports/build_month1_report.py`. The command below rebuilds it
from the saved result artifacts:

```powershell
conda run -n mse-ai python reports\build_month1_report.py
```

`reports/verify_month1_report.py` then checks that the file exists, contains
exactly eight pages, has meaningful extractable text on every page, includes
the correct page labels, and contains no unfinished placeholder markers. The
final PDF was also rendered to images and all eight pages were visually
inspected. No text, table, or figure is clipped.

### Month 1 measured result summary

| Week | Dataset | Model | Test accuracy | Macro F1 | CPU training time |
|---:|---|---|---:|---:|---:|
| 1 | MNIST | Logistic regression | 92.19% | 92.08% | 32.6 seconds |
| 2 | MNIST | 128-unit MLP | 96.80% | 96.77% | 105.9 seconds |
| 3 | MNIST | Small CNN | 98.60% | 98.60% | 430.8 seconds |
| 3 | CIFAR-10 | Small CNN | 71.38% | 71.07% | 1,078.2 seconds |

These are fixed-seed baseline runs on one CPU machine, not confidence
intervals or universal claims. Multiple seeds belong to the later benchmark
stage. The different MNIST and CIFAR-10 percentages also describe different
task difficulty and should not be treated as a direct model-quality ranking.

### Gate status and required student action

All Week 1–4 technical artifacts are ready. Gate 1 nevertheless remains
unchecked because the roadmap requires the student—not the code—to explain
how training works, how loss behaves, how CNN filters create feature maps, why
validation and test sets are separate, and why measured accuracy changes.

`reports/gate1_self_check.md` provides ten beginner questions. The student
should answer them in their own words and connect the explanations to the
measured Month 1 results. Only after those answers are reviewed should Gate 1
be checked and Month 2 FedAvg work begin. FedAvg, Non-IID partitioning,
FedProx, and all Federated Unlearning implementations remain blocked for now.

---

## 2026-08-16 — Gate 1 evidence audit and exact config-only reruns

### Why a second audit was necessary

The Week 1–4 experiments and report were working, but an independent evidence
audit found two important differences between “the files exist” and “the
milestone is reproducible”:

1. Most Month 1 files had not yet been committed. A folder can contain a
   `.git` directory while new files remain untracked. Untracked work has no
   stable historical version to recover or cite.
2. The neural runners selected CPU or GPU from machine availability. A saved
   config could therefore behave differently on another machine even though
   the JSON itself had not changed. The general `requirements.txt` also used
   lower version bounds, which are useful for development but do not identify
   the exact environment that produced the reported numbers.

The audit also noticed that the PDF builder loaded `metrics.json` but repeated
some headline numbers as typed text. Those copies matched at the time, yet
they could become stale after a future rerun. These were evidence and
provenance weaknesses, not model-accuracy failures, and they were corrected
before declaring the technical side of Gate 1 ready.

### What changed in every Month 1 config

Each JSON config now records:

- `device: "cpu"`, so an available GPU cannot silently change execution;
- `runner`, the exact script that interprets the config;
- `dataset_version`, so the dataset identity is explicit;
- `environment_manifest`, pointing to the verified Python/package versions;
- `strict_environment: true`, which stops before training if those versions
  differ;
- `code_revision: "month1-gate1"`, the Git tag identifying the complete
  Month 1 source and documentation.

The CIFAR-10 config additionally records the archive filename, exact size, and
MD5. The runtime manifest is
`environment/month1_cpu_runtime.json`, and its exact direct-package lock is
`environment/month1_cpu_requirements.txt`. The verified setup uses Python
3.10.20, PyTorch 2.12.1 on CPU, TorchVision 0.27.1, and fixed seed 42.

This does not mean every future experiment must use these old versions. It
means a result must name the environment that produced it. A later milestone
can create a new manifest deliberately rather than changing the meaning of an
old config silently.

### Stronger automatic result evidence

The runners now save additional evidence into `metrics.json` automatically:

- the actual Python, platform, device, and package snapshot;
- all class names and the complete numeric confusion matrix;
- the true and predicted labels shown in each CNN feature-map figure;
- the source config, runner, dataset version, environment manifest, and code
  tag inherited from the config.

Saving the confusion matrix as numbers matters. A plot is helpful for a human,
but code cannot reliably recover exact counts from image pixels. With the
numeric matrix, the report can calculate the largest error and the verifier
can prove that all matrix cells sum to the test-set size.

### Clean config-only rerun results

All four experiments were started again using only their documented commands
and saved JSON configs. They reproduced the original metrics exactly:

| Experiment | Test accuracy | Macro F1 | Fresh CPU training time |
|---|---:|---:|---:|
| Week 1 MNIST logistic regression | 92.19% | 92.08% | 29.2 seconds |
| Week 2 MNIST MLP | 96.80% | 96.77% | 103.2 seconds |
| Week 3 MNIST CNN | 98.60% | 98.60% | 407.2 seconds (6.8 minutes) |
| Week 3 CIFAR-10 CNN | 71.38% | 71.07% | 1,075.7 seconds (17.9 minutes) |

Training time changes slightly with background computer activity; accuracy
and macro F1 are the deterministic result checked here. The reruns did not
change the architectures, optimizers, learning rates, data splits, epochs, or
seed. Their purpose was to prove the strengthened configs reproduce the same
experiments, not to search for higher accuracy.

### Report correction and verification

`reports/build_month1_report.py` now calculates its experimental statements
from the fresh JSON results. Headline accuracies, model improvements, split
sizes, parameter counts, epoch counts, training times, class rankings, largest
confusion, feature-map labels, and the CIFAR archive checksum are no longer
duplicated as hand-typed result numbers.

The rebuilt `output/pdf/month1_experiment_report.pdf` remains eight pages. Its
structural verifier passed, it was rendered to eight PNG pages, and every page
was visually inspected again. All text, tables, figures, captions, headers,
and footers remain readable with no clipping or overlap.

### Automated Gate 1 technical verifier

`reports/verify_gate1_artifacts.py` now checks all non-student evidence in one
command. Among other checks, it verifies:

- all mandatory experiment and provenance fields;
- config/result agreement and measured accuracy/F1 ranges;
- exact Python and package versions against the lock;
- numeric confusion-matrix shape and test-set totals;
- CIFAR-10 archive size and MD5;
- strict loading and output shapes for all three neural checkpoints;
- the six SISA questions and nine literature-matrix columns;
- report page count, measured values, and absence of hard-coded current
  headline values in the builder;
- Git tracking, a clean worktree, and the `month1-gate1` tag pointing to the
  committed evidence.

Raw datasets, checkpoints, plots, and `metrics.json` outputs remain ignored as
required by the project rules. They are reproduced from versioned configs;
the final PDF is committed as the Month 1 deliverable. The source, configs,
environment locks, literature notes, learning material, verifier, and report
are versioned together under the local `month1-gate1` tag. Nothing was pushed
or published to the remote repository.

### Gate status

The technical and version-control conditions for Month 1 are now proven. Gate
1 remains deliberately unchecked for one reason only: the roadmap requires
the student to explain how training works, how loss behaves, and why accuracy
changes. The new `reports/gate1_study_guide.md` teaches those ideas with the
measured project examples, while `reports/gate1_evidence.md` separates
technical proof from human-understanding proof.

The next action is still the student's ten-question explanation check.
FedAvg and all Month 2 implementation remain blocked until those answers are
reviewed and Gate 1 is checked.

## 2026-08-16 — Gate 1 passed: student explanation review

The student completed all ten questions in
`reports/gate1_self_check.md`. The answers were reviewed separately from the
automatic artifact audit because a script can confirm files and numbers, but
it cannot decide whether a person understands what those numbers mean.

### Review decision

**Gate 1 passed: 10 of 10 answers meet the required standard.** The student
demonstrated the three abilities named explicitly by Gate M1:

1. **How training works:** the answer correctly described the forward pass,
   cross-entropy calculation, `loss.backward()`, and `optimizer.step()`. The
   important distinction is clear: backward computes gradients, while the
   optimizer uses those gradients to change the weights.
2. **How loss works:** the answer correctly explained that accuracy records
   whether the highest-scoring class is correct, while cross-entropy also
   reacts to confidence. Therefore, loss can improve while the number of
   correct predictions—and thus accuracy—stays unchanged.
3. **Why accuracy changes:** the answers connected performance to learned
   weight updates, model structure, dataset difficulty, augmentation,
   dropout, and evaluation protocol instead of treating accuracy as an
   isolated number.

The remaining answers also showed that the student can separate train,
validation, and test roles; explain filters, feature maps, and pooling; use
the measured 96.80% MLP and 98.60% CNN results without making a universal
claim; explain why CIFAR-10 is harder than MNIST; distinguish a random seed,
configuration, and checkpoint; and identify the bird/cat/dog weaknesses that
later motivate class- and client-level Non-IID analysis.

Two precision notes were recorded for future use. First, logits are raw class
scores; PyTorch cross-entropy performs the internal transformation needed to
compare them with the label. Second, the observed CNN advantage is evidence
for this model/configuration and single seed, not proof that every CNN always
outperforms every MLP. These are refinements, not errors, so no re-answer is
required.

### Repository and roadmap update

- `README.md` now checks Gate 1 and sets Month 2, Week 5 as the current unit.
- `reports/gate1_evidence.md` now records the human-understanding evidence as
  passed rather than waiting.
- `reports/verify_gate1_artifacts.py` now supports both the historical open
  state and the reviewed closed state. The original `month1-gate1` tag remains
  the exact experiment revision, while `gate1-complete` identifies the later
  review decision.
- The historical Month 1 PDF is not rewritten merely to alter its status
  paragraph. Its experiment results remain tied to the original revision;
  the self-check, evidence checklist, README, progress log, and completion tag
  provide the dated closure record.

Gate discipline now permits **Month 2, Week 5 only**: learn the FL roles and
workflow and read McMahan et al. (2017) using the fixed six-question template.
Hand-written FedAvg belongs to Week 6 and has not been started. An FL
framework, FedProx, and every Federated Unlearning method remain blocked by
their later gates.

## 2026-08-16 — Month 2, Week 5 learning package prepared

Work began on Month 2 only after Gate 1 was reviewed, checked, committed, and
tagged. This unit prepares the Federated Learning concepts that must be
understood before writing FedAvg. It does not claim that the student has
completed Week 5 merely because explanatory files now exist.

### Primary-paper reading

`literature/fedavg_2017_six_questions.md` records the McMahan et al. (2017)
paper using exactly the six required headings: Problem, Gap, Idea, Assumption,
Evaluation, and Limitation. The note was checked against the original
AISTATS/PMLR paper rather than a secondary tutorial.

The central paper result is that clients can perform several local optimizer
steps before communicating and the server can sample-weight their returned
models. This often exchanges more inexpensive local computation for fewer
expensive communication rounds. The note also preserves qualifications that
are important for research integrity:

- the experiments are primarily empirical and controlled/synchronous;
- a common global initialization is important before model averaging;
- very large local-epoch counts can plateau or diverge;
- raw data remaining local does not prove that model updates reveal nothing;
- the study establishes ordinary FL, not Federated Unlearning.

The paper's row was added to `literature/literature_matrix.md` with all nine
required columns. Its “Possible Gap” cell is explicitly a later research
question and does not select the thesis contribution before Week 19.

### Beginner learning guide

`reports/month2_week5_fl_concepts.md` connects FL to the Gate 1 training
loop. The guide explains:

- client, server, local model, global model, communication round, local epoch,
  and aggregation;
- one complete server → clients → server round;
- that `loss.backward()` computes local gradients and
  `optimizer.step()` changes local weights, while server aggregation creates
  the next global weights;
- why FedAvg weights local models by example count;
- the roles and trade-offs of client fraction \(C\), local epochs \(E\), batch
  size \(B\), and learning rate \(\eta\);
- the introductory difference between IID and Non-IID clients;
- common misconceptions, including the difference between data locality and a
  formal privacy guarantee.

The weighted-average example uses clients with 100, 200, and 700 examples. If
their returned scalar weights are 1.0, 2.0, and 3.0, FedAvg produces 2.6, not
the unweighted client mean of 2.0. This example makes the meaning of
sample-count weighting concrete before it appears in code.

### Required student action

`reports/month2_week5_self_check.md` contains ten beginner questions. The
student must trace a round, distinguish local optimization from aggregation,
calculate a two-client weighted average, explain \(C/E/B\), discuss the
privacy caveat, and connect ordinary FedAvg to the later unlearning problem.

Week 5 remains **in progress** until those answers are supplied and reviewed.
No experiment was run in this reading/concept unit, so no experiment config or
result file was created. Week 6 FedAvg code, Week 7 framework use, Month 3
Non-IID/FedProx, and all Federated Unlearning code remain blocked.
