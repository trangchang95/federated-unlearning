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

## 2026-08-17 — Month 1 literature sanity check completed

The four completed centralized baseline accuracies were checked against
original papers and authoritative official framework examples. This work asks
a narrow question: are the saved results believable for the corresponding
dataset and model family? It does not rerun training, tune a model, or treat
experiments with different protocols as if they were directly comparable.

### Evidence method

`reports/month1_literature_sanity_check.md` labels three kinds of statement
separately:

1. **Project facts** come from the saved Month 1 configs, runners,
   architectures, and `metrics.json` outputs.
2. **Literature facts** come from an exact paper table/figure/page or a named
   section of an official framework example.
3. **Inferences** are the conservative plausibility decisions made after the
   two fact sets are placed together.

This separation matters for a beginner because seeing two nearby accuracy
numbers does not mean the experiments are equivalent. A different train/test
split, model, optimizer, epoch count, augmentation rule, or random seed can
change the result.

The check used two sources for each MNIST linear/MLP family, two for the
MNIST CNN, and two for the CIFAR-10 CNN. Peer-reviewed or original benchmark
papers were used where a suitably clear baseline existed. For modest
CIFAR-10 CNNs, official TensorFlow and PyTorch examples were used because they
expose their small architectures and training budgets more clearly than the
highly optimized research models that would give misleading SOTA comparisons.

### Result-by-result decision

| Project baseline | Measured accuracy | Literature sanity decision |
|---|---:|---|
| MNIST Logistic Regression | 92.19% | `plausible / consistent with literature` |
| MNIST one-hidden-layer MLP | 96.80% | `plausible / consistent with literature` |
| MNIST Small CNN | 98.60% | `plausible / consistent with literature` |
| CIFAR-10 Small CNN | 71.38% | `plausible / consistent with literature` |

The closest linear benchmark reports 91.7% MNIST accuracy; shallow MLP
references report 95.3% and 97.2%; modest primary-paper MNIST CNN references
report roughly 99.01%–99.05% with longer training; and the closest official
CIFAR-10 example reports 71.63% for a similarly scaled 10-epoch Adam run.
These facts make the project results believable. They do not prove
implementation correctness, superiority, replication, or statistical
equivalence.

One especially important protocol difference was preserved in the report.
The TensorFlow CIFAR-10 tutorial monitors the official test set as validation
data every epoch. The project instead reserves 5,000 training examples for
validation and leaves the canonical 10,000-image test set for final
evaluation. Therefore, the 71.38% and 71.63% values must not be presented as a
controlled 0.25-percentage-point agreement.

### Literature records and limitations

Three newly used papers were recorded with exactly the plan's six questions—
Problem, Gap, Idea, Assumption, Evaluation, and Limitation:

- `literature/lecun_1998_six_questions.md`;
- `literature/fashion_mnist_2017_six_questions.md`;
- `literature/scherer_2010_six_questions.md`.

Their required rows and persistent DOI/arXiv links were added to
`literature/literature_matrix.md`. Official tutorials are recorded in the
sanity report's verified-source ledger rather than treated as research papers.

The main limitation remains unchanged: each project neural-network result is
currently one fixed-seed run. A fixed seed supports repeatability, but it does
not show how much accuracy varies across seeds. The correct conclusion is
therefore **literature sanity only**. No experiment was run in this unit, so no
new config/result pair was required. Gate 1 remains complete, Month 2 Week 5
remains the current learning unit, and no later-gate implementation was
started.

---

## 2026-08-17 — Consistency and reproducibility audit of Month 1

### Why

The Month 1 report and this log had not been independently checked end-to-end
against the actual repository state (git history, the remote, and the PDF's
extracted text) since Gate 1 closed. This entry documents three things that
audit found and corrected, plus one that it found and is recording as a known
limitation.

### Correction: the repository has in fact been pushed to a public remote

The 2026-08-16 "Gate 1 evidence audit" entry above states *"Nothing was
pushed or published to the remote repository."* That statement is wrong.
`git remote -v` shows `origin` pointing at
`https://github.com/trangchang95/federated-unlearning`, `git reflog show
origin/main` shows two prior pushes, and the GitHub page confirms the
repository is public with all commits through Month 2 Week 5 visible. No raw
dataset, checkpoint, or `results/` output leaked — `.gitignore` correctly
excludes `data/*` and `results/*`, so only source code, configs, the
environment lock, literature notes, and the built PDF report are public,
which is not sensitive information. The decision, confirmed with the student,
is to leave the repository public and correct this log rather than change
visibility. Future entries should not repeat the earlier claim; the repo has
been public since at least the Week 1 commit's push.

### Fixed: the Month 1 PDF had a text-extraction bug in every bullet point

`reports/build_month1_report.py` built its bullet lists with ReportLab's
named `"circle"` glyph (Unicode U+25CF), drawn through ReportLab's internal
bullet-rendering path rather than as ordinary paragraph text. That path
produced a broken `ToUnicode` mapping in the embedded font: the bullet
renders as a normal dot on screen, but copying text from the PDF, running it
through a screen reader, or extracting it programmatically (confirmed with
`pypdf`) returned `â—�` instead of the bullet. This would not have been
caught by the earlier visual page-by-page inspection, since it only affects
extracted text, not the rendered image. The fix switches the bullet to a
literal ASCII hyphen, which uses the same text path as normal body copy and
is confirmed to extract cleanly. The report was rebuilt; page count (8),
structural verification, and every measured number are unchanged — only the
bullet character and one status sentence (below) differ from the previous
PDF bytes.

### Fixed: the PDF's Gate 1 status line was stale the same day it was written

Page 8 of the original PDF said *"Gate 1 status: technical artifacts are
ready for review, but the README gate must remain unchecked..."* — true when
that paragraph was drafted, but Gate 1 was reviewed and passed later the same
day (see the "Gate 1 passed" entry above), and the PDF was deliberately never
rewritten to track that, per the reasoning given in that entry. On reflection,
a process-status sentence is different from a measured result: freezing
accuracy numbers to their producing run is correct practice, but leaving a
now-false status claim in the delivered report is not something a reader
handed only the PDF (e.g. a supervisor) could detect. The sentence was
updated to state the actual outcome (passed 16 August 2026) and point to
`README.md`/`PROGRESS.md` for the current milestone. This is a wording
correction, not a re-litigation of the Gate 1 decision itself.

### Known limitation, not fixed: Week 1 logistic regression is not bit-reproducible run-to-run

Comparing the very first Week 1 run (validation accuracy 91.71%, test 92.20%)
to the current `results/month1_week1/metrics.json` (validation accuracy
91.76%, test 92.19%) shows a difference of one flipped prediction out of
10,500 test examples. The 2026-08-16 "compliance update" entry describes this
as "only rounding," which is not accurate — 92.20% and 92.19% at four
decimal places are two different underlying counts of correct predictions,
not the same value shown with different precision. The likely cause is
non-deterministic multi-threaded BLAS reduction order in scikit-learn's
`lbfgs` solver, which `random_state=42` does not fully control. This is worth
recording plainly because the project leans heavily on "seed 42 makes a run
reproducible": that claim holds exactly for the PyTorch experiments (Weeks
2-3 reran bit-for-bit identical accuracy/F1 across the Gate 1 audit reruns)
but only approximately for the scikit-learn baseline, to within about one
example in ten thousand. The gap is too small to change any conclusion in
this project, so no rerun was triggered; if exact determinism ever matters
later, pinning `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` before training
would be the fix. `metrics.json` remains the authoritative current value; the
91.71%/24.8s pair in the original Week 1 entry above is superseded evidence
from before the config-driven rerun, not a data-entry error.

### What was not changed

The measured accuracy, F1, timing, and split numbers throughout this log and
the PDF are unchanged. This entry only corrects process claims (the push
statement, the stale gate sentence) and the PDF's non-text-extractable
bullet glyphs; it does not rerun any experiment or alter any reported result.

---

## 2026-08-18 — Month 2, Week 5 passed: FL concept review

The student supplied answers to all ten questions in
`reports/month2_week5_self_check.md`. They were reviewed against the beginner
guide and the primary FedAvg paper note. The result is **PASS (10/10)**, so
Week 5 is complete.

### What the answers demonstrate

The student can now explain the complete server-to-clients-to-server workflow
without treating Federated Learning as a black box. In particular, the
answers correctly distinguish two different weight-changing operations:

1. `optimizer.step()` acts during local training and changes one client's
   local model using gradients from that client's minibatch;
2. aggregation acts after local training and combines the returned client
   models into the next global model.

The weighted-average calculation is also correct: for returned scalar weights
1.0 and 3.0 from clients with 20 and 80 examples, respectively, FedAvg gives
`0.2(1.0) + 0.8(3.0) = 2.6`, not the unweighted mean 2.0. The remaining
answers correctly separate a local epoch from a communication round, explain
the roles of client fraction `C`, local epochs `E`, and batch size `B`, and
identify the trade-off between fewer communications and greater client drift.

The privacy and evaluation answers are important for later thesis work. The
student correctly states that local raw data alone does not provide a formal
privacy guarantee because model updates can reveal information. The student
also explains that sample-weighted global accuracy can hide poor performance
for a small client, so later experiments must report client-level utility.

### Gate and next step

`README.md` now checks Week 5 and identifies Month 2, Week 6 as the current
unit. Gate 2 remains open: no FedAvg implementation or multi-client experiment
has been claimed yet. The next permitted task is a hand-written FedAvg
implementation with explicit local training and sample-weighted aggregation.
An FL framework remains blocked until that implementation works; Non-IID,
FedProx, and Federated Unlearning remain blocked by later gates.

No experiment was run for this conceptual review, so no config or result file
was created.

---

## 2026-08-18 — Month 2, Week 6 completed: FedAvg implemented by hand

Week 6 implements the FedAvg mechanism directly, without Flower or another FL
framework. This order matters: the project can now inspect every local weight
update and server aggregation step instead of depending on a library before
the algorithm is understood.

### What was implemented

The responsibilities are deliberately separated across three modules:

1. `algorithms/fedavg.py` performs the sample-count-weighted average of model
   state tensors. It validates client counts and tensor compatibility, rejects
   non-finite values, avoids mutating/aliasing inputs, and gives an explicit
   policy for non-floating buffers.
2. `clients/federated_client.py` deep-copies the current global model, creates
   a fresh local SGD optimizer, trains only on one client's dataset, and
   returns detached model tensors plus the exact `len(dataset)` count.
3. `server/fedavg_server.py` deterministically selects clients, keeps the
   round-start global model unchanged while all selected clients train,
   aggregates their results, and only then installs the next global model.

For selected clients, the implementation computes
`sum((n_k / sum_selected(n_j)) * local_state_k)`. The beginner scalar case is
verified directly: 20 examples returning 1.0 and 80 examples returning 3.0
produce `0.2(1.0) + 0.8(3.0) = 2.6`.

The server also records a transparent communication estimate. A dense state
payload is the sum of `tensor elements × bytes per element`. Each selected
client counts one full-model download and one full-model upload. This is a
repeatable tensor-byte estimate, not measured network bandwidth or latency.

### Verification evidence

`tests/test_fedavg.py` contains 13 deterministic synthetic checks. The command

```powershell
conda run -n mse-ai python reports\verify_week6_fedavg.py
```

passed on 2026-08-18. It proves, among other invariants:

- exact sample-count weighting, including the 20/80 example;
- no mutation or shared tensor storage between local/global states;
- fresh, isolated client training and preserved caller RNG state;
- one-client/full-batch equivalence to centralized SGD;
- an unequal 4/2-client server round matching an explicit manual aggregate;
- tensor-exact reproducibility across two independent two-round CPU runs;
- round 2 starts from the updated round-1 state;
- a hand-calculated 24-byte model payload and matching upload/download totals;
- clear rejection of invalid, incompatible, or non-finite updates; and
- absence of any active/imported FL framework in the Week 6 code.

`reports/month2_week6_fedavg_implementation.md` explains the complete workflow
and its limitations in beginner language.

### Scope and roadmap decision

No MNIST experiment was run in Week 6. The checks use tiny in-memory tensors
and datasets because their purpose is algorithm correctness; they are unit
tests, not research results, so no experiment config/result pair was created.
They do not establish classification accuracy, convergence, privacy, real
client isolation, network performance, IID/Non-IID behavior, or a centralized
comparison.

`README.md` now checks Week 6 and moves the current unit to Week 7. A framework
may now be inspected because the hand-written implementation works. Gate 2
remains open until the Week 8 multi-client MNIST experiment and
centralized-versus-FedAvg comparison are complete and understood. FedProx,
Non-IID experiments, and every Federated Unlearning method remain blocked.

---

## 2026-08-18 — Month 2, Week 7 completed: Flower inspected transparently

Flower was introduced only after the hand-written FedAvg implementation and
its 13 tests passed. The framework is used as a compatibility reference, not
as a replacement for `algorithms/fedavg.py`.

### Version and environment decision

The current Flower line has moved beyond version 1.30, but pip metadata and
the official changelog show that Flower 1.31 and newer require Python 3.11.
The project's verified `mse-ai` runtime uses Python 3.10.20. Flower 1.30.0 is
therefore the newest compatible release for this environment and is pinned in
`requirements.txt` and `environment/month2_cpu_requirements.txt`. The exact
runtime is recorded in `environment/month2_cpu_runtime.json`.

This avoids changing Python under previously reproduced work merely to claim
the newest framework version. The report states the constraint plainly; it
does not describe 1.30.0 as Flower's newest release overall.

### What was inspected and compared

`server/flower_compat.py` converts copied PyTorch state tensors to the NumPy
arrays used by Flower 1.30, calls Flower's real weighted aggregation helper,
and converts copied results back to the original tensor order, shape, dtype,
and device. The hand-written implementation remains untouched and is compared
against this independent framework path.

The source-level mapping is:

- the project's server corresponds to Flower's server/strategy role;
- `train_client` corresponds to a client `fit` path;
- returned tensor state corresponds to serialized `Parameters`;
- the project's example count corresponds to `FitRes.num_examples`; and
- both aggregation paths multiply each returned layer by the client example
  count, add corresponding layers, and divide by the total count.

`reports/month2_week7_flower_compatibility.md` explains these mappings and the
limits of this comparison in beginner language.

### Verification and a meaningful difference

The command

```powershell
conda run -n mse-ai python reports\verify_week7_flower.py
```

passed with Flower 1.30.0 and **20 tests**: 13 hand-written FedAvg tests plus
seven framework compatibility tests. Flower matches the hand-written result
for the 20/80 scalar example (2.6) and an unequal two-tensor example. Both the
out-of-place helper and Flower's default in-place `FedAvg.aggregate_fit` path
are checked. Dtype, shape, key order, non-aliasing, and the supported NumPy
float dtypes are explicit.

The audit also records a real default-policy difference. The project selects
`ceil(C × K)` clients. With `C=0.3` and five clients, that means two clients.
Flower 1.30's legacy strategy uses integer truncation plus its minimum-client
setting; with `min_fit_clients=1`, it selects one. Aggregation agrees once the
same clients return, but different selection can change convergence and
communication, so future results must record the policy instead of calling
the two paths identical.

### Scope and next step

No dataset training, accuracy result, real network process, or privacy claim
was produced in Week 7. These are compatibility tests, not experiments, so no
experiment config/result pair was created. No Non-IID, FedProx, or Federated
Unlearning code was introduced.

`README.md` now checks Week 7 and makes Week 8 the current unit. Gate 2 remains
open. The next task is a modest, config-driven IID MNIST comparison using the
hand-written FedAvg implementation and a matched centralized SGD baseline.

---

## 2026-08-18 — Month 2, Week 8 completed: first multi-client MNIST comparison

Week 8 now has a real, config-driven experiment comparing the hand-written
FedAvg workflow with centralized SGD. This completes the technical Month 2
deliverable, but it does not close Gate 2 yet: the student explanation and the
required `K`/`E` changes still need review.

### Reproducibility was frozen before training

The experiment is defined by
`configs/month2_week8_mnist_iid_fedavg_vs_centralized.json` and executed by
`experiments/iid/month2_week8_mnist_fedavg.py`. Before MNIST training began:

- all 46 synthetic/unit tests passed;
- the exact Python 3.10.20 CPU environment passed validation;
- the producing files were committed as `2fb17f49b27a3b0d57a3e3edafa8e2bf514af549`;
- tag `month2-week8` was created at that same commit;
- the worktree was clean; and
- `reports/verify_week8_fedavg.py` returned `PRE-RUN` rather than pretending
  that a missing experiment had already passed.

The runner records the resolved tag/commit and refuses an untracked, dirty, or
mismatched revision. It also refuses to overwrite an existing result unless
`--overwrite` is supplied deliberately.

### What was compared

The canonical protocol uses MNIST with 51,000 training, 9,000 validation, and
10,000 official test examples. Five clients receive deterministic, disjoint,
equal-sized IID training shards of 10,200 examples each. The settings are:

| Symbol/setting | Value |
|---|---:|
| Total clients `K` | 5 |
| Client fraction `C` | 1.0 (all five every round) |
| Local epochs `E` | 1 |
| Batch size `B` | 128 |
| Communication rounds `R` | 5 |
| Centralized epochs | 5 |
| SGD learning rate | 0.1 |
| Fixed main seed | 42 |

Both methods use the same training split, `SimpleMLP` architecture, initial
weights, plain SGD settings, batch size, and 255,000 training-example
exposures. Validation selects each method's checkpoint. The test set is
evaluated only after that selection.

This is a matched-*exposure* comparison, not identical optimization.
Centralized SGD follows one continuous trajectory through the complete split.
FedAvg creates five local trajectories during each round and then performs a
sample-weighted average of their model states.

### Measured result

| Method | Test loss | Test accuracy | Macro F1 | Best validation step |
|---|---:|---:|---:|---:|
| Centralized SGD | 0.187003 | 94.75% | 94.68% | epoch 5 (94.31%) |
| Hand-written FedAvg | 0.319598 | 90.99% | 90.86% | round 5 (90.28%) |

FedAvg is **3.76 percentage points lower in test accuracy** and 3.82 points
lower in macro F1. This negative comparison is reported plainly. It does not
mean the FedAvg implementation failed: its validation accuracy rises from
10.42% before training to 84.73%, 88.12%, 89.29%, 89.88%, and 90.28% over the
five rounds, while validation loss falls every round. It also does not prove
that centralized training is universally better; this is one architecture,
one IID split, one training budget, and one fixed seed.

The five IID client test subsets each contain 2,000 examples. Centralized
client accuracies range from 94.10% to 95.50%; FedAvg client accuracies range
from 90.35% to 91.50%. These values make client-level utility visible, but
they are not Non-IID results and there is no unlearning target in Week 8.

### Why the step and communication totals differ

Centralized training performs
`ceil(51,000 / 128) × 5 = 399 × 5 = 1,995` optimizer steps. Each FedAvg client
has 10,200 examples, so each performs `ceil(10,200 / 128) = 80` steps per
round. Across five clients and five rounds, FedAvg performs
`80 × 5 × 5 = 2,000` local optimizer steps. The five-step difference comes
from rounding each client's final partial minibatch separately; it is not
extra training data.

The `SimpleMLP` dense state is 407,080 bytes. Counting one full-model download
and one upload for every client in every round gives
`407,080 × 2 × 5 × 5 = 20,354,000` bytes (19.411 MiB). This is deterministic
tensor-payload accounting, not real network bandwidth, latency, or total
protocol traffic.

### Exact rerun and artifact verification

The canonical run was executed a second time with the same tagged code,
config, environment, and seeds. `reports/compare_week8_reruns.py` compared the
two complete JSON trees after removing only the two measured timing fields.
Every remaining value was exactly identical, including splits, client
partitions, histories, losses, accuracies, F1 values, communication totals,
and checkpoint hashes. CPU times changed from 111.384 to 90.148 seconds for
centralized training/validation and from 92.255 to 88.971 seconds for FedAvg,
which is expected and is why timing is not treated as deterministic.

The final command

```powershell
conda run -n mse-ai python reports\verify_week8_fedavg.py
```

passed the config, environment, Git provenance, scope, split, partition,
history, optimizer-step, communication, metric, checkpoint, plot, generated
summary, and durable-report checks. The detailed versioned report is
`reports/month2_week8_centralized_vs_fedavg.md`; raw outputs remain under the
gitignored `results/month2_week8_mnist_iid_comparison/` directory.

### Gate and next step

`README.md` now checks Week 8. `reports/gate2_self_check.md` asks the student to
explain the result and convergence curve, then propose and run separate
config-driven changes for `K=10` and `E=2` without overwriting the canonical
evidence. Gate 2 remains unchecked until those answers and hands-on results
are reviewed. Non-IID partitioning, FedProx, and Federated Unlearning remain
blocked; none was introduced in this unit.

---

## 2026-08-18 — Gate 2 readiness made explicit without doing the student's work

The technical Week 8 experiment was already complete, but Gate 2 could not be
closed yet because its educational evidence was still blank. A separate
readiness workflow now makes that boundary machine-checkable without answering
the questions, inventing predictions, creating variant configs, or running the
required hands-on changes for the student.

### Three states now have different meanings

`reports/verify_gate2_readiness.py` always verifies the protected canonical
Week 8 evidence first. It then reports one of three states:

- `WAITING`: valid required evidence has not been supplied yet;
- `FAIL`: an existing file, claim, path, result, or gate marker contradicts
  the reproducibility rules; or
- `READY`/`PASS`: both controlled variants and the human learning review are
  complete, with `PASS` reserved for a committed gate closure.

The current state is deliberately `WAITING`. The checker lists the exact
remaining work: Questions 1–8, the two predictions, pre-run reviewer approval,
the tagged `K=10` and `E=2` configs/runs, both post-run interpretations, and a
dated final conceptual review. `reports/gate2_variant_evidence.json` starts in
`awaiting_student` state with null variant paths, so missing experiments cannot
be mistaken for zero-valued or unsuccessful experiments.

### Why the workflow has two human reviews

The self-check now separates **pre-run proposal approval** from the final Gate
2 review. The producing Git tag must already contain the student's eight
answers, both predictions, and `APPROVED TO RUN`; therefore the predictions
are auditable as predictions made before training. A SHA-256 snapshot binds
that approved pre-run text across both producing tags and the final self-check,
so a prediction cannot be rewritten after its result is known. After both
runs, the student must fill the saved-result sections and receive a separate
dated `PASS` that cannot predate the proposal approval. The script checks only
that substantive fields exist and that the review markers are coherent. It
does not use keywords to pretend it can judge whether the explanations are
conceptually correct.

### Variant evidence is protected before it can close the gate

The future variants are restricted to the intended changes:

- `K=10`, while `C=1`, `E=1`, `B=128`, `R=5`, and five centralized epochs
  remain fixed; and
- `E=2`, while `K=5`, `C=1`, `B=128`, and `R=5` remain fixed and centralized
  epochs become ten to preserve the matched exposure budget.

The verifier normalizes Windows paths so spellings such as `folder` and
`folder/.` cannot alias the canonical or sibling output directory. It requires
tagged configs, clean-start provenance, unchanged result-producing code, the
exact Month 2 runtime, seeded IID splits, complete histories, derived optimizer
steps/exposures/communication, internally consistent confusion and per-client
counts/accuracy, loadable checkpoints, valid plots, and reports/summaries
generated exactly from saved JSON. Per-client macro F1 remains a structurally
range-checked runner-reported value because the frozen runner does not log a
confusion matrix for each client; it must not be described as independently
recomputed. The arithmetic is derived from each config rather than accepted as
hand-copied constants. No accuracy direction or monotonic curve is required,
so a negative variant result remains reportable.

### Verification and scope

All 70 tests pass, including 24 Gate 2 readiness tests for controlled config
deltas, matched budgets, path aliases, placeholder answers, explicit state
transitions, low-accuracy-but-valid evidence, and the exact closure marker.
The deterministic Week 8 report check and the full canonical Week 8 post-run
verifier still pass. The canonical result-producing files were not changed.

The README exact-rerun instructions were also corrected: the runner must be
executed from tag `month2-week8`, because a newer `main` commit is intentionally
not allowed to impersonate the producing revision. For the normal beginner
readiness command, `--allow-waiting` suppresses Conda's misleading error footer
while leaving the printed state as `WAITING`; automated gate checks omit that
flag and receive exit code 2.

Gate 2 remains unchecked. No Non-IID partitioning, FedProx, Federated
Unlearning, or proposed method was started.

---

## 2026-09-06 — Gate 2 Part A reviewed: FedAvg round mechanics and Week 8 result

The student answered all eight Part A questions in
`reports/gate2_self_check.md`, tracing one FedAvg round, explaining why every
selected client must start from the same global model, defining `K`/`C`/`E`/
`B`/`R` with their values in the canonical run, separating optimizer-step
counts from training-example exposures, reporting the 3.76-percentage-point
FedAvg-versus-centralized gap without overgeneralizing it, distinguishing
"validation curves improved" from "the method provably converged," reconstructing
the communication-byte formula and naming what it omits, and explaining why
per-client utility and single-seed repeatability are each narrower claims than
they might first appear.

### Review decision

**Part A: pass (8 of 8 answers meet the required standard.)** The answers
correctly separate `optimizer.step()` (client-side, local weight change) from
sample-weighted aggregation (server-side, next global model), correctly state
the canonical run's `K=5, C=1.0, E=1, B=128, R=5`, and correctly avoid two
common overclaims this gate specifically checks for: treating one run's
negative FedAvg result as universal, and treating rising validation curves as
proof of convergence to a stable limit.

One precision note, not requiring a re-answer: question 7's answer explains
the communication formula and its omissions correctly but does not show the
numeric substitution — the `SimpleMLP` state is 407,080 bytes, and
`407,080 × 2 × 5 × 5 = 20,354,000` bytes is the reported estimate. This is the
same kind of refinement recorded (without demanding a rewrite) during the
Gate 1 review.

### What remains open

Part B (Questions 9–10: propose a `K=10` variant and an `E=2` variant, each
with a written prediction) is still blank, along with the pre-run proposal
review and the final review section. Per this repository's rule against
inventing predictions before an experiment runs, and per the gate's own audit
design — the producing Git tag must already contain the approved predictions,
hashed so they cannot be edited after the result is known — these two
proposals and predictions must be written by the student before any `K=10` or
`E=2` config is created, tagged, or run. No such config was created in this
session. Gate 2 therefore remains open; Month 3 Non-IID/FedProx work and all
Federated Unlearning implementation remain blocked.

---

## 2026-09-07 — Gate 2 variant setup: tag-gap correction and K=10/E=2 configs prepared

### Correction: three documented Git tags did not actually exist

Preparing to run the Part B variants required resolving the `month2-week8`
tag (the Gate 2 readiness verifier hard-depends on it to locate the canonical
commit). `git tag -l` and `git show-ref --tags` returned nothing on this
clone, and `git reflog --all` and `git fsck --unreachable --tags` showed no
trace that any tag had ever existed, on this clone or on `origin`. This
contradicts three earlier entries above, which state that `month1-gate1`,
`gate1-complete`, and `month2-week8` were each created.

The commits those entries already name were verified to exist with matching
content — `2fb17f49b27a3b0d57a3e3edafa8e2bf514af549` (named in the Week 8
entry) is the exact commit whose config already reads
`"code_revision": "month2-week8"`. So this was a documentation gap, not a
lost or divergent commit: the tags were retroactively created pointing at the
exact commits already on record, not at any new or different commit:

| Tag | Commit | Matches entry |
|---|---|---|
| `month1-gate1` | `b1934ec` (Month 1: reproducible centralized baselines and Gate 1 evidence) | 2026-08-16 Gate 1 evidence audit |
| `gate1-complete` | `3b45e3e` (Gate 1: record student self-check pass) | 2026-08-16 Gate 1 passed |
| `month2-week8` | `2fb17f4` (Complete literature audit and prepare verified FedAvg run) | 2026-08-18 Week 8 completed |

No commit content changed; only the missing tag pointers were added. Future
entries should not repeat the earlier "tag was created" claims as evidence of
anything beyond the commit existing — verify with `git tag -l` before citing
a tag as proof.

### Correction: the canonical Week 8 config had an uncommitted, unintended edit

Before this session, `configs/month2_week8_mnist_iid_fedavg_vs_centralized.json`
had a dirty (uncommitted) local change setting `number_of_clients` to `10`
directly on the canonical file — a direct edit of the protected canonical
config, which the project rules explicitly forbid. It was never committed, so
no result was ever produced from it. It was restored to its committed `K=5`
state with `git restore`, and the intended `K=10` change was moved into its
own new config instead (below).

### Part B configs created and reviewed

Two new configs were created from the restored canonical file, changing only
the field Question 9/10 identify plus `experiment_name`, `output_subdirectory`,
and `code_revision`:

- `configs/month2_week8_mnist_k10_fedavg_vs_centralized.json` —
  `number_of_clients: 10`, everything else unchanged.
- `configs/month2_week8_mnist_e2_fedavg_vs_centralized.json` —
  `local_epochs: 2`, `centralized_epochs: 10`, everything else unchanged.

Both were checked against `reports/verify_gate2_readiness.py`'s
`VARIANT_RULES` (field set, required values, and allowed-differences list)
before being recorded. `reports/gate2_variant_evidence.json` now names both
config paths with `status: "configs_ready"`; `metrics`/`report` remain `null`
because neither has been run yet.

A structural issue was also found and fixed in `reports/gate2_self_check.md`:
the student's Part B answers for Questions 9 and 10 were written directly
under each bullet prompt, but the field the verifier actually reads is the
`**Proposed changes and prediction:**` marker below that, which was still
empty — so the automated check would have reported both predictions as
missing despite them being fully written. The existing text (unchanged,
nothing added or reworded) was relocated under that marker for both
questions.

### Pre-run proposal review recorded

**Review date:** 2026-09-07. **Result:** APPROVED TO RUN. Both proposals
change only their intended field, keep the canonical config's other values
untouched, and use new identity fields that don't alias the canonical run.
The K=10 arithmetic (5,100 examples/client, 40,708,000 predicted
communication bytes, 2,000 total local steps) and the E=2 arithmetic (4,000
predicted FedAvg steps, 3,990 predicted centralized steps, unchanged
20,354,000 communication bytes) both check out against the runner's
step/communication formulas. One precision note, not blocking: the E=2
answer doesn't state the training-example-exposures number itself (510,000)
even though its own step-count numbers are consistent with that value.

### What is not yet done

Neither variant has been executed. This machine (macOS, system Python 3.9.6,
no conda) does not have the Windows/Python-3.10.20/`mse-ai` environment the
canonical run and `environment/month2_cpu_runtime.json` document — that gap
must be resolved (installing a matching Python 3.10.20 environment here, or
running on the original machine) before either tagged config can be executed.
Gate 2 remains open; no K=10/E=2 result exists, and no Non-IID, FedProx, or
Federated Unlearning work has been started.

---

## 2026-09-07 — Canonical Week 8 evidence re-baselined from Windows to this Mac

### What happened

A Python 3.10.20 environment was built on this Mac (system Python 3.9.6 had
no path to the exact pinned version; a user-level pyenv build was used
instead of Homebrew or an admin-rights installer, after first fixing a
missing `_lzma` module by compiling `xz` into a local, non-admin prefix).
With the exact pinned package versions from
`environment/month2_cpu_requirements.txt` installed, the canonical Week 8
config was rerun on this machine as a prerequisite check: `reports/
verify_gate2_readiness.py` will not evaluate the K=10/E=2 variants until the
canonical result independently verifies, and no local canonical result
existed on this machine at all (`results/` and `data/` are correctly
gitignored, so nothing had ever synced here).

Hand-written FedAvg reproduced exactly: 90.99% test accuracy, 90.86% macro
F1, identical to the numbers recorded in the Week 8 entries above. Centralized
SGD reproduced almost exactly but not bit-for-bit: 94.76%/94.69% here versus
94.75%/94.68% on the original Windows machine — a difference of exactly one
test example out of 10,000. This is consistent with ordinary cross-platform
floating-point non-determinism (macOS's math libraries reducing
matrix-multiply sums in a different order than Windows/MKL did), not a bug in
the implementation; `torch.use_deterministic_algorithms(True)` guarantees
determinism *within* one environment, not bit-identical results *across*
environments. This is the same category of gap as the Week 1 logistic-
regression non-determinism recorded above, at a smaller magnitude.

### The decision, and what it costs

`reports/verify_week8_fedavg.py` requires the committed canonical report to
match freshly computed metrics exactly, so this one-example difference is
enough to fail it outright with the original Windows-produced report left in
place. Two options were considered: document the gap and keep the Windows
result as canonical (meaning the K=10/E=2 variants would need to run on the
original Windows machine, since a variant must be compared against a
canonical baseline from the same environment), or accept this Mac as the new
reference environment for Gate 2's remaining work. The student chose to
re-baseline to this Mac, after being shown explicitly what that requires:
Gate 2's readiness verifier hardcodes the canonical tag name (`month2-week8`)
as a Python constant rather than reading it from the config, so re-baselining
means **force-moving that tag to a new commit** rather than creating a new
tag name. This is a real cost, stated plainly: every entry above that cites
`month2-week8` was written when that tag pointed at commit
`2fb17f49b27a3b0d57a3e3edafa8e2bf514af549` (the original Windows-producing
revision). That commit still exists and is unchanged; only what the tag name
currently resolves to has moved. A reader following an old citation of
`month2-week8` from before 2026-09-07 should resolve it against this entry,
not assume the tag still points where it did when that entry was written.

### What changed

- `environment/month2_cpu_runtime.json`: `operating_system` updated from
  `Windows 10 10.0.19045` to `macOS-26.5.1-arm64-arm-64bit`, `captured_on`
  updated to `2026-09-07`, and a new `previously_captured_on` field records
  the superseded Windows description inline. `python_version` and every
  pinned package version are unchanged — the same exact versions were
  installed here.
- `reports/month2_week8_centralized_vs_fedavg.md` was rebuilt from this
  machine's metrics (94.76%/94.69% centralized; FedAvg unchanged).
- `reports/month2_week8_k10_variant_report.md` and
  `reports/month2_week8_e2_variant_report.md` were generated from this same
  environment, so all three results (canonical, K=10, E=2) are now mutually
  consistent — produced by the same machine, same environment manifest, same
  session.
- The `month2-week8`, `month2-week8-k10`, and `month2-week8-e2` tags were all
  force-moved to point at the single new commit containing the above, so
  `reports/verify_gate2_readiness.py`'s requirement that a variant's
  implementation files not differ from the canonical tag's files now compares
  a commit to itself for all three.

### What did not change

No model code, no algorithm, no configuration value (`K`, `E`, `B`, `R`,
seeds, learning rate) changed. The measured FedAvg numbers are bit-identical
to the original Windows run. Only the centralized path's last-decimal
accuracy and the environment provenance changed, and both are documented
here rather than silently overwritten.

---

## 2026-09-07 — Gate 2 passed: K=10/E=2 variants run, interpreted, and reviewed

Both Part B variants from `reports/gate2_self_check.md` were executed against
their approved, tagged configs (`month2-week8-k10`, `month2-week8-e2`),
verified, and interpreted. `reports/gate2_variant_evidence.json` now records
`status: "complete"` with both variants' metrics and report paths.

### K=10 result (Question 9)

Predicted and actual agreed exactly: 5,100 examples per client, 2,000 total
local optimizer steps, 40,708,000 communication bytes. Centralized SGD is
unchanged at 94.76%/94.69% F1 (it does not depend on `K`). Hand-written FedAvg
dropped to 89.55%/89.37% F1 (from 90.99%/90.86% at K=5), widening the gap to
centralized from −3.77pp to −5.21pp accuracy (−3.83pp to −5.32pp F1). More,
smaller IID shards produced a lower-utility FedAvg model in this one run —
reported as measured, not generalized to every `K`.

### E=2 result (Question 10)

Predicted and actual agreed exactly: 510,000 training-example exposures,
4,000 FedAvg local steps, 3,990 centralized steps (at `centralized_epochs=10`,
preserving the matched-exposure budget), and unchanged communication
(20,354,000 bytes, since `K` and `R` did not change). Centralized SGD rose to
96.17%/96.13% F1 and FedAvg rose to 92.31%/92.21% F1; the gap to centralized
(−3.86pp accuracy / −3.93pp F1) stayed essentially the level it was at `E=1`
(−3.77pp / −3.83pp) rather than closing — doubling local epochs improved both
methods' absolute utility here without changing FedAvg's relative standing.

### Review decision

**Gate 2 status:** CLOSED

Reviewed against the passing standard in `reports/gate2_self_check.md`: the
student distinguishes client-side optimizer updates from server aggregation,
correctly explains the Week 8 result and convergence curves, reports the
negative centralized-vs-FedAvg gap honestly at both K=5 and both variants,
explains matched exposure versus differing optimizer trajectories, and
produced and interpreted two separate reproducible `K`/`E` variants with
correct step-count and communication arithmetic in every case. No claim was
made about Non-IID behavior, privacy, unlearning, or real network performance.
`README.md` now checks Gate 2 and moves the current unit to Month 3, Week 9.

### What remains blocked

Non-IID partitioning, FedProx, and every Federated Unlearning method remain
blocked by Gates 3 and 4. Nothing beyond FedAvg was implemented in this
session.
