# Month 1 Centralized Baselines — Literature Sanity Check

**Checked:** 2026-08-17
**Scope:** MNIST logistic regression, MNIST one-hidden-layer MLP, MNIST small
CNN, and CIFAR-10 small CNN
**Purpose:** decide whether the four saved Month 1 accuracies are believable
for their dataset and model family.

This is a **sanity check**, not a replication study. A sanity check asks, “Is
this result in a believable neighborhood?” A replication study would require
matching the architecture, data split, preprocessing, optimizer, training
budget, software version, and random seeds. None of the references matches all
of those conditions.

## 1. How to read the evidence

The report keeps three kinds of statement separate:

- **Project fact:** read from this repository's saved config, source code, or
  `metrics.json` output.
- **Literature fact:** explicitly reported by the cited paper or official
  framework example. When a source reports error, the corresponding accuracy
  is the arithmetic conversion `100% - error`; it is not a new experiment.
- **Inference:** this report's cautious interpretation of the two fact sets.

The allowed classifications are:

- `plausible / consistent with literature` — the value is believable given
  nearby, although not identical, published or official examples;
- `needs investigation` — the value conflicts with reasonably close evidence
  or has another clear warning sign;
- `insufficiently comparable` — the available evidence is too different to
  support even a useful sanity judgment.

## 2. Executive decision

| Project result | Closest useful literature context | Classification |
|---|---|---|
| MNIST Logistic Regression: **92.19%** | Xiao et al. report **91.7%** for both L1 and L2 logistic regression; LeCun et al. report **88.0%** for an older basic linear classifier. | `plausible / consistent with literature` |
| MNIST MLP: **96.80%** | LeCun et al. report **95.3%** for a 300-hidden-unit MLP; Xiao et al. report **97.2%** for a 100-ReLU-unit MLP. | `plausible / consistent with literature` |
| MNIST Small CNN: **98.60%** | Two modest primary-paper CNN references report **99.01%–99.05%**, but use 20 or 60 passes/epochs rather than the project's 5 epochs. | `plausible / consistent with literature` |
| CIFAR-10 Small CNN: **71.38%** | The closest official example reports **71.63%** for a similarly scaled CNN trained with Adam for 10 epochs; a shorter, smaller official example provides lower-budget context. | `plausible / consistent with literature` |

**Decision:** none of the four values currently needs investigation on
literature-plausibility grounds. This does not prove that the implementations
are bug-free or that any pair of results is statistically equivalent.

## 3. Project facts being checked

These facts come from the versioned configs and runners plus the locally saved
Month 1 metrics. The experiment code revision is `month1-gate1`; the recorded
environment is Python 3.10.20 on CPU with PyTorch 2.12.1, TorchVision 0.27.1,
and scikit-learn 1.7.1.

| Model | Project protocol | Measured fact |
|---|---|---|
| MNIST Logistic Regression | OpenML `mnist_784` v1; all 70,000 examples randomly repartitioned with stratification into 49,000 train / 10,500 validation / 10,500 test; pixels divided by 255; scikit-learn `LogisticRegression(max_iter=300, random_state=42)` with its pinned-version defaults. | **92.19%** test accuracy; **92.08%** macro F1. |
| MNIST MLP | Official MNIST files; 51,000 train / 9,000 validation / canonical 10,000 test; `784→128 ReLU→10`; 101,770 parameters; pixels in `[0,1]`; Adam, learning rate 0.001, batch 128, 5 epochs, fixed seed 42; validation-selected epoch-5 checkpoint. | **96.80%** test accuracy; **96.77%** macro F1. |
| MNIST Small CNN | Official MNIST files; 51,000 / 9,000 / canonical 10,000 split; `Conv32→pool→Conv64→pool→AdaptiveAvgPool(4×4)→Dense128`, dropout 0.25; 151,306 parameters; normalization mean 0.1307 / standard deviation 0.3081; Adam 0.001, batch 128, 5 epochs, fixed seed 42; no augmentation; validation-selected epoch-5 checkpoint. | **98.60%** test accuracy; **98.60%** macro F1. |
| CIFAR-10 Small CNN | Official 50,000/10,000 files with 45,000 train / 5,000 validation / canonical 10,000 test; same two-convolution family, 151,882 parameters; crop/flip training augmentation, channel normalization, dropout 0.25, weight decay 0.0001; Adam 0.001, batch 128, 10 epochs, fixed seed 42; validation-selected epoch-10 checkpoint. | **71.38%** test accuracy; **71.07%** macro F1. |

Local audit trail:

- configs: `configs/month1_week1_mnist_logistic_regression.json`,
  `configs/month1_week2_mnist_mlp.json`,
  `configs/month1_week3_mnist_cnn.json`, and
  `configs/month1_week3_cifar10_cnn.json`;
- runners: `experiments/month1_week1_mnist_baseline.py`,
  `experiments/month1_week2_mnist_mlp.py`, and
  `experiments/month1_week3_cnn.py`;
- architectures: `models/simple_mlp.py` and `models/small_cnn.py`;
- results: the corresponding `results/**/metrics.json` files. The `results/`
  directory is intentionally gitignored under the repository rules.

## 4. MNIST Logistic Regression evidence

| Primary source | Literature-reported fact and exact location | Differences that block direct comparison |
|---|---|---|
| LeCun, Bottou, Bengio & Haffner (1998), *Gradient-Based Learning Applied to Document Recognition* | MNIST has 60,000 training and 10,000 test examples (§III-A). A 7,850-parameter basic linear classifier has **12.0% test error = 88.0% accuracy** (§III-C.1 and Fig. 9). | Historical generic linear classifier, not confirmed to use the same multinomial logistic objective or L2 regularization; official split, preprocessing, and optimizer differ. |
| Xiao, Rasul & Vollgraf (2017), *Fashion-MNIST* | The paper also benchmarks original MNIST. Logistic regression with `C=1`, one-vs-rest, and either L1 or L2 penalty has **0.917 = 91.7% mean test accuracy**. Algorithms were repeated five times with shuffled training order (§3 and Table 3 continued, printed p. 5 / PDF page 5 of 6). | Official 60,000/10,000 split; the authors' companion benchmark runner standardizes inputs; one-vs-rest formulation, training/stopping details, and scikit-learn version differ. |

- **Project fact:** the project measured **92.19%** on its disjoint random
  49,000/10,500/10,500 repartition.
- **Literature fact:** the closer logistic reference reports **91.7%**, while
  the older generic linear reference reports **88.0%**.
- **Inference:** `plausible / consistent with literature`. The project is only
  0.49 percentage points above the closer value, but that difference is **not
  evidence of superiority**. Repartitioning all 70,000 examples changes which
  original MNIST test images can enter training, and the scaling, objective,
  solver, and evaluation size are not matched.

## 5. MNIST MLP evidence

| Primary source | Literature-reported fact and exact location | Differences that block direct comparison |
|---|---|---|
| LeCun et al. (1998) | A fully connected `28×28–300–10` network without distortion/deslanting has **4.7% test error = 95.3% accuracy** (§III-C.5 and Fig. 9). | 300 hidden units rather than 128 ReLU units; historical backpropagation/training procedure rather than five epochs of Adam; preprocessing differs. |
| Xiao et al. (2017) | `MLPClassifier` with one hidden layer of 100 ReLU units has **0.972 = 97.2% mean MNIST test accuracy**; the paper averages five shuffled-order repetitions (§3 and Table 3 continued, printed p. 5 / PDF page 5 of 6). | 100 rather than 128 hidden units; standardized inputs; official training allocation but no project-style validation selection; optimizer and stopping budget differ. |

- **Project fact:** the project measured **96.80%** for `784→128→10`, Adam
  0.001, batch 128, and 5 epochs on one fixed seed.
- **Literature fact:** comparable shallow MLPs report **95.3%** and **97.2%**.
- **Inference:** `plausible / consistent with literature`. The project result
  falls between the two reported values, but the interval is descriptive—not
  a confidence interval and not proof of statistical agreement.

## 6. MNIST Small-CNN evidence

| Primary source | Literature-reported fact and exact location | Differences that block direct comparison |
|---|---|---|
| LeCun et al. (1998), LeNet-5 | A roughly 60,000-parameter convolutional network has **0.95% error = 99.05% accuracy** without artificial distortion. Architecture: Fig. 2 and §II-B; MNIST protocol: §III-A; result and learning curve: §III-B and Fig. 5. The paper specifies 20 training passes, with convergence around 10–12. | 32×32 padded input, 5×5 convolutions, trainable average subsampling, tanh-like units/RBF outputs, specialized historical optimization, and 20 rather than 5 passes. |
| Scherer, Müller & Behnke (2010), *Evaluation of Pooling Operations…* | A shallow `28×28→Conv(112,9×9)→MaxPool(5×5)→10` network trained for 60 epochs has **0.99% error = 99.01% accuracy** (§4.5, printed p. 99 / PDF page 8 of 10). | One wide-filter convolutional layer rather than two 3×3 layers; 112 maps; 5×5 pooling; 60 rather than 5 epochs; §4.5 does not state the split, preprocessing, seed, MNIST-specific augmentation, learning-rate schedule, or variability. |

- **Project fact:** the project measured **98.60%** for a 151,306-parameter,
  two-convolution CNN trained for 5 epochs on one fixed seed.
- **Literature fact:** the selected primary-paper CNNs report
  **99.01%–99.05%** after 20 or 60 passes/epochs.
- **Inference:** `plausible / consistent with literature`. Being 0.41–0.45
  percentage points below these references is believable for the deliberately
  shorter five-epoch budget. The differences prevent attributing the gap to
  epoch count alone.

## 7. CIFAR-10 Small-CNN evidence

In this scoped search, no suitably close peer-reviewed non-SOTA paper was found
that also exposes a small architecture and modest training budget clearly
enough for this check. The evidence therefore uses authoritative official
framework examples, as allowed by the review protocol.

| Official source | Literature/official-source fact and exact location | Differences that block direct comparison |
|---|---|---|
| TensorFlow Core, *Convolutional Neural Network (CNN)* | CIFAR-10 50,000/10,000 files scaled by 255; `Conv32→pool→Conv64→pool→Conv64→Dense64→10` (122,570 parameters calculated from the listed layers); Adam and 10 epochs; displayed test accuracy **0.7163 = 71.63%** (“Download and prepare,” “Create the convolutional base,” “Add Dense layers,” “Compile and train,” and “Evaluate the model”). | Three convolutions vs. two, Dense64 vs. Dense128, default batch size vs. 128, no augmentation/weight decay shown, and no explicit fixed seed. The tutorial monitors the **test set as validation data every epoch**; although the displayed code does not select a best epoch, this protocol still differs from the project's separate 5,000-image validation split. |
| PyTorch, *Training a Classifier* | CIFAR-10; a LeNet-like `Conv6→Conv16→FC120→FC84→10` with 62,006 parameters calculated from the listed layers; batch 4, SGD 0.001 with momentum 0.9, 2 epochs, no augmentation; current official rendered output is **52%** on 10,000 test images (§§1–5; result in §5). | About 41% of the project's parameters, two rather than ten epochs, SGD rather than Adam, different normalization, and no explicit fixed seed. The displayed value is one unseeded rendered run, so it is lower-budget context rather than a stable benchmark statistic; “Last Verified: Not Verified” is an additional caution. |

- **Project fact:** the project measured **71.38%** for a 151,882-parameter,
  two-convolution CNN trained with Adam for 10 epochs on one fixed seed.
- **Literature fact:** the closest official example reports **71.63%** with a
  similar scale and 10-epoch Adam budget; a much smaller two-epoch example
  reports **52%**.
- **Inference:** `plausible / consistent with literature`. The 0.25-point
  proximity to TensorFlow is reassuring only at the level of plausibility. It
  must **not** be described as agreement to within 0.25 points, because the
  architectures, augmentation, batching, validation protocol, framework, and
  seeds are different.

## 8. Verified source ledger

All links below were opened and checked on 2026-08-17.

| ID | Verified source and persistent identifier | Type |
|---|---|---|
| R1 | LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998), *Gradient-Based Learning Applied to Document Recognition*. [DOI 10.1109/5.726791](https://doi.org/10.1109/5.726791); [author-hosted record/PDF](https://bottou.org/papers/lecun-98h). | Original peer-reviewed paper |
| R2 | Xiao, H., Rasul, K., & Vollgraf, R. (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747v2](https://arxiv.org/abs/1708.07747); [DataCite DOI 10.48550/arXiv.1708.07747](https://doi.org/10.48550/arXiv.1708.07747); [authors' official companion repository](https://github.com/zalandoresearch/fashion-mnist). | Original dataset/benchmark paper and companion artifact |
| R3 | Scherer, D., Müller, A., & Behnke, S. (2010), *Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition*. [DOI 10.1007/978-3-642-15825-4_10](https://doi.org/10.1007/978-3-642-15825-4_10); [author-hosted PDF](https://amueller.github.io/papers/icann2010_maxpool.pdf). | Original peer-reviewed conference paper |
| R4 | [TensorFlow Core “Convolutional Neural Network (CNN)”](https://www.tensorflow.org/tutorials/images/cnn), last updated 2024-08-16. No DOI/arXiv identifier. | Authoritative framework tutorial |
| R5 | [PyTorch “Training a Classifier”](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial), created 2017-03-24, rendered page updated 2026-05-08 and marked “Last Verified: Not Verified.” No DOI/arXiv identifier. | Authoritative framework tutorial |

## 9. Limitations and correct conclusion

1. The neural-network project results are currently **one fixed-seed run
   each**. A fixed seed makes a run repeatable; it does not measure variation
   across seeds.
2. Several official examples also show one unseeded rendered run rather than
   a mean and standard deviation. Their displayed outputs can change when a
   page is regenerated with a newer framework.
3. Architectures, parameter counts, data allocations, preprocessing,
   augmentation, epochs, batches, optimizers, regularization, checkpoint
   selection, software versions, and seeds differ. Accuracy values therefore
   cannot be ranked as if they came from one controlled experiment.
4. The logistic-regression project split is especially non-comparable to the
   official MNIST split because all 70,000 images were shuffled before the
   49,000/10,500/10,500 repartition.
5. No new training run, hyperparameter search, or test-set selection was
   performed for this review. The check only audits the already completed
   Month 1 results against sources.

The defensible conclusion is:

> All four saved Month 1 accuracies are plausible for their dataset and model
> family according to the selected primary and official references. This
> establishes literature sanity only—not direct comparability, replication,
> statistical equivalence, or superiority.
