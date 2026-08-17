# LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition

**Venue:** *Proceedings of the IEEE*, 86(11):2278–2324
**DOI:** <https://doi.org/10.1109/5.726791>

This note records only the six questions required by the thesis plan. The
Month 1 sanity check uses the paper's MNIST linear, MLP, and LeNet-5 results as
historical model-family anchors, not as directly comparable experiments.

## 1. Problem

How can a document-recognition system learn useful features and a classifier
jointly from examples, instead of depending on a pipeline of separately
hand-designed modules whose errors may accumulate?

## 2. Gap

Traditional recognition systems relied heavily on hand-crafted feature
extractors and separately optimized stages. The authors argue that gradient-
based methods make it possible to optimize a complete multi-stage recognition
system toward one overall objective, including learning local image features.

## 3. Idea

Use gradient-based learning end to end. For handwritten digits, the paper
develops convolutional networks with local receptive fields, shared weights,
and subsampling so that features can be learned directly from pixels. LeNet-5
is then connected to a graph-transformer-network approach for full document
recognition.

## 4. Assumption

The relevant digit experiments assume labeled, size-normalized MNIST images
and a fixed training/test task. Convolution further assumes that useful local
patterns can appear at different image positions, making weight sharing and
some translation tolerance appropriate. The historical optimization and
preprocessing choices are part of the reported system.

## 5. Evaluation

For regular MNIST, §III-A states 60,000 training and 10,000 test examples. The
model comparison in §III-C and Fig. 9 reports 12.0% test error (88.0% accuracy)
for a basic 7,850-parameter linear classifier and 4.7% error (95.3% accuracy)
for a `28×28–300–10` one-hidden-layer network without distortion/deslanting.
The LeNet-5 experiment reports 0.95% test error (99.05% accuracy) without
artificial distortion; its architecture is in Fig. 2/§II-B and its MNIST
learning result is in §III-B/Fig. 5.

## 6. Limitation

These are historical baselines, not controlled matches for the project's
modern scikit-learn/PyTorch runs. Objectives, activations, preprocessing,
optimization, parameter counts, and training budgets differ. The cited result
locations do not provide modern multi-seed means and uncertainty intervals.
LeNet-5 also uses 32×32 padding, trainable average subsampling, and RBF outputs
rather than the project's max-pooling/ReLU classifier. The paper therefore
supports only a broad plausibility judgment here.

## Sources

- Author-hosted record and paper:
  <https://bottou.org/papers/lecun-98h>
- DOI record: <https://doi.org/10.1109/5.726791>
