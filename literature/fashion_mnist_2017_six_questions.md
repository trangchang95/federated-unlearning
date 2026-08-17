# Xiao et al. (2017) — Fashion-MNIST

**Paper:** *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine
Learning Algorithms*
**arXiv:** <https://arxiv.org/abs/1708.07747>

This note follows the thesis plan's fixed six-question method. The relevance
to Month 1 is Table 3's side-by-side results on the original MNIST dataset.

## 1. Problem

How can researchers obtain a small, easy-to-load image-classification dataset
with MNIST's convenient format but a more challenging recognition task?

## 2. Gap

Original MNIST had become highly saturated and was often too easy to
distinguish meaningful improvements. Harder datasets existed, but they did not
offer the same drop-in file format, image dimensions, class count, and modest
computational cost.

## 3. Idea

Create Fashion-MNIST: 70,000 grayscale 28×28 fashion-product images in ten
classes, stored with the same 60,000/10,000 file structure as MNIST. Publish a
benchmark table that runs ordinary machine-learning classifiers on both
Fashion-MNIST and original MNIST for side-by-side context.

## 4. Assumption

The selected ten fashion categories form a useful single-label classification
task, and matching MNIST's dimensions and file layout makes algorithm transfer
straightforward. The benchmark also assumes that five repetitions with
shuffled training order give a useful average test-accuracy summary for the
included algorithms.

## 5. Evaluation

Section 3 says every algorithm is repeated five times with shuffled training
order and reports average test accuracy. In Table 3 continued (printed p. 5),
the original-MNIST column reports 0.917 accuracy for `LogisticRegression` with
`C=1`, one-vs-rest, and either L1 or L2 penalty. It reports 0.972 for a
one-hidden-layer `MLPClassifier` with 100 ReLU units. These are the two values
used in the Month 1 sanity check.

## 6. Limitation

The paper's main contribution is Fashion-MNIST, not a fully specified study of
original-MNIST optimization. Table 3 does not pin every library version,
optimizer, stopping detail, preprocessing choice, or seed, and it reports an
average without a standard deviation. The logistic experiment uses the
official 60,000/10,000 split and one-vs-rest formulation; the project's
logistic run repartitions all 70,000 examples. The MLP architecture and
training budget also differ. The values are close model-family anchors, not
direct replications.

## Sources

- Primary paper: <https://arxiv.org/abs/1708.07747>
- DataCite DOI: <https://doi.org/10.48550/arXiv.1708.07747>
- Authors' official repository:
  <https://github.com/zalandoresearch/fashion-mnist>
