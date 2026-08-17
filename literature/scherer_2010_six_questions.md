# Scherer et al. (2010) — Evaluation of Pooling Operations

**Venue:** ICANN 2010, pp. 92–101
**DOI:** <https://doi.org/10.1007/978-3-642-15825-4_10>

This note follows the thesis plan's fixed six-question method. Its shallow
MNIST max-pooling CNN is used only as one architecture-family sanity anchor.

## 1. Problem

Which local pooling operation gives a convolutional vision model the most
useful invariance: trainable subsampling, max pooling, overlapping pooling, or
a smoother windowed aggregation?

## 2. Gap

Earlier recognition systems used different pooling operations inside
different architectures. Because many components changed at once, their
results could not isolate whether the pooling operation itself caused a
performance difference.

## 3. Idea

Hold the surrounding convolutional architecture as fixed as possible while
changing the pooling rule. Directly compare subsampling and max pooling,
overlapping and non-overlapping regions, and several window functions across
vision tasks.

## 4. Assumption

Local aggregation should provide tolerance to small image translations, and a
mostly fixed architecture makes pooling variants meaningfully comparable. The
experiments also assume that results on datasets such as NORB, Caltech-101,
and MNIST provide useful evidence about these pooling choices.

## 5. Evaluation

The main experiments compare pooling operations on NORB and Caltech-101.
Section 4.5 additionally reports a shallow MNIST architecture: 28×28 input,
one convolutional layer with 112 feature maps and 9×9 filters, non-overlapping
5×5 max pooling, and ten outputs. After 60 epochs of online backpropagation,
the paper reports 0.99% test error, equivalent to 99.01% accuracy (printed
p. 99; PDF page 8 of 10).

## 6. Limitation

The MNIST result is a short auxiliary subsection rather than the paper's main
controlled comparison. Section 4.5 does not restate the MNIST split,
preprocessing, seed, or variability. Its one wide-filter convolutional stage,
112 maps, 5×5 pooling, and 60 online-training epochs differ substantially from
the project's two 3×3 stages and five Adam epochs. It establishes that a
shallow max-pooling CNN can reach roughly 99%, not that the two runs are
numerically equivalent.

## Sources

- Publisher/DOI record:
  <https://doi.org/10.1007/978-3-642-15825-4_10>
- Author-hosted paper:
  <https://amueller.github.io/papers/icann2010_maxpool.pdf>
