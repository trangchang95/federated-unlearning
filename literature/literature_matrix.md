# Literature Matrix

The matrix uses the fixed columns required by the thesis plan. “Possible Gap”
records a question to investigate later; it does **not** lock the thesis
contribution before the Week 19 gap-analysis gate.

| Paper | Problem | Method | Assumption | Dataset | Metric | Result | Limitation | Possible Gap |
|---|---|---|---|---|---|---|---|---|
| Bourtoule et al. (2021), *Machine Unlearning* (SISA) | Remove a training point's influence without full retraining cost. | Shard data, train constituent models in isolation, slice incremental training with checkpoints, and aggregate predictions. Retrain only the affected constituent from the last unaffected checkpoint. | SISA is designed before training; shards are disjoint; isolated predictions can be aggregated; checkpoints can be stored; optional prior knowledge of request distribution. | Purchase, MNIST, SVHN, ImageNet, Mini-ImageNet, CIFAR-100 transfer learning. | Accuracy/top-1/top-5, retraining time, speed-up, shards, slices, deletion-request count. | With 20 shards/50 slices: 4.63× speed-up for 8 Purchase requests and 2.45× for 18 SVHN requests, with under 2 percentage points accuracy loss on simple tasks. Complex-task accuracy losses can be much larger without transfer learning. | Sharding weakens constituent models, especially on complex tasks; checkpoint storage grows with slices; speed gains hold only for a bounded request regime; requires SISA-aware initial training. | Centralized point-level SISA does not directly address client-level FU, FL aggregation, or Non-IID clients. Its accuracy loss from smaller heterogeneous shards is a question for later comparison, not yet a chosen contribution. |
| McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg) | Train one shared neural network from decentralized, private/large client datasets under expensive communication. | Select clients each synchronous round; start them from shared global weights; perform local minibatch SGD for one or more epochs; sample-weight and average returned local models. | Fixed clients/local datasets in controlled experiments; random client sampling; synchronous completed rounds; trusted coordinating server; common round initialization; raw data stays local but updates are not assumed perfectly private. | MNIST IID/Non-IID, Shakespeare IID/natural Non-IID, CIFAR-10 IID, and 10 million public posts grouped by over 500,000 authors. | Test accuracy, communication rounds to target accuracy, and round speed-up versus FedSGD/SGD. | Often 10–100× fewer rounds; CIFAR-10 reached 85% in 2,000 FedAvg rounds versus 99,000 SGD update-rounds (49.5×); word LSTM reached 10.5% in 35 rounds versus 820 for FedSGD (about 23×). | Primarily empirical; no general non-convex convergence/privacy guarantee; controlled synchronous simulation omits failures/corruption/changing data; very large local epoch counts can plateau or diverge. | Non-IID changes local update directions and later client contributions, but its effect on client-level unlearning is a question for later gates—not yet a chosen thesis contribution. |
| LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* | Learn document-recognition features and decisions jointly instead of relying on separately hand-designed stages. | End-to-end gradient-based learning; convolution, shared weights, and subsampling in LeNet; graph transformer networks for structured document recognition. | Labeled normalized inputs; useful local patterns recur across positions; shared weights and limited translation tolerance suit image recognition. | MNIST and multiple document-recognition/check-reading tasks. | Test error/accuracy and full-system recognition performance. | On regular MNIST: basic linear classifier 12.0% error (88.0% accuracy), `784–300–10` MLP 4.7% error (95.3%), and LeNet-5 without artificial distortion 0.95% error (99.05%). | Historical objectives, activations, preprocessing, optimizers, and budgets differ from modern runs; the cited comparisons do not provide modern multi-seed uncertainty. | The results are historical centralized model-family anchors only; they do not establish an FL/FU research gap or a directly comparable Month 1 benchmark. |
| Xiao et al. (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms* | Provide a more challenging but equally accessible drop-in replacement for saturated original MNIST. | Create a 70,000-image, ten-class 28×28 fashion dataset with MNIST-compatible files and benchmark common classifiers side by side on Fashion-MNIST and MNIST. | Fashion categories form a useful single-label task; matching MNIST's format supports drop-in use; five shuffled-order repetitions provide a useful average. | Fashion-MNIST and original MNIST. | Mean test accuracy across five repetitions. | Original-MNIST Table 3 reports 91.7% for `C=1` one-vs-rest logistic regression with L1 or L2 penalty and 97.2% for a one-hidden-layer 100-ReLU MLP. | Main focus is Fashion-MNIST; every library/optimizer/stopping detail and result variance is not reported; project splits and training protocols differ. | The MNIST column provides a close sanity anchor for Month 1, not a selected thesis contribution or evidence of statistical equivalence. |
| Scherer et al. (2010), *Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition* | Determine which local pooling operation best supports invariant visual recognition. | Compare subsampling, max pooling, overlaps, and window functions while keeping the surrounding architecture mostly fixed. | Local aggregation provides translation tolerance; mostly fixed architectures allow meaningful pooling comparisons. | NORB, Caltech-101, and an auxiliary MNIST experiment. | Test error under alternative pooling designs. | Section 4.5 reports 0.99% MNIST test error (99.01% accuracy) for a shallow one-convolution, 112-map max-pooling model trained for 60 epochs. | The MNIST subsection does not restate split, preprocessing, seed, or variability; architecture and 60-epoch budget differ substantially from the project. | This is an architecture-family sanity anchor only; pooling under FL/Non-IID conditions remains a later empirical question, not a chosen contribution. |

## Sources

### SISA

- <https://arxiv.org/abs/1912.03817>
- <https://doi.org/10.1109/SP40001.2021.00019>

### FedAvg

- <https://proceedings.mlr.press/v54/mcmahan17a.html>
- <https://arxiv.org/abs/1602.05629>

### Month 1 baseline sanity sources

- LeCun et al. (1998): <https://doi.org/10.1109/5.726791> and
  <https://bottou.org/papers/lecun-98h>
- Xiao et al. (2017): <https://arxiv.org/abs/1708.07747> and
  <https://doi.org/10.48550/arXiv.1708.07747>
- Scherer et al. (2010):
  <https://doi.org/10.1007/978-3-642-15825-4_10> and
  <https://amueller.github.io/papers/icann2010_maxpool.pdf>
