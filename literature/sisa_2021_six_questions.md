# Bourtoule et al. (2021) — Machine Unlearning (SISA)

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

## 1. Problem

How can a trained model remove the influence of a requested training point
without paying the full computational cost of retraining the complete model
from the beginning?

## 2. Gap

Full retraining without the deleted point provides a strong reference for
forgetting, but it is expensive for large datasets and neural networks. Earlier
methods discussed by the authors were limited in adaptive learning settings
and did not provide an efficient general solution for stateful training such
as stochastic gradient descent on deep networks.

## 3. Idea

SISA means **Sharded, Isolated, Sliced, and Aggregated** training. The training
data are split into disjoint shards, with one isolated constituent model per
shard. Each shard is further divided into ordered slices, and a model checkpoint
is stored before later slices are introduced. At prediction time, constituent
outputs are aggregated. To unlearn one point, only its constituent model is
restored to the checkpoint before the affected slice and retrained without the
point; unaffected shard models do not change.

## 4. Assumption

The service provider designs the training pipeline for SISA before deletion
requests arrive; each point belongs to exactly one shard; constituent models
are trained independently and can be aggregated; training is incremental so a
saved checkpoint can restart it; and the provider accepts storage overhead for
slice checkpoints. Knowledge of the future request distribution is optional,
but the paper assumes it when evaluating its optimized data placement variant.

## 5. Evaluation

The paper evaluates Purchase, MNIST, SVHN, ImageNet/Mini-ImageNet, and a
transfer-learning setting from ImageNet to CIFAR-100. It compares SISA with
full retraining and a baseline trained on only one shard-sized data fraction.
The main measurements are retraining/unlearning time or speed-up, classification
accuracy (including top-1/top-5 for complex tasks), the number of shards and
slices, and the number of deletion requests. With 20 shards and 50 slices, the
reported speed-ups over full retraining are 4.63× for eight Purchase requests
and 2.45× for eighteen SVHN requests, with less than two percentage points of
accuracy loss on those simpler tasks.

## 6. Limitation

More shards mean fewer examples per constituent model, which can substantially
reduce accuracy on complex tasks. Checkpoints add storage cost, slicing has
diminishing speed returns, and the speed advantage depends on the number of
requests remaining in a useful regime (the paper summarizes this as fewer than
three times the number of shards). SISA must be planned at initial training
time; it is not a drop-in repair for an arbitrary already-trained model. Most
importantly for this thesis, it studies centralized point-level unlearning, not
client-level Federated Unlearning or Non-IID client data.

## Sources

- Primary paper: <https://arxiv.org/abs/1912.03817>
- IEEE publication record: <https://doi.org/10.1109/SP40001.2021.00019>
- Authors' experiment repository: <https://github.com/cleverhans-lab/machine-unlearning>
