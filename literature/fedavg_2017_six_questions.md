# McMahan et al. (2017) — FederatedAveraging

**Paper:** *Communication-Efficient Learning of Deep Networks from
Decentralized Data*  
**Venue:** AISTATS 2017, PMLR 54:1273–1282

This note follows the thesis plan's fixed six-question paper-reading method.
It records what the paper actually establishes and does not treat a possible
future research gap as an already chosen thesis contribution.

## 1. Problem

How can many devices jointly train one useful neural-network model when their
training data are distributed across the devices, are often private or too
large to upload, and communication with a central server is much more
expensive than local computation?

In ordinary centralized learning, the data are collected at one location. The
paper instead studies a central server that coordinates training while every
client keeps its raw examples locally.

## 2. Gap

Data-center distributed optimization methods were a poor match for the
federated setting considered by the paper. They commonly assumed relatively
reliable workers, IID and balanced data, many examples per worker, and cheap
communication. Mobile clients instead could have different data
distributions, different dataset sizes, intermittent availability, and slow
connections.

Naively performing only one distributed gradient step per communication round
(FedSGD) also required too many rounds. At the other extreme, training each
client independently to completion and averaging only once could produce a
poor global model.

## 3. Idea

**FederatedAveraging (FedAvg)** trades additional local computation for fewer
communications:

1. The server holds global weights \(w_t\) for round \(t\).
2. It randomly selects a fraction \(C\) of the clients.
3. Every selected client receives the same \(w_t\).
4. Each client trains that copy on its own data for \(E\) local epochs using
   minibatches of size \(B\).
5. Clients return their new local weights.
6. The server forms the next global weights by a sample-count-weighted average
   of the returned local weights.

For selected client set \(S_t\), the implementation form is:

\[
w_{t+1}
=
\sum_{k \in S_t}
\frac{n_k}{\sum_{j \in S_t} n_j}
w_{t+1}^{k},
\]

where \(n_k\) is client \(k\)'s number of local examples. This makes each
client's aggregation coefficient proportional to its local dataset size,
matching the example-weighted global objective. It does not claim that every
individual example has identical causal influence after nonlinear local
training.

The key parameters are client fraction \(C\), local epochs \(E\), local batch
size \(B\), and learning rate \(\eta\). FedSGD is a special endpoint with one
full-local-batch step; FedAvg allows multiple local optimizer steps before
communication.

## 4. Assumption

The main experiments use a controlled, synchronous system with a fixed set of
clients and fixed local datasets. In each round, a random client subset is
selected, selected clients start from the same global model, finish their local
work, and return an update for server aggregation. A central server is trusted
to coordinate the process.

The paper assumes that raw examples remain on clients, but it does **not**
assume this alone gives formal privacy. It notes that model updates may reveal
information and leaves stronger mechanisms such as differential privacy and
secure aggregation to future work.

## 5. Evaluation

The evaluation covers five neural-network architectures and four data
sources:

- MNIST with a two-hidden-layer network and a CNN, split across 100 clients in
  both IID form and a deliberately difficult Non-IID form where most clients
  contain examples from only two digit classes;
- Shakespeare next-character prediction with a two-layer LSTM and 1,146
  character-role clients, including a natural unbalanced Non-IID partition;
- CIFAR-10 image classification with a CNN and a balanced IID split across 100
  clients;
- next-word prediction with a word-level LSTM using 10 million public posts
  grouped into more than 500,000 author-clients.

The main measurements are test accuracy, communication rounds needed to reach
a target accuracy, and speed-up in rounds compared with FedSGD or ordinary
SGD. The headline result is a 10–100× reduction in communication rounds in
many tested settings. For CIFAR-10, FedAvg reached 85% test accuracy in 2,000
rounds versus 99,000 minibatch-update rounds for ordinary SGD to reach 85%
(49.5× fewer). On the large word-LSTM task, FedAvg reached 10.5% accuracy in
35 rounds versus 820 for FedSGD (about 23× fewer).

The results also show a trade-off: more local work often reduces the required
rounds, but very large \(E\) can cause training to plateau or diverge, and the
benefit is smaller in some strongly Non-IID settings.

## 6. Limitation

The study is mainly empirical and does not provide a general convergence
guarantee for non-convex FedAvg. It explicitly observes that averaging
arbitrary independently initialized neural networks can be bad; FedAvg relies
on clients starting each round from the same global model. Too many local
epochs can also cause plateauing or divergence.

Most experiments simulate a controlled synchronous process. Changing client
data, availability correlated with data distribution, failed clients,
malicious or corrupted updates, and production deployment are outside the
paper's scope. Its privacy argument is data minimization, not a proof that
shared model updates reveal nothing. For this thesis, the paper establishes
the FL training foundation but does not study how to remove a client's
already-aggregated influence.

## Sources

- Primary publication page:
  <https://proceedings.mlr.press/v54/mcmahan17a.html>
- Primary paper PDF:
  <https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf>
- Author preprint record: <https://arxiv.org/abs/1602.05629>
