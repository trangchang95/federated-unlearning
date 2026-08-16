# Month 2, Week 5 — Federated Learning Concepts

This guide assumes only the centralized training knowledge from Gate 1. It
explains what changes when training is distributed across clients and what
does **not** change. No FL framework or FedAvg implementation is introduced in
Week 5.

## 1. The one-sentence idea

Federated Learning (FL) trains a shared model by moving model parameters or
updates between a server and clients while leaving each client's raw training
examples on that client.

This is different from centralized learning:

```text
Centralized learning
--------------------
Client data ───────────────> one training machine
                              trains one model

Federated learning
------------------
                    global model
Server ─────────────────────────────> selected clients
       <─────────────────────────────
                    locally trained models/updates

Raw client examples stay at their clients.
```

Keeping raw examples local reduces the need to collect them centrally. It does
not automatically make the system private: an update can still contain
information about local data, so stronger privacy mechanisms are a separate
topic.

## 2. The seven Week 5 terms

| Term | Beginner meaning | Role in one round |
|---|---|---|
| Client | A participant that owns a local dataset; it can represent a phone, person, hospital, or organization | Receives the global model, trains a local copy, and returns weights or an update |
| Server | The coordinator; it does not train on the clients' raw examples | Selects clients, sends the global model, aggregates returned models |
| Local model | A temporary client-side copy of the global model | Its weights change through the same forward/loss/backward/optimizer loop learned in Gate 1 |
| Global model | The shared model stored by the server | It is the starting point for a round and becomes the weighted aggregate after the round |
| Communication round | One server-to-clients-to-server cycle | Send global weights → local training → receive results → aggregate |
| Local epoch | One complete pass through one client's local dataset | Controls how much a client trains before communicating again |
| Aggregation | Combining returned local models into the next global model | FedAvg uses a sample-count-weighted mean |

## 3. One communication round, slowly

Suppose the server starts round \(t\) with weights \(w_t\):

```text
1. Server selects clients A, B, and C.
2. Server sends the same w_t to A, B, and C.
3. A copies w_t and trains only on A's data.
   B copies w_t and trains only on B's data.
   C copies w_t and trains only on C's data.
4. Each client returns its locally updated weights.
5. Server computes a weighted average.
6. The result is w_(t+1), the global model for the next round.
```

Local training still uses the Gate 1 loop:

```text
forward pass
    ↓
cross-entropy loss
    ↓
loss.backward() computes gradients
    ↓
optimizer.step() changes that client's local weights
```

The new FL operation happens **after** those local optimizer steps:
aggregation changes the server's global weights.

## 4. Why the average is weighted

Consider three selected clients and pretend the model has only one scalar
weight:

| Client | Local examples \(n_k\) | Returned weight |
|---|---:|---:|
| A | 100 | 1.0 |
| B | 200 | 2.0 |
| C | 700 | 3.0 |

A simple client average would be:

\[
(1.0 + 2.0 + 3.0) / 3 = 2.0.
\]

FedAvg instead weights by the number of examples:

\[
(100/1000)(1.0) + (200/1000)(2.0) + (700/1000)(3.0) = 2.6.
\]

Why? A simple mean treats one 100-example client as equally influential as one
700-example client. Sample weighting instead makes each client's coefficient
proportional to its share of the 1,000 examples, matching the example-weighted
global objective. It does not guarantee identical causal influence for every
example after nonlinear local training. Later experiments must still inspect
per-client utility because a globally sensible weighting can hide poor
performance on a small client.

## 5. Four controls in FedAvg

| Symbol | Name | If increased | Main trade-off |
|---|---|---|---|
| \(C\) | Client fraction | More clients participate per round | More representative/smoother update, but more communication and waiting |
| \(E\) | Local epochs | Each client performs more local training before returning | Fewer rounds may be needed, but local models can drift farther apart |
| \(B\) | Local batch size | Each optimizer step uses more examples | Fewer steps and smoother gradients, but different memory/computation behavior |
| \(\eta\) | Learning rate | Each local optimizer step is larger | Can learn faster, but a value that is too large can make training unstable |

The 2017 paper found that extra local work often reduced communication rounds,
but very large \(E\) sometimes caused plateauing or divergence. Therefore,
“more local epochs” is not automatically “better.”

## 6. FedSGD versus FedAvg

- **FedSGD:** a client computes one full-local-dataset gradient step per round.
  It communicates frequently.
- **FedAvg:** a client can perform multiple minibatch optimizer steps over one
  or more local epochs before returning. It spends more computation locally to
  reduce communication rounds.

FedAvg does not invent a different backpropagation rule. It repeats the
ordinary local training loop and adds orchestration plus weighted aggregation.

## 7. What IID and Non-IID mean here

- **IID-like clients:** each client's local data roughly resembles the overall
  population. For MNIST, most clients might each contain a similar mixture of
  digits 0–9.
- **Non-IID clients:** local distributions differ. One client might contain
  mostly 0 and 1 while another contains mostly 8 and 9.

When Non-IID clients train locally, their models can move in different
directions. Averaging may still work, but convergence can be slower or less
stable. Week 5 only introduces this intuition. Month 3 will implement and
measure IID/Non-IID partitioning after Gate 2.

## 8. Common misconceptions

1. **“No raw data upload means perfect privacy.”** No. Model updates can leak
   information; FL is not itself a complete privacy guarantee.
2. **“A local epoch is a communication round.”** No. Local epochs occur inside
   a round, before a client sends its result back.
3. **“Every client must participate in every round.”** No. A fraction \(C\)
   can be selected.
4. **“FedAvg takes an ordinary unweighted average.”** Not when client dataset
   sizes differ; the standard rule weights by local sample count.
5. **“The global model belongs to one client.”** No. It is the server's shared
   aggregate; each selected client receives a copy.
6. **“A framework is needed to understand FL.”** No. Week 6 will implement the
   mechanism directly before any framework is considered in Week 7.

## 9. Why this foundation is required for Federated Unlearning

After many rounds, a target client's influence is mixed into a sequence of
global aggregates. Federated Unlearning will later ask how to remove that
client's contribution without fully retraining. We cannot define or evaluate
that removal correctly until ordinary FedAvg training, aggregation, and
client-level behavior are understood and reproduced.

This paragraph explains motivation only. FU implementation remains blocked
until Gates 2 and 3 are complete.

## 10. What to remember before Week 6

You should be able to say, without memorizing code:

- what the client and server each do;
- the difference between local and global models;
- the difference between a local epoch and a communication round;
- why all selected clients start from the same global weights;
- why FedAvg weights a client by its local example count;
- how \(C\), \(E\), \(B\), and learning rate affect the workflow;
- why local-data retention is not a formal privacy guarantee.

## Source

- McMahan et al. (2017), primary AISTATS/PMLR publication:
  <https://proceedings.mlr.press/v54/mcmahan17a.html>
- Fixed six-question reading:
  [`../literature/fedavg_2017_six_questions.md`](../literature/fedavg_2017_six_questions.md)
