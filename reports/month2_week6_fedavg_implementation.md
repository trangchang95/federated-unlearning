# Month 2, Week 6 — Hand-Written FedAvg

**Completed:** 2026-08-18
**Purpose:** implement the mechanism from the FedAvg paper directly before
using any Federated Learning framework.

This unit proves that the project can perform one or more correct Federated
Learning rounds on small in-memory client datasets. It is an algorithm and
software-correctness milestone. It is **not yet** the Week 8 MNIST experiment,
an accuracy comparison, or evidence about convergence.

## 1. What changed after centralized learning?

Centralized Month 1 training used one model and one dataset:

```text
training batch
    → forward pass
    → loss
    → loss.backward()
    → optimizer.step()
    → the one model's weights change
```

FedAvg keeps that local training loop, but surrounds it with a server/client
round:

```text
Server has global weights w_t
          │
          ├── sends a copy of w_t to selected Client A
          ├── sends a copy of w_t to selected Client B
          └── sends a copy of w_t to selected Client C
                         │
             each client trains locally
             optimizer.step() changes only
             that client's copied weights
                         │
             clients return (weights, n_k)
                         │
          server computes the weighted average
                         │
                  global weights w_(t+1)
```

The important boundary is:

- `optimizer.step()` changes a **local** model during client training;
- FedAvg aggregation creates the **next global** model after all selected
  clients have returned.

## 2. Where each responsibility lives

| File | Responsibility | What it deliberately does not do |
|---|---|---|
| `algorithms/fedavg.py` | Validate and sample-weight client model tensors; clone states; estimate dense tensor payload bytes | It does not train a model or select clients |
| `clients/federated_client.py` | Deep-copy `w_t`, create fresh local SGD, train on one client's dataset, return state plus the exact example count | It never changes the server's global model |
| `server/fedavg_server.py` | Select clients, keep `w_t` unchanged while they train, aggregate their returns, and load `w_(t+1)` | It does not access a framework or pretend to be a real network |
| `tests/test_fedavg.py` | Check the mathematics and workflow on tiny deterministic tensors/datasets | It does not report research accuracy |

This separation is useful for debugging. If a future result is wrong, we can
ask whether the problem comes from local learning, aggregation, orchestration,
or evaluation instead of treating FedAvg as one opaque block.

## 3. The weighted-average calculation

For selected clients (S_t), client (k) returns locally trained weights
(w_{t+1}^{k}) and owns (n_k) training examples. The implementation uses:

\[
w_{t+1}
=
\sum_{k \in S_t}
\frac{n_k}{\sum_{j \in S_t} n_j}
w_{t+1}^{k}.
\]

The denominator contains the examples of the **selected** clients in that
round. For the Week 5 scalar example:

| Client | Examples | Returned scalar weight | FedAvg coefficient |
|---|---:|---:|---:|
| A | 20 | 1.0 | 20/100 = 0.2 |
| B | 80 | 3.0 | 80/100 = 0.8 |

Therefore:

\[
0.2(1.0) + 0.8(3.0) = 2.6.
\]

An unweighted mean would produce 2.0 and would incorrectly give the
20-example client the same coefficient as the 80-example client.

## 4. One implemented round, step by step

1. The server chooses
   `ceil(client_fraction × number_of_clients)` clients, with at least one.
2. Selection uses a private seeded random generator. The selected IDs are
   sorted so tensor additions occur in a stable order.
3. The server retains the current global model (w_t). It does not update it
   while individual clients are training.
4. Each selected client receives a deep copy of that same (w_t).
5. Each client creates a **fresh SGD optimizer**. Reusing an optimizer from
   another client could transfer momentum or other hidden optimizer state and
   would no longer represent basic FedAvg.
6. The client performs the configured local epochs with ordinary forward,
   cross-entropy, backward, and optimizer steps.
7. The client returns detached CPU tensor copies, `len(client_dataset)`, its
   mean local loss, and its number of optimizer steps.
8. The server validates that every returned state has the same tensor names,
   shapes, dtypes, and devices, then computes the sample-weighted state.
9. Only now does the server load (w_{t+1}) into the global model.

## 5. Reproducibility safeguards

Three random processes are intentionally separated:

- the future experiment runner must seed PyTorch **before constructing the
  initial global model**;
- the server derives client selection from the saved seed and round number;
- each local client seed is derived from the saved seed, round, and client ID,
  so changing loop order does not silently change a client's random stream.

Local training uses a dedicated DataLoader generator and a temporary PyTorch
RNG context. After that client finishes, the caller's CPU/Python RNG states
are unchanged. Two independent CPU servers with the same initial state,
datasets, and settings produce tensor-exact equal states through two rounds in
the synthetic check.

## 6. Defensive checks

The implementation stops with a clear error instead of silently producing a
misleading global model when it sees:

- no selected client or an empty model state;
- a zero/negative/non-integer example count;
- mismatched tensor keys, shapes, dtypes, or devices;
- `NaN` or infinite floating-point values;
- an empty client dataset or invalid local hyperparameter;
- a repeated/decreasing completed round number.

Floating parameters are averaged normally. Half-precision states accumulate
in float32 before converting back. For integer or Boolean buffers, a
fractional average has no faithful value, so Week 6 preserves the buffer only
when every client returned the same value; otherwise it raises. The current
`SimpleMLP` and `SmallCNN` contain no BatchNorm counters, so this policy is
compatible with the planned first experiments. It must be revisited before a
future architecture introduces differing non-floating buffers.

## 7. Communication-cost meaning

The helper counts a dense model state's raw tensor payload:

\[
S = \sum_{tensor} number\_of\_elements \times bytes\_per\_element.
\]

For (m) selected clients, one simulated round counts:

\[
download = mS, \qquad upload = mS, \qquad total = 2mS.
\]

The tiny test model has six float32 values, so (S=6×4=24) bytes. With two
clients, one round counts 48 download bytes plus 48 upload bytes = 96 bytes.
This is a reproducible **tensor-payload estimate**, not observed bandwidth. It
does not include serialization, headers, retries, compression, latency, or
parallel wall-clock behavior.

## 8. What the 13 synthetic tests prove

Run from the repository root:

```powershell
conda run -n mse-ai python -m unittest discover -s tests -p test_fedavg.py -v
```

Observed result on 2026-08-18:

```text
Ran 13 tests
OK
```

The suite proves:

- the 20/80 scalar example equals 2.6;
- multiple tensors use sample weighting without mutation or shared storage;
- malformed, non-finite, and incompatible states are rejected;
- half-precision accumulation and the explicit integer-buffer policy work;
- selection is deterministic, unique, and follows the ceiling rule;
- local training leaves the global model and caller RNG unchanged;
- identical clients do not inherit one another's local updates;
- one client with one full-batch SGD step matches centralized SGD from the
  same initialization;
- an unequal-size 4/2-client server round equals an explicit manual weighted
  aggregate;
- two independent two-client runs are tensor-exact equal through two rounds;
- round 2 begins from the updated round-1 model; and
- download/upload payload arithmetic is independently checked from 24 bytes.

`reports/verify_week6_fedavg.py` reruns this suite and also checks that the
Week 6 modules do not import an FL framework.

## 9. What this unit does not prove

- It does not produce MNIST accuracy, F1, or a convergence curve.
- It does not compare centralized learning with FedAvg; that is Week 8.
- It does not simulate IID/Non-IID partitions; that begins in Month 3 after
  Gate 2.
- It is sequential in one Python process. Clients are logical objects, not
  separate machines, processes, or security boundaries.
- Keeping datasets in separate objects does not prove privacy.
- Payload-byte arithmetic is not a real network benchmark.
- It does not use Flower or another framework.
- It does not implement FedProx or any Federated Unlearning method.

No experiment config or result file was created because these are unit tests,
not a research run. The first MNIST FL run in Week 8 must be config-driven and
must automatically save every mandatory thesis field, including nonzero
communication cost.

## 10. Roadmap decision

Week 6 is complete when the implementation, this explanation, and the
verification suite all pass. Gate 2 remains open because it additionally
requires a multi-client MNIST experiment, centralized-versus-FedAvg evidence,
and the student's ability to change clients/local epochs and explain
convergence.

The next roadmap unit is Week 7. An FL framework may be inspected only now
that hand-written FedAvg works; it must be treated as a comparison/adapter,
not as the source of the project's understanding.
