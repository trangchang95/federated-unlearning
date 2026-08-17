# Month 2, Week 7 — Flower Compatibility Without a Black Box

**Checked:** 2026-08-18
**Framework:** Flower 1.30.0
**Purpose:** understand how a framework represents and aggregates the same
FedAvg updates that Week 6 already implemented by hand.

## 1. Why Flower is introduced only now

Week 6 established the mechanism first:

```text
same global w_t
    → fresh local copies and local SGD
    → (client state, client example count)
    → sample-count-weighted average
    → w_(t+1)
```

A framework is useful for standardized client/server interfaces, message
transport, lifecycle management, failure handling, and deployment. Those
features should surround an understood algorithm; they should not replace the
student's explanation of why the weights become (w_{t+1}).

The Week 7 adapter therefore does one narrow job. It converts the same
PyTorch state tensors into Flower's NumPy representation, calls Flower's
weighted aggregation helper, converts the result back, and compares it with
the hand-written result.

## 2. Version decision

The current Flower documentation lists newer releases, including 1.32.1, but
Flower 1.31 and newer require Python 3.11. The thesis's verified `mse-ai`
environment uses Python 3.10.20. Pip confirmed that 1.30.0 is the newest
available Flower release compatible with that interpreter.

The project therefore pins `flwr==1.30.0` instead of changing Python midway
through a reproduced milestone. This is a compatibility choice, not a claim
that 1.30.0 is the newest Flower release overall.

The exact Month 2 runtime is recorded in:

- `environment/month2_cpu_runtime.json`;
- `environment/month2_cpu_requirements.txt`.

Official framework references:

- Flower changelog: <https://flower.ai/docs/framework/ref-changelog.html>
- `FedAvg` strategy API:
  <https://flower.ai/docs/framework/main/en/ref-api/flwr.server.strategy.FedAvg.html>
- strategy explanation:
  <https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html>
- NumPy-array conversion API:
  <https://flower.ai/docs/framework/ref-api/flwr.common.ndarrays_to_parameters.html>

## 3. Concept mapping

| Hand-written project concept | Flower 1.30 concept | Meaning |
|---|---|---|
| `FedAvgServer` | server plus `FedAvg` strategy | Coordinates rounds and decides aggregation behavior |
| `train_client(...)` | client `fit(...)` path | Receives global parameters and returns locally trained parameters |
| `dict[str, Tensor]` | list of NumPy arrays serialized as `Parameters` | Model-state representation crossing the framework boundary |
| `number_of_examples` | `FitRes.num_examples` | Weight attached to a client's returned parameters |
| `weighted_average_model_states(...)` | strategy aggregation helper/`aggregate_fit(...)` | Produces the next sample-weighted global parameters |

Flower's built-in strategy also has explicit concepts for client proxies,
failures, fit configuration, evaluation configuration, and metrics. Week 7
does not start a real server because the goal is semantic inspection, not a
network/deployment experiment.

## 4. What the Flower source does

The inspected Flower 1.30 helper receives pairs of:

```text
(list of model-layer NumPy arrays, num_examples)
```

It:

1. sums all client example counts;
2. multiplies every returned layer by that client's count;
3. adds corresponding layers;
4. divides each layer sum by the total count.

That is the same formula implemented in Week 6. The comparison adapter is
`server/flower_compat.py`; it copies arrays before passing them to Flower so
the check cannot mutate the PyTorch inputs. The adapter is version-guarded and
supports NumPy-compatible `float16`, `float32`, and `float64` tensors; the
current MLP/CNN use `float32`.

## 5. Verified agreement

Run:

```powershell
conda run -n mse-ai python reports\verify_week7_flower.py
```

The verifier runs the 13 hand-written tests and seven Flower compatibility
tests. On 2026-08-18 all **20 tests passed**.

The Flower-specific evidence proves:

- the installed version is exactly 1.30.0;
- Flower also produces 2.6 for the 20/80 scalar example;
- Flower and the hand-written aggregator match on a two-tensor, unequal 4/2
  example without mutating the inputs;
- the public `FedAvg.aggregate_fit(...)` strategy path returns the same array
  as the transparent helper adapter with `inplace=False`;
- Flower's default in-place strategy path independently returns 2.6 for the
  20/80 example;
- dtype, shape, key order, output storage independence, and the explicit
  rejection of unsupported `bfloat16` conversion are checked; and
- framework client-selection rounding is tested rather than assumed equal.

This shows mathematical agreement for the tested aggregation inputs. It does
not prove that every framework configuration produces the same training run.

## 6. Important policy difference: fractional selection

The project's Week 6 rule is explicit:

```text
selected clients = ceil(C × K), at least one
```

For (C=0.3) and (K=5), it selects `ceil(1.5) = 2` clients.

Flower 1.30's legacy `FedAvg.num_fit_clients` uses an integer truncation plus
its configured minimum. With `min_fit_clients=1`, the same numbers select
`max(int(1.5), 1) = 1` client.

Neither rule changes the weighted-average formula after clients return, but
it changes **which and how many clients train**, so it can change convergence,
communication cost, and reproducibility. The Week 8 hand-written experiment
will keep the project's documented ceiling rule. Any later Flower experiment
must record Flower's minimum-client settings and must not be labeled identical
unless the selected count is matched deliberately.

## 7. What Week 7 does not claim

- No MNIST/CIFAR training was run.
- No accuracy, F1, convergence, speed, or bandwidth result was produced.
- No real client process or network connection was started.
- Flower was not used to rewrite or bypass `algorithms/fedavg.py`.
- Framework installation does not make FL private or distributed by itself.
- No Non-IID, FedProx, or Federated Unlearning code was introduced.

These are unit/compatibility checks, not experiments, so they do not require
an experiment config/result pair.

## 8. Roadmap decision

Week 7 is complete when the pinned environment, transparent adapter, source
mapping, and 20-test verifier pass. The next unit is Week 8: a modest,
config-driven IID MNIST experiment using the hand-written implementation,
with five clients to start and an honest centralized-versus-FedAvg report.

Gate 2 remains open until that experiment works and the student can change the
client count/local epochs and explain the observed convergence behavior.
