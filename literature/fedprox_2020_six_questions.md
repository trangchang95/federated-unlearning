# Li et al. (2020) — FedProx

**Paper:** *Federated Optimization in Heterogeneous Networks*
**Venue:** MLSys 2020

This note follows the thesis plan's fixed six-question paper-reading method.
It is read directly against this project's own Week 9-10 measurements: both
show FedAvg's accuracy falling as data heterogeneity rises, which is exactly
the gap this paper targets.

## 1. Problem

Real federated networks are heterogeneous in two ways FedAvg's original
analysis mostly sets aside: **statistical heterogeneity** (clients hold
Non-IID data — this project's Weeks 9-10) and **systems heterogeneity**
(clients differ in compute, memory, and connectivity, so forcing every
selected client to complete the exact same number of local epochs is
unrealistic — slower devices either get dropped or straggle). The paper asks
how to optimize one shared model well under both kinds of heterogeneity at
once, with some convergence guarantee.

## 2. Gap

FedAvg has two specific weaknesses this paper targets directly:

- **No tolerance for partial local work.** FedAvg implicitly assumes every
  selected client finishes the same fixed `E` local epochs. A framework that
  can only average complete, uniform local solutions must either wait for
  the slowest device or discard stragglers — both costly at scale.
- **No convergence guarantee under Non-IID data with variable local work.**
  Prior analyses generally assumed IID data or a fixed, uniform number of
  local steps; neither matches this project's Week 9-10 setting, where
  FedAvg's measured accuracy already degrades as heterogeneity increases,
  with no theoretical account of *why* or *how much*.

## 3. Idea

**FedProx** generalizes FedAvg with two changes that work together:

1. **A proximal term.** Each selected client minimizes
   \(h_k(w; w^t) = F_k(w) + \frac{\mu}{2}\lVert w - w^t\rVert^2\)
   instead of the plain local loss \(F_k(w)\), where \(w^t\) is the current
   global model. The extra term penalizes a local model for drifting far
   from the point every client started this round from — directly limiting
   the client-drift effect this project's Week 9-10 confusion matrices and
   accuracy gaps already show empirically. \(\mu=0\) recovers plain FedAvg
   exactly, so FedProx is a strict generalization, not a separate method.
2. **Tolerance for inexact/partial local solutions.** A client that only
   manages a \(\gamma\)-inexact local solution (less work than a fully
   converged or fixed-epoch update, e.g. due to limited compute or time) can
   still contribute safely, because the proximal term bounds how far even a
   partial solution can drift.

Aggregation itself is unchanged from FedAvg: the server still forms a
sample-count-weighted average of returned local models.

## 4. Assumption

The convergence analysis assumes **bounded dissimilarity** between clients'
local objectives and the true global objective (a formal bound on how
different local gradients can be from each other), not literal IID data. It
covers non-convex objectives, which is why it applies to neural networks
rather than only convex models. \(\mu\) is treated as a tunable
hyperparameter chosen per problem, not a fixed universal constant — the paper
is explicit that too large a \(\mu\) can slow convergence by over-restricting
local movement, while too small a \(\mu\) approaches plain FedAvg's
instability under heavy heterogeneity.

## 5. Evaluation

The paper evaluates on both synthetic Non-IID datasets (with a controllable
heterogeneity parameter, conceptually similar to this project's Dirichlet
\(\alpha\) knob) and real federated datasets (MNIST, FEMNIST, Shakespeare,
Sent140), under both statistical heterogeneity and systems heterogeneity
(clients performing variable amounts of local work rather than a fixed `E`).
Headline results: FedProx improves average test accuracy and training
stability over FedAvg specifically as heterogeneity increases, and unlike
FedAvg it retains a convergence guarantee even when clients contribute
different amounts of local work.

## 6. Limitation

\(\mu\) is an extra hyperparameter that must be tuned; a poorly chosen value
can make FedProx no better than FedAvg or slow it down. Reported accuracy
gains over FedAvg are sometimes modest in mild-heterogeneity regimes — the
paper itself shows the benefit grows with heterogeneity severity, which
matches this project's own Week 9-10 finding that FedAvg's accuracy drop is
nonlinear in Non-IID severity. The convergence theory relies on the bounded-
dissimilarity assumption, which is not verified for every real dataset. Like
McMahan et al. (2017), FedProx addresses *training* under heterogeneity; it
says nothing about removing a client's contribution afterward, which remains
this thesis's actual research question.

## Sources

- Publication page: <https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html>
- Author preprint: <https://arxiv.org/abs/1812.06127>
