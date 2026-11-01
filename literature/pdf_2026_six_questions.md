# Nguyen and Pham (2026) — Not All Retained Clients Are Equal: Client Selection for Federated Unlearning (PDF)

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: arXiv preprint 2610.31786v1 (cs.LG), submitted 22 Oct 2026;
the arXiv page comments say "Accepted at NewInML @ NeurIPS 2026" (a workshop;
no proceedings DOI seen). Metadata taken from the arXiv abstract page. Full
text read from the arXiv HTML rendering (all sections, Tables 1–4); figures
were not inspected, and Appendix A (CNN architecture, baseline details) was not
read. Short name "PDF" = "path-dependent", not the file format.

## 1. Problem

Remove a designated target client's influence from a trained FedAvg model
without retraining from scratch, when retained clients are only intermittently
available (Abstract; Sec. 1; Sec. 3.2, Eq. 5 defines success as being within
tolerances δ (retained loss) and ε_f (target loss) of a retrain-on-retained-
clients reference).

## 2. Gap / Method

**Gap** (Sec. 1–2): existing client-level FU treats unlearning as a stateless
post-hoc step and treats all retained clients as equally useful; retraining-
based methods use all retained clients and so cost a lot of computation,
communication and storage.

**Method** (Sec. 4): keep two lightweight, target-agnostic statistics per
client during training, computed on the classification layer only —
a CLP-weighted (critical-learning-period) average update direction z̄_i and an
average "server alignment" s̄_i (Eqs. 6–9; CLPs detected online from a
federated-gradient-norm jump above threshold τ_CLP, Sec. 4.1). On a request,
score each retained client by pd_i = β·s̄_i + (1−β)·(1 − cos(z̄_i, z̄_tar))
(Eqs. 10–11) and run retraining-style unlearning rounds with only the top-k
clients (Eq. 12). Storage O(N·d_cls) (Table 2).

## 3. Assumption

- The server can compute and store per-client classification-layer updates
  every round (Sec. 4.1) and detect CLPs with a hand-set threshold.
- Clients in the top-k are available for T_unlearn = 20 rounds (Sec. 5.1).
- Cosine similarity of historical update directions is a good proxy for
  "overlapping knowledge" with the target (Sec. 4.2) — argued, not proven.
- Unlearning scenario is one label-flipping client (Sec. 5.1).

## 4. Evaluation

- Datasets/models (Sec. 5.1): FMNIST (CNN), CIFAR-10 and CIFAR-100 (ResNet-18).
- Clients/partition (Sec. 5.1): N = 64, M = 16 sampled per round; IID and
  Dirichlet α ∈ {1, 0.1}.
- Hyperparameters (Sec. 5.1): lr 0.01, batch 32, τ_CLP = 0.01, λ = 0.7,
  β ∈ {0.1, 0.5, 0.9}, T_unlearn = 20, k = 25 in Table 1.
- Baselines (Sec. 5.1): Retrain, CONDA, FedRecovery, FOUL, FedEraser,
  Exact-Fun, FUSED, GDFA. (FedEraser and GDFA are in this thesis's matrix;
  PGA/Halimi is **not** a baseline here.)
- Metrics (Sec. 5.1): FA (accuracy on target-client data), RA, MIA-AUC gap to
  Retrain, Comm (GB), Comp (PFLOPs), space complexity (Table 2).
- Forgetting is measured as FA on the target client's data plus an MIA-AUC gap
  to a single Retrain run — **no retrain-vs-retrain noise floor** is shown.
- Number of training rounds T_learn, number of seeds, and compute hardware are
  **not stated** in the sections read.

## 5. Main results (with table/section references)

- Table 1, FMNIST α=0.1: Retrain RA 88.20; PDF RA 82.60 / 85.97 / 86.89 for
  β = 0.1 / 0.5 / 0.9; FA 0.00 / 0.72 / 0.44 vs Retrain 0.44.
- Table 1, CIFAR-10 α=0.1: Retrain RA 55.28, FA 6.93; PDF RA 45.22 / 54.48 /
  57.10 (the β=0.9 value exceeds Retrain's), FA 0.42 / 0.00 / 0.39. CIFAR-10 at
  α=0.1 has Retrain RA of only 55.28, so the baseline itself is weak.
- Table 1, CIFAR-100: PDF RA up to 49.13 (α=0.1) and 54.38 (α=1.0) vs best
  baseline 42.22 / 47.50 and Retrain 55.17 / 58.92.
- Table 1, Comm/Comp, CIFAR-10: Retrain 845.48 GB / 47.45 PFLOPs vs PDF
  259.46 GB / 18.7–20.0 PFLOPs (Sec. 5.2: ≈69% less comm, 57–61% less comp).
- Table 3 (FMNIST, α=0.1): RA rises with k (β=0.5: 79.26 at k=10, 85.97 at
  k=25, 86.40 at k=30) while comm/comp grow.
- Table 4 (ablation): reversed ranking or selecting target-coherent clients
  lowers RA (CIFAR-10 α=0.1: PDF 54.48 vs Rev 41.09 vs high-Coh 37.65).

## 6. Limitation

The paper has no Limitations section (Sec. 6 is a short conclusion). From this
reading (scout's view, not the authors'):
- One target client, one attack type (label flipping), one run per cell as far
  as the text shows — no ± values in Tables 1, 3, 4.
- FA of 0.00 is *lower* than Retrain's FA (e.g. 0.44, 6.93); "lower is better"
  rewards over-forgetting, which the thesis's noise-floor test would flag.
- No FedProx / heterogeneity-aware training baseline, and no PGA baseline.
- Statistics rely on a hand-tuned CLP threshold and the classification layer
  only; no analysis of when that proxy fails.
- Server-side cache O(N·d_cls) is small but must exist **before** the request;
  clients must still be reachable for 20 rounds.

**Relation to the thesis's baselines:** it is a *retraining-based* method that
keeps Retrain as the reference and compares to FedEraser and GDFA under
Dirichlet α, so it is a direct neighbour of the thesis's Non-IID-aware idea
(selecting which retained clients to trust under heterogeneity).

## For the tutor

1. In PDF's score pd_i = β·s̄_i + (1−β)(1−c_i), what does a *large* β favour
   and what does a small β favour, and which Table 1 rows show this?
2. Why is "FA = 0.00" in Table 1 not obviously better than Retrain's FA of
   0.44? What reference would the thesis use to decide?
3. Which three facts needed to reproduce Table 1 are missing from the paper's
   text (hint: rounds, seeds, ...)? How would that limit citing it as a
   baseline?

Source: https://arxiv.org/abs/2610.31786 (HTML render https://arxiv.org/html/2610.31786v1) — VERIFIED (title, authors, arXiv ID, year, workshop note from arXiv page; no DOI).
