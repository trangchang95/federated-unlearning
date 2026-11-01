# Reddy, K, Raza, Prakash (2026) — Machine Unlearning for Federated Socially Fair k-Means Clustering

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: arXiv preprint 2610.13717v1 (submitted 9 Oct 2026; 53 pages
including appendices; the arXiv page lists code and scoped Lean proofs). No
venue beyond arXiv seen. Full text read from the arXiv HTML rendering:
Introduction (contributions), Sec. 6 (Experiments) and Sec. 7; Sections 2–5
(algorithms/theorems) and the appendices were **not** read. **Scope flag:**
federated k-means on tabular data, not neural networks.

## 1. Problem

Exact unlearning (data deletion) in federated socially fair k-means, where
deletions change group balance (Abstract; Sec. 1).

## 2. Gap / Method

Keep separate local summaries per group so unaffected institutions' summaries
stay valid; after a deletion rebuild only touched summaries, reweight and
resample; same-seed replay reproduces retraining exactly (Sec. 1 contributions,
Theorem 5). Fair refinement uses a guarded Fair-Lloyd update.

## 3. Assumption

Preprocessing/encoding fixed; both groups nonempty; binary canonical learner
(Appendix L has a multi-group prototype); exactness does not erase influence on
a fitted scaler (Sec. 7).

## 4. Evaluation

- Data (Sec. 6): Adult, Bank, Credit (tabular) and 2018 ACS Census records
  (Folktables); local single-machine runs.
- Clients: k = 10 for the whole-client deletion study; heterogeneity studied as
  a "scoped heterogeneity effect" (P1), not as Dirichlet α.
- Seeds: five seeds per configuration (Table 2).
- Retrain reference: yes ("fresh training"); all 810 replay comparisons match
  exactly (Sec. 6).
- Cost: algorithm and end-to-end wall time; no communication bytes (Sec. 7:
  network, stragglers unmeasured).

## 5. Main results (with table/section references)

- Table 2: one preselected client per dataset, five seeds: replay is 35.79×
  faster than retraining for the T = 0 model, 2.357× at T = 1 and 1.696× at
  T = 2.
- Table 1: record deletion gains are modest: direct replay total algorithm-time
  ratio 1.061 (P2) and 1.056 (P3).
- Table 3: fair updates lower worst-group cost by 3.15–5.07% but raise
  population-average cost by 0.53–4.72%.

## 6. Limitation

Authors (Sec. 7): single desktop, warm caches, five seeds; network/secure
aggregation/stragglers unmeasured; exactness conditional on fixed preprocessing.
Scout: not a deep model, so not comparable to FedAvg-based methods.

**Relation to the thesis's baselines:** only the idea of "replay with the same
seed equals Retrain" is relevant; it matches the thesis's reproducibility rules.

## For the tutor

1. Why does same-seed replay reproduce retraining exactly here but not for a
   neural network trained with FedAvg?
2. Why is whole-client deletion so much faster (35.79×) than record deletion
   (≈1.06×)?
3. Why is this k-means paper out of the thesis's main comparison?

Source: https://arxiv.org/abs/2610.13717 (HTML render https://arxiv.org/html/2610.13717v1) — VERIFIED (title, authors, arXiv ID, year from arXiv page; no venue/DOI; algorithm/theory sections and appendices not read).
