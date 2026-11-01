# Yin, Yao, Dressler, Wang, Tao (2026) — Efficient Exact Unlearning for Federated Zeroth-Order Optimization (FedZOUL)

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: arXiv preprint 2610.35211v1 (submitted 26 Oct 2026; 36 pages, 8
figures per the arXiv page). No venue beyond arXiv seen. Full text read from
the arXiv HTML rendering: Introduction, Sec. VI (Experiments) and VII; the
theory in Sec. III–V and the appendix proofs were **not** read, so theoretical
claims are the authors' statements from the abstract/Introduction.
**Scope flag:** the experiments remove *samples* held by target clients
(Sec. VI-B 2), not whole clients, and use zeroth-order (gradient-free)
training; marginal relevance to a client-level FedAvg thesis.

## 1. Problem

Exact unlearning (output distribution equals retraining on retained data) for
federated zeroth-order optimization (FedZOO), which communicates only random
seeds and scalars (Abstract; Intro).

## 2. Gap / Method

**Gap** (Intro): first/second-order FU methods bring back backprop, big model
states or tensor communication, losing the benefit of FedZOO.

**Method** (Abstract; Intro): use total-variation (TV) stability so that
distributions before and after deletion overlap and computation is reusable by
coupling. FedZOO-S gets stability from client + mini-batch subsampling;
FedZOUL-S keeps earlier updates and restarts from the first affected step.
FedZOO-NS adds Gaussian noise; FedZOUL-NS reuses affected updates through
rejection–reflection coupling.

## 3. Assumption

- Training is a specific zeroth-order, seed-pool algorithm (K = 4096
  accumulators), not FedAvg (Sec. VI-A 3).
- Exactness holds for the paper's own algorithm; it needs the saved state.
- Deletion requests arrive as a stream (Sec. VI-B 2).

## 4. Evaluation

- Datasets (Sec. VI-A 1): MNIST (MLP), FashionMNIST and SVHN (2-layer CNN),
  CIFAR-10 (ResNet-18), plus TOFU (Llama-3.2-1B/3B) for LLM unlearning.
- Partition: Dirichlet α ∈ {0.1, 0.5, 1.0}; 20% of clients are given 20% label-
  flipped samples as the forget set.
- Setup (Sec. VI-A 3): M = 100 clients, m = 10 per round, R = 300 rounds
  (vision), E = 10 local steps, batch 32.
- Baselines (Sec. VI-A 2): Re-optimization (retrain), FATS-SU, FedRR, FUDD;
  FedKSeed Pro and DeComFL for training; Oblivionis for TOFU. **Not** FedEraser
  or PGA.
- Metrics (Sec. VI-A 4): test accuracy, ROUGE-L, wall-clock time, bytes,
  peak GPU memory. **No MIA, no forget-accuracy vs noise floor** on vision
  tasks.
- Seeds / number of runs: not stated in the sections read.

## 5. Main results (with table/section references)

- Sec. VI-B 2: FedZOUL-S is up to 13.37 pp better than FedZOO-S after removing
  low-quality data and within 2.77 pp of re-optimization's final accuracy.
- Sec. VI-B 2: runtime at the final request is 2.2% (S) and 12.4% (NS) lower
  than re-optimization on CIFAR-10; 22.5% / 56.1% lower on TOFU 3B.
- Sec. VI-B 2: after 15 CIFAR-10 requests FedZOUL-NS uses 389.1 MB, 5.7% less
  than re-optimization; on TOFU 3B after five requests 12.38 MB / 1.89 MB vs
  27.08 GB for Oblivionis.
- Sec. VI-B 3: accuracy of FedZOUL-S peaks at 68.27% (m/M = 0.5), FedZOUL-NS
  72.73% at 0.9 on FashionMNIST.

## 6. Limitation

No Limitations section in what was read (Sec. VII is a conclusion). Scout:
- On CIFAR-10 the speed gain over full re-optimization is only 2.2%/12.4%;
  the large wins are against first-order baselines in *communication*.
- Final accuracy tracked only as test accuracy; forgetting is implicit in
  "exact" claims, not measured by a forgetting metric against a noise floor.
- Sample-level, not client-level; zeroth-order training is rare in practice.

**Relation to the thesis's baselines:** only "re-optimization" (= Retrain) is
shared. Useful as a reference for *exact* unlearning via replay, not as a
baseline.

## For the tutor

1. What does "exact unlearning" promise that approximate FU does not?
2. Why is the CIFAR-10 runtime saving over retraining so small (2.2%)?
3. Why is this paper's unlearning target (samples) different from the thesis's
   (clients)?

Source: https://arxiv.org/abs/2610.35211 (HTML render https://arxiv.org/html/2610.35211v1) — VERIFIED (title, authors, arXiv ID, year from arXiv page; no venue/DOI; theory and appendices not read).
