# Chen, Lv, Hu, Huang, Bai (2026) — Split Federated Unlearning (SURE)

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: arXiv preprint 2610.36465v1 (submitted 27 Oct 2026; 13 pages per
the arXiv page). No venue beyond arXiv seen. Full text read from the arXiv HTML
rendering: Introduction, Sec. 5 (all subsections), Sec. 6–7, Appendix C.1–C.2;
the theory sections (Sec. 3–4), Appendix A/B proofs and Appendix C.3–C.4 were
**not** read in detail, so theory claims below are the authors' own statements
from the Introduction/abstract only.

## 1. Problem

Unlearning in *split* federated learning (SFL), where a model is cut into a
client-side part and a server-side part that are each aggregated, so removing a
client or a class is harder than in plain FL (Intro; Fig. 1).

## 2. Gap / Method

**Gap** (Intro): FedEraser/FedRecovery handle aggregation, "split unlearning"
handles partitioned execution, but nothing handles both; three stated
challenges: cross-partition coupling, historical influence propagation,
mode-dependent feasibility.

**Method** (Intro; Sec. 5.1): formulate SFU for client-wise and class-wise
removal under three update modes (Client-only / Server-only / Joint) with
"mode-matched Retrain" as reference; propose SURE = dual-importance estimation
(forgotten vs retained importance), selective parameter perturbation inside
permitted partitions, importance-guided recovery by the normal SFL protocol
(≤ 40 recovery rounds, App. C.1).

## 3. Assumption

- A retained-data recovery stage is available (Sec. 6 admits this).
- Reference = mode-matched retraining, which the authors say does not
  necessarily equal full retraining (Sec. 6).
- Target client is a single designated client (10 clients, one request,
  Sec. 5.1); multi-client results are in App. A.3 (not read).

## 4. Evaluation

- Datasets/models (Sec. 5.1): MNIST/LeNet, CIFAR-10/CNN, CIFAR-10/ResNet-18,
  CIFAR-100/ResNet-18; ResNet-18 cut after the first residual block (nine cut
  positions in Fig. 3).
- Clients/partition: ten clients; IID and *extreme* non-IID (disjoint label
  subsets: one class per client on MNIST/CIFAR-10, ten on CIFAR-100, App. C.1);
  class-wise runs use Dirichlet α = 0.1.
- Rounds (App. C.1): 50 (MNIST), 100/500 (CIFAR-10 CNN, IID/non-IID),
  200/1000 (ResNet-18, IID/non-IID); one local epoch, SGD momentum 0.9.
- Seeds (App. C.1, Sec. 5.1): main results mean ± SD over three runs, seeds
  41–43.
- Baselines (Sec. 5.1): Fine-tune, FedEraser, FedRecovery, SalUn, MoDe,
  FedOSD, NoT, FedCARE, FedMemPrune, all adapted to SFL. **No PGA/Halimi.**
- Metrics (Sec. 5.1): Retain, Forget, Test accuracy, MIA ROC-AUC, Avg. Gap =
  mean absolute difference to mode-matched Retrain over the four metrics;
  communication (GB) and computation (TFLOPs).
- Retrain baseline: yes (mode-matched). No retrain-vs-retrain noise floor, but
  SDs over 3 seeds are reported.
- Storage cost: not reported for SURE; Sec. 5.5 notes FedRecovery's cost
  excludes stored history.

## 5. Main results (with table/section references)

- Sec. 5.2 / Table 1 (client-wise, extreme non-IID, CIFAR-10/ResNet-18,
  Server-only): FedRecovery keeps 68.70% Forget; SalUn and MoDe drop Test to
  13.86% and 21.58%; SURE 0.40% Forget, 56.30% Test, Avg. Gap 1.76 pp.
- Sec. 5.2 (CIFAR-10/CNN, Joint): SURE Retain 44.22, Test 39.47, Forget 0.00,
  Avg. Gap 2.08 pp. CIFAR-100/ResNet-18 "reveals remaining gaps".
- Table 3 (ablation, CIFAR-10/ResNet-18, client-wise, Dir α=0.1, split 2):
  full SURE Avg. Gap 0.41 / 1.63 / 1.90 pp (Client-only / Server-only / Joint);
  Retrain Forget there is 68.14 / 65.37 / 64.13, i.e. the "forgotten" client
  still has ≈65% accuracy in the reference.
- Table 4: CIFAR-10/ResNet-18 Joint: SURE 218 GB, 1,370 TFLOPs vs Fine-tune
  213 GB, 1,370 TFLOPs and MoDe 449 GB, 2,890 TFLOPs.
- Sec. 5.3: split position changes Test/Retain/Forget in opposite directions
  for Client-only vs Server-only.

## 6. Limitation

Authors' own (Sec. 6): effectiveness varies by mode, split position and
scenario; recovery needs retained data; no formal forgetting guarantee; mode-
matched agreement ≠ full-retrain equivalence. Scout's additions:
- Only 10 clients and a single removal in the main tables; not the 50–100
  client regime used elsewhere.
- Extreme non-IID (one class per client) makes "client forgetting" equal to
  "class forgetting", which weakens what the Forget metric shows.
- Split learning is a different architecture from the thesis's plain FedAvg.

**Relation to the thesis's baselines:** shares FedEraser as baseline and
Retrain as the reference, and evaluates under non-IID, but in SFL only; it
does not test PGA/PGD and no plain-FL setting. Useful mainly as a reminder
that the reference-model definition matters.

## For the tutor

1. What is "mode-matched Retrain" and why can it differ from full retraining?
2. In Table 3, Retrain's Forget accuracy is ≈65%. What does that tell you
   about using "Forget accuracy → 0" as a goal?
3. Why would this paper's results not directly transfer to the thesis's FedAvg
   setting?

Source: https://arxiv.org/abs/2610.36465 (HTML render https://arxiv.org/html/2610.36465v1) — VERIFIED (title, authors, arXiv ID, year from arXiv page; no venue/DOI; theory sections and appendices A, B, C.3–C.4 not read).
