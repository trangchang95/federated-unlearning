# Che et al. (2023) — Fast Federated Machine Unlearning with Nonlinear Functional Theory (FFMU)

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: the local PDF is the PMLR camera-ready (page 1 header:
"Proceedings of the 40th International Conference on Machine Learning, Honolulu,
Hawaii, USA. PMLR 202, 2023"), 28 pages (main text 1-9, appendix A.1-A.5 from
page 17 on). Page numbers below are PDF pages. Read from a pdfplumber text
extraction of this local file; figures were not inspected. Math in the extraction
is partly garbled, so formal statements are quoted only where the layout was
readable.

## 1. Problem

How can a federated system remove a specified part of the training data from a
trained FL model, faster than "train, then retrain or unlearn", while keeping a
certified unlearning guarantee, when the server cannot see local data (Sec. 1,
p. 1-2)?

## 2. Gap

Existing centralized and federated unlearning methods "sequentially perform two
expensive operations": train on the whole dataset, then retrain or unlearn
(Sec. 1, p. 1). PCMU (the authors' earlier centralized method) does both at once,
but the paper says that averaging local PCMU models with standard aggregation such
as FedAvg makes it "difficult to guarantee the certified radius and budget of data
removals of the global model" (Sec. 1, p. 2).

## 3. Idea

Three stages (Sec. 3, pp. 3-5):
1. Each edge device k trains a local model and uses PCMU: gradient smoothed with
   Gaussian noise (sigma) and quantized to {-1, 0, 1} by a quantizer Q.
2. The local model is written as the output of a Nemytskii operator,
   O(q_k)(x) = E over noise of f(x; Q(G_k + noise)) (Eq. 13).
3. The server takes the unweighted mean of local gradients, G = (G_1 + ... + G_K) / K
   (Eq. 14), and outputs O(q) (Eq. 15).

Theory: O is Frechet differentiable (Thm 2); it has a global Lipschitz constant
C / sqrt(2 pi sigma) in the gradient distance (Thm 3); if all pairwise local
gradient distances are at most d, the global model output is within a bound of
each local model output in L^p (Thm 4); as sigma goes to 0 the smooth model
converges to the non-smooth one, which has no certificate (Thm 5).

Timing of unlearning: FFMU trains "a machine unlearning model in advance" using the
certified budget of removals; it "performs one-time operation of simultaneous
training and unlearning" and needs no re-unlearning for a new request "as long as
the ratio of actual data removals is below the certified budget" (Sec. 4, p. 8).
It is therefore a train-time design, not a repair of an already trained FedAvg
model.

## 4. Assumption

- The local certificate comes from PCMU (Thm 1, p. 3: certified radius R and
  certified removal budget B' for the quantized gradient). FFMU claims to keep it
  "to a certain degree" (wording used on pp. 2, 3 and 5), not exactly.
- Quantizer Q is C-Lipschitz; pairwise local gradient distances are bounded by d
  (Thms 3-4). How d behaves under heterogeneous clients is not discussed anywhere
  in the paper (inference: more heterogeneous clients give a larger d and a looser
  bound; not shown by the paper).
- Thm 4 statement, quoted from the extraction: distance at most (K-1) C d divided
  by sqrt(2 pi K sigma) (p. 5, Eq. 12; Eq. 45, p. 20). The proof's own step (Eq. 47,
  p. 21) gives ||G - G_k|| <= (K-1) d / K, which combined with Eq. 46 (constant
  C / sqrt(2 pi sigma)) would give a factor (K-1)/K on C d / sqrt(2 pi sigma). The
  extraction cannot show where the K sits relative to the radical, so whether the
  printed bound and the proof agree is UNCONFIRMED; check the page image before
  quoting the bound. The proof of Thm 3 (Eqs. 42-43, p. 20) is displayed with the
  sigma = 1 normalisation and reads C d / sqrt(2 pi).
- Global aggregation is an unweighted mean of gradients, while the FL objective
  written in Sec. 2.1 (Eq. 1, p. 2) is sample-weighted (N_k / N).
- Removal is a fraction of the data (5% to 20%, p. 6; appendix text on p. 21 says
  "between 2% and 20%", which does not match the tables), not a whole client.
  Which samples or clients hold the removed data is not stated.
- The guarantee needs the actual removal ratio to stay below the certified budget
  (p. 8).

## 5. Evaluation

Setup (Sec. 4, p. 6; appendix A.4, pp. 27-28):

- Datasets and models: Fashion-MNIST with a CNN (60,000 train / 10,000 test),
  CIFAR-10 with LeNet (50,000 / 10,000), SVHN with ResNet-18 (73,257 / 26,032)
  (Table 16, p. 28).
- Clients: "Number of Edge Devices 100" (Table 16, p. 28). It appears only in the
  appendix hyperparameter table.
- Partition: NOT STATED. A keyword search of the whole extracted text found no
  "non-IID", "IID", "heterogeneity" or "Dirichlet" outside the reference list. No
  Non-IID or IID-vs-Non-IID experiment exists.
- Hyperparameters: Table 16 (p. 28) lists sigma 0.1, quantization threshold sigma^2,
  removal ratio 20%, local epochs 2, global epochs 200, batch size 50, learning rate
  0.1. Appendix A.4 (p. 27) separately says the networks are trained with SGD for
  120 epochs, initial learning rate 0.05, batch size 500, decayed by 0.1 at 1/2 and
  3/4 of the epochs. These two descriptions differ (200 vs 120 epochs, 0.1 vs 0.05,
  50 vs 500); the paper does not say which applies where. Sec. 4.5 (p. 8) varies
  sigma "from 0.025 to 0.03", while A.4 gives a sweep over 0.025 to 1 and Table 16
  defaults to 0.1.
- Seeds: "we run each experiment for 3 trials for obtaining more stable results"
  (A.4, p. 27). No mean, standard deviation or error bars are shown in Tables 1-4
  or 5-15, so the reported cells are not identified as means.
- Compute: 8 NVIDIA GeForce GTX 2080 Ti GPUs, 2 Xeon E5-2650 v4 CPUs, Python 3.7.3;
  about 5 days for all experiments (A.4, p. 27).
- Baselines (p. 6): Retrain, Knowledge Distillation (Wu et al. 2022a), Rapid
  Retraining, MacForget, FedEraser, VeriFi, Class-Discriminative Pruning, UN, RCAD,
  IJ, Noisy-GD. Only FedEraser and RCAD were run from official code; the other eight
  were implemented "in terms of the algorithm description from the original papers"
  (A.4, p. 27). The authors' code is promised "after the paper is accepted" (p. 27).

Forgetting measures (Sec. 4, p. 7): Accuracy, Error_f (error on forgotten data),
Error_r (error on remembered data), Error_t (error on test data); the goal is to
match the Retrain row. Table column order is Accuracy | Error_t | Error_r | Error_f
(Table 1, p. 6). No membership inference, no distance to the retrained weights, no
backdoor test.

Results (Table 1, p. 6, Fashion-MNIST, 10% removal; accuracy %, then Error_t /
Error_r / Error_f, then seconds Training + Unlearning = Total):

- Retrain: 84.40; 15.60 / 14.70 / 15.75; 211 + 1,636 = 1,847.
- FedEraser: 81.64; 18.36 / 17.70 / 16.72; 369 + 3,610 = 3,979.
- FFMU: 84.55; 15.45 / 14.88 / 15.07; 1,288 + 0 = 1,288 (the row reads correctly in
  the local PDF: unlearning time 0).
- Table 2 (p. 6, 20% removal): Retrain 84.39, total 2,538 (198 + 2,340); FFMU 84.55,
  total 1,288; FedEraser 79.51, total 3,979.
- CIFAR-10 (Tables 3-4, p. 7): 10% Retrain 53.66, total 1,831; FFMU 54.84, total 986.
  20% Retrain 54.97, total 3,318; FFMU 54.84, total 986.
- SVHN appears only in the appendix (Tables 11-15, appendix A.2; Table 14 is on p. 25, the other pages were not noted): 5% Retrain 90.62,
  total 2,085; FFMU 90.71, total 1,826; FedEraser 78.10, total 12,244. 20% Retrain
  90.12, total 6,597; FFMU 90.71, total 1,826.
- Ratios computed here from those tables (Retrain total / FFMU total), not printed
  in the paper: Fashion-MNIST 5% 0.86x (1,113 / 1,288), 10% 1.43x, 20% 1.97x;
  CIFAR-10 5% 1.03x (1,015 / 986), 10% 1.86x, 20% 3.37x; SVHN 5% 1.14x, 20% 3.61x.
- Paper's claim (p. 7): FFMU gives "at least 22.09% and 9.43% improvement of
  absolute accuracy difference" on Fashion-MNIST and CIFAR-10, relative to the other
  baselines' differences from Retrain; "at least 20.35% and 14.98% decrease of
  absolute error difference" on the two datasets (p. 8). Appendix (p. 21): FFMU
  within 3% accuracy, 3% Error_t, 3% Error_r and 2% Error_f of Retrain.
- Ablation (Fig. 1, p. 8, 20% removal, three datasets) and sigma / removal-ratio
  sweeps (Fig. 2, p. 8): read from the text only; figures not inspected.

Speed-up definition: total wall-clock seconds, Training + Unlearning, for the
whole pipeline (Tables 1-4 columns "Runtime (s)"). It includes the original
training. FFMU's total is the same at every removal ratio (1,288 s Fashion-MNIST,
986 s CIFAR-10, 1,826 s SVHN) because it is trained once for a budget of removals;
its total is larger than Retrain's at 5% on Fashion-MNIST (1,288 vs 1,113 s) and
smaller at higher ratios, since Retrain must be rerun for each ratio. So the
speed-up depends on the removal ratio and on how many requests arrive. FFMU's
training phase is about 6 times Retrain's original training on Fashion-MNIST
(1,288 vs 211 s, Table 1), and about 6 times on CIFAR-10 (986 vs 165 s, Table 3).
Whether communication time is included is not stated.

Storage of historical information: no statement about what the server or devices
must store. FFMU has no per-round update history like FedEraser, but per-device
state (local models, gradients) is not described.

Weak-evaluation flags:

- Partition, Non-IID, and client-level removal are absent; the 100-device setup is
  given only in an appendix table.
- Hyperparameter descriptions conflict (Table 16 vs A.4).
- Three trials are mentioned but no variance is reported.
- Accuracy in the FFMU rows is identical across removal ratios (84.55, 54.84,
  90.71), because one model is trained once; only Error_r and Error_f change,
  because they are evaluated on different removed subsets. FFMU test accuracy is
  slightly above Retrain's in several rows (for example 84.55 vs 84.40 and 54.84 vs
  53.66 at 10%), which is not what "closest to retrain" would predict; the paper
  does not comment.
- The FedEraser runtime row is identical in Table 1 and Table 2 (369 / 3,610 /
  3,979 s) although Retrain and other baselines change, and the appendix shows
  FedEraser growing with the ratio (5%: 2,127 s, 8%: 3,186 s, 15%: 5,713 s). The
  20% Fashion-MNIST FedEraser time therefore looks copied from the 10% table;
  do not use it as a measured value.
- Eight of eleven baselines were reimplemented from descriptions.
- No MIA or weight-distance check; the guarantee is a closeness bound that is
  conditional on d.

## 6. Limitation

The certificate is a closeness bound between the global and local models, valid only
if local gradients stay within distance d and the quantizer is Lipschitz; it is not
an (epsilon, delta) unlearning statement for the final global model. Unlearning
happens at training time against a fixed budget, so the method cannot repair an
already trained FedAvg or FedProx model. Removal is by data ratio, so client-level
removal is not shown. The FL partition is not stated, and no Non-IID setting is
tested. Efficiency depends on the removal ratio and is front-loaded into a longer
training phase. For this thesis: FFMU is the efficiency comparison that folds
unlearning into training; whether its d-dependent bound stays informative under
Dirichlet Non-IID is an open question and would be a new result.

## Sources

- Local PDF checked via text extraction (pdfplumber); figures not inspected:
  `articles/7. che23bFast-Federated-Machine-Unlearning-with-Nonlinear-Functional-Theory.pdf`
  (PMLR camera-ready, ICML 2023).
- PMLR page (fetched directly): <https://proceedings.mlr.press/v202/che23b.html> — Proceedings of the 40th ICML, PMLR vol. 202, pp. 4241-4268, 2023; authors Tianshi Che, Yang Zhou, Zijie Zhang, Lingjuan Lyu, Ji Liu, Da Yan, Dejing Dou, Jun Huan.
- Corroborating records seen in search results only (not read): ACM DL <https://dl.acm.org/doi/10.5555/3618408.3618577>, DBLP <https://dblp.org/rec/conf/icml/CheZZLL0DH23.html>, OpenReview <https://openreview.net/forum?id=6wQKmKiDHw>.
- No arXiv version found.

Still unconfirmed: figure content (Figs. 1-2 and appendix Figs. 3-4); the exact
placement of K under the radical in the Thm 4 bound; whether Table 2's FedEraser
time is a printing error (visible in the text, cause unknown); exact pages of
appendix Tables 5-13 and 15 (numbers themselves read from the text).

Source: articles/7. che23bFast-Federated-Machine-Unlearning-with-Nonlinear-Functional-Theory.pdf — local PDF checked via text extraction (pdfplumber); figures not inspected; metadata VERIFIED on the PMLR page — VERIFIED
