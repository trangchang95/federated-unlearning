# Wu, Zhu, Mitra (2022) — Federated Unlearning with Knowledge Distillation

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: the local PDF is arXiv v1 (24 Jan 2022), the only version. The
arXiv page lists no journal reference, no comments and no DOI other than the
arXiv one. A Crossref title+author search found no published version. Treat as
an arXiv-only preprint. The page-1 stamp of the local PDF confirms
"arXiv:2201.09441v1 [cs.LG] 24 Jan 2022" (7 pages). Numbers, tables and section
references were rechecked against a pdfplumber text extraction of the local PDF;
figures were not inspected.

## 1. Problem

Remove one client's contribution from a trained FL global model without
retraining from scratch, without asking clients to participate again, and
without any client data (Sec. 1, Sec. 3.1).

## 2. Gap

The only earlier FL unlearning work cited (FedEraser, Liu et al.) needs clients
to retrain and needs extra communication rounds (Sec. 1, Sec. 2). Retraining
compares poorly on cost, and clients may have deleted their data (Sec. 3.2).
Comparing test accuracy against a retrained model is judged unconvincing
because FL is stochastic (Sec. 3.3).

## 3. Idea

Two server-side steps (Algorithm 1, Sec. 4):
1. Subtract the target client's accumulated historical updates from the final
   global model: M'_F = M_F - (1/N) * sum over rounds t=1..F-1 of the target's
   update (Eq. 3). The authors set the target's contribution to zero in every
   round ("lazy" strategy) instead of rescaling by N/(N-1), to limit accumulated
   skew (Sec. 4.1). This leaves an unquantified skew term epsilon_t.
2. Repair the damaged model by knowledge distillation on the server: the
   original global model M_F is the teacher, the subtracted model M'_F is the
   student, and training uses outsourced *unlabeled* data with soft labels at
   temperature T (Eq. 4); T is set back to 1 afterwards (Sec. 4.2). If labeled
   server data exists, a low-weight hard-label term can be added (Sec. 4.2).

Evaluation trick: the target client is made a backdoor attacker, so forgetting
can be read from backdoor attack success rate (Sec. 3.3).

## 4. Assumption

- Server stores every round's update of every client that may later be removed
  (Sec. 4).
- Server has extra unlabeled data that is not any client's data (Sec. 4, Sec. 5.1).
- Every round all N clients participate and are averaged with equal weight 1/N
  (Eq. 1; "for simplicity", Sec. 4.1). This differs from sample-weighted FedAvg
  and from client sampling.
- Backdoor features in the teacher are dormant on clean unlabeled inputs, so
  they are not transferred to the student (Sec. 1, Sec. 5.2). This is asserted,
  not tested beyond the trigger-input ASR.
- One target client (the attacker) at a time; no sequential requests.

## 5. Evaluation

Setup (Sec. 5.1, Fig. 1-3, Table 1):

- Datasets and clients: MNIST with 10 clients; CIFAR-10 with 10 clients; GTSRB
  with 5 clients (images cropped to 112x112). In each, one client is the
  attacker and the unlearning target.
- Models: MNIST 2 Conv + 2 MaxPool + 2 FC; CIFAR-10 VGG11; GTSRB AlexNet.
- Partition IID or Non-IID: NOT stated. No heterogeneity experiment.
- Backdoor: MNIST digit 1 predicted as 9; CIFAR-10 truck predicted as car;
  GTSRB stop sign predicted as speed limit 120 km/h (Sec. 5.1, Fig. 1). Poison
  fraction, trigger size and any model-replacement scaling: not stated.
- Distillation data: "some data (excluding the clients' data)" from outside the
  clients, labels ignored, size equal to one client's dataset (Sec. 5.1). Exact
  source pool is not named; it reads as drawn from the same dataset
  (in-distribution).
- Temperature T: "high", value NOT given. Distillation epochs k: NOT given for
  Table 1; the text says (MNIST, Fig. 3) test accuracy is "quickly recovered
  within five epochs of distillation training" (Sec. 5.2). Distillation loss form,
  optimizer, LR, batch size: NOT given (Algorithm 1 lists epochs k and temperature
  T as parameters without values).
- FL rounds, local epochs, LR, batch size, seeds: NOT given in the text. The number
  of rounds in Fig. 2 is only visible in the figure, which was not inspected.
- Compute and software: not reported.

Results (Table 1, test acc / backdoor "atk acc", percent):

| Dataset | Training | UL-Subtraction | UL-Distillation | Post-Training | Re-Training |
|---|---|---|---|---|---|
| MNIST | 98.0 / 99.7 | 52.2 / 0 | 97.7 / 0 | 98.5 / 0 | 98.2 / 0 |
| CIFAR-10 | 80.8 / 99.4 | 10.0 / 0 | 78.8 / 6.4 | 81.4 / 7.3 | 79.5 / 7.0 |
| GTSRB | 93.0 / 100 | 3.9 / 0 | 92.1 / 0 | 94.0 / 0 | 92.7 / 0 |

- Text claims (Sec. 5.2): UL-Distillation accuracy within 1% of retraining
  (differences from Table 1: MNIST 0.5, CIFAR-10 0.7, GTSRB 0.6 points).
  Subtraction alone never exceeds 60% accuracy on MNIST (Fig. 2).
- CIFAR-10 ASR of 7.0% on the retrained model is explained by ordinary
  misclassification into the target label (Sec. 5.2).
- "Post-Training" continues FL without the attacker, so it does need clients.

Speed-up: NONE reported. The paper has no timing table, no wall-clock number,
no communication count and no energy measurement. Efficiency is argued
qualitatively (server-side only, no client communication; Sec. 1, Sec. 6). The
only quantitative statement close to cost is "recovered within five epochs of
distillation" on MNIST (Sec. 5.2, Fig. 3). Any speed-up figure must therefore be
produced by a reproduction, not cited.

Weak-evaluation flags:

- Forgetting is measured only by backdoor ASR. ASR is a proxy for removing the
  attacker's *backdoor*, not the clean-data influence of an ordinary client.
- The subtraction step alone already gives ASR 0 with accuracy collapsed
  (MNIST 52.2, CIFAR-10 10.0, GTSRB 3.9), so ASR 0 is not informative about
  forgetting by itself. Reviewer inference: distillation from the original
  teacher M_F on clean data may still transfer knowledge learned from the
  target client's clean data; only ASR is checked, and no MIA or
  distance-to-retrain check is reported.
- Single run per cell, no seeds or variance, even though Sec. 3.2 says
  retraining itself gives different results each time.
- No efficiency measurement (see above), no storage-cost measurement for the
  stored per-round updates.
- No Non-IID, no partial participation, equal-weight averaging assumed.
- No comparison with FedEraser or any other FU baseline; only retrain.
- Key hyperparameters (T, k, LR, rounds) missing, so the table is not
  reproducible from the text alone. No code link found in the PDF.

## 6. Limitation

Server must store every round's per-client update and hold unlabeled
distillation data; the method needs the teacher to be the original model that
contains the target's influence. Equal-weight, full-participation averaging is
assumed. The skew term is never bounded, and recovery is shown only empirically
on three small vision setups with a single attacker. Efficiency claim is
unmeasured. For this thesis it is a server-side, client-free comparison point;
whether its distillation step preserves non-target knowledge under Non-IID is an
open question, not something the paper shows.

## Sources

- Local PDF checked via text extraction (pdfplumber); figures not inspected: `articles/6. 2201.09441v1Federated-Unlearning-with-Knowledge-Distillation.pdf` (arXiv v1)
- arXiv abs page (checked): <https://arxiv.org/abs/2201.09441> — v1 submitted 24 Jan 2022, cs.LG and cs.CR, no journal reference or comments listed
- Crossref title+author search (checked): no published version found for this paper
- Not reached: Semantic Scholar API returned HTTP 429, DBLP was not reachable. The arXiv and Crossref checks above were done in an earlier run and not re-fetched in this one.

Still unconfirmed: figure content (Figs. 1-3), including the number of FL rounds and the distillation-epoch axis.

Source: articles/6. 2201.09441v1Federated-Unlearning-with-Knowledge-Distillation.pdf — local PDF checked via text extraction (pdfplumber); figures not inspected; arXiv-only, no peer-reviewed venue found — VERIFIED as arXiv preprint (title, authors, year confirmed; no venue exists to confirm)
