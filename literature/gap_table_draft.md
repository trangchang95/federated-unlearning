# Gap Table (Week 19 draft)

**DRAFT — does not lock the thesis contribution before the Week 19 gate.**

Each cell is Y / N / Partial / `?`, followed by the section, table or figure it rests on. A cell with no citable support is `?`. Column meanings:

- **Non-IID**: the paper runs a Non-IID (or heterogeneity-varied) experiment on the method.
- **Sequential**: the method handles more than one unlearning request in a row.
- **Efficient**: a measured cost saving over retraining is reported.
- **Verification**: forgetting is checked by something stronger than target-data accuracy alone (formal statement, MIA, distance to retrain, watermark).
- **Limitation**: main limitation, in one line.

Evidence grade per row:

All rows: cells checked against the local PDFs via text extraction (pdfplumber); figures not inspected. Section numbers are the papers' own; page numbers are PDF pages.

- FedEraser (arXiv v3) and KD-FU (arXiv v1): rechecked against the local text extraction; section numbers are those of those arXiv versions.
- FFMU and SIFU: checked against the local PDFs (PMLR camera-ready).
- SISA (arXiv v3): checked against the local text extraction; two section references in the earlier draft were corrected (Definition III.1 sits in Sec. III-A, p. 4; sequential/batch analysis is in Secs. V-B and V-C, pp. 8-9).

| Method | Non-IID | Sequential | Efficient | Verification | Limitation |
|---|---|---|---|---|---|
| SISA (Bourtoule et al. 2021; centralized point-level reference) | N — centralized, no clients; shards are a uniform partition of one dataset (Sec. IV-B, p. 6); no data-skew or non-IID text found (only "heterogeneous" constituent models, Sec. IV-B). | Y — sequential and batch request settings analysed separately (Sec. V-B, p. 8; Sec. V-C, p. 9); speed gain holds only while requests stay below about 3x the number of shards (Sec. I, p. 2; Sec. VII-A, p. 11). | Y — 4.63x (Purchase, 8 requests) and 2.45x (SVHN, 18 requests) with 20 shards / 50 slices (abstract p. 1; Sec. I, p. 2). | Partial — exact unlearning by construction: Definition III.1 (Sec. III-A, p. 4) requires the model distributions with and without the point to be identical; verification by outsiders is left open (Sec. IX, p. 14); no empirical forgetting test. | Sharding lowers accuracy on complex tasks (Sec. VII-A, pp. 10-11); slice checkpoints cost storage (App. C); must be set up before training; point-level, not client-level FL. |
| FedEraser (Liu et al. 2021) | N — partition not stated and no Non-IID experiment (Sec. IV-A; local note, Evaluation). | N — one target client per request, requests not chained (Alg. 1, Sec. II-B; local note, Assumption). | Y — FedRetrain / FedEraser time 3.40x, 4.84x, 4.07x, 4.01x (Table II); analytic r^-1 x Delta-t = 4x (Sec. III-C). Excludes original training and update storage. | Partial — target accuracy and loss (Fig. 1b, Table III), MIA (Fig. 2, Sec. IV-B-3), last-layer angle to retrain on Adult only (Fig. 3, Sec. IV-C); no formal guarantee. | Stores all clients' updates every Delta-t rounds (unmeasured); needs remaining clients online; single runs; IID-like evaluation only. |
| KD-FU (Wu, Zhu, Mitra 2022; arXiv only) | N — partition not stated, no heterogeneity experiment (Sec. 5.1; local note, Evaluation). | N — one target client at a time (Sec. 4.1: "client N is the target client"; Sec. 5.1: one attacker per experiment). | ? — no timing, communication or energy number anywhere; only "within five epochs" of distillation on MNIST (Sec. 5.2, Fig. 3). | Partial — backdoor attack success rate vs retrain only (Table 1, Sec. 3.3); subtraction alone already gives ASR 0 with collapsed accuracy, so ASR is weak evidence. | Needs stored per-round updates plus extra unlabeled server data; equal-weight full participation; key hyperparameters missing; no FedEraser comparison. |
| FFMU (Che et al. 2023) | N — no IID / non-IID / heterogeneity / Dirichlet text anywhere in the paper; partition not stated (Sec. 4 p. 6; A.4 p. 27; Table 16 p. 28, which lists 100 edge devices); figures not inspected. | Partial — text claims a one-time train-and-unlearn covers "a series of" requests while the removal ratio stays below the certified budget (Sec. 4, p. 8); the sweep in Fig. 2b varies ratio, not a sequence, and removal is by data ratio, not by client (p. 6). | Partial — Retrain total / FFMU total, computed from Tables 5, 1, 2, 8, 3, 4, 11, 15: Fashion-MNIST 0.86x (5%), 1.43x (10%), 1.97x (20%); CIFAR-10 1.03x, 1.86x, 3.37x; SVHN 1.14x (5%), 3.61x (20%). Totals include original training; FFMU's training phase is about 6x Retrain's (Table 1, p. 6). | Partial — Error_f / Error_r / Error_t next to Retrain (Sec. 4, p. 7; Tables 1-4); closeness bound global-to-local, conditional on gradient distance d (Thm 4, p. 5); no MIA or weight distance. | Train-time design, so it cannot repair an already trained model (Sec. 4, p. 8); sample-level; certificate is a closeness bound, not an (epsilon, delta) statement for the final model; conflicting hyperparameters (Table 16 vs A.4, pp. 27-28). |
| SIFU (Fraboni et al. 2024) | Partial — only skewed partitions: one class per client (MNIST, FMNIST), Dirichlet 0.1 (CIFAR-10/100), one celebrity per client (CelebA) (Sec. 5.1, p. 7); Dirichlet 1 in the watermark scenario (Sec. 5.3, p. 8); no IID control, no controlled sweep. | Y — Algorithm 3 and Thm 3 cover a series of requests (Sec. 4, pp. 5-7); experiments use U = 3 requests of 10 clients each (Sec. 5.3, p. 8; App. E, p. 15). | Y (in aggregation rounds, not wall-clock) — 52% fewer iterations than Scratch on average and FedEraser 12% more than Scratch (App. E, p. 16, Fig. 4); SIFU best on 4 of 5 datasets in the time-constrained scenario (Sec. 5.2, p. 7). Per-dataset values are in figures, not inspected. | Partial — (epsilon, delta)-unlearning of the noised rollback model under smoothness (Def. 1, Thms 1-3, pp. 3-7), but experiments set B = 1 for CNNs and check the bound only empirically (Sec. 5.1, p. 7; App. F, p. 17); also a watermark check (Sec. 5.3, p. 8). | Proof as read covers full participation and gradient steps, experiments sample clients and use SGD (App. B, p. 14; Table 2, p. 15); epsilon = 10; one checkpoint per client stored, size unmeasured (p. 7); Non-IID effect not isolated. |

## Cells that most clearly point to a Non-IID gap

1. FedEraser Non-IID = N (Sec. IV-A; no Non-IID experiment, partition unstated).
2. KD-FU Non-IID = N (Sec. 5.1; no heterogeneity experiment).
3. FFMU Non-IID = N (whole paper: no IID / non-IID text; partition unstated, Sec. 4 p. 6, A.4 p. 27, Table 16 p. 28).
4. SISA Non-IID = N (centralized; Sec. IV-B, p. 6, uniform shards).
5. SIFU Non-IID = Partial (Sec. 5.1 p. 7, Sec. 5.3 p. 8): the only method tested on skewed partitions, but with no IID control and no controlled sweep, so the effect of heterogeneity level is unmeasured.

No method in the table varies the heterogeneity level. The survey (IEEE TNNLS 36(7), Sec. VII-F, p. 11713) says the influence of data heterogeneity on FU algorithms is "under-explored" and that the efficacy of unlearning methods within FedAvg variants designed for non-IID data is "an unexplored dimension". These cells state where evidence is missing; they do not choose the thesis contribution before the Week 19 gate.
