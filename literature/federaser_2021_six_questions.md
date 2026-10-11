# Liu et al. (2021) — FedEraser (arXiv title: "Federated Unlearning")

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: the local PDF is arXiv v3 (6 May 2021), titled "Federated
Unlearning". The published version is titled "FedEraser: Enabling Efficient
Client-Level Data Removal from Federated Learning Models" (IWQoS 2021). Only
the arXiv v3 text was read (page-1 stamp confirms "arXiv:2012.13891v3 [cs.LG]
6 May 2021", 10 pages); the IEEE version was not read, so wording or numbers in the
published version may differ. Numbers, tables and section references were rechecked
against a pdfplumber text extraction of the local PDF; figures were not inspected.

## 1. Problem

After FL training, how can the server remove the influence of one client's data
from the global model much faster than retraining from scratch, given that every
round's starting model already contains the target client's influence
(the paper calls this "forward coupling", Sec. I)?

## 2. Gap

Centralized machine-unlearning methods (e.g., SISA) assume one-shot training and
do not handle FL's iterative rounds (Sec. I, Sec. V-A). Retraining from scratch
on the remaining clients is the reference but costs the full FL training time.

## 3. Idea

Trade server storage for reconstruction time (Sec. III-A). During normal FL
training the server keeps every client's update at intervals of Δt rounds
(rounds 1, 1+Δt, 1+2Δt, ..., T = floor(E/Δt) retained rounds). To unlearn client
k_u, it replays only the retained rounds. At each retained round, the remaining
clients run a short calibration training (E_cali local epochs, calibration
ratio r = E_cali/E_loc) starting from the current *calibrated* model. The server
keeps the stored update's norm but takes its direction from the fresh
calibration update (Eq. 1), aggregates the calibrated updates with the usual
sample-count weights over the K-1 remaining clients (Eq. 2), and adds them to
the model (Eq. 3). Round 1 needs no calibration because the initial model has
no target influence (Sec. III-B-1). Claimed cost reduction: r^-1 × Δt (Sec. III-C).

## 4. Assumption

- The server stores historical client updates (every Δt rounds, all clients) and
  the round index; storage is accepted as the price of speed.
- The remaining clients are online, still hold their training data, and can run
  E_cali local epochs for each retained round (calibration training is
  client-side, Algorithm 1). The target client does not participate.
- Horizontal FL, all K clients aggregated each round (Sec. II-B, Algorithm 1).
  No client sampling and no dynamic client set are modelled.
- One target client per unlearning request. Requests are not chained.
- Directional (normalized) calibration is assumed adequate: keep the old update
  magnitude, replace direction. No proof is given.

## 5. Evaluation

Setup (Sec. IV-A unless stated):

- Datasets: UCI Adult (48,842 records, binary), Purchase (197,324 records,
  labels created by unsupervised clustering into 2 classes), MNIST (70,000,
  described as 32x32), CIFAR-10 (50,000 train / 10,000 test).
- Models (Table I): Adult 2 FC layers; Purchase 3 FC layers; MNIST 2 Conv + 2 FC;
  CIFAR-10 2 Conv + 2 Pool + 2 FC.
- Clients: K = 20 (Sec. IV-A-5); varied 5 to 25 in Sec. IV-F (Purchase and
  CIFAR-10 only, Fig. 6).
- Partition: NOT stated anywhere in the text (a keyword search for "IID",
  "partition", "seed", "epoch", "learning rate" and "batch" found only the sentence
  that learning rate, training epochs and batch size use "the same settings" for all
  methods, Sec. IV-A-5). No Non-IID experiment. The introduction mentions
  "inherently heterogeneous data distribution across parties" only as a reason why
  training data may become outdated (Sec. I), not as an experimental setting.
- Fixed FedEraser hyperparameters: r = E_cali/E_loc = 0.5 and Δt = 2 (Sec.
  IV-A-5; chosen "empirically" as a trade-off, Sec. III-C).
- NOT stated: number of FL rounds E, local epochs E_loc, learning rate, batch
  size, optimizer, seeds, which client is the target, how many targets tried.
  The paper says only "the same settings" are used for all methods (Sec. IV-A-5).
- Compute: Intel Core i7 9400 CPU, GPU named "GeForce GTX 2070" in the paper,
  PyTorch 1.4.0, CUDA 10.1, Python 3.7.3.
- Baselines (Sec. IV-A-4): FedRetrain (retrain on the K-1 remaining clients),
  FedAccum (sum retained updates of remaining clients with no calibration),
  and FedAvg with no unlearning. Code link given is a Dropbox zip (footnote 1),
  not checked here.
- Stored updates and storage cost: retained = all clients' updates at rounds
  1, 1+Δt, ... (Sec. III-A). No byte count or storage-cost measurement appears
  anywhere in the paper.

How forgetting was measured (Sec. IV-A-3, IV-B):

- Accuracy on the target client's data, compared with FedRetrain (Fig. 1b).
  FedEraser mean target accuracy 0.831 over the four datasets, close to
  FedRetrain; Purchase 0.934 vs FedRetrain 0.952; CIFAR-10 0.556 vs 0.577;
  FedAccum on CIFAR-10 0.339.
- Loss on target data (Table III), e.g. MNIST FedEraser 1.03e-3 vs FedRetrain
  7.85e-4; Purchase 3.85e-3 vs 2.28e-3; CIFAR-10 2.03e-2 vs 3.51e-2.
- Prediction difference P_diss (Eq. 4) is defined but no numeric P_diss result
  was found in the text read.
- Membership-inference attack (shadow model = original FedAvg model), Fig. 2 and
  Sec. IV-B-3: mean attack precision about 0.50 and recall 0.726 on FedEraser
  models; Adult F1 0.714 (original), 0.563 (FedEraser), 0.571 (FedRetrain);
  CIFAR-10 F1 0.951 (original), 0.629 (FedEraser), 2.02% below FedRetrain.
- Last-layer parameter angle to FedRetrain (Adult only, Fig. 3): mean 3.99
  degrees for FedEraser vs 10.56 degrees for FedAccum.

Utility (Sec. IV-B-1, Fig. 1a): FedEraser test accuracy Adult 0.853, MNIST 0.986,
Purchase 0.943, CIFAR-10 0.562; average gap to FedRetrain 0.61% (text: "0.52%
lower" than FedRetrain on CIFAR-10). FedAccum CIFAR-10 0.408. The CIFAR-10 model is
weak (0.562, within 0.52% of the baseline), so forgetting comparisons there are made
on a poor model.

Speed-up and how it is computed:

- Analytic (Sec. III-C): time ratio = r^-1 × Δt, so r = 0.5, Δt = 2 gives 4×.
- Measured (Table II, wall-clock seconds, FedRetrain / FedEraser): Adult
  1.69e2 / 4.97e1 = 3.40×; MNIST 1.08e3 / 2.23e2 = 4.84×; Purchase 3.75e2 /
  9.21e1 = 4.07×; CIFAR-10 3.01e3 / 7.51e2 = 4.01×. The paper's text reports
  3.4×, 4.8×, 4.1×, 4.0×, consistent with these ratios. (Ratios computed here
  from the table; the paper prints only the seconds and the rounded text values.)
- The denominator is retraining time, so the speed-up excludes the original
  training and the cost of storing updates. Whether communication time is
  included is not stated. Single timing per cell, no variance.
- Sweeps: r from 0.1 (10× on Adult, 9.6× MNIST, 8.4× Purchase; Sec. IV-D) and
  Δt up to 10 ("12×" on Adult, MNIST, Purchase; Sec. IV-E).

Weak-evaluation flags:

- No seeds or variance reported anywhere; every number looks like a single run.
- Partition not stated; no Non-IID experiment at all, though the intro (Sec. I)
  cites heterogeneous data as a motivation.
- Retrain baseline is present (good), but forgetting is judged mainly by
  "target accuracy near retrain", which under an IID split is weak evidence:
  a retrained model also classifies the target client's data well.
- MIA is run against one shadow model that is the original model itself;
  no MIA threshold or dataset split described.
- Internal inconsistencies visible in the text (checked against the local PDF; the
  figures were not inspected): the Δt used in the r-sweep (Sec. IV-D) is not
  stated; "12× speed-up" at Δt = 10 on Adult (Sec. IV-E) is not what the quoted
  times give: 19.1 s at Δt = 10 vs Table II's FedRetrain 169 s is about 8.8×, and
  the paper does not say what the 12× is relative to; and CIFAR-10 test accuracy is
  0.562 at 20 clients (Sec. IV-B-1) but 32.4% at 5 clients and 7.2% lower at 25
  clients in the number-of-clients experiment (Sec. IV-F), a far lower model than
  in the main results. (An earlier draft also flagged a mismatch between the sweep
  ranges in the text and the Fig. 4/5 axes; that came from reading the figures and
  is not confirmed, so it is dropped.)
- Storage cost and client-side calibration cost per unlearning request are not
  reported.
- Sec. IV-E finds target accuracy rising with Δt, which the authors read as
  less forgetting, so the 4× speed-up sits on a forgetting/speed trade-off that
  is only shown on IID-like data.

## 6. Limitation

Needs stored updates for all clients (storage grows with model size × K × T,
not measured) and needs remaining clients online with their data to calibrate,
which fails if clients leave or delete data. Speed-up is bounded by r^-1 × Δt and
degrades forgetting when Δt grows. Evaluated only on small models and datasets,
one target client, no seeds, no Non-IID, no sequential requests, no formal
guarantee. Directly relevant to this thesis as the calibration-based baseline;
its behaviour under Dirichlet-type Non-IID is untested, so any Non-IID claim
here would be a new result.

## Scout check (2026-10-10)

Checked against arXiv v3 (arXiv MCP HTML text, full paper; sections cited by the paper's own
numbering). Every number and reference in the sections above matched the paper. Additions only:

- Gold standard: the paper calls FedRetrain "the baseline" and says it "provides an upper bound
  on the prediction performance of the unlearned model" (Sec. IV-A-4). It does not use the term
  gold standard and does not compare FedEraser against a second retrain.
- Calibration details for the retrain/noise-floor design: calibration interval Delta-t = 2 and
  calibration ratio r = E_cali / E_loc = 0.5, both in Sec. IV-A-5 (the paragraph headed
  "Experiment Environment"); the paper says they were chosen empirically as a trade-off
  (Sec. III-C). The actual E_cali and E_loc, the number of rounds E, learning rate and batch size
  are not stated ("the same settings" for all methods, Sec. IV-A-5). At each retained round, the
  non-target clients train E_cali epochs from the current calibrated model (Sec. III-B-1,
  Algorithm 1 CaliTrain).
- What the server stores: each client's update U_k at rounds t_1 = 1, t_(j+1) = t_j + Delta-t,
  with T = floor(E / Delta-t) retained rounds as printed, plus the round index (Sec. III-A).
  As printed, the rounds 1, 1 + Delta-t, ... and the count floor(E / Delta-t) differ by one when E
  is not a multiple of Delta-t; the paper does not address this. Size: not stated and not
  measured; model sizes are given only as layer types (Table I) and E is not given.
  Derived here (scout, not in the paper): stored entries = K x T update vectors.
- r = 1.0 is said to make FedEraser degenerate to FedRetrain (Sec. IV-D).
- The only dispersion figure in the paper is a standard deviation of 0.0011 for MNIST target
  accuracy across different calibration ratios (Sec. IV-D). That is spread over r, not over
  seeds, so it is not a noise floor.
- P_diss (Eq. 4) compares the original model with the unlearned model on the target data, not
  the unlearned model with FedRetrain.
- Additional internal inconsistencies (all from the arXiv v3 text; the 12x item is the one
  already listed in the older "Weak-evaluation flags" above, restated here with all baselines):
  - Adult, Sec. IV-D: r = 0.1 takes 10.1 s and is called 10x faster than FedRetrain, and r = 1.0
    takes 100.1 s. That implies a FedRetrain time of about 100 s, but Table II gives 169 s
    (1.69e2). Same pattern for MNIST (40.3 s, "9.6x" implies about 387 s versus Table II 1.08e3 s)
    and Purchase (19.6 s, "8.4x" implies about 165 s versus Table II 375 s). The sweeps in
    Sec. IV-D seem to use another training setting than Table II; the paper does not say.
  - Adult, "12x" at Delta-t = 10 (Sec. IV-E, Fig. 5a): the text gives 36.7 s at Delta-t = 1 and
    19.1 s at Delta-t = 10. No baseline quoted in the text reproduces 12x: 36.7 / 19.1 = 1.9x
    (relative to Delta-t = 1), 169 / 19.1 = 8.8x (relative to Table II FedRetrain), and
    100.1 / 19.1 = 5.2x (relative to the Sec. IV-D r = 1.0 time). The paper does not say what the
    12x is relative to. The r used in the Sec. IV-E sweep: not stated. Separately, 36.7 s at
    Delta-t = 1 is below Table II's 49.7 s at Delta-t = 2 (with r = 0.5 there), although a smaller
    Delta-t should take longer at the same r; whether r matches is not stated.
  - Purchase, Sec. IV-E: the text gives the range 80.1% to 92.1% for the testing data and
    81.3% to 93.7% for the target data over Delta-t = 1 to 10. The main result at Delta-t = 2 is
    0.943 on test data (Sec. IV-B-1), which is above the top of the testing range (92.1%); the
    main target-data result is 0.934 (Sec. IV-B-2), which lies inside the target range. So the
    inconsistency is for the test split only.
- Not an inconsistency, but a reading note: the "average difference of only 0.61%" (Sec. IV-B-1)
  is a gap on test data. Target-data gaps are larger: Purchase 0.952 - 0.934 = 1.8 percentage
  points (computed here from the Sec. IV-B-2 values; the paper also says 1.8%).
- Flag for the thesis: no Non-IID, no seeds, forgetting by target accuracy near FedRetrain, and
  storage not measured (Sec. IV-A; Sec. IV-B-2). The matrix row for this paper already records
  these; its Table II ratios were recomputed (169/49.7 = 3.40, 1080/223 = 4.84, 375/92.1 = 4.07,
  3010/751 = 4.01) and are correct.

## For the tutor

1. FedEraser keeps the norm of the stored update and takes only the direction from a fresh
   calibration update (Eq. 1, Sec. III-B-2). Why can the stored update's direction not be
   reused (Sec. I, Sec. III-A)?
   (Open discussion, scout inference, the paper does not address it: does the stored norm
   itself still carry some of the target client's influence? See the key.)
2. Sec. III-C gives a Speedup of r^-1 x Delta-t = 4x for r = 0.5, Delta-t = 2, and Table II
   measures 3.4x to 4.8x. What is the denominator of that Speedup, what is left out of it
   (original training, storage), and what does Sec. IV-E say happens to forgetting as
   Delta-t grows?
3. The paper judges forgetting by accuracy on the target client's data being close to
   FedRetrain (Fig. 1b, Sec. IV-B-2). Using CLAUDE.md section 9, why is that not enough, and
   what would you add?

Tutor key (paper references given where the paper says it; otherwise scout inference):
1. Paper: the retained updates were computed from global models that already contain the target
   client's influence ("forward coupling", Sec. I; Sec. III-A), so they must be calibrated
   against the current calibrated model using the non-target clients' data. Open discussion
   (scout inference): the norm is kept as is, so any influence it carries is not removed; the
   paper offers no argument or test either way, and the last-layer angle to FedRetrain
   (3.99 degrees, Fig. 3, Adult only) is the only related evidence.
2. Denominator: FedRetrain time (Table II), so the Speedup excludes original training and the
   cost of storing updates; communication time is not stated to be included. Sec. IV-E: target
   accuracy rises with Delta-t (Adult 83.8% to 84.4%; Purchase 81.3% to 93.7%), which the authors
   read as less forgetting, so the Speedup trades against forgetting.
3. Missing: M_retrain^(-A) comparison within the noise floor (two retrains with different
   seeds), and a Non-IID split. Target accuracy near FedRetrain alone is not enough
   (CLAUDE.md section 9).

## Sources

- Local PDF checked via text extraction (pdfplumber); figures not inspected: `articles/5. 2012.13891v3-Federated-Unlearning.pdf` (arXiv v3, 6 May 2021)
- arXiv abs page (checked): <https://arxiv.org/abs/2012.13891> — lists v1 27 Dec 2020, v2 21 Feb 2021, v3 6 May 2021; comment "To be published in ... 29th IEEE/ACM IWQoS, 2021"
- Crossref record (checked): title "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models", authors Gaoyang Liu, Xiaoqiang Ma, Yang Yang, Chen Wang, Jiangchuan Liu, 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS), issued 25 Jun 2021, pp. 1-10, DOI 10.1109/IWQOS52092.2021.9521274 (<https://api.crossref.org/works?query.title=FedEraser+Enabling+Efficient+Client-Level+Data+Removal+from+Federated+Learning+Models>)
- Not reached: the IEEE Xplore page and DBLP returned no usable content, so the published PDF and DBLP entry were not inspected. The arXiv and Crossref records above were checked in an earlier run and not re-fetched in this one.

Still unconfirmed: figure content (Figs. 1-6), including the sweep axis ranges; the published IWQoS text.

Source: articles/5. 2012.13891v3-Federated-Unlearning.pdf — local PDF checked via text extraction (pdfplumber); figures not inspected; venue and year verified via arXiv abs page and Crossref (earlier run) — VERIFIED (arXiv v3 text only; published IWQoS text not read)

Revision 2026-10-11: after the reviewer's draft-check, only the new Scout check and For the tutor parts were edited (inconsistency list, section labels, tutor key); the older answers were not touched.

Revision 2026-10-10: no statement in the existing answers was contradicted by the paper, so nothing was corrected. Added "Scout check" and "For the tutor"; re-confirmed arXiv v3 authors/title (arXiv MCP) and the IWQoS title, authors, 25 Jun 2021, pp. 1-10, DOI 10.1109/IWQOS52092.2021.9521274 (Crossref API, fetched directly). Re-read via the arXiv MCP HTML text, not the local PDF; figures still not inspected.
