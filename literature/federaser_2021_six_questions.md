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

## Sources

- Local PDF checked via text extraction (pdfplumber); figures not inspected: `articles/5. 2012.13891v3-Federated-Unlearning.pdf` (arXiv v3, 6 May 2021)
- arXiv abs page (checked): <https://arxiv.org/abs/2012.13891> — lists v1 27 Dec 2020, v2 21 Feb 2021, v3 6 May 2021; comment "To be published in ... 29th IEEE/ACM IWQoS, 2021"
- Crossref record (checked): title "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models", authors Gaoyang Liu, Xiaoqiang Ma, Yang Yang, Chen Wang, Jiangchuan Liu, 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS), issued 25 Jun 2021, pp. 1-10, DOI 10.1109/IWQOS52092.2021.9521274 (<https://api.crossref.org/works?query.title=FedEraser+Enabling+Efficient+Client-Level+Data+Removal+from+Federated+Learning+Models>)
- Not reached: the IEEE Xplore page and DBLP returned no usable content, so the published PDF and DBLP entry were not inspected. The arXiv and Crossref records above were checked in an earlier run and not re-fetched in this one.

Still unconfirmed: figure content (Figs. 1-6), including the sweep axis ranges; the published IWQoS text.

Source: articles/5. 2012.13891v3-Federated-Unlearning.pdf — local PDF checked via text extraction (pdfplumber); figures not inspected; venue and year verified via arXiv abs page and Crossref (earlier run) — VERIFIED (arXiv v3 text only; published IWQoS text not read)
