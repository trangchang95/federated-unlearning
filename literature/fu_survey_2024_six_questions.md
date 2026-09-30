# Romandini et al. (2024/2025) — Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary. Because the
paper is a survey, it focuses on the taxonomy, the recommended evaluation
metrics, and the open problems. The survey's descriptions of other papers are
NOT used here as sources for those papers.

Version note: the local PDF is the published IEEE Transactions on Neural Networks
and Learning Systems version, vol. 36, no. 7, July 2025, pp. 11697-11717 (page 1
header), DOI 10.1109/TNNLS.2024.3478334. Page 1 footer: received 18 February 2024,
revised 6 July 2024, accepted 6 October 2024, published 25 October 2024, current
version 9 July 2025; licensed CC BY 4.0. arXiv 2401.05146 (v1 10 Jan 2024, v3 5 Nov
2024) is a different file and was not compared. Page numbers below are the journal
page numbers printed on the PDF. Read from a pdfplumber text extraction; figures
and tables (Tables I-V) were not inspected, and their cell contents are NOT in the
extracted text.

## 1. Problem

Give researchers one map of federated unlearning (FU): background, why unlearning
is needed even though data stay local, requirements and metrics for an FU
algorithm, a literature review, and open problems, so that removing a client's,
class's or sample's contribution from an FL model can be done without full
retraining (Sec. I, pp. 11697-11699).

## 2. Gap

Earlier surveys mostly cover machine unlearning (MU), or cover FU only in general
or from a security angle; the authors want a survey with practical design and
implementation guidance and their own experiments (Sec. I-A, I-B, p. 11698;
Table I, p. 11699, compares surveys but its cells are not in the extraction). The
paper also reports that most FU works appeared recently: "more than 60% of the
articles proposing new FU methods, reviewed in this article, were published after
2022" (p. 11698).

## 3. Idea

Structure (section headings in the PDF): I Introduction; II Background (MU, FL,
catastrophic forgetting and DP); III Federated Unlearning (objectives, why unlearning
is needed if data are not shared, challenges of adapting MU to FL, entity
performing unlearning); IV Design and Implementation Guidelines for Efficient FU
Algorithms (requirements, metrics); V Literature Review of FU Methods; VI Distilled
Lessons Learned; VII Open Problems and Future Directions; VIII Conclusion;
Appendix (experimental setup).

Taxonomy (Sec. III-A, V):

- By objective (Sec. III-A, p. 11701): sample unlearning (a subset S_u of one
  client's data), class unlearning (all samples of a class across clients), client
  unlearning ("peculiar to federated settings", the client's entire local dataset).
  Client unlearning is a special case of sample unlearning when S_u = D_u.
- By technique within each objective (Sec. V, pp. 11707-11711):
  - Client unlearning (V-A): 1) recalibration of historical updates (FedEraser, FRU,
    FedRecover, VeriFi, FAST are described in this subsection), 2) knowledge distillation (Wu et al.),
    3) gradient modification, 4) clustering, 5) Bayesian FL, 6) differential
    privacy, 7) other approaches (for example TV-stability, federated random forest,
    federated K-means).
  - Class unlearning (V-B): channel pruning with TF-IDF, and momentum degradation
    (MoDe).
  - Sample unlearning (V-C): 1) gradient modification, 2) quantization, 3)
    reinforcement learning, 4) other approaches.
- Table IV (p. 11708) compares reviewed methods by objective, technique, where the
  work is done (server, target client, remaining clients, mixtures), the metric
  identifiers from Table III, whether proxy data are needed, and whether code is
  open. Text on p. 11707 states these columns; the table cells are not in the
  extraction, so the per-method entries and the identifiers are unconfirmed.
- Entity performing unlearning (Sec. III-D, pp. 11703-11704): server (can keep
  historical models and updates but must link updates to identities), target client
  (has the data, security risk from malicious clients), remaining clients (hold the
  retain data), or mixed.

Design requirements (Sec. IV-A, p. 11704): (1) improved efficiency relative to
retraining; (2) retained or recovered performance (test performance comparable to the
retrained model; not necessarily to the original model); (3) effective forgetting
(comparable to the retrained model on forgetting data, and significantly different
from the original model on it); (4) same privacy level as regular FL.

Metrics discussed in Sec. IV-B (pp. 11704-11707), named from the text; the
identifier scheme in Table III is not in the extraction, so no IDs are used here:

- Efficiency: speed-up as the ratio of the time the baseline (for example the
  retrained model) needs to reach a test performance over the time the unlearned
  model needs (T_w* / T_wu); cumulative client-to-server communication cost, or the
  ratio Bytes_w* / Bytes_wu (higher is better; attributed to Halimi et al.); number
  of clients participating in recovery.
- Retained or recovered performance: test accuracy and loss; expected calibration
  error; symmetric absolute percentage error of test accuracy (SAPE, Eq. 4, range
  [0, 1], lower is better); for class unlearning, accuracy on the retained classes
  only.
- Forgetting verification: performance (loss, accuracy) on the target client's train
  data compared with the retrained model; the authors note this "can only be
  evaluated locally by clients since their data are private" (p. 11705); backdoor
  attack success rate; membership inference backdoor (MIB); membership inference
  attacks (MIA) with accuracy, precision, recall, where the unlearned and retrained
  models should show similar sensitivity; forgetting rate (FR, Eq. 5); reconstruction
  error and activation distance; loss, accuracy, influence function and KL
  divergence on "marker" subsets (Gao et al.); KL divergence of the output on
  forgetting data against a uniform distribution (Eq. 6); KL divergence between
  weight distributions of the unlearned and retrained models; for class unlearning,
  accuracy on the removed class (should approach 0) and KL between per-class
  accuracy distributions.
- Which metrics the survey recommends per objective (Table III's three rightmost
  columns) is UNCONFIRMED: those cells are not in the extraction.

Open problems named (Sec. VII, pp. 11712-11714), in the PDF's lettering: A
Standardization, B Incentive to Avoid Unlearning, C Multiple Unlearning Request,
D Feature Unlearning, E Security and Privacy, F Data Heterogeneity Among Clients.
Key statements:

- A: "a notable deficiency in standardization, particularly concerning the choice of
  datasets ... and the metrics employed" (p. 11712).
- C: "almost all the existing FU methods overlook" clients that "continually request
  multiple, sequential unlearning requests" (p. 11713).
- F: heterogeneity can make unlearning "easier" at low participation and high
  heterogeneity (Sec. III-B), "the influence of data heterogeneity on FU algorithms
  is often under-explored", and "the efficacy of proposed unlearning methods within
  FedAvg variants designed to address non-IID data [40], [101] is still an
  unexplored dimension" (p. 11713). References 40 and 101 are Li et al. (ICDE 2022,
  FL on non-IID data silos) and Mora, Fantini, Bellavista (empirical evaluation of
  FL algorithms with heterogeneous data). FedProx is not named in that sentence.

## 4. Assumption

Scope: horizontal FL with FedAvg-style aggregation and a single global model
(Sec. II-B); FU is triggered by a client, class or sample request; feature-level
unlearning is placed in vertical FL and called open (VII-D, p. 11713). The survey's
own experiments use a pretrained visual transformer and assume the evaluator can
measure accuracy on the target client's train data.

## 5. Evaluation

The survey's own experiments (Sec. III-B, pp. 11701-11703; Appendix, pp.
11714-11715):

- Datasets: CIFAR-100, Caltech-2011 birds, FGVC-Aircraft. Model: MiT-B0 visual
  transformer (about 3.6M parameters) pretrained on ImageNet-1k, with a new
  classification head; AdamW, client learning rate 3e-4, round-wise decay 0.998, 5
  local epochs, batch size 32, weight decay 1e-3; 100 FL rounds.
- Partitions: IID (about equal per-class examples) and non-IID by Dirichlet label
  skew with concentration parameter alpha = 0.1, following Hsu et al. (Appendix
  B, p. 11714).
- Clients per dataset for Fig. 4 and Table II: CIFAR-100, 100 clients with 10 chosen
  per round; Birds, 29 clients with 5 per round; Aircraft, 65 clients with 7 per
  round. Natural-forgetting runs (Figs. 5-6): 500 clients with 20 per round (4%
  participation), target client u participating only in the first 50 rounds
  (Fig. 5-6 captions, Appendix B).
- Seeds: not found in the text.
- Metrics used: accuracy on test data and on client u's train data, and KL divergence
  of the output distribution on client u's data to a uniform distribution
  (Appendix C, p. 11715).
- Table II ("Accuracy (%) of a global model trained with or without (w/o) a specific
  client in the federation", p. 11702): the numbers are NOT in the extracted text and
  could not be read. The text states only the qualitative result: test accuracy is
  "very similar" for the two models, while the model "exhibits significantly higher
  accuracy on client u's train data when the federation includes client u", with the
  same trend under homogeneous and heterogeneous data (p. 11702).
- Natural forgetting (Figs. 5-6, pp. 11702-11703): with homogeneous data, after
  client u leaves, "the global model steadily keeps generating more meaningful
  predictions than the retrained baseline on client u's train data for all the
  tracked rounds"; with heterogeneous data "catastrophic forgetting seems to help
  natural unlearning: after a few rounds, there is no sensible difference between the
  tracked metrics for the two versions" (p. 11703). The authors add that this holds
  in "the particular case of a large-scale federation with a low participation rate
  of clients and non-IID local datasets" (Sec. VI item 2, p. 11712).

Evidence labels for this thesis:

- (a) Established consensus, as the survey states it: retraining from scratch is the
  reference; an FU method should beat it on efficiency while staying close in test
  performance (Sec. IV-A). For efficiency and recovered performance "there is quite
  a consensus on the metrics", speed-up ratio and test accuracy; for forgetting
  verification the works "are less aligned", with MIA and backdoor success rate the
  most used (Sec. VI item 3, p. 11712). The survey also says a server that keeps all
  updates faces "unfeasible storage capacity" (Sec. III-C-2, p. 11703); that is an
  argument about a method class, not a measurement.
- (b) One paper's result: the memorization and natural-forgetting experiments above,
  on pretrained-ViT fine-tuning with one setup per dataset.
- (c) Contested or untested: whether Non-IID data help or hurt unlearning. The
  survey suggests heterogeneity "could be convenient for an FU algorithm" (Sec. III-C-5,
  p. 11703) while calling its effect on FU algorithms under-explored (VII-F). The
  thesis hypothesis H2 predicts degradation. The survey's evidence concerns natural
  forgetting after a client leaves, not an unlearning algorithm, so the two are not
  yet directly in conflict.

Weak-evaluation flags:

- Own experiments evaluate no unlearning algorithm; one alpha value (0.1); pretrained
  ViT; seeds not reported in the text read.
- Table II numbers, Tables III-IV contents and metric IDs could not be checked.

## 6. Limitation

A survey: useful for vocabulary, the taxonomy, metric names, and the open-problem
list, not for evaluation baselines. It condenses many methods in a few pages;
anything about a specific method must come from that method's own paper. Its own
experiments show memorization and slow natural forgetting but do not evaluate an
unlearning method. It names data heterogeneity (including FedAvg variants for
non-IID data), sequential requests, and standardization as open, which lines up with
this thesis's angle, but that alignment does not lock the contribution before the
Week 19 gate.

## Sources

- Local PDF checked via text extraction (pdfplumber); figures and Tables I-V not inspected:
  `articles/9. Federated_Unlearning_A_Survey_on_Methods_Design_Guidelines_and_Evaluation_Metrics.pdf`
  (IEEE TNNLS 36(7), July 2025).
- arXiv abs page (fetched directly): <https://arxiv.org/abs/2401.05146v3> — Nicolo Romandini, Alessio Mora, Carlo Mazzocca, Rebecca Montanari, Paolo Bellavista; v1 10 Jan 2024, v2 16 Feb 2024, v3 5 Nov 2024; journal ref TNNLS 2024.
- Crossref record (fetched directly): <https://api.crossref.org/works/10.1109/TNNLS.2024.3478334> — IEEE TNNLS vol. 36 issue 7 pp. 11697-11717, July 2025.

Still unconfirmed: Table II numbers; Table III metric identifiers and per-objective
recommendations; Table IV per-method entries; Table I; figure content; seeds of the
survey's experiments.

Source: articles/9. Federated_Unlearning_A_Survey_...pdf — local PDF checked via text extraction (pdfplumber); figures and tables not inspected; metadata VERIFIED on the journal PDF header, arXiv and Crossref — VERIFIED
