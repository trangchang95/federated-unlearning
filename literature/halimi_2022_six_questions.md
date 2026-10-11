# Halimi et al. (2022) — Federated Unlearning: How to Efficiently Erase a Client in FL?

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: the text read is arXiv 2207.05521 **v3** (submitted 20 Oct 2023; v1 12 Jul 2022,
v2 9 Sep 2022), read through the arXiv MCP HTML rendering. Cite as arXiv preprint 2207.05521;
the first posting is 2022, so the file and the matrix use 2022. No journal or proceedings
reference is listed on the arXiv abstract page, and the arXiv API record has no comment, journal
reference or DOI field other than the arXiv DOI. Semantic Scholar (fetched 2026-10-10) lists the
venue as "arXiv.org", year 2022, DBLP key journals/corr/abs-2207-05521 (a CoRR entry); the DBLP
page itself was not reached. The string "Keywords: Machine Learning, ICML" in the v3 text appears
to be template text; no venue is confirmed by it (scout inference). No workshop or conference
version was found in these records. v1 and v2 were not read, so numbers may differ from them. Section numbers below are the paper's own (Sec. 1 to 6,
Appendix A to D). Figures were not inspected as images: the MCP text has the captions and the
sentences that quote numbers, but not the plotted values, so every curve-level claim below is
"per caption/text". The HTML rendering has no page numbers.

Naming note: the paper calls its local step "projected gradient descent (PGD)" and says in a
footnote that it is technically projected gradient ascent (Eq. 3 and footnote, Sec. 4.2). The
thesis plan calls the same method PGA (called PGD by Halimi et al.). This note uses PGA from
here on, and "PGD" only inside quotes or when naming a paper's own wording.

## 1. Problem

One client (the target client) wants to leave a cross-silo FL system and have the influence of
its whole local dataset removed from the trained global model, without the server or any client
having stored update history (abstract; Sec. 1; Sec. 4.1). Retraining from scratch is the
"gold standard" but too costly in communication and computation (Sec. 1).

## 2. Gap

Earlier client-removal methods (FedEraser, Liu et al. 2021; Wu et al. 2022) need the server to
store the updates from each client "in every round" (Sec. 1; Sec. 2), which the authors consider
infeasible when updates are only held ephemerally for privacy reasons; Wu et al. also need
extra unlabeled server data (Sec. 1; Sec. 2). This is Halimi's characterization: FedEraser
itself retains updates only every Delta-t rounds (Sec. III-A of Liu et al.; Delta-t = 2 in its
experiments), see federaser_2021_six_questions.md. Other FU work targets a class (Wang et al.
2022) or sample subsets with convex-loss guarantees and inverse Hessians (Liu et al. 2022)
(Sec. 2); these two papers are as characterised by Halimi and are not in the matrix, so cite
them only as `[CITE?]`. Naive gradient ascent on
a whole client dataset fails because cross-entropy is unbounded, so the model drifts to an
arbitrary one (Sec. 4.2, "Local Unlearning").

## 3. Idea

Two phases (Sec. 4.2, Algorithm 1, Fig. 1):

1. Local unlearning at the target client. Reference model
   w_ref = (1/(N-1)) * sum over j != i of w_j^(T-1) = (N * w^T - w_i^(T-1)) / (N-1)
   (Sec. 4.2; same in Algorithm 1). The target client can compute it alone from the final
   global model w^T and its own last local model w_i^(T-1). The client starts at w_ref and
   maximizes its own local loss F_i (Eq. 2) inside the L2 ball of radius delta around w_ref,
   using the projected step w <- P(w + eta_u * grad F_i(w; b)) (Eq. 3). Early stopping when the
   distance condition with threshold tau fires (see the open point in the setup block).
2. FL post-training. The server and the N-1 non-target clients run FedAvg for T_p rounds
   starting from the locally unlearned model w_i^u, with the target excluded (Algorithm 1,
   "FL post-training").

Why the ball: it keeps the ascent from running off to a random model, and the radius delta
trades forgetting against damage to non-target clients' performance; "the combined choice of
delta and w_ref can be used to balance the performance trade-off" (Appendix A, schematic and
Fig. 10). The L2 choice is justified by reference to Thudi et al. 2021 as characterised by
Halimi (Appendix A; not in the matrix, so `[CITE?]`).

Why the ball is centred on w_ref and not on the final global model w^T: the paper says the
unlearned model should be "sufficiently close to a reference model that has effectively learned
the other clients' data distributions" (Sec. 4.2). It does not say why w^T is not used. Scout
inference, not in the paper: w^T was trained with the target client, so a ball around it would be
centred on a model that contains the target's influence.

## 4. Assumption

- Cross-silo setting: few clients, all participate each round, each holds a large dataset
  (Sec. 1; Sec. 3). One target client per request (Sec. 1).
- The target client keeps w^T (or can get it) and its own last-round local model w_i^(T-1)
  (Sec. 4.2). So "no stored history" means no history on the server or other clients; the
  target does need two vectors from the end of training. The paper does not discuss this
  requirement beyond the formula.
- Scout inference, not in the paper: the identity w_ref = (N w^T - w_i^(T-1))/(N-1) equals
  the average of the other clients' models only if FedAvg used equal weights 1/N. The paper
  partitions the training images equally across clients (Sec. 5, "Datasets and Model
  Architecture"), so equal weights hold in its experiments. With unequal client sizes (weights
  p_i = n_i / sum n_l, Sec. 3) the formula would need p_i. The paper does not discuss this.
  Derivation (scout, answer key for tutor Q3; uses the paper's own index convention and full
  participation): if w^T = sum_j p_j w_j^(T-1) with sum_j p_j = 1, then
  sum_{j != i} p_j w_j^(T-1) = w^T - p_i w_i^(T-1), so the p-weighted average of the non-target
  clients' models is (w^T - p_i w_i^(T-1)) / (1 - p_i). For p_j = 1/N this is
  (N w^T - w_i^(T-1)) / (N - 1), the paper's formula. For unequal p_j, the paper's formula
  with 1/N is not that weighted average, and it is also not the unweighted mean of the
  non-target models.
- "Approximate unlearning" aimed at performance close to retraining; no formal guarantee,
  because certified methods are said to be restricted to convex losses (Sec. 4.1). Evaluation
  is empirical via efficacy, fidelity, efficiency (Sec. 4.1; Sec. 5).
- Sec. 5 ("Unlearning Scenarios") says the goal of the backdoor and flip scenarios is "to
  deliberately differentiate the target client's data distribution from the data distribution
  of the other clients", and that a successful method should perform poorly on the target's
  distribution and well on the others'. Scout inference, not in the paper: the efficacy metrics
  therefore only work when the target's data is distinguishable by construction.

## 5. Evaluation

### Setup extraction

| Item | Value | Where |
|---|---|---|
| Datasets | MNIST, EMNIST (balanced), CIFAR-10 | Sec. 5, "Datasets and Model Architecture" |
| Model | CNN of McMahan et al. 2017: two 5x5 conv layers, FC 512 + ReLU, softmax; the paper states one architecture "for all datasets" and one count, 1,663,370 total parameters. Per-dataset parameter counts are not stated (scout inference: input size and class count differ across MNIST, EMNIST and CIFAR-10, so one count cannot hold for all three) | Sec. 5 |
| Number of clients N | 5 (target has 66% of its images modified) and 10 (80%) | Sec. 5.1; Sec. 5.2 |
| Partition | training images split equally across N clients (sample-balanced; IID is implied but the word "IID" is not used). No Dirichlet, no label skew. The target is made different by modifying 66% / 80% of its images | Sec. 5; Sec. 5.1; Sec. 5.2 |
| Scenarios | (i) backdoor: 3x3 pixel-pattern trigger via Adversarial Robustness Toolbox, label changed to 9 (MNIST), 't' (EMNIST), truck (CIFAR-10); (ii) horizontal flip, label kept; CIFAR-10 dropped from (ii) because "the accuracy of retraining on a hold-out test set of flipped images for N=5 is 71.14%. This value is similar to the accuracy on a hold-out test set of regular images", making CIFAR-10 "an inappropriate dataset for this scenario" (Sec. 5.2) | Sec. 5.1; Sec. 5.2 |
| Training rounds T of the original FL | not stated | Appendix A lists optimizer settings but no T |
| Local epochs (FL training) | E = 1 | Appendix A |
| FL training optimizer | SGD, momentum 0.9, lr 0.01, batch 128, FedAvg | Appendix A |
| Unlearning optimizer | SGD, momentum 0.9, lr eta_u = 0.01, batch B_u = 1024, E_u = 5 epochs | Appendix A |
| Projection radius delta | one third of the average Euclidean distance between w_ref and a random model, averaged over 10 random models (so it is computed per run from w_ref; the numeric value is not stated) | Appendix A |
| Gradient clipping | gradient L2 clipping with radius 5 | Appendix A. Algorithm 1 lists a "clipping radius delta" as an input, while Appendix A uses delta for the ball radius and 5 for gradient clipping. Not clear whether Algorithm 1's "clipping radius delta" is the ball radius or the gradient clip (5); the paper does not say |
| Early-stopping threshold tau | chosen "via a grid search over [2, 6]"; selected value and the grid selection criterion not stated | Appendix A |
| Early-stopping rule as printed | stop when ||w - w_i^(T-1)||_2 < tau (distance to the target client's own last local model) | Algorithm 1; Sec. 4.2 text. Open point: the text says the purpose is "to avoid learning an arbitrary model", which would suggest a cap on how far w moves. The printed inequality stops when w comes within tau of the target's model. I could not reconcile this from the text; ask the tutor or check the authors' code (no code link found in v3) |
| FL post-training | SGD, momentum 0.9, lr eta_p = 0.01, batch B_p = 128; local epochs E_p not given in Appendix A (Algorithm 1 lists E_p as an input); rounds T_p: 1 round for the backdoor accuracy and membership-inference results (Fig. 2 caption; Fig. 3 text), 5 rounds for the clean accuracy in Fig. 4 and for flipped and clean accuracy in Figs. 7, 8 (captions). The text also quotes clean and backdoor accuracy together at 1 round for MNIST (Sec. 5.1; App. C), and clean and flipped accuracy together at 1 round for MNIST N=10 (App. D). Figs. 6, 11, 12, 13 show accuracy versus rounds | Appendix A; Sec. 5.1; Sec. 5.2 |
| Baseline / gold standard | "Retraining from scratch" with the target excluded, started from a randomly initialized model; also the FedAvg model before unlearning as a reference. Number of retrain rounds not stated. No other FU method is run | Sec. 1; Sec. 5.1 (Fig. 6 discussion); Appendix C |
| Forgetting metrics (efficacy) | backdoor accuracy = share of triggered test images classified as the attacker's label; flipped accuracy; membership-inference accuracy against the target's data, using Shokri et al. 2017 (FedAvg model used as the shadow model, "for simplicity") and Yeom et al. 2018 (loss threshold), both as described in Halimi's App. B and not in the matrix (`[CITE?]` outside this note) | Sec. 5, "Efficacy"; Sec. 5.1; Appendix B |
| Retained-performance metric (fidelity) | clean accuracy on a hold-out set of unmodified images | Sec. 5, "Fidelity" |
| Cost metric (efficiency) | communication only: total MB of model updates the clients send to the server. Computation not measured ("even though we do not explicitly measure computation costs", Sec. 5.2). Wall-clock time: not stated. Storage: no server/client history is stored (abstract), but the two vectors the target needs are not sized | Sec. 5, "Efficiency"; Sec. 5.2 |
| Seeds / repeats | not stated | no mention anywhere in Sec. 5 or Appendix A |
| Variance / error bars | not stated in the text; whether figures show them was not checked (figures not inspected) | Sec. 5 |
| Compute hardware | not stated | |
| Comparison of PGA result to retrain within a noise floor | not done as far as the text shows: "similar to retraining" is the authors' reading of the figures, with no stated number of runs and no second retrain with a different seed | Sec. 5.1 (Fig. 2, 4, 6); Sec. 5.2 |

### Results quoted in the text (everything else is in figures not inspected)

- Backdoor, N=5, MNIST: after 1 round of post-training, clean accuracy 95.3% and backdoor
  accuracy 11.38%; retraining needs more than 5 rounds for similar performance (Sec. 5.1, Fig. 6
  discussion). N=10, MNIST: 96.6% clean and 12.35% backdoor after 1 round; retrain needs more
  than 5 rounds (Appendix C).
- Communication, backdoor, N=5, MNIST: to reach 98.13% clean accuracy, 167 MB (PGA) versus
  453 MB (retrain), 2.7x (Sec. 5.1, Fig. 5). "Up to 24x" for the backdoor scenario (Sec. 5.1;
  the dataset that gives 24x is not named in the text).
- Flip, N=5, MNIST: to reach 98.13% clean accuracy, 566 MB versus 1039 MB, 1.8x; up to 5.8x
  overall (Sec. 5.2, Fig. 9).
- Flip, N=10, MNIST: after 1 round, clean accuracy 95.47% and flipped accuracy 55.57%; retrain
  needs more than 10 rounds for similar performance (Appendix D). Retrain's flipped accuracy for
  this setting is not quoted in the text (figure not inspected), so whether 55.57% is close to
  retrain is not settled here. The Fig. 7 comparison uses 5 rounds.
- The Introduction states a reduction "by 5x to 24x" (Sec. 1, contribution 2), but the worked
  examples in Sec. 5.1 and Sec. 5.2 are 2.7x and 1.8x, and the flipping maximum is 5.8x. The
  lower end of 5x is not backed by a number I could find in the text.
- Membership inference (Fig. 3): shown for N=5 only; both attack accuracies for PGA are
  reported as "similar" to retraining and much lower than FedAvg (Sec. 5.1).

### Weak-evaluation flags

- No seeds, no variance, no noise floor: "similar to retraining" is not tested against the gap
  between two retrains (Sec. 5; Appendix A). This is the main point for the thesis.
- Partition is equal-sized and not label-skewed; no Non-IID experiment for the clients. The only
  heterogeneity is the target client's modified data, which is by design far from the others
  (Sec. 5, "Unlearning Scenarios"; Sec. 5.1).
- Forgetting is measured on the attacker's artefact (backdoor, flip) and MIA, not on distance to
  a retrained model's parameters or outputs. The flip scenario was dropped for CIFAR-10 because
  retrain's accuracy on flipped images (71.14%, N=5) is similar to its accuracy on regular images
  (Sec. 5.2, the paper's reason). Scout inference, not in the paper: the flip metric is
  dataset-dependent.
- Figs. 2 and 3 are read at 1 round of post-training and Fig. 4 at 5 rounds (captions). The text
  gives clean and backdoor accuracy together at 1 round for MNIST only (Sec. 5.1; App. C), so for
  EMNIST and CIFAR-10 the text does not show efficacy and fidelity at the same round (figures
  not inspected).
- Cost is communication only; the target client's local ascent (5 epochs on its whole dataset)
  and the tau grid search are not counted (Sec. 5.2; Appendix A). The tau grid search needs
  evaluation runs, and its selection criterion is not stated.
- Small CNN, cross-silo N = 5 and 10, three image datasets; no timing of retrain versus PGA.
- No comparison with FedEraser or any other FU method.

## 6. Limitation

The method is empirical only, with no guarantee (Sec. 4.1). It was tested with an equal split
and an obviously different target client; how the reference model w_ref, the radius delta (set
from distances to random models) and the early-stopping rule behave when client sizes and label
distributions differ is untested. The post-training rounds needed to recover accuracy and the
dependence on tau are not reported as a sweep. For this thesis: PGA is the baseline that needs
no stored history but does need a good delta, tau and a correct w_ref; behaviour under Dirichlet
client splits is not shown in this paper.

## For the tutor

1. Why does the target client maximize its loss inside an L2 ball around w_ref and not around
   the final global model w^T, and not with no ball at all? Which reference model is w_ref
   (Sec. 4.2), how can the target compute it alone, and what does the radius delta change in
   the trade-off between forgetting and retained accuracy on the non-target clients
   (Appendix A, Fig. 10)?
2. Fig. 2 shows the PGA model's backdoor accuracy as "similar to retraining" after 1 round.
   Using CLAUDE.md section 9, what extra comparison is needed before the thesis may call a
   client "forgotten", and why is one retrain per setting not enough? Why did the authors drop
   the CIFAR-10 flip scenario (Sec. 5.2), and what does that say about when a metric can
   separate a retrained model from the original one?
3. The paper gives w_ref = (N w^T - w_i^(T-1))/(N-1) as the average of the other clients'
   models (Sec. 4.2). Under what condition on the aggregation weights is that true, and what
   must change when the thesis uses Dirichlet client splits with unequal sizes (Sec. 3 weights
   p_i; Sec. 5 equal partition)?

Tutor key (scout notes, not paper claims unless a section is cited):
1. The paper's reason: the unlearned model should stay close to a reference model that has
   learned the other clients' data (Sec. 4.2); delta trades forgetting against retained accuracy
   (Appendix A, Fig. 10). Without a ball, loss is unbounded and ascent drifts to an arbitrary
   model (Sec. 4.2). Why not w^T: the paper does not say; scout inference is that w^T contains the
   target's influence. w_ref is computed from w^T and the target's own last local model (Sec. 4.2).
2. Needed: a second retrain with a different seed to get the noise floor, then check the
   PGA model lies within that gap on the metrics (CLAUDE.md section 9). Paper's reason for dropping
   CIFAR-10 flip: retrain's flipped accuracy (71.14%) is similar to its regular-image accuracy
   (Sec. 5.2). Scout inference: a metric is only useful where the retrained model and the original
   model differ on it by more than run-to-run variation.
3. See the derivation in section 4: with weights p_j, the average is
   (w^T - p_i w_i^(T-1)) / (1 - p_i); it equals the paper's formula only for p_j = 1/N, which
   holds for the equal split in Sec. 5. Under Dirichlet splits the target's p_i = n_i / sum n_l
   must be used (scout inference; also requires full participation and the paper's index
   convention).

## Sources

- arXiv MCP (HTML rendering of v3, full text read): arXiv 2207.05521v3, 20 Oct 2023.
- arXiv abstract page (fetched directly): <https://arxiv.org/abs/2207.05521> — title "Federated Unlearning: How to Efficiently Erase a Client in FL?", authors Anisa Halimi, Swanand Kadhe, Ambrish Rawat, Nathalie Baracaldo, v1 12 Jul 2022, v2 9 Sep 2022, v3 20 Oct 2023, DOI 10.48550/arXiv.2207.05521; no journal reference listed.
- arXiv MCP search record: same title, same four authors, categories cs.LG and cs.CR, first published 2022-07-12.
- arXiv API record (fetched 2026-10-10, <http://export.arxiv.org/api/query?id_list=2207.05521>): no comment, journal_ref or non-arXiv DOI field.
- Semantic Scholar API (fetched 2026-10-10): venue "arXiv.org", year 2022, DBLP key journals/corr/abs-2207-05521. No workshop or conference version listed.
- Not reached: the DBLP web page (access denied). A workshop version cannot be fully excluded.

Still unconfirmed: figure content (Figs. 2 to 13) including error bars and the dataset behind "24x"; the selected values of delta and tau; the direction of the early-stopping inequality; whether a code release exists; differences between v1/v2 and v3.

Source: arXiv 2207.05521v3 (arXiv MCP full text) — VERIFIED as arXiv preprint (title, authors, year, arXiv id from arXiv); no peer-reviewed venue found
