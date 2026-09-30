# Fraboni et al. (2024) — SIFU: Sequential Informed Federated Unlearning

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: the local PDF is the PMLR camera-ready (page 1 footer: "Proceedings of
the 27th International Conference on Artificial Intelligence and Statistics
(AISTATS) 2024, Valencia, Spain. PMLR: Volume 238"), 19 pages (main text pp. 1-9,
checklist p. 12, appendices A-F pp. 13-19). arXiv 2211.11656 has five versions
(v1 21 Nov 2022 to v5 16 Mar 2024); the arXiv versions were not compared with this
file. Page numbers are PDF pages. Read from a pdfplumber text extraction of the
local file; figures were not inspected (all quantitative plots are figures, so
per-dataset numbers are not available from the text). Math in the extraction is
partly garbled; only readable formulas are quoted.

## 1. Problem

Remove one or several clients' contribution from a FedAvg-trained global model with
a formal unlearning guarantee, without extra client-side computation, without
server-held data or client Hessians, and without assuming convex local losses
(abstract p. 1; Sec. 1 p. 2).

## 2. Gap

Most centralized MU schemes need high-order quantities or extra sensitive client
information, so they do not carry over to FL (Sec. 1, p. 2). Existing FU methods
with FedAvg either need client-side Hessians or independent server data, or lack
guarantees (FedEraser is "not backed by theoretical guarantees"; Sec. 2.3, p. 3).
Table 1 (p. 4) summarizes each method's working assumptions and claims SIFU is the
only one ticking all columns; the tick marks are not recoverable from the text
extraction.

## 3. Idea

Use each client's measured influence to decide how far back to roll the model
(Sec. 3, Algorithm 2 "IFU", pp. 3-5).

- Learning phase: while running FedAvg, the server computes for every round n and
  client i a bounded sensitivity Psi(n, i) (Eq. 9) from the client updates it
  already receives (p. 5).
- Unlearning of client c: T = the largest round n with Psi(n, c) <= Lambda*
  (Eq. 12), where Lambda* is set from the chosen noise scale sigma and the budget
  (epsilon, delta) (Eq. 11). Then theta_tilde = theta_T + Gaussian noise
  N(0, sigma^2 I), and FedAvg is run for N_tilde rounds on the remaining clients
  starting from theta_tilde (Algorithm 2, p. 5).
- SIFU (Algorithm 3, Sec. 4, pp. 5-7) handles a series of requests {W_u}: the client
  set shrinks to I_u = I_(u-1) minus W_u, the rollback index is recomputed with the
  training history H(u), and earlier requests stay covered (Thm 3).
- Low sigma makes SIFU behave like Scratch, high sigma like Last (Sec. 5.4, p. 9).
- The sigma in Table 2 is the same sigma as in Theorem 2 and Lambda*: it is chosen as
  a hyperparameter, and (epsilon, delta) = (10, 0.01) only sets the threshold that
  picks the rollback round (App. E, p. 15: "The unlearning budget plays the important
  role of identifying in the training history the global model to perturb").

## 4. Assumption

What the "provable" claim covers, precisely:

- Definition 1 (p. 3, Eqs. 2-3): a randomized mechanism f_m achieves
  (epsilon, delta)-unlearning of a data point {x_m, y_m} from M(D) if, for every
  measurable set S of parameters and D_-m := D minus that point,
  P(f_m(M(D)) in S) <= e^epsilon P(f_m(M(D_-m)) in S) + delta, and the same with the
  two swapped. This is a differential-privacy-style indistinguishability statement
  between the mechanism applied to the full-data model and applied to the
  without-the-data model. It is written for a single data point; Thm 2 applies it
  "of client c" without restating it for a client, so the client version is an
  extension the paper takes as given. It is not a claim that the model equals a
  retrained model, and it says nothing about accuracy.
- Object certified by Thm 2 (p. 4): the noised model theta_n + N(0, sigma(n,c)^2 I),
  with sigma(n,c) := [2(ln 1.25 - ln delta)]^(1/2) epsilon^(-1) Psi(n,c) (Eq. 10),
  "under smoothness assumptions". The proof (App. C, p. 14) says it "follows directly
  from Theorem 1 coupled with Theorem A.1 of (Dwork and Roth, 2014)", the Gaussian
  mechanism. The paper does not state a range for epsilon; whether epsilon = 10 lies
  inside the conditions of the cited theorem was not checked here.
- Thm 1 (p. 4): for smooth local losses (Lipschitz-continuous gradients),
  alpha(n,c) <= Psi(n,c), where alpha(n,c) is the actual distance between FedAvg
  run with and without client c after n rounds (Eq. 6) and Psi(n,c) is the sum over
  s = 0..n-1 of B^((n-s-1)K) times delta_c(I, theta_s) (Eq. 9); delta_c is the norm
  of the difference between the aggregated update with and without c (Eq. 7).
- B(f, eta), from Appendix B (pp. 13-14): B = 1 + eta L if the losses are L-smooth
  (Eq. 26, includes non-convex); B = 1 if smooth and convex with eta <= 2/L (Eq. 27);
  B < 1 if smooth and mu-strongly convex, with the learning-rate condition eta <=
  2/(L + mu) (Eq. 28; the exact formula is garbled in the extraction and is not
  reconstructed here). Main text (p. 4): B < 1, = 1, > 1 for strongly convex, convex,
  non-convex.
- Proof structure of Thm 1 (App. B.2.2, p. 14): uses Lemma 3.7 of Hardt et al.
  (2016) (contractivity of one gradient step), applied K times; the proof writes the
  aggregation as a weighted sum over all M clients, and Algorithm 1 (p. 3) sends the
  model to every client in I. Appendix B.1 defines the trained models with "K GD
  steps". Experiments instead sample m of M = 100 clients per round and use
  minibatch SGD with batch size B (Table 2, App. E p. 15). How client sampling and
  minibatch randomness are coupled between the runs with and without c is not
  addressed in the text I read. Treat this as an open question; do not assume the
  theorem covers the experimental protocol.
- Non-convex case: the formal bound holds with B = 1 + eta L > 1, which enters as
  B^((n-s-1)K) (Eq. 9). In every experiment "we set B = 1. The choice for B is
  experimentally justified in Appendix F" (Sec. 5.1, p. 7), including for the CNNs.
  Appendix F (p. 17, Fig. 7): "we empirically measure alpha and Psi for all our
  datasets with a large set of hyper-parameter choices ... The bound from Theorem 1
  holds for every experimental scenario"; the caption says it holds "even when
  setting B = 1 in neural networks". So for the neural-network experiments the
  guarantee is checked empirically, not proved. Computing alpha needs a second
  training run without client c, so this check is not available at deployment.
  Reviewer inference, not in the paper: with the true B > 1 the sum would grow with
  the number of rounds and local steps, which would push the rollback round earlier
  and shrink the savings.
- Thm 3 (p. 7): "The model theta^(N_u)_u obtained with SIFU satisfies
  (epsilon, delta)-unlearning for every client in current and previous unlearning
  requests." The proof (App. D, p. 15) is an induction that, as read, shows the
  perturbed model theta'_u (the noised rollback model, before the N_u retraining
  rounds) unlearns every client in F_u. The step from the perturbed model to the
  model after N_u FedAvg rounds on the remaining clients is not visible in the text
  I read; check the proof on the page image.
- Server-side state: the server tracks Psi(n, i) for each client and round, and "one
  global model checkpoint per client" (p. 7). The text says "The computational cost
  for this operation is negligible" and, on p. 5, that only one version of the global
  model must be kept per client who may later ask to be unlearned. Storage of these
  M checkpoints is not quantified. The model history H(u) is "not actually stored"
  (p. 6).
- FedAvg convergence guarantees are inherited from Wang et al. (2020) and Li et al.
  (2020) (p. 5). "The retraining with IFU is generally faster" than random-init
  retraining is stated as a claim (p. 5), not proved.
- Client cost: none in the main protocol; if clients use gradient masking they can
  compute their own delta_c (Eq. 13, p. 5).
- The unlearned unit is a whole client; requests are client removals.

## 5. Evaluation

Setup (Sec. 5.1 p. 7; Table 2, App. E p. 15):

- Datasets: adapted versions of CIFAR-10, CIFAR-100, MNIST, FashionMNIST, CelebA.
  M = 100 clients, 100 data points each (p. 7).
- Partition (all skewed): MNIST and FashionMNIST, one class per client, each class in
  10 clients; CIFAR-10 and CIFAR-100, per-client class ratios from a Dirichlet
  distribution with parameter 0.1; CelebA, each client owns one celebrity's samples
  (p. 7). "With these five datasets, we consider different levels of heterogeneity
  based on labels and features distribution" (p. 7). In the watermark scenario the
  data are partitioned with Dirichlet parameter alpha = 1 (Sec. 5.3, p. 8). No IID
  control and no controlled sweep of the heterogeneity level were found: the two
  Dirichlet values (0.1 and 1) belong to different scenarios and datasets.
- Models: logistic regression on MNIST (convex case); on the other four datasets a
  network with three convolutional layers and two fully connected layers (Sec. 5.1
  p. 7; App. E p. 15).
- Table 2 (App. E, p. 15), columns K (SGD steps), B (batch size), M (clients),
  m (sampled clients), sigma (noise standard deviation), eta (learning rate), C
  (defined in App. E only through "clipping value C" for the DP baseline): CIFAR10
  5, 20, 100, 5, 0.05, 0.01, 0.2; CIFAR100 5, 20, 100, 5, 0.05, 0.02, 0.2; MNIST 10,
  100, 100, 10, 0.05, 0.01, 0.5; FMNIST 5, 20, 100, 10, 0.1, 0.02, 0.5; CelebA 10, 20,
  100, 20, 0.1, 0.01, 0.5.
- Budget: epsilon = 10 and delta = 0.01 for SIFU, DP and Last (App. E, p. 15).
- Seeds: "we replicate each unlearning scenario on 10 different seeds and plot in
  Figure 4 to 6 their averaged results" (p. 15); the Fig. 4 caption says "variability
  estimated on 10 seeds" (p. 16). Whether Figs. 2-3 use the same 10 seeds was not
  found.
- Compute: Nvidia 1080TI GPU (p. 16). Number of initial training rounds N: not found
  in the text.
- Baselines (Sec. 5.1, pp. 7-8): Scratch, Fine-Tuning, DP, FedEraser, FedAccum, and
  Last (adapted from Neel et al. 2021). All share the same local learning rate, SGD
  steps and optimizer. How FedEraser's calibration and storage interval were
  configured was not found in the main text.
- Scenarios: (1) time-constrained (Sec. 5.2, Fig. 2): unlearning time limited to 25%,
  33% and 50% of the training time (p. 7); (2) watermark verification (Sec. 5.3,
  Fig. 3, following Sommer et al. 2020): U = 3 sequential requests of 10 random
  clients each (for MNIST and FashionMNIST, 10 clients owning the same class), then
  fine-tuning on the remaining data until accuracy on remaining clients exceeds
  93% (MNIST), 99.9% (CelebA), 90% (FashionMNIST, CIFAR10, CIFAR100), with a minimum
  of 50 and a maximum of 10,000 rounds (p. 8); (3) the same sequential protocol
  without watermarks (App. E, Fig. 4, p. 16).

How forgetting was measured: absolute accuracy difference to Scratch on the forget
set and on the retain set (Fig. 2 caption, p. 8); accuracy of the unlearned clients,
"the lower the better" (Figs. 3-4 captions); the watermark check of Sommer et al.
(Sec. 5.3 title: "Verifying Unlearning with Watermarking"). No membership-inference
attack and no parameter distance to the retrained model were found.

Cost definition: the number of FL aggregation rounds (Figs. 3-4 captions: "Total
amount of aggregation rounds"). Sec. 5.1 (p. 7) also says success is evaluated by
"running time", and the time-constrained scenario uses a time limit; how running
time was measured was not found. Whether the initial training rounds are counted
was not stated for SIFU.

Results (text only; per-dataset values are in figures, not inspected):

- Sec. 5.2 (p. 7), time-constrained scenario: "SIFU outperforms its competitors on 4
  out of 5 datasets"; on every neural-network experiment "SIFU is the only method to
  achieve a good trade-off between forgetting quality and accuracy on the retain
  set". The four datasets are not named in the text.
- App. E (p. 16), sequential scenario (Fig. 4): "SIFU requires a sensibly lower
  number of iterations than Scratch (52% faster on average) to achieve similar
  unlearning performances"; FedEraser has comparable unlearning "while however
  requiring a higher number of iterations than Scratch (12% on average)".
- App. E (p. 17, Fig. 5 caption): at optimal sigma the speed-ups are "between
  two-fold and five-fold". Relation to the 52% figure is not stated.
- Sec. 5.3 (p. 9): on every dataset where a CNN is used, SIFU outperforms every other
  method in the watermark scenario.

Storage of historical information: see Assumption (per-client sensitivity plus one
checkpoint per client, not quantified).

Weak-evaluation flags:

- Non-IID only, at two Dirichlet values in different scenarios; no IID control and
  no sweep, so the effect of heterogeneity level cannot be read from this paper.
- 100 samples per client, small models; results shown as figures only.
- Formal guarantee not proved for the CNN runs (B = 1); it relies on an empirical
  check that needs a retrained-without-c run; epsilon = 10.
- Proof covers full participation and gradient steps (as read); experiments use
  client sampling and SGD.
- Forgetting judged against Scratch by accuracy difference and watermark; no MIA or
  distance-to-retrain.
- The 52% and 12% values are averages of round counts across five datasets in one
  scenario (Fig. 4); the text gives no per-dataset numbers.
- Storage and tracking cost stated as negligible (computation) but storage unmeasured.

## 6. Limitation

The certificate covers the noised rollback model under smoothness with a sensitivity
bound; the CNN experiments set B = 1, which the theory does not license for
non-convex losses; the proof as read assumes full participation and gradient steps;
the extension to the model after retraining is not visible in the text read.
Speed depends on how far back the rollback round T can go, which depends on client
c's measured influence Psi(n, c). Reviewer hypothesis (not tested in the paper):
under stronger heterogeneity a client's update differs more from the others, Psi
grows, T moves earlier, and the round savings shrink. Cost is counted in rounds, not
measured wall-clock; storage of one checkpoint per client is unmeasured. For this
thesis: SIFU is the sequential baseline with a certificate under stated assumptions;
the thesis must say "(epsilon, delta)-unlearning of the noised rollback model under
smoothness, with B set empirically in the CNN experiments", not "provably unlearns".

## Sources

- Local PDF checked via text extraction (pdfplumber); figures not inspected:
  `articles/8. fraboni24a-SIFU-Sequential Informed Federated Unlearning for Efficient and Provable Client Unlearning in Federated Optimization.pdf`
  (PMLR camera-ready, AISTATS 2024).
- PMLR page (fetched directly): <https://proceedings.mlr.press/v238/fraboni24a.html> — AISTATS 2024, PMLR 238, pp. 3457-3465, 2-4 May 2024; authors Yann Fraboni, Martin Van Waerebeke, Kevin Scaman, Richard Vidal, Laetitia Kameni, Marco Lorenzi. Note the PDF prints the author order Fraboni, Van Waerebeke, Vidal, Kameni, Scaman, Lorenzi (first-page layout), while the PMLR page and arXiv list Scaman before Vidal; the order was not reconciled.
- arXiv abs page (fetched directly): <https://arxiv.org/abs/2211.11656> — five versions, v1 21 Nov 2022, v5 16 Mar 2024, no journal reference listed.

Still unconfirmed: all figure content (Figs. 2-7) including which four datasets SIFU
wins on and the per-dataset round counts; the strongly-convex formula for B
(garbled); the exact wording of Thm 1-3 formulas beyond what is quoted (math
layout garbled); whether the epsilon = 10 setting fits the cited Gaussian-mechanism
theorem's conditions; the step from perturbed to retrained model in Thm 3; author
order (PDF vs PMLR/arXiv listing).

Source: articles/8. fraboni24a-SIFU-...pdf — local PDF checked via text extraction (pdfplumber); figures not inspected; metadata VERIFIED on the PMLR page and arXiv — VERIFIED
