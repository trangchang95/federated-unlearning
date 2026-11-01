# Ding, Huang, Duan (2026) — Initiating Removal from Operational Symptoms in Federated Vehicular Networks (ATONE)

This note follows the thesis plan's fixed six-question paper-reading method.
It is intentionally concise rather than a chapter-style summary.

Version note: arXiv preprint 2610.22584v1 (submitted 15 Oct 2026). No venue
beyond arXiv seen. Full text read from the arXiv HTML rendering: Introduction
and Sec. V (Performance Evaluation) and VI; Sec. II–IV (method derivation,
Theorems 1–2) read only as far as quoted in the Introduction/abstract.

## 1. Problem

Federated unlearning normally starts from a known target ("forget client X").
Here a faulty contribution is only discovered as a delayed *symptom* (e.g. a
pedestrian labelled road), so the system must infer what to remove (Abstract;
Sec. I).

## 2. Gap / Method

**Gap** (Sec. I): FU assumes the request names the target; poisoning defences
inspect updates at submission time; post-hoc tracing needs a labelled
misclassified input and stored checkpoints.

**Method** (Abstract; Sec. I): ATONE puts unlearning inside the aggregation
rule: project everyday errors and symptoms onto a subspace of recent global
increments, propagate each contribution's surviving effect with curvature
estimated from those increments, score all contributions in one backward pass,
suppress suspects every round, eliminate confirmed ones after a one-round
probe. Exactness proved "when training is flat outside the subspace".

## 3. Assumption

- A symptom-reporting signal and some recent driving data exist at vehicles.
- Harm persists along flat directions in a low-dimensional subspace (q = 8
  default).
- Clients may be unavailable (range of roadside unit), and a compromised client
  could fabricate a symptom (Sec. I).

## 4. Evaluation

- Setup (Sec. V-A): CARLA simulation, N = 50 vehicles, LR-ASPP + MobileNetV3
  segmentation; Foggy Cityscapes, ACDC, BDD100K; physical MentorPi/Raspberry
  Pi 5 testbed on Foggy Cityscapes.
- Partition: "each vehicle receives a different share of the training split" —
  no Dirichlet α or shard rule stated.
- Faulty contribution: pedestrians relabelled as road for one vehicle during
  one window.
- Baselines (Sec. V-A): FedRecover, FedOSD, FedEditor (told the target);
  FLForensics, FLTracer (trace, then FedRecover removes). **Not** FedEraser/PGA.
- Metrics: FmIoU, RmIoU, TmIoU, MIA success, residual harm H̄, reaction delay,
  on-board time and uplink.
- Retrain reference: yes (agreement with Retrain within 0.36 pts).
- Rounds, seeds and number of runs: not stated in the sections read.

## 5. Main results (with table/section references)

- Sec. V-B / Table I: within 0.36 pts of Retrain on FmIoU/RmIoU/TmIoU and
  within 3.3 pts on MIA; H̄ is 43–61% below FedEditor in eight of nine
  settings (FedEditor is the strongest baseline).
- Sec. V-B: delay ≤ 5.7 rounds vs 12.3–36.7 for tracing baselines; testbed
  time and uplink 2.7–7.3× and 1.5–4.1× lower than those baselines.
- Fig. 7a: scoring < 6 ms up to N = 200 and storage 206 MB vs FLTracer 63.5 s
  and 10.6 GB.

## 6. Limitation

Not a classification benchmark and not non-IID-controlled; the removal target
is *discovered*, not requested. Scout: one fault type (pedestrian→road); the
claim "eliminate exactly" depends on a flat-outside-subspace assumption; no
Limitations section in what was read.

**Relation to the thesis's baselines:** shares only Retrain-as-reference. A
useful adjacent idea (inferring the target) but out of the thesis's
client-level, known-target scope; kept as borderline-in-scope.

## For the tutor

1. How does ATONE's problem differ from a "forget client X" request?
2. What does "within 0.36 points of Retrain" assume about Retrain?
3. Which data-heterogeneity details are missing, and why does that matter for
   the thesis?

Source: https://arxiv.org/abs/2610.22584 (HTML render https://arxiv.org/html/2610.22584v1) — VERIFIED (title, authors, arXiv ID, year from arXiv page; no venue/DOI; method sections and theorems only partly read).
