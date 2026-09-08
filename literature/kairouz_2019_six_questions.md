# Kairouz et al. (2019/2021) — Advances and Open Problems in Federated Learning

**Paper:** *Advances and Open Problems in Federated Learning*
**Venue:** Foundations and Trends in Machine Learning, Vol. 14, No. 1-2 (also
widely circulated as an arXiv preprint from 2019)

This note follows the thesis plan's fixed six-question paper-reading method.
It is a large, multi-author survey rather than a single-method paper, so the
"Idea" and "Evaluation" sections below describe how it organizes the field
rather than one experiment.

## 1. Problem

Federated Learning research had grown quickly after McMahan et al. (2017),
but work was scattered across communication efficiency, privacy, robustness,
fairness, and personalization as separate threads. The survey asks: what does
the *whole* FL problem look like once every one of these axes is considered
together, and where is the field still missing methods or even a clear
problem statement?

It also formalizes a distinction the thesis plan already uses informally:
**statistical heterogeneity** (clients hold different, often non-IID, data
distributions) versus **systems heterogeneity** (clients differ in compute,
memory, connectivity, and availability, and are not reliably addressable the
way data-center workers are).

## 2. Gap

Before this survey, most FL papers treated their own axis (e.g., compression
for communication, or a single privacy mechanism) largely in isolation. Two
gaps motivated a unifying document:

- **Cross-device vs. cross-silo conflation.** Many papers did not clearly
  separate the setting FedAvg targeted (millions of unreliable, unaddressable
  mobile/edge clients, "cross-device") from a different setting with a
  handful of reliable, addressable organizations such as hospitals or banks
  ("cross-silo"). Techniques reasonable in one setting (e.g., assuming every
  client answers every round) are not reasonable in the other.
- **No shared map of open problems.** Without one, it was hard for a new
  researcher to see which combinations of heterogeneity, privacy, and
  robustness requirements were already handled and which were not.

## 3. Idea

The survey's contribution is a taxonomy and a systematic open-problems list,
not one algorithm. It organizes FL work around several core challenges, each
with its own advances and open problems:

- **Communication efficiency**: compression, quantization, and structured or
  sketched updates that reduce what each round transmits, extending FedAvg's
  own strategy of trading local computation for fewer rounds.
- **Statistical heterogeneity (Non-IID data)**: client-drift-aware objectives
  (the survey explicitly discusses proximal-term methods, i.e., the family
  FedProx belongs to), personalization, and multi-task formulations, since a
  single global model can serve skewed clients poorly.
- **Systems heterogeneity and partial participation**: handling clients that
  drop out, arrive late, or never participate, without assuming synchronous,
  fully reliable workers.
- **Privacy**: secure aggregation (the server learns only the sum/average of
  updates, not any individual client's update) and differential privacy
  (adding calibrated noise so a single client's data cannot be confidently
  inferred from the published model), and the trade-offs each introduces
  against utility and communication cost.
- **Robustness and fairness**: resistance to poisoned or adversarial updates
  from a minority of clients, and ensuring the trained model does not perform
  much worse for some clients or subgroups than others.

## 4. Assumption

The survey is written primarily from the **cross-device** perspective:
massive numbers of resource-constrained, intermittently available clients
that the server cannot individually track between rounds, communicating over
unreliable networks. Cross-silo FL is discussed explicitly as a distinct
setting with different practical constraints (fewer, addressable,
higher-availability participants), and the survey is careful not to present
techniques suited to one setting as automatically transferable to the other.

It assumes the reader already understands basic FedAvg-style training
(client selection, local computation, server aggregation) and builds the
taxonomy on top of that foundation rather than re-deriving it.

## 5. Evaluation

As a survey, it reports no new benchmark numbers of its own. Its
"evaluation" is the breadth and organization of prior work it assembles:
dozens of cited communication-compression, privacy, robustness, and
personalization methods are grouped under the taxonomy above, each with the
setting (cross-device or cross-silo), the problem it addresses, and open
sub-problems the authors identify as unresolved at the time of writing.

## 6. Limitation

The paper proposes no new algorithm to adopt directly, and it is a snapshot
of open problems as of 2019 (largely written by authors at Google and
collaborating institutions), so its emphasis skews toward production-scale
cross-device concerns; smaller academic cross-silo studies receive
comparatively less depth. Some problems it flags as open (for example,
FedProx-style heterogeneity handling, and various secure-aggregation
protocols) have since seen substantial follow-up work the survey obviously
could not include. For this thesis, its main value is vocabulary and
scoping: it gives a precise way to say *which* kind of heterogeneity a
client-level experiment is testing (statistical vs. systems) and confirms
that this thesis's chosen path — statistical heterogeneity via Non-IID
partitioning, then client-level Federated Unlearning — sits squarely inside
one well-recognized branch of the field rather than a fringe combination.

## Sources

- Foundations and Trends in Machine Learning record:
  <https://doi.org/10.1561/2200000083>
- Author preprint: <https://arxiv.org/abs/1912.04977>
