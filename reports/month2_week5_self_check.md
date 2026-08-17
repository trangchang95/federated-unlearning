# Month 2, Week 5 — Beginner Self-Check

Read `[month2_week5_fl_concepts.md](month2_week5_fl_concepts.md)` and the
six-question FedAvg note before answering. Use your own words. Short answers
are acceptable when they explain the reason, not only the definition.

1. What are the different responsibilities of a client and the server in one
  communication round?
   Answer: A client owns its local data, receives a copy of the global model, trains that copy using its local dataset, and sends the updated model or update back to the server. The server coordinates the round: it selects clients, sends the global model, collects their results, and aggregates them into the next global model. The server does not need to train directly on the clients' raw examples.
2. Trace one complete round from global model w_t to global model
  w_{t+1}. At which step does `optimizer.step()` act, and at which step
   does aggregation act?
   Answer:At the beginning of round t, the server has the global model wt​. It sends the same wt​ to the selected clients. Each client creates a local copy and trains it using the normal forward pass, loss calculation, backward pass, and optimizer steps. `optimizer.step()` changes the weights of that client's local model. After local training finishes, the clients return their updated models to the server. The server then performs aggregation to combine them and produce the new global model wt+1​.
3. What is the difference between one local epoch and one communication
  round? Can a communication round contain more than one local epoch?
   Answer: A local epoch is one complete pass through one client's local dataset. A communication round is the larger server-client cycle in which the server sends the global model, clients perform local training, return their models, and the server aggregates them. Yes, one communication round can contain multiple local epochs. For example, if E=5, every selected client may perform five local epochs before communicating with the server again.
4. Client A has 20 examples and returns weight 1.0. Client B has 80 examples
  and returns weight 3.0. What scalar weight does sample-weighted FedAvg
   produce? Why is the answer not the simple average 2.0?
   Answer: FedAvg produces 2.6. Client A owns 20% of the 100 examples, while client B owns 80%, so their returned weights receive coefficients 0.2 and 0.8 respectively: 0.2(1.0)+0.8(3.0)=2.6. The answer is not the simple average 2.0 because a simple average would give the 20-example client and the 80-example client equal influence despite representing different numbers of training examples.
5. What do C, E, and B control? Give one possible benefit and
  one possible cost of increasing local epochs E.
   Answer: C controls the fraction of clients selected in each communication round. E controls the number of local epochs each selected client performs before communicating again. B controls the local minibatch size used for each optimizer step. Increasing E can reduce the number of communication rounds because clients perform more learning locally, but it can also make local models drift farther apart, especially when client data distributions are different.
6. Why must selected clients start a round from the same global model rather
  than independently initialized models?
   Answer: Selected clients should start from the same global model so that their returned models represent different local updates applied to a common starting point. If clients started from independently initialized models, their final weights would reflect both different initializations and different local data, so averaging them would no longer have the intended FedAvg meaning. Starting from the same wt​ makes the differences mainly come from each client's local training during that round.
7. Why does “raw data stays on the client” not prove that Federated Learning
  is perfectly private?
   Answer: Keeping raw data on the client reduces the need to send training examples to a central server, but it does not prove perfect privacy. Clients still send model parameters or updates, and those updates may contain information about the local data that produced them. Therefore, Federated Learning by itself is not a formal privacy guarantee; additional privacy mechanisms would be needed for stronger guarantees.
8. Suppose global accuracy improves but one small client's accuracy becomes
  worse. Why could sample-weighted aggregation hide this, and what additional
   measurement should the experiment report?
   Answer: Sample-weighted aggregation gives a small client less influence because its dataset represents only a small fraction of the total examples. Therefore, improvements on larger clients can increase global accuracy while poor performance on a small client is hidden by the aggregate metric. The experiment should also report per-client accuracy or other client-level utility metrics, not only global accuracy.
9. In your own words, explain why FedAvg uses more local computation to reduce
  communication, and why setting a very large number of local epochs is not
   automatically better.
   Answer: FedAvg allows each client to perform multiple local optimizer steps before sending its result back to the server. This spends more computation on the client but can reduce how often server-client communication is required. However, using a very large number of local epochs is not automatically better because local models can move farther away from each other, especially when their data distributions differ. This can cause slower convergence, plateauing, or unstable training.
10. Why must this project make ordinary FedAvg work before implementing any
  Federated Unlearning method?
    Answer: Federated Unlearning operates on a model whose parameters have already been influenced by clients through ordinary federated training and aggregation. Therefore, we first need a correct and reproducible FedAvg baseline so that we understand how client contributions enter the global model. Otherwise, if an unlearning result is wrong, we would not know whether the problem comes from the unlearning method or from the underlying federated training implementation. FedAvg therefore provides both the technical foundation and the reference point required to evaluate Federated Unlearning.



## Passing standard

A passing response should correctly trace the client/server workflow,
distinguish local optimization from server aggregation, calculate the weighted
example in Question 4, explain the main parameter trade-offs, and avoid
claiming that FL alone guarantees privacy. Before the dated review below,
Week 6 remained blocked until these answers were reviewed.

## Review result

**Review date:** 2026-08-18

**Result:** PASS (10/10 answers meet the Week 5 standard)

| Question | What the answer demonstrated | Result |
|---:|---|---|
| 1 | Correctly separates client-side data and local training from server-side coordination and aggregation | Pass |
| 2 | Correctly traces one complete round and distinguishes local `optimizer.step()` from server aggregation | Pass |
| 3 | Correctly distinguishes a local epoch from a communication round and explains that one round may contain several local epochs | Pass |
| 4 | Correctly calculates the sample-weighted result as 2.6 and explains why an unweighted mean would misrepresent dataset sizes | Pass |
| 5 | Correctly explains client fraction `C`, local epochs `E`, batch size `B`, and the communication-versus-drift trade-off | Pass |
| 6 | Correctly explains why selected clients must begin from the same global weights | Pass |
| 7 | Correctly rejects the claim that keeping raw data local is a complete privacy guarantee | Pass |
| 8 | Correctly explains how global accuracy can hide poor utility for a small client and asks for client-level measurements | Pass |
| 9 | Correctly explains why extra local computation may reduce communication while very large `E` can harm convergence | Pass |
| 10 | Correctly identifies working FedAvg as the technical and evaluation foundation required before Federated Unlearning | Pass |

No conceptual correction is required. Question 8 can later be extended with
summary measures such as worst-client accuracy or macro-average client
accuracy, but the supplied per-client recommendation already meets the Week 5
requirement.

Week 5 is complete. Hand-written FedAvg is now permitted as the Week 6 task.
An FL framework remains blocked until that hand-written implementation works.
