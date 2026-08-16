# Month 2, Week 5 — Beginner Self-Check

Read [`month2_week5_fl_concepts.md`](month2_week5_fl_concepts.md) and the
six-question FedAvg note before answering. Use your own words. Short answers
are acceptable when they explain the reason, not only the definition.

1. What are the different responsibilities of a client and the server in one
   communication round?

   Answer:

2. Trace one complete round from global model \(w_t\) to global model
   \(w_{t+1}\). At which step does `optimizer.step()` act, and at which step
   does aggregation act?

   Answer:

3. What is the difference between one local epoch and one communication
   round? Can a communication round contain more than one local epoch?

   Answer:

4. Client A has 20 examples and returns weight 1.0. Client B has 80 examples
   and returns weight 3.0. What scalar weight does sample-weighted FedAvg
   produce? Why is the answer not the simple average 2.0?

   Answer:

5. What do \(C\), \(E\), and \(B\) control? Give one possible benefit and
   one possible cost of increasing local epochs \(E\).

   Answer:

6. Why must selected clients start a round from the same global model rather
   than independently initialized models?

   Answer:

7. Why does “raw data stays on the client” not prove that Federated Learning
   is perfectly private?

   Answer:

8. Suppose global accuracy improves but one small client's accuracy becomes
   worse. Why could sample-weighted aggregation hide this, and what additional
   measurement should the experiment report?

   Answer:

9. In your own words, explain why FedAvg uses more local computation to reduce
   communication, and why setting a very large number of local epochs is not
   automatically better.

   Answer:

10. Why must this project make ordinary FedAvg work before implementing any
    Federated Unlearning method?

    Answer:

## Passing standard

A passing response should correctly trace the client/server workflow,
distinguish local optimization from server aggregation, calculate the weighted
example in Question 4, explain the main parameter trade-offs, and avoid
claiming that FL alone guarantees privacy. Week 6 remains blocked until these
answers are reviewed.
