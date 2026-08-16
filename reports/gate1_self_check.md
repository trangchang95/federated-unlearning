# Gate 1 Beginner Self-Check

Gate 1 is not only a code checklist. The Month 1 plan says the student must be
able to explain how training works, how loss works, and why accuracy changes.
Answer these questions in your own words without copying the progress log.

1. For one training batch, what happens in the forward pass, loss calculation,
  backward pass, and optimizer step?

Answer:
Forward pass: Images pass through the model and generate 10 logits, corresponding to 10 digit classes.

Loss: Cross-entropy compares these logits with the true labels to measure the error.

Backward pass: loss.backward() calculates the gradient for each weight; this step does not change the weights yet.

Optimizer step: optimizer.step() uses the gradients to update the weights according to the learning rate.

The next batch is then processed using the updated model.

2. Why are training, validation, and test data kept separate? What would go
  wrong if the test set were used to choose the best epoch?

Answer: Train: Used to update the weights.

Validation: Used to select the best epoch/checkpoint.

Test: Used only for the final evaluation.

If test accuracy is used to select the epoch, information from the test set has influenced the model selection process. In this case, the test set is no longer independent, and the final results might be overly optimistic. In the current project, the checkpoint is selected using validation accuracy before being evaluated on the test set.

3. What does cross-entropy loss measure that accuracy does not show directly?
  Give an example where loss improves but accuracy stays unchanged.

Answer: Accuracy only cares about which class has the highest score; cross-entropy also reflects the model's confidence level. For example, if 7 out of 10 samples are still correct, the accuracy remains 70%, but if the model increases the probability for the 7 correct labels and reduces its confidence in the 3 incorrect predictions, the loss can decrease even if the accuracy remains unchanged. Therefore, loss and accuracy are related but do not measure the same thing.

4. What is a convolution filter, what is a feature map, and what does pooling
  do?

Answer: Filter: A small weight matrix (e.g., 3x3) that slides over the image to detect local patterns.

Feature map: A map that records the response level of a filter at each position on the image.

Pooling: Reduces the dimensions of the feature map; max pooling keeps the strongest response in each small region.

In the project's CNN, convolution helps preserve the spatial structure of the image instead of flattening it from the beginning like a Multi-Layer Perceptron (MLP).

5. Why did the MNIST CNN outperform the MLP in this project? State only what
  the experiment supports, without claiming the result is universal.

Answer: The measured results show that the MLP achieved 96.80%, while the CNN achieved 98.60%, which is 1.80 percentage points higher. This result supports the hypothesis that convolution better leverages the structure of neighboring pixels compared to an MLP that flattens the image. However, this experiment does not prove that a CNN is always better than an MLP because the two models have different architectures, different numbers of parameters, and were run with only a single fixed seed.

6. Why was CIFAR-10 accuracy much lower than MNIST accuracy, even though both
  experiments used the same basic CNN architecture?

Answer: The two datasets have different levels of difficulty. MNIST consists of grayscale digits, typically centered in the image with a simple background; CIFAR-10 consists of colored objects with diverse backgrounds, viewing angles, textures, and positions. Some classes like cats, dogs, and birds look quite similar at a 32x32 resolution. Therefore, achieving 98.60% on MNIST and 71.38% on CIFAR-10 does not mean the CIFAR-10 training failed; the learning curves for CIFAR-10 still showed improvement over the 10 epochs.

7. In the CIFAR-10 run, why was validation accuracy higher than training
  accuracy? Why is that not automatically a data leak?

Answer: During training, CIFAR-10 images are made more difficult using random crops and horizontal flips, and dropout is enabled. During validation, random augmentations are not used, and dropout is turned off via model.eval(). Thus, having a training accuracy of 65.36% but a validation accuracy of 72.24% at epoch 10 can be explained by the protocol. This difference alone does not prove data leakage; leakage only occurs if data or information is misused across splits.

8. What do random seed 42, the saved JSON config, and `best_model.pt` each
  contribute to reproducibility?

Answer: Seed 42: Makes sources of randomness such as splits, initialization, batch order, and augmentations repeatable.

JSON config: Records the dataset, model, optimizer, learning rate, batch size, epochs, seed, and experiment parameters.

best_model.pt: Saves the exact learned weights of the best checkpoint according to validation.

These three components solve three different problems; a single seed is not enough to reproduce the entire experiment.



9. Looking at the CIFAR-10 class results, which classes were hardest and what
  confusion was most visible? Why should later Non-IID experiments report
   class/client-level behavior rather than only global accuracy?

Answer: The hardest classes are bird (53.6%), cat (54.5%), and dog (55.1%). The most prominent confusion is 233/1000 dogs being predicted as cats; conversely, 110 cats are predicted as dogs.

In Non-IID Federated Learning, if one client mostly contains cats and dogs while another contains automobiles, the global accuracy might look good but could hide the fact that the first client is performing very poorly. Therefore, subsequent experiments need to track per-class and per-client utility, especially evaluating the model on non-target clients after unlearning.

10. In one paragraph, explain why the project is not allowed to start FedAvg
  until this gate is understood and checked.

Answer: FedAvg does not replace the basic training loop; instead, it places multiple local training loops on the clients and adds data partitioning, communication rounds, and model aggregation.

If one does not understand the forward pass, loss, backward pass, optimizer, how to split train-validation-test, and how to read metrics, it will be very difficult to tell whether a poorly performing global model is failing due to local training, data splitting, evaluation, or aggregation.

Gate 1 therefore requires a working centralized baseline, and the practitioner must be able to explain the results before increasing the complexity to Federated Learning.

Based on current progress, all technical artifacts for Month 1 have been completed; the missing piece to close Gate 1 is this self-check. The syllabus explicitly specifies that Gate M1 must ensure the learner can explain how the model is trained, how the loss works, and why the accuracy changes before moving on to FedAvg.



## Passing standard

A passing response does not need mathematical notation. It should be accurate,
use the student's own language, connect explanations to the measured Month 1
results, and distinguish evidence from guesses. After review, the Gate 1
checkbox in `README.md` can be updated and Gate 2 can begin.

## Review result

**Review date:** 2026-08-16

**Result:** PASS (10/10 answers meet the Gate 1 standard)

The answers demonstrate the three abilities required by Gate M1: explaining
how a model is trained, explaining what loss measures, and explaining why
accuracy changes. The review also checked the broader Month 1 foundations
needed to interpret an experiment correctly.

| Question | What the answer demonstrated | Result |
|---:|---|---|
| 1 | Correctly separates forward computation, cross-entropy loss, gradient calculation, and the weight update performed by `optimizer.step()` | Pass |
| 2 | Correctly assigns training, model selection, and final evaluation to the train, validation, and test splits | Pass |
| 3 | Correctly explains that loss uses confidence information while accuracy uses only the winning class | Pass |
| 4 | Correctly explains convolution filters, feature maps, pooling, and spatial structure | Pass |
| 5 | Uses the measured 96.80% and 98.60% results, calculates the 1.80 percentage-point difference, and avoids a universal claim | Pass |
| 6 | Correctly relates the MNIST/CIFAR-10 gap to task difficulty and uses the learning curve rather than accuracy alone to judge training | Pass |
| 7 | Correctly explains the effects of augmentation, dropout, and evaluation mode without misdiagnosing leakage | Pass |
| 8 | Correctly distinguishes repeatable randomness, recorded configuration, and saved learned weights | Pass |
| 9 | Correctly identifies the hardest CIFAR-10 classes and dog/cat confusion, then connects them to client-level Non-IID evaluation | Pass |
| 10 | Correctly explains why FedAvg builds on—not replaces—the centralized training loop and why the gate prevents ambiguous debugging | Pass |

Two precision notes are useful for future explanations, but neither changes
the pass decision:

- logits are the model's raw class scores; PyTorch cross-entropy internally
  converts those scores into the comparison needed for the class label;
- this project's CNN result supports an interpretation about spatial
  structure under this configuration, not a general proof that every CNN
  always beats every MLP.

Gate 1 is therefore complete. The next roadmap unit is Month 2, Week 5:
Federated Learning concepts and the six-question reading of the FedAvg paper.
