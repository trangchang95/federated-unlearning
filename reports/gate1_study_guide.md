# Gate 1 Study Guide for a Beginning AI Student

This guide prepares you for the questions in
[`gate1_self_check.md`](gate1_self_check.md). It teaches the concepts, but it
does not replace your own explanation. Read one section, close the guide, and
then explain the idea aloud or in writing as if you were teaching a friend.
Do not memorize sentences. Gate 1 is passed when the ideas make sense to you.

## A small vocabulary map

| Term | Plain-language meaning |
|---|---|
| Example | One image together with its correct label |
| Batch | A small group of examples processed at the same time |
| Parameter or weight | A number the model is allowed to change while learning |
| Logit | One raw class score produced by the model |
| Loss | A number describing how unsuitable the current predictions are |
| Gradient | Information about how each parameter should change to affect the loss |
| Epoch | One complete pass through the training data |
| Checkpoint | A saved copy of the model parameters at one point in training |
| Metric | A measurement used to evaluate behavior, such as accuracy or macro F1 |

## 1. What happens during one training batch?

The Week 2 code expresses the complete learning cycle:

```text
images + correct labels
        |
        v
1. Forward pass: model(images) -> ten class scores per image
        |
        v
2. Loss: compare the scores with the correct labels
        |
        v
3. Backward pass: calculate a gradient for every trainable parameter
        |
        v
4. Optimizer step: use the gradients to change the parameters
        |
        v
the next batch sees a slightly updated model
```

The corresponding lines are:

```python
optimizer.zero_grad()
logits = model(images)
loss = loss_function(logits, labels)
loss.backward()
optimizer.step()
```

`zero_grad()` clears gradients left by the previous batch. The forward pass
uses the current parameters but does not change them. The backward pass
computes gradients but also does not change the parameters. The optimizer
step is the operation that actually updates them. The learning rate limits
the size of that update.

### Tiny mental example

Suppose an image is the digit 7. The model initially gives its largest score
to class 1. The loss is large because the scores disagree with the true label
7. Backpropagation finds which parameter changes would reduce that error.
The optimizer makes a small change. One batch will not teach the entire task;
thousands of repeated updates gradually improve the model.

### Explain it yourself

Use the words **input**, **scores**, **correct label**, **loss**, **gradient**,
and **weight update**. Also state which step actually changes the weights.

Common mistake: saying that backpropagation itself updates the weights.
Backpropagation calculates the gradients; the optimizer uses them to update
the weights.

## 2. Why keep training, validation, and test data separate?

The three splits answer different questions:

| Split | Question it answers | May it affect model selection? |
|---|---|---|
| Training | What patterns should the model learn? | Yes; it changes the weights |
| Validation | Which epoch or setting should be selected? | Yes; it selects among candidates |
| Test | How well does the finished choice work on untouched examples? | No |

The experiments save the checkpoint with the highest validation accuracy and
evaluate that selected checkpoint on the test set. If test accuracy were used
to choose the best epoch, information from the test set would influence the
choice. The test set would no longer be a clean final examination, and the
reported result would tend to be too optimistic.

An analogy is useful: training data is practice material, validation data is
a mock exam used to adjust study strategy, and test data is the final exam.
Repeatedly looking at the final exam while choosing a strategy makes it no
longer an independent final exam.

### Explain it yourself

Give each split one job, then explain what becomes biased if the test split is
used to choose the epoch.

Common mistake: saying the validation set trains the weights. It is evaluated
after each epoch, but the optimizer never updates weights from validation
examples.

## 3. What does cross-entropy loss show beyond accuracy?

Accuracy uses only the class with the largest predicted score. It treats both
of these correct predictions equally:

| Model output for the true class | Predicted class | Accuracy contribution |
|---:|---|---:|
| 51% probability | Correct | 1 correct answer |
| 95% probability | Correct | 1 correct answer |

Cross-entropy also responds to confidence. For one example, its contribution
is the negative logarithm of the probability assigned to the correct class.
You do not need to calculate logarithms for Gate 1. The useful idea is:

- more probability on the correct class usually lowers loss;
- high confidence in a wrong class produces a large penalty;
- therefore loss can improve even when no example crosses the decision
  boundary and accuracy stays unchanged.

For example, suppose ten predictions keep the same winning classes, so seven
remain correct and accuracy stays at 70%. If the model assigns more
probability to the seven correct labels and less extreme confidence to its
three mistakes, its loss can still decrease.

In the Week 2 MLP run, validation loss fell from 0.2561 to 0.1205 while
validation accuracy rose from 92.64% to 96.43%. These two measurements are
related, but they do not encode identical information.

### Explain it yourself

Contrast **winning class only** with **the full confidence distribution**, and
give a same-accuracy/different-confidence example.

Common mistake: saying lower loss always guarantees higher accuracy on the
next measurement. It does not; accuracy changes only when a winning class
changes from wrong to correct or vice versa.

## 4. What are a convolution filter, feature map, and pooling?

An MLP first flattens a 28 x 28 image into 784 numbers. A CNN keeps the rows
and columns so it can process local neighborhoods.

```text
small learned filter        whole image
      3 x 3              slide across locations
         \                       /
          +---- local responses-+
                     |
                     v
                one feature map
                     |
                     v
             pooling reduces width and height
```

- A **filter** is a small grid of learned weights. It is reused at every
  image location to respond to a local pattern.
- A **feature map** contains that filter's response at every location. Bright
  or strong areas mean the filter responded strongly there.
- **Max pooling** keeps the strongest response in each small region, reducing
  spatial size and computation while retaining prominent evidence.

The filters are not manually programmed as “digit 7 detector” or “cat
detector.” Training learns their values from gradients. Early filters often
respond to simple local properties such as edges, strokes, color changes, or
textures. Later layers can combine those responses into more useful patterns.

For the MNIST model, the shape progression is approximately:

```text
1 x 28 x 28 input
  -> 32 x 28 x 28 after the first convolution
  -> 32 x 14 x 14 after max pooling
  -> 64 x 14 x 14 after the second convolution
  -> 64 x 7 x 7 after max pooling
  -> 64 x 4 x 4 after adaptive pooling
  -> ten final class scores
```

### Explain it yourself

Describe what moves, what is learned, what becomes a map, and what pooling
reduces. Point to the saved feature-map figure in the Month 1 report.

Common mistake: saying a feature map is just another copy of the input image.
It represents where one learned filter responded strongly.

## 5. What does the MNIST CNN-versus-MLP result support?

The measured evidence is:

| Model | Test accuracy | Trainable parameters |
|---|---:|---:|
| 128-unit MLP | 96.80% | 101,770 |
| Small CNN | 98.60% | 151,306 |

The CNN is 1.80 percentage points higher in these saved configurations. This
is consistent with the explanation that preserving nearby-pixel structure is
helpful for MNIST. It does **not** prove that every CNN is better than every
MLP, or that convolution alone caused all of the difference. The models also
have different parameter counts and architectures, and only one fixed seed
was measured.

Good research language separates observation and interpretation:

- **Observation:** the saved CNN run achieved 98.60%, compared with 96.80%
  for the saved MLP run.
- **Supported interpretation:** local image processing is a plausible reason
  and is consistent with the experiment.
- **Unsupported universal claim:** CNNs always improve every image task by
  1.80 points.

### Explain it yourself

State the two numbers, the limited interpretation, and at least one reason not
to claim a universal law.

## 6. Why is CIFAR-10 accuracy lower than MNIST accuracy?

The numbers come from different tasks:

| MNIST | CIFAR-10 |
|---|---|
| Grayscale digits | Color objects |
| Mostly centered | Varying positions and viewpoints |
| Plain background | Complex natural backgrounds |
| One channel | Three color channels |
| Classes differ by strokes | Several animal classes share shapes and textures |

The same basic CNN reaches 98.60% on MNIST and 71.38% on CIFAR-10. The lower
CIFAR-10 result is not evidence that training failed. Both training and
validation curves improve through ten epochs. It establishes a realistic
baseline for a harder dataset using a deliberately small model.

### Explain it yourself

Compare the visual complexity of the datasets and use the improving curves as
evidence that “lower final accuracy” is not the same as “the model learned
nothing.”

Common mistake: comparing the percentages as though both models took the same
exam. Dataset difficulty changes what a percentage means.

## 7. Why can validation accuracy exceed training accuracy?

The CIFAR-10 experiment intentionally makes training harder:

- random cropping and horizontal flipping alter training images;
- dropout temporarily disables part of the classifier in `model.train()`;
- validation uses stable images and disables dropout in `model.eval()`.

Training accuracy is therefore measured under more difficult conditions than
validation accuracy. In epoch 10 it is 65.36%, while validation accuracy is
72.24%. This difference is explainable by the protocol and is not, by itself,
proof of data leakage.

Data leakage would require inappropriate information crossing a boundary—for
example, validation images appearing in training or test results guiding
checkpoint selection. The Week 3 runner creates disjoint index sets, applies
augmentation only through the training view, selects with validation, and
tests only after selection.

### Explain it yourself

Name both training-only mechanisms and distinguish an expected train/eval
mode difference from actual data leakage.

## 8. What makes a run reproducible?

Three artifacts solve different parts of the problem:

| Artifact | What it preserves | What it does not guarantee alone |
|---|---|---|
| Random seed 42 | Repeatable split, initial weights, batch order, and random augmentation sequence | It does not describe all settings |
| Saved JSON config | Dataset, model, optimizer, epochs, batch size, seed, output path, and required thesis fields | It does not contain learned weights |
| `best_model.pt` | The learned weights from the best validation epoch | It does not fully explain how those weights were produced |

Together with versioned source code and recorded library/environment details,
these make the run much easier to repeat and inspect. A fixed seed means
repeatable randomness, not universal truth. Later benchmark experiments must
use multiple recorded seeds to measure variation.

In this repository, each Month 1 config now also names the CPU device, exact
environment manifest, runner file, dataset version, and `month1-gate1` Git
revision. This prevents a machine with an available GPU or different library
versions from silently calling a different run “the same experiment.”

### Explain it yourself

Give one distinct job to the seed, config, and checkpoint. Do not describe
them as three copies of the same thing.

## 9. Why inspect class-level and later client-level behavior?

The CIFAR-10 CNN does not perform equally across classes:

| Hardest classes | Accuracy | Easier examples | Accuracy |
|---|---:|---|---:|
| Bird | 53.6% | Frog | 85.3% |
| Cat | 54.5% | Automobile | 87.6% |
| Dog | 55.1% |  |  |

The largest visible confusion is **233 dogs predicted as cats**. The reverse
confusion is also visible: **110 cats predicted as dogs**.

Imagine a later Non-IID client holding mostly cats and dogs while another
client holds mostly automobiles. A global accuracy average could look
acceptable even while the first client receives much worse predictions. This
is why the thesis rules treat non-target-client utility as a first-class
measurement and why later Non-IID evaluation must retain per-class and
per-client views.

### Explain it yourself

Name the hardest classes and largest confusion, then construct a two-client
example showing what global accuracy could hide.

## 10. Why does Gate 1 block FedAvg?

Federated Learning does not replace the centralized training loop. It repeats
that loop on several clients and adds communication plus model aggregation:

```text
centralized foundation
  training loop + loss + evaluation + reproducibility
                          |
                          v
federated additions
  client data partitions + local training + weighted aggregation + rounds
```

Without the foundation, a falling global accuracy could be caused by the
local optimizer, bad data splitting, incorrect evaluation, or aggregation.
It would be difficult to identify the real problem. Gate 1 makes sure there
is a known-working centralized reference and that the student can reason
about its behavior before adding multiple clients and communication rounds.

This is why “the code runs” is insufficient. A thesis student must be able to
defend what the code measures, why the result changed, and which conclusions
are supported.

### Explain it yourself

Write one paragraph that connects the existing training loop to the extra FL
components and explains how the gate reduces debugging and research risk.

## A safe way to complete the self-check

1. Study only one section at a time.
2. Close this guide.
3. Answer the corresponding question in two to five sentences.
4. Add one measured project fact when relevant.
5. Mark guesses with language such as “a plausible explanation is,” rather
   than presenting them as proven facts.
6. Review the answer against the common mistake, but rewrite it in your own
   language.

You do not need mathematical notation. You do need to show the causal order
of training, the different jobs of the data splits, the difference between
loss and accuracy, and the limits of what one experiment proves.
