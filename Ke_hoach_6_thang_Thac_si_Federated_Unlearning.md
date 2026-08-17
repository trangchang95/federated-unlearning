# Kế hoạch thực hiện luận văn Thạc sĩ ứng dụng
## Federated Unlearning cho môi trường Federated Learning không đồng nhất

**Thời lượng:** 6 tháng
**Đối tượng:** Học viên Thạc sĩ ứng dụng, chưa có nền tảng về Federated Learning (FL)
**Định hướng:** Xây dựng prototype + thực nghiệm có đối chứng + đề xuất một cải tiến có phạm vi vừa phải
**Mục tiêu cuối:** Có luận văn hoàn chỉnh và kết quả đủ tốt để phát triển thành bài báo nếu kết quả thực nghiệm có ý nghĩa.

---

# 1. Tổng quan đề tài

## 1.1. Tên đề tài đề xuất

### Tên tiếng Việt

**Xây dựng và đánh giá giải pháp Federated Unlearning cho môi trường Federated Learning không đồng nhất**

### Tên tiếng Anh

**Development and Evaluation of a Federated Unlearning Solution for Heterogeneous Federated Learning Environments**

Có thể điều chỉnh tên sau khi xác định contribution cuối cùng.

---

## 1.2. Bối cảnh

Federated Learning cho phép nhiều client cùng huấn luyện một mô hình mà không cần tập trung dữ liệu thô về server.

Mô hình cơ bản:

```text
Client 1 ─┐
Client 2 ─┤
Client 3 ─┼──> Federated Server ──> Global Model
Client 4 ─┤
Client 5 ─┘
```

Tuy nhiên, sau khi mô hình toàn cục đã được huấn luyện, một client có thể yêu cầu rút lui hoặc yêu cầu xóa ảnh hưởng của dữ liệu đã đóng góp.

Câu hỏi đặt ra:

> Làm thế nào loại bỏ ảnh hưởng của một client khỏi mô hình toàn cục mà không phải huấn luyện lại toàn bộ mô hình từ đầu?

Đây là bài toán **Federated Unlearning (FU)**.

---

# 2. Mục tiêu của luận văn

## 2.1. Mục tiêu tổng quát

Xây dựng và đánh giá một giải pháp Federated Unlearning có khả năng:

1. Xác định client cần được unlearn.
2. Loại bỏ ảnh hưởng của client mục tiêu khỏi global model.
3. Duy trì hiệu năng trên dữ liệu không thuộc client cần xóa.
4. Giảm chi phí tính toán/thời gian so với full retraining.
5. Đánh giá mức độ forgetting bằng các metric phù hợp.
6. Hoạt động trong môi trường dữ liệu không đồng nhất (Non-IID).

## 2.2. Mục tiêu kỹ thuật

Học viên cần hoàn thành:

- Centralized ML/DL baseline.
- FedAvg implementation.
- FedProx implementation/benchmark.
- IID và Non-IID FL.
- Full retraining baseline.
- Ít nhất một Federated Unlearning baseline.
- Proposed improvement.
- Evaluation framework.
- Prototype/demo.

---

# 3. Phạm vi phù hợp với Thạc sĩ ứng dụng

## 3.1. Nên làm

- PyTorch.
- MNIST.
- CIFAR-10.
- Có thể bổ sung FEMNIST/CIFAR-100 nếu đủ thời gian.
- FedAvg.
- FedProx.
- Một hoặc hai FU baselines.
- IID/Non-IID.
- Client-level unlearning.
- So sánh với full retraining.
- Đánh giá efficiency và forgetting.
- Prototype có thể chạy local hoặc nhiều process/container.

## 3.2. Không bắt buộc

- Theorem mới.
- Formal proof phức tạp.
- Cryptographic protocol.
- Blockchain.
- Differential privacy mới.
- Foundation models.
- UAV/IoT deployment thực tế ngay từ đầu.
- Đồng thời xử lý quá nhiều bài toán: Non-IID + Sequential + Privacy + Verification + Dynamic Client + UAV.

## 3.3. Hướng mở rộng nếu tiến độ tốt

- Sequential unlearning.
- Dynamic client participation.
- Membership Inference Attack.
- Client-verifiable unlearning.
- Edge/IoT/UAV case study.

---

# 4. Research Questions

## RQ1 — Forgetting effectiveness

**Can federated unlearning effectively remove the contribution of a target client?**

## RQ2 — Data heterogeneity

**How does Non-IID data distribution affect federated unlearning performance?**

## RQ3 — Utility preservation

**Can the model preserve comparable performance on non-target clients after unlearning?**

## RQ4 — Efficiency

**Can federated unlearning achieve comparable model utility to full retraining with substantially lower computational cost?**

## RQ5 — Verification

**How can the effectiveness of forgetting be evaluated or verified?**

RQ5 có thể được xem là phần mở rộng nếu scope 6 tháng bị giới hạn.

---

# 5. Research hypothesis

Có thể sử dụng các giả thuyết sau:

### H1

Federated Unlearning có thể đạt model utility gần với full retraining nhưng với chi phí thấp hơn.

### H2

Mức độ Non-IID càng cao thì chất lượng unlearning và model utility càng suy giảm.

### H3

Một phương pháp FU có cơ chế bảo toàn non-target knowledge sẽ tốt hơn phương pháp FU cơ bản trong môi trường Non-IID.

### H4

Có thể sử dụng khoảng cách với retrained model và/hoặc Membership Inference Attack để đánh giá mức độ forgetting.

---

# 6. Kiến trúc thực nghiệm tổng thể

```text
                    Dataset
                       |
              +--------+--------+
              |                 |
        Centralized          Partition
        Training             to Clients
              |                 |
              |          +------+------+------+
              |          |      |      |      |
              |         C1     C2     C3    ... Cn
              |          |      |      |      |
              |          +------+------+- ----+
              |                 |
              |              FedAvg
              |                 |
              |            Global Model
              |                 |
              |       Client k requests deletion
              |                 |
              |                 v
              |        Federated Unlearning
              |                 |
              |                 v
              |          Unlearned Model
              |                 |
              +--------+--------+
                       |
                 Evaluation
                       |
       +---------------+----------------+
       |               |                |
    Utility        Forgetting       Efficiency
```

---

# 7. Bộ paper chính

## 7.1. Foundation

### P1 — FedAvg

**McMahan et al., 2017**

> Communication-Efficient Learning of Deep Networks from Decentralized Data

Venue: AISTATS 2017.

Vai trò:

- Paper nền tảng của FedAvg.
- Hiểu client/server.
- Local training.
- Model averaging.
- Communication rounds.

---

### P2 — FL Survey

**Kairouz et al., 2019**

> Advances and Open Problems in Federated Learning

Vai trò:

- Tổng quan FL.
- Statistical heterogeneity.
- Systems heterogeneity.
- Privacy.
- Communication.
- Open problems.

Không cần đọc toàn bộ.

---

### P3 — FedProx

**Li et al., 2020**

> Federated Optimization in Heterogeneous Networks

Venue: MLSys 2020.

Vai trò:

- Hiểu heterogeneity.
- Non-IID.
- Client drift.
- Baseline FL thứ hai.

---

# 7.2. Machine Unlearning

### P4 — SISA

**Bourtoule et al., 2021**

> Machine Unlearning

Venue: IEEE Symposium on Security and Privacy.

Vai trò:

- Hiểu machine unlearning.
- Hiểu tại sao full retraining tốn kém.
- Làm quen với efficient unlearning.

---

# 7.3. Federated Unlearning

### P5 — FedEraser

**Liu et al., 2020**

> Federated Unlearning

Vai trò:

- Một trong những công trình nền tảng về FU.
- Khai thác historical model updates.
- Client-level unlearning.
- So sánh với retraining.

---

### P6 — Knowledge Distillation FU

**Wu, Zhu & Mitra, 2022**

> Federated Unlearning with Knowledge Distillation

Vai trò:

- FU + knowledge distillation.
- Baseline thực nghiệm.
- Forgetting và model recovery.

---

### P7 — FFMU

**Che et al., 2023**

> Fast Federated Machine Unlearning with Nonlinear Functional Theory

Venue: ICML 2023.

Vai trò:

- Efficient FU.
- Theoretical perspective.
- Certified guarantees.
- Hiểu hướng nghiên cứu hiện đại.

Không yêu cầu học viên hiểu toàn bộ phần toán.

---

### P8 — SIFU

**Fraboni et al., 2024**

> SIFU: Sequential Informed Federated Unlearning for Efficient and Provable Client Unlearning in Federated Optimization

Venue: AISTATS 2024.

Vai trò:

- Sequential client unlearning.
- Formal guarantees.
- Convex/non-convex optimization.
- Hiểu hướng mở rộng sau client-level FU cơ bản.

---

### P9 — FU Survey

**Romandini et al., 2025**

> Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics

Venue: IEEE Transactions on Neural Networks and Learning Systems.

Vai trò:

- Literature map.
- Taxonomy.
- Evaluation metrics.
- Research gaps.
- Là paper quan trọng nhất để xác định contribution.

Nên đọc sau khi học viên đã có kiến thức FL và FU cơ bản.

---

### P10 — Practical FU, 2026

**Towards practical federated unlearning: A knowledge distillation solution**

Venue: Alexandria Engineering Journal, 2026.

Vai trò:

- Practical FU.
- Non-IID.
- Sequential settings.
- Membership inference.
- Communication overhead.

Dùng để cập nhật literature ngay trước khi chốt contribution.

---

# 8. Cách đọc paper

Không yêu cầu học viên viết summary dài.

Mỗi paper phải trả lời 6 câu:

1. **Problem:** Paper giải quyết vấn đề gì?
2. **Gap:** Phương pháp trước có hạn chế gì?
3. **Idea:** Ý tưởng chính là gì?
4. **Assumption:** Paper giả định điều gì?
5. **Evaluation:** Dataset/metric nào?
6. **Limitation:** Điểm yếu là gì?

Tạo file:

```text
literature_matrix.xlsx
```

với các cột:

| Paper | Problem | Method | Assumption | Dataset | Metric | Result | Limitation | Possible Gap |
|---|---|---|---|---|---|---|---|---|

---

# 9. Lộ trình 6 tháng

---

## MONTH 1 — ML/DL Foundation

### Mục tiêu

Học viên có thể:

- Python.
- NumPy/Pandas.
- PyTorch.
- Neural network.
- CNN.
- Training/evaluation.
- Git.
- Experiment logging.

### Week 1 — Python + ML

Học:

- Python.
- NumPy.
- Pandas.
- Matplotlib.
- Git.
- train/validation/test.

Bài tập:

> MNIST classifier.

### Week 2 — Neural Networks

Học:

- forward propagation.
- loss.
- backpropagation.
- optimizer.
- learning rate.

### Week 3 — CNN

Học:

- convolution.
- pooling.
- feature maps.
- CNN.

Train CNN trên MNIST/CIFAR-10.

### Week 4 — Experimental methodology

Học:

- accuracy.
- precision.
- recall.
- F1.
- confusion matrix.
- random seed.
- checkpoint.
- logging.

Đọc:

**P4 — Bourtoule et al. 2021, SISA**

Chỉ đọc phần motivation và concept.

### Deliverable M1

- CNN trên MNIST.
- CNN trên CIFAR-10.
- Git repository.
- Experiment report 5–8 trang.
- Có baseline centralized training.

### Gate M1

Học viên phải giải thích được:

> Model được train thế nào, loss hoạt động ra sao, và tại sao accuracy thay đổi.

---

# MONTH 2 — Federated Learning Fundamentals

## Week 5 — FL concepts

Đọc:

**P1 — McMahan et al. 2017**

Học:

- client.
- server.
- local model.
- global model.
- communication round.
- local epochs.
- aggregation.

### Week 6 — FedAvg

Tự implement:

```text
for each round:
    select clients

    for each client:
        receive global model
        train locally
        send local model

    server:
        aggregate local models
        create global model
```

Công thức:

w_(t+1) = Σ_k (n_k/n) w_(t+1)^k

### Week 7 — FL framework

Sau khi tự implement mới sử dụng:

- Flower hoặc framework tương đương.

Không dùng framework như black box.

### Week 8 — First FL experiment

So sánh:

- Centralized.
- FedAvg.

Trên MNIST.

### Deliverable M2

Prototype:

```text
Server
 ├── Client 1
 ├── Client 2
 ├── Client 3
 ├── Client 4
 └── Client 5
```

Report:

> Centralized Learning vs Federated Learning.

### Gate M2

Học viên phải:

- tự giải thích FedAvg;
- tự thay đổi number of clients;
- tự thay đổi local epochs;
- tự giải thích convergence.

---

# MONTH 3 — Non-IID Federated Learning

## Week 9 — IID vs Non-IID

Đọc lại:

- P1 — FedAvg.
- P2 — Kairouz survey.

Thực nghiệm:

```text
IID:
C1 = mixed classes
C2 = mixed classes
C3 = mixed classes
```

vs.

```text
Non-IID:
C1 = mostly class A
C2 = mostly class B
C3 = mostly class C
```

## Week 10 — Dirichlet partition

Dùng:

Dirichlet(alpha)

Thử:

- α = 1.0
- α = 0.5
- α = 0.1

## Week 11 — FedProx

Đọc:

**P3 — Li et al. 2020**

Implement/experiment:

- FedAvg.
- FedProx.

## Week 12 — Benchmark

Tạo bảng:

| Setting | FedAvg Acc | FedProx Acc | Rounds | Time |
|---|---:|---:|---:|---:|
| IID | | | | |
| α=1.0 | | | | |
| α=0.5 | | | | |
| α=0.1 | | | | |

### Deliverable M3

Report:

> Experimental Study of Federated Learning under IID and Non-IID Data.

Khoảng 10–15 trang.

### Gate M3

Học viên phải giải thích:

- statistical heterogeneity;
- client drift;
- tại sao Non-IID ảnh hưởng FL;
- FedProx giải quyết vấn đề gì.

---

# MONTH 4 — Federated Unlearning

## Week 13 — Machine Unlearning

Đọc:

**P4 — SISA**

Học:

- data deletion.
- exact vs approximate unlearning.
- retraining.
- efficient unlearning.

Thực nghiệm:

```text
Train full dataset
        ↓
Delete subset
        ↓
Full retraining
```

Đo:

- accuracy;
- training time.

## Week 14 — Federated Unlearning

Đọc:

**P5 — FedEraser**

Hiểu:

```text
FL training
     ↓
Historical updates
     ↓
Client requests deletion
     ↓
FedEraser
     ↓
Unlearned model
```

## Week 15 — KD-based FU

Đọc:

**P6 — Wu et al.**

Implement/reproduce baseline nếu khả thi.

## Week 16 — Baseline reproduction

So sánh:

```text
Full Retraining
      vs
FedEraser
      vs
KD-based FU
```

### Deliverable M4

Report:

> Reproduction Study of Federated Unlearning Methods.

Phải có:

- implementation;
- configuration;
- results;
- comparison;
- discussion.

### Gate M4 — critical gate

Nếu chưa reproduce được ít nhất một FU baseline:

**Không được vội đề xuất thuật toán mới.**

Chuyển scope sang applied comparative study nếu cần.

---

# MONTH 5 — Research Contribution

## Week 17 — Advanced literature

Đọc:

**P7 — FFMU**

và:

**P8 — SIFU**

Mục tiêu:

- hiểu current research direction;
- hiểu efficiency;
- hiểu sequential unlearning;
- hiểu formal guarantee ở mức concept.

## Week 18 — FU survey

Đọc:

**P9 — Romandini et al. 2025**

Lập taxonomy:

```text
Federated Unlearning
│
├── Client-level
├── Data-level
├── Efficient
├── Sequential
├── Privacy-preserving
├── Verifiable
└── Non-IID-aware
```

## Week 19 — Identify research gap

Đối chiếu:

| Existing method | Non-IID | Sequential | Efficient | Verification | Limitation |
|---|---:|---:|---:|---:|---|
| FedEraser | | | | | |
| KD-FU | | | | | |
| FFMU | | | | | |
| SIFU | | | | | |
| 2026 method | | | | | |

Chỉ sau bảng này mới chốt proposed method.

## Week 20 — Proposed method

Ưu tiên:

### Option A — Efficient FU

hoặc:

### Option B — Non-IID-aware FU

**Khuyến nghị: Option B.**

Problem:

```text
Target client
     ↓
Forget
     +
Preserve non-target knowledge
     ↓
Unlearned model
```

Không yêu cầu algorithm quá phức tạp.

### Deliverable M5

- Proposed method.
- Architecture diagram.
- Algorithm/pseudocode.
- Preliminary experiment.
- Research gap statement.

---

# MONTH 6 — Final Evaluation + Thesis

## Week 21 — Final benchmark

Datasets:

- MNIST.
- CIFAR-10.

Optional:

- CIFAR-100/FEMNIST.

Settings:

- IID.
- Non-IID.
- Different number of clients.
- Different target clients.

## Week 22 — Metrics

### Utility

- Accuracy.
- F1.

### Forgetting

- Distance to retrained model.
- Target-data performance.
- Non-target performance.

### Privacy

Nếu đủ thời gian:

- Membership Inference Attack.
- MIA AUC/accuracy.

### Efficiency

- Training time.
- Unlearning time.
- Communication cost.
- Number of rounds.
- Storage cost.

## Week 23 — Ablation

Ví dụ:

```text
Baseline
Baseline + Component A
Baseline + Component B
Proposed A+B
```

Phân tích:

- component nào thực sự có tác dụng;
- khi Non-IID tăng thì sao;
- khi số client tăng thì sao;
- khi số client bị xóa tăng thì sao.

## Week 24 — Thesis + demo

Hoàn thiện:

- Thesis.
- Presentation.
- Demo.
- Source code.
- README.
- Configuration.
- Results.

---

# 10. Evaluation protocol

## 10.1. Gold standard

Model chuẩn để đánh giá forgetting:

```text
Original FL
     ↓
Global model M
```

và:

```text
Remove Client A
     ↓
Train FL again without A
     ↓
M_retrain^(-A)
```

Sau đó:

```text
M_FU
   vs
M_retrain^(-A)
```

Mục tiêu:

M_FU ≈ M_retrain^(-A)

nhưng:

Time_FU << Time_retrain

---

# 11. Metrics

## 11.1. Model utility

Accuracy và F1 trên:

- toàn bộ test set;
- non-target clients.

## 11.2. Forgetting effectiveness

So sánh:

M_FU với M_retrain^(-A)

Có thể dùng:

- parameter distance;
- prediction agreement;
- representation similarity.

## 11.3. Target forgetting

Đánh giá model trên dữ liệu của target client.

## 11.4. Non-target preservation

Đánh giá trên dữ liệu của các client còn lại.

Đây là metric rất quan trọng.

## 11.5. Efficiency

Tính:

Speedup = Time_retrain / Time_FU

và:

Communication ratio =
Communication_FU / Communication_retrain

## 11.6. Privacy

Nếu triển khai được:

- Membership Inference Attack.
- Attack accuracy.
- Attack AUC.

---

# 12. Experimental matrix

Tối thiểu:

| Dimension | Values |
|---|---|
| Dataset | MNIST, CIFAR-10 |
| FL | FedAvg, FedProx |
| Distribution | IID, Non-IID |
| Dirichlet α | 1.0, 0.5, 0.1 |
| Clients | 5, 10, 20 |
| Unlearning | 1 client |
| Target client | Several clients |
| Baseline | Retraining, FedEraser/KD-FU |
| Proposed | Proposed method |

Nếu đủ thời gian:

| Extension | Values |
|---|---|
| Deletion ratio | 10%, 20%, 30% |
| Sequential deletion | 2–3 clients |
| MIA | Yes |
| Dataset | CIFAR-100/FEMNIST |

---

# 13. Repository structure

```text
federated-unlearning/
│
├── data/
│
├── models/
│
├── clients/
│
├── server/
│
├── algorithms/
│   ├── fedavg.py
│   ├── fedprox.py
│   ├── federated_unlearning.py
│   └── proposed_method.py
│
├── experiments/
│   ├── iid/
│   ├── noniid/
│   ├── unlearning/
│   └── ablation/
│
├── evaluation/
│   ├── utility.py
│   ├── forgetting.py
│   ├── efficiency.py
│   └── privacy.py
│
├── configs/
│
├── results/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# 14. Quản lý thí nghiệm

Mỗi experiment phải lưu:

```text
dataset
number_of_clients
client_fraction
local_epochs
learning_rate
batch_size
alpha
random_seed
algorithm
target_client
number_of_rounds
accuracy
f1
unlearning_time
communication_cost
```

Không được chạy experiment thủ công rồi copy số liệu vào Excel mà không lưu configuration.

Khuyến nghị:

- YAML/JSON configuration.
- fixed random seeds.
- automatic result logging.
- version control bằng Git.

---

# 15. Milestone/Gate system

## Gate 1 — End of Month 1

**Must have:**

- PyTorch.
- CNN.
- Centralized baseline.

## Gate 2 — End of Month 2

**Must have:**

- FedAvg tự implement.
- Multi-client FL.
- Centralized vs FL.

## Gate 3 — End of Month 3

**Must have:**

- IID/Non-IID.
- FedAvg/FedProx.
- Benchmark.

## Gate 4 — End of Month 4

**Must have:**

- FU baseline.
- Full retraining.
- Initial forgetting/efficiency results.

## Gate 5 — End of Month 5

**Must have:**

- Research gap.
- Proposed improvement.
- Preliminary results.

## Gate 6 — End of Month 6

**Must have:**

- Final experiments.
- Thesis.
- Prototype.
- Reproducible code.

---

# 16. Tiêu chí đánh giá tiến độ

| Tiêu chí | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---:|---:|---:|---:|---:|---:|
| Theory | 80% | 70% | 50% | 40% | 30% | 20% |
| Coding | 50% | 70% | 80% | 80% | 80% | 60% |
| Experiment | 20% | 50% | 80% | 90% | 90% | 90% |
| Research | 10% | 10% | 20% | 40% | 80% | 80% |
| Writing | 20% | 20% | 30% | 40% | 60% | 100% |

---

# 17. Phân bổ thời gian hàng tuần

Khuyến nghị:

**15–20 giờ/tuần**

| Hoạt động | Giờ/tuần |
|---|---:|
| Reading | 3–4 |
| Theory | 2–3 |
| Coding | 6–8 |
| Experiment | 3–4 |
| Documentation | 1–2 |
| Supervisor meeting | 1 |

Tỷ trọng theo thời gian:

```text
Month 1
Learning >>> Coding

Month 2
Learning >> Coding

Month 3
Coding ≈ Experiment

Month 4
Experiment >>> Reading

Month 5
Research ≈ Experiment

Month 6
Experiment + Thesis
```

---

# 18. Kết quả đầu ra kỳ vọng

Cuối 6 tháng, học viên phải có:

## Knowledge

- ML/DL.
- FL.
- FedAvg.
- FedProx.
- Non-IID.
- Machine Unlearning.
- Federated Unlearning.

## Implementation

- Centralized training.
- FedAvg.
- FedProx.
- FU baseline.
- Proposed method.

## Research

- Literature review.
- Research gap.
- Research question.
- Proposed solution.
- Experimental validation.

## System

- FL server.
- Multiple clients.
- Training.
- Unlearning request.
- Unlearning execution.
- Evaluation dashboard/report.

## Thesis

Cấu trúc:

```text
Chapter 1 — Introduction
Chapter 2 — Background
Chapter 3 — Literature Review
Chapter 4 — Proposed Method
Chapter 5 — Experimental Evaluation
Chapter 6 — Discussion
Chapter 7 — Conclusion
```

---

# 19. Rủi ro và phương án dự phòng

## Risk 1 — Học viên chưa hiểu FL

**Action:** Quay lại FedAvg tự implement, chưa dùng framework.

## Risk 2 — Không reproduce được FU

**Action:** Chọn một baseline đơn giản hơn; không phát triển algorithm mới.

## Risk 3 — Proposed method không tốt hơn baseline

Không được che giấu kết quả.

Phân tích:

- tại sao;
- limitation;
- setting nào phương pháp tốt/xấu.

Một luận văn ứng dụng tốt không nhất thiết phải thắng mọi baseline.

## Risk 4 — Scope quá rộng

Ưu tiên:

**FedAvg → Non-IID → Client-level FU → Efficiency/Forgetting**

Bỏ:

- UAV;
- blockchain;
- cryptography;
- advanced privacy;
- foundation models.

## Risk 5 — Không đủ compute

Bắt đầu:

- MNIST.
- CIFAR-10.
- 5–10 clients.

Không cần train model lớn.

---

# 20. Tiêu chí thành công của luận văn

Luận văn được xem là đạt mục tiêu nếu chứng minh được:

### Requirement 1

Client-level unlearning thực hiện được.

### Requirement 2

Model sau FU vẫn giữ được utility chấp nhận được.

### Requirement 3

Kết quả FU có hành vi gần với retraining without target client.

### Requirement 4

FU có chi phí thấp hơn full retraining.

### Requirement 5

Đánh giá được ảnh hưởng của Non-IID.

### Requirement 6

Có một cải tiến có cơ sở từ limitation của baseline.

Không bắt buộc phải chứng minh một theorem mới.

---

# 21. Định hướng contribution phù hợp nhất

Sau khi học viên hoàn thành M4, ưu tiên xem xét:

## Preferred direction

### Non-IID-Aware Federated Unlearning

Problem:

> Existing FU methods may degrade when client data are highly heterogeneous.

Proposed objective:

```text
                    Target Client
                         |
                         v
                    FORGET
                         |
                         v
                +----------------+
                | Proposed FU    |
                +----------------+
                         |
              +----------+----------+
              |                     |
              v                     v
       Remove target          Preserve others
        contribution            knowledge
```

Mục tiêu:

Accuracy_non-target_after_FU

≈

Accuracy_non-target_after_retraining

đồng thời:

Time_FU << Time_retraining.

---

# 22. Các hướng mở rộng sau luận văn

Nếu kết quả tốt, có thể phát triển research line:

```text
Federated Unlearning
        |
        +-- Efficient FU
        |
        +-- Non-IID FU
        |
        +-- Sequential FU
        |
        +-- Dynamic Client FU
        |
        +-- Verifiable FU
        |
        +-- Privacy-Preserving FU
        |
        +-- Edge/UAV FU
```

Đối với Master Applied, chỉ cần hoàn thành nhánh:

**Non-IID + Client-level FU + Efficiency/Forgetting**

là đã đủ scope.

---

# 23. Tóm tắt roadmap một trang

```text
MONTH 1
ML/DL
  └── PyTorch + CNN + SISA concept
        ↓
MONTH 2
Federated Learning
  └── FedAvg + centralized vs FL
        ↓
MONTH 3
Heterogeneous FL
  └── IID/Non-IID + FedProx
        ↓
MONTH 4
Federated Unlearning
  └── Retraining + FedEraser + KD-FU
        ↓
MONTH 5
Research
  └── FFMU + SIFU + TNNLS Survey
  └── Research gap
  └── Proposed Non-IID-aware FU
        ↓
MONTH 6
Evaluation
  └── Benchmark
  └── Ablation
  └── Forgetting
  └── Efficiency
  └── Thesis + Demo
```

---

# 24. Danh sách tài liệu chính cần tải và đọc

1. McMahan et al. (2017), **Communication-Efficient Learning of Deep Networks from Decentralized Data**, AISTATS.
2. Kairouz et al. (2019), **Advances and Open Problems in Federated Learning**.
3. Li et al. (2020), **Federated Optimization in Heterogeneous Networks**, MLSys.
4. Bourtoule et al. (2021), **Machine Unlearning**, IEEE S&P.
5. Liu et al. (2020), **Federated Unlearning / FedEraser**.
6. Wu, Zhu & Mitra (2022), **Federated Unlearning with Knowledge Distillation**.
7. Che et al. (2023), **Fast Federated Machine Unlearning with Nonlinear Functional Theory**, ICML.
8. Fraboni et al. (2024), **SIFU**, AISTATS.
9. Romandini et al. (2025), **Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics**, IEEE TNNLS.
10. **Towards practical federated unlearning: A knowledge distillation solution**, 2026.

---

# 25. Kết luận

Với học viên **chưa có FL background**, không nên bắt đầu bằng câu hỏi:

> "Làm Federated Unlearning bằng phương pháp gì?"

Mà phải đi theo chuỗi:

**Understand ML → Understand FL → Implement FL → Understand Non-IID → Understand Unlearning → Reproduce FU → Identify limitation → Propose improvement → Evaluate.**

Lộ trình này giảm rủi ro đáng kể và phù hợp với bản chất **Thạc sĩ ứng dụng**.

**Mục tiêu hợp lý nhất:** đến cuối tháng 4 học viên đã có một FU baseline chạy ổn định; tháng 5 mới xác định novelty; tháng 6 tập trung chứng minh bằng thực nghiệm và hoàn thiện luận văn.
