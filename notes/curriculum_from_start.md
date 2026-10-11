# Curriculum: học lại từ đầu dự án (backfill)

Mục đích: hiểu đến mức bảo vệ được (L2/L3) mọi thứ đã làm từ Month 1 đến nay, trước khi học tiếp phần unlearning.
- Mỗi buổi ~45 phút với `claude --agent tutor`. Tổng 16 buổi ≈ 12 h.
- Đề xuất xếp lịch: W1–W4 làm 4–5 buổi/tuần, trong lúc các lane khác build Week 13–14 (chưa chạy canonical được vì phải chờ mức L2).
- Câu mở đầu: `Backfill: buổi <mã>.`

Tutor đánh dấu `[x]` khi đạt mức yêu cầu. Kết quả ghi vào `notes/understanding_map.md` và `notes/learning_log.md`.

## M0 — Bức tranh tổng
- [ ] **0.1 Đề tài trong 1 trang** — `Ke_hoach_6_thang_...md` mục 2 (mục tiêu), `Ke_hoach_v4_FU.md` mục 1–3, `README.md`
  - Bài tập: vẽ lại pipeline data → partition → FL → yêu cầu xóa → FU → evaluation; nói RQ1–RQ5 bằng lời của mình.
  - Mức yêu cầu: L2.

## M1 — Nền ML (Gate 1)
- [ ] **1.1 Từ logistic regression đến MLP**
  - Code: `experiments/month1_week1_mnist_baseline.py`, `experiments/month1_week2_mnist_mlp.py`, `models/simple_mlp.py`
  - Câu hỏi còn mở từ trước: gradient của softmax + cross-entropy rút gọn thành gì, vì sao?
  - Bài tập: tính tay 1 bước gradient cho 3 class; bug-hunt sai axis softmax.
  - Mức yêu cầu: L2.
- [ ] **1.2 CNN và overfitting**
  - Code: `models/small_cnn.py`, `experiments/month1_week3_cnn.py`
  - Kết quả: `results/month1_week3_mnist_cnn`, `results/month1_week3_cifar10_cnn`
  - Bài tập: đọc log train/val, chỉ ra điểm bắt đầu overfit; predict-then-run khi đổi learning rate.
  - Mức yêu cầu: L2.
- [ ] **1.3 Phương pháp thực nghiệm của repo**
  - Code: `experiments/experiment_utils.py`, một file `reports/verify_*.py`, `tests/`, `configs/*.json`
  - Tài liệu: `reports/gate1_self_check.md`, `reports/gate1_study_guide.md`
  - Bài tập: lấy một con số trong report Month 1, truy ngược về JSON → config → commit (number audit).
  - Mức yêu cầu: L2.

## M2 — Federated Learning (Gate 2)
- [ ] **2.1 Khái niệm FL và paper FedAvg**
  - Paper: `articles/1.*`, `literature/fedavg_2017_six_questions.md`
  - Tài liệu: `reports/month2_week5_fl_concepts.md`
  - Bài tập: đọc 3 vòng (xem `docs/huong_dan_MCP_doc_paper.md`).
  - Mức yêu cầu: L2.
- [ ] **2.2 Code FedAvg** ⭐
  - Code: `algorithms/fedavg.py`, `server/fedavg_server.py`, `clients/federated_client.py`, `clients/iid_partition.py`, `tests/test_fedavg.py`
  - Bài tập: map công thức w = Σ(n_k/n)·w_k vào từng dòng code; bug-hunt (aliasing global model, sai trọng số n_k/n); modify (đổi số local epochs, dự đoán kết quả trước khi chạy).
  - Mức yêu cầu: **L3**.
- [ ] **2.3 Centralized vs FedAvg; các biến thể E=2 và K=10; communication bytes**
  - Code: `experiments/iid/month2_week8_mnist_fedavg.py`
  - Tài liệu: các file `reports/month2_week8_*.md`, `reports/month2_week7_flower_compatibility.md` (lướt)
  - Bài tập: giải thích vì sao FedAvg hội tụ chậm hơn centralized; tính tay số bytes truyền trong 1 round.
  - Mức yêu cầu: L2.

## M3 — Non-IID và FedProx (Gate 3)
- [ ] **3.1 IID vs Non-IID chia theo shard**
  - Code: `clients/noniid_partition.py`, `experiments/noniid/month3_week9_mnist_noniid_fedavg.py`
  - Tài liệu: `reports/month3_week9_iid_vs_noniid.md`
  - Mức yêu cầu: L2.
- [ ] **3.2 Dirichlet α và client drift** ⭐
  - Code: `clients/dirichlet_partition.py`, `tests/test_dirichlet_partition.py`
  - Tài liệu: `reports/month3_week10_dirichlet_benchmark.md`
  - Bài tập: predict-then-run histogram class per client ở α=0.1 so với α=1.0; tìm client gần rỗng; giải thích client drift bằng sơ đồ.
  - Mức yêu cầu: **L3** (partition quyết định nhóm target-dominant class trong luận văn).
- [ ] **3.3 FedProx** ⭐
  - Paper: `articles/3.*`, `literature/fedprox_2020_six_questions.md`
  - Code: `clients/fedprox_client.py`, `server/fedprox_server.py`, `tests/test_fedprox.py`
  - Lưu ý: `algorithms/fedprox.py` chỉ là stub.
  - Bài tập: map số hạng proximal (μ/2)‖w − w_t‖² vào code; modify μ = 0 và kiểm tra có ra đúng FedAvg không.
  - Mức yêu cầu: **L3**.
- [ ] **3.4 Benchmark và cách đọc kết quả**
  - Tài liệu: `reports/month3_week11_fedprox.md`, `reports/month3_week12_benchmark.md`
  - Bài tập: number audit một ô trong bảng; trả lời câu "μ có bị chọn dựa trên test set không?"; rehearsal câu hỏi Gate 3, dựa trên câu trả lời của chính bạn trong `reports/gate3_self_check.md`.
  - Mức yêu cầu: L2.

## M4 — Machine Unlearning và hạ tầng đo
- [ ] **4.1 Machine unlearning và SISA**
  - Paper: `articles/4.*`, `literature/sisa_2021_six_questions.md`
  - Nội dung: exact vs approximate unlearning; vì sao retrain là gold standard; vì sao retrain tốn kém.
  - Mức yêu cầu: L2.
- [ ] **4.2 Hạ tầng đo lường**
  - Code: `evaluation/three_way.py`, `evaluation/efficiency.py`, `evaluation/utility.py`, cùng các file test tương ứng
  - Bài tập: giải thích 3 split (global / target / non-target), phase timer, cách tính storage bytes.
  - Mức yêu cầu: L2.
- [ ] **4.3 Week 13: retrain tập trung vs gold standard federated**
  - Code: `experiments/unlearning/month4_week13_mnist_full_retraining.py`, file config tương ứng
  - Bài tập: tự giải thích vì sao bản retrain tập trung này **chưa phải** M_retrain^(-A); noise floor là gì; vì sao không so parameter distance giữa các model khác init.
  - Mức yêu cầu: L2.

## M5 — Federated Unlearning: tổng quan tài liệu
- [ ] **5.1 FedEraser** ⭐
  - Paper: `articles/5.*`, `literature/federaser_2021_six_questions.md`
  - Nội dung: lưu historical updates, calibration, chi phí storage.
  - Mức yêu cầu: L2 (lên L3 khi đã có code).
- [ ] **5.2 KD-FU, FFMU, SIFU, survey: chỉ ở mức khái niệm**
  - Tài liệu: các file six_questions tương ứng, `literature/gap_table_draft.md`
  - Bài tập: mỗi phương pháp nói được 1 câu "ý tưởng" và 1 câu "điểm yếu"; giải thích vì sao luận văn không chọn chúng làm baseline.
  - Mức yêu cầu: L2.

## M6 — Định vị luận văn
- [ ] **6.1 Mục tiêu, contribution và lý do chọn thuật toán**
  - Tài liệu: v4 mục 1–4 và 8; bản gốc mục 2, 20–21
  - Bài tập: rehearsal 2 phút "đề tài và 6 mục tiêu là gì; vì sao chọn PGA + FedEraser thay vì KD-FU; proposed method bảo toàn non-target bằng cách nào"; trả lời câu "code này ai viết?".
  - Mức yêu cầu: L2.

⭐ = mục sẽ được hội đồng hỏi sâu; ưu tiên làm trước nếu thiếu giờ.
