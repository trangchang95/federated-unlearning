# Learning log

## Syllabus checklist
- [ ] FL: round, local epochs, FedAvg n_k/n
- [ ] Non-IID, Dirichlet α, client drift
- [ ] FedProx μ
- [ ] MU: exact vs approximate; retrain as gold standard and its cost; SISA idea
- [ ] FU: client-level; M_retrain^(-A)
- [ ] PGA: gradient ascent, L2-ball projection, recovery rounds
- [ ] FedEraser: stored updates, calibration, storage cost
- [ ] Evaluation: 3 splits, agreement, noise floor, shared vs target-dominant classes
- [ ] Why parameter distance across different inits is meaningless
- [ ] Backdoor probe; MIA (concept)
- [ ] Speedup, cost-to-recover; seeds and mean ± std

## Sessions
<!-- tutor appends: date · topic · levels changed · misconceptions · next step -->

- **2026-10-11 · Backfill 0.1 — Đề tài trong 1 trang.** Lecture + 10 MCQ (7/9 câu hợp lệ đúng; câu 10 bị thu hồi vì tutor giảng theo v3 đã bị thay thế).
  - Levels: 0.1, noise floor, 3-split, over-unlearning → **L1**. Chưa đạt L2 (explain-back đã giao, chưa trả lời).
  - Misconception chính: **neo mốc so sánh vào M_original** thay vì M_retrain^(-A) (gây sai cả Check 4c và 7b). Phụ: utility bị gán cho target split; phát biểu "khớp" theo kiểu "nằm trong khoảng" thay vì Δ ≤ NF.
  - Tutor finding (không phải bug code): `Ke_hoach_v3_FU_AI_tutor.md` vẫn nằm ở repo root với tên đề tài + contribution khác v4 ("benchmark-first"). Nguy cơ trình bày sai contribution trước hội đồng → đề nghị main session đánh dấu OBSOLETE hoặc archive. Tutor không được sửa file ngoài `notes/`.
  - Next: thu explain-back 3 câu (contribution v4 / lý do chọn PGA + FedEraser / nguyên tắc so sánh 1 câu) → nếu đạt thì tick 0.1 và chuyển sang buổi 1.1 (logistic → MLP).
  - Explain-back lần 1 (2026-10-11): câu 1 Partial (thiếu "proposed method là một thuật toán" + "Non-IID-aware" + vai trò evaluation framework = mục tiêu 5, không phải contribution); câu 2 bỏ trống → tutor đã giảng cơ chế FedEraser (stored update + calibration, trả storage) vs PGA (ascent trong L2 ball quanh w_ref + recovery, không storage) và 2 lý do KD-FU optional; câu 3 Partial (thiếu noise floor, target split shared/dominant, agreement/KL). Giao lại câu 1 + 3.
