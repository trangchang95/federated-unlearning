# Kế hoạch luận văn — v4 (giữ mục tiêu gốc, cập nhật thiết kế nghiên cứu)

## Xây dựng và đánh giá giải pháp Federated Unlearning cho môi trường Federated Learning không đồng nhất
*Development and Evaluation of a Federated Unlearning Solution for Heterogeneous Federated Learning Environments*

| | |
|---|---|
| Nền tảng | `Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md` (bản gốc). **Giữ nguyên:** tên đề tài, mục tiêu tổng quát và kỹ thuật (mục 2), RQ1–RQ5, Requirement 1–6, hướng contribution Non-IID-aware FU (mục 21), cấu trúc thesis |
| Thay đổi | Chỉ thay đổi thiết kế nghiên cứu: cách đo, baseline, hướng thuật toán đề xuất, lịch theo trạng thái thực tế |
| Ngân sách | **15–20 h/tuần** × 26 tuần (12/10/2026 → ~11/04/2027) ≈ 390–520 h |
| Máy | MacBook Air M5 16 GB (MPS); Colab là phương án dự phòng cho CIFAR-10 |
| Trạng thái (11/10/2026) | Gate 1–2 đạt. Gate 3: đã có câu trả lời (sửa sau review), chờ tutor xác nhận. Week 13 (retrain tập trung) đã chuẩn bị, chưa chạy. Đã có six-questions cho P1–P9 và `gap_table_draft.md` |
| Cách làm | AI build (code, run, nháp) → tutor dạy lại → học viên chứng minh hiểu ở mức L2/L3 (xem `CLAUDE.md` §7) |
| Ngân sách AI | **Claude Pro**: lane chạy tuần tự; tutor dùng Sonnet (Opus chỉ cho core logic và rehearsal). **G08–G09 có thể trễ**; nếu trễ, cắt theo thứ tự: extensions → client-count ablation → FedProx slice. Không cắt core grid, walkthrough hay rehearsal |
| Thay thế | v2 và v3 (đã bỏ; v3 đã đổi tên và mục tiêu đề tài, không còn áp dụng) |

---

## 1. Giữ nguyên từ bản gốc

**Mục tiêu tổng quát** (bản gốc 2.1). Giải pháp FU có khả năng:
1. Xác định client cần unlearn.
2. Loại bỏ ảnh hưởng của client đó khỏi global model.
3. Duy trì hiệu năng trên dữ liệu không thuộc client cần xóa.
4. Giảm chi phí so với full retraining.
5. Đánh giá mức độ forgetting bằng metric phù hợp.
6. Hoạt động trong môi trường Non-IID.

**Mục tiêu kỹ thuật** (bản gốc 2.2): centralized baseline · FedAvg · FedProx · IID/Non-IID · full retraining · ≥ 1 FU baseline · proposed improvement · evaluation framework · prototype/demo.

**Requirement 1–6** (bản gốc mục 20) là tiêu chí nghiệm thu luận văn. Mục 8 bên dưới chỉ rõ bằng chứng cho từng requirement.

**Mục tiêu cuối** (bản gốc): luận văn hoàn chỉnh, kết quả đủ tốt để phát triển thành bài báo nếu thực nghiệm có ý nghĩa.

**Ánh xạ mục tiêu → cách thực hiện**

| Mục tiêu | Thực hiện bằng |
|---|---|
| 1. Xác định client | Demo prototype: server nhận yêu cầu xóa theo client ID, tra cứu dữ liệu và update đã lưu của client đó, rồi chạy FU. Trong thí nghiệm, target client được chọn theo quy tắc T1/T2/T3 (mục 6) |
| 2. Loại bỏ ảnh hưởng | FedEraser, PGA, (KD-FU), **proposed method** |
| 3. Duy trì hiệu năng | Utility trên non-target clients và global test, so với M_retrain |
| 4. Giảm chi phí | Speedup, bytes, storage, cost-to-recover |
| 5. Đánh giá forgetting | Evaluation framework: noise floor, agreement/KL, gap theo nhóm class, backdoor probe, MIA |
| 6. Non-IID | IID và Dirichlet α ∈ {1.0, 0.5, 0.1}; slice FedProx |

**Contribution** (bản gốc mục 21): Non-IID-Aware Federated Unlearning — quên target client, đồng thời giữ kiến thức non-target:
- Accuracy_non-target sau FU ≈ sau retrain;
- Time_FU << Time_retrain.

---

## 2. Thay đổi về nghiên cứu và thuật toán (và lý do)

| # | Bản gốc | v4 | Lý do |
|---|---|---|---|
| 1 | Baseline: FedEraser và/hoặc KD-FU | **Retrain + FedEraser + PGA** bắt buộc; **KD-FU** chạy nếu reproduce được | PGA (Halimi et al., arXiv 2207.05521) không cần server lưu update, đơn giản, được dùng làm baseline trong các nghiên cứu thực nghiệm FU. Thêm PGA giúp proposed method có ít nhất 2 baseline khác cơ chế (lưu lịch sử vs gradient ascent). KD-FU cần dữ liệu không nhãn ở server và mới chỉ là preprint, nên để optional |
| 2 | "Gần retrain" đo bằng parameter distance, agreement, target acc | Thêm **noise floor**: khoảng cách giữa 2 retrain khác seed. M_FU chỉ được coi là "≈ M_retrain" khi gap không vượt noise floor | Ở α thấp, M_retrain cũng có acc thấp trên target data. Nếu chỉ thấy acc giảm thì không chứng minh được là đã quên |
| 3 | Target / non-target performance | Tách target data thành **shared-class** và **target-dominant class** (target giữ ≥ 50% mẫu của class) | Đây là cơ chế Non-IID làm FU khó hơn. Đồng thời là chỗ proposed method phải chứng minh hiệu quả |
| 4 | Parameter distance | Chỉ dùng khi các model **cùng init**. Ưu tiên prediction agreement / KL output | So khoảng cách tham số giữa các init khác nhau thì không có ý nghĩa |
| 5 | Speedup = thời gian | Thời gian + bytes truyền + storage (FedEraser) + recovery rounds; thêm **cost-to-recover** (chi phí đến khi đạt utility bằng retrain) | Cách tính công bằng giữa các phương pháp có/không lưu lịch sử |
| 6 | Forgetting chỉ qua accuracy, MIA optional | Thêm **backdoor probe**. MIA giữ như bản gốc (mục 11.6), hiệu chỉnh theo M_retrain | Backdoor là cách đánh giá forgetting phổ biến trong FU (KD-FU và nhiều nghiên cứu thực nghiệm dùng) |
| 7 | Gap analysis ở Week 19, dựa trên P1–P10 | Giữ mốc gap analysis. Bổ sung các công trình 2025–2026 về Non-IID FU trước khi chốt method | Hướng Option B đã có nhiều công trình mới (mục 6). Phải khác biệt với chúng |
| 8 | P10 "Towards practical FU…" (AEJ 2026) | Chưa tìm thấy nguồn → **UNVERIFIED**, không cite cho đến khi lit-scout verify được | Quy tắc citation |
| 9 | Ma trận: tích đầy đủ các chiều | Giữ **mọi chiều**, nhưng tổ chức thành core grid + slice + ablation (mục 7) | Tích đầy đủ lên tới hàng nghìn run, không khả thi trên MacBook Air |

---

## 3. Research Questions — giữ câu hỏi gốc, thêm cách trả lời

| RQ (bản gốc) | Cách đo trong v4 | Bằng chứng |
|---|---|---|
| RQ1 — FU có loại bỏ được đóng góp của target client không? | Gap (acc, agreement) giữa M_FU và M_retrain^(-A) trên target split, so với noise floor; backdoor ASR so với M_retrain | Bảng gap per-split; bảng ASR |
| RQ2 — Non-IID ảnh hưởng FU thế nào? | Gap theo α ∈ {IID, 1.0, 0.5, 0.1}, tách shared-class / target-dominant | Biểu đồ gap theo α |
| RQ3 — Có giữ được performance trên non-target không? | Acc/F1 trên non-target clients và global test so với M_retrain (trong phạm vi noise floor) | Bảng utility |
| RQ4 — Có đạt utility gần retrain với chi phí thấp hơn đáng kể không? | Speedup, communication ratio, storage, cost-to-recover | Bảng cost |
| RQ5 — Đánh giá / kiểm chứng forgetting thế nào? | So sánh các metric (acc gap, agreement, backdoor, MIA): khi nào chúng cho kết luận khác nhau | Bảng so sánh metric |

**Hypotheses (giữ H1–H4 gốc, chỉ làm rõ để kiểm định được)**
- **H1:** FU (FedEraser, PGA) đạt utility non-target trong phạm vi noise floor của retrain, với Speedup > 1 khi đã tính recovery.
- **H2:** α càng thấp thì gap trên target split càng lớn (vượt noise floor ở α=0.1), chủ yếu ở target-dominant class; utility non-target giảm theo.
- **H3:** Proposed method (có cơ chế bảo toàn non-target) thu hẹp gap ở α thấp so với baseline cơ sở của nó, và không làm giảm non-target utility.
- **H4:** Khoảng cách tới M_retrain (agreement/KL) và MIA/backdoor đã hiệu chỉnh theo M_retrain là đủ để đánh giá forgetting. Chỉ dùng target acc thì không đủ ở α thấp.

---

## 4. Proposed method (Option B — Non-IID-aware FU)

**Đã có công trình gần với hướng này.** Proposed method phải khác các công trình dưới đây, và gap table ở W11 phải chỉ rõ điểm khác:

| Công trình | Ý tưởng chính | Trạng thái verify |
|---|---|---|
| GDFA (CVPR 2026) | Đưa model về vùng phẳng của loss; lọc task vector theo hướng nhất quán | Chỉ có nguồn thứ cấp; cần bản CVPR Open Access |
| arXiv 2510.13606 | Unlearning 1 round cho heterogeneity cao | arXiv |
| FedCCCU (arXiv 2510.07022) | Ràng buộc cross-client để giảm tác động lên non-target; benchmark heterogeneity thực tế | arXiv |
| Lethe (arXiv 2608.01094) | Benchmark 12 phương pháp FU, có phần thay đổi heterogeneity | arXiv; chưa đọc full text |

**Ứng viên** (chưa kiểm chứng; chọn ở W11 dựa trên failure mode quan sát được ở Gate 4):
1. **Class-ownership-aware PGA**: bán kính projection hoặc trọng số ascent theo mức target-dominance của từng class.
2. **Recovery có bảo toàn non-target**: ràng buộc trên các class shared trong các vòng post-training chỉ với non-target clients.
3. **Calibration của FedEraser theo heterogeneity**: số vòng hoặc tỷ lệ calibration theo α hoặc theo phân phối class của target.

**Yêu cầu:** đơn giản (gắn vào một baseline), có pseudocode, ablation từng thành phần (bản gốc Week 23). Kết quả âm vẫn được báo cáo (Risk 3 của bản gốc).

---

## 5. Lộ trình 26 tuần

Khung tuần gốc (Week 13–24) được giãn ra 26 tuần lịch, có thêm thời gian học lại và buffer.

| Tuần | Từ ngày | Tuần gốc | Việc chính | Học viên chứng minh | Gate / mốc |
|---|---|---|---|---|---|
| W1–W2 | 12/10 | Gate 3 + Week 13 | Commit; tutor xác nhận Gate 3. Chạy Week 13 (retrain tập trung). Build **federated retrain** M_retrain^(-A) + noise floor + `evaluation/forgetting.py` | Backfill M0–M3 (`notes/curriculum_from_start.md`) | **Gate 3** (25/10) |
| W3–W4 | 26/10 | Week 13–14 | FedEraser: đọc, build, đo storage. Backfill M4–M5 | FedEraser L3; giải thích noise floor | — |
| W5–W6 | 09/11 | Week 14–15 | PGA build; KD-FU: đọc và thử reproduce (dừng nếu > 1 tuần chưa chạy được) | PGA L3 | — |
| W7–W8 | 23/11 | Week 16 | So sánh Retrain vs FedEraser vs PGA (vs KD-FU) trên MNIST × 4 mức α; backdoor probe; viết report M4 | Number audit bảng M4; nêu failure mode ở α=0.1 | **Gate 4** (06/12) |
| W9–W10 | 07/12 | Week 17–18 | FFMU, SIFU, survey (P7–P9) + 4 công trình 2025–2026 ở mục 4; taxonomy | Giải thích từng công trình bằng 2 câu: ý tưởng, điểm yếu | — |
| W11 | 21/12 | Week 19 | Gap table; **chốt proposed method**; trình GVHD | Bảo vệ lựa chọn trong 5 phút | Research gap statement |
| W12–W15 | 28/12 | Week 20 | Proposed method: thiết kế, build, kết quả sơ bộ (MNIST α=0.1, 0.5) | Proposed method L3 | **Gate 5** (24/01) |
| W16–W17 | 25/01 | Week 21 | Final benchmark grid (mục 7) chạy qua đêm | Number audit | — |
| W18 | 08/02 | — | **Buffer Tết** (~06/02/2027) | — | — |
| W19–W20 | 15/02 | Week 22–23 | Metrics bổ sung (MIA), ablation (component, α, số client, số client bị xóa) | Giải thích từng ablation | — |
| W21–W23 | 01/03 | Week 24 | Viết thesis Ch.1–7 (Ch.2–3 có nháp từ trước); reviewer draft-check | Mỗi chương qua tutor check | Bản thảo đầy đủ |
| W24–W25 | 22/03 | Week 24 | Slides, demo prototype, 3 buổi rehearsal | Mock defense | — |
| W26 | 05/04 | — | Buffer, chỉnh sửa, nộp | — | **Gate 6** |

**Phân bổ giờ mỗi tuần (15–20 h)**

| Hoạt động | Giờ |
|---|---:|
| Tutor: walkthrough, bài tập, backfill | 6–8 |
| Đọc paper (3 vòng, có MCP) | 3–4 |
| Giao việc cho AI, đọc kết quả, review | 2–3 |
| Viết (nháp thesis, báo cáo) | 2–3 |
| Gặp GVHD | 1 |

**Gate 4 giữ nguyên quy tắc gốc:** chưa reproduce được ít nhất một FU baseline thì chưa đề xuất thuật toán mới, và chuyển sang so sánh ứng dụng.

---

## 6. Evaluation protocol (bổ sung cho mục 10–11 của bản gốc)

- **Gold standard:** M_retrain^(-A) = FL retrain với cùng config, seed và số round, chỉ bỏ target client. Retrain tập trung của Week 13 chỉ là reference, **không** dùng làm gold standard.
- **Noise floor:** gap giữa các retrain khác seed; báo cáo mean ± std.
- **Target client:** quy tắc cố định trước khi xem kết quả. T1 = client có kích thước trung vị; T2 = client lớn nhất; T3 = client có tỷ lệ cao nhất của một class.
- **Splits:** global test · non-target clients · target (shared / target-dominant).
- **Metrics:** giữ toàn bộ metric của bản gốc mục 11; thêm noise floor, agreement/KL, backdoor ASR, cost-to-recover, storage bytes.
- **Thống kê:** ≥ 3 seed × 3 target; không dùng chữ "significantly" khi chưa có kiểm định.

---

## 7. Ma trận thực nghiệm (giữ đủ các chiều của bản gốc mục 12)

| Khối | Dataset | FL | Phân phối | Clients | Target × seed | Methods |
|---|---|---|---|---|---|---|
| **Core** | MNIST, CIFAR-10 | FedAvg | IID, α 1.0 / 0.5 / 0.1 | 10 | 3 × 3 | Retrain, FedEraser, PGA, Proposed (+KD-FU) |
| **Slice FedProx** | MNIST, CIFAR-10 | FedProx | α 0.1 | 10 | 3 × 3 | như Core |
| **Ablation số client** | MNIST | FedAvg | α 0.5, 0.1 | 5, 20 | 1 × 3 | như Core |
| **Mở rộng** (bản gốc) | MNIST | FedAvg | α 0.1 | 10 | — | Deletion ratio 10/20/30%, sequential 2–3 client, MIA |

**Ước tính khối lượng:** khoảng 150 lần train FL đầy đủ + khoảng 400 run FU. **Chưa đo thời gian thật:** cần smoke test ở W1–W2.
- Nếu CIFAR-10 > 40 phút/run: chuyển sang Colab, hoặc giảm còn {IID, 0.1}.

---

## 8. Bằng chứng cho Requirement 1–6

| Requirement (bản gốc) | Bằng chứng |
|---|---|
| R1. Client-level unlearning thực hiện được | FedEraser, PGA, proposed chạy được trên core grid; demo |
| R2. Utility sau FU chấp nhận được | Bảng RQ3 |
| R3. Hành vi gần retrain | Bảng RQ1 so với noise floor |
| R4. Chi phí thấp hơn retrain | Bảng RQ4 |
| R5. Đánh giá được ảnh hưởng của Non-IID | Biểu đồ RQ2 |
| R6. Cải tiến có cơ sở từ limitation của baseline | Proposed method (mục 4) + failure mode từ Gate 4 + ablation |

---

## 9. Paper

Danh sách P1–P9 giữ nguyên. Bổ sung:
- Halimi et al. (PGA, arXiv 2207.05521).
- 4 công trình ở mục 4.
- FedQUIT (arXiv 2408.07587) — protocol cost-to-recover.
- Nguyen et al., Empirical Study of FU (PMLR v222).

P10 để UNVERIFIED. Cách đọc: `docs/huong_dan_MCP_doc_paper.md`.

---

## 10. Rủi ro (bổ sung cho bản gốc mục 19)

| Rủi ro | Xử lý |
|---|---|
| Bài báo không đủ sức thuyết phục nếu chỉ dùng MNIST/CIFAR-10 với Dirichlet tổng hợp (đúng điểm FedCCCU phê phán) | Nếu kết quả Gate 5 tốt, thêm FEMNIST (heterogeneity tự nhiên) ở W19–W20, thay cho một phần ablation |
| Proposed method trùng với GDFA / FedCCCU / 2510.13606 | Gap table ở W11 phải chỉ ra điểm khác; nếu không khác được thì chọn ứng viên khác |
| AI build nhanh hơn tốc độ hiểu | Giới hạn WIP 3 change note ở L0; chỉ chạy canonical khi core đạt ≥ L2 |
| CIFAR-10 chậm trên MacBook | Smoke test sớm; Colab; giảm số mức α của CIFAR-10 |
| KD-FU không reproduce được | Bỏ (đây là optional); vẫn còn FedEraser + PGA |
