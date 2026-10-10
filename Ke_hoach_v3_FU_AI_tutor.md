# Kế hoạch luận văn — bản sửa đổi v3 (AI build, tutor dạy lại)
## Đánh giá Federated Unlearning cấp client trong môi trường dữ liệu không đồng nhất

**Ngân sách:** 5 h/tuần × 26 tuần ≈ **130 h**, từ 12/10/2026 đến ~12/04/2027
**Máy:** MacBook Air (Apple Silicon M5, 16 GB, MPS). Giờ chạy máy qua đêm không tính vào 5 h.
**Trạng thái:** đang làm Gate 3 (IID/Non-IID + FedProx)
**Mục tiêu số 1:** hiểu 100% đồ án và bảo vệ được từng khái niệm, từng dòng code, từng con số trước hội đồng
**Mô hình làm việc:** AI build (code, chạy thí nghiệm, phân tích, viết nháp) → tutor dạy lại → bạn chứng minh đã hiểu. Giờ của bạn dành cho việc **hiểu và kiểm tra**, không phải để gõ code.
**Bản gốc:** `Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md` (giữ lại để đối chiếu)

---

## 1. Thay đổi so với bản gốc

| # | Bản gốc | Bản v2 | Lý do |
|---|---|---|---|
| 1 | 15–20 h/tuần (~450 h) | 5 h/tuần (~130 h) | Ràng buộc thực tế. Phạm vi phải giảm khoảng 70% |
| 2 | Contribution: thuật toán Non-IID-aware mới (Option B) | **Benchmark-first**: một protocol đánh giá phân tầng theo heterogeneity. Thuật toán mới thành phần bổ sung (W13–W15) | Hướng này đã có nhiều công trình (mục 9). Một thuật toán đứng một mình dễ bị phương pháp đã công bố vượt trội |
| 3 | Baselines: FedEraser + KD-FU | **Retrain + PGA + FedEraser** | PGA đơn giản và không cần lưu update. KD-FU cần proxy data ở server và chỉ là arXiv preprint |
| 4 | Gap analysis ở tuần 19 | Chốt gap ở **W9** | Biết sớm mình đóng góp gì |
| 5 | Forgetting = target acc giảm | Forgetting = **khớp M_retrain trong phạm vi noise floor**, tách shared-class và target-dominant | Ở α=0.1, M_retrain cũng có acc thấp trên target data, nên acc giảm không chứng minh đã quên |
| 6 | Speedup = wall-clock | Wall-clock + bytes + storage + recovery rounds | Đánh giá chi phí công bằng |
| 7 | Matrix: 2 dataset × 2 FL × 4 α × 3 #client | MNIST đủ 4 α. CIFAR-10 chỉ {IID, 0.1}. 10 clients. FedProx chỉ dùng ở Gate 3 | Vừa sức tính toán của MacBook Air |
| 8 | Báo cáo tháng 10–15 trang | Báo cáo ngắn 3–5 trang và viết dần Ch.2 | Tiết kiệm giờ, thesis không dồn cuối kỳ |
| 9 | 4 subagent | 4 subagent mới: **tutor**, lit-scout, experiment-engineer, reviewer | Mục tiêu là hiểu bài, nên cần tutor |
| 10 | Học viên tự viết code | **AI viết toàn bộ code và nháp**. Bạn phải đạt mức hiểu L2/L3 trước khi kết quả được đưa vào thesis | Điểm nghẽn chuyển từ việc code sang việc hiểu. Giờ tiết kiệm được dồn cho walkthrough, cải tiến thuật toán và rehearsal |

---

## 2. Đề tài và contribution

**Tên đề xuất**
- VN: *Đánh giá Federated Unlearning cấp client trong môi trường Federated Learning dữ liệu không đồng nhất*
- EN: *Evaluating Client-Level Federated Unlearning under Non-IID Data: A Heterogeneity-Stratified Protocol*

**Contribution chính (core, bắt buộc)**
1. Một protocol đánh giá FU cấp client, gồm:
   - so sánh với **noise floor** (khoảng cách giữa hai lần retrain khác seed);
   - tách target data thành **shared-class** và **target-dominant class**;
   - kế toán chi phí đầy đủ (thời gian, bytes, storage, recovery).
2. Áp dụng protocol cho Retrain, PGA, FedEraser trên MNIST/CIFAR-10 với 4 mức α.
3. Kết luận thực nghiệm: phương pháp nào khớp retrain ở mức Non-IID nào, và với chi phí bao nhiêu.

**Cải tiến thuật toán (W13–W15; mặc định là làm, vì AI viết code)**
- Chỉ làm khi pass Gate 4 đúng hạn và gap check ở W9 không cho thấy trùng với công trình đã có.
- Chi phí thật nằm ở **giờ để hiểu** (~10–15 h), không phải giờ code.
- Một cải tiến nhỏ gắn lên PGA hoặc FedEraser, chọn ở W9 dựa trên failure mode quan sát được.
- Ví dụ ứng viên (**chưa kiểm chứng**): điều chỉnh bán kính projection của PGA, hoặc số vòng calibration của FedEraser, theo mức Non-IID.
- Đánh giá bằng chính protocol ở trên. Kết quả âm vẫn được báo cáo.

**Requirement 6 của bản gốc** ("một cải tiến có cơ sở từ limitation của baseline"): đề xuất với GVHD rằng cải tiến ở đây là **cải tiến về đánh giá**, xuất phát từ hạn chế trong cách các baseline tự đánh giá. Cải tiến thuật toán ở trên là phần bổ sung. Nếu nó thất bại, Requirement 6 vẫn được đáp ứng nhờ cải tiến về đánh giá.

---

## 3. Research questions và hypotheses (bản sửa)

| RQ | Câu hỏi | Bằng chứng trả lời |
|---|---|---|
| RQ1 | M_FU có khớp M_retrain^(-A) trong phạm vi noise floor trên target split không? | Bảng gap per-split vs noise floor |
| RQ2 | Gap thay đổi thế nào theo α, xét riêng shared-class và target-dominant? | Biểu đồ gap theo α, tách 2 nhóm class |
| RQ3 | Utility trên non-target clients và global test có giữ được không? | Bảng acc/F1 per-split |
| RQ4 | Chi phí so với retrain (thời gian, bytes, storage, cost-to-recover)? | Bảng cost, Speedup |
| RQ5 | Các metric forgetting (acc gap, agreement, backdoor ASR) có bất đồng không, và khi nào? | Bảng so sánh metric; đây là một kết quả của luận văn |

- **H1:** PGA và FedEraser đạt utility trên non-target trong phạm vi noise floor, với Speedup > 1 khi đã tính recovery.
- **H2:** Gap trên target split tăng khi α giảm và vượt noise floor ở α=0.1, chủ yếu ở các target-dominant class.
- **H3:** Ở α thấp, chỉ dùng target accuracy sẽ cho kết luận khác với khi so với noise floor.
- **H4:** Backdoor ASR sau FU tiến về mức của M_retrain, nhưng không đủ để kết luận về phần clean data.

---

## 4. Phạm vi

| Làm | Không làm |
|---|---|
| FedAvg, FedProx (chỉ ở Gate 3), 10 clients | 5/20 clients (chỉ làm nếu còn giờ, chỉ trên MNIST) |
| MNIST: IID, α = 1.0, 0.5, 0.1 · CIFAR-10: IID, α = 0.1 | CIFAR-100, FEMNIST |
| Retrain, PGA, FedEraser | KD-FU, FFMU, SIFU (chỉ đọc concept) |
| Unlearn 1 client, 3 target client | Sequential deletion, data-level unlearning |
| Backdoor probe (grid nhỏ), cải tiến thuật toán (W13–W15) | MIA (chỉ làm nếu tutor xác nhận bạn đã đạt L2 với mọi phần core) |
| CLI demo: train → yêu cầu xóa → unlearn → báo cáo | Dashboard, container, UAV, blockchain |

---

## 5. Lộ trình 26 tuần

**Một tuần mẫu (~5 h)**

| Việc | Giờ | Ai làm |
|---|---:|---|
| Giao việc cho AI: nói mục tiêu, duyệt kế hoạch, đọc báo cáo kết quả | 0,5 | Bạn + experiment-engineer / lit-scout / main session |
| Walkthrough với tutor: change note, code, kết quả | 2,5 | Bạn + tutor |
| Bài tập: predict-then-run, explain-back, modify, bug-hunt | 1,5 | Bạn (tutor chấm) |
| Đọc lại nháp thesis của tuần, ghi câu hỏi | 0,5 | Bạn |

AI chạy ngoài 5 h này: build code, chạy grid qua đêm, viết nháp.

**Mức hiểu**

| Mức | Nghĩa |
|---|---|
| L0 | AI vừa build xong |
| L1 | Đã walkthrough với tutor |
| L2 | Bạn tự giải thích lại được, không nhìn code |
| L3 | Bạn tự sửa một chỗ, dự đoán đúng kết quả, giải thích được chỗ lệch |

**Lộ trình**

| Tuần | Bắt đầu | Giờ | AI build | Bạn học / chứng minh | Gate |
|---|---|---:|---|---|---|
| W1–W2 | 12/10 | 10 | Hoàn tất code Gate 3: partitioner + histogram, FedProx, benchmark 4 mức α × 3 seeds, cờ `exclude_clients`, báo cáo M3 nháp | Walkthrough FedAvg, FedProx, Dirichlet; predict-then-run: acc khi α giảm | **Gate 3** (25/10) |
| W3–W4 | 26/10 | 10 | Retrain harness, `evaluation/` (3 split, agreement, phân loại class, noise floor). Smoke test và đo t_train. lit-scout tóm tắt Halimi + Liu | Machine unlearning, gold standard, noise floor; đọc 2 paper qua tutor | Bảng retrain vs retrain |
| W5–W6 | 09/11 | 10 | PGA + test + change note | PGA lên L3 (modify: bán kính projection) | — |
| W7–W8 | 23/11 | 10 | FedEraser + đo storage/bytes + change note | FedEraser lên L3 (modify: số vòng calibration) | **Gate 4** (06/12) |
| W9 | 07/12 | 5 | Backdoor probe. lit-scout lập gap table (GDFA, 2510.13606, FedCCCU, Lethe) | Đọc gap table với tutor. **Chốt contribution + chọn cải tiến** | Gap statement |
| W10–W12 | 14/12 | 15 | Chạy full grid qua đêm, sinh bảng và hình | Number audit từng bảng; tự giải thích từng kết quả RQ1–RQ4 | **Gate 5** (03/01) |
| W13–W15 | 04/01 | 15 | Cải tiến thuật toán + ablation (hoặc MIA nếu bỏ cải tiến) | Cải tiến lên L3; trả lời được "tại sao nó tốt/xấu ở α=0.1" | — |
| W16–W17 | 25/01 | 10 | Main session viết nháp Ch.2–4; reviewer draft-check | Tutor kiểm tra từng chương; bạn sửa câu chữ thành giọng của mình | — |
| W18 | 08/02 | 5 | — | **Buffer Tết** (~06/02/2027) | — |
| W19–W21 | 15/02 | 15 | Nháp Ch.5–7, abstract, intro; reviewer critique toàn bộ | Mỗi con số trong Ch.5 đã qua number audit | Bản thảo đầy đủ |
| W22–W24 | 08/03 | 15 | Slides + demo CLI | 3 buổi rehearsal; tutor ra câu hỏi "show me in your code" | — |
| W25–W26 | 29/03 | 10 | Sửa theo góp ý | Mock defense cuối; mọi mục đạt mức yêu cầu | **Gate 6** |
| **Tổng** | | **130** | | | |

**Điều kiện đi tiếp (thay cho hạn chót viết code)**
- Không chạy canonical grid nếu core function chưa đạt L2.
- Không đưa kết quả hay chương nào vào thesis nếu bạn chưa đạt mức yêu cầu: core L3, phần còn lại L2.

**Quy tắc khi trễ** (trễ giờ *hiểu*, không phải giờ code; cắt theo thứ tự):
1. Bỏ MIA.
2. Thu hẹp cải tiến thuật toán (chỉ giữ ablation 1 thành phần), hoặc bỏ hẳn.
3. CIFAR-10 giảm còn 2 target client.
4. Bỏ α=1.0 trên MNIST.
5. Bỏ FedEraser, chỉ giữ Retrain vs PGA (vẫn trả lời được RQ1–RQ5).

Không bao giờ cắt walkthrough hay rehearsal để giữ scope.

---

## 6. Evaluation protocol

**Gold standard.** M_retrain^(-A) dùng cùng config, seed và số round với run gốc, chỉ bỏ target client.

**Noise floor.** Với mỗi (dataset, α, target), so sánh hai retrain khác seed. M_FU được coi là "khớp" khi gap của nó so với M_retrain không lớn hơn gap retrain-vs-retrain (báo cáo mean ± std).

**Chọn target client.** Quy tắc cố định trong config, quyết định trước khi xem kết quả:
- T1 = client có kích thước dữ liệu trung vị;
- T2 = client lớn nhất;
- T3 = client nắm tỷ lệ cao nhất của một class nào đó.

**Phân loại class.** Class c là *target-dominant* nếu target client giữ ≥ 50% số mẫu train của c. Các class còn lại là *shared*. Ở IID không có class dominant.

| Nhóm metric | Metric | Split |
|---|---|---|
| Utility | Accuracy, macro-F1 | global test, non-target clients |
| Forgetting | gap acc vs M_retrain; prediction agreement với M_retrain; (KL output) | target (shared / dominant) |
| Probe | Backdoor ASR so với ASR của M_retrain | trigger set |
| Cost | wall-clock (gồm recovery/calibration), bytes truyền, storage của FedEraser, Speedup, cost-to-recover | — |

**Lưu ý**
- Không dùng parameter distance giữa các model có init khác nhau (không có ý nghĩa). Chỉ dùng khi cùng init.
- Báo cáo mean ± std trên 3 seeds × 3 target. Không dùng chữ "significantly" nếu không có test.

**Backdoor probe (grid riêng, nhỏ):** MNIST, α ∈ {IID, 0.1}, 3 seeds, target T1. Target client gắn trigger vào một phần dữ liệu của mình và đổi nhãn.

---

## 7. Grid thực nghiệm và tính toán

| Dataset | α | Train gốc | Retrain (3 target × 3 seed) | FU runs (PGA + FedEraser) |
|---|---|---:|---:|---:|
| MNIST | IID, 1.0, 0.5, 0.1 | 12 | 36 | 72 |
| CIFAR-10 | IID, 0.1 | 6 | 18 | 36 |
| **Tổng** | | **18** | **54** | **108** |

→ 72 lần train FL đầy đủ + 108 lần chạy FU + grid backdoor nhỏ.

**Ước tính thời gian máy.** Chưa đo được, phải đo bằng smoke test ở W5.
- Công thức: Tổng ≈ (72 × t_train) + (108 × t_FU).
- Nếu t_train(CIFAR-10) > 40 phút: giảm số round hoặc giảm còn 2 target; Colab free là phương án dự phòng.

**Lưu ý MacBook Air (không có quạt)**
- Chạy khi cắm sạc, dùng `caffeinate -i`, mỗi lần một process, có checkpoint/resume cho từng run.
- Ghi wall-clock của từng run để phát hiện throttling.
- MPS không đảm bảo kết quả lặp lại bit-by-bit: kiểm tra determinism bằng smoke test trên CPU, và dựa vào nhiều seed.
- Ghi device vào result JSON.

---

## 8. Đọc paper (giảm còn những gì bảo vệ cần)

| Mức | Paper | Trạng thái verify |
|---|---|---|
| Đọc kỹ với tutor | Liu et al., FedEraser, IWQoS 2021 | Verified (IEEE Xplore 9521274) |
| Đọc kỹ với tutor | Halimi et al., Federated Unlearning: How to Efficiently Erase a Client in FL? | arXiv 2207.05521 verified; venue chưa xác nhận |
| Đọc kỹ với tutor | Li et al., FedProx, MLSys 2020 | Cần lit-scout verify |
| Đọc phần evaluation | Romandini et al., FU survey | Venue TNNLS chưa verify |
| Đọc phần evaluation | Nguyen et al., Empirical Study of FU (PMLR v222, 2024) | Verified (proceedings.mlr.press) |
| Đọc phần evaluation | FedQUIT, arXiv 2408.07587 (protocol cost-to-recover) | arXiv verified |
| Skim (abstract + hình + kết quả) | GDFA (CVPR 2026) | Chỉ có nguồn thứ cấp; cần CVPR Open Access |
| Skim | Towards Robust Knowledge Removal in FL with High Data Heterogeneity, arXiv 2510.13606 | arXiv |
| Skim | FedCCCU, arXiv 2510.07022 | arXiv verified |
| Skim | Lethe, arXiv 2608.01094 — **kiểm tra overlap với contribution trước W9** | arXiv; chưa đọc full text |
| Skim | Wu et al. KD-FU (arXiv 2201.09441); FFMU (ICML 2023); SIFU (AISTATS 2024); SISA (S&P 2021) | KD-FU verified; các bài khác cần verify |
| Bỏ | P10 "Towards practical federated unlearning…" (AEJ 2026) | **Không tìm thấy** |
| Bỏ | Kairouz survey (chỉ tra cứu khi cần) | — |

**Quy trình đọc:** tự trả lời 6 câu trước → lit-scout kiểm tra và bổ sung setup → tutor hỏi 2–3 câu kiểm tra.

---

## 9. Lập luận để trình GVHD về việc đổi hướng

1. **Thời gian:** 130 h thay vì ~450 h như kế hoạch gốc giả định. Một thuật toán mới cộng với ablation không vừa trong ngân sách này.
2. **Mức độ cạnh tranh của hướng Non-IID-aware FU:** đã có nhiều phương pháp, gồm một phương pháp nhắm thẳng vào việc unlearn target client dưới Non-IID (GDFA, CVPR 2026), một phương pháp one-round cho heterogeneity cao (arXiv 2510.13606), và FedCCCU (2025).
3. **Khoảng trống về đánh giá:** FedCCCU cho rằng các đánh giá FU hiện tại dựa trên Non-IID tổng hợp, dễ phóng đại độ robust. Protocol có noise floor và tách class giải quyết đúng vấn đề này. Cần đối chiếu với Lethe ở W9.
4. **Vẫn đáp ứng Requirement 1–5.** Requirement 6 được đáp ứng bằng cải tiến về đánh giá, cộng thêm một cải tiến thuật toán nhỏ ở W13–W15.
5. **Phù hợp mục tiêu Thạc sĩ ứng dụng:** prototype, thực nghiệm có đối chứng, và học viên hiểu sâu, giải thích được trước hội đồng.

---

## 10. Dùng AI

| Agent | Làm gì | Ranh giới |
|---|---|---|
| experiment-engineer | Viết **toàn bộ** code (kể cả PGA, FedEraser, cải tiến), test, chạy grid, vẽ hình, làm demo. Viết **change note** cho mỗi thay đổi | Mỗi thay đổi chỉ một khái niệm, < 150 dòng, để vừa một buổi walkthrough |
| lit-scout | Verify và tóm tắt paper (6 câu + setup + 3 câu hỏi cho tutor), lập gap table | Chỉ dùng nguồn gốc của paper |
| main session | Viết nháp thesis/báo cáo theo evidence rules | Câu chữ phải là giọng bạn bảo vệ được |
| reviewer | Critique trước gate và buổi gặp GVHD; draft-check từng chương; liệt kê phần chưa đạt mức hiểu | Chỉ đọc |
| tutor | Walkthrough mọi thứ AI làm, ra bài tập, chấm mức L0–L3, gate self-check, rehearsal | Không build gì trong thesis |

**Vòng lặp mỗi thay đổi**
1. Bạn nói mục tiêu.
2. AI build và ghi `notes/changes/<ngày>_<chủ đề>.md`.
3. Tutor walkthrough → bài tập → cập nhật `notes/understanding_map.md`.
4. Nếu tutor phát hiện lỗi, chuyển lại cho AI sửa. Mỗi lần sửa, mức hiểu reset về L0.

**Câu hỏi hội đồng chắc chắn sẽ gặp:** *"Code này ai viết?"*
- Trả lời trung thực: AI hỗ trợ viết code.
- Rồi chứng minh hiểu ngay tại chỗ: mở file, giải thích từng bước, nói được vì sao chọn cách này thay vì cách khác.
- Mức L3 tồn tại chính để trả lời câu này.

---

## 11. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| **Ảo tưởng là đã hiểu** (đọc code AI thấy "hợp lý" nhưng không tự tái tạo được) | Qua L1 nhanh nhưng trượt L2/L3; không dự đoán được kết quả | Không bỏ qua explain-back và predict-then-run. Rehearsal "show me in your code" từ W10 |
| AI build nhanh hơn tốc độ hiểu | Change note ở L0 dồn > 3 cái | Dừng build cho đến khi đã walkthrough hết các change đang dồn |
| AI viết sai mà trông đúng | Test pass nhưng kết quả lạ; tutor phát hiện lệch so với paper | Test bằng toy case tính tay được; reviewer critique; trích section paper cho từng bước |
| 5 h/tuần không đủ để hiểu hết | Trễ hơn 1 tuần ở bất kỳ gate nào | Áp dụng quy tắc cắt ở mục 5 |
| Lethe/FedCCCU đã làm protocol tương tự | Kết quả kiểm tra overlap ở W9 | Thu hẹp novelty: tập trung vào noise floor theo α trên client-level; dựa nhiều hơn vào cải tiến thuật toán |
| FedEraser không reproduce được | Gate 4 không đạt | Chỉ dùng Retrain + PGA |
| CIFAR-10 quá chậm trên máy | t_train > 40 phút | Giảm round/target, hoặc dùng Colab |

---

## 12. Cấu trúc thesis

1. Introduction
2. Background: FL, Non-IID, MU, FU
3. Related Work: các phương pháp FU và cách chúng tự đánh giá
4. Evaluation Protocol (contribution; thay cho "Proposed Method")
5. Experiments: trả lời RQ1–RQ4
6. Discussion: RQ5, các hạn chế, khi nào metric đánh giá sai
7. Conclusion
