# Dùng arXiv MCP để hiểu paper

> **Giả định cần kiểm tra:** server là `arxiv-mcp-server` (blazickjp), đăng ký trong Claude Code với tên `arxiv`.
> Kiểm tra bằng `claude mcp list`, hoặc gõ `/mcp` trong Claude Code. Nếu tên khác, sửa `mcp__arxiv` trong `tools:` của `.claude/agents/lit-scout.md` và `tutor.md`.
> Tên tool bên dưới lấy từ trang mô tả server. Phiên bản bạn cài có thể khác, nên đối chiếu với `/mcp`.

## 1. Tool cần dùng

| Việc | Tool | Ghi chú |
|---|---|---|
| Tìm paper | `search_papers` (`query`, `date_from`, `sort_by`, `max_results`) | Dùng để quét công trình mới, ví dụ "federated unlearning non-IID" từ 2025 |
| Đọc nhanh | `get_abstract` | Vòng 1 |
| Xem cấu trúc | `get_paper_outline` | Lấy `section_id` cho bước sau |
| Đọc từng phần | `read_paper_section` (`paper_id`, `section_id`) | Vòng 2. Đọc từng section thay vì cả bài |
| Lấy công thức chuẩn | `get_paper_latex_section` | Vòng 3. LaTeX giữ đúng công thức, tốt hơn text trích từ PDF |
| Tải về để dùng lại | `download_paper`, `list_papers` | |
| Ai trích dẫn / ai được trích | `citation_graph` | Tìm công trình nối tiếp FedEraser hoặc PGA |
| Xuất BibTeX | `export_citations` | Vẫn phải verify venue bản chính thức |

Server có thể có thêm prompt như `deep-paper-analysis`, `compare_papers`, `summarize_paper`. Trong Claude Code, prompt của MCP thường xuất hiện dưới dạng lệnh `/mcp__arxiv__<tên>` (gõ `/` để xem danh sách).

## 2. Đọc một paper: 3 vòng, kết hợp với tutor

| Vòng | Thời gian | Làm gì | Tutor kiểm tra |
|---|---|---|---|
| 1. Bức tranh | ~10 phút | `get_abstract` + `get_paper_outline`. Tự viết 3 dòng: bài toán, ý tưởng, kết quả chính | "Paper này giải quyết bài toán gì mà retrain không giải quyết được?" |
| 2. Cơ chế | ~30 phút | `read_paper_section` cho phần method và experiments. Xem figure/table chính trong PDF ở `articles/` | Explain-back thuật toán, kèm sơ đồ |
| 3. Chi tiết | khi cần | `get_paper_latex_section` lấy công thức → map vào code (`file:line`) | Bài modify trên code tương ứng |

Câu mở đầu cho tutor:
`Dạy mình paper arXiv <id> theo 3 vòng. Dùng get_paper_outline trước. Mỗi vòng hỏi mình 1 câu.`

## 3. arXiv ID của các paper chính

Chỉ ghi ID đã verify. Paper chưa có ID thì dùng `search_papers` với tên paper, rồi để lit-scout verify.

| Paper | arXiv ID | Lưu ý |
|---|---|---|
| FedEraser (Liu et al.) | 2012.13891 | Bản chính thức ở IWQoS 2021; bản arXiv có thể khác. Khi cite thì dùng bản IWQoS |
| SISA (Bourtoule et al.) | 1912.03817 | |
| KD-FU (Wu et al.) | 2201.09441 | Chỉ là preprint |
| PGA (Halimi et al.) | 2207.05521 | Venue chưa xác nhận |
| FedQUIT | 2408.07587 | Protocol đo cost-to-recover |
| FedCCCU | 2510.07022 | Kiểm tra overlap với contribution |
| Robust knowledge removal, high heterogeneity | 2510.13606 | Kiểm tra overlap |
| Lethe | 2608.01094 | Kiểm tra overlap; full text chưa đọc được |

## 4. Bẫy khi dùng MCP

- **Tóm tắt do prompt hay AI tạo ra không phải là nguồn.** Chỉ cite paper gốc, và mọi con số phải kiểm tra lại trên table trong PDF.
- **Text trích từ PDF hay làm hỏng bảng và công thức.** Lấy số từ PDF ở `articles/`, lấy công thức từ LaTeX.
- **arXiv không phải venue.** Bản arXiv có thể khác bản đã xuất bản (FedEraser là ví dụ). lit-scout phải ghi rõ dùng bản nào.
- **Paper ngoài arXiv thì MCP không có.** Ví dụ bản IWQoS của FedEraser, hoặc paper CVPR như GDFA. Với những paper này, dùng PDF local hoặc trang publisher.

## 5. Phân vai

| Ai | Dùng MCP để |
|---|---|
| lit-scout (chạy nền trong `/goal`) | Tìm, verify, viết six_questions có số section, quét công trình mới mỗi tháng |
| tutor (`claude --agent tutor`) | Mở đúng section hoặc công thức khi dạy bạn |
| bạn | Vòng 1 tự làm, để tự trả lời trước khi đọc tóm tắt của AI |
