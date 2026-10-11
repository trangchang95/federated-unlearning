# Học với tutor

Tutor phải chạy ở **main thread** để hỏi đáp qua lại với bạn. Subagent chạy nền chỉ trả về một bản tóm tắt, không hỏi đáp được.

```bash
cd ~/Developer/School/thesis/federated-unlearning
claude --agent tutor                 # mặc định: Sonnet
claude --agent tutor --model opus    # chỉ khi: walkthrough/L3 core logic, rehearsal, mock defense
```

Core logic = `algorithms/federated_unlearning.py`, `algorithms/proposed_method.py`, `evaluation/forgetting.py`. Mỗi buổi xong: `/clear`.

Mở một terminal riêng cho tutor, song song với session `/goal` của các lane khác.

## Câu mở đầu (copy một dòng)

| Mục đích | Câu mở đầu |
|---|---|
| Học lại từ đầu dự án (tự chọn buổi tiếp theo) | `Backfill: buổi tiếp theo trong notes/curriculum_from_start.md.` |
| Học lại một buổi cụ thể | `Backfill: buổi 2.2.` |
| Walkthrough một thay đổi do AI làm | `Walkthrough notes/changes/<file>.md. Bắt đầu từ "why".` |
| Hàng đợi từ PROGRESS | `Đọc "Blocked on student" trong PROGRESS.md và dạy mục đầu tiên.` |
| Đọc paper | `Dạy mình paper <tên>, dùng literature/<file>_six_questions.md và arXiv MCP để mở đúng section. Đi từ intuition đến figure/table chính.` |
| Hiểu một bảng kết quả | `Number audit: bảng <...> trong reports/<...>.md. Mình tự truy từng số về results/.` |
| Lên L3 | `Bài modify cho <function>: cho mình dự đoán trước, rồi tự sửa và chạy.` |
| Bug-hunt | `Bug-hunt trên bản copy của <function> trong notes/tutor/.` |
| Gate | `Gate <N>: hỏi từng câu, chấm bài mình viết trong reports/gate<N>_self_check.md.` |
| Rehearsal | `Rehearsal 20 phút về <chủ đề>, có câu "show me in your code".` |

## Quy tắc cho bạn
- Trả lời bằng lời của mình trước khi nhìn code hoặc paper. Nếu đoán, nói rõ là đang đoán.
- Không hiểu thì nói "chưa hiểu chỗ X", đừng gật đầu cho qua.
- Mỗi buổi kết thúc khi tutor đã cập nhật `notes/understanding_map.md`.
