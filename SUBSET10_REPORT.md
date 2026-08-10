# Báo cáo validation subset 10 video

Ngày chạy: 2026-08-10

## Phạm vi

10 video thuộc 8 nhóm dữ liệu, tổng 2,1 giờ nội dung và khoảng 1,14 GB:

`L21_V001`, `L21_V006`, `L22_V001`, `L23_V013`, `L24_V045`, `L25_V058`,
`L26_V208`, `L27_V003`, `L29_V013`, `L30_V059`.

## Artifact

- OmniShotCut, stride-5 sampling, SigLIP2 Base clustering, SO400M final
  embedding, Whisper large-v3, semantic scenes và Gemini batch preparation:
  70/70 checks hợp lệ.
- 648 semantic scenes.
- 3.744 final-frame vectors, dimension 1.152.
- 648 BGE-M3 vectors, dimension 1.024.
- Gemini Batch: 10/10 jobs thành công với `gemini-3.6-flash`.
- Parser: 647/648 enriched scenes. Scene còn lại bị `RECITATION` và dùng
  fallback ASR + visual, không bị loại khỏi index.
- Index: `workspace/offline_index_subset10`.

## Retrieval benchmark

50 Gemini pseudo-label của `L21_V001`, tìm kiếm trên toàn bộ subset 10 video:

| Metric | Kết quả |
|---|---:|
| MRR | 0,6180 |
| Recall@1 | 0,50 |
| Recall@5 | 0,78 |
| Recall@10 | 0,92 |
| Recall@20 | 0,98 |
| Recall@50 | 0,98 |
| Recall@100 | 0,98 |
| Mean query latency | 1,126 giây |

KIS: MRR 0,6562; Recall@5 0,8158. QA retrieval: MRR 0,4972; Recall@5
0,6667.

Các số liệu này chưa phải điểm dự đoán vòng sơ loại vì query vẫn là pseudo-label
và ground truth chỉ thuộc một video. Mục đích của lượt chạy là kiểm tra
end-to-end, hard negatives, độ ổn định artifact và khả năng mở rộng đa video.

## QA end-to-end

12 QA pseudo-query được chạy qua hybrid retrieval, shortlist 24 frame và
Gemini VQA:

| Metric | Kết quả |
|---|---:|
| Frame hit rate | 0,4167 |
| Answer match rate | 0,5833 |
| Joint frame + answer accuracy | 0,3333 |
| Mean latency | 8,534 giây |

Gemini trả response thành công 12/12. Kết quả cho thấy lỗi chính vẫn nằm ở chọn
evidence frame/candidate ranking; chỉ cải thiện prompt VQA sẽ không đủ.

## TRAKE end-to-end

10 TRAKE pseudo-query đa video, mỗi query có ba sự kiện theo thứ tự:

| Metric | Kết quả |
|---|---:|
| Video selection accuracy | 1,00 |
| Strictly increasing sequence | 1,00 |
| Per-event temporal hit rate | 0,4667 |
| All-events sequence accuracy | 0,10 |
| Mean latency | 4,396 giây |

Nhánh chọn video và ràng buộc thứ tự hoạt động tốt, nhưng SigLIP Base
frame-level alignment chưa đủ chính xác. TRAKE cần coarse-to-fine alignment và
scene-aware temporal constraints trước khi dùng trong thi đấu.
