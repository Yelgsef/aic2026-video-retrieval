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
