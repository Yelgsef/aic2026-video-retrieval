# Trạng thái tối ưu toàn pipeline

Ngày: 2026-08-11

## Đã build

- PP-OCRv6 Small trên final keyframe độ phân giải gốc, Paddle CUDA 12.6.
- Lưu polygon, confidence, OCR raw/searchable và crop cần Gemini.
- Gemini Batch fallback giới hạn tối đa 80 crop/video.
- Gemini Batch normalization có điều kiện cho Whisper large-v3.
- Parser merge OCR crop và ASR correction, không mất raw data.
- Scene document hợp nhất Gemini contact sheet, PP-OCR và normalized ASR.
- Query decomposition thành visual/OCR/ASR sub-query.
- Fusion selection theo task với paired bootstrap.
- Subset 20 video đa dạng, scheduler checkpoint/resume.
- Motion-aware TRAKE reranker dạng candidate-only ablation.
- Giao diện web cục bộ cho KIS, QA và TRAKE.
- Subset 20 đã hoàn tất 220/220 scheduler jobs và toàn bộ Gemini scene/OCR/ASR.
- Offline index subset 20: 1.088 scene, 5.897 frame, SigLIP2 + OpenCLIP + BGE-M3.
- Independent Gemini text index: 1.088 x 768 vectors. Query service cache bản dịch Việt-Anh,
  visual paraphrase, English keywords và query embedding trong SQLite.
- KIS benchmark sau text-first + query expansion: MRR 0,1248; R@1 6,90%; R@5 16,09%;
  R@20 34,48%; R@100 77,01%. Nhánh này chỉ bật cho KIS vì thử nghiệm QA làm giảm R@5/R@10/R@100.
- Gemini scene evidence đã đổi từ số frame cố định sang ngân sách 8-25 theo duration/shot count.
  Smoke test L24_V012 tạo 4 scene/65 frame; chưa submit lại batch toàn corpus.
- OpenCLIP index có key sidecar và incremental reuse; chạy lại reuse 5.897/5.897,
  decode 0 frame.
- Validation silver đa video: 60 query, 3 scene/video trên đủ 20 video.
- Retrieval profile chuyên biệt theo task; CLI và HTTP service dùng chung
  `OfflineSearchEngine`, không còn hai implementation khác nhau.

## Scheduler subset 20 đã hoàn tất

Subset 20 gồm 3,45 giờ video, 2,47 GB. Log:

```text
workspace/scheduler/subset20.stdout.log
workspace/scheduler/subset20.stderr.log
```

Kiểm tra trạng thái:

```powershell
uv run --project environments/siglip2 python scripts/run_dataset_scheduler.py --status
```

Scheduler dùng SQLite WAL, artifact validation, stale-lease recovery và tối đa
ba lần thử. Tắt máy không làm mất trạng thái; chạy lại cùng lệnh để tiếp tục.

OCR decode video tuần tự và inference batch 16; không random-seek từng frame.
Môi trường `environments/ocr` tách biệt PyTorch và tự nạp cuDNN/cuBLAS DLL trên
Windows.

## Gemini auxiliary batch

Pipeline chỉ chuẩn bị request. Đăng ký và submit có chủ đích:

```powershell
uv run --project environments/gemini python scripts/gemini_batch_generic.py register --task asr-normalize --video-id L21_V001
uv run --project environments/gemini python scripts/gemini_batch_generic.py submit   --task asr-normalize --video-id L21_V001

uv run --project environments/gemini python scripts/gemini_batch_generic.py register --task ocr-crops --video-id L21_V001
uv run --project environments/gemini python scripts/gemini_batch_generic.py submit   --task ocr-crops --video-id L21_V001
```

Không submit auxiliary batch rỗng. Sau download, chạy parser tương ứng rồi
rebuild scene documents và global indexes.

## Giao diện

```powershell
uv run --project environments/siglip2 python scripts/serve_offline_query.py --index workspace/offline_index_subset10
```

Mở `http://127.0.0.1:8765/`.

## Benchmark hiện tại

- Multi-video baseline: MRR 0,574; R@1 41,7%; R@10 85,0%.
- KIS quality candidate: R@10 tổng 93,3%; KIS MRR 0,555 -> 0,567.
- QA reasoning v4: frame hit 91,7%; answer/joint 66,7%; khoảng 31,9 giây/query.
  Router theo loại câu hỏi, temporal strip, relation-aware crop và adjudication
  Gemini chỉ chạy khi hai lượt đầu bất đồng.
- TRAKE cấu hình giữ lại: video accuracy 100%; event hit 90%; full-sequence
  accuracy 70%; strictly increasing 100%; khoảng 4,83 giây/chuỗi.
  Ablation mới xác nhận cửa sổ 2 giây tốt hơn cửa sổ 1 giây, multi-scale
  1/2/4 giây và global-fallback weight 0,20; ba biến thể sau đều chỉ đạt
  event hit 80% và full-sequence 60%, nên không được bật production.
  X-CLIP Base/Patch32 8-frame candidate reranker cũng đã được triển khai và đo:
  event hit 76,7%, full-sequence 50%, latency 12,64 giây/chuỗi. Kết quả thấp hơn
  baseline 2 giây nên X-CLIP chỉ giữ làm ablation, không bật trong service.

Đây là pseudo/silver benchmark, chưa phải strong human-verified benchmark.

Benchmark nhãn object-grounded bên ngoài (90 KIS, 40 QA, 20 TRAKE; tất cả vẫn
`verified=false`) cho thấy một miền hard-negative mới. Trên subset20, KIS 87
query đạt MRR 0,0893/R@10 19,54%; QA retrieval 36 query đạt MRR 0,0741/R@10
16,67%. QA end-to-end công bằng trên 36 query: frame hit 2,78%, semantic answer
5,56%, joint 0%, latency 26,22 giây. TRAKE 20 chuỗi: video accuracy 10%, event
hit/full-sequence 0%, dù stride-5 feature phủ đủ 60/60 ground-truth event
ranges. Pipeline hiện yếu rõ rệt với count + spatial relation + object-only
descriptions; không được dùng bộ này để tuyên bố production quality.

Object-spatial v1 đã tái sử dụng YOLO boxes của đủ 20/20 video: đếm cực đại
theo từng frame (không đếm trùng xuyên thời gian), tâm bbox theo left/center/
right và top/middle/bottom, cùng các quan hệ left/right/above/below/near. KIS
hard-label tăng MRR 0,0893 -> 0,1098 và R@10 19,54% -> 21,84%. QA retrieval
tăng MRR 0,0741 -> 0,0810, R@5 8,33% -> 13,89% và R@10 16,67% -> 22,22%.
TRAKE object video fusion weight 0,9 tăng video accuracy 10% -> 25%, nhưng
event hit chỉ 3,33% và full-sequence vẫn 0%; scene alignment object-only vẫn
là điểm yếu chưa giải quyết.

Fusion optimizer hiện cache bảy component scores và dùng 5-fold validation
theo `video_id`. KIS-count và spatial profiles bị từ chối vì giảm R@100 hoặc
CV MRR. Chỉ QA-count được bật, và chỉ rerank trong top-150 candidate cũ. Live
benchmark: MRR 0,0810 -> 0,0900, R@1 2,78% -> 5,56%, R@5 giữ 13,89%, R@100
50% -> 61,11%; R@10 giảm nhẹ 22,22% -> 19,44%. Candidate ngoài top-150 giữ
nguyên thứ tự để không phá recall.

QA reasoning v4 phân loại câu hỏi thành color/OCR/count/object/spatial/motion/
general. Motion dùng năm frame theo thứ tự thời gian; object, OCR và color mới
kích hoạt crop, đồng thời giữ thêm crop rộng để không mất quan hệ chủ thể-vật
thể. Nếu đáp án broad-scene và high-resolution không tương đương ngữ nghĩa,
Gemini thực hiện một lượt phân xử có điều kiện. Trên 12 multi-range pseudo-query,
frame hit tăng 83,3% -> 91,7% và joint accuracy tăng 58,3% -> 66,7%. Một query
gặp lỗi 504 trong lượt benchmark; retry cho 504/DEADLINE_EXCEEDED đã được bổ
sung và rerun mục tiêu hoàn tất. Đây vẫn là pseudo benchmark, không thay thế
strong human-verified validation.

## Điều kiện trước production

- Tạo strong labels có human verification.
- Chỉ bật motion reranker nếu TRAKE ablation tăng điểm.
- Re-run fusion selection trên strong labels.
- Dataset audit hiện thấy 873 video đọc được, không phải đúng 800; cần chốt
  danh sách batch 1 chính thức trước khi submit Gemini toàn corpus.
