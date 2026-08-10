# Trạng thái tối ưu toàn pipeline

Ngày: 2026-08-11

## Đã build

- PP-OCRv6 trên final keyframe độ phân giải gốc.
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

## Scheduler đang chạy

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

## Điều kiện trước production

- Chờ subset-20 hoàn tất.
- Tạo strong labels có human verification.
- Chỉ bật motion reranker nếu TRAKE ablation tăng điểm.
- Re-run fusion selection trên strong labels.
- Dataset audit hiện thấy 873 video đọc được, không phải đúng 800; cần chốt
  danh sách batch 1 chính thức trước khi submit Gemini toàn corpus.
