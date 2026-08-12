# AIC 2026 Multimodal Video Retrieval

Hệ thống xử lý video offline và truy vấn cho ba dạng bài AIC: KIS, QA và TRAKE. Pipeline tạo semantic scenes từ video, lập chỉ mục đa phương thức, hợp nhất nhiều nhánh retrieval và trả về `video_id`, `frame_id` cùng đáp án hoặc chuỗi sự kiện.

## Trạng thái

Đã triển khai:

- OmniShotCut, sampling và keyframe coverage;
- SigLIP2 visual embeddings và OpenCLIP ensemble tùy chọn;
- Faster-Whisper ASR, PP-OCR và Gemini enrichment;
- semantic scenes, object/spatial metadata và OCR-to-frame evidence;
- BGE-M3, lexical/fuzzy retrieval, cross-encoder reranking và fusion;
- QA bằng evidence frames, KIS ranking và TRAKE temporal alignment;
- frontend localhost và định dạng submission top 100.

Dataset, model weights, virtual environments và `workspace/` không được commit. Xem [PIPELINE.md](PIPELINE.md) để biết kiến trúc và các lệnh build.

## Cài môi trường

Yêu cầu Windows, Python/uv, FFmpeg và NVIDIA CUDA nếu chạy GPU.

```cmd
git submodule update --init --recursive
uv sync --project environments/omnishotcut
uv sync --project environments/siglip2
uv sync --project environments/asr
uv sync --project environments/ocr
uv sync --project environments/objects
uv sync --project environments/gemini
copy .env.example .env
```

Điền `GEMINI_API_KEY` trong `.env`. Không commit file này.

## Chạy frontend

Ví dụ với index kiểm thử 10 video:

```cmd
uv run --project environments/siglip2 python scripts/serve_offline_query.py --index workspace/offline_index_queries10
```

Mở `http://127.0.0.1:8765`. Sau khi đổi code hoặc index, dừng server bằng `Ctrl+C` và chạy lại vì model/index được giữ trong RAM.

## Kiểm tra nhanh bằng CLI

```cmd
uv run --project environments/siglip2 python scripts/query_offline_index.py --help
uv run --project environments/siglip2 python -m compileall -q aic_pipeline scripts
```

Notebook minh họa được giữ tại `offline_pipeline_demo.ipynb`. Format pseudo-label nằm trong [PSEUDO_LABEL_FORMAT.md](PSEUDO_LABEL_FORMAT.md).

## Cấu trúc repository

```text
aic_pipeline/   thư viện sampling, indexing, retrieval, QA và temporal alignment
scripts/        entry points cho preprocessing, index, benchmark và serving
environments/   uv projects theo từng nhóm model
tests/          unit/regression tests
web/            frontend truy vấn localhost
third_party/    Git submodules, gồm OmniShotCut
workspace/      output sinh ra cục bộ, không commit
models/         checkpoint cục bộ, không commit
archive/        video dataset, không commit
```

## Quy tắc tái lập

- Metadata dùng `video_id`, `scene_index`, `frame_id` và timestamp gốc.
- Không dùng video ID ground truth để lọc khi benchmark end-to-end.
- Không để Gemini trả lời nếu không có bằng chứng trong OCR, ASR hoặc frame.
- Chỉ commit code, cấu hình, test và tài liệu; không commit dữ liệu hay secrets.

