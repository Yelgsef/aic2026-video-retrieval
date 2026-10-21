# Repository guide

Tài liệu này mô tả cấu trúc và cách vận hành code hiện tại trong repository.
Các artifact dữ liệu, checkpoint và index nằm trên máy hoặc object storage; chúng
không được commit cùng source code.

## 1. Cấu trúc repository

| Đường dẫn | Vai trò |
|---|---|
| `aic_pipeline/` | Module Python dùng chung cho xử lý dữ liệu, retrieval, fusion, QA, KIS và TRAKE. |
| `scripts/` | CLI cho scheduler, từng stage offline, build index, query, serving và validation. |
| `backend/` | FastAPI app, request schema, runtime dùng chung, query, QA và media service. |
| `frontend/` | Web app chính cho KIS, QA, TRAKE và kiểm tra frame; FastAPI phục vụ cùng origin. |
| `environments/` | Các project `uv` tách theo workload và dependency. |
| `tests/` | Unit tests cho sampling, retrieval, fusion, routing, grounding và temporal alignment. |
| `docs/` | Data flow, repository guide và hướng dẫn Query Transformer. |
| `third_party/OmniShotCut/` | Git submodule OmniShotCut dùng cho shot boundary detection. |
| `archive/video/` | Video nguồn; manifest trỏ tới file trong archive. |
| `archive/` | Một số artifact/feature cũ được lưu để tham chiếu; không phải output mặc định của flow hiện tại. |
| `models/`, `weights/` | Checkpoint tải hoặc chuẩn bị cục bộ; không commit. |
| `workspace/` | Manifest, intermediate artifacts, embeddings, scene documents, index, cache và scheduler state; không commit. |
| `web/` | Giao diện cũ với API contract trước lần tách frontend/backend; Flow C hiện chạy `frontend/`. |

Không commit `.env`, dữ liệu video, model weights, `workspace/` hoặc index database.
Không xóa scheduler SQLite nếu cần resume các job.

## 2. Pipeline theo flow

### Flow A — video tới scene evidence

Scheduler riêng: `scripts/run_flow_a_scheduler.py`. Các stage và dependency hiện
tại:

```text
OmniShotCut -> stride-5 sampling -> SigLIP2 Base embeddings
                                  -> SigLIP2 SO400M final embeddings
ASR large-v3 --------------------> semantic scenes
SigLIP2 Base --------------------> semantic scenes
```

ASR có thể chạy độc lập với nhánh OmniShotCut/sampling. Semantic scenes cần cả
SigLIP2 Base features và ASR. Scheduler Flow A chỉ quản lý các stage trên; các
stage enrichment, OCR/YOLO và build global index thuộc scheduler rộng hơn.

Khởi tạo/chạy cho một video:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project environments/siglip2 \
  python scripts/run_flow_a_scheduler.py --init --run \
  --shard-count 1 --shard-index 0 --worker-id flow-a --video-ids L21_V001
```

Xem trạng thái:

```bash
uv run --project environments/siglip2 \
  python scripts/run_flow_a_scheduler.py --status
```

### Flow B — enrichment và global retrieval index

`scripts/run_dataset_scheduler.py` quản lý các stage trên mỗi video rộng hơn:
YOLO-World, full-resolution PP-OCR, chuẩn bị ASR normalization, chuẩn bị 9Router
scene/OCR requests. `run_full_dataset_supervisor.py` và
`run_gemini_subset_supervisor.py` điều phối các stage và build index cho dataset
đầy đủ hoặc subset. Tên file có chữ `gemini` còn được giữ để tương thích; remote
enrichment hiện dùng 9Router.

Các bước tạo search index được chạy sau khi scene documents đã sẵn sàng:

```text
scene documents + final frame metadata/embeddings
  -> build_offline_hybrid_index.py
  -> build_bge_m3_scene_index.py
  -> build_openclip_frame_index.py (optional)
  -> build_milvus_index.py
```

Milvus Lite chứa vector collections HNSW cho ANN search. `search.sqlite3` vẫn
phục vụ FTS/fuzzy và metadata lookup. Parquet/NumPy là artifact nguồn để build
lại index; Milvus DB là derived artifact. Xem
[`DATA_FLOW_AND_STORAGE.md`](DATA_FLOW_AND_STORAGE.md) để biết đầy đủ input/output
theo stage.

Để nhập index dump đời cũ, dùng `scripts/import_legacy_offline_index.py`; script
chọn Parquet/vector cần cho serving, dựng lại FTS và OpenCLIP sidecar, rồi chạy
`scripts/build_milvus_index.py` trên output.

### Flow C — truy vấn online

`frontend/` gửi query tới `backend/`; backend nạp `OfflineSearchEngine`, Milvus
Lite và manifest một lần khi khởi động. KIS/QA dùng chung retrieval. QA có thể
gửi frame đã chọn đến AI Studio hoặc 9Router. TRAKE truy vấn từng event rồi
kiểm tra thứ tự bằng `frame_id`; dense `TemporalAligner` chưa được nối vào
request handler hiện tại. Candidate pool của API được cố định ở 100 cho mỗi
query/event; `top_k` điều khiển số kết quả trả về.

Khởi động với index đầy đủ:

```bash
uv run --project environments/siglip2 \
  python scripts/serve_offline_query.py \
  --index workspace/offline_index_all_873 --host 127.0.0.1 --port 8765
```

Hoặc dùng index nhỏ hiện có để thử giao diện:

```bash
uv run --project environments/siglip2 \
  python scripts/serve_offline_query.py \
  --index workspace/offline_index_builder_refactor_test --port 8765 --no-warmup
```

Mở `http://127.0.0.1:8765`. `GET /api/health` hiển thị model, index và
Milvus collections đã nạp. Dừng service bằng Ctrl+C trước khi đổi code hoặc
index. Có thể cấu hình `MILVUS_URI`, `MILVUS_COLLECTION_PREFIX`, `AIC_DEVICE`,
`AIC_HOST`, `AIC_PORT` qua environment; URI mặc định là
`<index>/milvus_lite.db`.

API chính:

| Endpoint | Chức năng |
|---|---|
| `GET /api/health` | Tình trạng engine và index. |
| `POST /api/route` | Phân loại query KIS/QA/TRAKE. |
| `POST /api/query`, `POST /api/query/stream` | Retrieval và kết quả cho KIS/QA/TRAKE. |
| `POST /api/qa/answer`, `POST /api/qa/models` | Gọi QA provider và lấy danh sách model. |
| `GET /api/frame/{video_id}/{frame_id}` | Giải mã/cached frame từ video nguồn. |
| `GET /videos/{video_id}.mp4` | Stream video nguồn. |
| `GET /api/videos/{video_id}/metadata` | FPS, frame count và duration. |

Backend phân giải `video_id` qua `workspace/manifests/videos.parquet`; frontend
không cần biết đường dẫn tuyệt đối của video.

## 3. Python environments

| Project | Workload chính |
|---|---|
| `environments/omnishotcut` | Shot segmentation với OmniShotCut. |
| `environments/siglip2` | Sampling/index, SigLIP, BGE-M3, Milvus, query và web service. |
| `environments/asr` | Faster-Whisper transcription. |
| `environments/ocr` | PP-OCRv6 full-resolution OCR. |
| `environments/objects` | YOLO-World object detection. |
| `environments/gemini` | 9Router/VLM request preparation, calls và parsing. |

Cài dependency theo workload bằng `uv sync --project environments/<name>`.
Không dùng chung một virtualenv vì các project có dependency và runtime khác nhau.

## 4. Workspace artifact contract

Các đường dẫn runtime chính:

| Đường dẫn | Nội dung |
|---|---|
| `workspace/manifests/videos.parquet` | Video ID, source path và thông tin media sau audit. |
| `workspace/shots/{video_id}/` | Kết quả shot boundary. |
| `workspace/samples/{video_id}/` | Sampled frame IDs/metadata. |
| `workspace/features/{encoder_id}/{video_id}/` | Dense visual features theo encoder. |
| `workspace/keyframes/{video_id}/` | Final keyframe metadata. |
| `workspace/asr/faster_whisper_large_v3/{video_id}.json` | Transcript và timestamps từ Faster-Whisper large-v3. |
| `workspace/semantic_scenes/{video_id}/` | Scene-level evidence của Flow A. |
| `workspace/objects_yoloworld/{video_id}/` | Object detections/spatial evidence. |
| `workspace/ocr_fullres/{video_id}/` | Full-resolution OCR, crop và provenance. |
| `workspace/scene_documents/{video_id}/` | Unified scene documents dùng cho index. |
| `workspace/scheduler/jobs.sqlite3` | Trạng thái scheduler tổng quát. |
| `workspace/scheduler/flow_a_jobs.sqlite3` | Trạng thái scheduler riêng Flow A. |
| `workspace/offline_index*/` | Parquet metadata, vector `.npy`, SQLite và Milvus Lite index. |

Một index serving cần tối thiểu scene/frame Parquet, vector nguồn tương ứng,
`search.sqlite3`, `milvus_lite.db` và `milvus_report.json`. Nếu dùng OpenCLIP,
cần cả `openclip_frame_keys.parquet`. Sau khi thay đổi embedding hoặc metadata,
build lại Milvus trước khi khởi động service. Giữ các file Parquet/vector/SQLite/
Milvus của cùng một lần build với nhau.

## 5. Code map

- `aic_pipeline/audit.py`, `paths.py`, `io.py`: audit dataset và quy ước đường dẫn/I/O.
- `sampling.py`, `keyframe_selection.py`, `siglip_retrieval.py`: sampling, chọn keyframe và visual retrieval helpers.
- `offline_search_engine.py`, `retrieval_fusion.py`, `fusion_profiles.py`: truy vấn, retrieval branches và fusion.
- `query_understanding.py`, `query_transformer.py`, `task_router.py`: parse/expand query và chọn task.
- `kis_review.py`, `qa_gemini.py`, `qa_reasoning.py`, `scene_evidence.py`: review KIS và chọn/ground QA evidence.
- `temporal_alignment.py`, `temporal_reranker.py`, `xclip_reranker.py`: helpers đánh giá/refine thứ tự thời gian; không phải tất cả đều đang được gọi từ API.
- `backend/runtime.py`, `services/`, `app.py`: khởi tạo engine và HTTP contract.
- `scripts/README.md`: nhóm script production, build, validation và utility.

## 6. Validation và dữ liệu cục bộ

`tests/` chứa unit tests. Các script `evaluate_*`, `benchmark_*`,
`optimize_fusion_profiles.py`, `select_fusion_config.py` và các builder
`build_*pseudolabels.py` dùng cho validation/tuning; chúng không thay thế
entrypoint xử lý dataset.

Dataset, checkpoint, Milvus DB, cache và scheduler state là dữ liệu cục bộ.
Trước khi dọn hoặc thay index, backup đúng artifact cần resume/serve. Không chỉ
backup vector: Parquet metadata, SQLite FTS và Milvus report/database cũng cần
được giữ đồng bộ. Không commit `.env` hoặc credential.
