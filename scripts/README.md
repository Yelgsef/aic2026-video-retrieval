# Script map

Đây là bản đồ ngắn để tránh chạy nhầm script thử nghiệm thay cho pipeline production.

## Production orchestration

- `run_flow_a_scheduler.py`: scheduler riêng cho Flow A, dùng `workspace/scheduler/flow_a_jobs.sqlite3`.
- `run_dataset_scheduler.py`: scheduler rộng hơn có trạng thái/resume cho các stage local theo video.
- `run_full_dataset_supervisor.py`: chạy scheduler nhiều GPU, enrichment và build index.
- `run_gemini_subset_supervisor.py`: enrichment 9Router và build index cho subset. Tên `gemini` chỉ được giữ để tương thích lệnh cũ.
- `serve_offline_query.py`: compatibility entrypoint to `backend/` FastAPI, serving `frontend/` and Milvus Lite queries.
- `query_offline_index.py`: CLI retrieval.

## Offline stages

- `audit_dataset.py`, `run_omnishotcut.py`, `build_sampling_manifest.py`
- `build_siglip2_sample_index.py`, `select_keyframes.py`, `build_siglip2_final_index.py`
- `transcribe_faster_whisper.py`, `extract_fullres_ocr.py`, `extract_yoloworld_objects.py`
- `gemini_batch_scenes.py` chuẩn bị scene request; `run_9router_enrichment.py` thực hiện remote VLM call.
- `prepare_gemini_*` và `parse_gemini_*` giữ tên artifact/schema cũ nhưng được dùng cho cả response 9Router.
- `build_unified_scene_documents.py`, `build_offline_hybrid_index.py`, `build_bge_m3_scene_index.py`, `build_openclip_frame_index.py`

## Validation và benchmark

Các file `benchmark_*`, `evaluate_*`, `build_*pseudolabels.py`, `build_validation_*`, `optimize_fusion_profiles.py`, `select_fusion_config.py` và `run_qa_regression_direct.py` chỉ đánh giá/tinh chỉnh pipeline; chúng không phải entrypoint để embed toàn dataset.

## Utilities

- `export_keyframe_jpgs.py`: xuất final keyframes để kiểm tra bằng mắt.
- `validate_keyframe_coverage.py`: audit coverage của final keyframes.
- `test_9router_request.py`: smoke test endpoint/model/credential 9Router.
- `smoke_query_transformer.py`: smoke test Qwen Query Transformer.
