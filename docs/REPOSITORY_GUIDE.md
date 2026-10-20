# Repository guide

Tài liệu này mô tả repository đang dùng, không mô tả các baseline đã bị loại bỏ.

## 1. Thư mục

| Đường dẫn | Vai trò | Có commit? |
|---|---|---|
| `aic_pipeline/` | Thư viện Python dùng chung cho retrieval, fusion, QA, KIS và TRAKE | Có |
| `scripts/` | Entrypoint chạy stage, scheduler, serving và benchmark | Có |
| `environments/` | Các môi trường `uv` độc lập theo workload | Có, trừ `.venv` |
| `tests/` | Unit test cho sampling, fusion, grounding, routing và temporal alignment | Có |
| `frontend/` | Frontend tĩnh cho KIS/QA/TRAKE | Có |
| `backend/` | FastAPI API, retrieval runtime và media/QA services | Có |
| `third_party/OmniShotCut` | Git submodule của OmniShotCut | Có dưới dạng submodule |
| `archive/video/` | Video dataset gốc | Không |
| `models/`, `weights/` | Checkpoint tải về | Không |
| `workspace/` | Manifest, embeddings, scene documents, index và scheduler state | Không |
| `exports/` | Kết quả xuất để xem/nộp | Không |

Không xóa `workspace/scheduler/jobs.sqlite3` nếu muốn resume. Không commit `.env`, dataset, model hoặc index.

## 2. Module runtime

### Video và keyframe

- `audit.py`, `paths.py`, `io.py`: tìm dataset, chuẩn hóa ID và I/O.
- `sampling.py`: stride theo shot và temporal sampling.
- `keyframe_selection.py`: chunk, medoid, redundancy elimination và coverage fallback.
- `siglip_retrieval.py`: tải/encode SigLIP2 và cosine retrieval.

### Evidence và query

- `ocr_filter.py`: loại watermark, timestamp và lower-third nhiễu.
- `object_spatial.py`, `object_grounding.py`: count, vị trí, quan hệ và crop grounding.
- `scene_evidence.py`: chọn frame evidence thích ứng theo câu hỏi.
- `query_understanding.py`, `query_transformer.py`: intent, answer type, constraints và query variants.
- `router_chat.py`: client OpenAI-compatible cho 9Router.

### Retrieval và task logic

- `offline_search_engine.py`: nạp index và chạy các nhánh retrieval.
- `retrieval_fusion.py`, `fusion_profiles.py`: normalize, fusion và profile theo intent.
- `gemini_text_retrieval.py`: vector text backend lựa chọn tương thích Gemini.
- `qa_gemini.py`, `qa_reasoning.py`: evidence selection, VQA và answer variants.
- `kis_review.py`: review pack và exact-frame refinement chỉ dành cho KIS.
- `temporal_alignment.py`, `temporal_reranker.py`, `xclip_reranker.py`: xếp chuỗi sự kiện TRAKE.
- `task_router.py`, `submission.py`: route task và định dạng output.

## 3. Môi trường uv

| Project | Workload |
|---|---|
| `environments/omnishotcut` | shot segmentation |
| `environments/siglip2` | embeddings, index, retrieval và service |
| `environments/asr` | Faster-Whisper |
| `environments/ocr` | PaddleOCR/PP-OCR |
| `environments/objects` | YOLO-World/object extraction |
| `environments/gemini` | chuẩn bị/parse VLM jobs và gọi 9Router/direct Gemini |

Khởi tạo bằng `uv sync --project environments/<name>`. Không dùng một `.venv` chung vì CUDA/Paddle/Python constraints khác nhau.

## 4. Offline lifecycle

```text
audit dataset
  -> OmniShotCut
  -> stride-5 sampling
  -> SigLIP2 Base candidate embeddings
  -> temporal chunks + medoids + redundancy + coverage audit
  -> SigLIP2 SO400M final embeddings
  -> ASR / OCR / objects / remote scene enrichment
  -> unified scene documents
  -> SigLIP2 + OpenCLIP + BGE-M3 + lexical indexes
```

Scheduler lưu một job cho mỗi `(video_id, stage, config_hash)`. Artifact hợp lệ được reconcile thành `completed`; job stale/retry có thể tiếp tục mà không xử lý lại phần đã xong.

Chạy riêng Flow A cho một video (cần tạo `videos.parquet` bằng audit trước):

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

Chạy hai GPU nên đặt mỗi worker trong một cửa sổ `tmux` riêng, cùng `--shard-count 2` và khác `--shard-index 0/1`.

## 5. Artifact contract

Các đường dẫn chính dưới `workspace/`:

- `manifests/videos.parquet`: danh sách video đã audit.
- `shots/<video_id>/`: shot boundaries OmniShotCut.
- `sampling/<video_id>/`: sampled-frame manifest.
- `features/<encoder_id>/<video_id>/`: visual embeddings theo encoder.
- `keyframes/<video_id>/`: final keyframe metadata.
- `asr/<video_id>/`: transcript và timestamps.
- `ocr_fullres/<video_id>/`: OCR lines/crops và frame provenance.
- `objects/<video_id>/`: object detections/spatial metadata.
- `semantic_scenes/<video_id>/`: scene-level multimodal metadata.
- `scheduler/jobs.sqlite3`: trạng thái resume.
- `offline_index_all_873/`: index serving đầy đủ.

Index serving cần `scenes.parquet`, `frames.parquet`, các vector nguồn `.npy`, `search.sqlite3` và `milvus_lite.db` cùng `milvus_report.json`. Milvus Lite phục vụ ANN vector retrieval; SQLite FTS/fuzzy vẫn phục vụ lexical retrieval và metadata lookup. `frame_id`, `video_id`, timestamps và scene index phải luôn được giữ cùng vector. Sau khi rebuild vector, chạy lại `scripts/build_milvus_index.py` trước khi khởi động query engine.

## 6. Retrieval online

1. Router xác định KIS, QA hoặc TRAKE.
2. Query Transformer tùy chọn tạo variants nhưng không tự trả lời.
3. Engine lấy candidate từ visual, semantic text, OCR/ASR lexical và object/spatial evidence.
4. Score từng nhánh được normalize trước khi weighted fusion/RRF.
5. Reranker chỉ được trộn khi đã tải thành công; nếu không, hệ thống dùng raw fused score và renormalize trọng số còn hoạt động.
6. KIS trả ranked candidates/review data; QA frontend gửi evidence đã chọn tới VLM endpoint; TRAKE xếp candidates của từng event theo frame order. Dense `TemporalAligner` hiện chưa được gọi từ request handler.

Không cộng trực tiếp cosine từ hai model có phân phối khác nhau nếu chưa normalize. Thiếu OpenCLIP hoặc reranker phải fallback, không gán điểm 0 rồi vẫn giữ nguyên trọng số.

## 7. Service contract

Chạy server:

```bash
uv run --project environments/siglip2 python scripts/serve_offline_query.py \
  --index workspace/offline_index_all_873 --host 0.0.0.0 --port 8765
```

Endpoint:

- `GET /api/health`: model/index availability.
- `POST /api/query`, `POST /api/query/stream`: retrieval KIS/QA/TRAKE.
- `POST /api/qa/answer`, `POST /api/qa/models`: QA với frame do FE chọn.
- `GET /api/frame/{video_id}/{frame_id}`: trích frame qua manifest video path.
- `GET /videos/{video_id}.mp4`, `GET /api/videos/{video_id}/metadata`: media.

Backend mới nằm trong `backend/`, frontend trong `frontend/`; FastAPI phục vụ
frontend cùng origin. `video_id` được ánh xạ sang `video_path` từ manifest,
đường dẫn nguồn không được gửi tới browser. Restart service sau khi đổi code
hoặc index.

## 8. Remote VLM configuration

`.env` tối thiểu:

```env
NINEROUTER_CHAT_URL=http://localhost:20128/v1/chat/completions
NINEROUTER_MODEL=ag/gemini-3-flash-agent
NINEROUTER_KEY=
GEMINI_MODE=9router
GEMINI_API_KEY=
GEMINI_API_MODEL=gemini-2.5-flash
```

9Router là luồng enrichment mặc định. Một số tên file/stage còn chứa `gemini`
để giữ tương thích artifact và scheduler cũ. QA trong frontend Flow C gửi ảnh
và câu hỏi qua backend tới Google AI Studio hoặc 9Router theo provider.

## 9. Validation

- Retrieval: MRR, R@5, R@10, R@20, R@100.
- QA: video/frame-range match, answer exact/normalized/multi-answer accuracy.
- TRAKE: đúng video, event recall và full-sequence accuracy.
- Latency: retrieval, media decode, VLM first pass và refinement.

Pseudo-label chỉ là tập đánh giá; không được đưa expected `video_id/frame_id` vào retrieval input. Tinh chỉnh fusion trên một split và báo cáo trên split khác để hạn chế overfit.

## 10. Deploy và backup

Code lấy từ Git; dataset/model/index lấy từ GCS hoặc persistent volume. Trước khi destroy Vast instance:

1. Dừng worker sạch hoặc xác nhận SQLite đã flush.
2. Sync `workspace/offline_index_all_873` và artifact cần resume lên GCS.
3. Kiểm tra số file/kích thước ở đích.
4. Không chỉ backup vector: phải kèm Parquet metadata và SQLite.

Các key/API credential không được ghi vào command history, log, commit hoặc ảnh chụp màn hình.

## 11. Troubleshooting nhanh

- `LocalEntryNotFoundError` cùng `local_files_only=True`: model chưa có trong Hugging Face cache; tải trước rồi mới bật offline mode.
- `401 API key required`: `.env` chưa được process hiện tại load hoặc key không được gửi trong Authorization header.
- Scheduler đứng ở `pending`: dependency trước chưa completed hoặc artifact validation thất bại.
- Kết quả QA đúng video nhưng sai chữ: kiểm tra full-resolution OCR/crop và frame provenance trước khi tăng top-k.
- KIS sai “lần thứ hai/đầu tiên”: dùng dense review/refinement theo thời gian; không thay logic QA/TRAKE.
- Query chậm: tách latency retrieval khỏi VLM call; tăng dataset chủ yếu ảnh hưởng ANN/search nhẹ, còn decode và remote VLM thường chiếm phần lớn.
