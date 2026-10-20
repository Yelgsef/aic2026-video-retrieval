# AIC2026 — Multimodal Video Retrieval

Hệ thống xử lý video offline và cung cấp backend/frontend cho ba dạng truy vấn:

- KIS: tìm đoạn/frame phù hợp với mô tả.
- QA: tìm bằng chứng rồi trả lời bằng Gemini qua 9Router.
- TRAKE: tìm chuỗi sự kiện theo đúng thứ tự thời gian.

## Tài liệu

- [`PIPELINE.md`](PIPELINE.md): thiết kế pipeline, model, fusion và đường xử lý KIS/QA/TRAKE.
- [`docs/DATA_FLOW_AND_STORAGE.md`](docs/DATA_FLOW_AND_STORAGE.md): sơ đồ chia theo flow và artifact được tạo/đọc ở từng bước.
- [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md): bản đồ toàn bộ repository, artifact và cách vận hành/resume.
- [`scripts/README.md`](scripts/README.md): phân loại các script production, build, validation và utility.
- [`PSEUDO_LABEL_FORMAT.md`](PSEUDO_LABEL_FORMAT.md): định dạng nhãn đánh giá KIS/QA/TRAKE.
- [`docs/QWEN_BACKEND.md`](docs/QWEN_BACKEND.md): cấu hình Query Transformer local.

## Kiến trúc

~~~text
Video -> OmniShotCut/sampling -> SigLIP2/ASR/OCR/YOLO/semantic scenes
      -> Gemini enrichment -> BGE-M3/SigLIP2/OpenCLIP embeddings
      -> offline_index_all_873 -> Qwen query expansion (tùy chọn)
      -> retrieval/reranking/temporal alignment -> KIS/QA/TRAKE
~~~

Qwen chỉ phân tích query lúc truy vấn; bật Qwen không yêu cầu embed lại.

## Entrypoint chính

- `scripts/run_flow_a_scheduler.py`: chạy và resume riêng các stage của Flow A.
- `scripts/run_dataset_scheduler.py`: chạy và resume các stage offline rộng hơn theo video.
- `scripts/run_full_dataset_supervisor.py`: điều phối nhiều GPU và enrichment qua 9Router.
- `scripts/run_gemini_subset_supervisor.py`: supervisor enrichment/index cho một tập video. Tên file được giữ để tương thích lệnh cũ; luồng hiện tại gọi 9Router, không phải Gemini Batch API.
- `scripts/serve_offline_query.py`: backend KIS, QA và TRAKE.
- `scripts/query_offline_index.py`: truy vấn index bằng CLI.

Các script còn lại thuộc nhóm build artifact hoặc validation. Xem `scripts/README.md` trước khi chạy riêng từng stage.

## Trạng thái model production

| Chức năng | Model/backend |
|---|---|
| Shot boundary | OmniShotCut |
| Candidate visual encoder | SigLIP2 Base 224 |
| Final visual encoder | SigLIP2 SO400M 384 |
| Visual ensemble | OpenCLIP ViT-L/14 |
| Speech-to-text | Faster-Whisper large-v3 |
| OCR local | PP-OCR |
| Object evidence | YOLO-World; GroundingDINO dùng có điều kiện trong QA |
| Scene/text vectors | BGE-M3 |
| Query understanding | Qwen2.5 1.5B local, tùy chọn |
| VLM enrichment/QA | Gemini qua 9Router; direct Gemini API là backend lựa chọn |

BEiT-3 và pipeline CLIP cũ đã được loại khỏi repository vì không còn thuộc luồng production.

## Cài đặt

Yêu cầu: Python 3.10+, uv, FFmpeg, NVIDIA CUDA (nếu chạy GPU), Git.

~~~bash
git clone --branch pipeline/gemini-qa-spatial-grounding --recurse-submodules <REPO_URL> aic2026
cd aic2026
git submodule sync --recursive
git submodule update --init --recursive
uv sync --project environments/omnishotcut
uv sync --project environments/siglip2
uv sync --project environments/asr
uv sync --project environments/ocr
uv sync --project environments/objects
uv sync --project environments/gemini
~~~

~~~bash
nvidia-smi
uv run --project environments/siglip2 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
~~~

## Global offline index

Index đầy đủ: workspace/offline_index_all_873/

Các file quan trọng: scenes.parquet, frames.parquet, scene_embeddings.npy,
frame_embeddings.npy, frame_embeddings_openclip.npy, text_embeddings_bge_m3*.npy,
search.sqlite3, bge_m3_report.json và openclip_report.json.

~~~bash
uv run --project environments/siglip2 python -c "import pandas as pd; p='workspace/offline_index_all_873'; f=pd.read_parquet(p+'/frames.parquet'); s=pd.read_parquet(p+'/scenes.parquet'); print('frame videos=',f.video_id.nunique()); print('scene videos=',s.video_id.nunique()); print('frames=',len(f),'scenes=',len(s))"
cat workspace/offline_index_all_873/openclip_report.json
~~~

Index hoàn chỉnh có khoảng 873 video và 338862 frame OpenCLIP.

## Qwen local

Logic nằm ở aic_pipeline/query_transformer.py. Đặt model tại:

~~~text
models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
~~~

Chạy llama-server:

~~~bash
llama-server -m models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8081 --n-gpu-layers 99 --ctx-size 2048
curl http://127.0.0.1:8081/health
~~~

Service sẽ warm-up Qwen nếu model và llama-server khả dụng. Cache query nằm ở
workspace/offline_index_all_873/query_transformer_cache.sqlite3. Đổi model hoặc
prompt chỉ cần xóa cache; không cần embed lại.

## 9Router/Gemini

Tạo .env và không commit:

~~~env
NINEROUTER_CHAT_URL=http://localhost:20128/v1/chat/completions
NINEROUTER_MODEL=ag/gemini-3-flash-agent
NINEROUTER_KEY=<key-neu-router-yeu-cau>
GEMINI_MODE=9router
GEMINI_API_KEY=<key-neu-dung-gemini-api>
GEMINI_API_MODEL=gemini-2.5-flash
~~~

~~~bash
set -a; source .env; set +a
curl -sS "http://localhost:20128/v1/models"
~~~

9Router dùng cho Gemini enrichment và QA online; BGE-M3, SigLIP2, OpenCLIP
và Qwen có thể chạy local.

QA hỗ trợ hai mode: `9router` (mặc định) và `gemini_api`. Mode được chọn theo
từng request, không cần restart hoặc embed lại.

## Backend HTTP

~~~bash
HF_HOME=workspace/hf-cache HF_HUB_OFFLINE=1 uv run --project environments/siglip2 python scripts/serve_offline_query.py --index workspace/offline_index_all_873 --port 8765
~~~

Endpoint: GET /health, POST /route, POST /query (KIS), POST /answer (QA),
POST /trake (TRAKE), GET /frame và GET /video.

~~~bash
curl http://127.0.0.1:8765/health
~~~

## Frontend

Frontend ở web/index.html và gọi backend localhost. Mở http://127.0.0.1:8765.
Chọn AUTO, KIS, QA hoặc TRAKE. Sau khi đổi code/index, dừng server bằng Ctrl+C
rồi chạy lại.

## Query KIS

~~~bash
HF_HOME=workspace/hf-cache HF_HUB_OFFLINE=1 uv run --project environments/siglip2 python scripts/query_offline_index.py --index workspace/offline_index_all_873 --top-k 100 --candidate-k 500 --no-openclip "Một chiếc máy cày trên đồng lúa xanh"
~~~

Bỏ --no-openclip khi muốn dùng OpenCLIP fusion.

~~~bash
curl -X POST http://127.0.0.1:8765/query -H "Content-Type: application/json" -d '{"query":"Một chiếc máy cày trên đồng lúa xanh","top_k":20,"query_transformer":true}'
~~~

## Query QA

~~~bash
curl -X POST http://127.0.0.1:8765/answer -H "Content-Type: application/json" -d '{"query":"Dòng chữ nào trong cuốn sách được in đỏ?","vqa_candidates":12,"query_transformer":true}'
~~~

QA cần model/credential Gemini qua 9Router. UNCERTAIN thường do chữ quá nhỏ
hoặc bằng chứng hình ảnh yếu, không nhất thiết phải embed lại.

Gọi trực tiếp Gemini API thay vì 9Router:

~~~bash
curl -X POST http://127.0.0.1:8765/answer -H "Content-Type: application/json" -d '{"query":"Dòng chữ nào trong cuốn sách được in đỏ?","gemini_mode":"gemini_api","vqa_candidates":12,"query_transformer":true}'
~~~

Request không có `gemini_mode` dùng `GEMINI_MODE` trong `.env`, mặc định là
`9router`. Mode `gemini_api` cần `GEMINI_API_KEY`; mode `9router` cần các biến
`NINEROUTER_*` tương ứng.

## Query TRAKE

~~~bash
curl -X POST http://127.0.0.1:8765/trake -H "Content-Type: application/json" -d '{"query":"Đầu tiên có xe máy cày trên đồng lúa, sau đó có các vận động viên xe đạp","query_transformer":true}'
~~~

TRAKE yêu cầu ít nhất hai sự kiện có thứ tự.

## Google Cloud Storage

~~~bash
gcloud auth login --no-launch-browser
gcloud config set project aic2026-505115
gcloud storage ls gs://aic2026-505115-vertex/
~~~

Restore index:

~~~bash
mkdir -p workspace/offline_index_all_873
gcloud storage rsync --recursive gs://aic2026-505115-vertex/backups/aic2026/workspace/offline_index_all_873 workspace/offline_index_all_873
~~~

Restore Qwen:

~~~bash
mkdir -p models/qwen2.5-1.5b-instruct-gguf
gcloud storage cp gs://aic2026-505115-vertex/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf models/qwen2.5-1.5b-instruct-gguf/
~~~

Sync index/model:

~~~bash
gcloud storage rsync --recursive workspace/offline_index_all_873 gs://aic2026-505115-vertex/backups/aic2026/workspace/offline_index_all_873
gcloud storage cp models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf gs://aic2026-505115-vertex/models/qwen/
~~~

Chỉ destroy instance sau khi các file .npy, .parquet, .sqlite3 và
openclip_report.json xuất hiện trên GCS.

## Build lại index

Chỉ build lại khi thêm dữ liệu hoặc thay embedding model. Bật Qwen không cần:

~~~bash
uv run --project environments/siglip2 python scripts/build_offline_hybrid_index.py --output workspace/offline_index_all_873
uv run --project environments/siglip2 python scripts/build_bge_m3_scene_index.py --documents workspace/offline_index_all_873/scenes.parquet --output workspace/offline_index_all_873 --batch-size 16
CUDA_VISIBLE_DEVICES=1 uv run --project environments/siglip2 python scripts/build_openclip_frame_index.py --index workspace/offline_index_all_873 --batch-size 128 --force
~~~

OpenCLIP cần video gốc trong archive/video/.

## Chạy lâu trên Vast.ai

~~~bash
tmux new -s aic
~~~

Detach không dừng bằng Ctrl-B rồi D; vào lại bằng:

~~~bash
tmux attach -t aic
~~~

Không đóng SSH nếu process còn TTY=pts/... và không nằm trong tmux.

## Milvus (tùy chọn)

Milvus chỉ lưu vector và metadata; vẫn giữ .npy, .parquet, .sqlite3 làm dữ liệu
gốc. Nên tạo collection riêng: BGE-M3 dim 1024, SigLIP2 dim 1152 và OpenCLIP
dim 768. Mỗi vector cần video_id, scene_index/frame_id, timestamp và ID.

## Bảo mật và bàn giao

- Không commit .env, API key hoặc token.
- Cấp cộng tác viên quyền GCS tối thiểu, thường Storage Object Viewer.
- Offline index đủ cho retrieval; QA cần thêm code, evidence video/frame,
  Gemini/9Router và Qwen nếu muốn query expansion.
- Không chia sẻ Google account; thêm email cộng tác viên qua IAM project.
