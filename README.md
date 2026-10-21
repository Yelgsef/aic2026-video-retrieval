# AIC2026 Video Search

Web app tìm video theo mô tả, trả lời câu hỏi dựa trên frame và tìm chuỗi sự kiện theo thứ tự. Repository chứa source code; video, model weights và search index được tải riêng.

## Chạy ứng dụng

### 1. Chuẩn bị máy

- Linux hoặc Windows WSL2, Python 3.10, Git và [`uv`](https://docs.astral.sh/uv/).
- NVIDIA GPU được khuyến nghị. CPU vẫn chạy được nhưng truy vấn và khởi động model sẽ chậm hơn.
- Bạn cần bộ runtime artifacts gồm offline index, manifest và video nguồn. Repository hiện chưa phát hành bundle/download URL công khai cho chúng.

### 2. Tải source code và cài dependency

```bash
git clone --recurse-submodules https://github.com/tien140406/AIC2026.git
cd AIC2026
uv sync --project environments/siglip2
```

### 3. Đặt index, model và video

Đặt bộ index hoàn chỉnh tại `workspace/offline_index_all_873/`. Backend kiểm tra row alignment và fingerprint giữa Milvus, Parquet và các embedding source, nên hãy tải nguyên thư mục từ cùng một lần build; không chỉ chép các file liệt kê dưới đây:

```text
scenes.parquet
frames.parquet
scene_embeddings.npy
frame_embeddings.npy
text_embeddings_bge_m3.npy
search.sqlite3
milvus_lite.db
milvus_report.json
```

Index đầy đủ còn có thể chứa các embedding `.npy` cho summary/ASR/OCR và OpenCLIP. Giữ nguyên toàn bộ các file này nếu chúng có trong bundle.

Hãy lấy một **serving bundle đã build bằng code hiện tại** từ nơi lưu artifact của dự án. Bản backup legacy trên project storage có thể thiếu SQLite/Milvus artifacts và cần được import/build lại trước khi chạy; không dùng trực tiếp như serving bundle.

Backend cũng cần các model runtime trong `models/`: `siglip2-so400m-patch14-384`, `bge-m3` và `bge-reranker-v2-m3`. Sau khi cài dependency, có thể tải model public từ Hugging Face:

```bash
uv run --project environments/siglip2 hf download google/siglip2-so400m-patch14-384 \
  --local-dir models/siglip2-so400m-patch14-384
uv run --project environments/siglip2 hf download BAAI/bge-m3 \
  --local-dir models/bge-m3
uv run --project environments/siglip2 hf download BAAI/bge-reranker-v2-m3 \
  --local-dir models/bge-reranker-v2-m3
```

Nếu index có collection `aic_frame_openclip`, backend cần thêm model `models/openclip-vit-l14-laion2b/`; lấy đúng model artifact từ kho của dự án. Nếu index không có collection này thì OpenCLIP không cần. Qwen chỉ cần nếu muốn bật Query Transformer.

Ngoài index, cần có `workspace/manifests/videos.parquet` và video nguồn tại các đường dẫn trong cột `video_path`. Lấy manifest/video từ nơi lưu dataset của dự án. Nếu chuyển sang máy khác, cập nhật `video_path` cho đúng vị trí video. Search có thể khởi động khi thiếu video, nhưng preview frame và phát video sẽ không hoạt động.

### 4. Khởi động backend và giao diện

```bash
uv run --project environments/siglip2 \
  python scripts/serve_offline_query.py \
  --index workspace/offline_index_all_873 --host 127.0.0.1 --port 8765
```

Mở <http://127.0.0.1:8765>. Kiểm tra trạng thái tại <http://127.0.0.1:8765/api/health>. Lần đầu khởi động sẽ nạp index và model vào RAM/VRAM; thời gian phụ thuộc máy. Dừng bằng `Ctrl+C`.

### 5. Dùng chức năng

- **KIS** tìm video/frame theo mô tả.
- **QA** tìm candidates trước, sau đó gửi câu hỏi cùng các frame được chọn tới provider AI. Cần API key/provider tương ứng.
- **TRAKE** tìm video có chuỗi sự kiện xuất hiện đúng thứ tự.
- **Search with Qwen** là tùy chọn; cần `llama-server` và GGUF tại `models/qwen2.5-1.5b-instruct-gguf/`. Xem [hướng dẫn Qwen](docs/QWEN_BACKEND.md).

## Tài liệu

- [Architecture](docs/architecture.md): tổng quan dự án, Flow A-F và kiến trúc frontend/backend.
- [Data flow and storage](docs/DATA_FLOW_AND_STORAGE.md): input/output và vị trí artifact theo từng stage.
- [Repository guide](docs/REPOSITORY_GUIDE.md): cấu trúc mã nguồn và entry points.
- [Pipeline details](PIPELINE.md): model, retrieval, fusion và task behavior.
- [Qwen backend](docs/QWEN_BACKEND.md): cài model và Query Transformer.
- [Scripts](scripts/README.md): phân loại lệnh xử lý, build index và validation.

## Nếu muốn tự xử lý dữ liệu và build index

README này hướng dẫn chạy web với index đã được tạo. Muốn xử lý dataset từ đầu, cần video nguồn, checkpoint của các model, các environment tương ứng và chạy scheduler/build stages. Bắt đầu từ [architecture](docs/architecture.md), sau đó xem [data flow](docs/DATA_FLOW_AND_STORAGE.md) và [repository guide](docs/REPOSITORY_GUIDE.md).
