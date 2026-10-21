# Chạy Qwen local như backend cho AIC2026

Tài liệu này hướng dẫn chạy Qwen local qua `llama-server` với API tương thích
OpenAI. Qwen chỉ dùng để phân tích/mở rộng query; không cần embed lại dữ liệu.

## 1. Vị trí model

Pipeline tìm model theo một trong hai đường dẫn:

```text
models/qwen2.5-1.5b-instruct/
models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Bản GGUF được ưu tiên vì `query_transformer.py` có thể chạy qua
`llama-server`. Kiểm tra:

```bash
ls -lh models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Nếu chưa có model, tải bản Q4_K_M chính thức (~1.1 GB):

```bash
mkdir -p models/qwen2.5-1.5b-instruct-gguf
curl -fL \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -o models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

## 2. Cài llama.cpp

Trên Linux/Vast, dùng binary `llama-server` tương thích CUDA và đặt nó trong
`PATH`:

```bash
which llama-server
llama-server --version
```

Nếu chưa có binary, cài từ bản phát hành llama.cpp hoặc build bản CUDA. Không
dùng `llama-server` CPU nếu muốn tận dụng GPU.

Trên Windows, `query_transformer.py` cũng tự tìm binary trong:

```text
%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_*\llama-server.exe
```

## 3. Kiểm tra runtime

Flow C tự khởi động `llama-server` ở loopback trên một port tạm khi query đầu
tiên yêu cầu Qwen; không cần chạy một server riêng. Kiểm tra binary và thiết bị:

```bash
llama-server --version
llama-server --list-devices
```

Nếu binary liệt kê CUDA device, Qwen offload lên GPU. Nếu output là `(none)`,
backend chạy Qwen trên CPU với 4 threads. Trạng thái cụ thể hiển thị trong
`GET /api/health` → `query_transformer`.

## 4. Cách pipeline hiện tại sử dụng Qwen

File chính:

```text
aic_pipeline/query_transformer.py
```

Backend tự khởi động `llama-server` khi có cả binary và file GGUF. Model chỉ
được nạp khi query đầu tiên yêu cầu Qwen; vì vậy request Qwen đầu tiên có thể
mất thêm thời gian khởi động. Trường `query_transformer` trong `/api/health`
cho biết model/runtime có sẵn, thiết bị thực thi (`cuda` hoặc `cpu`) và server
đã chạy chưa. Khi binary có CUDA, backend offload model lên GPU; binary CPU-only
chạy model với tối đa bốn threads để không chiếm hết CPU của retrieval. Có thể
kiểm tra model server độc lập bằng API:

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-instruct-q4_k_m","messages":[{"role":"user","content":"Phân tích query: một người mặc áo xanh đi trên bãi biển"}],"temperature":0,"max_tokens":160}'
```

## 5. Chạy query có Qwen

`run_qa_regression_direct.py` hiện chủ động tắt query transformer để regression
ổn định. Muốn dùng Qwen, gọi `OfflineSearchEngine.search(...,
query_transformer=True)` hoặc dùng HTTP service:

```bash
HF_HOME=workspace/hf-cache HF_HUB_OFFLINE=1 \
uv run --project environments/siglip2 python scripts/serve_offline_query.py \
  --index workspace/offline_index_all_873 \
  --port 8765
```

Request tìm kiếm có thể bật/tắt transformer bằng trường JSON. Nếu yêu cầu Qwen
nhưng model/runtime không sẵn sàng, backend trả lỗi HTTP 503; nếu model lỗi khi
phân tích query, backend trả HTTP 502 thay vì âm thầm tìm bằng query thường.

```json
{
  "query": "Một người mặc áo xanh đi trên bãi biển",
  "query_transformer": true,
  "top_k": 20
}
```

Nếu muốn giữ hành vi cũ, đặt `query_transformer` thành `false`.

TRAKE chấp nhận các event `E1:`, `E2:`, `E3:` trên các dòng riêng hoặc ngăn
cách bằng dấu chấm phẩy. Backend tách event trước khi chuẩn hóa khoảng trắng,
rồi tùy chọn mở rộng từng event; thứ tự event không đổi.

## 6. Cache query

Kết quả phân tích query được lưu tại:

```text
workspace/offline_index_all_873/query_transformer_cache.sqlite3
```

Đổi model hoặc system prompt thì có thể xóa riêng cache này:

```bash
rm workspace/offline_index_all_873/query_transformer_cache.sqlite3
```

Không xóa các file embedding. Việc thêm hoặc thay Qwen **không yêu cầu
embed lại**.

## 7. Kiểm tra GPU và lỗi thường gặp

```bash
nvidia-smi
pgrep -af 'llama-server|serve_offline_query.py'
```

- `llama-server or Qwen GGUF is unavailable`: thiếu binary hoặc sai đường dẫn
  GGUF.
- `connection refused`: server chưa chạy hoặc sai port.
- VRAM không tăng: binary là bản CPU hoặc chưa đặt `--n-gpu-layers 99`.
- Query chậm: giảm `--ctx-size`, dùng GGUF Q4, hoặc giữ server chạy lâu để
  tránh startup lại cho mỗi request.

## 8. Bảo mật

Mặc định bind `127.0.0.1`, chỉ backend cùng máy truy cập được. Không bind
`0.0.0.0` trực tiếp ra Internet nếu chưa có firewall hoặc reverse proxy có
authentication.

