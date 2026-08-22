# Pipeline AIC 2026 hiện tại

Tài liệu này là nguồn mô tả duy nhất cho pipeline đang được triển khai. Các tài liệu pipeline cũ đã được hợp nhất vào đây.

## 1. Luồng tổng thể

```text
Video dataset
  -> audit và manifest
  -> OmniShotCut shot segmentation
  -> sampling stride 5 theo shot
  -> temporal chunks 67 frames, overlap 10
  -> SigLIP2 Base candidate features
  -> 1-2 medoid mỗi chunk + redundancy elimination
  -> coverage audit 103 frames + fallback ở cửa sổ trống
  -> SigLIP2 SO400M final visual embeddings
  -> ASR + OCR + object evidence + 9Router scene enrichment
  -> semantic scenes và OCR-to-frame mapping
  -> multimodal offline index
  -> query transformer
  -> hybrid retrieval và fusion
  -> task-specific QA / KIS / TRAKE output
```

## 2. Video preprocessing và keyframes

### Shot segmentation

- Model: OmniShotCut.
- Mục đích: tìm shot boundary để sampling không vượt qua chuyển cảnh.
- Output: `workspace/shots/<video_id>/`.

### Sampling và temporal coverage

- Sample mỗi 5 frame, tính lại từ `shot.start_frame`.
- Chia temporal chunks dài 67 frame, overlap 10 frame.
- Chọn 1-2 medoid trong mỗi chunk tùy độ đa dạng.
- Loại frame dư thừa bằng cosine similarity và khoảng cách thời gian.
- Audit theo cửa sổ 103 frame; chỉ chèn fallback khi cửa sổ không có frame.

Thiết kế này tránh hai cực đoan: chỉ giữ một medoid khiến mất chi tiết ngắn, còn giữ mọi anchor tạo quá nhiều frame gần giống nhau.

### Visual models

- SigLIP2 Base 224: embedding ứng viên nhanh.
- SigLIP2 SO400M 384: embedding final và retriever hình ảnh chính.
- OpenCLIP ViT-L/14: nhánh ensemble độc lập, chỉ bật khi index có vectors tương ứng.

## 3. Audio, OCR và object evidence

### ASR

- Faster-Whisper large-v3, CUDA FP16 khi có GPU.
- 9Router normalization chỉ dùng có điều kiện cho transcript nhiễu; không thay thế transcript gốc.

### OCR

- PP-OCR cho local/full-resolution OCR.
- 9Router crop OCR là fallback cho chữ khó hoặc nhỏ.
- Watermark, timestamp và lower-third lặp lại được lọc trước indexing.
- Index giữ `ocr_evidence_json`, ánh xạ từng chuỗi OCR về đúng `frame_id`. Đây là bắt buộc cho câu hỏi đọc biển, tên cây, mã số và tiêu đề sách.

### Object understanding

- YOLO-World/object metadata cho object count, vị trí chuẩn hóa và quan hệ chủ thể-vật thể.
- GroundingDINO crop được dùng có điều kiện trong QA khi cần nhìn vật thể nhỏ.

## 4. Semantic scene documents

Mỗi scene gồm:

- biên frame/time và shot IDs;
- representative frame IDs;
- visual summary;
- ASR exact và ASR context;
- OCR text và OCR-to-frame evidence;
- người, vật thể, hành động, địa điểm, sự kiện;
- object count, normalized position và relations;
- từ khóa tìm kiếm và provenance.

Scene lân cận được dùng làm temporal context khi câu hỏi và bằng chứng nằm hai phía của một boundary. Không merge shot ngắn một cách mù quáng.

## 5. Offline index

Local storage:

- Parquet cho metadata;
- NumPy arrays cho embeddings;
- SQLite FTS5 cho lexical/fuzzy search;
- cache SQLite cho Query Transformer; semantic text vectors dùng BGE-M3 local.

Production trên Vast.ai có thể đẩy vectors và scalar metadata sang Milvus. `video_id`, `scene_index`, `frame_id`, timestamps và đường dẫn media vẫn là khóa chuẩn.

Build index cục bộ:

```cmd
uv run --project environments/siglip2 python scripts/build_offline_hybrid_index.py --output workspace/offline_index
uv run --project environments/siglip2 python scripts/build_bge_m3_scene_index.py --documents workspace/offline_index/scenes.parquet --output workspace/offline_index --batch-size 16
uv run --project environments/siglip2 python scripts/build_openclip_frame_index.py --index workspace/offline_index
```

## 6. Query understanding

Query Transformer local chuẩn hóa tiếng Việt/Anh, phát hiện task, answer target, object/spatial constraints và sinh query variants. Nó không trực tiếp xem video và không được phép tạo đáp án.

Các answer type quan trọng gồm visible text, clock/time, count, color, object, motion và named entity. Với `named_entity` như “cây nào”, pipeline ưu tiên scene có tên thực thể thật trong OCR thay vì để VLM suy đoán bằng kiến thức ngoài.

## 7. Hybrid retrieval và fusion

Các nhánh retrieval:

1. SigLIP2 visual-text similarity.
2. OpenCLIP ensemble nếu khả dụng.
3. BGE-M3 trên retrieval text, summary, ASR và OCR/entities.
4. SQLite FTS5/BM25-like lexical retrieval.
5. Character n-grams và RapidFuzz.
6. Object/spatial constraints.
7. BGE reranker v2 M3 cross-encoder.

Score được robust-normalize và hợp nhất bằng RRF/normalized fusion theo intent. Candidate expansion phải có điều kiện: mở rộng recall rồi lọc theo loại bằng chứng, không gửi thẳng top 100 scene nhiễu cho VLM.

## 8. Task paths

### KIS

Query variants -> hybrid retrieval -> fusion -> reranker -> danh sách frame/video xếp hạng. Output hỗ trợ top 100 submission.

Các số/mã xuất hiện nguyên vẹn trong query có một exact-metadata branch để tránh
visual similarity nhầm `126` thành `152`. API cũng có tầng dense short-clip moment
refinement cho ablation (`moment_refinement=true`), nhưng mặc định tắt: benchmark
hiện tại cho thấy embedding ảnh tĩnh chưa đủ ổn định để tự đếm “lần thứ hai”.

KIS còn trả một Review Pack riêng gồm top 3 video, top scene theo từng video và
tối đa 25 dense stride-5 frame mỗi scene. Frontend cho phép phát đoạn ở 0.25×/0.5×,
dịch ±0.2 giây và chọn/copy exact frame. Review Pack không chạy trong QA hoặc TRAKE
và không tự ghi đè kết quả retrieval.

### QA

Hybrid retrieval -> evidence-aware candidate selection -> contact sheet -> 9Router QA -> optional full-resolution refinement -> answer variants và frame evidence.

Nguyên tắc QA:

- Không dùng `video_id` ground truth để lọc trong benchmark end-to-end.
- Exact text dùng nhiều frame phân bố theo thời gian và scene lân cận.
- Tên thực thể phải được grounded trong OCR/ASR/caption.
- Refinement không được thay đáp án có confidence bằng hoặc cao hơn bằng một chi tiết lân cận không đúng ngữ cảnh.
- Nếu index không có OpenCLIP, server tự fallback sang SigLIP2/BGE-M3.

### TRAKE

Tách query thành chuỗi sự kiện -> retrieve candidate scenes -> short-clip temporal evidence -> dynamic/global sequence alignment -> trả ordered frame IDs và video ID.

## 9. Scheduler cho toàn dataset

`scripts/run_dataset_scheduler.py` quản lý stage theo video, ghi trạng thái để có thể tiếp tục sau khi tắt máy. Các stage chính:

```text
sampling
siglip-base
siglip-so400m-final
asr-large-v3
semantic-scenes
objects-yoloworld
contact-sheet prepare -> parallel resumable 9Router enrichment -> parse
fullres-ocr
ASR/OCR normalization khi cần
global index build
```

Không rebuild toàn dataset chỉ để thêm một video đã có final embeddings và semantic scenes; build một index mới từ artifact có sẵn thường chỉ mất vài phút.

## 10. Validation

Đánh giá tách riêng:

- retrieval: MRR, R@5, R@10, R@20, R@100;
- QA: video match, frame-range match và answer accuracy/multi-answer match;
- TRAKE: đúng video và đúng toàn bộ thứ tự sự kiện;
- latency: retrieval, decode/contact sheet, 9Router first pass và refinement.

R@K chỉ cho biết scene đúng có nằm trong K kết quả đầu hay không. Tăng K có thể tăng recall nhưng cũng tăng nhiễu; QA chỉ nên mở rộng có kiểm soát và dựa trên evidence type.

## 11. Chạy dịch vụ

```cmd
uv run --project environments/siglip2 python scripts/serve_offline_query.py --index workspace/offline_index_all_873
```

Mở `http://127.0.0.1:8765`. Cần restart server sau khi đổi code hoặc index vì dữ liệu/model được nạp và giữ trong RAM.
