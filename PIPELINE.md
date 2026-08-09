# Pipeline AIC 2026

Tài liệu này mô tả toàn bộ pipeline mục tiêu. Các bước được đánh dấu rõ là
`ĐÃ BUILD`, `BASELINE`, `ĐÃ CHUẨN BỊ` hoặc `CHƯA BUILD` để tránh nhầm giữa
kiến trúc dự kiến và phần mềm hiện đã hoạt động.

## 1. Sơ đồ tổng thể

```text
VIDEO DATASET
      |
      v
[1] Dataset Audit + Manifest                         ĐÃ BUILD
      |
      v
[2] OmniShotCut: Shot Boundary Detection             ĐÃ BUILD
      |
      v
[3] Shot-local Sampling: stride 5                    ĐÃ BUILD
      |
      v
[4] SigLIP2 Base Patch16 224                         ĐÃ BUILD
    Embed toàn bộ sampled frames
      |
      v
[5] Adaptive Clustering + Redundancy Elimination     ĐÃ BUILD
      |
      v
[6] Final Keyframe Selection                         ĐÃ BUILD
      |
      v
[7] SigLIP2 SO400M Patch14 384                       ĐÃ BUILD
    Embed final keyframes cho image-text retrieval

VIDEO DATASET
      |
      v
[8] Whisper large-v3: ASR + Word Timestamps          ĐÃ BUILD

Shot + Final Keyframes + ASR
      |
      v
[9] Semantic Scene Merging                           ĐÃ BUILD
      |
      v
[10] Contact Sheet + Gemini Batch JSONL              ĐÃ BUILD
      |
      v
[11] Gemini OCR/Caption/Structured Metadata          ĐÃ CHUẨN BỊ
     Chưa submit toàn dataset
      |
      v
[12] Multi-Modal Semantic Fusion Engine              BASELINE ĐÃ BUILD
      |
      v
[13] Multi-Stage Hybrid Retrieval                    BASELINE ĐÃ BUILD
     Vector-Fuzzy Fusion có intent-aware weights
      |
      v
[14] Model-Based Multimodal Reranking                CHƯA BUILD
      |
      v
[15] Query API + User Interface                      CHƯA BUILD
```

## 2. Visual preprocessing và keyframe selection

```text
Video
  -> OmniShotCut
  -> Danh sách technical shots
  -> Lấy mỗi frame thứ 5 tính từ đầu từng shot
  -> SigLIP2 Base embedding
  -> Adaptive clustering trong từng shot
  -> Chọn cluster medoids
  -> Loại frame trùng lặp theo cosine similarity
  -> Final keyframes
  -> SigLIP2 SO400M embedding
```

SigLIP2 Base được dùng để xử lý số lượng sampled frames lớn với chi phí thấp.
SigLIP2 SO400M chỉ chạy trên final keyframes để tạo vector retrieval chất lượng
cao hơn.

## 3. Audio và semantic scenes

```text
Video
  -> Whisper large-v3
  -> Transcript, segments, words và timestamps

Technical shots + visual similarity + ASR context
  -> Gộp các shot liên tiếp
  -> Semantic scenes, tối đa 45 giây mỗi scene
  -> Chọn tối đa 6 representative frames cho Gemini
```

## 4. Gemini enrichment

```text
Semantic scene
  -> Contact sheet tối đa 6 frame
  -> Transcript tương ứng
  -> Gemini Batch JSONL
  -> Gemini structured response
       - summary_vi
       - OCR lines và frame evidence
       - people
       - objects
       - actions
       - locations
       - events
       - search keywords
       - uncertainties
```

Gemini Batch là bước enrichment qua API, không chạy trên GPU offline. Pipeline
offline vẫn có thể tạo index từ SigLIP2 và Whisper khi chưa có kết quả Gemini.

## 5. Multi-Modal Semantic Fusion Engine

Schema, parser và scene-document builder đã được build. Hiện nó chạy được khi
chỉ có ASR và tự bổ sung Gemini fields sau khi Batch responses xuất hiện.

```text
Mỗi semantic scene
  |
  +-- SigLIP2 SO400M image vectors
  +-- Whisper ASR text + timestamps
  +-- Gemini caption
  +-- Gemini OCR
  +-- People, objects, actions, locations, events
  +-- Video, shot, frame và time metadata
  |
  v
Unified Multimodal Scene Document
  |
  +-- Visual vectors: SigLIP2 SO400M
  +-- Text semantic vector: BGE-M3
  +-- Sparse/lexical fields: ASR + OCR + caption + entities
  +-- Provenance: nguồn và frame chứng minh cho từng thông tin
```

Model và công cụ dự kiến:

- Visual embedding: `google/siglip2-so400m-patch14-384`.
- Text semantic embedding: `BAAI/bge-m3`.
- ASR: Whisper `large-v3`.
- OCR/caption/metadata: Gemini Batch.
- Full-text index: SQLite FTS5/BM25.
- Fuzzy OCR/entity matching: RapidFuzz và character n-grams.

Không trộn trực tiếp vector SigLIP2 và BGE-M3 vì chúng thuộc hai embedding
spaces khác nhau. Mỗi nhánh tìm kiếm riêng, sau đó hợp nhất score/rank.

## 6. Multi-Stage Hybrid Retrieval: Vector-Fuzzy Fusion

```text
Query tiếng Việt hoặc tiếng Anh
  |
  v
Stage A - Query Understanding
  - chuẩn hóa Unicode và chính tả
  - phân loại visual / OCR / ASR / metadata intent
  - tách people / objects / actions / locations / events
  - query expansion Việt-Anh
  - tạo các query variants
  |
  v
Stage B - Parallel Candidate Generation
  +-- SigLIP2 query-to-image vector search
  +-- BGE-M3 query-to-scene-text vector search
  +-- SQLite FTS5/BM25 cho ASR, OCR và caption
  +-- RapidFuzz + character n-gram cho OCR/tên riêng bị sai
  +-- Metadata filters
  |
  v
Stage C - Vector-Fuzzy Fusion
  - intent-aware weighted Reciprocal Rank Fusion
  - exact OCR/entity match bonus
  - fuzzy-match score
  - max keyframe score + top-N mean theo scene
  - temporal deduplication
  |
  v
Stage D - Model-Based Reranking
  - BGE Reranker v2 M3 cho query-scene text
  - kết hợp visual, semantic, lexical và fuzzy evidence
  - optional VLM rerank top-N khi có GPU mạnh
  |
  v
Stage E - Result Diversification
  - loại frame/scene gần trùng
  - giữ coverage theo video và thời gian
  |
  v
Video ID + timestamp + keyframe + evidence
```

### Trọng số theo intent

Query hình ảnh sẽ ưu tiên SigLIP2. Query nhắc đến chữ hoặc biển báo sẽ tăng
trọng số OCR/FTS/Fuzzy. Query nhắc đến lời nói sẽ tăng trọng số ASR và BGE-M3.
Các trọng số cuối cùng phải được chọn bằng validation set, không cố định theo
cảm tính.

## 7. Những gì hybrid baseline hiện đã làm

```text
Query
  +-- SigLIP2 SO400M exact vector search trên final keyframes
  +-- SQLite FTS5 search trên ASR
  |
  v
Weighted Reciprocal Rank Fusion
  |
  v
Max keyframe score theo semantic scene
  |
  v
Video + scene + timestamp + representative frame IDs
```

Baseline hiện đã có BGE-M3 dense vectors, SQLite FTS5, character-trigram fuzzy
candidates, RapidFuzz reranking, rule-based query intent, Việt-Anh concept
expansion và BGE Reranker v2 M3. Nó chưa có learned intent model hoặc một model
dịch/query-expansion tổng quát. Vì vậy đây vẫn chưa phải chất lượng retrieval
cuối cùng.

## 8. Model inventory

| Stage | Model/công cụ | Trạng thái |
|---|---|---|
| Shot detection | OmniShotCut | Đã build |
| Sampled-frame embedding | SigLIP2 Base 224 | Đã build |
| Final image embedding | SigLIP2 SO400M 384 | Đã build |
| ASR | Whisper large-v3 | Đã build |
| OCR/caption/entities | Gemini Batch | Generator đã build, chưa chạy toàn dataset |
| Text semantic embedding | BGE-M3 | Đã build và smoke test trên L21_V001 |
| Full-text retrieval | SQLite FTS5/BM25 | ASR baseline đã build |
| Fuzzy OCR/entity retrieval | Character n-gram + RapidFuzz | Đã build |
| Query intent/expansion | Transformer + rule fallback | Rule fallback + concept expansion đã build |
| Text reranking | BGE Reranker v2 M3 | Đã build và smoke test |
| Q&A visual answering | Hybrid top-24 + Gemini contact sheet | Đã build; cần validation đa video |
| TRAKE temporal alignment | SigLIP2 Base stride-5 + dynamic programming | Đã build baseline |
| Optional local visual-language reranking | VLM top-N | Chưa build |

## 8.1. Persistent query service

Query service giữ SigLIP2 SO400M, BGE-M3 và BGE Reranker v2 M3 trong bộ nhớ
thay vì reload theo từng query:

```powershell
uv run --project environments/siglip2 python scripts/serve_offline_query.py
```

API cục bộ:

```text
GET  http://127.0.0.1:8765/health
POST http://127.0.0.1:8765/route
POST http://127.0.0.1:8765/query
POST http://127.0.0.1:8765/answer
POST http://127.0.0.1:8765/trake
Body: {"query": "...", "top_k": 100, "rerank_k": 30}
```

Smoke benchmark RTX 4060 trên index L21_V001:

- Load ba model: khoảng 21,5 giây một lần.
- Warm visual query: khoảng 0,09 giây.
- Warm ASR query có reranker: khoảng 0,27–0,48 giây.
- HTTP query đầu tiên: khoảng 1,8 giây do CUDA warm-up.
- VRAM giữ model: khoảng 4,58 GB.

Service chỉ bind `127.0.0.1`, không mở ra mạng ngoài.

Query response hiện có `task_plan`, candidate results và tối đa 100 submissions
cho KIS. Router phân biệt `KIS`, `QA` và `TRAKE`. Q&A submissions được đánh dấu
`requires_vqa=true`; TRAKE được đánh dấu `requires_temporal_alignment=true` để
không vô tình nộp output chưa hoàn chỉnh.

Q&A dùng hybrid retrieval lấy ít nhất 24 candidate scenes, giải mã một
representative frame cho mỗi candidate, ghép contact sheet 4 cột và gọi Gemini
để vừa chọn evidence frame vừa trả lời ngắn. Lời gọi mạng có timeout 45 giây,
retry giới hạn và fallback model để không treo query service vô hạn.

TRAKE tách query thành các sự kiện có thứ tự, truy hồi từng sự kiện độc lập để
chọn video có coverage đầy đủ, sau đó dùng SigLIP2 Base trên toàn bộ sampled
frames stride-5 và dynamic programming để tìm chuỗi frame tăng nghiêm ngặt.
Vì dùng stride 5, sai lệch chỉ do lưới sampling tối đa là 4 frame. Smoke test
trên `L21_V001` trả chuỗi `3278 -> 10102 -> 14286` đúng thứ tự. Đây là baseline
temporal alignment, chưa phải kết quả đã tune trên TRAKE validation thật.

## 9. Strong validation review

Tạo contact sheets và CSV để gắn strong labels:

```powershell
uv run --project environments/siglip2 python scripts/build_validation_review_pack.py `
  --video-id L21_V001 --scenes-per-video 10
```

Mở `workspace/validation/review_pack/index.html` để xem và điền
`annotations.csv`. Weak labels lấy từ mốc 30 giây chỉ dùng smoke test, không
dùng để chọn model hoặc fusion weights.

Khi chưa thể review thủ công, Gemini auto-labeler có thể tạo pseudo-label có
frame evidence và resume cache. Các nhãn này luôn giữ `verified=false`:

```powershell
uv run --project environments/gemini python scripts/auto_label_validation_pack.py `
  --pack workspace/validation/review_pack_50
```

Pack hiện tại có 50 pseudo-label trên L21_V001: 38 KIS, 12 Q&A, confidence
trung bình 0,948. Đây chưa phải strong validation đa video.

Đánh giá retrieval trên 50 pseudo-label này: Recall@1 = 0,14; Recall@5 = 0,58;
Recall@10 = 0,78; Recall@20 = 0,84; Recall@50 = 0,88; Recall@100 = 1,00;
warm latency trung bình khoảng 0,253 giây/query. Các số này chỉ dùng phát hiện
lỗi pipeline vì nhãn do Gemini tạo và toàn bộ mẫu chỉ thuộc một video.

## 10. Trạng thái chạy trên dataset

- Code visual pipeline đã build tới final SO400M embeddings.
- Code Whisper ASR và semantic-scene merging đã build.
- Code Gemini contact-sheet/Batch JSONL đã build nhưng chưa submit toàn dataset.
- Offline hybrid baseline đã smoke test trên `L21_V001`.
- BGE-M3 checkpoint đã được lưu tại `models/bge-m3`.
- BGE Reranker v2 M3 đã được lưu tại `models/bge-reranker-v2-m3`.
- Chưa chạy production trên toàn bộ khoảng 1.600 video.
- Scheduler có SQLite state, resume, retry, shard và artifact validation.
- Scheduler hiện vẫn cần persistent GPU workers để tránh reload model theo video.

## 11. Kết quả smoke test L21_V001

- Thời lượng khoảng 21 phút.
- 352 technical shots.
- 115 semantic scenes.
- Giảm khoảng 67,3% số đơn vị cần gửi Gemini.
- 462 representative frames dùng chuẩn bị contact sheets.
- 115 Gemini Batch requests, khoảng 15,1 MB JSONL.
- Offline index: 115 scenes, 704 SO400M keyframe vectors, dimension 1152.
- BGE-M3 index: 115 scene-text vectors, dimension 1024.

## 12. Thứ tự build tiếp theo

1. Submit một Gemini validation batch và chạy parser đã có trên responses thật.
2. Xây strong validation labels; weak 30-second labels hiện không đủ để tuning.
3. Xây transformer query intent và model-based query expansion Việt-Anh.
4. Chạy Recall@K, MRR, temporal hit rate và latency benchmark đầy đủ.
5. Chuyển SigLIP2/Whisper sang persistent GPU workers.
6. Tích hợp toàn bộ stage vào resumable production scheduler.
7. Chạy validation subset trước, sau đó mới chạy toàn dataset.
8. Xây Query API và giao diện tìm kiếm.

## 13. Nguyên tắc không được bỏ qua

- Không gọi weighted RRF baseline là model-based multimodal fusion.
- Không gọi FTS5 đơn thuần là Vector-Fuzzy Fusion hoàn chỉnh.
- Phải giữ riêng visual, semantic text, lexical và fuzzy evidence.
- Mọi kết quả phải truy ngược được về video, scene, frame và timestamp.
- Gemini là enrichment tùy chọn; offline retrieval phải vẫn hoạt động khi thiếu nó.
- Chỉ thay đổi model hoặc fusion weight sau khi có validation benchmark.
