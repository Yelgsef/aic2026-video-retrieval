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
[14] Model-Based Multimodal Reranking                BASELINE ĐÃ BUILD
     BGE Reranker; OpenCLIP ensemble cho KIS;
     coarse-to-fine + short-clip cho TRAKE
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

OCR dùng hai lớp chống nhiễu. Prompt yêu cầu Gemini bỏ timestamp, watermark
kênh, ticker/footer và website/hotline cố định nhưng vẫn giữ lower-third có tên
người, chức vụ, địa danh hoặc tiêu đề. Sau Batch, bộ lọc xác định tiếp tục loại
timestamp/date và watermark kênh lặp xuyên video. `ocr_text_raw` cùng lý do loại
được giữ để audit; chỉ `ocr_text` đã lọc đi vào BGE-M3, FTS5 và fuzzy index.
Không loại chỉ vì một chuỗi xuất hiện nhiều lần, vì logo/tên chương trình như
`60 giây` vẫn có thể là nội dung query hợp lệ.

Nhánh OCR độ phân giải gốc đã được bổ sung sau final-keyframe selection:

```text
Final keyframe gốc -> PP-OCRv6 tiếng Việt (GPU, sequential decode, batch 8)
  -> polygon + confidence
  -> loại timestamp/watermark
  -> confidence thấp: crop JPEG -> Gemini Batch minimal thinking
  -> merge local/Gemini -> scene document
```

Mỗi video giới hạn mặc định tối đa 80 crop Gemini, chọn theo confidence thấp
nhất. Vì vậy chi phí không tăng tuyến tính theo toàn bộ số frame.

ASR giữ `asr_text_raw`; chỉ segment có log-probability thấp, nhiều word
confidence thấp hoặc có chữ số mới được đưa vào Gemini Batch normalization.
Kết quả chuẩn hóa có proper nouns, changes, uncertainties và provenance riêng.

## 5. Multi-Modal Semantic Fusion Engine

Schema, parser và scene-document builder đã được build. Hiện nó chạy được khi
chỉ có ASR và tự bổ sung Gemini fields sau khi Batch responses xuất hiện.

```text
Mỗi semantic scene
  |
  +-- SigLIP2 SO400M image vectors
  +-- OpenCLIP ViT-L/14 image vectors (nhánh độc lập, dùng cho KIS)
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
  - tách riêng visual_query / ocr_query / asr_query
  |
  v
Stage B - Parallel Candidate Generation
  +-- SigLIP2 query-to-image vector search
  +-- OpenCLIP ViT-L/14 query-to-image search cho KIS
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
  - BGE Reranker v2 M3 cho text evidence
  - TRAKE: field-aware scene retrieval -> short-clip pooling -> temporal DP
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

Cấu hình production được chọn theo task thay vì bật mọi nhánh cùng lúc:

- KIS: RRF baseline + OpenCLIP ViT-L/14 ensemble. Trên 38 pseudo-query, MRR
  tăng `0,6562 -> 0,6647`, Recall@5 tăng `0,8158 -> 0,8421`.
- TRAKE: field-aware BGE scene retrieval + windowed coarse-to-fine decoder +
  short-clip pooling 2 giây. Trên 10 pseudo-query, event hit `0,90` và
  full-sequence accuracy `0,70`.
- QA: không tự động bật OpenCLIP/field-aware; tiếp tục dùng high-resolution và
  object-aware fallback đã validation.
- Normalized-score fusion và field-aware cho KIS đã được thử nhưng giảm MRR,
  vì vậy không bật mặc định. Các ablation được giữ dưới dạng option để đánh giá
  lại trên strong labels, tránh overfit vào subset nhỏ.

## 8. Model inventory

| Stage | Model/công cụ | Trạng thái |
|---|---|---|
| Shot detection | OmniShotCut | Đã build |
| Sampled-frame embedding | SigLIP2 Base 224 | Đã build |
| Final image embedding | SigLIP2 SO400M 384 | Đã build |
| Independent visual ensemble | OpenCLIP ViT-L/14 LAION-2B | Đã build, tự bật cho KIS |
| ASR | Whisper large-v3 | Đã build |
| OCR/caption/entities | Gemini Batch | Generator đã build, chưa chạy toàn dataset |
| Full-resolution OCR | PP-OCRv6 + conditional Gemini crops | Đã build và smoke test |
| ASR normalization | Conditional Gemini Batch | Đã build generator/parser |
| Text semantic embedding | BGE-M3 | Đã build và smoke test trên L21_V001 |
| Full-text retrieval | SQLite FTS5/BM25 | ASR baseline đã build |
| Fuzzy OCR/entity retrieval | Character n-gram + RapidFuzz | Đã build |
| Query intent/expansion | Transformer + rule fallback | Rule fallback + concept expansion đã build |
| Text reranking | BGE Reranker v2 M3 | Đã build và smoke test |
| Q&A visual answering | Hybrid top-24 + Gemini contact sheet | Đã build; cần validation đa video |
| TRAKE temporal alignment | Field-aware coarse retrieval + stride-5 short-clip + DP | Đã build, validation subset 10 |
| Optional local visual-language reranking | VLM top-N | Chưa build |
| Motion-aware temporal candidate rerank | Gated embedding-change prior | Đã build dạng ablation, chưa bật mặc định |

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

### Validation subset 10 video (2026-08-10)

Đã chạy end-to-end trên 10 video thuộc 8 nhóm, tổng 2,1 giờ video và khoảng
1,14 GB. Toàn bộ 70/70 artifact của bảy preprocessing stages hợp lệ. Kết quả:

- 648 semantic scenes và 3.744 final-frame vectors SO400M.
- 648 BGE-M3 scene vectors.
- 10 Gemini Batch jobs bằng `gemini-3.6-flash` đều thành công.
- 647/648 scene có Gemini OCR/caption/entities; một scene bị bộ lọc
  `RECITATION` và fallback sang ASR + visual.
- Không video nào bị bỏ qua khi dựng `workspace/offline_index_subset10`.

Đánh giá lại 50 pseudo-query của `L21_V001` với chín video còn lại làm hard
negatives: MRR = 0,6180; Recall@1 = 0,50; Recall@5 = 0,78; Recall@10 = 0,92;
Recall@20 = 0,98; Recall@100 = 0,98. Latency trung bình là 1,126 giây/query.
Riêng KIS có MRR = 0,6562 và Recall@5 = 0,8158; QA retrieval có MRR = 0,4972
và Recall@5 = 0,6667. Đây vẫn là pseudo-label tập trung vào một video, chưa
thay thế strong validation đa video.

QA end-to-end trên 12 pseudo-query: frame hit = 0,4167; answer match = 0,5833;
joint accuracy = 0,3333; latency trung bình = 8,534 giây. TRAKE end-to-end trên
10 query ba sự kiện: chọn đúng video = 1,00; thứ tự frame tăng đúng = 1,00;
event temporal hit = 0,4667; đúng toàn chuỗi = 0,10; latency = 4,396 giây.
Vì vậy QA evidence selection và TRAKE frame alignment vẫn là hai nhánh cần tối
ưu; chưa được xem là production-quality.

Sau tối ưu coarse-to-fine, TRAKE event hit tăng từ 0,4667 lên 0,8667 và đúng
toàn chuỗi tăng từ 0,10 lên 0,60, trong khi vẫn giữ chọn video/thứ tự ở 1,00.
QA grounding-query + hai temporal views tăng frame hit từ 0,4167 lên 0,50,
nhưng joint accuracy vẫn là 0,3333. Do đó TRAKE baseline đã được cải thiện đáng
kể; QA vẫn cần validation có nhiều accepted scene/frame ranges và reranking tốt
hơn cho các cảnh lặp lại.

QA sau đó được bổ sung frame-level SigLIP evidence reranking và high-resolution
second pass trên tối đa bốn frame của scene đã chọn. Answer match tăng lên 0,75
và joint accuracy tăng lên 0,50; frame hit giữ 0,50. Median latency là 9,381
giây, P90 10,908 giây và mean 14,512 giây do hai API outliers khoảng 40 giây.

Multi-range QA verifier đã tạo 32 accepted scene ranges cho 12 query và audit
100% range khớp artifact thật. Khi rescore cùng predictions, frame hit tăng từ
0,50 lên 0,8333 và joint accuracy tăng từ 0,50 lên 0,75. Các range này vẫn là
pseudo-label và giữ `verified=false`.

### Object-aware retrieval và QA

YOLO-World `yolov8s-worldv2` chạy offline trên final keyframes với vocabulary
được giới hạn theo AIC. Mỗi detection lưu label, confidence, bounding box chuẩn
hóa, video/frame/time/shot/scene; scene documents chỉ nhận detection từ ngưỡng
0,20 để giảm nhiễu. Smoke test `L24_V045`: 35/35 frame decode thành công, 127
raw detections trên 32 frame trong 6,558 giây warm; 5/6 scene có object metadata.

QA dùng Grounding DINO Tiny theo điều kiện: chỉ load khi query chứa vật thể cụ
thể, lấy tối đa bốn crop từ các high-resolution evidence frames, và luôn giữ
full-frame views làm fallback. Query về hướng/vị trí không dùng crop vì mất ngữ
cảnh không gian. Benchmark 12 multi-range pseudo-query sau gating: frame hit =
0,8333; answer match = 0,75; joint accuracy = 0,75; mean latency = 17,093 giây.
Grounding nhận đúng biển số `83-P3 881.43`; câu bãi biển không còn kích hoạt
nhầm. Grounding DINO chạy FP32 và dùng khoảng 668 MB VRAM riêng.

## 10. Trạng thái chạy trên dataset

- Code visual pipeline đã build tới final SO400M embeddings.
- Code Whisper ASR và semantic-scene merging đã build.
- Code YOLO-World offline object metadata và query-conditioned Grounding DINO đã build.
- Code Gemini contact-sheet/Batch JSONL đã build nhưng chưa submit toàn dataset.
- Offline hybrid pipeline đã chạy trên subset 10 video; index hoàn chỉnh nằm
  tại `workspace/offline_index_subset10`.
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
6. Chuyển object/SigLIP2/Whisper sang persistent GPU workers trên Vast.ai.
7. Chạy validation subset trước, sau đó mới chạy toàn dataset.
8. Xây Query API và giao diện tìm kiếm.

## 13. Nguyên tắc không được bỏ qua

- Không gọi weighted RRF baseline là model-based multimodal fusion.
- Không gọi FTS5 đơn thuần là Vector-Fuzzy Fusion hoàn chỉnh.
- Phải giữ riêng visual, semantic text, lexical và fuzzy evidence.
- Mọi kết quả phải truy ngược được về video, scene, frame và timestamp.
- Gemini là enrichment tùy chọn; offline retrieval phải vẫn hoạt động khi thiếu nó.
- Chỉ thay đổi model hoặc fusion weight sau khi có validation benchmark.
