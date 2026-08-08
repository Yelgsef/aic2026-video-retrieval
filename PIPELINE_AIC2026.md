# Pipeline đề xuất cho vòng Sơ tuyển AIC 2026

## 1. Mục tiêu thiết kế

Hệ thống phải tối ưu trực tiếp cho ba dạng truy vấn của AIC 2026:

1. **Textual KIS**: tìm đúng video và một `frame_id` nằm trong đoạn đáp án.
2. **Q&A**: tìm đúng video, đúng đoạn frame và trả lời đúng về mặt ngữ nghĩa.
3. **TRAKE**: tìm đúng video và căn chỉnh chính xác từng semantic keyframe theo thứ tự thời gian.

Điểm cuối của mỗi truy vấn được tính từ `R@1`, `R@5`, `R@20`, `R@50` và `R@100`. Vì vậy hệ thống phải đồng thời tối ưu:

- độ chính xác ở top đầu;
- recall trong top 100;
- độ chính xác định vị frame;
- sự đa dạng hợp lý giữa các hypothesis.

> Pipeline LMSKE trong paper `2401.04962` tối ưu video summarization, fidelity và compression ratio. Đối với AIC, keyframe extraction chỉ nên là tầng lập chỉ mục thô; đáp án cuối phải được tinh chỉnh trên video gốc.

## 2. Hiện trạng dataset

Dataset tại `C:\Users\ADMIN\AIC\archive` hiện có:

- 961 video;
- khoảng 176.816 keyframe;
- 873 file CLIP feature và 873 file map-keyframe;
- object metadata cho gần như toàn bộ keyframe;
- 3 transcript JSON và 1 SRT;
- khoảng 88 video chưa có CLIP feature/map-keyframe tương ứng, cần kiểm tra và xử lý bổ sung.

## 3. Kiến trúc tổng thể

```text
Video gốc
  ↓
Shot detection
  ↓
Coarse frame sampling + mandatory anchors
  ↓
Multi-modal feature extraction
  ├─ BEiT-3 visual embedding
  ├─ provided CLIP ViT-B/32 embedding
  ├─ transcript/ASR
  ├─ OCR
  ├─ object labels
  ├─ metadata
  └─ motion/quality signals
  ↓
Adaptive clustering có bảo toàn recall
  ↓
Redundancy elimination có điều kiện
  ↓
Frame / shot / video indexes
  ↓
Query parsing + translation + expansion
  ↓
Multi-channel coarse retrieval
  ↓
Video và shot reranking
  ↓
Dense temporal localization trên video gốc
  ↓
Task-specific inference
  ├─ Textual KIS
  ├─ Q&A
  └─ TRAKE
  ↓
Diversity-aware top-100 ranking
```

## 4. Offline preprocessing

### 4.1. Xây dựng manifest chuẩn

Mỗi video cần có một record thống nhất:

```json
{
  "video_id": "L21_V001",
  "video_path": "...",
  "fps": 30.0,
  "frame_count": 37860,
  "duration": 1262.0,
  "is_vfr": false,
  "map_keyframe_path": "...",
  "clip_feature_path": "...",
  "metadata_path": "...",
  "transcript_path": "..."
}
```

Yêu cầu:

- không suy ra `frame_id` chỉ từ `timestamp × FPS` nếu video có variable frame rate;
- lưu cả original frame index và PTS;
- kiểm tra đủ dữ liệu cho toàn bộ 961 video;
- đưa các video thiếu feature vào hàng đợi preprocessing riêng.

### 4.2. Shot detection

Sử dụng:

- OmniShotCut làm detector chính nếu checkpoint/code hoạt động ổn định;
- TransNetV2 làm fallback hoặc model ensemble;
- có thể thêm detector histogram đơn giản để phát hiện lỗi nghiêm trọng.

Hậu xử lý boundary:

- gộp hai boundary nếu cách nhau khoảng 2–4 frame;
- kiểm tra các shot quá ngắn;
- lưu `shot_id`, `start_frame`, `end_frame`, `start_pts`, `end_pts`;
- giữ một vùng bảo vệ vài frame quanh mỗi boundary.

Shot detection chỉ dùng để chia miền tìm kiếm. Không sử dụng shot boundary như đáp án semantic keyframe cuối cùng.

### 4.3. Sampling hai tầng

Không dùng duy nhất quy tắc “mỗi frame thứ 10”. Đề xuất:

| Độ dài shot | Sampling coarse |
|---|---:|
| ≤ 30 frame | stride 1–2 |
| 31–150 frame | stride 5 |
| > 150 frame | stride 10 |

Luôn bổ sung các mandatory anchor:

- shot start;
- shot middle;
- shot end;
- frame quanh boundary;
- local maximum của motion hoặc semantic change;
- frame bắt đầu xuất hiện OCR mới.

Tập frame stride 10 vẫn được giữ trong một **dense shadow index**, kể cả khi một số frame không được chọn làm keyframe cuối.

### 4.4. Multi-modal feature extraction

Mỗi sampled frame nên có:

- BEiT-3 image embedding từ checkpoint đã fine-tune cho image–text retrieval;
- CLIP ViT-B/32 feature được BTC cung cấp;
- OCR text và bounding box;
- object labels, confidence và bounding box;
- blur score, black-frame score, exposure score;
- motion score hoặc optical-flow magnitude;
- `video_id`, `shot_id`, original `frame_id`, timestamp và PTS.

Tất cả visual embedding phải được L2-normalize trước khi dùng cosine similarity.

Tạo ba cấp representation:

1. **Frame embedding**: từng sampled frame.
2. **Shot embedding**: weighted/top-k pooling các frame trong shot.
3. **Video embedding**: pooling các shot nổi bật, transcript và metadata.

Không average toàn bộ video một cách đơn giản, vì sự kiện ngắn dễ bị chìm trong các cảnh dài.

### 4.5. Transcript và OCR

Transcript hiện là khoảng trống lớn nhất của dataset.

Thứ tự ưu tiên:

1. lấy caption YouTube từ metadata nếu còn khả dụng;
2. chạy ASR offline cho video không có caption;
3. lưu segment text kèm `start_time`, `end_time` và frame range;
4. lập chỉ mục cả tiếng Việt lẫn bản dịch/embedding đa ngôn ngữ.

OCR nên được chạy trên sampled frame, sau đó:

- normalize Unicode;
- loại text lặp theo thời gian;
- giữ tên người, địa điểm, con số, tiêu đề và bảng biểu;
- liên kết mỗi đoạn OCR với frame range.

### 4.6. Adaptive clustering

Giữ ý tưởng LMSKE nhưng sửa để bảo toàn recall:

- cluster riêng trong từng shot;
- dùng cosine distance trên normalized embedding;
- `k_max = min(ceil(sqrt(n)), k_cap)`;
- dùng sampled silhouette hoặc MiniBatch K-Means cho shot dài;
- chọn medoid là frame thật gần cluster center;
- sắp xếp candidate theo thời gian.

Ngoài medoid, luôn giữ:

- shot start/middle/end;
- frame có OCR mới;
- frame có motion/semantic-change cực đại;
- frame quanh transition.

### 4.7. Redundancy elimination

Không loại frame chỉ dựa trên HSV histogram threshold `0.8` như LMSKE.

Chỉ xem hai frame là dư thừa khi đồng thời thỏa phần lớn điều kiện:

```text
embedding cosine similarity > 0.97–0.985
AND temporal distance nhỏ
AND OCR gần giống nhau
AND object set gần giống nhau
AND không thuộc vùng boundary quan trọng
```

Quy tắc bảo vệ:

- không loại frame giữa hai shot khác nhau;
- không loại frame có OCR mới;
- không loại mandatory anchor;
- giữ tối thiểu 3 frame đại diện cho shot có độ dài đáng kể;
- không xóa frame khỏi dense shadow index.

## 5. Xây dựng index

Nên có các index độc lập:

```text
frame_visual_index
shot_visual_index
video_visual_index
transcript_text_index
ocr_text_index
object_inverted_index
metadata_text_index
```

FAISS có thể được dùng cho dense visual/text embedding. BM25 hoặc inverted index dùng cho OCR, object label, transcript và metadata.

Mỗi kết quả trong index phải truy ngược được tới:

```text
video_id → shot_id → frame_id → timestamp/PTS → source path
```

## 6. Query processing

Với mỗi query tiếng Việt, tạo:

- query gốc;
- bản dịch tiếng Anh;
- phiên bản rút gọn;
- 2–5 paraphrase;
- danh sách chủ thể, thuộc tính, hành động, vật thể và bối cảnh;
- keyword dành riêng cho OCR/transcript;
- temporal constraint;
- negative constraint nếu có.

Schema đề xuất:

```json
{
  "task": "KIS|QA|TRAKE",
  "subject": [],
  "attributes": [],
  "actions": [],
  "objects": [],
  "scene": [],
  "ocr_terms": [],
  "speech_terms": [],
  "temporal_constraints": [],
  "events": []
}
```

Điểm retrieval ban đầu có thể dùng:

```text
score =
  0.40 × BEiT3_visual
+ 0.20 × provided_CLIP
+ 0.18 × transcript
+ 0.10 × OCR
+ 0.07 × objects
+ 0.05 × metadata
```

Các trọng số này chỉ là baseline và phải được tuning trên validation set.

## 7. Textual KIS pipeline

```text
Query expansion
→ retrieve top 100–300 shots
→ aggregate thành video candidates
→ rerank top 20–50 videos
→ chọn temporal windows tốt nhất
→ dense localization trên video gốc
→ tạo và xếp hạng tối đa 100 đáp án
```

Dense localization đề xuất:

```text
coarse hit tại frame f
→ đọc [f - 5 giây, f + 5 giây] với stride 2
→ chọn vùng score cao nhất
→ đọc lại vùng ±1 giây với stride 1
→ chọn frame ở giữa score plateau
```

Chọn trung tâm của một vùng score cao và ổn định thường an toàn hơn chọn frame có score cực đại duy nhất.

## 8. Q&A pipeline

```text
Retrieve event
→ lấy window ±5–15 giây
→ tập hợp transcript + OCR + contact sheet frames
→ sinh câu trả lời từ nhiều nguồn
→ consistency check
→ normalize answer
```

Các nguồn trả lời:

- VLM trên contact sheet 4–12 frame;
- VLM trên frame tốt nhất;
- transcript gần timestamp;
- OCR gần timestamp;
- object detection/tracking cho câu hỏi đếm.

Normalize câu trả lời:

- chuyển số chữ và số digit về cùng dạng;
- chuẩn hóa màu sắc, tên riêng và địa danh;
- chỉ trả lời ngắn gọn, không thêm giải thích nếu không cần;
- lưu confidence và evidence của từng nguồn.

Đối với câu hỏi đếm, cần tracking qua nhiều frame để tránh đếm trùng hoặc bỏ sót do che khuất.

## 9. TRAKE pipeline

### 9.1. Video retrieval

Cho chuỗi event `E1 ... En`, tính video score từ:

- coverage của toàn bộ event;
- global query similarity;
- temporal-order consistency;
- penalty nếu event bị đảo thứ tự hoặc cách nhau phi lý.

Không tìm từng event độc lập trên toàn dataset rồi ghép tùy ý.

### 9.2. Temporal alignment

Với mỗi top video candidate:

1. tạo similarity matrix `S[event, frame]`;
2. chạy dynamic programming/Viterbi;
3. bắt buộc `f1 < f2 < ... < fn`;
4. áp dụng constraint khoảng cách thời gian;
5. refine từng vùng ở stride 1;
6. sinh nhiều nghiệm alignment có score cao để phục vụ top-100.

Mục tiêu:

```text
maximize Σ S(Ej, fj)
subject to f1 < f2 < ... < fn
```

Với semantic moment là điểm chuyển trạng thái, ví dụ “chân vừa rời mặt đất” hoặc “vừa chạm đệm”, visual embedding chỉ nên dùng để tìm vùng. Frame cuối cần được xác định bằng:

- pose estimation;
- object/person tracking;
- optical flow;
- so sánh predicate giữa frame `t-1` và `t`.

## 10. Xếp hạng top 100

Vì Final Score dùng các mốc top khác nhau:

- **Top 1**: hypothesis có xác suất đúng cao nhất.
- **Top 5**: đa dạng video nếu vẫn chưa chắc video.
- **Top 20**: đa dạng video, shot và temporal window.
- **Top 50–100**: mở rộng temporal alternatives quanh ứng viên mạnh.

Không lấp các vị trí đầu bằng nhiều frame gần như giống nhau của cùng một video.

Khi đã chắc đúng video:

- KIS/Q&A: trải các frame candidates quanh vùng dự đoán;
- TRAKE: tạo các alignment alternatives bằng DP và local frame shifts.

Có thể dùng MMR hoặc quota theo video/shot để cân bằng relevance và diversity.

## 11. Validation nội bộ

Tạo ít nhất 100–300 query validation, bao gồm:

- cảnh tổng quát;
- hành động ngắn;
- OCR/text trên màn hình;
- nội dung lời nói;
- tên người/địa điểm;
- Q&A đếm, màu sắc, tên riêng;
- TRAKE gồm 3–5 event.

Metric cần đo:

```text
Video Recall@1/5/20/50/100
Frame hit rate tại ±5/±10/±30 frame
Exact ground-truth interval hit rate
Q&A answer accuracy
TRAKE per-event accuracy
TRAKE full-sequence accuracy
AIC Final Score mô phỏng
Latency/query
Index size
```

Chọn cấu hình theo AIC Final Score mô phỏng, không chọn theo F1/Fidelity/Compression Ratio của LMSKE.

## 12. Ablation experiments

Chạy các thí nghiệm sau trên cùng validation set:

1. Provided CLIP baseline.
2. BEiT-3 baseline.
3. CLIP + BEiT-3.
4. Thêm transcript.
5. Thêm OCR.
6. Thêm object index.
7. Có/không adaptive clustering.
8. Có/không redundancy elimination.
9. Stride 5 so với stride 10.
10. Có/không dense frame refinement.
11. Independent TRAKE alignment so với dynamic programming.

Chỉ giữ một thành phần nếu nó cải thiện metric AIC hoặc giảm đáng kể latency mà không làm mất recall.

## 13. Thứ tự triển khai

- [ ] Xây manifest đầy đủ cho 961 video.
- [ ] Xác định và xử lý khoảng 88 video thiếu feature/map.
- [ ] Chuẩn hóa frame ID, timestamp và PTS.
- [ ] Chạy shot detection và boundary post-processing.
- [ ] Tạo coarse sampling cùng mandatory anchors.
- [ ] Hoàn thiện transcript/ASR.
- [ ] Trích OCR.
- [ ] Trích BEiT-3 embeddings.
- [ ] Xây frame/shot/video indexes.
- [ ] Hoàn thiện Textual KIS retrieval.
- [ ] Thêm dense temporal localization ở stride 1.
- [ ] Hoàn thiện Q&A evidence aggregation.
- [ ] Hoàn thiện TRAKE dynamic programming.
- [ ] Tạo validation query và scorer theo công thức AIC.
- [ ] Chạy ablation và tune fusion weights.
- [ ] Thêm adaptive clustering/redundancy sau khi có baseline recall.
- [ ] Tối ưu top-100 ranking và diversity.

## 14. Cấu hình baseline khởi đầu

```yaml
shot_detection:
  primary: omnishotcut
  fallback: transnetv2
  boundary_merge_tolerance_frames: 3

sampling:
  short_shot_stride: 2
  medium_shot_stride: 5
  long_shot_stride: 10
  preserve_boundary_frames: true
  dense_refinement_stride: 1

features:
  visual_models:
    - beit3_retrieval
    - provided_clip_vit_b32
  transcript: true
  ocr: true
  objects: true
  metadata: true

redundancy:
  cosine_threshold: 0.98
  require_temporal_proximity: true
  preserve_ocr_changes: true
  preserve_boundaries: true
  minimum_frames_per_nontrivial_shot: 3

retrieval:
  coarse_shots: 300
  rerank_videos: 50
  dense_window_seconds: 5
  max_answers: 100

trake:
  ordered_alignment: dynamic_programming
  dense_refinement: true
  generate_alignment_alternatives: true
```

## 15. Kết luận

Pipeline đề xuất không loại bỏ ý tưởng của LMSKE mà sử dụng nó đúng vai trò:

- OmniShotCut/TransNetV2 để chia shot;
- BEiT-3 và CLIP để tạo semantic index;
- adaptive clustering để tạo compact retrieval index;
- redundancy elimination để giảm chi phí nhưng không xóa dense shadow index;
- dense frame refinement để đáp ứng đoạn ground truth dưới 10 frame;
- transcript, OCR, object và metadata để tăng recall;
- dynamic programming để căn chỉnh chuỗi TRAKE;
- task-aware ranking để tối ưu trực tiếp `R@1/5/20/50/100`.

Ưu tiên triển khai đầu tiên nên là **multi-modal retrieval + dense frame refinement + validation theo công thức AIC**. Adaptive clustering và redundancy elimination chỉ nên được tối ưu sau khi đã đo được ảnh hưởng của chúng lên recall.

## Tài liệu tham khảo

- Kailong Tan et al., [Large Model based Sequential Keyframe Extraction for Video Summarization](https://arxiv.org/abs/2401.04962), 2024.
- Wenhui Wang et al., [Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_as_a_Foreign_Language_BEiT_Pretraining_for_Vision_and_CVPR_2023_paper), CVPR 2023.
- Boyang Wang et al., [OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer](https://arxiv.org/abs/2604.24762), 2026.
- Tài liệu “Thông tin vòng Sơ tuyển AIC 2026” do Ban tổ chức cung cấp.
