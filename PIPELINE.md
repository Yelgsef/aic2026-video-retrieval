# Pipeline AIC 2026

## 1. Pipeline offline

```text
Video dataset
      |
      v
OmniShotCut
Phát hiện ranh giới shot
      |
      v
Sampling theo từng shot
Lấy mỗi frame thứ 5 tính từ đầu shot
      |
      v
SigLIP2 Base Patch16 224
Trích xuất embedding của sampled frames
      |
      v
Adaptive Clustering
Gom các frame có nội dung tương tự
      |
      v
Redundancy Elimination
Loại frame trùng lặp
      |
      v
Final Keyframes
      |
      v
SigLIP2 SO400M Patch14 384
Tạo embedding chất lượng cao cho final keyframes
```

## 2. Nhánh âm thanh

```text
Video dataset
      |
      v
Whisper large-v3
      |
      v
Transcript + timestamp
```

## 3. Semantic scene và Gemini

```text
Shot boundaries + Final keyframes + Transcript
                       |
                       v
              Semantic Scene Merging
              Gộp các shot liên tiếp
                       |
                       v
           Tối đa 6 frame đại diện mỗi scene
                       |
                       v
              Contact Sheet + ASR Context
                       |
                       v
                 Gemini Batch
                       |
                       v
       OCR + Caption + People + Objects
       Actions + Locations + Events + Keywords
```

## 4. Search pipeline

```text
SigLIP2 embeddings + Whisper ASR + Gemini metadata
                         |
                         v
              Multimodal Document Fusion
                         |
                         v
                Hybrid Retrieval Index
                 /          |          \
                v           v           v
         Vector Search  Text Search  Metadata Filter
                \           |           /
                 \          |          /
                          v
                   Candidate Fusion
                          |
                          v
                  Multimodal Reranking
                          |
                          v
             Video + Timestamp + Keyframe
```

## 5. Query flow

```text
Query tiếng Việt hoặc tiếng Anh
                 |
                 v
Chuẩn hóa query và nhận diện:
- người
- vật thể
- hành động
- địa điểm
- sự kiện
- chữ xuất hiện trong video
                 |
                 v
Tìm đồng thời trên:
- embedding hình ảnh SigLIP2
- transcript Whisper
- OCR và caption Gemini
- metadata video/scene
                 |
                 v
Fusion + Reranking
                 |
                 v
Top kết quả kèm video, timestamp và keyframe
```

## 6. Trạng thái hiện tại

### Đã hoàn thành

- OmniShotCut.
- Sampling stride 5 theo từng shot.
- SigLIP2 Base embedding.
- Adaptive clustering.
- Redundancy elimination.
- Final keyframe selection.
- SigLIP2 SO400M final embedding.
- Whisper large-v3 ASR pipeline.
- Semantic scene merging.
- Gemini contact-sheet generator.
- Gemini Batch JSONL generator.
- Scheduler có resume sau khi tắt máy.

### Đã chuẩn bị nhưng chưa chạy toàn dataset

- Whisper large-v3 cho toàn bộ video.
- Semantic scene merging cho toàn bộ video.
- Gemini Batch submission.

### Baseline offline đã xây dựng

- Global scene corpus từ semantic scenes, ASR và SO400M embeddings.
- SQLite FTS5 index cho transcript.
- Exact SigLIP2 keyframe search và max aggregation theo scene.
- Reciprocal-rank fusion giữa visual search và text search.
- Query CLI trả video, scene, timestamp và representative frame IDs.

### Chưa hoàn thành

- Parser kết quả Gemini Batch.
- Multimodal document fusion.
- Query attention và query expansion tiếng Việt/Anh.
- Learned candidate fusion và cross-encoder reranking.
- Giao diện tìm kiếm hoàn chỉnh.

## 7. Kết quả thử nghiệm L21_V001

- Video dài khoảng 21 phút.
- 352 technical shots.
- 115 semantic scenes sau khi gộp.
- Giảm khoảng 67,3% số scene cần gửi Gemini.
- 462 frame đại diện duy nhất.
- 115 Gemini Batch requests.
- JSONL khoảng 15,1 MB.

## 8. Bước tiếp theo

1. Submit Gemini Batch cho một nhóm validation nhỏ.
2. Đánh giá OCR, caption và metadata trả về.
3. Hoàn thiện parser kết quả Gemini.
4. Xây multimodal document fusion.
5. Xây hybrid retrieval index.
6. Xây và đánh giá query engine.
7. Chạy production scheduler trên toàn bộ dataset.
