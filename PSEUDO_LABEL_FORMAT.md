# Format pseudo-query và pseudo-label

File pseudo-label hiện tại:

```text
C:\Users\ADMIN\AIC\workspace\validation\review_pack_50\annotations_autolabeled.csv
```

Mỗi dòng trong CSV chứa cả câu truy vấn và ground truth tương ứng.

## 1. Schema CSV cho KIS và QA

```csv
query_id,query_type,difficulty,query_text_vi,answer,video_id,scene_id,start_frame,end_frame_exclusive,start_time,end_time,accepted_frame_id,verified,notes
```

| Trường | Ý nghĩa |
|---|---|
| `query_id` | ID duy nhất, ví dụ `strong_0051` |
| `query_type` | `KIS` hoặc `QA` |
| `difficulty` | `easy`, `medium` hoặc `hard` |
| `query_text_vi` | Câu truy vấn tiếng Việt tự nhiên |
| `answer` | Câu trả lời ngắn cho QA; để trống với KIS |
| `video_id` | Video chứa đáp án |
| `scene_id` | Scene ground truth, ví dụ `L21_V001:scene:000007` |
| `start_frame` | Frame đầu scene, inclusive |
| `end_frame_exclusive` | Frame kết thúc, exclusive |
| `start_time` | Thời gian bắt đầu scene, đơn vị giây |
| `end_time` | Thời gian kết thúc scene, đơn vị giây |
| `accepted_frame_id` | Frame đáp án đại diện |
| `verified` | `False` nếu do AI tạo; `True` sau khi người kiểm tra |
| `notes` | Nguồn nhãn, confidence và evidence |

## 2. Ví dụ KIS

```csv
strong_0051,KIS,medium,"Cảnh một người đàn ông mặc áo trắng đang phát biểu trước nhiều micro",,L22_V001,L22_V001:scene:000042,12450,12820,415.0,427.33,12605,False,"AI pseudo-label; confidence=0.94; evidence=Người đàn ông mặc áo trắng đứng trước cụm micro."
```

Pseudo-query:

```text
Cảnh một người đàn ông mặc áo trắng đang phát biểu trước nhiều micro
```

Pseudo-label tương ứng:

```json
{
  "video_id": "L22_V001",
  "scene_id": "L22_V001:scene:000042",
  "accepted_frame_id": 12605,
  "valid_frame_range": [12450, 12820]
}
```

## 3. Ví dụ QA

```csv
strong_0052,QA,medium,"Chiếc xe xuất hiện trong cảnh có màu gì?","Màu đỏ",L25_V058,L25_V058:scene:000018,8200,8510,273.33,283.67,8375,False,"AI pseudo-label; confidence=0.91; evidence=Frame cho thấy một chiếc xe màu đỏ."
```

Pseudo-query:

```text
Chiếc xe xuất hiện trong cảnh có màu gì?
```

Pseudo-label tương ứng:

```json
{
  "video_id": "L25_V058",
  "scene_id": "L25_V058:scene:000018",
  "accepted_frame_id": 8375,
  "answer": "Màu đỏ"
}
```

## 4. Schema JSONL cho TRAKE

TRAKE gồm nhiều sự kiện có thứ tự nên lưu bằng JSONL thay vì nhét nhiều frame
vào một ô CSV. Mỗi dòng là một JSON object hoàn chỉnh:

```json
{
  "query_id": "strong_trake_0001",
  "query_type": "TRAKE",
  "difficulty": "hard",
  "query_text_vi": "Tìm đoạn gồm: (1) người dẫn đứng trong trường quay; (2) cảnh sạt lở đất; (3) một người đàn ông phát biểu.",
  "video_id": "L21_V001",
  "events": [
    {
      "event_index": 0,
      "description_vi": "Người dẫn đứng trong trường quay",
      "accepted_frame_id": 3278,
      "start_frame": 3200,
      "end_frame_exclusive": 3400
    },
    {
      "event_index": 1,
      "description_vi": "Cảnh sạt lở đất",
      "accepted_frame_id": 10102,
      "start_frame": 10020,
      "end_frame_exclusive": 10220
    },
    {
      "event_index": 2,
      "description_vi": "Một người đàn ông phát biểu",
      "accepted_frame_id": 14286,
      "start_frame": 14200,
      "end_frame_exclusive": 14400
    }
  ],
  "verified": false,
  "confidence": 0.9,
  "notes": "AI pseudo-label; ordered temporal events"
}
```

Các `accepted_frame_id` phải tăng theo `event_index` và cùng thuộc một video.

## 4.1. Multi-range JSONL cho QA có cảnh lặp

Khi cùng evidence xuất hiện ở nhiều scene, giữ nhiều accepted ranges thay vì
ép evaluator chỉ chấp nhận scene ban đầu:

```json
{
  "query_id": "strong_0004",
  "query_type": "QA",
  "query_text_vi": "Nam MC đứng phía bên trái mặc áo sơ mi màu gì?",
  "answer": "Màu xanh dương",
  "accepted_ranges": [
    {
      "video_id": "L21_V001",
      "scene_id": "L21_V001:scene:000007",
      "start_frame": 921,
      "end_frame_exclusive": 1887,
      "source": "original_pseudo_label"
    },
    {
      "video_id": "L21_V001",
      "scene_id": "L21_V001:scene:000064",
      "start_frame": 22707,
      "end_frame_exclusive": 23163,
      "source": "gemini_equivalent_evidence"
    }
  ],
  "verified": false,
  "confidence": 0.95
}
```

Range bổ sung phải khớp một semantic scene thật và chỉ được chấp nhận khi hình
ảnh hỗ trợ cả mô tả câu hỏi lẫn reference answer.

## 5. Quy tắc chất lượng

- Query phải mô tả chi tiết thật sự nhìn thấy hoặc được transcript hỗ trợ.
- Không suy đoán danh tính, địa điểm hay sự kiện khi evidence không rõ.
- Không dùng chính `video_id`, `scene_id`, timestamp hoặc frame ID trong câu query.
- KIS phải có ít nhất một accepted frame hợp lệ.
- QA phải trả lời được từ hình ảnh hoặc transcript của scene đã gắn nhãn.
- TRAKE phải có ít nhất hai sự kiện và đúng thứ tự thời gian.
- Query không được trùng hoặc chỉ thay vài từ so với query cũ.
- Query `hard` nên có hard negatives tương tự ở video khác.
- Nhãn do AI tạo luôn để `verified=False`.
- `notes` phải ghi confidence và evidence ngắn gọn.

## 6. Phân bố difficulty gợi ý

- `easy`: một đối tượng hoặc hành động nổi bật, ít cảnh tương tự.
- `medium`: kết hợp người, vật thể, hành động hoặc bối cảnh.
- `hard`: OCR/tên riêng, chi tiết nhỏ, quan hệ giữa nhiều đối tượng, cảnh gần
  giống nhau giữa các video hoặc chuỗi sự kiện TRAKE.

Phân bố khởi đầu phù hợp:

```text
30% easy
40% medium
30% hard
```

## 7. Mẫu yêu cầu tạo thêm pseudo-label

```text
Hãy tạo thêm 150 pseudo-query từ 10 video trong subset:
- 90 KIS
- 40 QA
- 20 TRAKE
- 30% easy, 40% medium, 30% hard
- Không lặp lại 50 query cũ
- KIS/QA xuất thành CSV theo PSEUDO_LABEL_FORMAT.md
- TRAKE xuất thành JSONL theo PSEUDO_LABEL_FORMAT.md
- Giữ verified=False
- Ghi confidence và evidence cho từng nhãn
- Ưu tiên OCR, tên riêng, hành động nhỏ và hard negatives giữa các video
- Chạy kiểm tra scene/frame range trước khi lưu
```

## 8. File đầu ra gợi ý

```text
workspace/validation/subset10/pseudo_labels_kis_qa.csv
workspace/validation/subset10/pseudo_labels_trake.jsonl
workspace/validation/subset10/pseudo_label_report.json
```

`pseudo_label_report.json` nên ghi số lượng theo query type/difficulty, số query
trùng bị loại, số frame không hợp lệ, confidence trung bình và tỷ lệ nhãn đã
được người kiểm tra.
