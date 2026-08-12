from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.siglip_retrieval import l2_normalize
from aic_pipeline.text_embeddings import E5_ENCODER_ID, E5_MODEL_ID, mean_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multilingual E5 embeddings for object labels by shot.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--visual-encoder-id", required=True)
    parser.add_argument("--model-id", default=E5_MODEL_ID)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    visual_metadata_path = (
        PROJECT_ROOT / "workspace" / "retrieval" / args.visual_encoder_id
        / args.video_id / "metadata.parquet"
    )
    visual = pd.read_parquet(visual_metadata_path)
    shots = (
        visual[["video_id", "shot_id", "shot_start_frame", "shot_end_frame_exclusive"]]
        .drop_duplicates(["video_id", "shot_id"])
        .sort_values("shot_id")
        .reset_index(drop=True)
    )
    objects = pd.read_parquet(
        PROJECT_ROOT / "workspace" / "objects" / "corpus.parquet",
        columns=["video_id", "frame_id", "label_normalized", "confidence"],
        filters=[("video_id", "==", args.video_id)],
    )
    objects = objects[objects["confidence"] >= args.min_confidence]

    rows = []
    for shot in shots.itertuples(index=False):
        detections = objects[
            (objects.frame_id >= shot.shot_start_frame)
            & (objects.frame_id < shot.shot_end_frame_exclusive)
        ]
        if detections.empty:
            continue
        labels = (
            detections.groupby("label_normalized", as_index=False)["confidence"]
            .max().sort_values("confidence", ascending=False).head(20)
        )
        label_list = labels["label_normalized"].tolist()
        rows.append(
            {
                "video_id": args.video_id,
                "shot_id": int(shot.shot_id),
                "object_text": "objects: " + ", ".join(label_list),
                "object_labels": label_list,
                "object_count": len(label_list),
                "max_object_confidence": float(labels["confidence"].max()),
            }
        )
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise RuntimeError(f"No object metadata matched shots for {args.video_id}")

    output_dir = PROJECT_ROOT / "workspace" / "retrieval" / E5_ENCODER_ID / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "embeddings.npy").exists() and not args.force:
        raise FileExistsError(f"Object index exists: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=not args.allow_download)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, local_files_only=not args.allow_download
    ).to(device).eval()
    vectors = []
    passages = ["passage: " + text for text in metadata["object_text"]]
    for start in range(0, len(passages), args.batch_size):
        batch = passages[start : start + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs)
            embedding = mean_pool(output.last_hidden_state, inputs["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        vectors.append(embedding.cpu().numpy())
    embeddings = l2_normalize(np.concatenate(vectors))
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    metadata.to_parquet(output_dir / "metadata.parquet", index=False)
    config = {
        "model_id": args.model_id,
        "encoder_id": E5_ENCODER_ID,
        "video_id": args.video_id,
        "visual_encoder_id": args.visual_encoder_id,
        "shot_count": len(metadata),
        "dimension": int(embeddings.shape[1]),
        "min_confidence": args.min_confidence,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
