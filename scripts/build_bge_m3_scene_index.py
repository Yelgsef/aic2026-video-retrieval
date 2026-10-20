from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized BGE-M3 dense vectors for scene documents.")
    parser.add_argument("--documents", type=Path, default=ROOT / "workspace" / "offline_index" / "scenes.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--model-id", default=str(ROOT / "models" / "bge-m3"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_length < 1:
        parser.error("--max-length must be at least 1")
    table = pd.read_parquet(args.documents)
    required = {"row_id", "document_id", "video_id", "scene_index", "retrieval_text",
                "visual_summary", "asr_text", "asr_context", "ocr_text", "entities_text"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Scene documents missing required columns: {missing}")
    if table.empty:
        raise ValueError(f"No scene documents in {args.documents}")
    if table.row_id.duplicated().any():
        raise ValueError("Scene documents contain duplicate row_id values")
    if not np.array_equal(table.row_id.to_numpy(), np.arange(len(table))):
        raise ValueError("Scene row_id must match scenes.parquet row order")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_id, local_files_only=True, torch_dtype=dtype).to(device).eval()
    def encode_batch(texts: list[str]) -> np.ndarray:
        inputs = tokenizer(texts, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, 0].float()
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
        return hidden.cpu().numpy().astype(np.float32, copy=False)

    def join_values(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, str):
            return value
        try:
            return " ".join(str(item) for item in value)
        except TypeError:
            return str(value)

    field_texts = {
        "retrieval": table.retrieval_text.fillna("").astype(str).tolist(),
        "summary": table.visual_summary.fillna("").astype(str).tolist(),
        "asr": (table.asr_text.fillna("").astype(str) + " " + table.asr_context.fillna("").astype(str)).tolist(),
        "ocr_entities": [f"{join_values(ocr)} {entities}".strip()
                         for ocr, entities in zip(table.ocr_text, table.entities_text.fillna(""))],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    staged_paths: dict[str, Path] = {}
    report_path = args.output / "bge_m3_report.json"
    with tempfile.TemporaryDirectory(prefix=".bge_m3_staging_", dir=args.output) as staging_name:
        staging = Path(staging_name)
        dimension: int | None = None
        for field, texts in field_texts.items():
            filename = "text_embeddings_bge_m3.npy" if field == "retrieval" else f"text_embeddings_bge_m3_{field}.npy"
            path = staging / filename
            matrix = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(len(texts), int(model.config.hidden_size)))
            for start in range(0, len(texts), args.batch_size):
                batch = encode_batch(texts[start:start + args.batch_size])
                if dimension is None:
                    dimension = int(batch.shape[1])
                if batch.shape != (min(args.batch_size, len(texts) - start), dimension):
                    raise ValueError(f"Unexpected {field} embedding shape: {batch.shape}")
                matrix[start:start + len(batch)] = batch
            matrix.flush()
            del matrix
            staged_paths[filename] = path
        if dimension is None:
            raise RuntimeError("BGE-M3 produced no embeddings")
        report = {"schema_version": "0.2.0", "model": args.model_id, "rows": len(table),
                  "dimension": dimension, "fields": list(field_texts),
                  "device": str(device), "batch_size": args.batch_size,
                  "max_length": args.max_length, "documents": str(args.documents.resolve())}
        staged_report = staging / report_path.name
        staged_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        table[["row_id", "document_id", "video_id", "scene_index"]].to_parquet(
            staging / "text_embeddings_bge_m3.parquet", index=False)

        # Publish the report last so it marks a complete set of field matrices.
        report_path.unlink(missing_ok=True)
        for filename, staged in staged_paths.items():
            staged.replace(args.output / filename)
        (staging / "text_embeddings_bge_m3.parquet").replace(
            args.output / "text_embeddings_bge_m3.parquet")
        staged_report.replace(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
