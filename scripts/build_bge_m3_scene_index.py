from __future__ import annotations

import argparse
import json
import os
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
    table = pd.read_parquet(args.documents)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_id, local_files_only=True, torch_dtype=dtype).to(device).eval()
    def encode(texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), args.batch_size):
            inputs = tokenizer(texts[start:start + args.batch_size], padding=True, truncation=True,
                               max_length=args.max_length, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state[:, 0].float()
                hidden = torch.nn.functional.normalize(hidden, dim=-1)
            chunks.append(hidden.cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def join_values(value: object) -> str:
        if value is None:
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
    matrices = {field: encode(texts) for field, texts in field_texts.items()}
    matrix = matrices["retrieval"]
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "text_embeddings_bge_m3.npy", matrix)
    for field, field_matrix in matrices.items():
        if field != "retrieval":
            np.save(args.output / f"text_embeddings_bge_m3_{field}.npy", field_matrix)
    table[["row_id", "document_id", "video_id", "scene_index"]].to_parquet(
        args.output / "text_embeddings_bge_m3.parquet", index=False)
    report = {"model": args.model_id, "rows": len(matrix), "dimension": int(matrix.shape[1]),
              "fields": list(matrices),
              "device": str(device), "max_length": args.max_length}
    (args.output / "bge_m3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
