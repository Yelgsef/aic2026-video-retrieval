from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent Gemini text-first scene embeddings.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", default="gemini-embedding-001")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    scenes = pd.read_parquet(args.index / "scenes.parquet")
    destination = args.index / "text_embeddings_gemini.npy"
    partial = args.index / "text_embeddings_gemini.partial.npy"
    state_path = args.index / "text_embeddings_gemini_state.json"
    completed = 0
    if args.resume and partial.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("model") == args.model and state.get("rows") == len(scenes):
            completed = int(state.get("completed", 0))
    mode = "r+" if completed else "w+"
    matrix = np.lib.format.open_memmap(partial, mode=mode, dtype=np.float32,
                                       shape=(len(scenes), args.dimension))
    started = time.perf_counter()
    for start in range(completed, len(scenes), args.batch_size):
        end = min(len(scenes), start + args.batch_size)
        texts = scenes.retrieval_text.iloc[start:end].fillna("").astype(str).tolist()
        for attempt in range(4):
            try:
                response = client.models.embed_content(
                    model=args.model, contents=texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",
                                                    output_dimensionality=args.dimension))
                values = np.asarray([item.values for item in response.embeddings], dtype=np.float32)
                values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
                matrix[start:end] = values; matrix.flush(); break
            except Exception:
                if attempt == 3: raise
                time.sleep(2 ** attempt)
        state_path.write_text(json.dumps({"model": args.model, "rows": len(scenes),
                                          "completed": end}), encoding="utf-8")
        print(json.dumps({"completed": end, "rows": len(scenes)}), flush=True)
    del matrix
    os.replace(partial, destination)
    scenes[["row_id", "document_id", "video_id", "scene_index"]].to_parquet(
        args.index / "text_embeddings_gemini.parquet", index=False)
    state_path.write_text(json.dumps({"model": args.model, "rows": len(scenes),
                                      "completed": len(scenes), "dimension": args.dimension,
                                      "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
