from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]


class OcrLine(BaseModel):
    text: str = Field(description="Exact visible text; preserve spelling and diacritics")
    frame_label: str = Field(description="Frame label where the text is visible")


class SceneDescription(BaseModel):
    summary_vi: str
    ocr_lines: list[OcrLine]
    people: list[str]
    objects: list[str]
    actions: list[str]
    location: list[str]
    event: list[str]
    search_keywords_vi: list[str]
    uncertainties: list[str]


def make_contact_sheet(rows: pd.DataFrame, output: Path, max_images: int) -> list[dict]:
    rows = rows.sort_values("pts_time")
    if len(rows) > max_images:
        positions = [round(i * (len(rows) - 1) / (max_images - 1)) for i in range(max_images)]
        rows = rows.iloc[positions]

    tiles: list[Image.Image] = []
    metadata: list[dict] = []
    tile_width, tile_height, label_height = 448, 252, 28
    for _, row in rows.iterrows():
        image_path = Path(row["image_path"])
        with Image.open(image_path) as source:
            tile = ImageOps.fit(source.convert("RGB"), (tile_width, tile_height))
        canvas = Image.new("RGB", (tile_width, tile_height + label_height), "white")
        canvas.paste(tile, (0, label_height))
        label = f"F{int(row.frame_id)} | {float(row.pts_time):.2f}s"
        ImageDraw.Draw(canvas).text((8, 7), label, fill="black")
        tiles.append(canvas)
        metadata.append({"label": label, "frame_id": int(row.frame_id), "pts_time": float(row.pts_time)})

    columns = 2
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows_count * (tile_height + label_height)), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * (tile_height + label_height)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--shot-id", type=int, default=27)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--model", default="gemini-3.6-flash")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in .env or the process environment")

    manifest_path = ROOT / "workspace" / "keyframe_images" / args.video_id / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    shot_rows = manifest[(manifest.shot_id == args.shot_id) & manifest.image_exists.astype(bool)]
    if shot_rows.empty:
        available = manifest.groupby("shot_id").size().sort_values(ascending=False)
        args.shot_id = int(available.index[0])
        shot_rows = manifest[(manifest.shot_id == args.shot_id) & manifest.image_exists.astype(bool)]

    sheet_path = ROOT / "workspace" / "contact_sheets" / args.video_id / f"shot_{args.shot_id:06d}.jpg"
    frames = make_contact_sheet(shot_rows, sheet_path, args.max_images)
    prompt = """Analyze this chronological contact sheet from one video shot for Vietnamese video retrieval.
The labels above tiles identify frames and timestamps. Return Vietnamese structured data.
Rules:
- Describe only evidence visible in the images; do not infer identities or unsupported facts.
- OCR must contain only text you can actually read. Preserve original spelling, casing, numbers, and Vietnamese diacritics. Do not repair or guess unclear text.
- Associate every OCR line with its frame label.
- Use concise concrete nouns/actions useful for search. Put ambiguity in uncertainties.
- Empty evidence must be an empty list, never invented content.
"""

    client = genai.Client(api_key=api_key)
    with Image.open(sheet_path) as image:
        response = client.models.generate_content(
            model=args.model,
            contents=[prompt, image.copy()],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SceneDescription,
                temperature=0.1,
            ),
        )
    parsed = SceneDescription.model_validate_json(response.text)
    usage = response.usage_metadata
    payload = {
        "video_id": args.video_id,
        "shot_id": args.shot_id,
        "model": args.model,
        "contact_sheet": str(sheet_path),
        "frames": frames,
        "description": parsed.model_dump(),
        "usage": usage.model_dump() if usage else None,
    }
    output = ROOT / "workspace" / "gemini_smoke" / args.video_id / f"shot_{args.shot_id:06d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "contact_sheet": str(sheet_path), "model": args.model, "usage": payload["usage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
