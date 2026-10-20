from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def select_evenly(rows: pd.DataFrame, maximum: int) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("max representative frames must be at least 1")
    rows = rows.sort_values("frame_id").drop_duplicates("frame_id")
    if len(rows) > maximum:
        positions = [0] if maximum == 1 else sorted(
            {round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)}
        )
        rows = rows.iloc[positions]
    return [
        {"frame_id": int(row.frame_id), "pts_time": float(row.pts_time), "shot_id": int(row.shot_id)}
        for row in rows.itertuples(index=False)
    ]


def asr_for_interval(asr: dict[str, Any] | None, start: float, end: float) -> tuple[str, str]:
    if not asr:
        return "", ""
    words = [
        word
        for segment in asr.get("segments", [])
        for word in segment.get("words", [])
        if start <= (float(word["start"]) + float(word["end"])) / 2 < end
    ]
    segments = [
        segment for segment in asr.get("segments", [])
        if min(float(segment["end"]), end) > max(float(segment["start"]), start)
    ]
    exact = " ".join(str(word["text"]).strip() for word in words if str(word["text"]).strip())
    context = " ".join(dict.fromkeys(str(segment["text"]).strip() for segment in segments if str(segment["text"]).strip()))
    return exact, context


def build_technical_shots(ranges: list[list[int]], fps: float,
                          intra_labels: list[str | None]) -> list[dict[str, Any]]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps}")
    if not ranges:
        raise ValueError("Shot file contains no shot ranges")
    if len(intra_labels) != len(ranges):
        intra_labels = [None] * len(ranges)
    shots = []
    for shot_id, (start_frame, end_frame) in enumerate(ranges):
        start_frame, end_frame = int(start_frame), int(end_frame)
        if end_frame <= start_frame:
            raise ValueError(f"Invalid frame range for shot {shot_id}: {start_frame}:{end_frame}")
        shots.append({
            "shot_id": shot_id,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "start_time": start_frame / fps,
            "end_time": end_frame / fps,
            "duration": (end_frame - start_frame) / fps,
            "intra_label": intra_labels[shot_id],
        })
    return shots


def merge_adjacent_shots(technical: list[dict[str, Any]], centroids: dict[int, np.ndarray],
                         asr_segments: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    scenes = []
    current = {"shots": [technical[0]], "boundary_similarities": [], "merge_reasons": []}
    transition_names = {"Dissolve", "Wipes", "Zoom", "Fade", "Transition"}
    for previous, following in zip(technical, technical[1:]):
        previous_center = centroids.get(previous["shot_id"])
        following_center = centroids.get(following["shot_id"])
        similarity = float(previous_center @ following_center) if previous_center is not None and following_center is not None else -1.0
        boundary_time = following["start_time"]
        asr_bridge = any(float(segment["start"]) < boundary_time < float(segment["end"])
                         for segment in asr_segments)
        reasons = []
        if similarity >= args.visual_cosine:
            reasons.append("visual_similarity")
        if min(previous["duration"], following["duration"]) <= args.short_shot_seconds and similarity >= args.short_shot_cosine:
            reasons.append("short_shot")
        if asr_bridge and similarity >= args.asr_bridge_cosine:
            reasons.append("shared_asr_segment")
        if (previous["intra_label"] in transition_names or following["intra_label"] in transition_names) and similarity >= args.transition_cosine:
            reasons.append("transition_continuity")
        proposed_duration = following["end_time"] - current["shots"][0]["start_time"]
        if reasons and proposed_duration <= args.max_scene_seconds:
            current["shots"].append(following)
            current["boundary_similarities"].append(similarity)
            current["merge_reasons"].append(reasons)
        else:
            scenes.append(current)
            current = {"shots": [following], "boundary_similarities": [], "merge_reasons": []}
    scenes.append(current)
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge adjacent technical shots into semantic scenes.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--encoder-id", default="google_siglip2_base_patch16_224")
    parser.add_argument("--visual-cosine", type=float, default=0.90)
    parser.add_argument("--short-shot-seconds", type=float, default=0.75)
    parser.add_argument("--short-shot-cosine", type=float, default=0.78)
    parser.add_argument("--asr-bridge-cosine", type=float, default=0.75)
    parser.add_argument("--transition-cosine", type=float, default=0.75)
    parser.add_argument("--max-scene-seconds", type=float, default=45.0)
    parser.add_argument("--max-representative-frames", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_representative_frames < 1:
        parser.error("--max-representative-frames must be at least 1")

    output_dir = ROOT / "workspace" / "semantic_scenes" / args.video_id
    parquet_path = output_dir / "scenes.parquet"
    if parquet_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {parquet_path}")
    shots = json.loads((ROOT / "workspace" / "shots" / f"{args.video_id}.json").read_text(encoding="utf-8"))
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    video = videos.loc[videos.video_id == args.video_id]
    if len(video) != 1:
        raise ValueError(f"Expected one video row for {args.video_id}")
    fps = float(video.iloc[0].fps)
    feature_dir = ROOT / "workspace" / "features" / args.encoder_id
    samples = pd.read_parquet(feature_dir / f"{args.video_id}.parquet").sort_values("embedding_row")
    vectors = np.load(feature_dir / f"{args.video_id}.f16.npy", mmap_mode="r")
    if len(samples) != len(vectors):
        raise ValueError("Feature metadata and embeddings differ")
    final_path = ROOT / "workspace" / "keyframes" / args.encoder_id / f"{args.video_id}.parquet"
    final_frames = pd.read_parquet(final_path)
    asr_path = ROOT / "workspace" / "asr" / "faster_whisper_large_v3" / f"{args.video_id}.json"
    asr = json.loads(asr_path.read_text(encoding="utf-8")) if asr_path.exists() else None

    normalized_vectors = np.asarray(vectors, dtype=np.float32)
    normalized_vectors /= np.clip(np.linalg.norm(normalized_vectors, axis=1, keepdims=True), 1e-12, None)
    centroids: dict[int, np.ndarray] = {}
    for shot_id, group in samples.groupby("shot_id"):
        centroids[int(shot_id)] = normalize(normalized_vectors[group.embedding_row.astype(int)].mean(axis=0))

    technical = build_technical_shots(shots["ranges"], fps, shots.get("intra_labels", []))
    asr_segments = asr.get("segments", []) if asr else []
    scenes = merge_adjacent_shots(technical, centroids, asr_segments, args)

    output_rows = []
    json_scenes = []
    for scene_index, scene in enumerate(scenes):
        scene_shots = scene["shots"]
        shot_ids = [int(item["shot_id"]) for item in scene_shots]
        start_time = float(scene_shots[0]["start_time"])
        end_time = float(scene_shots[-1]["end_time"])
        representatives = select_evenly(final_frames[final_frames.shot_id.isin(shot_ids)], args.max_representative_frames)
        exact, context = asr_for_interval(asr, start_time, end_time)
        scene_id = f"{args.video_id}:scene:{scene_index:06d}"
        record = {
            "scene_id": scene_id,
            "video_id": args.video_id,
            "scene_index": scene_index,
            "start_frame": int(scene_shots[0]["start_frame"]),
            "end_frame_exclusive": int(scene_shots[-1]["end_frame_exclusive"]),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "shot_ids": shot_ids,
            "shot_count": len(shot_ids),
            "representative_frames": representatives,
            "asr_exact": exact,
            "asr_context": context,
            "boundary_similarities": scene["boundary_similarities"],
            "merge_reasons": scene["merge_reasons"],
        }
        json_scenes.append(record)
        output_rows.append({
            "scene_id": scene_id,
            "video_id": args.video_id,
            "scene_index": scene_index,
            "start_frame": record["start_frame"],
            "end_frame_exclusive": record["end_frame_exclusive"],
            "start_time": start_time,
            "end_time": end_time,
            "shot_ids": shot_ids,
            "shot_count": len(shot_ids),
            "asr_exact": exact,
            "asr_context": context,
            "representative_frame_ids": [item["frame_id"] for item in representatives],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_parquet(parquet_path, index=False)
    (output_dir / "scenes.json").write_text(json.dumps({"video_id": args.video_id, "scenes": json_scenes}, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_boundaries = sum(scene["shot_count"] - 1 for scene in output_rows)
    report = {
        "pipeline_version": "0.2.0",
        "video_id": args.video_id,
        "technical_shots": len(technical),
        "semantic_scenes": len(output_rows),
        "merged_boundaries": merged_boundaries,
        "compression_ratio": 1.0 - len(output_rows) / len(technical),
        "maximum_scene_seconds": max(row["end_time"] - row["start_time"] for row in output_rows),
        "maximum_shots_per_scene": max(row["shot_count"] for row in output_rows),
        "scenes_with_asr": sum(bool(row["asr_exact"]) for row in output_rows),
        "parameters": vars(args),
        "outputs": {"parquet": str(parquet_path), "json": str(output_dir / "scenes.json")},
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
