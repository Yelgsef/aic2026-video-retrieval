from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_video(video_id: str) -> Path:
    import pandas as pd

    videos = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    rows = videos[videos["video_id"] == video_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one video for {video_id}, got {len(rows)}")
    return Path(rows.iloc[0]["video_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an AIC video with faster-whisper.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--language", default="vi")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    video_path = find_video(args.video_id)
    output_dir = PROJECT_ROOT / "workspace" / "asr" / "faster_whisper_large_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.video_id}.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Transcript exists: {output_path}")

    # CTranslate2 on Windows can reuse CUDA/cuDNN DLLs shipped with the SigLIP torch runtime.
    torch_lib = PROJECT_ROOT / "environments" / "siglip2" / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.is_dir():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

    from faster_whisper import WhisperModel

    started = time.time()
    print(f"Loading {args.model} on {args.device} ({args.compute_type})", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments_generator, info = model.transcribe(
        str(video_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    segments = []
    words = []
    for index, segment in enumerate(segments_generator):
        segment_words = []
        for word in segment.words or []:
            item = {
                "start": float(word.start),
                "end": float(word.end),
                "text": word.word,
                "probability": float(word.probability),
            }
            words.append({"segment_id": index, **item})
            segment_words.append(item)
        segments.append(
            {
                "segment_id": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
                "words": segment_words,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"Transcribed {index + 1} segments through {segment.end:.1f}s", flush=True)

    document = {
        "schema_version": "0.1.0",
        "video_id": args.video_id,
        "video_path": str(video_path),
        "engine": "faster-whisper",
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "duration_after_vad": float(info.duration_after_vad),
        "segment_count": len(segments),
        "word_count": len(words),
        "segments": segments,
        "words": words,
        "full_text": " ".join(segment["text"] for segment in segments),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({key: document[key] for key in (
        "video_id", "model", "language", "duration", "duration_after_vad",
        "segment_count", "word_count", "elapsed_seconds"
    )}, ensure_ascii=False, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
