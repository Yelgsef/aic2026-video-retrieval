from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


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

    # Reuse the CUDA runtime libraries installed with the SigLIP PyTorch environment.
    siglip_site = PROJECT_ROOT / "environments" / "siglip2" / ".venv"
    windows_torch_lib = siglip_site / "Lib" / "site-packages" / "torch" / "lib"
    if windows_torch_lib.is_dir():
        os.environ["PATH"] = str(windows_torch_lib) + os.pathsep + os.environ.get("PATH", "")
    linux_nvidia = siglip_site / "lib" / "python3.10" / "site-packages" / "nvidia"
    linux_cuda_libs = [linux_nvidia / component / "lib" for component in ("cublas", "cudnn")]
    linux_cuda_libs = [path for path in linux_cuda_libs if path.is_dir()]
    if linux_cuda_libs and not os.environ.get("AIC_ASR_CUDA_PATH_READY"):
        environment = os.environ.copy()
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*(str(path) for path in linux_cuda_libs), current]
        ).rstrip(os.pathsep)
        environment["AIC_ASR_CUDA_PATH_READY"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)

    from faster_whisper import WhisperModel

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
    for index, segment in enumerate(segments_generator):
        segment_words = []
        for word in segment.words or []:
            item = {
                "start": float(word.start),
                "end": float(word.end),
                "text": word.word,
                "probability": float(word.probability),
            }
            segment_words.append(item)
        segments.append(
            {
                "segment_id": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(segment.avg_logprob),
                "words": segment_words,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"Transcribed {index + 1} segments through {segment.end:.1f}s", flush=True)

    document = {
        "schema_version": "0.2.0",
        "video_id": args.video_id,
        "model": args.model,
        "language": info.language,
        "duration": float(info.duration),
        "segments": segments,
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({
        "video_id": args.video_id,
        "duration": document["duration"],
        "model": args.model,
        "language": info.language,
        "segments": len(segments),
        "words": sum(len(segment["words"]) for segment in segments),
    }, ensure_ascii=False, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
