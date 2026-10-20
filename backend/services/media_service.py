from __future__ import annotations

import subprocess
from pathlib import Path

import cv2


def extract_frame(video_path: Path, frame_id: int, width: int | None = 320) -> bytes:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {frame_id}")
    if width and frame.shape[1] > width:
        height = max(1, round(frame.shape[0] * width / frame.shape[1]))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")
    return encoded.tobytes()


def probe_video(video_path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    import json
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("Video has no readable video stream")
    stream = streams[0]
    numerator, _, denominator = str(stream.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / max(1.0, float(denominator or 1))
    return {"width": int(stream.get("width", 0)), "height": int(stream.get("height", 0)),
            "fps": fps, "frame_count": int(stream.get("nb_frames", 0) or 0),
            "duration_seconds": float(stream.get("duration", 0) or 0)}
