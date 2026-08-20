from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.task_router import route_query
from aic_pipeline.qa_gemini import GeminiQAEngine, local_exact_ocr_answer, qa_grounding_query
from aic_pipeline.temporal_alignment import TemporalAligner
from aic_pipeline.kis_review import build_kis_review_pack
from aic_pipeline.retrieval_fusion import robust_unit
from aic_pipeline.submission import build_qa_submissions, build_trake_submissions


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent localhost AIC query service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--index", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--device")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    engine = OfflineSearchEngine(ROOT, args.index, args.device)
    model_load_seconds = time.perf_counter() - started
    manifest = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    video_paths = {str(row.video_id): Path(row.video_path).resolve() for row in manifest.itertuples()}
    frame_cache = ROOT / "workspace" / "ui_frame_cache"
    frame_cache.mkdir(parents=True, exist_ok=True)
    qa_engine: dict[str, GeminiQAEngine] = {}
    temporal_aligner = [None]
    kis_frame_cache: dict[str, pd.DataFrame] = {}
    warmup_seconds = 0.0
    if not args.no_warmup:
        warmup_started = time.perf_counter()
        if engine.query_transformer.available:
            # Pay Qwen/llama.cpp model startup once before the service reports
            # ready; subsequent uncached query parses take about one second.
            engine.query_transformer._start_server()
        engine.search("người đang hoạt động", top_k=1, candidate_k=20, rerank_k=4,
                      field_aware=True, openclip=engine.openclip_vectors is not None,
                      online_gemini_text=False, query_transformer=False)
        temporal_aligner[0] = TemporalAligner(ROOT, args.device)
        temporal_aligner[0]._event_vectors(["người bước đi", "người ngồi xuống"])
        warmup_seconds = time.perf_counter() - warmup_started
    print(json.dumps({"event": "ready", "load_seconds": model_load_seconds,
                      "warmup_seconds": warmup_seconds,
                      "ready_seconds": time.perf_counter() - started,
                      **engine.health()}), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, data: bytes, content_type: str,
                       extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = (ROOT / "web" / "index.html").read_bytes()
                self.send_bytes(200, data, "text/html; charset=utf-8",
                                {"Cache-Control": "no-cache"})
            elif parsed.path == "/health":
                self.send_json(200, engine.health())
            elif parsed.path == "/frame":
                try:
                    query = parse_qs(parsed.query)
                    video_id = query["video_id"][0]
                    frame_id = max(0, int(query["frame_id"][0]))
                    if video_id not in video_paths:
                        raise ValueError("unknown video_id")
                    target = frame_cache / f"{video_id}_{frame_id:09d}.jpg"
                    if not target.exists():
                        import cv2
                        capture = cv2.VideoCapture(str(video_paths[video_id]))
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ok, frame = capture.read()
                        capture.release()
                        if not ok:
                            raise ValueError("cannot decode frame")
                        ok, encoded = cv2.imencode(".jpg", frame,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if not ok:
                            raise ValueError("cannot encode frame")
                        target.write_bytes(encoded.tobytes())
                    self.send_bytes(200, target.read_bytes(), "image/jpeg")
                except (KeyError, ValueError, IndexError) as exc:
                    self.send_json(400, {"error": str(exc)})
            elif parsed.path == "/video":
                try:
                    video_id = parse_qs(parsed.query)["video_id"][0]
                    path = video_paths.get(video_id)
                    if path is None or not path.is_file():
                        raise ValueError("unknown video_id")
                    size = path.stat().st_size
                    start, end = 0, size - 1
                    range_header = self.headers.get("Range")
                    if range_header and range_header.startswith("bytes="):
                        bounds = range_header[6:].split("-", 1)
                        start = int(bounds[0] or 0)
                        end = min(int(bounds[1]) if bounds[1] else size - 1, size - 1)
                    length = max(0, end - start + 1)
                    self.send_response(206 if range_header else 200)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(length))
                    if range_header:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = length
                        while remaining:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (KeyError, ValueError, IndexError) as exc:
                    self.send_json(400, {"error": str(exc)})
            else:
                self.send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/route":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length))
                    self.send_json(200, route_query(str(request["query"])).to_dict())
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self.send_json(400, {"error": str(exc)})
                return
            if self.path == "/answer":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length))
                    question = str(request["query"])
                    task = route_query(question)
                    forced_task = str(request.get("task", "")).upper()
                    if task.task != "QA" and forced_task != "QA":
                        raise ValueError("/answer accepts only QA queries")
                    request_started = time.perf_counter()
                    vqa_candidates = int(request.get("vqa_candidates", 12))
                    candidate_k = max(int(request.get("candidate_k", 30)), vqa_candidates)
                    retrieval_started = time.perf_counter()
                    retrieval = engine.search(qa_grounding_query(question), top_k=candidate_k,
                                              rerank_k=int(request.get("rerank_k", 20)),
                                              fusion_mode="normalized", field_aware=True,
                                              openclip=engine.openclip_vectors is not None,
                                              calibrate_openclip=False, reranker_fusion="rank",
                                              query_transformer=bool(request.get("query_transformer", True)))
                    retrieval_seconds = time.perf_counter() - retrieval_started
                    answer = (local_exact_ocr_answer(ROOT, question, retrieval["results"][0])
                              if retrieval["results"] else None)
                    if answer is None:
                        gemini_mode = str(request.get("gemini_mode", os.getenv("GEMINI_MODE", "9router"))).strip().lower()
                        if gemini_mode not in {"9router", "gemini_api"}:
                            raise ValueError("gemini_mode must be 9router or gemini_api")
                        if gemini_mode not in qa_engine:
                            qa_engine[gemini_mode] = GeminiQAEngine(ROOT, mode=gemini_mode)
                        answer = qa_engine[gemini_mode].answer(question, retrieval["results"],
                                                     vqa_candidates)
                        answer["gemini_mode"] = gemini_mode
                    answer_variants = [answer.get("answer", "")]
                    if (answer.get("first_answer") and
                            answer.get("first_answer") != answer.get("answer")):
                        answer_variants.append(answer["first_answer"])
                    submissions = build_qa_submissions(
                        retrieval["results"], answer_variants,
                        min(100, int(request.get("top_k", 100))))
                    timings = {"retrieval": round(retrieval_seconds, 4),
                               "qa": answer.get("timings", {}),
                               "total": round(time.perf_counter() - request_started, 4)}
                    self.send_json(200, {"task_plan": task.to_dict(), "answer": answer,
                                         "retrieval": retrieval, "submissions": submissions,
                                         "timings": timings})
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self.send_json(400, {"error": str(exc)})
                except Exception as exc:
                    self.send_json(500, {"error": type(exc).__name__, "detail": str(exc)})
                return
            if self.path == "/trake":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length))
                    query = str(request["query"])
                    task = route_query(query)
                    forced_task = str(request.get("task", "")).upper()
                    if task.task != "TRAKE" and forced_task != "TRAKE":
                        raise ValueError("/trake requires a query with at least two ordered events")
                    if len(task.events) < 2:
                        raise ValueError("TRAKE cần ít nhất hai sự kiện, ví dụ: (1) ...; (2) ...")
                    # Score every ordered event independently. A video must cover the
                    # whole sequence; one excellent match must not dominate missing events.
                    request_started = time.perf_counter()
                    event_retrievals = []
                    for event in task.events:
                        event_retrievals.append(engine.search(
                            event, top_k=50, candidate_k=150, rerank_k=12,
                            fusion_mode="rrf", field_aware=True, openclip=False,
                            calibrate_openclip=False, reranker_fusion="rank",
                            online_gemini_text=False,
                            query_transformer=bool(request.get("query_transformer", True))))
                    per_event = []
                    per_event_object = []
                    all_videos = set()
                    for retrieval in event_retrievals:
                        scores = {}
                        object_scores = {}
                        for result in retrieval["results"]:
                            video = result["video_id"]
                            scores[video] = max(scores.get(video, float("-inf")),
                                                float(result["fusion_score"]))
                            if result.get("object_spatial_score") is not None:
                                object_scores[video] = max(object_scores.get(video, 0.0),
                                                           float(result["object_spatial_score"]))
                        per_event.append(scores)
                        per_event_object.append(object_scores)
                        all_videos.update(scores)
                    video_scores = {}
                    ordered_videos = sorted(all_videos)
                    fusion_units = robust_unit(np.array([
                        sum(scores.get(video, 0.0) for scores in per_event) / len(per_event)
                        for video in ordered_videos], dtype=np.float32))
                    object_units = robust_unit(np.array([
                        sum(scores.get(video, 0.0) for scores in per_event_object) / len(per_event_object)
                        for video in ordered_videos], dtype=np.float32))
                    for video in all_videos:
                        available = [scores[video] for scores in per_event if video in scores]
                        coverage = len(available) / len(per_event)
                        # Coverage is primary; mean event evidence breaks ties.
                        position = ordered_videos.index(video)
                        object_weight = 0.9 if any(per_event_object) else 0.0
                        video_scores[video] = {"coverage": coverage,
                                               "mean_event_score": sum(available) / len(available),
                                               "mean_object_score": float(object_units[position]),
                                               "video_fusion_score": float(
                                                   (1 - object_weight) * fusion_units[position]
                                                   + object_weight * object_units[position]),
                                               "event_scores": [scores.get(video) for scores in per_event]}
                    video_id = max(video_scores,
                                   key=lambda video: (video_scores[video]["coverage"],
                                                      video_scores[video]["video_fusion_score"]))
                    event_windows = [[result for result in retrieval["results"]
                                      if result["video_id"] == video_id][:5]
                                     for retrieval in event_retrievals]
                    if temporal_aligner[0] is None:
                        temporal_aligner[0] = TemporalAligner(ROOT)
                    alignment_started = time.perf_counter()
                    alignment = temporal_aligner[0].align(
                        video_id, list(task.events), event_windows,
                        decoder="windowed", short_clip=True, clip_seconds=2.0,
                        motion_rerank=bool(request.get("motion_rerank", False)))
                    feature_metadata = pd.read_parquet(
                        ROOT / "workspace" / "features" / "google_siglip2_base_patch16_224"
                        / f"{video_id}.parquet")
                    submissions = build_trake_submissions(
                        video_id, alignment["events"],
                        feature_metadata.frame_id.astype(int).tolist(),
                        maximum=min(100, int(request.get("top_k", 100))))
                    self.send_json(200, {"task_plan": task.to_dict(), "alignment": alignment,
                                         "video_candidates": video_scores, "submissions": submissions,
                                         "timings": {
                                             "events": [item.get("timings", {}) for item in event_retrievals],
                                             "alignment": round(time.perf_counter() - alignment_started, 4),
                                             "total": round(time.perf_counter() - request_started, 4)}})
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self.send_json(400, {"error": str(exc)})
                except Exception as exc:
                    self.send_json(500, {"error": type(exc).__name__, "detail": str(exc)})
                return
            if self.path != "/query":
                self.send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                started_query = time.perf_counter()
                query = str(request["query"])
                task = route_query(query)
                # Validation shows the independent OpenCLIP branch helps KIS,
                # while field-specific BGE vectors help ordered TRAKE events.
                response = engine.search(query, top_k=int(request.get("top_k", 100)),
                                         candidate_k=int(request.get("candidate_k", 300)),
                                         rerank_k=int(request.get("rerank_k", 20)),
                                         fusion_mode=str(request.get("fusion_mode", "normalized")),
                                         openclip=bool(request.get("openclip", True)),
                                         field_aware=bool(request.get("field_aware", True)),
                                         calibrate_openclip=bool(request.get(
                                             "calibrate_openclip", task.task == "KIS")),
                                         reranker_fusion=str(request.get("reranker_fusion", "auto")),
                                         online_gemini_text=bool(request.get("online_gemini_text", False)),
                                         query_transformer=bool(request.get("query_transformer", True)))
                if (task.task == "KIS" and response.get("results")
                        and bool(request.get("moment_refinement", False))):
                    selected_video = str(response["results"][0]["video_id"])
                    windows = [item for item in response["results"]
                               if str(item["video_id"]) == selected_video][:8]
                    if temporal_aligner[0] is None:
                        temporal_aligner[0] = TemporalAligner(ROOT, args.device)
                    original_frame = int(response["results"][0]["best_frame_id"])
                    refined = temporal_aligner[0].refine_kis_moment(
                        selected_video, query, windows)
                    response["results"][0]["best_frame_id"] = int(refined["frame_id"])
                    response["kis_moment_refinement"] = {
                        "original_frame_id": original_frame, **refined,
                    }
                if task.task == "KIS" and response.get("results"):
                    review_video_ids = []
                    for item in response["results"]:
                        value = str(item["video_id"])
                        if value not in review_video_ids:
                            review_video_ids.append(value)
                        if len(review_video_ids) >= min(5, int(request.get("review_videos", 3))):
                            break
                    review_tables = []
                    feature_root = ROOT / "workspace" / "features" / "google_siglip2_base_patch16_224"
                    for video_id in review_video_ids:
                        if video_id not in kis_frame_cache:
                            path = feature_root / f"{video_id}.parquet"
                            kis_frame_cache[video_id] = (pd.read_parquet(
                                path, columns=["video_id", "frame_id", "pts_time"])
                                if path.exists() else engine.frames[engine.frames.video_id == video_id])
                        review_tables.append(kis_frame_cache[video_id])
                    review_frames = pd.concat(review_tables, ignore_index=True) if review_tables else engine.frames
                    response["kis_review_pack"] = build_kis_review_pack(
                        query, response["results"], review_frames,
                        maximum_videos=min(5, int(request.get("review_videos", 3))),
                        scenes_per_video=min(5, int(request.get("review_scenes", 3))),
                        frames_per_scene=min(25, int(request.get("review_frames", 25))))
                response["latency_seconds"] = time.perf_counter() - started_query
                self.send_json(200, response)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.send_json(500, {"error": type(exc).__name__, "detail": str(exc)})

        def log_message(self, format: str, *values: object) -> None:
            print(json.dumps({"client": self.client_address[0], "message": format % values}), flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
