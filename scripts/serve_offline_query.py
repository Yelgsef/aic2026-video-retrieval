from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.task_router import route_query
from aic_pipeline.qa_gemini import GeminiQAEngine
from aic_pipeline.temporal_alignment import TemporalAligner


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent localhost AIC query service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--index", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--device")
    args = parser.parse_args()
    started = time.perf_counter()
    engine = OfflineSearchEngine(ROOT, args.index, args.device)
    qa_engine = [None]
    temporal_aligner = [None]
    print(json.dumps({"event": "ready", "load_seconds": time.perf_counter() - started, **engine.health()}), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_json(200, engine.health())
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
                    if task.task != "QA":
                        raise ValueError("/answer accepts only QA queries")
                    vqa_candidates = int(request.get("vqa_candidates", 24))
                    candidate_k = max(int(request.get("candidate_k", 30)), vqa_candidates)
                    retrieval = engine.search(question, top_k=candidate_k,
                                              rerank_k=int(request.get("rerank_k", 30)))
                    if qa_engine[0] is None:
                        qa_engine[0] = GeminiQAEngine(ROOT)
                    answer = qa_engine[0].answer(question, retrieval["results"],
                                                 vqa_candidates)
                    self.send_json(200, {"task_plan": task.to_dict(), "answer": answer,
                                         "retrieval": retrieval})
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
                    if task.task != "TRAKE":
                        raise ValueError("/trake requires a query with at least two ordered events")
                    # Score every ordered event independently. A video must cover the
                    # whole sequence; one excellent match must not dominate missing events.
                    event_retrievals = [engine.search(event, top_k=100, rerank_k=30)
                                        for event in task.events]
                    per_event = []
                    all_videos = set()
                    for retrieval in event_retrievals:
                        scores = {}
                        for result in retrieval["results"]:
                            video = result["video_id"]
                            scores[video] = max(scores.get(video, float("-inf")),
                                                float(result["fusion_score"]))
                        per_event.append(scores)
                        all_videos.update(scores)
                    video_scores = {}
                    for video in all_videos:
                        available = [scores[video] for scores in per_event if video in scores]
                        coverage = len(available) / len(per_event)
                        # Coverage is primary; mean event evidence breaks ties.
                        video_scores[video] = {"coverage": coverage,
                                               "mean_event_score": sum(available) / len(available),
                                               "event_scores": [scores.get(video) for scores in per_event]}
                    video_id = max(video_scores,
                                   key=lambda video: (video_scores[video]["coverage"],
                                                      video_scores[video]["mean_event_score"]))
                    if temporal_aligner[0] is None:
                        temporal_aligner[0] = TemporalAligner(ROOT)
                    alignment = temporal_aligner[0].align(video_id, list(task.events))
                    self.send_json(200, {"task_plan": task.to_dict(), "alignment": alignment,
                                         "video_candidates": video_scores})
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
                response = engine.search(str(request["query"]), top_k=int(request.get("top_k", 100)),
                                         candidate_k=int(request.get("candidate_k", 300)),
                                         rerank_k=int(request.get("rerank_k", 30)))
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
