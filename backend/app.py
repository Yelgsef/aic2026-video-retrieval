from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.runtime import Runtime
from backend.schemas import QaAnswerRequest, QueryRequest
from backend.services.media_service import extract_frame, probe_video
from backend.services.qa_service import answer_with_images
from backend.services.query_service import execute_query


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = await asyncio.to_thread(Runtime, settings)
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.runtime.close)


app = FastAPI(title="AIC Flow C", lifespan=lifespan)


@app.get("/api/health")
async def health(request: Request) -> dict:
    return request.app.state.runtime.engine.health()


@app.post("/api/route")
async def route(body: dict) -> dict:
    from aic_pipeline.task_router import route_query
    return route_query(str(body.get("query", ""))).to_dict()


@app.post("/api/query")
async def query(body: QueryRequest, request: Request) -> dict:
    try:
        data = await asyncio.to_thread(
            execute_query, request.app.state.runtime, body.type, body.query.strip(),
            top_k=body.top_k, rerank_k=body.rerank_k,
            openclip=body.openclip, query_transformer=body.query_transformer,
            reranker_enabled=body.reranker_enabled,
        )
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Query failed: {error}") from error
    return {"type": body.type, "data": data}


@app.post("/api/query/stream")
async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    async def events():
        yield json.dumps({"event": "start", "type": body.type}, ensure_ascii=False) + "\n"
        try:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict] = asyncio.Queue()

            def publish(message: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, message)

            result_task = asyncio.create_task(asyncio.to_thread(
                execute_query, request.app.state.runtime, body.type, body.query.strip(),
                top_k=body.top_k, rerank_k=body.rerank_k,
                openclip=body.openclip, query_transformer=body.query_transformer,
                reranker_enabled=body.reranker_enabled,
                progress_callback=publish if body.type in {"KIS", "QA"} else None,
            ))
            final_preview_sent = False
            while not result_task.done():
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                final_preview_sent |= message.get("final") is True
                yield json.dumps(message, ensure_ascii=False) + "\n"
            await asyncio.sleep(0)
            while not queue.empty():
                message = queue.get_nowait()
                final_preview_sent |= message.get("final") is True
                yield json.dumps(message, ensure_ascii=False) + "\n"
            data = await result_task
            if body.type == "TRAKE":
                for rank, row in enumerate(data.get("trake_rows", []), 1):
                    yield json.dumps({"event": "trake_row", "rank": rank, "row": row}, ensure_ascii=False) + "\n"
                for rank, row in enumerate(data.get("trake_rejected", []), 1):
                    yield json.dumps({"event": "trake_rejected", "rank": rank, "row": row}, ensure_ascii=False) + "\n"
                data = {key: value for key, value in data.items()
                        if key not in {"results", "event_results", "trake_rows", "trake_rejected"}}
                yield json.dumps({"event": "complete", "data": data}, ensure_ascii=False) + "\n"
                return
            results = data.get("results", [])
            if not final_preview_sent:
                for offset in range(0, len(results), 5):
                    yield json.dumps({"event": "batch", "items": results[offset:offset + 5]}, ensure_ascii=False) + "\n"
            metadata = {key: value for key, value in data.items() if key != "results"}
            yield json.dumps({"event": "complete", "data": metadata, "total": len(results)}, ensure_ascii=False) + "\n"
        except Exception as error:
            yield json.dumps({"event": "error", "error": str(error)}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/frame/{video_id}/{frame_id}")
async def frame(video_id: str, frame_id: int, request: Request,
                width: int | None = Query(default=320, ge=16, le=1280)) -> Response:
    if frame_id < 0:
        raise HTTPException(status_code=400, detail="frame_id must be non-negative")
    runtime: Runtime = request.app.state.runtime
    try:
        source = runtime.resolve_video(video_id)
        target = runtime.settings.frame_cache / f"{video_id}_{frame_id:09d}_{width or 0}.jpg"
        if not target.exists():
            encoded = await asyncio.to_thread(extract_frame, source, frame_id, width)
            target.write_bytes(encoded)
        return Response(target.read_bytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/videos/{video_id}.mp4")
async def video(video_id: str, request: Request) -> FileResponse:
    try:
        source = request.app.state.runtime.resolve_video(video_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(source, media_type="video/mp4", filename=f"{video_id}.mp4")


@app.get("/api/videos/{video_id}/metadata")
async def video_metadata(video_id: str, request: Request) -> dict:
    try:
        source = request.app.state.runtime.resolve_video(video_id)
        row = request.app.state.runtime.video_metadata.loc[video_id]
        metadata = {
            "fps": float(row.fps), "frame_count": int(row.frame_count),
            "duration_seconds": float(row.duration_seconds),
        }
        if metadata["fps"] <= 0 or metadata["frame_count"] <= 0:
            metadata = await asyncio.to_thread(probe_video, source)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"video_id": video_id, "file": source.name, "source": "ffprobe", **metadata}


@app.post("/api/qa/answer")
async def qa_answer(body: QaAnswerRequest) -> dict:
    try:
        answer = await answer_with_images(body.provider, body.api_key, body.model,
                                          body.system_prompt, body.question,
                                          [image.model_dump() for image in body.images])
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"text": answer}


@app.post("/api/qa/models")
async def qa_models(body: dict) -> dict:
    provider = body.get("provider", "ai_studio")
    key = str(body.get("api_key", "")).strip()
    if provider == "9router":
        import os
        import httpx
        from dotenv import load_dotenv
        load_dotenv(settings.root / ".env")
        headers = {}
        key = key or os.environ.get("NINEROUTER_KEY", "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = os.environ.get("NINEROUTER_CHAT_URL", "http://localhost:20128/v1/chat/completions")
        base = url.rsplit("/chat/completions", 1)[0]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{base}/models", headers=headers)
            payload = response.json()
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"Could not reach 9Router: {error}") from error
        if not response.is_success:
            raise HTTPException(status_code=502, detail=f"9Router returned HTTP {response.status_code}")
        return {"models": [{"id": item["id"], "name": item.get("name", item["id"]), "description": ""}
                           for item in payload.get("data", []) if item.get("id")]}
    if provider != "ai_studio":
        raise HTTPException(status_code=400, detail="Unsupported QA provider")
    if not key:
        raise HTTPException(status_code=400, detail="AI Studio API key is required")
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get("https://generativelanguage.googleapis.com/v1beta/models",
                                        params={"pageSize": 1000},
                                        headers={"x-goog-api-key": key})
        payload = response.json()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not reach AI Studio: {error}") from error
    if not response.is_success:
        raise HTTPException(status_code=502, detail=payload.get("error", {}).get("message", "AI Studio error"))
    models = [{"id": item.get("name", "").removeprefix("models/"),
               "name": item.get("displayName") or item.get("name", ""),
               "description": item.get("description", "")}
              for item in payload.get("models", [])
              if "generateContent" in item.get("supportedGenerationMethods", [])]
    return {"models": [item for item in models if item["id"]]}


FRONTEND = settings.root / "frontend"
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
