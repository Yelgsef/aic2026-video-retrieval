from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from .query_understanding import QueryPlan


class StructuredQuery(BaseModel):
    query_vi: str
    query_en: str
    visual_queries_en: list[str] = Field(default_factory=list, max_length=3)
    exact_terms: list[str] = Field(default_factory=list)
    objects_en: list[str] = Field(default_factory=list)
    answer_target: str = ""


class QueryTransformerUnavailable(RuntimeError):
    """Raised when Qwen was requested but its local runtime is not installed."""


class QueryTransformerError(RuntimeError):
    """Raised when the configured Qwen runtime fails to transform a query."""


class LocalQueryTransformer:
    """Lazy, cached local Qwen query parser.

    Query expansion is cached, invalid generations are rejected by Pydantic,
    and the regular retrieval pipeline continues to preserve the source query.
    """

    def __init__(self, root: Path, index: Path,
                 model_name: str = "qwen2.5-1.5b-instruct") -> None:
        self.root, self.index = root.resolve(), index.resolve()
        self.model_path = self.root / "models" / model_name
        self.gguf_path = (self.root / "models" / f"{model_name}-gguf" /
                          "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        self.database = self.index / "query_transformer_cache.sqlite3"
        self.model = None
        self.tokenizer = None
        self.server: subprocess.Popen | None = None
        self.server_port: int | None = None
        self._device_listing: str | None = None
        self.index.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    @property
    def model_available(self) -> bool:
        return self.gguf_path.is_file() or (self.model_path / "config.json").is_file()

    @property
    def available(self) -> bool:
        return self.use_gguf or (self.model_path / "config.json").is_file()

    @property
    def use_gguf(self) -> bool:
        return self.gguf_path.is_file() and self._llama_server() is not None

    def status(self) -> dict[str, object]:
        gguf_available = self.gguf_path.is_file()
        transformers_available = (self.model_path / "config.json").is_file()
        llama_server = self._llama_server()
        devices = self._available_gpu_devices() if llama_server is not None else []
        use_gguf = gguf_available and llama_server is not None
        backend = "llama.cpp" if use_gguf else (
            "transformers" if transformers_available else None)
        return {
            "available": self.available,
            "model_available": gguf_available or transformers_available,
            "backend": backend,
            "llama_server_available": llama_server is not None,
            "server_running": bool(self.server and self.server.poll() is None),
            "execution_device": "cuda" if devices else "cpu",
            "model_path": str(self.gguf_path if use_gguf else self.model_path),
        }

    def close(self) -> None:
        if self.server is None or self.server.poll() is not None:
            return
        self.server.terminate()
        try:
            self.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait(timeout=5)
        self.server = None
        self.server_port = None

    def _llama_server(self) -> Path | None:
        command = shutil.which("llama-server")
        if command:
            return Path(command)
        packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
        matches = list(packages.glob("ggml.llamacpp_*/llama-server.exe"))
        return matches[0] if matches else None

    def _available_devices(self) -> str:
        if self._device_listing is None:
            executable = self._llama_server()
            if executable is None:
                self._device_listing = ""
            else:
                try:
                    result = subprocess.run([str(executable), "--list-devices"], capture_output=True,
                                            text=True, timeout=10, check=False)
                    self._device_listing = result.stdout if result.returncode == 0 else ""
                except (OSError, subprocess.TimeoutExpired):
                    self._device_listing = ""
        return self._device_listing

    def _available_gpu_devices(self) -> list[str]:
        return [line.strip() for line in self._available_devices().splitlines()
                if line.strip() and not line.lower().startswith("available devices")
                and line.strip().casefold() != "(none)"]

    def _start_server(self) -> None:
        if self.server is not None and self.server.poll() is None:
            return
        executable = self._llama_server()
        if executable is None or not self.gguf_path.exists():
            raise FileNotFoundError("llama-server or Qwen GGUF is unavailable")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.server_port = int(probe.getsockname()[1])
        gpu_layers = "99" if self._available_gpu_devices() else "0"
        self.server = subprocess.Popen(
            [str(executable), "-m", str(self.gguf_path), "--host", "127.0.0.1",
             "--port", str(self.server_port), "--n-gpu-layers", gpu_layers,
             "--threads", "4", "--threads-batch", "4", "--ctx-size", "2048"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + 120
        health = f"http://127.0.0.1:{self.server_port}/health"
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                with urllib.request.urlopen(health, timeout=0.5) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.2)
        self.server.terminate()
        raise TimeoutError("llama-server startup exceeded 120 seconds")

    def _expand_gguf(self, query: str) -> StructuredQuery:
        self._start_server()
        schema = StructuredQuery.model_json_schema()
        system = (
            "Parse Vietnamese video-retrieval queries and return JSON only. Translate faithfully. "
            "Never add colors, clothing, actions, objects or facts absent from the source. Preserve all "
            "numbers and names. số áo=bib number; vận động viên xe đạp=cyclist; đồng hồ thời gian=race clock. "
            "màu xanh=blue unless the Vietnamese source explicitly says xanh lá cây=green. "
            "exact_terms may contain only literal spans copied from the source. objects_en may contain only "
            "visible COCO objects. answer_target is a short enum such as clock_time, visible_text, count, color, "
            "person_identity, location or empty."
        )
        payload = json.dumps({
            "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": query}],
            "temperature": 0, "max_tokens": 160,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "structured_query", "strict": True, "schema": schema}},
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server_port}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        generated = result["choices"][0]["message"]["content"].strip()
        start, end = generated.find("{"), generated.rfind("}")
        return StructuredQuery.model_validate_json(generated[start:end + 1])

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.available:
            raise FileNotFoundError(f"Query Transformer missing: {self.model_path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        try:
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True, device_map="cpu",
                quantization_config=quantization, low_cpu_mem_usage=True).eval()
        except (ImportError, RuntimeError, ValueError):
            # Portable Windows fallback when bitsandbytes is unavailable.
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True).eval()

    def _cached(self, key: str) -> StructuredQuery | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return StructuredQuery.model_validate_json(row[0]) if row else None

    def expand(self, query: str) -> StructuredQuery:
        key = hashlib.sha256(f"qwen2.5-1.5b-q4:v3:{query}".encode("utf-8")).hexdigest()
        cached = self._cached(key)
        if cached:
            return cached
        if not self.available:
            raise QueryTransformerUnavailable(
                f"Qwen is unavailable: model or runtime missing ({self.gguf_path})")
        try:
            if self.use_gguf:
                value = self._expand_gguf(query)
                with sqlite3.connect(self.database) as connection:
                    connection.execute("INSERT OR REPLACE INTO cache VALUES(?,?)",
                                       (key, value.model_dump_json()))
                return value
            self._load()
        except QueryTransformerUnavailable:
            raise
        except Exception as exc:
            raise QueryTransformerError(f"Qwen query transformation failed: {exc}") from exc
        schema = json.dumps(StructuredQuery.model_json_schema(), ensure_ascii=False)
        system = (
            "You parse Vietnamese video-retrieval queries. Return JSON only. Preserve every number, "
            "proper name, quoted string, spatial relation and event order. Do not invent visual details. "
            "Translate clothing faithfully: Vietnamese áo means shirt/top, and quần means pants/trousers; "
            "never change one into the other. Preserve the source clothing item and its color. "
            "query_en and visual_queries_en must be natural English suitable for CLIP retrieval. "
            "exact_terms contains bib numbers, license plates, names and quoted OCR strings. Schema: " + schema
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": query}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        try:
            output = self.model.generate(**inputs, max_new_tokens=320, do_sample=False)
            generated = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True).strip()
            start, end = generated.find("{"), generated.rfind("}")
            value = StructuredQuery.model_validate_json(generated[start:end + 1])
            with sqlite3.connect(self.database) as connection:
                connection.execute("INSERT OR REPLACE INTO cache VALUES(?,?)",
                                   (key, value.model_dump_json()))
            return value
        except Exception as exc:
            raise QueryTransformerError(f"Qwen query transformation failed: {exc}") from exc

    def enhance(self, plan: QueryPlan) -> tuple[QueryPlan, StructuredQuery]:
        value = self.expand(plan.normalized)
        # Generative parsers may suggest useful semantic paraphrases, but they
        # must never invent hard filters. Only literal source spans/numbers may
        # become exact constraints; object filters are restricted to the
        # detector vocabulary already supported by the pipeline.
        source_folded = plan.normalized.casefold()
        safe_exact = []
        for term in value.exact_terms:
            cleaned = " ".join(str(term).strip().split())
            if cleaned and (cleaned.casefold() in source_folded or
                            (cleaned.isdigit() and re.search(
                                rf"(?<!\d){re.escape(cleaned)}(?!\d)", plan.normalized))):
                safe_exact.append(cleaned)
        object_vocabulary = {
            "person", "bicycle", "motorcycle", "car", "bus", "truck", "boat",
            "airplane", "train", "cell phone", "laptop", "dog", "cat", "microphone",
        }
        safe_objects = [item.casefold().strip() for item in value.objects_en
                        if item.casefold().strip() in object_vocabulary]
        generated_variants = [value.query_en, *value.visual_queries_en]
        if "áo" in plan.normalized.casefold() and "quần" not in plan.normalized.casefold():
            generated_variants = [re.sub(r"\b(shorts?|pants?|trousers)\b", "shirt", text,
                                          flags=re.IGNORECASE)
                                  for text in generated_variants]
        if (any(cue in plan.normalized.casefold() for cue in ("màu xanh", "áo xanh"))
                and "xanh lá" not in plan.normalized.casefold()):
            generated_variants = [re.sub(r"\bgreen\b", "blue", text, flags=re.IGNORECASE)
                                  for text in generated_variants]
        if (generated_variants[0] != value.query_en
                or generated_variants[1:] != value.visual_queries_en):
            value = value.model_copy(update={
                "query_en": generated_variants[0],
                "visual_queries_en": generated_variants[1:],
            })
        variants = tuple(dict.fromkeys([*plan.variants, *generated_variants]))
        enhanced = replace(
            plan,
            variants=variants,
            exact_terms=tuple(dict.fromkeys([*plan.exact_terms, *safe_exact])),
            objects=tuple(dict.fromkeys([*plan.objects, *safe_objects])),
            # Deterministic answer-slot detection is more reliable than free
            # generation; use the model only when rules found no target.
            answer_target=plan.answer_target,
            transformer_used=True,
        )
        return enhanced, value
