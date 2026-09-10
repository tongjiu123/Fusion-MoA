#!/usr/bin/env python3
"""Fusion-MOA v0.9.4.5 generic OpenAI-compatible Gateway."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping
import uuid

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from v0945_next.task_identity import resolve_task_identity
from v0945_next.gateway_integration import (
    GenericMOAOrchestrator,
    ModelEndpoint,
)
from v0945_next.request_controls import (
    normalize_request_body,
    request_uses_thinking,
)
from v0945_next.response_normalizer import (
    ThinkingStreamNormalizer,
    normalize_completion_payload,
)


ROOT = Path(__file__).resolve().parent
MODEL_GENERAL = "fusion-moa-v0945-general"
MODEL_SOLO = "fusion-moa-v0945-solo-d12"
MODEL_GENERAL_V0944_COMPAT = "fusion-moa-v0944-general"
MODEL_SOLO_V0944_COMPAT = "fusion-moa-v0944-solo-d12"
MODEL_GENERAL_V0943_COMPAT = "fusion-moa-v0943-general"
MODEL_SOLO_V0943_COMPAT = "fusion-moa-v0943-solo-d12"
MODEL_GENERAL_V0942_COMPAT = "fusion-moa-v0942-general"
MODEL_SOLO_V0942_COMPAT = "fusion-moa-v0942-solo-d12"
MODEL_POLICIES = {
    MODEL_GENERAL: {"managed": True},
    MODEL_SOLO: {"managed": False},
    MODEL_GENERAL_V0944_COMPAT: {"managed": True},
    MODEL_SOLO_V0944_COMPAT: {"managed": False},
    MODEL_GENERAL_V0943_COMPAT: {"managed": True},
    MODEL_SOLO_V0943_COMPAT: {"managed": False},
    MODEL_GENERAL_V0942_COMPAT: {"managed": True},
    MODEL_SOLO_V0942_COMPAT: {"managed": False},
}

DEFAULT_MODELS = {
    "D12": "Qwen/Qwen3.8-27B",
    "D16": "fusion-moa-d16-kat-coder-v2.5-dev",
    "D9": "TeichAI/Qwen3.6-27B-Fable-5-v1",
    "D13": "Jackrong/Qwopus3.6-35B-A3B-v1",
}
DEFAULT_URLS = {
    "D12": "http://127.0.0.1:8001/v1",
    "D16": "http://127.0.0.1:8002/v1",
    "D9": "http://127.0.0.1:8004/v1",
    "D13": "http://127.0.0.1:8006/v1",
}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def _safe_component(value: Any, fallback: str) -> str:
    if isinstance(value, str) and SAFE_COMPONENT.fullmatch(value):
        return value
    return fallback


RUN_ID = _safe_component(os.environ.get("FUSION_RUN_ID"), "v0945-adhoc")
RUNS_DIR = Path(
    os.environ.get(
        "FUSION_RUNS_DIR",
        str(ROOT / "runs" / "fusion-moa-v0945" / RUN_ID),
    )
)
TRACE_PATH = Path(
    os.environ.get("FUSION_TRACE_PATH", str(RUNS_DIR / "gateway.jsonl"))
)
PORT = int(os.environ.get("FUSION_PORT", "19025"))


def _version_env(name: str, default: str | None = None) -> str | None:
    """Prefer this version's settings without losing the parent env aliases."""
    return os.environ.get(
        name.replace("V0944_", "V0945_", 1),
        os.environ.get(name, default),
    )


def _analyst_cells() -> tuple[str, str, str]:
    configured = _version_env(
        "V0944_ANALYST_CELLS",
        os.environ.get(
            "V0943_ANALYST_CELLS",
            os.environ.get(
                "V0942_ANALYST_CELLS",
                os.environ.get(
                    "V094_ANALYST_CELLS",
                    os.environ.get(
                        "V093_ANALYST_CELLS",
                        os.environ.get(
                            "V092_ANALYST_CELLS",
                            os.environ.get("V09_ANALYST_CELLS", "D16,D9,D13"),
                        ),
                    ),
                ),
            ),
        ),
    )
    cells = tuple(
        item.strip() for item in configured.split(",") if item.strip()
    )
    if (
        len(cells) != 3
        or len(set(cells)) != 3
        or any(not SAFE_COMPONENT.fullmatch(cell) for cell in cells)
    ):
        raise RuntimeError(
            "V0945_ANALYST_CELLS must name exactly three distinct safe Cell IDs"
        )
    return cells


ANALYST_CELLS = _analyst_cells()


def _endpoint(actor: str) -> ModelEndpoint:
    model = _version_env(
        f"V0944_{actor}_MODEL",
        os.environ.get(
            f"V0943_{actor}_MODEL",
            os.environ.get(
                f"V0942_{actor}_MODEL",
                os.environ.get(
                    f"V094_{actor}_MODEL",
                    os.environ.get(
                        f"V093_{actor}_MODEL",
                        os.environ.get(
                            f"V092_{actor}_MODEL",
                            os.environ.get(
                                f"V09_{actor}_MODEL",
                                DEFAULT_MODELS.get(actor, ""),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    base_url = os.environ.get(
        f"{actor}_BASE_URL",
        DEFAULT_URLS.get(actor, ""),
    )
    if not model or not base_url:
        raise RuntimeError(
            f"Cell {actor} requires V0945_{actor}_MODEL and {actor}_BASE_URL"
        )
    return ModelEndpoint(model=model, base_url=base_url)


orchestrator = GenericMOAOrchestrator(
    executor=_endpoint("D12"),
    analysts={actor: _endpoint(actor) for actor in ANALYST_CELLS},
    run_id=RUN_ID,
    trace_path=TRACE_PATH,
    analyst_deadline_seconds=float(
        _version_env(
            "V0944_ANALYST_DEADLINE_SECONDS",
            os.environ.get(
                "V0943_ANALYST_DEADLINE_SECONDS",
                os.environ.get(
                    "V0942_ANALYST_DEADLINE_SECONDS",
                    os.environ.get(
                        "V094_ANALYST_DEADLINE_SECONDS",
                        os.environ.get(
                            "V093_ANALYST_DEADLINE_SECONDS",
                            os.environ.get("V092_ANALYST_DEADLINE_SECONDS", "120"),
                        ),
                    ),
                ),
            ),
        )
    ),
    analyst_max_tokens=int(
        _version_env(
            "V0944_ANALYST_MAX_TOKENS",
            os.environ.get(
                "V0943_ANALYST_MAX_TOKENS",
                os.environ.get(
                    "V0942_ANALYST_MAX_TOKENS",
                    os.environ.get(
                        "V094_ANALYST_MAX_TOKENS",
                        os.environ.get(
                            "V093_ANALYST_MAX_TOKENS",
                            os.environ.get("V092_ANALYST_MAX_TOKENS", "768"),
                        ),
                    ),
                ),
            ),
        )
    ),
    executor_timeout_seconds=float(
        _version_env(
            "V0944_EXECUTOR_TIMEOUT_SECONDS",
            os.environ.get(
                "V0943_EXECUTOR_TIMEOUT_SECONDS",
                os.environ.get(
                    "V0942_EXECUTOR_TIMEOUT_SECONDS",
                    os.environ.get(
                        "V094_EXECUTOR_TIMEOUT_SECONDS",
                        os.environ.get(
                            "V093_EXECUTOR_TIMEOUT_SECONDS",
                            os.environ.get("V092_EXECUTOR_TIMEOUT_SECONDS", "3300"),
                        ),
                    ),
                ),
            ),
        )
    ),
    executor_max_tokens=int(
        _version_env(
            "V0944_EXECUTOR_MAX_TOKENS",
            os.environ.get(
                "V0943_EXECUTOR_MAX_TOKENS",
                os.environ.get(
                    "V0942_EXECUTOR_MAX_TOKENS",
                    os.environ.get(
                        "V094_EXECUTOR_MAX_TOKENS",
                        os.environ.get(
                            "V093_EXECUTOR_MAX_TOKENS",
                            os.environ.get("V092_EXECUTOR_MAX_TOKENS", "8192"),
                        ),
                    ),
                ),
            ),
        )
    ),
    output_recovery_max_tokens=int(
        _version_env(
            "V0944_OUTPUT_RECOVERY_MAX_TOKENS",
            os.environ.get(
                "V0943_OUTPUT_RECOVERY_MAX_TOKENS",
                os.environ.get(
                    "V0942_OUTPUT_RECOVERY_MAX_TOKENS",
                    os.environ.get(
                        "V094_OUTPUT_RECOVERY_MAX_TOKENS",
                        os.environ.get("V093_OUTPUT_RECOVERY_MAX_TOKENS", "512"),
                    ),
                ),
            ),
        )
    ),
    panel_integrator_actor=_version_env(
        "V0944_PANEL_INTEGRATOR_CELL",
        os.environ.get(
            "V0943_PANEL_INTEGRATOR_CELL",
            os.environ.get(
                "V0942_PANEL_INTEGRATOR_CELL",
                os.environ.get("V094_PANEL_INTEGRATOR_CELL"),
            ),
        ),
    ),
)


@asynccontextmanager
async def lifespan(_app):
    try:
        yield
    finally:
        await orchestrator.aclose()


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


def _error(error_type: str, message: str, status_code: int):
    return JSONResponse(
        {"error": {"type": error_type, "message": message}},
        status_code=status_code,
    )


def _sse_data(payload: Any) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"data: {data}\n\n".encode("utf-8")


def _sse_event(event_name: str, payload: Mapping[str, Any]) -> bytes:
    data = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")


def _sse_status(payload: Mapping[str, Any]) -> bytes:
    return _sse_event("moa.status", payload)


def _buffered_completion_chunks(
    payload: Mapping[str, Any],
    *,
    public_model: str,
    include_usage: bool,
    allow_implicit_thinking: bool,
) -> list[dict[str, Any]]:
    payload = normalize_completion_payload(
        payload,
        allow_implicit_thinking=allow_implicit_thinking,
    )
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, Mapping) else {}
    message = choice.get("message")
    message = message if isinstance(message, Mapping) else {}
    chunk_id = payload.get("id")
    if not isinstance(chunk_id, str) or not chunk_id:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = payload.get("created")
    if not isinstance(created, int):
        created = int(time.time())

    def chunk(delta: Mapping[str, Any], finish_reason: Any = None):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "delta": dict(delta),
                    "finish_reason": finish_reason,
                }
            ],
        }

    chunks = [chunk({"role": "assistant"})]
    delta: dict[str, Any] = {}
    for key in ("content", "reasoning_content", "refusal"):
        value = message.get(key)
        if isinstance(value, str) and value:
            delta[key] = value
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        delta["tool_calls"] = [
            {"index": index, **dict(call)}
            for index, call in enumerate(tool_calls)
            if isinstance(call, Mapping)
        ]
    if delta:
        chunks.append(chunk(delta))
    chunks.append(chunk({}, choice.get("finish_reason") or "stop"))
    usage = payload.get("usage")
    if include_usage and isinstance(usage, Mapping):
        chunks.append(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": public_model,
                "choices": [],
                "usage": dict(usage),
            }
        )
    return chunks


class DisconnectAwareStreamingResponse(StreamingResponse):
    """Cancel the SSE generator as soon as ASGI reports a disconnect."""

    async def __call__(self, scope, receive, send) -> None:
        async with anyio.create_task_group() as task_group:
            async def stream_response() -> None:
                try:
                    await self.stream_response(send)
                except OSError:
                    pass
                finally:
                    task_group.cancel_scope.cancel()

            async def listen_for_disconnect() -> None:
                await self.listen_for_disconnect(receive)
                task_group.cancel_scope.cancel()

            task_group.start_soon(stream_response)
            task_group.start_soon(listen_for_disconnect)
        if self.background is not None:
            await self.background()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "v0.9.4.5",
        "run_id": RUN_ID,
        "port": PORT,
        "topology": "single_openai_compatible_gateway",
        "scope": "general_chat_and_tool_use",
        "executor_model": orchestrator.executor.model,
        "boundaries": orchestrator.boundaries,
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "local"}
            for model in MODEL_POLICIES
        ],
    }


@app.get("/metrics")
def metrics():
    return orchestrator.metrics()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    try:
        body = await req.json()
    except Exception:
        return _error("invalid_request", "request body must be valid JSON", 400)
    if not isinstance(body, dict):
        return _error(
            "invalid_request",
            "request body must be a JSON object",
            400,
        )
    try:
        body = normalize_request_body(body)
    except ValueError as exc:
        return _error("invalid_request", str(exc), 400)

    model = body.get("model")
    policy = MODEL_POLICIES.get(model)
    if policy is None:
        return _error("invalid_model", f"unsupported model: {model}", 404)
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(not isinstance(message, Mapping) for message in messages)
    ):
        return _error(
            "invalid_request",
            "messages must be a non-empty array of JSON objects",
            400,
        )
    metadata = (
        body.get("metadata")
        if isinstance(body.get("metadata"), Mapping)
        else {}
    )
    identity = resolve_task_identity(messages, metadata)
    if body.get("stream") is True:
        progress_events = metadata.get("moa_progress_events") is True
        panel_summary_events = (
            metadata.get("moa_panel_summary_events") is True
        )
        stream_options = body.get("stream_options")
        include_usage = (
            isinstance(stream_options, Mapping)
            and stream_options.get("include_usage") is True
        )
        tools = body.get("tools")
        has_tools = isinstance(tools, list) and bool(tools)
        orchestration_body = dict(body)
        if has_tools:
            orchestration_body["stream"] = False
            orchestration_body.pop("stream_options", None)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        allow_implicit_thinking = request_uses_thinking(body)
        thinking_stream = ThinkingStreamNormalizer(
            allow_implicit_thinking=allow_implicit_thinking
        )

        async def stream_sink(item: dict[str, Any]) -> None:
            data = item.get("data")
            if item.get("kind") == "chunk" and isinstance(data, Mapping):
                for normalized in thinking_stream.feed(data):
                    await queue.put({"kind": "chunk", "data": normalized})
                return
            await queue.put(item)

        completion_task = asyncio.create_task(
            orchestrator.complete(
                orchestration_body,
                identity=identity,
                public_model=model,
                managed=bool(policy["managed"]),
                stream_sink=stream_sink,
            )
        )

        async def event_stream():
            emitted_model_chunks = False
            last_keepalive = time.monotonic()
            try:
                while not completion_task.done() or not queue.empty():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if time.monotonic() - last_keepalive >= 15.0:
                            last_keepalive = time.monotonic()
                            yield b": keep-alive\n\n"
                        continue
                    kind = item.get("kind")
                    data = item.get("data")
                    if kind == "chunk" and isinstance(data, Mapping):
                        emitted_model_chunks = True
                        yield _sse_data(dict(data))
                    elif (
                        kind == "status"
                        and progress_events
                        and isinstance(data, Mapping)
                    ):
                        yield _sse_status(data)
                    elif (
                        kind == "panel"
                        and panel_summary_events
                        and isinstance(data, Mapping)
                    ):
                        yield _sse_event("moa.panel", data)

                try:
                    result = completion_task.result()
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    yield _sse_data(
                        {
                            "error": {
                                "type": "gateway_stream_error",
                                "message": str(exc)[:240],
                            }
                        }
                    )
                    yield _sse_data("[DONE]")
                    return

                for chunk in thinking_stream.flush():
                    emitted_model_chunks = True
                    yield _sse_data(chunk)

                if not emitted_model_chunks:
                    if 200 <= result.status_code < 300:
                        for chunk in _buffered_completion_chunks(
                            result.payload,
                            public_model=model,
                            include_usage=include_usage,
                            allow_implicit_thinking=allow_implicit_thinking,
                        ):
                            yield _sse_data(chunk)
                    else:
                        yield _sse_data(result.payload)
                elif not 200 <= result.status_code < 300:
                    yield _sse_data(result.payload)
                if progress_events:
                    yield _sse_status(
                        {"stage": "completed", "message": "本次请求已完成"}
                    )
                yield _sse_data("[DONE]")
            finally:
                if not completion_task.done():
                    completion_task.cancel()
                    await asyncio.gather(
                        completion_task,
                        return_exceptions=True,
                    )

        return DisconnectAwareStreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-MOA-Stream": "true",
                "X-Fusion-MOA-Version": "v0.9.4.5",
            },
        )

    completion_task = asyncio.create_task(
        orchestrator.complete(
            body,
            identity=identity,
            public_model=model,
            managed=bool(policy["managed"]),
        )
    )
    try:
        while True:
            done, _ = await asyncio.wait(
                {completion_task},
                timeout=0.25,
            )
            if completion_task in done:
                result = completion_task.result()
                break
            if await req.is_disconnected():
                completion_task.cancel()
                with suppress(asyncio.CancelledError):
                    await completion_task
                return _error(
                    "client_disconnected",
                    "client disconnected before completion",
                    499,
                )
    finally:
        if not completion_task.done():
            completion_task.cancel()
            await asyncio.gather(completion_task, return_exceptions=True)
    response_headers = {
        **result.headers,
        "X-Fusion-MOA-Version": "v0.9.4.5",
    }
    return JSONResponse(
        normalize_completion_payload(
            result.payload,
            allow_implicit_thinking=request_uses_thinking(body),
        ),
        status_code=result.status_code,
        headers=response_headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
