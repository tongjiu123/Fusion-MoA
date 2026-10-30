"""Keep Anthropic SSE alive and forward caller-supplied Fusion task IDs."""
import asyncio
import os
from contextlib import suppress
from urllib.parse import urlsplit

from litellm.integrations.custom_logger import CustomLogger

if os.environ.get('FUSION_TERMINAL_COMPAT') == '1':
    from stream_compat import install
    install()


class AnthropicHeartbeat(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        if os.environ.get('FUSION_TERMINAL_COMPAT') == '1':
            from stream_compat import install
            install()
        params = kwargs.get("litellm_params") or {}
        if params.get("custom_llm_provider") != "openai":
            return
        request = params.get("proxy_server_request") or {}
        outgoing = (kwargs.get("additional_args") or {}).get("complete_input_dict")
        if not isinstance(outgoing, dict) or not isinstance(outgoing.get("messages"), list):
            return
        if os.environ.get('FUSION_TERMINAL_COMPAT') == '1':
            from stream_compat import normalize_reasoning_history
            outgoing['messages'] = normalize_reasoning_history(outgoing['messages'])

        original_body = request.get("body") or {}
        metadata = original_body.get("metadata") or {}
        if isinstance(metadata, dict) and "task_id" in metadata:
            # An explicit field wins even when empty/invalid; never fall back
            # to a header, user/account identity, or task-name heuristic.
            task_id = metadata["task_id"]
        else:
            task_id = next(
                (
                    value
                    for name, value in (request.get("headers") or {}).items()
                    if name.lower() == "x-fusion-task-id"
                ),
                None,
            )
        if not isinstance(task_id, str) or not task_id:
            return

        outgoing_metadata = outgoing.get("metadata")
        if outgoing_metadata is None:
            outgoing_metadata = {}
        if not isinstance(outgoing_metadata, dict):
            return

        # LiteLLM 1.93.0 OpenAI acompletion/async_streaming calls pre_call
        # synchronously with this same data dict, before SDK create(**data).
        # Do not move this to async_log_pre_api_call: that logging hook may
        # run after serialization. Copy metadata to avoid mutating shared data.
        outgoing["metadata"] = {**outgoing_metadata, "task_id": task_id}

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ):
        request = request_data.get("proxy_server_request", {})
        path = urlsplit(request.get("url", "")).path.rstrip("/")
        if path not in ("/messages", "/v1/messages"):
            async for chunk in response:
                yield chunk
            return

        iterator = response.__aiter__()
        pending = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(iterator.__anext__())
                ready, _ = await asyncio.wait({pending}, timeout=15)
                if not ready:
                    yield b'event: ping\ndata: {"type":"ping"}\n\n'
                    continue
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    break
                pending = None
                yield chunk
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with suppress(asyncio.CancelledError):
                    await pending
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()


proxy_handler_instance = AnthropicHeartbeat()
