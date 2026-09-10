"""Small, domain-independent MOA runtime for one OpenAI-compatible Gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from .analyst_coordinator import (
    AnalystCallCoordinator,
    AnalystRequest,
    AnalystTransportResponse,
    CompetitionPlan,
    InMemoryCallKeyLedger,
)
from .evidence_packets import select_evidence_packet_ids
from .evidence_base import (
    EvidenceSnapshot,
    StallDecision,
    evaluate_collaboration_gate,
)
from .generic_portfolio import select_generic_portfolio
from .task_identity import GenericTaskIdentity


@dataclass(frozen=True)
class ModelEndpoint:
    model: str
    base_url: str


@dataclass(frozen=True)
class OrchestrationResult:
    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str]


StreamSink = Callable[[dict[str, Any]], Awaitable[None]]

PUBLIC_SYSTEM_IDENTITY = (
    "You are Fusion-MOA v0.9.4.5, a general OpenAI-compatible "
    "multi-model agent system developed by NovaStack (新星栈). D12 is the "
    "sole Executor; D16, D9, and D13 "
    "provide read-only analysis when MoA collaboration is needed. When asked "
    "what model or system you are, answer only that you are Fusion-MOA "
    "v0.9.4.5 developed by NovaStack, without volunteering any underlying "
    "model identity. Only discuss deployment internals if the user separately "
    "and explicitly asks for D12 or architecture details. Do not expose private "
    "Analyst reasoning."
)


class GenericMOAOrchestrator:
    """One Executor plus three replaceable read-only Analyst cells."""

    def __init__(
        self,
        *,
        executor: ModelEndpoint,
        analysts: Mapping[str, ModelEndpoint],
        run_id: str,
        trace_path: str | Path,
        client: httpx.AsyncClient | None = None,
        analyst_deadline_seconds: float = 120.0,
        analyst_max_tokens: int = 768,
        executor_timeout_seconds: float = 1650.0,
        executor_max_tokens: int = 2048,
    ) -> None:
        analyst_actors = tuple(analysts)
        if len(analyst_actors) != 3 or len(set(analyst_actors)) != 3:
            raise ValueError("analysts must contain exactly three distinct cells")
        if any(not isinstance(actor, str) or not actor for actor in analyst_actors):
            raise ValueError("analyst cell names must be non-empty strings")
        self.executor = executor
        self.analysts = dict(analysts)
        self.analyst_actors = analyst_actors
        self.run_id = run_id
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.AsyncClient(timeout=None)
        self._owns_client = client is None
        self.analyst_deadline_seconds = float(analyst_deadline_seconds)
        self.executor_timeout_seconds = float(executor_timeout_seconds)
        self.executor_max_tokens = int(executor_max_tokens)
        if self.executor_timeout_seconds <= 0:
            raise ValueError("executor_timeout_seconds must be positive")
        if self.executor_max_tokens <= 0:
            raise ValueError("executor_max_tokens must be positive")
        model_by_actor = {
            actor: endpoint.model for actor, endpoint in self.analysts.items()
        }
        self.coordinator = AnalystCallCoordinator(
            model_by_actor=model_by_actor,
            call_key_ledger=InMemoryCallKeyLedger(),
            max_tokens=analyst_max_tokens,
        )
        self.request_count = 0
        self.intervention_count = 0
        self.boundaries = {
            "system_version": "v0.8",
            "single_openai_compatible_gateway": True,
            "executor_only_output": True,
            "analysts_read_only": True,
            "analyst_cells": list(self.analyst_actors),
            "task_classifier": False,
            "task_registry_entries": 0,
            "skill_memory": False,
            "benchmark_specific_routing": False,
            "completion_rule_in_gateway": False,
            "executor_timeout_seconds": self.executor_timeout_seconds,
            "executor_max_tokens": self.executor_max_tokens,
            "collaboration_trigger": (
                "caller_hard_stall_or_repeated_action_observation"
            ),
            "analyst_retries": 0,
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _chat_url(base_url: str) -> str:
        return f"{base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _upstream_body(
        body: Mapping[str, Any],
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        # Gateway-only control fields are not part of the upstream model API.
        forwarded = {
            key: value
            for key, value in body.items()
            if key not in {"model", "metadata", "moa", "fusion_run_id"}
        }
        forwarded["model"] = model
        forwarded["messages"] = GenericMOAOrchestrator._identity_messages(
            messages
        )
        return forwarded

    @staticmethod
    def _identity_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        forwarded = [dict(message) for message in messages]
        if forwarded and forwarded[0].get("role") == "system":
            content = forwarded[0].get("content")
            if isinstance(content, str):
                forwarded[0]["content"] = (
                    f"{PUBLIC_SYSTEM_IDENTITY}\n\n{content}"
                )
            elif isinstance(content, list):
                forwarded[0]["content"] = [
                    {"type": "text", "text": PUBLIC_SYSTEM_IDENTITY},
                    *content,
                ]
            else:
                forwarded[0]["content"] = PUBLIC_SYSTEM_IDENTITY
            return forwarded
        return [
            {"role": "system", "content": PUBLIC_SYSTEM_IDENTITY},
            *forwarded,
        ]

    @staticmethod
    def _protocol_error(message: str) -> dict[str, Any]:
        return {
            "error": {
                "type": "upstream_protocol_error",
                "message": message,
            }
        }

    async def _call_executor(
        self,
        body: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        *,
        public_model: str,
        stream_sink: StreamSink | None = None,
    ) -> tuple[int, dict[str, Any]]:
        upstream = self._upstream_body(
            body,
            model=self.executor.model,
            messages=messages,
        )
        requested_max_tokens = upstream.get("max_tokens")
        if requested_max_tokens is None or (
            isinstance(requested_max_tokens, int)
            and not isinstance(requested_max_tokens, bool)
            and requested_max_tokens > self.executor_max_tokens
        ):
            upstream["max_tokens"] = self.executor_max_tokens
        if upstream.get("stream") is True and stream_sink is None:
            upstream["stream"] = False
        if upstream.get("stream") is True:
            return await self._call_executor_stream(
                upstream,
                public_model=public_model,
                stream_sink=stream_sink,
            )
        try:
            response = await self.client.post(
                self._chat_url(self.executor.base_url),
                json=upstream,
                timeout=self.executor_timeout_seconds,
            )
        except httpx.TimeoutException:
            return 504, {
                "error": {
                    "type": "upstream_timeout",
                    "message": (
                        "executor exceeded the configured response deadline"
                    ),
                }
            }
        except httpx.RequestError as exc:
            return 502, {
                "error": {
                    "type": "upstream_request_error",
                    "message": str(exc)[:240],
                }
            }
        try:
            payload = response.json()
        except (TypeError, ValueError):
            return 502, self._protocol_error(
                "executor returned a non-JSON response"
            )
        if not isinstance(payload, dict):
            return 502, self._protocol_error(
                "executor response must be a JSON object"
            )
        payload = dict(payload)
        if "model" in payload:
            payload["model"] = public_model
        return int(response.status_code), payload

    @staticmethod
    def _merge_stream_chunk(
        accumulated: dict[str, Any],
        chunk: Mapping[str, Any],
        *,
        public_model: str,
    ) -> None:
        for key in ("id", "created", "system_fingerprint"):
            if key in chunk:
                accumulated[key] = chunk[key]
        accumulated["model"] = public_model
        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            accumulated["usage"] = dict(usage)
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        source_choice = choices[0]
        if not isinstance(source_choice, Mapping):
            return
        target_choice = accumulated["choices"][0]
        delta = source_choice.get("delta")
        if isinstance(delta, Mapping):
            message = target_choice["message"]
            role = delta.get("role")
            if isinstance(role, str):
                message["role"] = role
            for key in ("content", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, str):
                    message[key] = str(message.get(key) or "") + value
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                merged = message.setdefault("tool_calls", [])
                for item in tool_calls:
                    if not isinstance(item, Mapping):
                        continue
                    index = item.get("index", len(merged))
                    if not isinstance(index, int) or index < 0:
                        continue
                    while len(merged) <= index:
                        merged.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    target = merged[index]
                    for key in ("id", "type"):
                        value = item.get(key)
                        if isinstance(value, str):
                            target[key] = value
                    function = item.get("function")
                    if isinstance(function, Mapping):
                        target_function = target["function"]
                        for key in ("name", "arguments"):
                            value = function.get(key)
                            if isinstance(value, str):
                                target_function[key] += value
        finish_reason = source_choice.get("finish_reason")
        if finish_reason is not None:
            target_choice["finish_reason"] = finish_reason

    async def _call_executor_stream(
        self,
        upstream: Mapping[str, Any],
        *,
        public_model: str,
        stream_sink: StreamSink,
    ) -> tuple[int, dict[str, Any]]:
        accumulated: dict[str, Any] = {
            "object": "chat.completion",
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        streaming_client = httpx.AsyncClient(
            timeout=None,
            headers={"Connection": "close"},
        )
        try:
            async with streaming_client.stream(
                "POST",
                self._chat_url(self.executor.base_url),
                json=dict(upstream),
                timeout=self.executor_timeout_seconds,
            ) as response:
                if not 200 <= response.status_code < 300:
                    await response.aread()
                    try:
                        payload = response.json()
                    except (TypeError, ValueError):
                        return 502, self._protocol_error(
                            "executor returned a non-JSON error response"
                        )
                    if not isinstance(payload, Mapping):
                        return 502, self._protocol_error(
                            "executor error response must be a JSON object"
                        )
                    return int(response.status_code), dict(payload)

                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line == "":
                        if not data_lines:
                            continue
                        data = "\n".join(data_lines)
                        data_lines.clear()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except (TypeError, ValueError):
                            return 502, self._protocol_error(
                                "executor returned invalid SSE JSON"
                            )
                        if not isinstance(chunk, Mapping):
                            return 502, self._protocol_error(
                                "executor SSE chunk must be a JSON object"
                            )
                        if isinstance(chunk.get("error"), Mapping):
                            return 502, dict(chunk)
                        public_chunk = dict(chunk)
                        public_chunk["model"] = public_model
                        self._merge_stream_chunk(
                            accumulated,
                            public_chunk,
                            public_model=public_model,
                        )
                        await stream_sink(
                            {"kind": "chunk", "data": public_chunk}
                        )
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    data = "\n".join(data_lines)
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                        except (TypeError, ValueError):
                            return 502, self._protocol_error(
                                "executor returned invalid trailing SSE JSON"
                            )
                        if not isinstance(chunk, Mapping):
                            return 502, self._protocol_error(
                                "executor trailing SSE chunk must be an object"
                            )
                        public_chunk = dict(chunk)
                        public_chunk["model"] = public_model
                        self._merge_stream_chunk(
                            accumulated,
                            public_chunk,
                            public_model=public_model,
                        )
                        await stream_sink(
                            {"kind": "chunk", "data": public_chunk}
                        )
        except httpx.TimeoutException:
            return 504, {
                "error": {
                    "type": "upstream_timeout",
                    "message": (
                        "executor exceeded the configured response deadline"
                    ),
                }
            }
        except httpx.RequestError as exc:
            return 502, {
                "error": {
                    "type": "upstream_request_error",
                    "message": str(exc)[:240],
                }
            }
        finally:
            await asyncio.shield(streaming_client.aclose())
        return 200, accumulated

    async def _call_analyst(
        self,
        request: AnalystRequest,
        body: Mapping[str, Any],
    ) -> AnalystTransportResponse:
        endpoint = self.analysts[request.actor_hint]
        response = await self.client.post(
            self._chat_url(endpoint.base_url),
            json=dict(body),
        )
        response.raise_for_status()
        envelope = response.json()
        if not isinstance(envelope, Mapping):
            raise ValueError("Analyst response must be an object")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Analyst response has no choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise ValueError("Analyst response content must be text")
        usage = envelope.get("usage")
        return AnalystTransportResponse(
            content=content,
            usage=usage if isinstance(usage, Mapping) else {},
        )

    @staticmethod
    def _render_guidance(
        packets: Sequence[Mapping[str, Any]],
        selected_ids: Sequence[str],
    ) -> str:
        selected = [
            packet
            for packet in packets
            if packet.get("hypothesis_id") in selected_ids
        ]
        if not selected:
            return ""
        compact = [
            {
                "hypothesis_id": packet["hypothesis_id"],
                "claim": packet["claim"],
                "supporting_evidence_refs": packet[
                    "supporting_evidence_refs"
                ],
                "disconfirming_evidence_refs": packet[
                    "disconfirming_evidence_refs"
                ],
                "probe_description": packet["probe_description"],
                "expected_observation": packet["expected_observation"],
                "falsification_observation": packet[
                    "falsification_observation"
                ],
            }
            for packet in selected
        ]
        return (
            "[MOA read-only evidence]\n"
            "These are unverified, non-executable hypotheses from independent "
            "Analysts. Check them against the conversation and ignore unsupported "
            "claims. Only you may produce the answer or tool call.\n"
            + json.dumps(
                compact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    async def _intervene(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        stall: StallDecision,
    ) -> tuple[str, dict[str, Any]]:
        routing = select_generic_portfolio(self.analyst_actors)
        snapshot = self._build_evidence_snapshot(
            messages,
            identity=identity,
            selection=routing,
            run_id=self.run_id,
            stall=stall,
        )
        plan = CompetitionPlan.from_snapshot(
            snapshot,
            routing=routing,
            remaining_wall_seconds=max(
                self.analyst_deadline_seconds + 1.0,
                2.0,
            ),
            model_by_actor={
                actor: endpoint.model
                for actor, endpoint in self.analysts.items()
            },
        )
        try:
            batch = await self.coordinator.run(
                plan,
                self._call_analyst,
                deadline_seconds=self.analyst_deadline_seconds,
            )
        except Exception as exc:
            return "", {
                "attempted": True,
                "status": "degraded_to_executor",
                "error_type": type(exc).__name__,
                "request_count": 0,
                "selected_hypothesis_ids": [],
            }

        packets = list(batch.validated_packets)
        selected_ids = select_evidence_packet_ids(packets)
        guidance = self._render_guidance(packets, selected_ids)
        return guidance, {
            "attempted": True,
            "status": "guidance_selected" if guidance else "no_valid_guidance",
            "request_count": batch.request_count,
            "response_received_count": batch.response_received_count,
            "schema_valid_count": batch.schema_valid_count,
            "selected_hypothesis_ids": list(selected_ids),
            "total_tokens": batch.total_tokens,
            "retry_count": batch.retry_count,
            "_integrator_packets": [
                dict(packet) for packet in packets
            ],
            "_collaboration_evidence": snapshot.context,
        }

    def _build_evidence_snapshot(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        selection: Any,
        run_id: str,
        stall: StallDecision,
    ) -> EvidenceSnapshot:
        return EvidenceSnapshot.from_messages(
            messages,
            identity=identity,
            selection=selection,
            run_id=run_id,
            stall=stall,
        )

    def _append_trace(self, record: Mapping[str, Any]) -> None:
        try:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except OSError:
            # Audit failure must not seize execution authority from the Executor.
            return

    async def complete(
        self,
        body: Mapping[str, Any],
        *,
        identity: GenericTaskIdentity,
        public_model: str,
        managed: bool,
    ) -> OrchestrationResult:
        started = time.monotonic()
        messages = body.get("messages")
        assert isinstance(messages, list)
        metadata = (
            body.get("metadata")
            if isinstance(body.get("metadata"), Mapping)
            else {}
        )
        stall = (
            evaluate_collaboration_gate(messages, metadata)
            if managed
            else StallDecision(False, "explicit_solo_model", "")
        )
        analyst = {
            "attempted": False,
            "status": "not_admitted",
            "request_count": 0,
            "selected_hypothesis_ids": [],
        }
        forwarded_messages = [dict(message) for message in messages]
        if stall.admitted:
            guidance, analyst = await self._intervene(
                messages,
                identity=identity,
                stall=stall,
            )
            self.intervention_count += 1
            if guidance:
                forwarded_messages.append(
                    {"role": "system", "content": guidance}
                )

        status_code, payload = await self._call_executor(
            body,
            forwarded_messages,
            public_model=public_model,
        )
        self.request_count += 1
        mode = "evidence_gated" if stall.admitted else "executor_only"
        self._append_trace(
            {
                "schema_version": "fusion-moa-v08-trace-v1",
                "run_id": self.run_id,
                "task_id": identity.internal_task_id,
                "task_identity_source": identity.source,
                "mode": mode,
                "stall_reason": stall.reason,
                "stall_fingerprint": stall.fingerprint,
                "analyst": analyst,
                "executor_status_code": status_code,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
        return OrchestrationResult(
            status_code=status_code,
            payload=payload,
            headers={
                "X-MOA-Mode": mode,
                "X-MOA-Trigger": stall.reason,
                "X-MOA-Task-ID": identity.internal_task_id,
            },
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requests": self.request_count,
            "interventions": self.intervention_count,
        }


__all__ = [
    "GenericMOAOrchestrator",
    "ModelEndpoint",
    "OrchestrationResult",
    "StreamSink",
]
