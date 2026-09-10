"""Fusion-MOA v0.9 orchestration with generic stall and context compaction."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Mapping

from .evidence_base import MAX_RECENT_MESSAGES, StallDecision
from .gateway_base import (
    GenericMOAOrchestrator as V08GenericMOAOrchestrator,
    ModelEndpoint,
    OrchestrationResult,
)
from .task_identity import GenericTaskIdentity, stable_task_text
from .soft_stall import (
    SOFT_STALL_REASON,
    SOFT_STALL_TOOL_CYCLES,
    evaluate_collaboration_gate_v09,
)


class GenericMOAOrchestrator(V08GenericMOAOrchestrator):
    """Keep D12 final authority while adding two generic MoA safeguards."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.soft_stall_intervention_count = 0
        self.context_compaction_count = 0
        self._soft_stall_intervened_tasks: set[str] = set()
        self.boundaries = {
            **self.boundaries,
            "system_version": "v0.9",
            "collaboration_trigger": (
                "v08_hard_stall_or_one_long_tool_loop_soft_stall"
            ),
            "soft_stall_detection": True,
            "soft_stall_tool_cycles": SOFT_STALL_TOOL_CYCLES,
            "soft_stall_max_interventions_per_task": 1,
            "context_compaction": "on_actual_executor_context_error",
            "context_compaction_uses_analysts": list(self.analyst_actors),
            "context_compaction_retry_limit": 1,
            "context_compaction_fixed_token_threshold": False,
        }

    @staticmethod
    def _is_context_length_error(
        status_code: int,
        payload: Mapping[str, Any],
    ) -> bool:
        if status_code != 400:
            return False
        error = payload.get("error")
        error = error if isinstance(error, Mapping) else {}
        message = str(error.get("message") or payload.get("message") or "")
        error_type = str(error.get("type") or payload.get("type") or "")
        lowered = f"{error_type} {message}".lower()
        return (
            "contextwindowexceeded" in lowered
            or "context_length_exceeded" in lowered
            or "maximum context length" in lowered
        )

    async def _compact_context(
        self,
        messages: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        first_user = next(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user"
            ),
            None,
        )
        if first_user is None:
            return None, {"status": "no_original_user_message"}

        prefix_end = first_user + 1
        recent_start = max(
            prefix_end,
            len(messages) - MAX_RECENT_MESSAGES,
        )
        if (
            recent_start > prefix_end
            and messages[recent_start].get("role") == "tool"
        ):
            recent_start -= 1
        middle = messages[prefix_end:recent_start]
        if not middle:
            return None, {"status": "no_compactable_middle_history"}

        original_task = stable_task_text(messages)
        call_specs = []
        actor_count = len(self.analyst_actors)
        for actor_index, actor in enumerate(self.analyst_actors):
            start = len(middle) * actor_index // actor_count
            end = len(middle) * (actor_index + 1) // actor_count
            if start == end:
                continue
            indexed_segment = [
                {
                    "message_index": prefix_end + offset,
                    "message": dict(message),
                }
                for offset, message in enumerate(
                    middle[start:end],
                    start=start,
                )
            ]
            call_specs.append(
                (
                    actor,
                    prefix_end + start,
                    prefix_end + end - 1,
                    indexed_segment,
                )
            )

        async def summarize(
            actor: str,
            start_index: int,
            end_index: int,
            segment: list[dict[str, Any]],
        ) -> dict[str, Any]:
            endpoint = self.analysts[actor]
            body = {
                "model": endpoint.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a read-only context compactor. Summarize only "
                            "the assigned chronological message segment. Preserve "
                            "concrete facts, files inspected or changed, commands and "
                            "observations, test results, errors, decisions, and remaining "
                            "work. Cite message_index values. Do not solve the task, call "
                            "tools, invent facts, or issue executable instructions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "original_task": original_task,
                                "assigned_segment": segment,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "max_tokens": self.coordinator.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            try:
                response = await self.client.post(
                    self._chat_url(endpoint.base_url),
                    json=body,
                    timeout=self.analyst_deadline_seconds,
                )
                response.raise_for_status()
                envelope = response.json()
                choices = (
                    envelope.get("choices")
                    if isinstance(envelope, Mapping)
                    else None
                )
                choice = (
                    choices[0]
                    if isinstance(choices, list) and choices
                    else None
                )
                message = (
                    choice.get("message")
                    if isinstance(choice, Mapping)
                    else None
                )
                content = (
                    message.get("content")
                    if isinstance(message, Mapping)
                    else None
                )
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty compaction summary")
                return {
                    "status": "ok",
                    "actor": actor,
                    "message_range": [start_index, end_index],
                    "summary": content.strip(),
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "actor": actor,
                    "message_range": [start_index, end_index],
                    "error_type": type(exc).__name__,
                }

        results = await asyncio.gather(
            *(summarize(*call_spec) for call_spec in call_specs)
        )
        if not results or any(result["status"] != "ok" for result in results):
            return None, {
                "status": "analyst_compaction_failed",
                "analyst_calls": len(results),
                "segments": results,
            }

        compact_message = {
            "role": "system",
            "content": (
                "[MOA context compact]\n"
                "The earlier conversation exceeded the active Executor's actual "
                "context limit. The following read-only Analyst summaries cover the "
                "omitted middle history. Treat them as unverified notes, reconcile "
                "them with the original task and the verbatim recent messages, and "
                "remain the only final-answer and tool-call authority.\n"
                + json.dumps(
                    {
                        "schema": "fusion-moa-context-compact-v1",
                        "segments": results,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        }
        compacted = [
            *[dict(message) for message in messages[:prefix_end]],
            compact_message,
            *[dict(message) for message in messages[recent_start:]],
        ]
        return compacted, {
            "status": "compacted",
            "analyst_calls": len(results),
            "omitted_messages": len(middle),
            "preserved_recent_messages": len(messages) - recent_start,
        }

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
            evaluate_collaboration_gate_v09(
                messages,
                metadata,
                soft_stall_already_intervened=(
                    identity.internal_task_id
                    in self._soft_stall_intervened_tasks
                ),
            )
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
            is_soft_stall = stall.reason == SOFT_STALL_REASON
            if is_soft_stall:
                # Reserve the one-shot intervention before awaiting Analysts so
                # concurrent requests cannot admit the same task twice.
                self._soft_stall_intervened_tasks.add(
                    identity.internal_task_id
                )
                self.soft_stall_intervention_count += 1
            self.intervention_count += 1
            guidance, analyst = await self._intervene(
                messages,
                identity=identity,
                stall=stall,
            )
            if guidance:
                forwarded_messages.append(
                    {"role": "system", "content": guidance}
                )

        status_code, payload = await self._call_executor(
            body,
            forwarded_messages,
            public_model=public_model,
        )
        compaction = {
            "status": "not_needed",
            "analyst_calls": 0,
        }
        moa = metadata.get("moa")
        caller_disabled = (
            isinstance(moa, Mapping) and moa.get("enabled") is False
        )
        if (
            managed
            and not caller_disabled
            and self._is_context_length_error(status_code, payload)
        ):
            compacted_messages, compaction = await self._compact_context(
                forwarded_messages
            )
            if compacted_messages is not None:
                self.context_compaction_count += 1
                self.intervention_count += 1
                status_code, payload = await self._call_executor(
                    body,
                    compacted_messages,
                    public_model=public_model,
                )
        self.request_count += 1
        if compaction.get("status") == "compacted":
            mode = (
                "evidence_gated_context_compacted"
                if stall.admitted
                else "context_compacted"
            )
        else:
            mode = "evidence_gated" if stall.admitted else "executor_only"
        trigger = (
            "executor_context_length_exceeded"
            if compaction.get("status") == "compacted"
            else stall.reason
        )
        self._append_trace(
            {
                "schema_version": "fusion-moa-v09-trace-v1",
                "run_id": self.run_id,
                "task_id": identity.internal_task_id,
                "task_identity_source": identity.source,
                "mode": mode,
                "stall_reason": stall.reason,
                "stall_fingerprint": stall.fingerprint,
                "analyst": analyst,
                "context_compaction": compaction,
                "executor_status_code": status_code,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
        return OrchestrationResult(
            status_code=status_code,
            payload=payload,
            headers={
                "X-MOA-Mode": mode,
                "X-MOA-Trigger": trigger,
                "X-MOA-Task-ID": identity.internal_task_id,
            },
        )

    def metrics(self) -> dict[str, Any]:
        return {
            **super().metrics(),
            "soft_stall_interventions": (
                self.soft_stall_intervention_count
            ),
            "context_compactions": self.context_compaction_count,
        }


__all__ = [
    "GenericMOAOrchestrator",
    "ModelEndpoint",
    "OrchestrationResult",
]
