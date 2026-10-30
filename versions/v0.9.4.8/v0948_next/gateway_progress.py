"""Fusion-MOA v0.9.1 progress-gated orchestration."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping, Sequence

from .gateway_base import (
    ModelEndpoint,
    OrchestrationResult,
)
from .task_identity import GenericTaskIdentity
from .gateway_context import (
    GenericMOAOrchestrator as V09GenericMOAOrchestrator,
)

from .evidence_episode import (
    StallDecision,
    action_item_signatures,
)
from .soft_stall import (
    SOFT_STALL_REASON,
    SOFT_STALL_TOOL_CYCLES,
    count_tool_cycles,
    evaluate_collaboration_gate_v091,
)


class GenericMOAOrchestrator(V09GenericMOAOrchestrator):
    """Add bounded state progression without changing D12 final authority."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_soft_stall_round: dict[str, int] = {}
        self._hard_episode_by_task: dict[str, dict[str, Any]] = {}
        self.orchestration_event_count = 0
        self.panel_event_count = 0
        self.soft_stall_event_count = 0
        self.hard_stall_event_count = 0
        self.analyst_request_count = 0
        self.d12_internal_retry_count = 0
        self.action_resample_count = 0
        self.boundaries = {
            **self.boundaries,
            "system_version": "v0.9.1",
            "collaboration_trigger": (
                "hard_stall_episode_or_periodic_long_tool_loop_soft_stall"
            ),
            "hard_stall_episode_guidance_reuse": True,
            "soft_stall_max_interventions_per_task": None,
            "soft_stall_interval_tool_cycles": SOFT_STALL_TOOL_CYCLES,
            "progress_gated_repeat_guard": True,
            "repeat_reason_marker": "REPEAT_REASON:",
            "progress_gated_action_resample_limit": 1,
            "supervisor": False,
            "long_term_memory": False,
        }

    @staticmethod
    def _repeat_guard_context(stall: StallDecision) -> str:
        stalled = stall.stalled_action
        observation = stall.last_observation
        if not isinstance(stalled, Mapping):
            return ""
        return (
            "[MOA progress-gated repeat guard]\n"
            "The following action and observation define the current stall "
            f"episode at state_epoch={stall.state_epoch}.\n"
            "stalled_action="
            + json.dumps(
                stalled,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\nlast_observation="
            + json.dumps(
                observation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\nRepeating an action is not forbidden. Repeat it only when a "
            "relevant precondition has changed, or when the repetition is an "
            "intentional retry, polling operation, or nondeterminism check. "
            "If the state has not changed and repetition is intentional, include "
            "`REPEAT_REASON:` in assistant.content and state the expected new "
            "information. Otherwise choose a materially different action. D12 "
            "remains the only tool-call and final-answer authority."
        )

    @staticmethod
    def _resample_instruction() -> str:
        return (
            "[MOA progress-gate retry]\n"
            "You selected the same action that already produced the same "
            "observation. Repeating it is allowed only if a relevant precondition "
            "has changed, or the repetition is an intentional retry, polling "
            "operation, or nondeterminism check. Otherwise choose a materially "
            "different action. If repetition is intentional, state "
            "`REPEAT_REASON:` and the expected new information. This is the only "
            "internal resampling attempt."
        )

    @staticmethod
    def _response_message(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return None
        message = choice.get("message")
        return message if isinstance(message, Mapping) else None

    @classmethod
    def _has_repeat_reason(cls, payload: Mapping[str, Any]) -> bool:
        message = cls._response_message(payload)
        if message is None:
            return False
        content = message.get("content")
        return (
            isinstance(content, str)
            and re.search(
                r"(?im)^\s*REPEAT_REASON\s*:\s*\S",
                content,
            )
            is not None
        )

    @classmethod
    def _repeats_stalled_action(
        cls,
        payload: Mapping[str, Any],
        stalled_signatures: Sequence[str],
    ) -> bool:
        if not stalled_signatures:
            return False
        message = cls._response_message(payload)
        if message is None:
            return False
        proposed = set(action_item_signatures(message))
        return bool(proposed.intersection(stalled_signatures))

    async def _panel_guidance(
        self,
        messages: list[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        stall: StallDecision,
    ) -> tuple[str, dict[str, Any], bool]:
        task_id = identity.internal_task_id
        is_soft = stall.reason == SOFT_STALL_REASON
        if is_soft:
            self._hard_episode_by_task.pop(task_id, None)
            self._last_soft_stall_round[task_id] = count_tool_cycles(messages)
            self.soft_stall_event_count += 1
            should_call_panel = True
        else:
            self.hard_stall_event_count += 1
            episode_id = stall.episode_id or stall.fingerprint
            existing = self._hard_episode_by_task.get(task_id)
            if existing and existing["episode_id"] == episode_id:
                return (
                    str(existing.get("guidance") or ""),
                    {
                        "attempted": False,
                        "status": "episode_guidance_reused",
                        "request_count": 0,
                        "selected_hypothesis_ids": list(
                            existing.get("selected_hypothesis_ids") or []
                        ),
                    },
                    True,
                )
            self._hard_episode_by_task[task_id] = {
                "episode_id": episode_id,
                "guidance": "",
                "selected_hypothesis_ids": [],
            }
            should_call_panel = True

        assert should_call_panel
        self.panel_event_count += 1
        guidance, analyst = await self._intervene(
            messages,
            identity=identity,
            stall=stall,
        )
        self.analyst_request_count += int(
            analyst.get("request_count", 0) or 0
        )
        if not is_soft:
            current = self._hard_episode_by_task.get(task_id)
            if (
                current is not None
                and current["episode_id"]
                == (stall.episode_id or stall.fingerprint)
            ):
                current["guidance"] = guidance
                current["selected_hypothesis_ids"] = list(
                    analyst.get("selected_hypothesis_ids") or []
                )
        return guidance, analyst, False

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
        task_id = identity.internal_task_id
        stall = (
            evaluate_collaboration_gate_v091(
                messages,
                metadata,
                last_soft_stall_round=self._last_soft_stall_round.get(
                    task_id
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
        panel_reused = False
        forwarded_messages = [dict(message) for message in messages]
        if stall.admitted:
            self.intervention_count += 1
            self.orchestration_event_count += 1
            guidance, analyst, panel_reused = await self._panel_guidance(
                messages,
                identity=identity,
                stall=stall,
            )
            repeat_guard = self._repeat_guard_context(stall)
            combined = "\n\n".join(
                part for part in (guidance, repeat_guard) if part
            )
            if combined:
                forwarded_messages.append(
                    {"role": "system", "content": combined}
                )
        elif managed:
            self._hard_episode_by_task.pop(task_id, None)

        executor_messages = forwarded_messages
        status_code, payload = await self._call_executor(
            body,
            executor_messages,
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
                executor_messages
            )
            self.analyst_request_count += int(
                compaction.get("analyst_calls", 0) or 0
            )
            if compacted_messages is not None:
                executor_messages = compacted_messages
                self.context_compaction_count += 1
                self.intervention_count += 1
                self.orchestration_event_count += 1
                self.d12_internal_retry_count += 1
                status_code, payload = await self._call_executor(
                    body,
                    executor_messages,
                    public_model=public_model,
                )

        repeat_guard = {
            "attempted": False,
            "status": "not_needed",
        }
        if (
            managed
            and stall.admitted
            and 200 <= status_code < 300
            and self._repeats_stalled_action(
                payload,
                stall.stalled_action_signatures,
            )
        ):
            if self._has_repeat_reason(payload):
                repeat_guard["status"] = "allowed_with_reason"
            else:
                repeat_guard = {
                    "attempted": True,
                    "status": "resample_requested",
                }
                self.action_resample_count += 1
                self.orchestration_event_count += 1
                self.d12_internal_retry_count += 1
                retry_messages = [
                    *executor_messages,
                    {
                        "role": "system",
                        "content": self._resample_instruction(),
                    },
                ]
                status_code, payload = await self._call_executor(
                    body,
                    retry_messages,
                    public_model=public_model,
                )
                if not 200 <= status_code < 300:
                    repeat_guard["status"] = "retry_error"
                elif self._repeats_stalled_action(
                    payload,
                    stall.stalled_action_signatures,
                ):
                    if self._has_repeat_reason(payload):
                        repeat_guard["status"] = "allowed_with_reason_on_retry"
                    else:
                        repeat_guard["status"] = "retry_exhausted"
                else:
                    repeat_guard["status"] = "changed_action"

        self.request_count += 1
        if compaction.get("status") == "compacted":
            mode = (
                "progress_gated_context_compacted"
                if stall.admitted
                else "context_compacted"
            )
        else:
            mode = "progress_gated" if stall.admitted else "executor_only"
        trigger = (
            "executor_context_length_exceeded"
            if compaction.get("status") == "compacted"
            else stall.reason
        )
        self._append_trace(
            {
                "schema_version": "fusion-moa-v091-trace-v1",
                "run_id": self.run_id,
                "task_id": task_id,
                "task_identity_source": identity.source,
                "mode": mode,
                "stall_reason": stall.reason,
                "stall_fingerprint": stall.fingerprint,
                "stall_episode_id": stall.episode_id,
                "state_epoch": stall.state_epoch,
                "stalled_action_signatures": list(
                    stall.stalled_action_signatures
                ),
                "panel_reused": panel_reused,
                "analyst": analyst,
                "context_compaction": compaction,
                "repeat_guard": repeat_guard,
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
                "X-MOA-Task-ID": task_id,
                "X-MOA-Stall-Episode": stall.episode_id,
                "X-MOA-Panel-Reused": str(panel_reused).lower(),
                "X-MOA-Repeat-Guard": repeat_guard["status"],
            },
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requests": self.request_count,
            "interventions": self.intervention_count,
            "orchestration_events": self.orchestration_event_count,
            "context_compact_events": self.context_compaction_count,
            "panel_events": self.panel_event_count,
            "soft_stall_events": self.soft_stall_event_count,
            "hard_stall_events": self.hard_stall_event_count,
            "analyst_requests": self.analyst_request_count,
            "d12_internal_retry": self.d12_internal_retry_count,
            "action_resamples": self.action_resample_count,
        }


__all__ = [
    "GenericMOAOrchestrator",
    "ModelEndpoint",
    "OrchestrationResult",
]
