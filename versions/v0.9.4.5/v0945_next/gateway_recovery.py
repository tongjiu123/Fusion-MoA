"""Fusion-MOA v0.9.2 clean recovery and rolling compaction."""

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
from .gateway_progress import (
    GenericMOAOrchestrator as V091GenericMOAOrchestrator,
)

from .context_compact import RollingCompactCheckpointStore
from .evidence_episode import (
    EpisodeState,
    StallDecision,
    action_item_signatures,
    recovery_evidence,
)
from .soft_stall import (
    PANEL_ONCE_REASON,
    SOFT_STALL_REASON,
    SOFT_STALL_TOOL_CYCLES,
    count_tool_cycles,
    evaluate_collaboration_gate_v092,
)


class GenericMOAOrchestrator(V091GenericMOAOrchestrator):
    """Keep D12 final authority while making two bounded recovery paths useful."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._hard_episode_by_task: dict[str, EpisodeState] = {}
        self._compact_checkpoints = RollingCompactCheckpointStore()
        self.recovery_generation_count = 0
        self.recovery_exhausted_count = 0
        self.compact_build_count = 0
        self.compact_reuse_count = 0
        self.compact_extension_count = 0
        for key in (
            "progress_gated_repeat_guard",
            "repeat_reason_marker",
            "progress_gated_action_resample_limit",
        ):
            self.boundaries.pop(key, None)
        self.boundaries = {
            **self.boundaries,
            "system_version": "v0.9.2",
            "clean_recovery": True,
            "clean_recovery_per_hard_episode": 1,
            "clean_recovery_terminal_marker": "LOOP_STALLED:",
            "rolling_compact_checkpoint": "task_local_process_memory",
            "compact_checkpoint_public_api_change": False,
            "task_classifier": False,
            "task_registry_entries": 0,
            "benchmark_specific_routing": False,
            "supervisor": False,
            "long_term_memory": False,
        }

    @staticmethod
    def _stall_context(stall: StallDecision) -> str:
        if not isinstance(stall.stalled_action, Mapping):
            return ""
        return (
            "[MOA stall context]\n"
            "The current trajectory contains a confirmed no-progress episode.\n"
            "stalled_action="
            + json.dumps(
                stall.stalled_action,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\nrepeated_observation="
            + json.dumps(
                stall.last_observation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\nChoose a materially different action. The same action is valid "
            "only after a relevant state change or with `REPEAT_REASON:` and "
            "the expected new information. D12 remains the only tool-call and "
            "final-answer authority."
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
    def _has_tool_calls(cls, payload: Mapping[str, Any]) -> bool:
        message = cls._response_message(payload)
        return (
            message is not None
            and isinstance(message.get("tool_calls"), list)
            and bool(message["tool_calls"])
        )

    @classmethod
    def _has_repeat_reason(cls, payload: Mapping[str, Any]) -> bool:
        message = cls._response_message(payload)
        content = message.get("content") if message else None
        return (
            isinstance(content, str)
            and re.search(
                r"(?im)^\s*REPEAT_REASON\s*:\s*\S",
                content,
            )
            is not None
        )

    @classmethod
    def _is_loop_stalled(cls, payload: Mapping[str, Any]) -> bool:
        message = cls._response_message(payload)
        content = message.get("content") if message else None
        return (
            not cls._has_tool_calls(payload)
            and isinstance(content, str)
            and re.search(
                r"(?im)^\s*LOOP_STALLED\s*:\s*\S",
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

    @staticmethod
    def _append_guidance(
        messages: Sequence[Mapping[str, Any]],
        guidance: str,
        stall_context: str,
    ) -> list[dict[str, Any]]:
        forwarded = [dict(message) for message in messages]
        combined = "\n\n".join(
            item for item in (guidance, stall_context) if item
        )
        if combined:
            forwarded.append({"role": "system", "content": combined})
        return forwarded

    @staticmethod
    def _clean_recovery_messages(
        messages: Sequence[Mapping[str, Any]],
        *,
        stall: StallDecision,
        guidance: str,
        reasoned_repeat_used: bool,
    ) -> list[dict[str, Any]]:
        first_user = next(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user"
            ),
            None,
        )
        prefix = (
            [dict(message) for message in messages[: first_user + 1]]
            if first_user is not None
            else [dict(messages[0])]
        )
        choices = [
            "1. a materially different tool action;",
            "3. `LOOP_STALLED:` with the concrete unresolved blocker and no "
            "tool call.",
        ]
        if not reasoned_repeat_used:
            choices.insert(
                1,
                "2. the same action once with `REPEAT_REASON:` and the "
                "expected new information;",
            )
        recovery = {
            "schema": "fusion-moa-clean-recovery-v1",
            "evidence": recovery_evidence(messages, stall),
            "analyst_guidance": guidance,
        }
        prefix.append(
            {
                "role": "system",
                "content": (
                    "[MOA clean recovery]\n"
                    "The repeated loop history has been removed from this one "
                    "recovery generation. Use the original task and the bounded "
                    "evidence below. Do not reconstruct or continue the loop.\n"
                    + json.dumps(
                        recovery,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\nProduce exactly one of:\n"
                    + "\n".join(choices)
                ),
            }
        )
        return prefix

    async def _panel_guidance(
        self,
        messages: list[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        stall: StallDecision,
    ) -> tuple[str, dict[str, Any], bool, EpisodeState | None]:
        task_id = identity.internal_task_id
        is_panel_once = stall.reason == PANEL_ONCE_REASON
        is_soft = stall.reason == SOFT_STALL_REASON
        if is_panel_once:
            self._hard_episode_by_task.pop(task_id, None)
            self.panel_once_event_count += 1
            episode = None
        elif is_soft:
            self._hard_episode_by_task.pop(task_id, None)
            self._last_soft_stall_round[task_id] = count_tool_cycles(messages)
            self.soft_stall_event_count += 1
            episode = None
        else:
            self.hard_stall_event_count += 1
            episode_id = stall.episode_id or stall.fingerprint
            existing = self._hard_episode_by_task.get(task_id)
            if existing is not None and existing.episode_id == episode_id:
                return (
                    existing.guidance,
                    {
                        "attempted": False,
                        "status": "episode_guidance_reused",
                        "request_count": 0,
                        "selected_hypothesis_ids": list(
                            existing.selected_hypothesis_ids
                        ),
                    },
                    True,
                    existing,
                )
            episode = EpisodeState(episode_id=episode_id)
            self._hard_episode_by_task[task_id] = episode

        self.panel_event_count += 1
        guidance, analyst = await self._intervene(
            messages,
            identity=identity,
            stall=stall,
        )
        self.analyst_request_count += int(
            analyst.get("request_count", 0) or 0
        )
        if episode is not None:
            episode.guidance = guidance
            episode.selected_hypothesis_ids = tuple(
                analyst.get("selected_hypothesis_ids") or ()
            )
        return guidance, analyst, False, episode

    async def complete(
        self,
        body: Mapping[str, Any],
        *,
        identity: GenericTaskIdentity,
        public_model: str,
        managed: bool,
    ) -> OrchestrationResult:
        started = time.monotonic()
        raw_messages = body.get("messages")
        assert isinstance(raw_messages, list)
        messages = [dict(message) for message in raw_messages]
        metadata = (
            body.get("metadata")
            if isinstance(body.get("metadata"), Mapping)
            else {}
        )
        task_id = identity.internal_task_id
        stall = (
            evaluate_collaboration_gate_v092(
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
        episode: EpisodeState | None = None
        guidance = ""
        if stall.admitted:
            self.intervention_count += 1
            self.orchestration_event_count += 1
            (
                guidance,
                analyst,
                panel_reused,
                episode,
            ) = await self._panel_guidance(
                messages,
                identity=identity,
                stall=stall,
            )
        elif managed:
            self._hard_episode_by_task.pop(task_id, None)

        checkpoint_messages, checkpoint = self._compact_checkpoints.reuse(
            task_id,
            messages,
        )
        if checkpoint_messages is not None:
            executor_base = checkpoint_messages
            self.compact_reuse_count += 1
        else:
            executor_base = messages
        stall_context = self._stall_context(stall) if stall.admitted else ""
        executor_messages = self._append_guidance(
            executor_base,
            guidance,
            stall_context,
        )
        status_code, payload = await self._call_executor(
            body,
            executor_messages,
            public_model=public_model,
        )

        compaction: dict[str, Any] = {
            "status": checkpoint.get("status", "not_needed"),
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
            compacted, detail = await self._compact_context(executor_base)
            compaction = dict(detail)
            self.analyst_request_count += int(
                compaction.get("analyst_calls", 0) or 0
            )
            if compacted is not None:
                if checkpoint_messages is None:
                    saved = self._compact_checkpoints.save_build(
                        task_id,
                        messages,
                        compacted,
                        compaction,
                    )
                    compaction["status"] = "checkpoint_built"
                    compaction["checkpoint_saved"] = saved is not None
                    self.compact_build_count += 1
                else:
                    saved = self._compact_checkpoints.save_extension(
                        task_id,
                        messages,
                        executor_base,
                        compacted,
                        compaction,
                    )
                    compaction["status"] = "checkpoint_extended"
                    compaction["checkpoint_saved"] = saved is not None
                    self.compact_extension_count += 1
                self.context_compaction_count += 1
                self.intervention_count += 1
                self.orchestration_event_count += 1
                self.d12_internal_retry_count += 1
                executor_base = compacted
                executor_messages = self._append_guidance(
                    executor_base,
                    guidance,
                    stall_context,
                )
                status_code, payload = await self._call_executor(
                    body,
                    executor_messages,
                    public_model=public_model,
                )

        recovery = {
            "attempted": False,
            "status": "not_needed",
        }
        is_hard_episode = (
            managed
            and episode is not None
            and stall.reason != SOFT_STALL_REASON
        )
        if (
            is_hard_episode
            and 200 <= status_code < 300
            and self._repeats_stalled_action(
                payload,
                stall.stalled_action_signatures,
            )
        ):
            assert episode is not None
            if (
                self._has_repeat_reason(payload)
                and not episode.reasoned_repeat_used
            ):
                episode.reasoned_repeat_used = True
                recovery["status"] = "allowed_reasoned_repeat"
            elif episode.recovery_used:
                recovery = {
                    "attempted": False,
                    "status": "episode_recovery_exhausted_fallback_original",
                }
                self.recovery_exhausted_count += 1
            else:
                original_status_code = status_code
                original_payload = dict(payload)
                episode.recovery_used = True
                self.recovery_generation_count += 1
                self.orchestration_event_count += 1
                self.d12_internal_retry_count += 1
                recovery = {
                    "attempted": True,
                    "status": "generation_requested",
                }
                recovery_messages = self._clean_recovery_messages(
                    messages,
                    stall=stall,
                    guidance=guidance,
                    reasoned_repeat_used=episode.reasoned_repeat_used,
                )
                status_code, payload = await self._call_executor(
                    body,
                    recovery_messages,
                    public_model=public_model,
                )
                if not 200 <= status_code < 300:
                    recovery["status"] = "generation_error_fallback_original"
                    self.recovery_exhausted_count += 1
                    status_code = original_status_code
                    payload = original_payload
                elif self._is_loop_stalled(payload):
                    recovery["status"] = "loop_stalled"
                elif self._repeats_stalled_action(
                    payload,
                    stall.stalled_action_signatures,
                ):
                    if (
                        self._has_repeat_reason(payload)
                        and not episode.reasoned_repeat_used
                    ):
                        episode.reasoned_repeat_used = True
                        recovery["status"] = "allowed_reasoned_repeat"
                    else:
                        recovery["status"] = (
                            "repeat_guard_exhausted_fallback_original"
                        )
                        self.recovery_exhausted_count += 1
                        status_code = original_status_code
                        payload = original_payload
                elif self._has_tool_calls(payload):
                    recovery["status"] = "changed_action"
                else:
                    recovery["status"] = (
                        "invalid_recovery_output_fallback_original"
                    )
                    self.recovery_exhausted_count += 1
                    status_code = original_status_code
                    payload = original_payload

        terminal = (
            200 <= status_code < 300
            and not self._has_tool_calls(payload)
        )
        if terminal:
            self._compact_checkpoints.discard(task_id)
            self._hard_episode_by_task.pop(task_id, None)

        self.request_count += 1
        if recovery["attempted"]:
            mode = "clean_recovery"
        elif compaction.get("status") == "checkpoint_extended":
            mode = "compact_checkpoint_extended"
        elif compaction.get("status") == "checkpoint_built":
            mode = "compact_checkpoint_built"
        elif checkpoint_messages is not None:
            mode = "compact_checkpoint_reused"
        else:
            mode = "progress_gated" if stall.admitted else "executor_only"
        trigger = (
            "clean_recovery"
            if recovery["attempted"]
            else (
                "executor_context_length_exceeded"
                if compaction.get("status")
                in {"checkpoint_built", "checkpoint_extended"}
                else stall.reason
            )
        )
        self._append_trace(
            {
                "schema_version": "fusion-moa-v092-trace-v1",
                "run_id": self.run_id,
                "task_id": task_id,
                "task_identity_source": identity.source,
                "mode": mode,
                "stall_reason": stall.reason,
                "stall_episode_id": stall.episode_id,
                "state_epoch": stall.state_epoch,
                "panel_reused": panel_reused,
                "analyst": analyst,
                "context_compaction": compaction,
                "recovery": recovery,
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
                "X-MOA-Recovery": recovery["status"],
                "X-MOA-Compact": str(
                    compaction.get("status") or "not_needed"
                ),
            },
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requests": self.request_count,
            "interventions": self.intervention_count,
            "orchestration_events": self.orchestration_event_count,
            "panel_events": self.panel_event_count,
            "recovery_generations": self.recovery_generation_count,
            "recovery_exhausted": self.recovery_exhausted_count,
            "compact_builds": self.compact_build_count,
            "compact_reuses": self.compact_reuse_count,
            "compact_extensions": self.compact_extension_count,
            "soft_stall_events": self.soft_stall_event_count,
            "hard_stall_events": self.hard_stall_event_count,
            "analyst_requests": self.analyst_request_count,
            "d12_internal_retry": self.d12_internal_retry_count,
        }


__all__ = [
    "GenericMOAOrchestrator",
    "ModelEndpoint",
    "OrchestrationResult",
]
