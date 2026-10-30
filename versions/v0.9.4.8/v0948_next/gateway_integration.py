"""Fusion-MOA v0.9.4.8: task-initial solving and bounded reactive collaboration."""

from __future__ import annotations

import json
from copy import deepcopy
import time
from typing import Any, Mapping, Sequence
import uuid

from .gateway_base import (
    ModelEndpoint,
    OrchestrationResult,
    StreamSink,
)
from .task_identity import GenericTaskIdentity
from .gateway_recovery import (
    GenericMOAOrchestrator as V092GenericMOAOrchestrator,
)

from .evidence_snapshot import (
    RESULT_CYCLE_REASON,
    EpisodeState,
    StallDecision,
    build_bounded_evidence_snapshot,
    evaluate_collaboration_gate_v093 as evaluate_structural_gate,
    recovery_evidence,
)
from .soft_stall import (
    PANEL_ONCE_REASON,
    SOFT_STALL_REASON,
    evaluate_collaboration_gate_v0948,
)
from .panel_public import (
    PANEL_BRIEF_FOOTER,
    PANEL_BRIEF_HEADER,
    build_public_panel_event,
)
from .collective_solver import (
    SAMPLING_FIELDS, _candidate_content, collect_candidates, guidance_text, synthesis_messages,
)
from .proposal_review import (
    MEMO_INSTRUCTION, REVIEW_INSTRUCTION, REVIEW_ROLES, review_packet, review_guidance,
)
from .collaboration_deadline import before_deadline
from .request_controls import MOA_REQUEST_MODES


EXACT_ACTION_CYCLE_REASON = "repeated_action_observation"


class GenericMOAOrchestrator(V092GenericMOAOrchestrator):
    """Preserve the parent orchestration and carry faithful shared evidence."""

    def __init__(
        self,
        *,
        output_recovery_max_tokens: int = 512,
        panel_integrator_actor: str | None = None,
        default_collaboration_mode: str = "auto",
        solver_profiles: Mapping[str, Any] | None = None,
        panel_strategy: str = "legacy",
        proposal_review_profiles: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if default_collaboration_mode not in MOA_REQUEST_MODES:
            raise ValueError("unsupported default collaboration mode")
        self.default_collaboration_mode = default_collaboration_mode
        self.solver_profiles = dict(solver_profiles or {})
        if panel_strategy not in {"legacy", "proposal_review"}:
            raise ValueError("unsupported panel strategy")
        self.panel_strategy = panel_strategy
        self.proposal_review_profiles = dict(proposal_review_profiles or {})
        self.d12_proposal_request_count = 0
        self.always_event_count = 0
        self.initial_panel_event_count = 0
        self._initial_panel_tasks: set[str] = set()
        self.output_recovery_max_tokens = int(output_recovery_max_tokens)
        if self.output_recovery_max_tokens <= 0:
            raise ValueError("output_recovery_max_tokens must be positive")
        self.exact_action_cycle_count = 0
        self.result_cycle_count = 0
        self.output_contract_recovery_count = 0
        self.output_contract_recovery_exhausted_count = 0
        self.panel_integrator_actor = (
            panel_integrator_actor or self.analyst_actors[-1]
        )
        if self.panel_integrator_actor not in self.analysts:
            raise ValueError(
                "panel_integrator_actor must name one configured Analyst cell"
            )
        self.panel_integrator_request_count = 0
        self.panel_integrator_failure_count = 0
        self.d12_deliberation_count = 0
        self.d12_deliberation_fail_open_count = 0
        self.panel_once_event_count = 0
        self.boundaries = {
            **self.boundaries,
            "system_version": "v0.9.4.8",
            "collaboration_trigger": (
                "initial_panel_or_always_or_explicit_panel_once_or_caller_hard_stall_or_exact_action_"
                "cycle_or_result_cycle_or_periodic_long_tool_loop_soft_stall"
            ),
            "request_collaboration_modes": ["auto", "panel_once", "always", "initial_panel"],
            "default_collaboration_mode": self.default_collaboration_mode,
            "always_collaboration": "one_parallel_solver_round_per_external_request",
            "always_synthesis": "D12_direct_without_integrator",
            "initial_panel": "once_per_existing_task_identity_then_existing_reactive_gates",
            "initial_panel_state": "process_local_task_id_attempt_marker",
            "collaboration_event_budget_seconds": self.analyst_deadline_seconds,
            "collaboration_deadline": "one_absolute_monotonic_deadline_including_integrator",
            "partial_panel": "completed_candidates_are_normal_success",
            "solver_profiles": self.solver_profiles,
            "panel_strategy": self.panel_strategy,
            "proposal_review_profiles": self.proposal_review_profiles,
            "single_turn_explicit_panel": True,
            "single_turn_panel_limit_per_request": 1,
            "single_turn_panel_benchmark_specific": False,
            "exact_action_cycle": True,
            "result_cycle_detection": (
                "tail_period_over_exact_action_tool_name_and_exact_observation"
            ),
            "output_contract_recovery": (
                "tools_and_finish_reason_length_without_valid_tool_calls"
            ),
            "output_contract_recovery_limit": 1,
            "output_contract_recovery_max_tokens": (
                self.output_recovery_max_tokens
            ),
            "context_compaction_retry_limit": 2,
            "rolling_compact_checkpoint": "task_local_process_memory",
            "panel_integrator": "one_read_only_sequential_synthesis",
            "panel_integrator_actor": self.panel_integrator_actor,
            "d12_deliberation": (
                "accept_brief_or_evidence_reject_or_report_stalled"
            ),
            "d12_deliberation_per_hard_episode": 1,
            "d12_deliberation_output_authority": "D12_only",
            "d12_deliberation_internal_marker": (
                "fail_open_to_original_d12_when_tools_have_no_tool_calls"
            ),
            "optional_moa_failure_policy": "fail_open_to_original_d12",
            "openai_compatible_streaming": True,
            "streaming_no_tools": "upstream_d12_sse",
            "streaming_with_tools": "buffered_final_d12_response_as_sse",
            "streaming_panel_progress": (
                "optional_status_and_public_summary_events"
            ),
            "streaming_panel_summary": (
                "opt_in_integrated_brief_only_no_hidden_reasoning"
            ),
            "public_thinking_normalization": (
                "explicit_think_or_prompt_open_implicit_close"
            ),
            "public_content_contract": "final_answer_without_think_markup",
            "public_system_identity": "NovaStack_Fusion-MOA_v0.9.4.8",
            "implicit_thinking_transport": "discarded_before_public_output",
            "d12_default_reasoning_effort": "medium_unless_caller_overrides",
            "public_reasoning_transport": (
                "explicit_think_blocks_optional_reasoning_content"
            ),
            "thinking_normalization_scope": "public_d12_response_only",
            "task_classifier": False,
            "task_registry_entries": 0,
            "benchmark_specific_routing": False,
            "completion_rule_in_gateway": False,
            "supervisor": False,
            "long_term_memory": False,
        }

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
                "2. one intentional retry with `REPEAT_REASON:` and the "
                "expected new information;",
            )
        evidence = {
            "schema": "fusion-moa-clean-recovery-v2",
            "evidence": recovery_evidence(messages, stall),
            "analyst_guidance": guidance,
        }
        prefix.append(
            {
                "role": "system",
                "content": (
                    "[MOA clean recovery]\n"
                    "The repeated tail has produced no new observable evidence. "
                    "This recovery view contains the original task and bounded "
                    "cycle evidence instead of the repeated history. Do not infer "
                    "that different tool arguments are semantically identical; "
                    "only the exact observable result cycle is established.\n"
                    + json.dumps(
                        evidence,
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

    def _panel_integrator_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        stall: StallDecision,
        analyst_packets: Sequence[Mapping[str, Any]],
        collaboration_evidence: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if stall.reason == PANEL_ONCE_REASON:
            integration_goal = (
                "The caller explicitly requested one single-turn Panel "
                "consultation. Produce a concise, evidence-bound brief that "
                "helps D12 answer the original request directly."
            )
        elif stall.reason == SOFT_STALL_REASON:
            integration_goal = (
                "This is a periodic collaboration checkpoint, not proof of "
                "failure or a requirement to recover or change direction."
            )
        else:
            integration_goal = (
                "Produce a concise recovery brief that helps D12 choose a "
                "supported next direction."
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are the read-only Fusion-MOA Panel Integrator. "
                    "Synthesize the independent Analyst opinions into one short "
                    "brief for D12. Do not call tools, write code, decide task "
                    "completion, or produce the user's final answer. "
                    "Use the same frozen evidence snapshot supplied to the Analysts; "
                    "do not become a fourth Analyst or invent additional hypotheses. "
                    "Distinguish observed facts, proposed hypotheses, disagreements, "
                    "and abstentions. A hypothesis or citation is not proof. "
                    "Preserve useful claims, reasons, supporting and "
                    "disconfirming references, proposed probes, expected observations, "
                    "and falsification conditions. A single supported hypothesis may "
                    "be useful; do not present it as majority agreement. Abstention "
                    "means no hypothesis from that Analyst, not task failure or "
                    "support for another packet. If all abstain, report that limitation "
                    "without inventing an Analyst-supported direction. Respect current "
                    "user updates while retaining earlier constraints that were not "
                    "withdrawn or superseded. Treat truncated evidence as incomplete. "
                    + integration_goal
                    + " Summarize the supported options and their limits, including "
                    "what evidence would reject a hypothesis. If evidence conflicts "
                    "or is insufficient, say so; D12 alone chooses the next action."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "schema": "fusion-moa-panel-integration-v1",
                        "stall_reason": stall.reason,
                        "stall_evidence": collaboration_evidence,
                        "independent_analyst_packets": [
                            dict(packet) for packet in analyst_packets
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]

    async def _integrate_panel_guidance(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        stall: StallDecision,
        analyst_packets: Sequence[Mapping[str, Any]],
        collaboration_evidence: Mapping[str, Any] | None = None,
        deadline: float | None = None,
    ) -> str:
        if deadline is None:
            deadline = time.monotonic() + self.analyst_deadline_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("collaboration deadline reached before integration")
        endpoint = self.analysts[self.panel_integrator_actor]
        self.panel_integrator_request_count += 1
        self.analyst_request_count += 1
        response = await before_deadline(self.client.post(
            self._chat_url(endpoint.base_url),
            json={
                "model": endpoint.model,
                "messages": self._panel_integrator_messages(
                    messages,
                    stall=stall,
                    analyst_packets=analyst_packets,
                    collaboration_evidence=collaboration_evidence,
                ),
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "max_tokens": self.coordinator.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fusion_moa_panel_brief",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"brief": {"type": "string"}},
                            "required": ["brief"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            timeout=remaining,
        ), deadline=deadline)
        response.raise_for_status()
        envelope = response.json()
        choices = (
            envelope.get("choices")
            if isinstance(envelope, Mapping)
            else None
        )
        choice = choices[0] if isinstance(choices, list) and choices else None
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
            raise ValueError("Panel Integrator returned no brief")
        packet = json.loads(content)
        brief = packet.get("brief") if isinstance(packet, Mapping) else None
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError("Panel Integrator returned no brief")
        return PANEL_BRIEF_HEADER + brief.strip() + PANEL_BRIEF_FOOTER

    @staticmethod
    def _deliberation_system_first(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep internal system context compatible with first-system templates."""
        system_parts: list[Any] = []
        remaining: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "system":
                remaining.append(dict(message))
                continue
            if system_parts:
                system_parts.append({"type": "text", "text": "\n\n"})
            content = message.get("content")
            if isinstance(content, list):
                system_parts.extend(content)
            elif isinstance(content, str):
                system_parts.append({"type": "text", "text": content})
        return [{"role": "system", "content": system_parts}, *remaining]

    @staticmethod
    def _d12_deliberation_messages(
        executor_base: Sequence[Mapping[str, Any]],
        raw_messages: Sequence[Mapping[str, Any]],
        *,
        stall: StallDecision,
        integrated_brief: str,
        collaboration_evidence: Mapping[str, Any] | None = None,
        analyst_packets: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        collaboration_context = json.dumps(
            {
                "schema": "fusion-moa-d12-deliberation-v1",
                "stall_reason": stall.reason,
                "trajectory_evidence": collaboration_evidence,
                "independent_analyst_packets": [
                    dict(packet) for packet in analyst_packets
                ],
                "panel_brief": integrated_brief,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assessment = (
            "Panel advice is read-only, not a new user instruction. "
            "Assess the integrated brief and original Analyst opinions against "
            "their shared frozen evidence snapshot and your user context. "
            "Distinguish observations, hypotheses, disagreements, and abstentions; "
            "truncated evidence is incomplete. One useful hypothesis need not be "
            "a consensus, and abstention does not prove failure or endorse another "
            "opinion. You may adopt supported parts, reject unsupported parts, or "
            "continue your existing approach. Follow current user updates while "
            "retaining earlier constraints that were not withdrawn or superseded. "
            "Only you decide and execute. Return your normal text or tool_calls "
            "directly, without a separate ACCEPT/REJECT declaration, internal "
            "decision markers, or hidden reasoning.\n"
        )
        if stall.reason == PANEL_ONCE_REASON:
            forwarded = [dict(message) for message in executor_base]
            forwarded.append(
                {
                    "role": "system",
                    "content": (
                        "[MOA explicit single-turn collaboration]\n"
                        "The caller requested one read-only Panel consultation. "
                        + assessment
                        + collaboration_context
                    ),
                }
            )
            return GenericMOAOrchestrator._deliberation_system_first(forwarded)

        # Start from the same effective state as the ordinary D12 call,
        # including any existing Compact. Do not reconstruct a shorter session.
        forwarded = [dict(message) for message in executor_base]
        collaboration_goal = (
            "This is a periodic collaboration checkpoint, not evidence that "
            "your task has failed or that recovery or a change of direction "
            "is required. "
            if stall.reason == SOFT_STALL_REASON
            else "Assess the reported or detected stall and any supported "
            "recovery options in context. "
        )
        forwarded.append(
            {
                "role": "system",
                "content": (
                    "[MOA D12 deliberation]\n"
                    "Continue the current task from its existing execution state; "
                    "do not restart merely because Panel advice arrived. Preserve "
                    "completed work that remains valid. If the Panel has no new "
                    "supported hypothesis, continue the outstanding user request. "
                    + collaboration_goal
                    + assessment
                    + collaboration_context
                ),
            }
        )
        return GenericMOAOrchestrator._deliberation_system_first(forwarded)

    @staticmethod
    def _finish_reason(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return ""
        value = choice.get("finish_reason")
        return value if isinstance(value, str) else ""

    @classmethod
    def _is_internal_deliberation_only(
        cls,
        body: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        tools = body.get("tools")
        if not isinstance(tools, list) or not tools:
            return False
        if cls._has_tool_calls(payload):
            return False
        message = cls._response_message(payload)
        content = message.get("content") if message else None
        if not isinstance(content, str):
            return False
        normalized = content.lstrip().upper()
        return normalized.startswith(
            ("BRIEF_DECISION: ACCEPT", "BRIEF_DECISION: REJECT")
        )

    @staticmethod
    def _declared_tool_names(body: Mapping[str, Any]) -> set[str]:
        tools = body.get("tools")
        if not isinstance(tools, list):
            return set()
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            function = function if isinstance(function, Mapping) else {}
            name = function.get("name")
            if isinstance(name, str) and name:
                names.add(name)
        return names

    @classmethod
    def _has_valid_tool_calls(
        cls,
        payload: Mapping[str, Any],
        body: Mapping[str, Any],
    ) -> bool:
        message = cls._response_message(payload)
        calls = message.get("tool_calls") if message else None
        if not isinstance(calls, list) or not calls:
            return False
        declared_names = cls._declared_tool_names(body)
        if not declared_names:
            return False
        for call in calls:
            if not isinstance(call, Mapping):
                return False
            if call.get("type") not in (None, "function"):
                return False
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not call_id:
                return False
            if not isinstance(function, Mapping):
                return False
            name = function.get("name")
            arguments = function.get("arguments")
            if name not in declared_names or not isinstance(arguments, str):
                return False
            try:
                decoded = json.loads(arguments)
            except (TypeError, ValueError):
                return False
            if not isinstance(decoded, Mapping):
                return False
        return True

    @classmethod
    def _needs_output_contract_recovery(
        cls,
        body: Mapping[str, Any],
        status_code: int,
        payload: Mapping[str, Any],
    ) -> bool:
        return (
            200 <= status_code < 300
            and bool(cls._declared_tool_names(body))
            and cls._finish_reason(payload) == "length"
            and not cls._has_valid_tool_calls(payload, body)
        )

    @classmethod
    def _valid_output_recovery(
        cls,
        body: Mapping[str, Any],
        status_code: int,
        payload: Mapping[str, Any],
    ) -> bool:
        if not 200 <= status_code < 300:
            return False
        if cls._has_valid_tool_calls(payload, body):
            return True
        message = cls._response_message(payload)
        content = message.get("content") if message else None
        calls = message.get("tool_calls") if message else None
        return (
            cls._finish_reason(payload) != "length"
            and not calls
            and isinstance(content, str)
            and bool(content.strip())
        )

    def _output_contract_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        body: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        tool_names = sorted(self._declared_tool_names(body))
        return self._deliberation_system_first([
            *[dict(message) for message in messages],
            {
                "role": "system",
                "content": (
                    "[MOA output-contract recovery]\n"
                    "The previous generation reached its output limit before "
                    "forming a valid response. Do not repeat the long reasoning. "
                    "Return exactly one concise result: (1) one valid tool call "
                    "using a declared tool and JSON-object arguments; (2) a brief "
                    "final answer with no tool call if no tool is needed; or "
                    "(3) `OUTPUT_CONTRACT_STALLED:` followed by the concrete "
                    "reason no valid action can be produced. Declared tools: "
                    + json.dumps(tool_names, ensure_ascii=False)
                ),
            },
        ])

    async def _recover_output_contract(
        self,
        body: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        *,
        public_model: str,
        original_status_code: int,
        original_payload: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        self.output_contract_recovery_count += 1
        self.orchestration_event_count += 1
        self.d12_internal_retry_count += 1
        try:
            recovery_body = dict(body)
            recovery_body["max_tokens"] = min(
                self.output_recovery_max_tokens,
                self.executor_max_tokens,
            )
            status_code, payload = await self._call_executor(
                recovery_body,
                self._output_contract_messages(messages, body),
                public_model=public_model,
            )
            if self._valid_output_recovery(body, status_code, payload):
                return status_code, payload, {
                    "attempted": True,
                    "status": "recovered",
                }
        except Exception:
            pass
        self.output_contract_recovery_exhausted_count += 1
        return original_status_code, dict(original_payload), {
            "attempted": True,
            "status": "fallback_original",
        }

    def _build_evidence_snapshot(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        selection: Any,
        run_id: str,
        stall: StallDecision,
    ) -> Any:
        return build_bounded_evidence_snapshot(
            messages,
            identity=identity,
            selection=selection,
            run_id=run_id,
            stall=stall,
        )

    async def _panel_guidance(
        self,
        messages: list[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        stall: StallDecision,
        deadline: float | None = None,
        consultation=None,
    ) -> tuple[str, dict[str, Any], bool, EpisodeState | None]:
        if deadline is None:
            deadline = time.monotonic() + self.analyst_deadline_seconds
        try:
            (
                guidance,
                analyst,
                panel_reused,
                episode,
            ) = await super()._panel_guidance(
                messages,
                identity=identity,
                stall=stall,
                deadline=deadline,
                consultation=consultation,
            )
        except Exception as exc:
            return (
                "",
                {
                    "attempted": True,
                    "status": "degraded_to_executor",
                    "error_type": type(exc).__name__,
                    "request_count": 0,
                    "selected_hypothesis_ids": [],
                },
                False,
                self._hard_episode_by_task.get(
                    identity.internal_task_id
                ),
            )
        analyst = dict(analyst)
        if consultation is not None:
            # Reuse the same admission/episode lifecycle, without an Integrator.
            return guidance, analyst, panel_reused, episode
        # Kept private until complete() consumes both before public events/trace.
        analyst_packets = analyst.get("_integrator_packets", [])
        collaboration_evidence = analyst.get("_collaboration_evidence")
        if panel_reused:
            analyst["integrator"] = {
                "attempted": False,
                "status": "brief_reused",
                "actor": self.panel_integrator_actor,
            }
            return guidance, analyst, True, episode
        if not analyst_packets:
            if episode is not None:
                episode.guidance = ""
            analyst["integrator"] = {
                "attempted": False,
                "status": "no_valid_analyst_packets",
                "actor": self.panel_integrator_actor,
            }
            return "", analyst, False, episode

        def completed_candidates(attempted: bool, error_type: str):
            # Keep completed opinions; lack of an Integrator is not failure of the Panel.
            direct_guidance = (
                "[MOA completed analyst opinions; no integrated brief]\n"
                "The following read-only opinions completed within the collaboration budget. "
                "The Integrator was unavailable; this does not invalidate these opinions "
                "or require D12 to stop. Use the shared evidence and current user request.\n"
                + json.dumps(analyst_packets, ensure_ascii=False, separators=(",", ":"))
            )
            if episode is not None:
                episode.guidance = direct_guidance
            analyst["status"] = "completed_candidates_without_integrator"
            analyst["integrator"] = {
                "attempted": attempted, "status": "unavailable_use_completed_candidates",
                "actor": self.panel_integrator_actor, "error_type": error_type,
            }
            return direct_guidance, analyst, False, episode

        if time.monotonic() >= deadline:
            return completed_candidates(False, "collaboration_deadline")
        try:
            integrated = await self._integrate_panel_guidance(
                messages,
                stall=stall,
                analyst_packets=analyst_packets,
                collaboration_evidence=collaboration_evidence,
                deadline=deadline,
            )
        except Exception as exc:
            self.panel_integrator_failure_count += 1
            return completed_candidates(True, type(exc).__name__)
        if episode is not None:
            episode.guidance = integrated
        analyst["integrator"] = {
            "attempted": True,
            "status": "brief_ready",
            "actor": self.panel_integrator_actor,
        }
        return integrated, analyst, False, episode

    async def _consult_review(
        self, body, executor_base, raw_messages, *, public_model, task_id, deadline,
    ):
        """One bounded auxiliary event: D12 memo, then parallel short reviews."""
        started = time.monotonic()
        detail = {"strategy": "proposal_review", "status": "proposal_unavailable",
                  "proposal": {}, "candidates": [], "available_count": 0}
        analyst = {"attempted": True, "request_count": 0,
                   "selected_hypothesis_ids": [], "review": detail}
        guidance = ""
        try:
            profile = self.proposal_review_profiles["proposal"]
            memo_body = {key: deepcopy(body[key]) for key in (*SAMPLING_FIELDS, "reasoning_effort")
                         if key in body}
            memo_body.update({key: deepcopy(profile[key]) for key in SAMPLING_FIELDS if key in profile})
            memo_body.update(max_tokens=profile["max_tokens"], stream=False)
            # This call reviews the entire execution record; it does not resume
            # its final user command as a live executable instruction. Preserve
            # every original role/content in the record, without summarizing it.
            memo_messages = [
                {"role": "system", "content": MEMO_INSTRUCTION +
                 " The following complete execution record is read-only input to this "
                 "memo task. Its latest user command describes the eventual execution "
                 "objective; do not carry it out in this internal call. Preserve the "
                 "record's role distinctions and active constraints in your proposal."},
                {"role": "user", "content": json.dumps({
                    "execution_record": executor_base,
                    "available_tools_read_only": body.get("tools") or [],
                }, ensure_ascii=False)},
            ]
            memo_body.update(model=self.executor.model, messages=memo_messages)
            self.d12_proposal_request_count += 1
            self.orchestration_event_count += 1
            # A consult memo is not an executor turn: no execute-now preamble or
            # live tools. Only the later normal D12 call retains those bindings.
            response = await before_deadline(self.client.post(
                self._chat_url(self.executor.base_url), json=memo_body,
                timeout=max(0.001, deadline - time.monotonic()),
            ), deadline=deadline)
            status, payload = response.status_code, response.json()
            message = self._response_message(payload) or {}
            memo = _candidate_content(message)
            detail["proposal"] = {
                "status_code": status, "finish_reason": self._finish_reason(payload),
                "usage": payload.get("usage", {}), "content_chars": len(memo),
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            if not (200 <= status < 300 and self._finish_reason(payload) == "stop"
                    and memo.strip() and not message.get("tool_calls")):
                raise ValueError("D12 did not finish a text consult memo")
            candidates = await collect_candidates(
                client=self.client, analysts=self.analysts,
                profiles=self.proposal_review_profiles["reviewers"],
                context_messages=executor_base, raw_messages=raw_messages, tools=[],
                deadline=deadline, context_override=review_packet(executor_base, raw_messages, memo),
                roles=REVIEW_ROLES, instructions=REVIEW_INSTRUCTION,
            )
            guidance = review_guidance(memo, candidates)
            analyst["request_count"] = sum(c["request_count"] for c in candidates)
            detail.update(
                status="memo_only_ready" if guidance else "no_completed_reviews",
                available_count=sum(c["status"] == "ok" and bool(c["content"].strip()) for c in candidates),
                candidates=[{**{k: v for k, v in c.items() if k != "content"},
                             "content_chars": len(c["content"])} for c in candidates],
            )
        except Exception as exc:
            detail["error_type"] = type(exc).__name__
        detail["elapsed_seconds"] = round(time.monotonic() - started, 6)
        analyst["status"] = detail["status"]
        return guidance, analyst

    def _evaluate_clean_recovery(
        self,
        body: Mapping[str, Any],
        payload: Mapping[str, Any],
        status_code: int,
        *,
        original_status_code: int,
        original_payload: Mapping[str, Any],
        episode: EpisodeState,
        stall: StallDecision,
        recovery: dict[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        if not 200 <= status_code < 300:
            recovery["status"] = "generation_error_fallback_original"
            self.recovery_exhausted_count += 1
            status_code = original_status_code
            payload = dict(original_payload)
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
                payload = dict(original_payload)
        elif self._has_tool_calls(payload):
            recovery["status"] = "changed_action"
        else:
            recovery["status"] = (
                "invalid_recovery_output_fallback_original"
            )
            self.recovery_exhausted_count += 1
            status_code = original_status_code
            payload = dict(original_payload)
        return status_code, payload, recovery

    async def complete(
        self,
        body: Mapping[str, Any],
        *,
        identity: GenericTaskIdentity,
        public_model: str,
        managed: bool,
        stream_sink: StreamSink | None = None,
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

        moa = metadata.get("moa")
        caller_disabled = isinstance(moa, Mapping) and moa.get("enabled") is False
        requested_mode = (
            moa.get("mode", self.default_collaboration_mode)
            if isinstance(moa, Mapping) else self.default_collaboration_mode
        )
        always = managed and not caller_disabled and requested_mode == "always"
        initial = (
            managed and not caller_disabled and requested_mode == "initial_panel"
            and task_id not in self._initial_panel_tasks
        )
        if initial:
            # Claim before any await; API retries and changing hard episodes do not reset it.
            self._initial_panel_tasks.add(task_id)
        direct_solvers = always or initial
        direct_deadline = (
            time.monotonic() + self.analyst_deadline_seconds if direct_solvers else None
        )

        async def emit_status(stage: str, message: str) -> None:
            if stream_sink is not None:
                await stream_sink(
                    {
                        "kind": "status",
                        "data": {"stage": stage, "message": message},
                    }
                )

        async def emit_panel(data: Mapping[str, Any]) -> None:
            if stream_sink is not None:
                await stream_sink(
                    {"kind": "panel", "data": dict(data)}
                )

        await emit_status("request_accepted", "请求已进入 Fusion-MOA")
        stall = (
            evaluate_structural_gate(messages, metadata)
            if direct_solvers else evaluate_collaboration_gate_v0948(
                messages,
                metadata,
                last_soft_stall_round=self._last_soft_stall_round.get(
                    task_id
                ),
                request_nonce=uuid.uuid4().hex,
            )
            if managed
            else StallDecision(False, "explicit_solo_model", "")
        )
        if stall.reason == EXACT_ACTION_CYCLE_REASON:
            self.exact_action_cycle_count += 1
        elif stall.reason == RESULT_CYCLE_REASON:
            self.result_cycle_count += 1

        checkpoint_messages, checkpoint = self._compact_checkpoints.reuse(task_id, messages)
        if checkpoint_messages is not None:
            executor_base = checkpoint_messages
            self.compact_reuse_count += 1
        else:
            executor_base = messages
        reviewing = self.panel_strategy == "proposal_review"

        async def consult_current_state(_messages, *, identity, stall, deadline):
            return await self._consult_review(
                body, executor_base, messages, public_model=public_model,
                task_id=task_id, deadline=deadline,
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
        collaboration_evidence: Mapping[str, Any] | None = None
        analyst_packets: Sequence[Mapping[str, Any]] = ()
        if direct_solvers:
            # Fresh candidates on every request; only Recovery uses hard-episode state.
            if stall.admitted:
                self.hard_stall_event_count += 1
                episode = self._hard_episode_by_task.get(task_id)
                if not isinstance(episode, EpisodeState) or episode.episode_id != stall.episode_id:
                    episode = EpisodeState(stall.episode_id)
                    self._hard_episode_by_task[task_id] = episode
            else:
                self._hard_episode_by_task.pop(task_id, None)
        elif stall.admitted:
            deadline = time.monotonic() + self.analyst_deadline_seconds
            await emit_status(
                "panel_started",
                "D12提案与专家审阅" if reviewing else "正在汇集三路只读分析",
            )
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
                deadline=deadline,
                **({"consultation": consult_current_state} if reviewing else {}),
            )
            collaboration_evidence = analyst.pop("_collaboration_evidence", None)
            analyst_packets = analyst.pop("_integrator_packets", [])
            partial_ready = analyst.get("status") == "completed_candidates_without_integrator"
            if reviewing:
                await emit_status("panel_completed" if guidance else "panel_failed",
                                  "定向审阅已交给D12" if guidance else "审阅不可用，D12继续原请求")
            elif partial_ready:
                await emit_status("panel_completed", "已完成的专家意见已交给D12")
            else:
                panel_event = build_public_panel_event(
                    guidance, analyst, panel_reused=panel_reused,
                    analyst_cells=self.analyst_actors,
                )
                await emit_panel(panel_event)
                await emit_status(
                    "panel_completed" if panel_event["stage"] == "panel_summary" else "panel_failed",
                    "三路只读分析与统一 Brief 已完成" if panel_event["stage"] == "panel_summary"
                    else "Panel 未生成可用 Brief，已回退 D12",
                )
        elif managed:
            self._hard_episode_by_task.pop(task_id, None)

        candidates: list[dict[str, Any]] = []
        collective: dict[str, Any] = {"attempted": direct_solvers, "status": "not_requested"}
        if direct_solvers:
            deadline = direct_deadline
            if initial:
                self.initial_panel_event_count += 1
            else:
                self.always_event_count += 1
            self.panel_event_count += 1
            self.intervention_count += 1
            self.orchestration_event_count += 1
            await emit_status("panel_started", "D12提案与专家审阅" if reviewing else "专家正在并行求解")
            try:
                if reviewing:
                    guidance, review_result = await consult_current_state(
                        messages, identity=identity, stall=stall, deadline=deadline,
                    )
                    collective.update(review_result["review"])
                    count = review_result["request_count"]
                else:
                    candidates = await collect_candidates(
                        client=self.client, analysts=self.analysts, profiles=self.solver_profiles,
                        context_messages=executor_base, raw_messages=messages,
                        tools=body.get("tools") or [],
                        deadline=deadline,
                    )
                    guidance = guidance_text(candidates)
                    count = sum(int(item.get("request_count", 0)) for item in candidates)
                self.analyst_request_count += count
                collective.update(
                    status=collective["status"] if reviewing else "candidates_ready" if guidance else "all_unavailable",
                    trigger="initial_panel" if initial else "always",
                    budget_seconds=self.analyst_deadline_seconds,
                    available_count=collective.get("available_count", 0) if reviewing else sum(bool((item.get("content") or "").strip()) for item in candidates),
                    request_count=count,
                    candidates=collective.get("candidates", []) if reviewing else [
                        {**{key: value for key, value in item.items() if key != "content"},
                         "content_chars": len(item.get("content") or "")}
                        for item in candidates
                    ],
                )
            except Exception as exc:
                guidance = ""
                collective.update(status="failed_open", error_type=type(exc).__name__)
            analyst = {
                "attempted": True, "status": collective["status"],
                "request_count": collective.get("request_count", 0),
                "selected_hypothesis_ids": [],
            }
            await emit_status(
                "panel_completed" if guidance else "panel_failed",
                "候选已交给 D12 直接综合" if guidance else "专家候选不可用，D12继续原请求",
            )
        synthesis_enabled = (direct_solvers or reviewing) and bool(guidance)

        recovery: dict[str, Any] = {
            "attempted": False,
            "status": "not_needed",
        }
        compaction: dict[str, Any] = {
            "status": checkpoint.get("status", "not_needed"),
            "analyst_calls": 0,
        }
        should_deliberate = bool(guidance) and not direct_solvers and not reviewing
        if should_deliberate and episode is not None:
            if episode.deliberation_used:
                should_deliberate = False
            else:
                episode.deliberation_used = True
        deliberation: dict[str, Any] = {
            "attempted": should_deliberate,
            "status": (
                "not_applicable_direct_synthesis"
                if direct_solvers
                else "requested"
                if should_deliberate
                else "episode_already_deliberated"
                if guidance
                else "not_needed"
            ),
        }
        if should_deliberate:
            self.d12_deliberation_count += 1
            self.orchestration_event_count += 1

        async def call_executor_path() -> tuple[
            int,
            dict[str, Any],
            list[dict[str, Any]],
        ]:
            nonlocal synthesis_enabled, should_deliberate
            if synthesis_enabled:
                try:
                    forwarded = self._deliberation_system_first(
                        [*executor_base, {"role": "system", "content": guidance}]
                        if reviewing else synthesis_messages(executor_base, candidates)
                    )
                except Exception as exc:
                    synthesis_enabled = False
                    collective.update(status="context_failed_open", error_type=type(exc).__name__)
                    forwarded = [dict(message) for message in executor_base]
            elif should_deliberate:
                forwarded = self._d12_deliberation_messages(
                    executor_base,
                    messages,
                    stall=stall,
                    integrated_brief=guidance,
                    collaboration_evidence=collaboration_evidence,
                    analyst_packets=analyst_packets,
                )
            else:
                forwarded = [dict(message) for message in executor_base]
            await emit_status(
                "d12_streaming"
                if body.get("stream") is True
                else "d12_generating",
                "D12 正在生成唯一最终响应",
            )
            current_status, current_payload = await self._call_executor(
                body,
                forwarded,
                public_model=public_model,
                stream_sink=stream_sink,
            )
            if (
                synthesis_enabled
                and current_status in {400, 422}
                and (reviewing or not self._is_context_length_error(current_status, current_payload))
            ):
                synthesis_enabled = False
                collective["status"] = "solver_context_failed_open"
                self.d12_internal_retry_count += 1
                forwarded = [dict(message) for message in executor_base]
                current_status, current_payload = await self._call_executor(
                    body, forwarded, public_model=public_model, stream_sink=stream_sink,
                )
            if should_deliberate:
                if self._is_context_length_error(
                    current_status,
                    current_payload,
                ):
                    # Optional advice must not displace the normal execution state.
                    # Only a context error from this unchanged base may invoke Compact.
                    should_deliberate = False
                    self.d12_deliberation_fail_open_count += 1
                    self.d12_internal_retry_count += 1
                    deliberation["status"] = "advice_context_error_fallback_to_base"
                    forwarded = [dict(message) for message in executor_base]
                    current_status, current_payload = await self._call_executor(
                        body,
                        forwarded,
                        public_model=public_model,
                        stream_sink=stream_sink,
                    )
                elif 200 <= current_status < 300:
                    if self._is_internal_deliberation_only(
                        body,
                        current_payload,
                    ):
                        self.d12_deliberation_fail_open_count += 1
                        deliberation["status"] = (
                            "internal_marker_fail_open_to_original_d12"
                        )
                        forwarded = [
                            dict(message) for message in executor_base
                        ]
                        current_status, current_payload = (
                            await self._call_executor(
                                body,
                                forwarded,
                                public_model=public_model,
                                stream_sink=stream_sink,
                            )
                        )
                    else:
                        deliberation["status"] = "d12_returned"
                else:
                    self.d12_deliberation_fail_open_count += 1
                    deliberation["status"] = "failed_open_to_original_d12"
                    forwarded = [
                        dict(message) for message in executor_base
                    ]
                    current_status, current_payload = (
                        await self._call_executor(
                            body,
                            forwarded,
                            public_model=public_model,
                            stream_sink=stream_sink,
                        )
                    )
            return current_status, current_payload, forwarded

        (
            status_code,
            payload,
            last_executor_messages,
        ) = await call_executor_path()

        compact_statuses: list[str] = []
        for compact_pass in range(2):
            if (
                not managed
                or caller_disabled
                or not self._is_context_length_error(
                    status_code,
                    payload,
                )
            ):
                break
            compact_source = executor_base
            compacted, detail = await self._compact_context(compact_source)
            compaction = dict(detail)
            self.analyst_request_count += int(
                compaction.get("analyst_calls", 0) or 0
            )
            if compacted is None:
                break
            checkpoint_before = self._compact_checkpoints.get(task_id)
            if checkpoint_before is None:
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
                    compact_source,
                    compacted,
                    compaction,
                )
                compaction["status"] = "checkpoint_extended"
                compaction["checkpoint_saved"] = saved is not None
                self.compact_extension_count += 1
            compaction["context_error_pass"] = compact_pass + 1
            compact_statuses.append(str(compaction["status"]))
            compaction["context_error_passes"] = list(compact_statuses)
            self.context_compaction_count += 1
            self.intervention_count += 1
            self.orchestration_event_count += 1
            self.d12_internal_retry_count += 1
            executor_base = compacted
            (
                status_code,
                payload,
                last_executor_messages,
            ) = await call_executor_path()

        if synthesis_enabled and self._is_context_length_error(status_code, payload):
            # Optional candidate text must not turn a fitting base request into failure.
            synthesis_enabled = False
            collective["status"] = "solver_context_length_failed_open"
            self.d12_internal_retry_count += 1
            status_code, payload, last_executor_messages = await call_executor_path()

        is_hard_episode = managed and episode is not None
        if (
            not recovery["attempted"]
            and is_hard_episode
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
                recovery["status"] = (
                    "episode_recovery_exhausted_fallback_original"
                )
                self.recovery_exhausted_count += 1
            else:
                original_status_code = status_code
                original_payload = dict(payload)
                original_executor_messages = last_executor_messages
                episode.recovery_used = True
                self.recovery_generation_count += 1
                self.orchestration_event_count += 1
                self.d12_internal_retry_count += 1
                recovery = {
                    "attempted": True,
                    "status": "generation_requested",
                }
                last_executor_messages = self._clean_recovery_messages(
                    messages,
                    stall=stall,
                    guidance=guidance,
                    reasoned_repeat_used=episode.reasoned_repeat_used,
                )
                status_code, payload = await self._call_executor(
                    body,
                    last_executor_messages,
                    public_model=public_model,
                )
                status_code, payload, recovery = (
                    self._evaluate_clean_recovery(
                        body,
                        payload,
                        status_code,
                        original_status_code=original_status_code,
                        original_payload=original_payload,
                        episode=episode,
                        stall=stall,
                        recovery=recovery,
                    )
                )
                if recovery["status"].endswith(
                    "_fallback_original"
                ):
                    last_executor_messages = original_executor_messages

        output_contract = {
            "attempted": False,
            "status": "not_needed",
        }
        if (
            managed
            and self._needs_output_contract_recovery(
                body,
                status_code,
                payload,
            )
        ):
            (
                status_code,
                payload,
                output_contract,
            ) = await self._recover_output_contract(
                body,
                last_executor_messages,
                public_model=public_model,
                original_status_code=status_code,
                original_payload=payload,
            )

        terminal = (
            200 <= status_code < 300
            and not self._has_tool_calls(payload)
            and self._finish_reason(payload) != "length"
        )
        if terminal:
            self._compact_checkpoints.discard(task_id)
            self._hard_episode_by_task.pop(task_id, None)

        self.request_count += 1
        if output_contract["attempted"]:
            mode = "output_contract_recovery"
        elif recovery["attempted"]:
            mode = "clean_recovery"
        elif compaction.get("status") == "checkpoint_extended":
            mode = "compact_checkpoint_extended"
        elif compaction.get("status") == "checkpoint_built":
            mode = "compact_checkpoint_built"
        elif reviewing and (direct_solvers or stall.admitted):
            mode = "proposal_review_execute" if synthesis_enabled else "proposal_review_fail_open_executor"
        elif direct_solvers:
            prefix = "initial_panel" if initial else "always"
            mode = prefix + ("_direct_synthesis" if synthesis_enabled else "_no_candidates_executor")
        elif deliberation["attempted"]:
            mode = (
                "panel_once_d12_deliberation"
                if stall.reason == PANEL_ONCE_REASON
                else "panel_integrated_d12_deliberation"
            )
        elif stall.reason == PANEL_ONCE_REASON:
            mode = "panel_once_fail_open_executor"
        elif checkpoint_messages is not None:
            mode = "compact_checkpoint_reused"
        else:
            mode = "cycle_gated" if stall.admitted else "executor_only"
        if output_contract["attempted"]:
            trigger = "output_contract_length_without_valid_tool_call"
        elif recovery["attempted"]:
            trigger = "clean_recovery"
        elif compaction.get("status") in {
            "checkpoint_built",
            "checkpoint_extended",
        }:
            trigger = "executor_context_length_exceeded"
        else:
            trigger = "task_initial_panel" if initial else "request_always" if always else stall.reason

        self._append_trace(
            {
                "schema_version": "fusion-moa-v0944-trace-v1",
                "run_id": self.run_id,
                "task_id": task_id,
                "task_identity_source": identity.source,
                "mode": mode,
                "stall_reason": stall.reason,
                "stall_episode_id": stall.episode_id,
                "state_epoch": stall.state_epoch,
                "panel_reused": panel_reused,
                "analyst": analyst,
                **({"collective": collective} if direct_solvers else {}),
                "d12_deliberation": deliberation,
                "context_compaction": compaction,
                "recovery": recovery,
                "output_contract_recovery": output_contract,
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
                "X-MOA-Deliberation": deliberation["status"],
                "X-MOA-Output-Contract": output_contract["status"],
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
            "panel_once_events": self.panel_once_event_count,
            "always_events": self.always_event_count,
            "initial_panel_events": self.initial_panel_event_count,
            "d12_proposal_requests": self.d12_proposal_request_count,
            "soft_stall_events": self.soft_stall_event_count,
            "hard_stall_events": self.hard_stall_event_count,
            "exact_action_cycles": self.exact_action_cycle_count,
            "result_cycles": self.result_cycle_count,
            "recovery_generations": self.recovery_generation_count,
            "recovery_exhausted": self.recovery_exhausted_count,
            "output_contract_recoveries": (
                self.output_contract_recovery_count
            ),
            "output_contract_recovery_exhausted": (
                self.output_contract_recovery_exhausted_count
            ),
            "panel_integrator_requests": (
                self.panel_integrator_request_count
            ),
            "panel_integrator_failures": (
                self.panel_integrator_failure_count
            ),
            "d12_deliberations": self.d12_deliberation_count,
            "d12_deliberation_fail_open": (
                self.d12_deliberation_fail_open_count
            ),
            "compact_builds": self.compact_build_count,
            "compact_reuses": self.compact_reuse_count,
            "compact_extensions": self.compact_extension_count,
            "analyst_requests": self.analyst_request_count,
            "d12_internal_retry": self.d12_internal_retry_count,
        }


__all__ = [
    "GenericMOAOrchestrator",
    "ModelEndpoint",
    "OrchestrationResult",
]
