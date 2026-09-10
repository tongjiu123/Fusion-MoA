"""Periodic stalls plus explicit single-turn collaboration for v0.9.4.5."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .evidence_episode import (
    StallDecision,
    evaluate_collaboration_gate as evaluate_exact_action_gate,
)
from .evidence_snapshot import (
    evaluate_collaboration_gate_v093 as evaluate_structural_gate,
)


SOFT_STALL_TOOL_CYCLES = 16
SOFT_STALL_REASON = "soft_stall_long_tool_loop"
PANEL_ONCE_REASON = "caller_panel_once"


def count_tool_cycles(messages: Sequence[Mapping[str, Any]]) -> int:
    """Count tool rounds across consecutive assistant fragments.

    A tool-call block and its accompanying text may be separate assistant
    messages. They still form one round, including parallel tool calls.
    """

    cycles = 0
    pending_tools = False
    for message in messages:
        if not isinstance(message, Mapping):
            pending_tools = False
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                pending_tools = True
        elif role == "tool":
            if pending_tools:
                cycles += 1
                pending_tools = False
        else:
            pending_tools = False
    return cycles


def _soft_decision(
    messages: Sequence[Mapping[str, Any]],
) -> StallDecision:
    return evaluate_exact_action_gate(
        messages,
        {
            "moa": {
                "hard_stall": True,
                "reason": SOFT_STALL_REASON,
            }
        },
    )


def evaluate_collaboration_gate_v09(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    *,
    soft_stall_already_intervened: bool = False,
) -> StallDecision:
    """Compatibility form for the inherited context-compaction layer."""

    base = evaluate_structural_gate(messages, metadata)
    if base.admitted or base.reason == "caller_disabled":
        return base
    if count_tool_cycles(messages) < SOFT_STALL_TOOL_CYCLES:
        return base
    if soft_stall_already_intervened:
        return StallDecision(False, "soft_stall_already_intervened", "")
    return _soft_decision(messages)


def evaluate_collaboration_gate_v093(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    *,
    last_soft_stall_round: int | None = None,
) -> StallDecision:
    """Preserve v0.9.2 periodic Soft Stall after structural stall checks."""

    base = evaluate_structural_gate(messages, metadata)
    if base.admitted or base.reason == "caller_disabled":
        return base

    tool_round = count_tool_cycles(messages)
    if tool_round < SOFT_STALL_TOOL_CYCLES:
        return base
    if (
        last_soft_stall_round is not None
        and tool_round - last_soft_stall_round < SOFT_STALL_TOOL_CYCLES
    ):
        return StallDecision(False, "soft_stall_interval_not_reached", "")
    return _soft_decision(messages)


def evaluate_collaboration_gate_v0945(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    *,
    last_soft_stall_round: int | None = None,
    request_nonce: str = "",
) -> StallDecision:
    """Add an explicit, domain-independent one-request Panel mode."""

    metadata = metadata if isinstance(metadata, Mapping) else {}
    moa = metadata.get("moa")
    if isinstance(moa, Mapping) and moa.get("enabled") is False:
        return evaluate_collaboration_gate_v093(
            messages,
            metadata,
            last_soft_stall_round=last_soft_stall_round,
        )
    if not (
        isinstance(moa, Mapping) and moa.get("mode") == "panel_once"
    ):
        return evaluate_collaboration_gate_v093(
            messages,
            metadata,
            last_soft_stall_round=last_soft_stall_round,
        )

    admitted = evaluate_exact_action_gate(
        messages,
        {
            "moa": {
                "hard_stall": True,
                "reason": PANEL_ONCE_REASON,
            }
        },
    )
    fingerprint = hashlib.sha256(
        (
            admitted.fingerprint
            + "\0"
            + request_nonce
            + "\0"
            + PANEL_ONCE_REASON
        ).encode("utf-8")
    ).hexdigest()
    return StallDecision(
        admitted=True,
        reason=PANEL_ONCE_REASON,
        fingerprint=fingerprint,
        evidence_refs=admitted.evidence_refs,
        episode_id=fingerprint,
        state_epoch=admitted.state_epoch,
        stalled_action=admitted.stalled_action,
        stalled_action_signatures=admitted.stalled_action_signatures,
        last_observation=admitted.last_observation,
        last_observation_signature=admitted.last_observation_signature,
    )


evaluate_collaboration_gate_v091 = evaluate_collaboration_gate_v093
evaluate_collaboration_gate_v092 = evaluate_collaboration_gate_v093


__all__ = [
    "SOFT_STALL_REASON",
    "SOFT_STALL_TOOL_CYCLES",
    "PANEL_ONCE_REASON",
    "count_tool_cycles",
    "evaluate_collaboration_gate_v09",
    "evaluate_collaboration_gate_v091",
    "evaluate_collaboration_gate_v092",
    "evaluate_collaboration_gate_v093",
    "evaluate_collaboration_gate_v0945",
]
