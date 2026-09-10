"""Exact action-aware cycle evidence for Fusion-MOA v0.9.4.5."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .evidence_base import (
    MAX_RECENT_MESSAGES,
    EvidenceSnapshot,
    EvidenceSnapshotError,
    _message_ref as snapshot_message_ref,
)
from .evidence_episode import (
    EpisodeState,
    StallDecision,
    action_item_signatures,
    action_projection,
    action_signature,
    evaluate_collaboration_gate as evaluate_exact_action_gate,
    observation_projection,
    observation_signature,
    recovery_evidence as exact_recovery_evidence,
)


RESULT_CYCLE_REASON = "repeated_result_cycle"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _message_ref(index: int, message: Mapping[str, Any]) -> str:
    projection = {
        "role": str(message.get("role") or "unknown")[:32],
        "content": message.get("content"),
        "name": message.get("name"),
        "tool_call_id": message.get("tool_call_id"),
    }
    digest = hashlib.sha256(
        _canonical_json(projection).encode("utf-8")
    ).hexdigest()[:20]
    return f"message:{index}:{digest}"


def _snapshot_or_none(
    messages: Sequence[Mapping[str, Any]],
    *,
    identity: Any,
    selection: Any,
    run_id: str,
    stall: StallDecision,
    source_refs: Sequence[str] | None = None,
    truncated_refs: frozenset[str] = frozenset(),
) -> EvidenceSnapshot | None:
    try:
        return EvidenceSnapshot.from_messages(
            messages,
            identity=identity,
            selection=selection,
            run_id=run_id,
            stall=stall,
            source_refs=source_refs,
            truncated_refs=truncated_refs,
        )
    except EvidenceSnapshotError:
        return None


def build_bounded_evidence_snapshot(
    messages: Sequence[Mapping[str, Any]],
    *,
    identity: Any,
    selection: Any,
    run_id: str,
    stall: StallDecision,
) -> EvidenceSnapshot:
    indexed = [
        (index, dict(message))
        for index, message in enumerate(messages)
        if isinstance(message, Mapping)
    ]
    if not indexed:
        raise EvidenceSnapshotError("at least one message is required")

    source_refs: dict[int, str] = {}
    trigger_indices: list[int] = []
    for evidence_ref in stall.evidence_refs:
        parts = evidence_ref.split(":", 2)
        if len(parts) != 3 or parts[0] != "message":
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        if 0 <= index < len(messages):
            trigger_indices.append(index)
            # Preserve the detector's existing source reference, including
            # references produced by a different structural detector.
            source_refs[index] = evidence_ref

    available = {index: message for index, message in indexed}
    required_indices = sorted(set(trigger_indices))[-2:]
    if not required_indices:
        required_indices = [index for index, _ in indexed[-2:]]
    user_indices = [
        index for index, message in indexed if message.get("role") == "user"
    ]
    if user_indices:
        required_indices = sorted(
            set(required_indices) | {user_indices[0], user_indices[-1]}
        )
    selected = [
        (index, dict(available[index]))
        for index in required_indices
        if index in available
    ]

    def snapshot_for(
        candidate: Sequence[tuple[int, Mapping[str, Any]]],
    ) -> EvidenceSnapshot | None:
        for index, _ in candidate:
            if index not in source_refs:
                source_refs[index] = snapshot_message_ref(index, available[index])
        truncated_refs = frozenset(
            source_refs[index]
            for index, message in candidate
            if message.get("content") != available[index].get("content")
        )
        return _snapshot_or_none(
            [message for _, message in candidate],
            identity=identity,
            selection=selection,
            run_id=run_id,
            stall=stall,
            source_refs=[source_refs[index] for index, _ in candidate],
            truncated_refs=truncated_refs,
        )

    snapshot = snapshot_for(selected)
    if snapshot is None:
        content_position = next(
            (
                position
                for position in range(len(selected) - 1, -1, -1)
                if isinstance(selected[position][1].get("content"), str)
            ),
            None,
        )
        if content_position is None:
            raise EvidenceSnapshotError(
                "required cycle evidence cannot fit the snapshot"
            )
        original_content = selected[content_position][1]["content"]
        low = 0
        high = len(original_content)
        fitted: list[tuple[int, dict[str, Any]]] | None = None
        fitted_snapshot: EvidenceSnapshot | None = None
        while low <= high:
            keep = (low + high) // 2
            candidate = [
                (index, dict(message))
                for index, message in selected
            ]
            candidate[content_position][1]["content"] = (
                original_content[-keep:] if keep else ""
            )
            candidate_snapshot = snapshot_for(candidate)
            if candidate_snapshot is None:
                high = keep - 1
            else:
                fitted = candidate
                fitted_snapshot = candidate_snapshot
                low = keep + 1
        if fitted is None or fitted_snapshot is None:
            raise EvidenceSnapshotError(
                "required cycle evidence cannot fit the snapshot"
            )
        selected = fitted
        snapshot = fitted_snapshot

    selected_indices = {index for index, _ in selected}
    for index, message in reversed(indexed):
        if (
            index in selected_indices
            or len(selected) >= MAX_RECENT_MESSAGES
        ):
            continue
        candidate = sorted(
            [*selected, (index, dict(message))],
            key=lambda item: item[0],
        )
        candidate_snapshot = snapshot_for(candidate)
        if candidate_snapshot is None:
            continue
        selected = candidate
        selected_indices.add(index)
        snapshot = candidate_snapshot

    if snapshot is None:
        raise EvidenceSnapshotError("evidence snapshot is too large")
    return snapshot


def _tool_rounds(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair standard OpenAI tool results with their declared tool names."""

    pending: dict[str, dict[str, Any]] = {}
    rounds: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                call_id = tool_call.get("id")
                function = tool_call.get("function")
                function = function if isinstance(function, Mapping) else {}
                tool_name = function.get("name")
                if (
                    isinstance(call_id, str)
                    and call_id
                    and isinstance(tool_name, str)
                    and tool_name
                ):
                    pending[call_id] = {
                        "action_index": index,
                        "action": message,
                        "tool_name": tool_name,
                    }
            continue
        if role != "tool":
            continue
        call_id = message.get("tool_call_id")
        source = pending.pop(call_id, None) if isinstance(call_id, str) else None
        explicit_name = message.get("name")
        tool_name = (
            explicit_name
            if isinstance(explicit_name, str) and explicit_name
            else source.get("tool_name") if source else None
        )
        if not isinstance(tool_name, str) or not tool_name:
            continue
        action = source.get("action") if source else {}
        action_index = int(source.get("action_index", index)) if source else index
        rounds.append(
            {
                "key": (
                    action_signature(action),
                    tool_name,
                    observation_signature(message),
                ),
                "tool_name": tool_name,
                "action_index": action_index,
                "action": action,
                "observation_index": index,
                "observation": message,
            }
        )
    return rounds


def _canonical_cycle(
    pattern: Sequence[tuple[str, str, str]],
) -> tuple[tuple[str, str, str], ...]:
    rotations = [
        tuple(pattern[index:]) + tuple(pattern[:index])
        for index in range(len(pattern))
    ]
    return min(rotations)


def detect_result_cycle(
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Find the shortest exact action-and-result cycle at the tail."""

    rounds = _tool_rounds(messages)
    for period in range(1, len(rounds) // 2 + 1):
        previous = rounds[-2 * period : -period]
        current = rounds[-period:]
        if [item["key"] for item in previous] != [
            item["key"] for item in current
        ]:
            continue
        pattern = _canonical_cycle([item["key"] for item in current])
        episode_id = hashlib.sha256(
            _canonical_json(
                {
                    "reason": RESULT_CYCLE_REASON,
                    "pattern": pattern,
                }
            ).encode("utf-8")
        ).hexdigest()
        return {
            "period": period,
            "episode_id": episode_id,
            "pattern": pattern,
            "tail": previous + current,
        }
    return None


def _result_cycle_decision(
    cycle: Mapping[str, Any],
) -> StallDecision:
    tail = cycle["tail"]
    latest = tail[-1]
    cycle_action_signatures = tuple(
        sorted(
            {
                signature
                for item in tail
                for signature in action_item_signatures(item["action"])
            }
        )
    )
    refs = tuple(
        sorted(
            {
                ref
                for item in tail
                for ref in (
                    _message_ref(int(item["action_index"]), item["action"]),
                    _message_ref(
                        int(item["observation_index"]),
                        item["observation"],
                    ),
                )
            }
        )
    )
    episode_id = str(cycle["episode_id"])
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "reason": RESULT_CYCLE_REASON,
                "episode_id": episode_id,
                "evidence_refs": refs,
            }
        ).encode("utf-8")
    ).hexdigest()
    return StallDecision(
        admitted=True,
        reason=RESULT_CYCLE_REASON,
        fingerprint=fingerprint,
        evidence_refs=refs,
        episode_id=episode_id,
        stalled_action=action_projection(latest["action"]),
        stalled_action_signatures=cycle_action_signatures,
        last_observation=observation_projection(latest["observation"]),
        last_observation_signature=str(latest["key"][2]),
    )


def evaluate_collaboration_gate_v093(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> StallDecision:
    """Evaluate caller hard stall plus exact action/result cycles.

    Periodic Soft Stall is composed by ``soft_stall.py`` so this structural
    detector has no circular dependency on the periodic gate.
    """

    base = evaluate_exact_action_gate(messages, metadata)
    if base.admitted or base.reason == "caller_disabled":
        return base
    cycle = detect_result_cycle(messages)
    if cycle is not None:
        return _result_cycle_decision(cycle)
    return base


def recovery_evidence(
    messages: Sequence[Mapping[str, Any]],
    stall: StallDecision,
) -> dict[str, Any]:
    evidence = exact_recovery_evidence(messages, stall)
    if stall.reason != RESULT_CYCLE_REASON:
        return evidence
    cycle = detect_result_cycle(messages)
    if cycle is None:
        return evidence
    evidence["result_cycle"] = {
        "period": cycle["period"],
        "rounds": [
            {
                "action": action_projection(item["action"]),
                "tool_name": item["tool_name"],
                "observation": observation_projection(item["observation"]),
            }
            for item in cycle["tail"]
        ],
    }
    return evidence


__all__ = [
    "EpisodeState",
    "RESULT_CYCLE_REASON",
    "StallDecision",
    "action_item_signatures",
    "build_bounded_evidence_snapshot",
    "detect_result_cycle",
    "evaluate_collaboration_gate_v093",
    "recovery_evidence",
]
