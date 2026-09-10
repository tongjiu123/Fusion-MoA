"""Stable stall episodes and clean-recovery evidence for v0.9.2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence, Tuple


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _message_projection(message: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "role": str(message.get("role") or "unknown")[:32],
        "content": message.get("content"),
    }
    for field_name in ("name", "tool_call_id", "tool_calls"):
        if field_name in message:
            projected[field_name] = message[field_name]
    return json.loads(_canonical_json(projected))


def _message_ref(index: int, message: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json(_message_projection(message)).encode("utf-8")
    ).hexdigest()[:20]
    return f"message:{index}:{digest}"


def observation_projection(message: Mapping[str, Any]) -> dict[str, Any]:
    projected = _message_projection(message)
    projected.pop("tool_call_id", None)
    return projected


def observation_signature(message: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(observation_projection(message)).encode("utf-8")
    ).hexdigest()


def action_projection(message: Mapping[str, Any]) -> dict[str, Any]:
    """Project one action without reasoning text or transport-generated IDs."""

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        normalized_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function")
            function = function if isinstance(function, Mapping) else {}
            normalized_calls.append(
                {
                    "type": str(tool_call.get("type") or "function"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                    },
                }
            )
        if normalized_calls:
            return {
                "role": str(message.get("role") or "assistant")[:32],
                "tool_calls": normalized_calls,
            }
    return {
        "role": str(message.get("role") or "assistant")[:32],
        "content": message.get("content"),
    }


def action_signature(message: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(action_projection(message)).encode("utf-8")
    ).hexdigest()


def action_item_signatures(message: Mapping[str, Any]) -> Tuple[str, ...]:
    projection = action_projection(message)
    tool_calls = projection.get("tool_calls")
    items = tool_calls if isinstance(tool_calls, list) else [projection]
    return tuple(
        hashlib.sha256(
            _canonical_json(item).encode("utf-8")
        ).hexdigest()
        for item in items
    )


def _episode_id(action: str, observation: str, state_epoch: int) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "action_signature": action,
                "observation_signature": observation,
                "state_epoch": state_epoch,
            }
        ).encode("utf-8")
    ).hexdigest()


def _trajectory_pairs(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build completed action/observation pairs and a structural state epoch."""

    pairs: list[dict[str, Any]] = []
    previous: tuple[str, str] | None = None
    state_epoch = 0
    for index in range(len(messages) - 1):
        first = messages[index]
        second = messages[index + 1]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            continue
        if (
            first.get("role") != "assistant"
            or second.get("role") not in {"tool", "user"}
        ):
            continue
        action = action_signature(first)
        observation = observation_signature(second)
        current = (action, observation)
        if previous is not None and current != previous:
            state_epoch += 1
        pairs.append(
            {
                "action_index": index,
                "action": first,
                "observation_index": index + 1,
                "observation": second,
                "action_signature": action,
                "observation_signature": observation,
                "state_epoch": state_epoch,
            }
        )
        previous = current
    return pairs


@dataclass(frozen=True)
class StallDecision:
    admitted: bool
    reason: str
    fingerprint: str
    evidence_refs: Tuple[str, ...] = ()
    episode_id: str = ""
    state_epoch: int = 0
    stalled_action: Mapping[str, Any] | None = None
    stalled_action_signatures: Tuple[str, ...] = ()
    last_observation: Mapping[str, Any] | None = None
    last_observation_signature: str = ""


@dataclass
class EpisodeState:
    episode_id: str
    guidance: str = ""
    selected_hypothesis_ids: tuple[str, ...] = ()
    deliberation_used: bool = False
    recovery_used: bool = False
    reasoned_repeat_used: bool = False


def recovery_evidence(
    messages: Sequence[Mapping[str, Any]],
    stall: StallDecision,
    *,
    progress_pairs: int = 4,
) -> dict[str, Any]:
    """Return bounded non-loop progress plus the confirmed stalled cycle."""

    pairs = _trajectory_pairs(messages)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stalled_pair = (
        action_signature(stall.stalled_action or {}),
        stall.last_observation_signature,
    )
    repeat_count = 0
    for pair in reversed(pairs):
        signature = (
            str(pair["action_signature"]),
            str(pair["observation_signature"]),
        )
        if signature == stalled_pair:
            if not selected:
                repeat_count += 1
            continue
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(
            {
                "action": action_projection(pair["action"]),
                "observation": observation_projection(
                    pair["observation"]
                ),
            }
        )
        if len(selected) >= progress_pairs:
            break
    selected.reverse()
    return {
        "confirmed_progress": selected,
        "stalled_action": stall.stalled_action,
        "repeated_observation": stall.last_observation,
        "consecutive_repeat_count": repeat_count,
    }


def _decision_from_pair(
    *,
    reason: str,
    pair: Mapping[str, Any],
    evidence_refs: Tuple[str, ...] | None = None,
) -> StallDecision:
    first = pair["action"]
    second = pair["observation"]
    state_epoch = int(pair["state_epoch"])
    episode_id = _episode_id(
        str(pair["action_signature"]),
        str(pair["observation_signature"]),
        state_epoch,
    )
    refs = evidence_refs or (
        _message_ref(int(pair["action_index"]), first),
        _message_ref(int(pair["observation_index"]), second),
    )
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "reason": reason,
                "episode_id": episode_id,
                "evidence_refs": refs,
            }
        ).encode("utf-8")
    ).hexdigest()
    return StallDecision(
        admitted=True,
        reason=reason,
        fingerprint=fingerprint,
        evidence_refs=refs,
        episode_id=episode_id,
        state_epoch=state_epoch,
        stalled_action=action_projection(first),
        stalled_action_signatures=action_item_signatures(first),
        last_observation=observation_projection(second),
        last_observation_signature=str(pair["observation_signature"]),
    )


def evaluate_collaboration_gate(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> StallDecision:
    """Detect explicit or repeated stalls without task classification."""

    metadata = metadata if isinstance(metadata, Mapping) else {}
    moa = metadata.get("moa")
    if isinstance(moa, Mapping) and moa.get("enabled") is False:
        return StallDecision(False, "caller_disabled", "")

    pairs = _trajectory_pairs(messages)
    if isinstance(moa, Mapping) and moa.get("hard_stall") is True:
        raw_reason = str(moa.get("reason") or "caller_reported_hard_stall")
        reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw_reason)[:80]
        reason = reason.strip("_.:-") or "caller_reported_hard_stall"
        if pairs:
            return _decision_from_pair(reason=reason, pair=pairs[-1])
        refs = tuple(
            _message_ref(index, message)
            for index, message in enumerate(
                messages[-2:],
                start=max(0, len(messages) - 2),
            )
            if isinstance(message, Mapping)
        )
        fingerprint = hashlib.sha256(
            _canonical_json(
                {"reason": reason, "evidence_refs": refs}
            ).encode("utf-8")
        ).hexdigest()
        return StallDecision(
            admitted=True,
            reason=reason,
            fingerprint=fingerprint,
            evidence_refs=refs,
            episode_id=fingerprint,
        )

    if (
        len(pairs) >= 2
        and pairs[-1]["action_signature"]
        == pairs[-2]["action_signature"]
        and pairs[-1]["observation_signature"]
        == pairs[-2]["observation_signature"]
        and pairs[-1]["state_epoch"] == pairs[-2]["state_epoch"]
    ):
        refs = tuple(
            sorted(
                {
                    _message_ref(
                        int(pairs[-2]["action_index"]),
                        pairs[-2]["action"],
                    ),
                    _message_ref(
                        int(pairs[-2]["observation_index"]),
                        pairs[-2]["observation"],
                    ),
                    _message_ref(
                        int(pairs[-1]["action_index"]),
                        pairs[-1]["action"],
                    ),
                    _message_ref(
                        int(pairs[-1]["observation_index"]),
                        pairs[-1]["observation"],
                    ),
                }
            )
        )
        return _decision_from_pair(
            reason="repeated_action_observation",
            pair=pairs[-1],
            evidence_refs=refs,
        )
    return StallDecision(False, "no_structural_stall", "")


__all__ = [
    "EpisodeState",
    "StallDecision",
    "action_item_signatures",
    "action_projection",
    "action_signature",
    "evaluate_collaboration_gate",
    "observation_projection",
    "observation_signature",
    "recovery_evidence",
]
