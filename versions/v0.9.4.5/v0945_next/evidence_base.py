"""Generic, immutable evidence snapshots and a structural collaboration gate."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence, Tuple

from .generic_portfolio import PortfolioSelection
from .task_identity import GENERAL_TASK_TYPE, GenericTaskIdentity


MAX_CONTEXT_BYTES = 65_536
MAX_RECENT_MESSAGES = 12
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


class EvidenceSnapshotError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise EvidenceSnapshotError("snapshot must be strict JSON") from exc


def _safe_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise EvidenceSnapshotError(f"{field_name} is invalid")
    return value


def _message_projection(message: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "role": str(message.get("role") or "unknown")[:32],
        "content": message.get("content"),
    }
    for field_name in ("name", "tool_call_id", "tool_calls"):
        if field_name in message:
            projected[field_name] = message[field_name]
    # Reject non-JSON client objects before they can enter an Analyst request.
    return json.loads(_canonical_json(projected))


def _message_ref(index: int, message: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json(_message_projection(message)).encode("utf-8")
    ).hexdigest()[:20]
    return f"message:{index}:{digest}"


def _signature(message: Mapping[str, Any]) -> str:
    projected = _message_projection(message)
    # OpenAI-compatible servers generate a new ID for every tool call. Those
    # transport IDs must not make an otherwise identical action/observation
    # pair look different to the structural loop detector.
    projected.pop("tool_call_id", None)
    tool_calls = projected.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_call.pop("id", None)
    return hashlib.sha256(
        _canonical_json(projected).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class StallDecision:
    admitted: bool
    reason: str
    fingerprint: str
    evidence_refs: Tuple[str, ...] = ()


def evaluate_collaboration_gate(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> StallDecision:
    """Admit collaboration only from caller evidence or a repeated trajectory.

    No task name, keyword, domain label, benchmark, or model confidence is used.
    The automatic rule requires the same assistant action and the same following
    observation to appear twice in the recent trajectory.
    """

    metadata = metadata if isinstance(metadata, Mapping) else {}
    moa = metadata.get("moa")
    if isinstance(moa, Mapping) and moa.get("enabled") is False:
        return StallDecision(False, "caller_disabled", "")
    if isinstance(moa, Mapping) and moa.get("hard_stall") is True:
        raw_reason = str(moa.get("reason") or "caller_reported_hard_stall")
        reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw_reason)[:80]
        reason = reason.strip("_.:-") or "caller_reported_hard_stall"
        refs = tuple(
            _message_ref(index, message)
            for index, message in enumerate(messages[-2:], start=max(0, len(messages) - 2))
            if isinstance(message, Mapping)
        )
        fingerprint = hashlib.sha256(
            _canonical_json(
                {"reason": reason, "evidence_refs": refs}
            ).encode("utf-8")
        ).hexdigest()
        return StallDecision(True, reason, fingerprint, refs)

    pairs: list[tuple[str, str, str, str]] = []
    for index in range(max(0, len(messages) - 10), len(messages) - 1):
        first = messages[index]
        second = messages[index + 1]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            continue
        if first.get("role") != "assistant":
            continue
        if second.get("role") not in {"tool", "user"}:
            continue
        pairs.append(
            (
                _signature(first),
                _signature(second),
                _message_ref(index, first),
                _message_ref(index + 1, second),
            )
        )

    if len(pairs) >= 2 and pairs[-1][:2] == pairs[-2][:2]:
        refs = tuple(sorted(set(pairs[-2][2:] + pairs[-1][2:])))
        fingerprint = hashlib.sha256(
            _canonical_json(
                {"pair": pairs[-1][:2], "evidence_refs": refs}
            ).encode("utf-8")
        ).hexdigest()
        return StallDecision(
            True,
            "repeated_action_observation",
            fingerprint,
            refs,
        )
    return StallDecision(False, "no_structural_stall", "")


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Compatibility surface for the audited v0.7 Evidence Packet machinery."""

    snapshot_id: str
    digest: str
    run_id: str
    session_id: str
    task_name: str
    task_family: str
    portfolio_id: str
    stall_reason: str
    stall_id: str
    evidence_version: str
    allowed_evidence_refs: Tuple[str, ...]
    _context_json: str = field(repr=False)

    @property
    def context(self) -> dict[str, Any]:
        return json.loads(self._context_json)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Mapping[str, Any]],
        *,
        identity: GenericTaskIdentity,
        selection: PortfolioSelection,
        run_id: str,
        stall: StallDecision,
        source_refs: Sequence[str] | None = None,
        truncated_refs: frozenset[str] = frozenset(),
    ) -> "EvidenceSnapshot":
        if not stall.admitted or not stall.fingerprint:
            raise EvidenceSnapshotError("an admitted structural stall is required")
        if selection.status != "selected" or selection.portfolio is None:
            raise EvidenceSnapshotError("a selected generic portfolio is required")

        safe_run_id = _safe_identifier(run_id, "run_id")
        task_name = _safe_identifier(identity.internal_task_id, "task_name")
        session_digest = hashlib.sha256(
            f"{safe_run_id}\0{task_name}".encode("utf-8")
        ).hexdigest()[:24]
        session_id = f"session-{session_digest}"
        stall_id = f"stall-{stall.fingerprint[:24]}"

        start = max(0, len(messages) - MAX_RECENT_MESSAGES)
        trajectory = []
        refs = []
        for index in range(start, len(messages)):
            message = messages[index]
            if not isinstance(message, Mapping):
                continue
            evidence_ref = (
                source_refs[index]
                if source_refs is not None
                else _message_ref(index, message)
            )
            refs.append(evidence_ref)
            entry = {
                "evidence_ref": evidence_ref,
                "message": _message_projection(message),
            }
            if evidence_ref in truncated_refs:
                entry["truncated"] = True
            trajectory.append(entry)
        if not trajectory:
            raise EvidenceSnapshotError("at least one message is required")

        allowed_refs = tuple(sorted(set(refs)))
        portfolio_id = selection.portfolio_id
        assert portfolio_id is not None
        base_context = {
            "phase": "v0.8-general-evidence",
            "run_id": safe_run_id,
            "session_id": session_id,
            "task_id": task_name,
            "task_type": GENERAL_TASK_TYPE,
            "portfolio_id": portfolio_id,
            "stall": {
                "reason": stall.reason,
                "fingerprint": stall.fingerprint,
                "trigger_evidence_refs": [
                    ref for ref in stall.evidence_refs if ref in allowed_refs
                ],
            },
            "allowed_evidence_refs": list(allowed_refs),
            "recent_trajectory": trajectory,
        }
        digest = hashlib.sha256(
            _canonical_json(base_context).encode("utf-8")
        ).hexdigest()
        snapshot_id = f"snapshot_{digest[:24]}"
        context_json = _canonical_json(
            {"evidence_snapshot_id": snapshot_id, **base_context}
        )
        if len(context_json.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise EvidenceSnapshotError("evidence snapshot is too large")

        return cls(
            snapshot_id=snapshot_id,
            digest=digest,
            run_id=safe_run_id,
            session_id=session_id,
            task_name=task_name,
            task_family=GENERAL_TASK_TYPE,
            portfolio_id=portfolio_id,
            stall_reason=stall.reason,
            stall_id=stall_id,
            evidence_version="v08-general-v1",
            allowed_evidence_refs=allowed_refs,
            _context_json=context_json,
        )


__all__ = [
    "EvidenceSnapshot",
    "EvidenceSnapshotError",
    "StallDecision",
    "evaluate_collaboration_gate",
]
