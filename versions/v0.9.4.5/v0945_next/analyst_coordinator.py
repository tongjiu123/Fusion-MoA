"""Independent, no-retry Analyst request coordination for v0.7."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Tuple

from .generic_portfolio import (
    AnalystRoleSpec,
    PortfolioSelection,
)
from .evidence_packets import PacketBinding, parse_evidence_packet
from .evidence_base import EvidenceSnapshot


SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "hypothesis_v3.json"
HYPOTHESIS_V3_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
MAX_RESPONSE_TOKENS = 8192
MAX_DEADLINE_SECONDS = 600.0
MAX_REQUEST_BYTES = 96 * 1024
CALL_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DuplicateAnalystBatchError(RuntimeError):
    """Raised when the same stall/evidence call key is consumed twice."""


class CallKeyLedger(Protocol):
    """Atomically consume an Analyst call key exactly once."""

    def consume_once(self, call_key: str) -> bool:
        """Return true only for the first consumption of ``call_key``."""


def _validate_call_key(call_key: Any) -> str:
    if not isinstance(call_key, str) or not CALL_KEY_PATTERN.fullmatch(call_key):
        raise ValueError("call_key must be a lowercase SHA-256 hex digest")
    return call_key


class InMemoryCallKeyLedger:
    """Thread-safe process-local call-key ledger."""

    def __init__(self, consumed_call_keys: Optional[set[str]] = None):
        self._consumed = set(consumed_call_keys or ())
        for call_key in self._consumed:
            _validate_call_key(call_key)
        self._lock = threading.Lock()

    def consume_once(self, call_key: str) -> bool:
        safe_key = _validate_call_key(call_key)
        with self._lock:
            if safe_key in self._consumed:
                return False
            self._consumed.add(safe_key)
            return True


@dataclass(frozen=True)
class AnalystTransportResponse:
    content: str
    usage: Mapping[str, Any]


@dataclass(frozen=True)
class AnalystRequest:
    """One role assignment over the same immutable Evidence Snapshot."""

    portfolio_id: str
    task_family: str
    role_spec: AnalystRoleSpec
    binding: PacketBinding
    snapshot: EvidenceSnapshot
    _response_schema_json: str = field(repr=False)

    @property
    def analyst_role(self) -> str:
        return self.role_spec.role

    @property
    def actor_hint(self) -> str:
        return self.role_spec.actor_hint

    @property
    def expected_hypothesis_id(self) -> str:
        return self.binding.hypothesis_id

    @property
    def valid_evidence_refs(self) -> Tuple[str, ...]:
        return self.snapshot.allowed_evidence_refs

    @property
    def response_schema(self) -> dict[str, Any]:
        return json.loads(self._response_schema_json)

    @property
    def system_prompt(self) -> str:
        return (
            "You are one independent, read-only diagnostic Analyst. "
            f"Your assigned strategy is {self.analyst_role}: "
            f"{self.role_spec.focus} "
            "Use only the frozen Evidence Snapshot in the user message. "
            "Return exactly one strict JSON Evidence Packet matching the schema, or "
            "a legal abstain packet when evidence is insufficient or conflicting. "
            "Independent agreement with another possible diagnosis is valid; do not "
            "invent a different conclusion for diversity. Do not emit commands, tool "
            "calls, patches, executable code, long chain-of-thought, confidence, or "
            "information-gain scores."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "task_family": self.task_family,
            "analyst_role": self.analyst_role,
            "actor_hint": self.actor_hint,
            "expected_hypothesis_id": self.expected_hypothesis_id,
            "evidence_snapshot_id": self.snapshot.snapshot_id,
            "valid_evidence_refs": list(self.valid_evidence_refs),
            "retry_policy": "none",
            "read_only": True,
        }

    def to_openai_request_body(
        self,
        *,
        model: str,
        max_tokens: int = 768,
    ) -> dict[str, Any]:
        if not isinstance(model, str) or not MODEL_NAME_PATTERN.fullmatch(model):
            raise ValueError("Analyst model name is invalid")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= MAX_RESPONSE_TOKENS
        ):
            raise ValueError(
                f"max_tokens must be in [1, {MAX_RESPONSE_TOKENS}]"
            )
        assignment = {
            "hypothesis_id": self.expected_hypothesis_id,
            "analyst_role": self.analyst_role,
            "focus": self.role_spec.focus,
        }
        wire_schema = self.response_schema
        # These two packet variants have mutually exclusive status constants,
        # so anyOf is equivalent here and works with vLLM's grammar backend.
        # Only the wire copy changes; final packet validation stays unchanged.
        wire_schema["anyOf"] = wire_schema.pop("oneOf")
        definitions = wire_schema.get("$defs")
        if isinstance(definitions, dict):
            for definition_name in (
                "evidence_refs",
                "supporting_evidence_refs",
            ):
                definition = definitions.get(definition_name)
                if isinstance(definition, dict):
                    # Current vLLM structured-output backends reject the JSON
                    # Schema uniqueItems keyword. The final fail-closed parser
                    # below the transport independently rejects duplicate refs.
                    definition.pop("uniqueItems", None)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "evidence_snapshot": self.snapshot.context,
                            "analyst_assignment": assignment,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"phase_d_{self.analyst_role}_v3",
                    "strict": True,
                    "schema": wire_schema,
                },
            },
        }
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError(
                f"serialized Analyst request exceeds {MAX_REQUEST_BYTES} bytes"
            )
        return body


@dataclass(frozen=True)
class CompetitionPlan:
    """Runtime-compatible container for one independent Analyst batch."""

    run_id: str
    session_id: str
    task_name: str
    remaining_wall_seconds: float
    call_key: str
    routing: PortfolioSelection
    snapshot: Optional[EvidenceSnapshot]
    requests: Tuple[AnalystRequest, ...]
    retry_policy: str = "none"
    development_only: bool = True
    hard_stall: bool = True

    @property
    def workspace_digest(self) -> str:
        return self.snapshot.digest if self.snapshot is not None else ""

    @property
    def stall_reason(self) -> str:
        return self.snapshot.stall_reason if self.snapshot is not None else ""

    @property
    def stall_id(self) -> str:
        return self.snapshot.stall_id if self.snapshot is not None else ""

    @property
    def evidence_version(self) -> str:
        return self.snapshot.evidence_version if self.snapshot is not None else ""

    @property
    def projected_workspace(self) -> Optional[dict[str, Any]]:
        if self.snapshot is None:
            return None
        return copy.deepcopy(self.snapshot.context["workspace"])

    @classmethod
    def from_snapshot(
        cls,
        snapshot: EvidenceSnapshot,
        *,
        routing: PortfolioSelection,
        remaining_wall_seconds: float,
        model_by_actor: Optional[Mapping[str, str]] = None,
    ) -> "CompetitionPlan":
        requests = build_analyst_requests(
            snapshot,
            routing,
            model_by_actor=model_by_actor,
        )
        call_key = hashlib.sha256(
            (
                f"v07\0{snapshot.digest}\0{routing.portfolio_id or 'abstain'}"
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            run_id=snapshot.run_id,
            session_id=snapshot.session_id,
            task_name=snapshot.task_name,
            remaining_wall_seconds=float(remaining_wall_seconds),
            call_key=call_key,
            routing=routing,
            snapshot=snapshot,
            requests=requests,
        )


@dataclass(frozen=True)
class AnalystBatchResult:
    raw_responses: dict[str, str] = field(repr=False)
    validated_packets: Tuple[dict[str, Any], ...] = field(repr=False)
    audit: Tuple[dict[str, Any], ...]
    request_count: int
    response_received_count: int
    schema_valid_count: int
    total_tokens: int
    total_request_wall_seconds: float
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit": [dict(item) for item in self.audit],
            "request_count": self.request_count,
            "response_received_count": self.response_received_count,
            "schema_valid_count": self.schema_valid_count,
            "schema_valid_rate": (
                self.schema_valid_count / self.request_count
                if self.request_count
                else None
            ),
            "total_tokens": self.total_tokens,
            "total_request_wall_seconds": self.total_request_wall_seconds,
            "retry_count": self.retry_count,
        }


AnalystCaller = Callable[
    [AnalystRequest, Mapping[str, Any]],
    Awaitable[AnalystTransportResponse],
]


def _response_schema(
    hypothesis_id: str,
    evidence_refs: Tuple[str, ...],
) -> dict[str, Any]:
    schema = copy.deepcopy(HYPOTHESIS_V3_SCHEMA)
    schema["$defs"]["hypothesis_id"]["enum"] = [hypothesis_id]
    for definition in ("evidence_refs", "supporting_evidence_refs"):
        schema["$defs"][definition]["items"]["enum"] = list(evidence_refs)
    return schema


def build_analyst_requests(
    snapshot: EvidenceSnapshot,
    routing: PortfolioSelection,
    *,
    model_by_actor: Optional[Mapping[str, str]] = None,
) -> Tuple[AnalystRequest, ...]:
    """Build three independent assignments or no calls for an untrusted family."""
    if routing.status != "selected" or routing.portfolio is None:
        return ()
    if (
        routing.task_type != snapshot.task_family
        or routing.portfolio_id != snapshot.portfolio_id
        or not routing.portfolio.supports(snapshot.task_family)
    ):
        return ()
    if len(routing.portfolio.roles) != 3:
        return ()
    actor_hints = [role.actor_hint for role in routing.portfolio.roles]
    roles = [role.role for role in routing.portfolio.roles]
    if len(set(actor_hints)) != 3 or len(set(roles)) != 3:
        return ()

    requests = []
    for role_spec in routing.portfolio.roles:
        hypothesis_id = f"h_{role_spec.actor_hint.lower()}_01"
        served_model = (
            model_by_actor.get(role_spec.actor_hint)
            if isinstance(model_by_actor, Mapping)
            else role_spec.actor_hint
        )
        if (
            not isinstance(served_model, str)
            or not MODEL_NAME_PATTERN.fullmatch(served_model)
        ):
            return ()
        binding = PacketBinding(
            hypothesis_id=hypothesis_id,
            analyst_model=served_model,
            analyst_role=role_spec.role,
            task_family=snapshot.task_family,
            evidence_snapshot_id=snapshot.snapshot_id,
            run_id=snapshot.run_id,
            session_id=snapshot.session_id,
            task_name=snapshot.task_name,
        )
        schema_json = json.dumps(
            _response_schema(hypothesis_id, snapshot.allowed_evidence_refs),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        requests.append(
            AnalystRequest(
                portfolio_id=snapshot.portfolio_id,
                task_family=snapshot.task_family,
                role_spec=role_spec,
                binding=binding,
                snapshot=snapshot,
                _response_schema_json=schema_json,
            )
        )
    return tuple(requests)


def _safe_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, Mapping):
        return {}
    result = {}
    for field_name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
    ):
        value = usage.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[field_name] = value
    return result


def _usage_total(usage: Mapping[str, int]) -> int:
    total = usage.get("total_tokens")
    if total is not None:
        return total
    return usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)


class AnalystCallCoordinator:
    """Prepare every body first, then call each read-only Analyst once."""

    def __init__(
        self,
        *,
        model_by_actor: Mapping[str, str],
        call_key_ledger: Optional[CallKeyLedger] = None,
        max_tokens: int = 768,
    ):
        if not isinstance(model_by_actor, Mapping):
            raise TypeError("model_by_actor must be a mapping")
        models = {}
        for actor, model in model_by_actor.items():
            if not isinstance(actor, str) or not actor:
                raise ValueError("model_by_actor keys must be non-empty strings")
            if not isinstance(model, str) or not MODEL_NAME_PATTERN.fullmatch(model):
                raise ValueError(f"invalid Analyst model for {actor}")
            models[actor] = model
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= MAX_RESPONSE_TOKENS
        ):
            raise ValueError(
                f"max_tokens must be in [1, {MAX_RESPONSE_TOKENS}]"
            )
        ledger = call_key_ledger or InMemoryCallKeyLedger()
        if not callable(getattr(ledger, "consume_once", None)):
            raise TypeError("call_key_ledger must implement consume_once")
        self.model_by_actor = models
        self.call_key_ledger = ledger
        self.max_tokens = max_tokens

    async def run(
        self,
        plan: CompetitionPlan,
        caller: AnalystCaller,
        *,
        deadline_seconds: float,
    ) -> AnalystBatchResult:
        if plan.retry_policy != "none":
            raise ValueError("v0.7 Analyst calls must use retry_policy=none")
        if not plan.requests:
            return AnalystBatchResult({}, (), (), 0, 0, 0, 0, 0.0, 0)
        if (
            not isinstance(deadline_seconds, (int, float))
            or isinstance(deadline_seconds, bool)
            or not math.isfinite(deadline_seconds)
            or not 0 < deadline_seconds <= MAX_DEADLINE_SECONDS
        ):
            raise ValueError(
                f"deadline_seconds must be in (0, {MAX_DEADLINE_SECONDS}]"
            )
        remaining_wall_seconds = plan.remaining_wall_seconds
        if (
            not isinstance(remaining_wall_seconds, (int, float))
            or isinstance(remaining_wall_seconds, bool)
            or not math.isfinite(remaining_wall_seconds)
            or remaining_wall_seconds <= 0
            or deadline_seconds > remaining_wall_seconds
        ):
            raise ValueError("deadline exceeds the trusted remaining wall budget")
        if not callable(caller):
            raise TypeError("caller must be callable")

        snapshot = plan.snapshot
        if snapshot is None or plan.routing.status != "selected":
            raise ValueError("Analyst requests require one selected frozen snapshot")
        if len(plan.requests) != 3:
            raise ValueError("a selected v0.7 portfolio must contain three requests")
        if (
            plan.run_id != snapshot.run_id
            or plan.session_id != snapshot.session_id
            or plan.task_name != snapshot.task_name
            or plan.routing.task_type != snapshot.task_family
            or plan.routing.portfolio_id != snapshot.portfolio_id
        ):
            raise ValueError("competition plan identity does not match its snapshot")
        for request in plan.requests:
            if request.snapshot is not snapshot:
                raise ValueError("all Analyst requests must share one snapshot object")
            binding = request.binding
            if (
                binding.run_id != snapshot.run_id
                or binding.session_id != snapshot.session_id
                or binding.task_name != snapshot.task_name
                or binding.task_family != snapshot.task_family
                or binding.evidence_snapshot_id != snapshot.snapshot_id
            ):
                raise ValueError("Analyst binding does not match the frozen snapshot")

        # This complete preparation pass happens before any network call.
        prepared = []
        for request in plan.requests:
            if not isinstance(request, AnalystRequest):
                raise TypeError("plan.requests must contain AnalystRequest instances")
            model = self.model_by_actor.get(request.actor_hint)
            if model is None:
                raise KeyError(f"Analyst model is not configured: {request.actor_hint}")
            if request.binding.analyst_model != model:
                raise ValueError("Analyst model binding does not match served model")
            prepared.append(
                (
                    request,
                    request.to_openai_request_body(
                        model=model,
                        max_tokens=self.max_tokens,
                    ),
                )
            )

        if not self.call_key_ledger.consume_once(plan.call_key):
            raise DuplicateAnalystBatchError(
                "Analyst batch call_key was already consumed"
            )

        async def invoke(request: AnalystRequest, body: Mapping[str, Any]):
            started = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    caller(request, body),
                    timeout=float(deadline_seconds),
                )
            except Exception as exc:
                return request, None, exc, time.monotonic() - started
            return request, response, None, time.monotonic() - started

        results = await asyncio.gather(
            *(invoke(request, body) for request, body in prepared)
        )
        raw_responses: dict[str, str] = {}
        validated_packets = []
        audit = []
        response_count = 0
        valid_count = 0
        total_tokens = 0
        total_wall = 0.0
        for request, response, error, wall_seconds in results:
            total_wall += wall_seconds
            if error is not None or response is None:
                audit.append(
                    {
                        "analyst_role": request.analyst_role,
                        "actor": request.actor_hint,
                        "portfolio_id": request.portfolio_id,
                        "task_family": request.task_family,
                        "evidence_snapshot_id": request.snapshot.snapshot_id,
                        "status": "request_error",
                        "error_type": type(error).__name__ if error else "unknown",
                        "schema_valid": False,
                        "request_wall_seconds": round(wall_seconds, 6),
                        "retry_count": 0,
                    }
                )
                continue
            raw = getattr(response, "content", None)
            usage = _safe_usage(getattr(response, "usage", {}))
            if not isinstance(raw, str):
                audit.append(
                    {
                        "analyst_role": request.analyst_role,
                        "actor": request.actor_hint,
                        "portfolio_id": request.portfolio_id,
                        "task_family": request.task_family,
                        "evidence_snapshot_id": request.snapshot.snapshot_id,
                        "status": "protocol_error",
                        "error_type": "invalid_transport_response",
                        "schema_valid": False,
                        "request_wall_seconds": round(wall_seconds, 6),
                        "retry_count": 0,
                    }
                )
                continue

            response_count += 1
            raw_responses[request.analyst_role] = raw
            outcome = parse_evidence_packet(
                raw,
                binding=request.binding,
                allowed_evidence_refs=request.valid_evidence_refs,
            )
            if outcome.valid and outcome.packet is not None:
                validated_packets.append(copy.deepcopy(outcome.packet))
                valid_count += 1
            tokens = _usage_total(usage)
            total_tokens += tokens
            audit.append(
                {
                    "analyst_role": request.analyst_role,
                    "actor": request.actor_hint,
                    "portfolio_id": request.portfolio_id,
                    "task_family": request.task_family,
                    "evidence_snapshot_id": request.snapshot.snapshot_id,
                    "status": "schema_valid" if outcome.valid else "schema_invalid",
                    "packet_status": (
                        outcome.packet.get("status")
                        if outcome.packet is not None
                        else None
                    ),
                    "schema_valid": outcome.valid,
                    "validation_errors": list(outcome.errors),
                    "response_sha256": hashlib.sha256(
                        raw.encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "response_bytes": len(raw.encode("utf-8", errors="replace")),
                    "usage": usage,
                    "total_tokens": tokens,
                    "request_wall_seconds": round(wall_seconds, 6),
                    "retry_count": 0,
                }
            )

        return AnalystBatchResult(
            raw_responses=raw_responses,
            validated_packets=tuple(validated_packets),
            audit=tuple(audit),
            request_count=len(plan.requests),
            response_received_count=response_count,
            schema_valid_count=valid_count,
            total_tokens=total_tokens,
            total_request_wall_seconds=round(total_wall, 6),
            retry_count=0,
        )


__all__ = [
    "AnalystBatchResult",
    "AnalystCallCoordinator",
    "AnalystRequest",
    "AnalystTransportResponse",
    "CompetitionPlan",
    "DuplicateAnalystBatchError",
    "InMemoryCallKeyLedger",
    "build_analyst_requests",
]
