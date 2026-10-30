"""Strict parsing and deterministic selection for v0.7 Evidence Packets.

This module deliberately contains no model calls, JSON repair, confidence
scores, embeddings, or generated mediation.  An analyst supplies only the
schema-defined payload; the gateway supplies and verifies the packet identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


_MAX_RAW_BYTES = 16_384
_MAX_STATEMENT_CHARS = 512
_MAX_EVIDENCE_REF_CHARS = 160
_MAX_EVIDENCE_REFS = 16

_HYPOTHESIS_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,200}$")

_HYPOTHESIS_FIELDS = frozenset(
    {
        "status",
        "hypothesis_id",
        "claim",
        "supporting_evidence_refs",
        "disconfirming_evidence_refs",
        "probe_description",
        "expected_observation",
        "falsification_observation",
    }
)
_ABSTAIN_FIELDS = frozenset(
    {"status", "hypothesis_id", "reason_code"}
)
_BOUND_FIELDS = (
    "analyst_model",
    "analyst_role",
    "task_family",
    "evidence_snapshot_id",
    "run_id",
    "session_id",
    "task_name",
)
_REASON_CODES = frozenset(
    {
        "insufficient_evidence",
        "wrong_task_family",
        "evidence_conflict",
        "no_falsifiable_probe",
    }
)
_FORBIDDEN_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:ignore|override|disregard)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+instructions?\b",
        r"\b(?:bash|sh)\s+-c\b",
        r"\bpython(?:3)?\s+-c\b",
        r"\brm\s+-[A-Za-z]*r[A-Za-z]*\b",
        r"\b(?:sudo|curl|wget)\b",
        r"\bpip(?:3)?\s+install\b",
        r"\bdocker\s+(?:run|exec|kill|rm)\b",
        r"\bchmod\s+(?:\+x|[0-7]{3,4})\b",
        r"\btool[ _-]?call\b",
        r"(?:^|\s)(?:diff --git|--- a/|\+\+\+ b/|@@\s+-\d)",
        r"\$\(|&&|\|\|",
        # Evidence Packets describe a falsifiable observation; they must not
        # carry imperative terminal instructions, executable snippets, or
        # patch operations into the sole executor's context.
        r"^\s*(?:(?:please|kindly)\s+|you\s+(?:must|should|need\s+to)\s+)?(?:run|execute|type|invoke|launch|install|delete|remove|write|edit|apply|replace|insert|append|create)\b",
        r"^\s*(?:请|务必|必须)?\s*(?:运行|执行|输入|调用|安装|删除|移除|应用补丁|修改文件|写入文件)\b",
        r"^\s*(?:please\s+)?use\s+(?:the\s+)?(?:shell|terminal|command(?:\s+line)?|tool)\s+(?:to\s+)?(?:run|execute|invoke|launch|delete|remove|write|edit)\b",
        r"^\s*(?:please\s+)?(?:issue|enter)\s+(?:the\s+)?command\b",
        r"\b(?:shell|terminal|command(?:\s+line)?)\b",
        r"\bwhoami\b",
        r"\bpython(?:3)?\s+(?:--?\S+\s+)*[A-Za-z0-9_./~-]+\.py\b",
        r"(?:^|[\s\"'`])(?:\.{0,2}/|/)?[A-Za-z0-9_.-]+\.(?:sh|py|ps1|bat|cmd)\b",
        r"\b(?:pytest|unittest|git|make|cmake|ninja|npm|npx|yarn|pnpm|cargo|mvn|gradle)\b\s+(?:--?|[A-Za-z0-9_./~])",
        r"\bgo\s+(?:test|build|run|install)\b",
        r"\b(?:touch|cat|ls|cp|mv|grep|rg|sed|awk|find)\b\s+(?:--?|[A-Za-z0-9_./~])",
        r"(?:^|;)\s*(?:please\s+)?import\s+[A-Za-z_][A-Za-z0-9_.]*\b",
        r"(?:^|;)\s*(?:please\s+)?from\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+",
        r"\b(?:os\.system|subprocess\.(?:run|call|popen)|open|eval|exec|compile|__import__)\s*\(",
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:[\"'\d\[\{\(]|[A-Za-z_])",
        r"\b(?:apply|make)\s+(?:this|the)\s+(?:change|patch)\b",
        r"^\s*(?:replace|insert|delete|remove|append|write|edit|create)\b",
        r"^\s*change\s+(?:this|the|[A-Za-z_][A-Za-z0-9_./-]*)\s+(?:to|from|by|with)\b",
        r"(?:^|\s)(?:>|>>|<)\s*(?:[/~.]|[A-Za-z0-9_])",
        r"\b(?:follow|obey)\s+(?:the\s+)?(?:snapshot|analyst|user)\s+(?:message|instructions?)\s+(?:instead\s+of|over)\s+(?:the\s+)?(?:system|developer)\s+(?:message|instructions?)\b",
        r"(?=.*\b(?:snapshot|analyst|user)\b)(?=.*\b(?:system|developer)\b)(?=.*\b(?:instructions?|message)\b)",
    )
)


@dataclass(frozen=True)
class PacketBinding:
    """Identity assigned by the gateway, never by an analyst response."""

    hypothesis_id: str
    analyst_model: str
    analyst_role: str
    task_family: str
    evidence_snapshot_id: str
    run_id: str
    session_id: str
    task_name: str


@dataclass(frozen=True)
class ParseOutcome:
    """Fail-closed parse result.  Invalid responses never carry a packet."""

    valid: bool
    packet: dict[str, Any] | None
    errors: tuple[str, ...] = ()


class _StrictJSONError(ValueError):
    pass


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise _StrictJSONError(f"non-finite JSON number: {value}")


def _strict_json_object(raw_response: str) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        raise _StrictJSONError("response must be a JSON string")
    if not raw_response or len(raw_response.encode("utf-8")) > _MAX_RAW_BYTES:
        raise _StrictJSONError("response size is outside the allowed bounds")
    try:
        value = json.loads(
            raw_response,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError, _StrictJSONError) as exc:
        raise _StrictJSONError(str(exc)) from exc
    if not isinstance(value, dict):
        raise _StrictJSONError("response must be one JSON object")
    if not _all_numbers_finite(value):
        raise _StrictJSONError("response contains a non-finite number")
    return value


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _is_safe_text(value: Any, *, maximum: int) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        return False
    if value != value.strip():
        return False
    # Packets are short statements, not a transport for multiline prompts,
    # terminal control sequences, code fences, or hidden formatting.
    if "```" in value:
        return False
    if any(pattern.search(value) for pattern in _FORBIDDEN_TEXT_PATTERNS):
        return False
    return not any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value
    )


def is_safe_evidence_statement(value: Any) -> bool:
    """Reject obvious executable, patch, tool, or prompt-injection text."""
    return _is_safe_text(value, maximum=_MAX_STATEMENT_CHARS)


def _valid_bound_value(value: Any) -> bool:
    return isinstance(value, str) and bool(_BOUND_ID_RE.fullmatch(value))


def _binding_errors(binding: PacketBinding) -> list[str]:
    if not isinstance(binding, PacketBinding):
        return ["binding must be PacketBinding"]
    errors: list[str] = []
    if not isinstance(binding.hypothesis_id, str) or not (
        _HYPOTHESIS_ID_RE.fullmatch(binding.hypothesis_id)
    ):
        errors.append("invalid bound hypothesis_id")
    for field in _BOUND_FIELDS:
        if not _valid_bound_value(getattr(binding, field)):
            errors.append(f"invalid bound {field}")
    return errors


def _evidence_ref_errors(
    value: Any,
    *,
    field: str,
    require_one: bool,
    allowed: frozenset[str],
) -> list[str]:
    if not isinstance(value, list):
        return [f"{field} must be an array"]
    if (require_one and not value) or len(value) > _MAX_EVIDENCE_REFS:
        return [f"{field} has an invalid item count"]
    if any(
        not _is_safe_text(item, maximum=_MAX_EVIDENCE_REF_CHARS)
        for item in value
    ):
        return [f"{field} contains an invalid reference"]
    if len(set(value)) != len(value):
        return [f"{field} contains duplicate references"]
    unknown = sorted(set(value) - allowed)
    if unknown:
        return [f"{field} contains references outside the snapshot"]
    return []


def _validate_hypothesis(
    candidate: Mapping[str, Any],
    *,
    allowed: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    if frozenset(candidate) != _HYPOTHESIS_FIELDS:
        errors.append("hypothesis fields do not match the closed schema")
        return errors
    hypothesis_id = candidate.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not (
        _HYPOTHESIS_ID_RE.fullmatch(hypothesis_id)
    ):
        errors.append("invalid hypothesis_id")
    for field in (
        "claim",
        "probe_description",
        "expected_observation",
        "falsification_observation",
    ):
        if not _is_safe_text(
            candidate.get(field), maximum=_MAX_STATEMENT_CHARS
        ):
            errors.append(f"invalid {field}")
    errors.extend(
        _evidence_ref_errors(
            candidate.get("supporting_evidence_refs"),
            field="supporting_evidence_refs",
            require_one=True,
            allowed=allowed,
        )
    )
    errors.extend(
        _evidence_ref_errors(
            candidate.get("disconfirming_evidence_refs"),
            field="disconfirming_evidence_refs",
            require_one=False,
            allowed=allowed,
        )
    )
    expected = candidate.get("expected_observation")
    falsification = candidate.get("falsification_observation")
    if isinstance(expected, str) and isinstance(falsification, str):
        if _normalize_statement(expected) == _normalize_statement(falsification):
            errors.append("expected and falsification observations must differ")
    return errors


def _validate_abstain(candidate: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if frozenset(candidate) != _ABSTAIN_FIELDS:
        errors.append("abstain fields do not match the closed schema")
        return errors
    hypothesis_id = candidate.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not (
        _HYPOTHESIS_ID_RE.fullmatch(hypothesis_id)
    ):
        errors.append("invalid hypothesis_id")
    if candidate.get("reason_code") not in _REASON_CODES:
        errors.append("invalid abstain reason_code")
    return errors


def parse_evidence_packet(
    raw_response: str,
    *,
    binding: PacketBinding,
    allowed_evidence_refs: Iterable[str],
) -> ParseOutcome:
    """Strictly parse one analyst response and attach server-owned identity.

    There is intentionally no retry or repair path.  The caller should treat an
    invalid result as an abstention and continue the D12 solo path.
    """

    errors = _binding_errors(binding)
    try:
        allowed = frozenset(allowed_evidence_refs)
    except TypeError:
        errors.append("allowed_evidence_refs must be iterable")
        allowed = frozenset()
    if any(
        not _is_safe_text(item, maximum=_MAX_EVIDENCE_REF_CHARS)
        for item in allowed
    ):
        errors.append("allowed_evidence_refs contains an invalid reference")
    if errors:
        return ParseOutcome(False, None, tuple(errors))

    try:
        candidate = _strict_json_object(raw_response)
    except _StrictJSONError as exc:
        return ParseOutcome(False, None, (str(exc),))

    status = candidate.get("status")
    if status == "hypothesis":
        errors = _validate_hypothesis(candidate, allowed=allowed)
    elif status == "abstain":
        errors = _validate_abstain(candidate)
    else:
        errors = ["status must be hypothesis or abstain"]

    if candidate.get("hypothesis_id") != binding.hypothesis_id:
        errors.append("hypothesis_id does not match the server binding")
    if errors:
        return ParseOutcome(False, None, tuple(errors))

    # Construct a new canonical object so caller-owned input and JSON decoder
    # internals cannot be mutated through the returned packet.
    packet = dict(candidate)
    for field in _BOUND_FIELDS:
        packet[field] = getattr(binding, field)
    return ParseOutcome(True, packet, ())


def _normalize_statement(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _is_canonical_packet(packet: Any) -> bool:
    if not isinstance(packet, dict):
        return False
    status = packet.get("status")
    payload_fields = (
        _HYPOTHESIS_FIELDS if status == "hypothesis" else _ABSTAIN_FIELDS
    )
    if status not in {"hypothesis", "abstain"}:
        return False
    if frozenset(packet) != payload_fields.union(_BOUND_FIELDS):
        return False
    hypothesis_id = packet.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not (
        _HYPOTHESIS_ID_RE.fullmatch(hypothesis_id)
    ):
        return False
    if any(not _valid_bound_value(packet.get(field)) for field in _BOUND_FIELDS):
        return False
    if status == "abstain":
        return packet.get("reason_code") in _REASON_CODES

    if any(
        not _is_safe_text(packet.get(field), maximum=_MAX_STATEMENT_CHARS)
        for field in (
            "claim",
            "probe_description",
            "expected_observation",
            "falsification_observation",
        )
    ):
        return False
    for field, require_one in (
        ("supporting_evidence_refs", True),
        ("disconfirming_evidence_refs", False),
    ):
        refs = packet.get(field)
        if not isinstance(refs, list):
            return False
        if (require_one and not refs) or len(refs) > _MAX_EVIDENCE_REFS:
            return False
        if any(
            not _is_safe_text(ref, maximum=_MAX_EVIDENCE_REF_CHARS)
            for ref in refs
        ):
            return False
        if len(refs) != len(set(refs)):
            return False
    return _normalize_statement(packet["expected_observation"]) != (
        _normalize_statement(packet["falsification_observation"])
    )


def _context_key(packet: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(packet[field]) for field in (
        "run_id",
        "session_id",
        "task_name",
        "task_family",
        "evidence_snapshot_id",
    ))


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def _is_precise_duplicate(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    if _normalize_statement(first["claim"]) != _normalize_statement(
        second["claim"]
    ):
        return False
    first_probe = tuple(
        _normalize_statement(first[field])
        for field in (
            "probe_description",
            "expected_observation",
            "falsification_observation",
        )
    )
    second_probe = tuple(
        _normalize_statement(second[field])
        for field in (
            "probe_description",
            "expected_observation",
            "falsification_observation",
        )
    )
    if first_probe != second_probe:
        return False
    first_refs = set(first["supporting_evidence_refs"]) | set(
        first["disconfirming_evidence_refs"]
    )
    second_refs = set(second["supporting_evidence_refs"]) | set(
        second["disconfirming_evidence_refs"]
    )
    return _jaccard(first_refs, second_refs) >= 0.8


def _pair_score(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[int, int, float, int]:
    first_refs = set(first["supporting_evidence_refs"]) | set(
        first["disconfirming_evidence_refs"]
    )
    second_refs = set(second["supporting_evidence_refs"]) | set(
        second["disconfirming_evidence_refs"]
    )
    first_probe = tuple(
        _normalize_statement(first[field])
        for field in (
            "probe_description",
            "expected_observation",
            "falsification_observation",
        )
    )
    second_probe = tuple(
        _normalize_statement(second[field])
        for field in (
            "probe_description",
            "expected_observation",
            "falsification_observation",
        )
    )
    return (
        int(first_probe != second_probe),
        int(
            _normalize_statement(first["claim"])
            != _normalize_statement(second["claim"])
        ),
        1.0 - _jaccard(first_refs, second_refs),
        len(first_refs | second_refs),
    )


def select_evidence_packet_ids(
    packets: Sequence[Mapping[str, Any] | None],
) -> list[str]:
    """Return zero to two validated packet IDs without rewriting packets.

    Validity is a hard gate.  Conservative duplicate removal happens before a
    deterministic diversity tie-break, and uncertainty always favors retaining
    a potentially complementary hypothesis.
    """

    if not isinstance(packets, Sequence) or isinstance(packets, (str, bytes)):
        return []
    candidates = [
        packet
        for packet in packets
        if _is_canonical_packet(packet) and packet["status"] == "hypothesis"
    ]
    if not candidates:
        return []

    contexts = {_context_key(packet) for packet in candidates}
    if len(contexts) != 1:
        return []

    # ID uniqueness is required for unambiguous Session Packet Store lookup.
    ids = [packet["hypothesis_id"] for packet in candidates]
    if len(ids) != len(set(ids)):
        return []

    ordered = sorted(candidates, key=lambda item: item["hypothesis_id"])
    representatives: list[Mapping[str, Any]] = []
    for candidate in ordered:
        if any(
            _is_precise_duplicate(candidate, representative)
            for representative in representatives
        ):
            continue
        representatives.append(candidate)

    if not representatives:
        return []
    if len(representatives) == 1:
        return [representatives[0]["hypothesis_id"]]

    best_pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
    best_key: tuple[tuple[int, int, float, int], tuple[str, str]] | None = None
    for index, first in enumerate(representatives[:-1]):
        for second in representatives[index + 1 :]:
            ids_key = (first["hypothesis_id"], second["hypothesis_id"])
            score = _pair_score(first, second)
            # Negated lexical comparison would obscure intent; update only on a
            # better score and keep the first lexically ordered pair on ties.
            key = (score, ids_key)
            if best_key is None or score > best_key[0]:
                best_pair = (first, second)
                best_key = key

    if best_pair is None:
        return []
    return [best_pair[0]["hypothesis_id"], best_pair[1]["hypothesis_id"]]


__all__ = [
    "PacketBinding",
    "ParseOutcome",
    "is_safe_evidence_statement",
    "parse_evidence_packet",
    "select_evidence_packet_ids",
]
