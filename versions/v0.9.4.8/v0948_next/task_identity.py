"""Stable task identity without domain, benchmark, or keyword routing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


GENERAL_TASK_TYPE = "general_task"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
EXPLICIT_ID_FIELDS = ("task_id", "instance_id", "task_name")


@dataclass(frozen=True)
class GenericTaskIdentity:
    """External caller identity plus a path-safe internal identity."""

    internal_task_id: str
    external_task_id: str
    source: str
    task_type: str = GENERAL_TASK_TYPE


def stable_task_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return the immutable task prefix, never later observations."""

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        if role in {"system", "developer"}:
            parts.append(content)
        elif role == "user":
            parts.append(content)
            break
    return "\n".join(parts)


def _internal_id(external_id: str) -> str:
    if IDENTIFIER_PATTERN.fullmatch(external_id):
        return external_id
    digest = hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9_.:-]+", "-", external_id).strip(".-:")
    prefix = prefix[:120].rstrip(".-:")
    return f"{prefix or 'task'}-{digest}"


def resolve_task_identity(
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> GenericTaskIdentity:
    """Prefer caller IDs; otherwise derive a stable content identity.

    Task content is never classified and caller-supplied ``task_type`` never
    changes orchestration.  That prevents benchmark vocabulary from selecting
    a task-specific policy.
    """

    metadata = metadata if isinstance(metadata, Mapping) else {}
    for field in EXPLICIT_ID_FIELDS:
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            external = value.strip()
            return GenericTaskIdentity(
                internal_task_id=_internal_id(external),
                external_task_id=external,
                source=f"metadata.{field}",
            )

    text = stable_task_text(messages)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    task_id = f"task-{digest[:24]}"
    return GenericTaskIdentity(
        internal_task_id=task_id,
        external_task_id=task_id,
        source="stable_task_content_sha256",
    )


__all__ = [
    "EXPLICIT_ID_FIELDS",
    "GENERAL_TASK_TYPE",
    "GenericTaskIdentity",
    "resolve_task_identity",
    "stable_task_text",
]
