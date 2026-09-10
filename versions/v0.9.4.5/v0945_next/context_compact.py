"""Task-local rolling checkpoints for Fusion-MOA v0.9.2 compaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .task_identity import stable_task_text


def _prefix_end(messages: Sequence[Mapping[str, Any]]) -> int | None:
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            return index + 1
    return None


@dataclass
class CompactCheckpoint:
    task_id: str
    original_task: str
    prefix_end: int
    covered_until: int
    covered_boundary: dict[str, Any]
    compact_message: dict[str, Any]


class RollingCompactCheckpointStore:
    """Reuse one task's compacted prefix while its client history is continuous."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, CompactCheckpoint] = {}

    def discard(self, task_id: str) -> None:
        self._checkpoints.pop(task_id, None)

    def get(self, task_id: str) -> CompactCheckpoint | None:
        return self._checkpoints.get(task_id)

    def reuse(
        self,
        task_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        checkpoint = self._checkpoints.get(task_id)
        if checkpoint is None:
            return None, {"status": "checkpoint_missing"}
        if (
            len(messages) < checkpoint.covered_until
            or stable_task_text(messages) != checkpoint.original_task
            or dict(messages[checkpoint.covered_until - 1])
            != checkpoint.covered_boundary
        ):
            self.discard(task_id)
            return None, {"status": "checkpoint_discarded_non_contiguous"}

        reused = [
            *[dict(message) for message in messages[: checkpoint.prefix_end]],
            dict(checkpoint.compact_message),
            *[
                dict(message)
                for message in messages[checkpoint.covered_until :]
            ],
        ]
        return reused, {
            "status": "checkpoint_reused",
            "covered_until": checkpoint.covered_until,
            "new_tail_messages": len(messages) - checkpoint.covered_until,
        }

    def save_build(
        self,
        task_id: str,
        raw_messages: Sequence[Mapping[str, Any]],
        compacted_messages: Sequence[Mapping[str, Any]],
        compaction: Mapping[str, Any],
    ) -> CompactCheckpoint | None:
        prefix_end = _prefix_end(raw_messages)
        preserved = compaction.get("preserved_recent_messages")
        if (
            prefix_end is None
            or not isinstance(preserved, int)
            or preserved < 0
            or len(compacted_messages) <= prefix_end
        ):
            return None
        covered_until = len(raw_messages) - preserved
        if covered_until <= prefix_end:
            return None
        compact_message = compacted_messages[prefix_end]
        if not isinstance(compact_message, Mapping):
            return None
        checkpoint = CompactCheckpoint(
            task_id=task_id,
            original_task=stable_task_text(raw_messages),
            prefix_end=prefix_end,
            covered_until=covered_until,
            covered_boundary=dict(raw_messages[covered_until - 1]),
            compact_message=dict(compact_message),
        )
        self._checkpoints[task_id] = checkpoint
        return checkpoint

    def save_extension(
        self,
        task_id: str,
        raw_messages: Sequence[Mapping[str, Any]],
        reused_messages: Sequence[Mapping[str, Any]],
        compacted_messages: Sequence[Mapping[str, Any]],
        compaction: Mapping[str, Any],
    ) -> CompactCheckpoint | None:
        previous = self._checkpoints.get(task_id)
        preserved = compaction.get("preserved_recent_messages")
        if (
            previous is None
            or not isinstance(preserved, int)
            or preserved < 0
            or len(compacted_messages) <= previous.prefix_end
        ):
            return None

        recent_start = len(reused_messages) - preserved
        newly_covered = max(
            0,
            recent_start - (previous.prefix_end + 1),
        )
        covered_until = min(
            len(raw_messages),
            previous.covered_until + newly_covered,
        )
        compact_message = compacted_messages[previous.prefix_end]
        if not isinstance(compact_message, Mapping):
            return None
        checkpoint = CompactCheckpoint(
            task_id=task_id,
            original_task=stable_task_text(raw_messages),
            prefix_end=previous.prefix_end,
            covered_until=covered_until,
            covered_boundary=dict(raw_messages[covered_until - 1]),
            compact_message=dict(compact_message),
        )
        self._checkpoints[task_id] = checkpoint
        return checkpoint


__all__ = [
    "CompactCheckpoint",
    "RollingCompactCheckpointStore",
]
