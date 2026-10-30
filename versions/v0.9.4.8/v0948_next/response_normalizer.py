"""Normalize D12 thinking markup at the public OpenAI-compatible boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_leading_thinking(content: str) -> tuple[str, str, bool]:
    """Split one complete leading ``<think>`` block without guessing malformed text."""

    stripped = content.lstrip()
    if not stripped.startswith(THINK_OPEN):
        return content, "", False
    close_at = stripped.find(THINK_CLOSE, len(THINK_OPEN))
    if close_at < 0:
        return content, "", False
    reasoning = stripped[len(THINK_OPEN) : close_at]
    final = stripped[close_at + len(THINK_CLOSE) :].lstrip(" \t\r\n")
    return final, reasoning, True


def split_unclosed_leading_thinking(content: str) -> tuple[str, str, bool]:
    """Separate a terminal leading thinking block that has no closing tag."""

    stripped = content.lstrip()
    if not stripped.startswith(THINK_OPEN):
        return content, "", False
    remainder = stripped[len(THINK_OPEN) :]
    if THINK_CLOSE in remainder:
        return content, "", False
    return "", remainder, True


def split_implicit_leading_thinking(content: str) -> tuple[str, str, bool]:
    """Split Qwen-style reasoning when ``<think>`` came from the prompt."""

    stripped = content.lstrip()
    if stripped.startswith(THINK_OPEN):
        return content, "", False
    close_at = stripped.find(THINK_CLOSE)
    if close_at < 0:
        return content, "", False
    reasoning = stripped[:close_at]
    final = stripped[close_at + len(THINK_CLOSE) :].lstrip(" \t\r\n")
    return final, reasoning, True


def normalize_message(
    message: Mapping[str, Any],
    *,
    allow_implicit_thinking: bool = False,
) -> dict[str, Any]:
    """Move leading thinking markup out of public ``content``."""

    normalized = deepcopy(dict(message))
    content = normalized.get("content")
    if not isinstance(content, str):
        return normalized
    final, extracted, changed = split_leading_thinking(content)
    implicit = False
    if not changed:
        if allow_implicit_thinking:
            final, extracted, changed = split_implicit_leading_thinking(content)
            implicit = changed
    if not changed:
        final, extracted, changed = split_unclosed_leading_thinking(content)
        if not changed:
            return normalized
    normalized["content"] = final
    if extracted and not implicit:
        existing = normalized.get("reasoning_content")
        if not isinstance(existing, str) or not existing:
            normalized["reasoning_content"] = extracted
    return normalized


def normalize_completion_payload(
    payload: Mapping[str, Any],
    *,
    allow_implicit_thinking: bool = False,
) -> dict[str, Any]:
    """Normalize every Chat Completions choice while preserving the envelope."""

    normalized = deepcopy(dict(payload))
    choices = normalized.get("choices")
    if not isinstance(choices, list):
        return normalized
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            choice["message"] = normalize_message(
                message,
                allow_implicit_thinking=allow_implicit_thinking,
            )
    return normalized


class ThinkingStreamNormalizer:
    """Statefully normalize a leading thinking block across SSE chunks."""

    def __init__(self, *, allow_implicit_thinking: bool = False) -> None:
        self._pending = ""
        self._resolved = False
        self._last_chunk: dict[str, Any] | None = None
        self._allow_implicit_thinking = allow_implicit_thinking

    @staticmethod
    def _choice_delta(
        chunk: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any] | None, Any]:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, None
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return None, None
        delta = choice.get("delta")
        return (
            delta if isinstance(delta, Mapping) else None,
            choice.get("finish_reason"),
        )

    @staticmethod
    def _with_delta(
        chunk: Mapping[str, Any],
        delta: Mapping[str, Any],
        *,
        finish_reason: Any = None,
    ) -> dict[str, Any]:
        rendered = deepcopy(dict(chunk))
        choices = rendered.get("choices")
        if not isinstance(choices, list) or not choices:
            return rendered
        choice = choices[0]
        if not isinstance(choice, dict):
            return rendered
        choice["delta"] = dict(delta)
        choice["finish_reason"] = finish_reason
        return rendered

    def _flush_pending(
        self,
        template: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not self._pending:
            return []
        content = self._pending
        self._pending = ""
        self._resolved = True
        return [self._with_delta(template, {"content": content})]

    def _flush_terminal_pending(
        self,
        template: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not self._pending:
            return []
        _, reasoning, changed = split_unclosed_leading_thinking(self._pending)
        if not changed:
            if self._allow_implicit_thinking:
                self._pending = ""
                self._resolved = True
                return []
            return self._flush_pending(template)
        self._pending = ""
        self._resolved = True
        if not reasoning:
            return []
        return [self._with_delta(template, {"reasoning_content": reasoning})]

    def feed(self, chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return zero or more normalized chunks for one upstream chunk."""

        self._last_chunk = deepcopy(dict(chunk))
        delta, finish_reason = self._choice_delta(chunk)
        if delta is None:
            return [deepcopy(dict(chunk))]
        content = delta.get("content")
        if self._resolved or not isinstance(content, str):
            if finish_reason is not None and self._pending:
                return [
                    *self._flush_terminal_pending(chunk),
                    self._with_delta(chunk, delta, finish_reason=finish_reason),
                ]
            return [deepcopy(dict(chunk))]

        outputs: list[dict[str, Any]] = []
        other_delta = dict(delta)
        other_delta.pop("content", None)
        if other_delta:
            outputs.append(self._with_delta(chunk, other_delta))

        self._pending += content
        final, reasoning, changed = split_leading_thinking(self._pending)
        implicit = False
        if not changed and self._allow_implicit_thinking:
            final, reasoning, changed = split_implicit_leading_thinking(
                self._pending
            )
            implicit = changed
        if changed:
            self._pending = ""
            self._resolved = True
            if reasoning and not implicit:
                outputs.append(
                    self._with_delta(chunk, {"reasoning_content": reasoning})
                )
            if final:
                outputs.append(self._with_delta(chunk, {"content": final}))
        else:
            candidate = self._pending.lstrip()
            could_be_thinking = self._allow_implicit_thinking or (
                not candidate
                or THINK_OPEN.startswith(candidate)
                or candidate.startswith(THINK_OPEN)
            )
            if not could_be_thinking:
                outputs.extend(self._flush_pending(chunk))

        if finish_reason is not None:
            outputs.extend(self._flush_terminal_pending(chunk))
            outputs.append(
                self._with_delta(chunk, {}, finish_reason=finish_reason)
            )
        return outputs

    def flush(self) -> list[dict[str, Any]]:
        """Flush a terminal partial block without leaking thinking markup."""

        if not self._pending or self._last_chunk is None:
            return []
        return self._flush_terminal_pending(self._last_chunk)


__all__ = [
    "THINK_CLOSE",
    "THINK_OPEN",
    "ThinkingStreamNormalizer",
    "normalize_completion_payload",
    "normalize_message",
    "split_implicit_leading_thinking",
    "split_leading_thinking",
    "split_unclosed_leading_thinking",
]
