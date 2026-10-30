"""Public request controls owned by the Fusion-MOA Gateway."""

from __future__ import annotations

from typing import Any, Mapping


MOA_REQUEST_MODES = frozenset({"auto", "panel_once", "always", "initial_panel"})
DEFAULT_REASONING_EFFORT = "medium"


def request_uses_thinking(body: Mapping[str, Any]) -> bool:
    chat_kwargs = body.get("chat_template_kwargs")
    return not (
        isinstance(chat_kwargs, Mapping)
        and chat_kwargs.get("enable_thinking") is False
    )


def normalize_request_body(
    body: Mapping[str, Any], *, default_mode: str = "auto"
) -> dict[str, Any]:
    """Normalize top-level MoA controls into internal request metadata."""

    normalized = dict(body)
    metadata_raw = body.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

    metadata_moa = metadata.get("moa")
    if metadata_moa is not None and not isinstance(metadata_moa, Mapping):
        raise ValueError("metadata.moa must be a JSON object")
    top_level_moa = body.get("moa")
    if "moa" in body and not isinstance(top_level_moa, Mapping):
        raise ValueError("moa must be a JSON object")

    moa: dict[str, Any] = {}
    if isinstance(metadata_moa, Mapping):
        moa.update(metadata_moa)
    if isinstance(top_level_moa, Mapping):
        moa.update(top_level_moa)

    mode = moa.get("mode", default_mode)
    if not isinstance(mode, str) or mode not in MOA_REQUEST_MODES:
        allowed = ", ".join(sorted(MOA_REQUEST_MODES))
        raise ValueError(f"moa.mode must be one of: {allowed}")
    if moa or default_mode != "auto":
        moa["mode"] = mode
        metadata["moa"] = moa

    normalized["metadata"] = metadata
    normalized.pop("moa", None)
    chat_kwargs_raw = normalized.get("chat_template_kwargs")
    if chat_kwargs_raw is None:
        chat_kwargs: dict[str, Any] = {}
    elif isinstance(chat_kwargs_raw, Mapping):
        chat_kwargs = dict(chat_kwargs_raw)
    else:
        return normalized
    if chat_kwargs.get("enable_thinking") is not False:
        chat_kwargs.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    normalized["chat_template_kwargs"] = chat_kwargs
    return normalized


__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "MOA_REQUEST_MODES",
    "normalize_request_body",
    "request_uses_thinking",
]
