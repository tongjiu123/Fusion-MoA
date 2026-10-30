"""Build the opt-in public Panel summary event without raw reasoning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PANEL_BRIEF_HEADER = "[MOA integrated panel brief]\n"
PANEL_BRIEF_FOOTER = (
    "\nThis brief is read-only advice. D12 remains the only "
    "tool-call and final-answer authority."
)


def build_public_panel_event(
    guidance: str,
    analyst: Mapping[str, Any],
    *,
    panel_reused: bool,
    analyst_cells: Sequence[str],
) -> dict[str, Any]:
    """Return an integrated brief plus non-sensitive execution evidence."""

    summary = guidance.strip() if isinstance(guidance, str) else ""
    if summary.startswith(PANEL_BRIEF_HEADER):
        summary = summary[len(PANEL_BRIEF_HEADER) :]
    if summary.endswith(PANEL_BRIEF_FOOTER):
        summary = summary[: -len(PANEL_BRIEF_FOOTER)]
    summary = summary.strip()

    integrator_raw = analyst.get("integrator")
    integrator_raw = (
        integrator_raw if isinstance(integrator_raw, Mapping) else {}
    )
    integrator = {
        key: integrator_raw[key]
        for key in ("actor", "attempted", "status", "error_type")
        if key in integrator_raw
    }
    analyst_result = {
        key: analyst[key]
        for key in (
            "attempted",
            "status",
            "error_type",
            "request_count",
            "response_received_count",
            "schema_valid_count",
            "total_tokens",
            "retry_count",
        )
        if key in analyst
    }
    success = bool(summary) and integrator.get("status") in {
        "brief_ready",
        "brief_reused",
    }
    return {
        "stage": "panel_summary" if success else "panel_failed",
        "status": "brief_ready" if success else "failed_open_to_d12",
        "summary_kind": "integrated_brief",
        "summary": summary if success else "",
        "panel_reused": bool(panel_reused),
        "analyst_cells": list(analyst_cells),
        "analyst": analyst_result,
        "integrator": integrator,
    }


__all__ = [
    "PANEL_BRIEF_FOOTER",
    "PANEL_BRIEF_HEADER",
    "build_public_panel_event",
]
