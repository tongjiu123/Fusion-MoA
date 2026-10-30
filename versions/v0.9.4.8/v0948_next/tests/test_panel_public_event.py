from __future__ import annotations

import unittest

from v0948_next.panel_public import (
    PANEL_BRIEF_FOOTER,
    PANEL_BRIEF_HEADER,
    build_public_panel_event,
)


class PublicPanelEventTests(unittest.TestCase):
    def test_success_exposes_integrated_brief_and_counts_only(self) -> None:
        event = build_public_panel_event(
            PANEL_BRIEF_HEADER + "下一步：先关联 Trace ID。" + PANEL_BRIEF_FOOTER,
            {
                "attempted": True,
                "status": "no_valid_guidance",
                "request_count": 3,
                "response_received_count": 3,
                "schema_valid_count": 3,
                "total_tokens": 1680,
                "_integrator_packets": [{"private": "must not leak"}],
                "integrator": {
                    "actor": "D13",
                    "attempted": True,
                    "status": "brief_ready",
                },
            },
            panel_reused=False,
            analyst_cells=("D16", "D9", "D13"),
        )

        self.assertEqual(event["stage"], "panel_summary")
        self.assertEqual(event["status"], "brief_ready")
        self.assertEqual(event["summary"], "下一步：先关联 Trace ID。")
        self.assertEqual(event["analyst"]["request_count"], 3)
        self.assertEqual(event["integrator"]["actor"], "D13")
        self.assertNotIn("_integrator_packets", event["analyst"])
        self.assertNotIn("private", str(event))

    def test_failure_reports_fallback_without_internal_output(self) -> None:
        event = build_public_panel_event(
            "",
            {
                "attempted": True,
                "status": "degraded_to_executor",
                "error_type": "DuplicateAnalystBatchError",
                "request_count": 0,
                "integrator": {
                    "actor": "D13",
                    "attempted": False,
                    "status": "no_valid_analyst_packets",
                },
            },
            panel_reused=False,
            analyst_cells=("D16", "D9", "D13"),
        )

        self.assertEqual(event["stage"], "panel_failed")
        self.assertEqual(event["status"], "failed_open_to_d12")
        self.assertEqual(event["summary"], "")
        self.assertEqual(
            event["analyst"]["error_type"],
            "DuplicateAnalystBatchError",
        )


if __name__ == "__main__":
    unittest.main()
